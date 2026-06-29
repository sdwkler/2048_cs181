# src/phase_1/search/run_search.py
import argparse
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.environments.base_env import board
from src.ntuple.feature_base import diff_pattern, feature, learning, pattern
from src.ntuple.loader import fast_mmap_load
from src.common import (
    ACTION_NAMES,
    add_common_args_3,
    apply_action,
    config_from_args,
    generate_pressure_boards,
    max_tile_value,
    popup_with_rng,
    progress,
    safe_mean,
    summarize_games,
    write_result_bundle,
)
from src.evaluators import HeuristicEvaluator, NTupleEvaluator
from src.phase_3.search.expectimax import ExpectimaxAgent

_PROCESS_MODEL_CACHE = {}

# =======================================================================
# 【衍生优化方案核心验证矩阵：6组】
# 参数含义: (exp_id, name, eval_type, use_beam_search, use_tt, depth)
# =======================================================================
SEARCH_CONFIGS = [
    # ---- 组 A: 人工启发式 (Heuristic) 验证 ----
    # 1. 基线 (纯全展开)
    ("2-A1", "Heuristic + Base", "heuristic", False, False, 3),
    # 2. 方案1：Beam Search (预测效果：提速，但由于人工打分不准，分数暴跌)
    ("2-A2", "Heuristic + BeamSearch", "heuristic", True, False, 3),
    # 3. 方案2：Hash Cache DAG 优化 (预测效果：无损提速，分数与基线完全一致，节点展开数下降)
    ("2-A3", "Heuristic + HashDAG", "heuristic", False, True, 3),
    
    # ---- 组 B: N-Tuple (Afterstate 权重) 验证 ----
    # 4. 基线
    ("2-B1", "NTuple + Base", "ntuple_afterstate", False, False, 3),
    # 5. 方案1：Beam Search (预测效果：大幅提速，且由于 N-Tuple 极其精准，分数几乎不掉)
    ("2-B2", "NTuple + BeamSearch", "ntuple_afterstate", True, False, 3),
    # 6. 方案2：Hash Cache DAG 优化 (预测效果：绝对无损，速度最快，哈希命中率极高)
    ("2-B3", "NTuple + HashDAG", "ntuple_afterstate", False, True, 3),
]


def build_ntuple(eval_type: str) -> learning:
    original_alloc = feature.alloc
    old_stdout = sys.stdout
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    try:
        feature.alloc = staticmethod(lambda num: range(num))
        tdl = learning()
        shapes = [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6], [4, 5, 6, 8, 9, 10]]
        for shape in shapes:
            tdl.add_feature(pattern(shape))
        for shape in shapes:
            tdl.add_feature(diff_pattern(shape))
    finally:
        feature.alloc = original_alloc
        sys.stdout = old_stdout
        devnull.close()

    weight_file = "models/2048_afterstate.bin"
    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"missing N-tuple weight file: {weight_file}")
    fast_mmap_load(tdl, weight_file)
    return tdl


def get_value_func(eval_type: str):
    if eval_type == "heuristic":
        return HeuristicEvaluator().evaluate
    if eval_type not in _PROCESS_MODEL_CACHE:
        _PROCESS_MODEL_CACHE[eval_type] = build_ntuple(eval_type)
    return NTupleEvaluator(_PROCESS_MODEL_CACHE[eval_type]).evaluate


def build_search_agent(eval_type: str, use_beam_search: bool, use_tt: bool, p4_prob: float):
    value_func = get_value_func(eval_type)
    return ExpectimaxAgent(
        value_func=value_func, 
        use_beam_search=use_beam_search,
        beam_width=2, # 只保留 Top 2
        use_tt=use_tt,
        p4_prob=p4_prob
    )


def search_game_worker(args):
    game_seed, eval_type, use_beam_search, use_tt, search_depth, max_game_steps, p4_prob = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    agent = build_search_agent(eval_type, use_beam_search, use_tt, p4_prob)

    b = board()
    popup_with_rng(b, rng, p4=p4_prob)
    popup_with_rng(b, rng, p4=p4_prob)

    score, steps = 0, 0
    step_times, compressions, b_effs, node_counts, tt_hit_rates = [], [], [], [], []
    while max_game_steps is None or steps < max_game_steps:
        start = time.perf_counter()
        
        action, comp_ratio, b_eff, node_cnt, tt_hits = agent.get_best_action(b, max_depth=search_depth)
        
        step_times.append(time.perf_counter() - start)
        compressions.append(comp_ratio)
        b_effs.append(b_eff)
        node_counts.append(node_cnt)
        # 计算该步的哈希命中率：命中次数 / (命中次数 + 实际展开节点数)
        hit_rate = tt_hits / max(1, tt_hits + node_cnt)
        tt_hit_rates.append(hit_rate)

        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1:
            break
        score += reward
        steps += 1
        popup_with_rng(next_b, rng, p4=p4_prob)
        b = next_b

    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(step_times),
        "compression_ratio": safe_mean(compressions),
        "b_eff": safe_mean(b_effs),
        "node_count": safe_mean(node_counts),
        "tt_hit_rate": safe_mean(tt_hit_rates),
    }


def evaluate_config(config, exp_id, name, eval_type, use_beam_search, use_tt, depth):
    p4_prob = getattr(config, 'p4_prob', 0.1) 
    seeds = [config.seed + i for i in range(config.search_games)]
    args_list = [
        (seed, eval_type, use_beam_search, use_tt, depth, config.max_game_steps, p4_prob)
        for seed in seeds
    ]
    records = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(search_game_worker, args) for args in args_list]
        for future in progress(as_completed(futures), total=len(futures), desc=name[:24], leave=False):
            records.append(future.result())

    summary = summarize_games(records)
    summary.update(
        {
            "experiment": exp_id,
            "variant": name,
            "depth": depth,
            "eval_type": eval_type,
            "use_beam_search": use_beam_search,
            "use_tt": use_tt,
            "b_eff": safe_mean(r["b_eff"] for r in records),
            "compression_ratio": safe_mean(r["compression_ratio"] for r in records),
            "node_count": safe_mean(r["node_count"] for r in records),
            "tt_hit_rate": safe_mean(r["tt_hit_rate"] for r in records),
        }
    )
    return summary, records


def run_experiment(config):
    rows, details = [], {"games": {}}
    print(f"========== 衍生优化方案(DAG vs BeamSearch) 验证 ==========")
    for exp_id, name, eval_type, use_beam_search, use_tt, depth in SEARCH_CONFIGS:
        summary, records = evaluate_config(
            config, exp_id, name, eval_type, use_beam_search, use_tt, depth
        )
        rows.append(summary)
        details["games"][exp_id] = records
        
        # 针对不同算法展示最关键的核心指标
        tt_info = f", TT_Hit={summary['tt_hit_rate']*100:.1f}%" if use_tt else ""
        print(
            f"[{exp_id}] {name}: Score={summary['average_score']:.0f} | "
            f"Nodes={summary['node_count']:.0f} | Time={summary['time_per_step_ms']:.1f}ms{tt_info}"
        )

    paths = write_result_bundle(config.output_dir, "search_optimizations", config, rows, details)
    print(f"\n✅ 优化消融实验结果已保存: {paths['md']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run phase-1 Expectimax Optimization experiments.")
    add_common_args_3(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()