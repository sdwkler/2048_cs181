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
from src.phase_3.search.expectimax import ExpectimaxAgent, GreedyAgent

_PROCESS_MODEL_CACHE = {}

# =======================================================================
# 【极其严谨的 3 组终极消融实验矩阵】
# 参数含义: (exp_id, name, algorithm, use_afterstate(拓扑), eval_type, leaf_mode(公式), depth, use_pruning(剪枝))
# =======================================================================
SEARCH_CONFIGS = [
    # 1. 传统 State NTuple (算力爆炸，但能感知危险)
    ("A-1", "State NTuple", "expectimax", False, "ntuple_state", "state", 2, False),
    
    # 2. 传统 Afterstate NTuple (算力极低，速度极快，但容易对危机盲目而暴毙)
    ("A-2", "Afterstate NTuple", "expectimax", True, "ntuple_afterstate", "afterstate", 2, False),
    
    # 3. RSZ 双轨制 Dual NTuple (算力极低，依靠 V_risk O(1) 预判危险，结合了速度与安全)
    ("A-3", "Dual Risk-Aware NTuple", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False),
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

    # 映射文件路径
    if eval_type == "ntuple_afterstate": weight_file = "models/2048_afterstate.bin"
    elif eval_type == "ntuple_state": weight_file = "models/2048_state.bin"
    elif eval_type == "ntuple_dual_mean": weight_file = "models/2048_dual_mean.bin"
    elif eval_type == "ntuple_dual_risk": weight_file = "models/2048_dual_risk.bin"
    else: raise ValueError(f"Unknown eval_type: {eval_type}")

    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Missing N-tuple weight file: {weight_file}")
    fast_mmap_load(tdl, weight_file)
    return tdl


def get_value_func(eval_type: str):
    """
    返回一个 tuple (mean_evaluator, risk_evaluator)
    对于不需要 risk 的传统模型，risk_evaluator 为 None
    """
    if eval_type == "heuristic":
        return HeuristicEvaluator().evaluate, None
        
    # 【核心】：加载双轨制模型
    if eval_type == "ntuple_dual":
        if "ntuple_dual_mean" not in _PROCESS_MODEL_CACHE:
            _PROCESS_MODEL_CACHE["ntuple_dual_mean"] = build_ntuple("ntuple_dual_mean")
            _PROCESS_MODEL_CACHE["ntuple_dual_risk"] = build_ntuple("ntuple_dual_risk")
        mean_func = NTupleEvaluator(_PROCESS_MODEL_CACHE["ntuple_dual_mean"]).evaluate
        risk_func = NTupleEvaluator(_PROCESS_MODEL_CACHE["ntuple_dual_risk"]).evaluate
        return mean_func, risk_func

    # 传统单模型
    if eval_type not in _PROCESS_MODEL_CACHE:
        _PROCESS_MODEL_CACHE[eval_type] = build_ntuple(eval_type)
    return NTupleEvaluator(_PROCESS_MODEL_CACHE[eval_type]).evaluate, None


def build_search_agent(algorithm: str, use_afterstate: bool, eval_type: str, leaf_mode: str, use_pruning: bool, p4_prob: float):
    value_func_mean, value_func_risk = get_value_func(eval_type)
    
    # 【这里是控制 AI 胆量大小的关键参数】
    # Beta = 2.0 表示一旦预测出未来会有分数跌幅，立刻给予 2 倍跌幅分数的极强惩罚，彻底绕开危险区
    risk_beta = 0.2
    
    if algorithm == "greedy":
        return GreedyAgent(value_func=value_func_mean)
        
    return ExpectimaxAgent(
        use_afterstate=use_afterstate, 
        value_func=value_func_mean,
        value_func_risk=value_func_risk,
        risk_beta=risk_beta,
        leaf_mode=leaf_mode,
        use_pruning=use_pruning,
        prune_top_k=2,
        p4_prob=p4_prob
    )


def search_game_worker(args):
    game_seed, algorithm, use_afterstate, eval_type, leaf_mode, search_depth, max_game_steps, use_pruning, p4_prob = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    agent = build_search_agent(algorithm, use_afterstate, eval_type, leaf_mode, use_pruning, p4_prob)

    b = board()
    popup_with_rng(b, rng, p4=p4_prob)
    popup_with_rng(b, rng, p4=p4_prob)

    score, steps = 0, 0
    step_times, compressions, b_effs, node_counts = [], [], [], []
    while max_game_steps is None or steps < max_game_steps:
        start = time.perf_counter()
        action, comp_ratio, b_eff, node = agent.get_best_action(b, max_depth=search_depth)
        step_times.append(time.perf_counter() - start)
        compressions.append(comp_ratio)
        b_effs.append(b_eff)
        node_counts.append(node)

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
    }


def evaluate_config(config, exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth, use_pruning):
    p4_prob = getattr(config, 'p4_prob', 0.1) 
    seeds = [config.seed + i for i in range(config.search_games)]
    args_list = [
        (seed, algorithm, use_afterstate, eval_type, leaf_mode, depth, config.max_game_steps, use_pruning, p4_prob)
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
            "algorithm": algorithm,
            "depth": depth,
            "leaf_mode": leaf_mode,
            "use_afterstate": use_afterstate,
            "use_pruning": use_pruning,
            "b_eff": safe_mean(r["b_eff"] for r in records),
            "compression_ratio": safe_mean(r["compression_ratio"] for r in records),
            "node_count": safe_mean(r["node_count"] for r in records),
        }
    )
    return summary, records


def run_experiment(config):
    rows, details = [], {"games": {}}
    for exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth, use_pruning in SEARCH_CONFIGS:
        summary, records = evaluate_config(
            config, exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth, use_pruning
        )
        rows.append(summary)
        details["games"][exp_id] = records
        print(
            f"{exp_id} {name}: score={summary['average_score']:.1f}, "
            f"time/step={summary['time_per_step_ms']:.2f}ms, "
            f"comp_ratio={summary['compression_ratio']:.3f}"
        )

    paths = write_result_bundle(config.output_dir, "search", config, rows, details)
    print(f"Search results saved: {paths['md']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run phase-3 Expectimax experiments.")
    add_common_args_3(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()