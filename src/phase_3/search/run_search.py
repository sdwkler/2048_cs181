import argparse
import math
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
from src.evaluators import NTupleEvaluator
from src.phase_3.search.expectimax import ExpectimaxAgent, GreedyAgent

# 【完美接入你 MCTS 中的极速查表评估器】
from src.phase_1.planning.mcts3_new_node import FastHeuristic

_PROCESS_MODEL_CACHE = {}

# =======================================================================
# 【极其严谨的 10 组终极消融实验矩阵】
# 参数: (exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth, use_pruning, risk_beta)
# =======================================================================
SEARCH_CONFIGS = [
    # 1. 基础对照组 (仅使用 FastHeuristic 评估)
    ("H-1", "State Heuristic (No Risk)", "expectimax", False, "heuristic", "state", 2, False, 0.0),
    ("H-2", "Afterstate Heuristic (No Risk)", "expectimax", True, "heuristic", "afterstate", 2, False, 0.0),
    
    # 2. Heuristic Mean + N-Tuple Risk 矩阵测试 (Beta 0.1 -> 0.4)
    ("H-3.1", "Heuristic + Risk (b=0.1)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.1),
    ("H-3.2", "Heuristic + Risk (b=0.2)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.2),
    ("H-3.3", "Heuristic + Risk (b=0.3)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.3),
    ("H-3.4", "Heuristic + Risk (b=0.4)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.4),
    ("H-3.5", "Heuristic + Risk (b=0.5)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.5),
    ("H-3.6", "Heuristic + Risk (b=0.6)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.6),
    ("H-3.7", "Heuristic + Risk (b=0.7)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.7),
    ("H-3.8", "Heuristic + Risk (b=0.8)", "expectimax", True, "heuristic_dual_risk", "afterstate_dual_risk", 2, False, 0.8),

    # 3. Dual N-Tuple (Mean + Risk) 矩阵测试 (Beta 0.1 -> 0.4)
    ("N-3.1", "Dual N-Tuple (b=0.1)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.1),
    ("N-3.2", "Dual N-Tuple (b=0.2)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.2),
    ("N-3.3", "Dual N-Tuple (b=0.3)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.3),
    ("N-3.4", "Dual N-Tuple (b=0.4)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.4),
    ("N-3.5", "Dual N-Tuple (b=0.5)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.5),
    ("N-3.6", "Dual N-Tuple (b=0.6)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.6),
    ("N-3.7", "Dual N-Tuple (b=0.7)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.7),
    ("N-3.8", "Dual N-Tuple (b=0.8)", "expectimax", True, "ntuple_dual", "afterstate_dual_risk", 2, False, 0.8),
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
    if eval_type == "ntuple_dual_mean": weight_file = "models/2048_dual_mean.bin"
    elif eval_type == "ntuple_dual_risk": weight_file = "models/2048_dual_risk.bin"
    elif eval_type == "ntuple_heuristic_risk": weight_file = "models/2048_heuristic_risk.bin"
    else: raise ValueError(f"Unknown eval_type: {eval_type}")

    if not os.path.exists(weight_file):
        raise FileNotFoundError(f"Missing N-tuple weight file: {weight_file}")
    fast_mmap_load(tdl, weight_file)
    return tdl


def get_value_func(eval_type: str):
    """
    返回 tuple (mean_evaluator, risk_evaluator)
    """
    # =======================================================
    # 【核心修复：类型适配包装器】
    # Expectimax 传过来的是 board 对象，但 FastHeuristic 需要的是 board.raw (整数)
    # 用一个 wrapper 桥接，并确保 FastHeuristic 在进程中是全局单例以共享缓存！
    # =======================================================
    if "heuristic" in eval_type and "fast_heuristic" not in _PROCESS_MODEL_CACHE:
        _PROCESS_MODEL_CACHE["fast_heuristic"] = FastHeuristic()
        
    def wrapped_heuristic(b: board, is_afterstate: bool) -> float:
        return _PROCESS_MODEL_CACHE["fast_heuristic"].evaluate(b.raw, is_afterstate)

    # 1. 纯人工规则模式 (仅返回 Mean)
    if eval_type == "heuristic":
        return wrapped_heuristic, None
        
    # 2. 人工规则作 Mean，刚刚蒸馏出的 N-Tuple 雷达作 Risk
    if eval_type == "heuristic_dual_risk":
        mean_func = wrapped_heuristic
        if "ntuple_heuristic_risk" not in _PROCESS_MODEL_CACHE:
            _PROCESS_MODEL_CACHE["ntuple_heuristic_risk"] = build_ntuple("ntuple_heuristic_risk")
        risk_func = NTupleEvaluator(_PROCESS_MODEL_CACHE["ntuple_heuristic_risk"]).evaluate
        return mean_func, risk_func

    # 3. 纯双轨 N-Tuple 模式
    if eval_type == "ntuple_dual":
        if "ntuple_dual_mean" not in _PROCESS_MODEL_CACHE:
            _PROCESS_MODEL_CACHE["ntuple_dual_mean"] = build_ntuple("ntuple_dual_mean")
            _PROCESS_MODEL_CACHE["ntuple_dual_risk"] = build_ntuple("ntuple_dual_risk")
        mean_func = NTupleEvaluator(_PROCESS_MODEL_CACHE["ntuple_dual_mean"]).evaluate
        risk_func = NTupleEvaluator(_PROCESS_MODEL_CACHE["ntuple_dual_risk"]).evaluate
        return mean_func, risk_func

    raise ValueError(f"Unhandled eval_type in get_value_func: {eval_type}")


def build_search_agent(algorithm: str, use_afterstate: bool, eval_type: str, leaf_mode: str, use_pruning: bool, p4_prob: float, risk_beta: float):
    value_func_mean, value_func_risk = get_value_func(eval_type)
    
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
    game_seed, algorithm, use_afterstate, eval_type, leaf_mode, search_depth, max_game_steps, use_pruning, p4_prob, risk_beta = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    agent = build_search_agent(algorithm, use_afterstate, eval_type, leaf_mode, use_pruning, p4_prob, risk_beta)

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


def evaluate_config(config, exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth, use_pruning, risk_beta):
    p4_prob = getattr(config, 'p4_prob', 0.1) 
    seeds = [config.seed + i for i in range(config.search_games)]
    args_list = [
        (seed, algorithm, use_afterstate, eval_type, leaf_mode, depth, config.max_game_steps, use_pruning, p4_prob, risk_beta)
        for seed in seeds
    ]
    records = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(search_game_worker, args) for args in args_list]
        for future in progress(as_completed(futures), total=len(futures), desc=name[:24], leave=False):
            records.append(future.result())

    summary = summarize_games(records)
    
    # 【核心：为你精准计算分数的标准差 (Standard Deviation)】
    scores = [r["score"] for r in records]
    avg_score = sum(scores) / max(1, len(scores))
    score_std = math.sqrt(sum((s - avg_score) ** 2 for s in scores) / max(1, len(scores)))
    
    summary.update(
        {
            "experiment": exp_id,
            "variant": name,
            "algorithm": algorithm,
            "depth": depth,
            "leaf_mode": leaf_mode,
            "use_afterstate": use_afterstate,
            "use_pruning": use_pruning,
            "risk_beta": risk_beta,
            "score_std": score_std,  # 记录方差/标准差
            "b_eff": safe_mean(r["b_eff"] for r in records),
            "compression_ratio": safe_mean(r["compression_ratio"] for r in records),
            "node_count": safe_mean(r["node_count"] for r in records),
        }
    )
    return summary, records


def run_experiment(config):
    rows, details = [], {"games": {}}
    for exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth, use_pruning, risk_beta in SEARCH_CONFIGS:
        summary, records = evaluate_config(
            config, exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth, use_pruning, risk_beta
        )
        rows.append(summary)
        details["games"][exp_id] = records
        # 终端打印加上了 std (标准差) 的输出
        print(
            f"[{exp_id}] {name}: score={summary['average_score']:.1f} (std: {summary['score_std']:.1f}), "
            f"time/step={summary['time_per_step_ms']:.2f}ms"
        )

    paths = write_result_bundle(config.output_dir, "search", config, rows, details)
    print(f"\n✅ Search results & Variance Data saved: {paths['md']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run phase-3 Expectimax experiments with RSZ Variance Analysis.")
    add_common_args_3(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()