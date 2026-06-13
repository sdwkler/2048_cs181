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
from src.phase_1.common import (
    ACTION_NAMES,
    add_common_args,
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
from src.phase_1.evaluators import HeuristicEvaluator, NTupleEvaluator
from src.phase_1.search.expectimax import ExpectimaxAgent, GreedyAgent


_PROCESS_MODEL_CACHE = {}


SEARCH_CONFIGS = [
    ("1-A", "Greedy Baseline", "greedy", False, "heuristic", "afterstate", 1),
    ("1-B", "Standard+Heuristic", "expectimax", False, "heuristic", "state", 3),
    ("1-C", "Afterstate+Heuristic", "expectimax", True, "heuristic", "afterstate", 3),
    ("1-D", "Standard+StateNTuple", "expectimax", False, "ntuple_state", "state", 3),
    ("1-E", "Standard+AfterstateNTuple", "expectimax", False, "ntuple_afterstate", "afterstate", 3),
    ("1-F", "Afterstate+StateNTuple", "expectimax", True, "ntuple_state", "state", 3),
    ("1-G", "Afterstate+AfterstateNTuple", "expectimax", True, "ntuple_afterstate", "afterstate", 3),
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

    weight_file = "models/2048_afterstate.bin" if eval_type == "ntuple_afterstate" else "models/2048_state.bin"
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


def build_search_agent(algorithm: str, use_afterstate: bool, eval_type: str, leaf_mode: str):
    value_func = get_value_func(eval_type)
    if algorithm == "greedy":
        return GreedyAgent(value_func=value_func)
    return ExpectimaxAgent(use_afterstate=use_afterstate, value_func=value_func, leaf_mode=leaf_mode)


def search_game_worker(args):
    game_seed, algorithm, use_afterstate, eval_type, leaf_mode, search_depth, max_game_steps = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    agent = build_search_agent(algorithm, use_afterstate, eval_type, leaf_mode)

    b = board()
    popup_with_rng(b, rng)
    popup_with_rng(b, rng)

    score, steps = 0, 0
    step_times, compressions, b_effs = [], [], []
    while max_game_steps is None or steps < max_game_steps:
        start = time.perf_counter()
        action, comp_ratio, b_eff = agent.get_best_action(b, max_depth=search_depth)
        step_times.append(time.perf_counter() - start)
        compressions.append(comp_ratio)
        b_effs.append(b_eff)

        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1:
            break
        score += reward
        steps += 1
        popup_with_rng(next_b, rng)
        b = next_b

    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(step_times),
        "compression_ratio": safe_mean(compressions),
        "b_eff": safe_mean(b_effs),
    }


def evaluate_config(config, exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth):
    seeds = [config.seed + i for i in range(config.search_games)]
    args_list = [
        (seed, algorithm, use_afterstate, eval_type, leaf_mode, depth, config.max_game_steps)
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
            "b_eff": safe_mean(r["b_eff"] for r in records),
            "compression_ratio": safe_mean(r["compression_ratio"] for r in records),
        }
    )
    return summary, records


def best_action_for_raw(raw: int, use_afterstate: bool, eval_type: str, leaf_mode: str, depth: int) -> int:
    value_func = get_value_func(eval_type)
    agent = ExpectimaxAgent(use_afterstate=use_afterstate, value_func=value_func, leaf_mode=leaf_mode)
    action, _, _ = agent.get_best_action(board(raw), max_depth=depth)
    return action


def decoupled_action_value(raw: int, action: int, value_func) -> float:
    after_raw, reward = apply_action(raw, action)
    if reward == -1:
        return -float("inf")
    return reward + value_func(board(after_raw))


def compute_regret(config) -> dict:
    board.lookup.init()
    pressure_boards = generate_pressure_boards(config.regret_boards, config.seed + 10_000)
    decoupled_value = get_value_func("ntuple_afterstate")
    regrets, disagreements = [], 0

    for raw in progress(pressure_boards, desc="Action regret", leave=False):
        standard_action = best_action_for_raw(raw, False, "ntuple_state", "state", config.search_depth)
        decoupled_action = best_action_for_raw(raw, True, "ntuple_afterstate", "afterstate", config.search_depth)
        if standard_action != decoupled_action:
            disagreements += 1
            regrets.append(
                decoupled_action_value(raw, decoupled_action, decoupled_value)
                - decoupled_action_value(raw, standard_action, decoupled_value)
            )

    return {
        "boards": len(pressure_boards),
        "disagreements": disagreements,
        "disagreement_rate": disagreements / max(1, len(pressure_boards)),
        "average_regret": safe_mean(regrets),
        "max_regret": max(regrets) if regrets else 0.0,
    }


def run_experiment(config):
    rows, details = [], {"games": {}, "regret": {}}
    for exp_id, name, algorithm, use_afterstate, eval_type, leaf_mode, depth in SEARCH_CONFIGS:
        summary, records = evaluate_config(
            config,
            exp_id,
            name,
            algorithm,
            use_afterstate,
            eval_type,
            leaf_mode,
            depth,
        )
        rows.append(summary)
        details["games"][exp_id] = records
        print(
            f"{exp_id} {name}: score={summary['average_score']:.1f}, "
            f"2048={summary['rate_2048']:.1%}, b_eff={summary['b_eff']:.2f}"
        )

    regret = compute_regret(config)
    details["regret"] = regret
    rows.append(
        {
            "experiment": "1-Regret",
            "variant": "1-D vs 1-G Action Disagreement Regret",
            "algorithm": "expectimax",
            "depth": config.search_depth,
            "leaf_mode": "mixed",
            "average_score": 0.0,
            "average_steps": 0.0,
            "time_per_step_ms": 0.0,
            "rate_1024": 0.0,
            "rate_2048": 0.0,
            "rate_4096": 0.0,
            "b_eff": 0.0,
            "compression_ratio": 0.0,
            "disagreement_rate": regret["disagreement_rate"],
            "average_regret": regret["average_regret"],
            "max_regret": regret["max_regret"],
        }
    )

    paths = write_result_bundle(config.output_dir, "search", config, rows, details)
    print(f"Search results saved: {paths['md']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run phase-1 Expectimax/Greedy experiments.")
    add_common_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()
