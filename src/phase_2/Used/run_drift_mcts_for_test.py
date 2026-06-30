import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.environments.base_env import board
from src.common import (
    ACTION_NAMES,
    add_common_args_2,
    config_from_args,
    entropy_from_counts,
    generate_pressure_boards,
    play_policy,
    progress,
    safe_mean,
    summarize_games,
    write_result_bundle,
)
from src.phase_2.Used.mcts3_node_for_test import MCTSAgent

# 动态生成实验 ID，保持干净直观
def planning_experiment_id(use_afterstate: bool, rsz_mode: bool, rollout_index: int) -> str:
    if not use_afterstate:
        prefix = "State"
    elif not rsz_mode:
        prefix = "Afterstate-Base"
    else:
        prefix = "RSZ-Afterstate"
    return f"3-{prefix}-{rollout_index}"


def run_single_board(args):
    raw, use_afterstate, rsz_mode, simulations, seed = args
    board.lookup.init()
    agent = MCTSAgent(use_afterstate=use_afterstate, rsz_mode=rsz_mode, seed=seed)
    start = time.perf_counter()
    action, root_variance, visit_entropy = agent.get_best_action(board(raw), num_simulations=simulations)
    elapsed_ms = 1000.0 * (time.perf_counter() - start)
    return {
        "action": action,
        "root_action_value_variance": root_variance,
        "root_action_values": {
            ACTION_NAMES[action]: value
            for action, value in agent.last_root_action_values.items()
        },
        "root_action_visits": {
            ACTION_NAMES[action]: visits
            for action, visits in agent.last_root_action_visits.items()
        },
        "visit_policy_entropy": visit_entropy,
        "time_per_decision_ms": elapsed_ms,
    }


def run_stability_for_board(raw: int, use_afterstate: bool, rsz_mode: bool, simulations: int, seed: int, repeats: int) -> dict:
    counts = {a: 0 for a in range(4)}
    for rep in range(repeats):
        result = run_single_board((raw, use_afterstate, rsz_mode, simulations, seed + rep * 9973))
        counts[result["action"]] += 1
    return {
        "policy_recommendation_entropy": entropy_from_counts(counts),
        "action_counts": {ACTION_NAMES[action]: count for action, count in counts.items() if count},
    }


def planning_game_worker(args):
    game_seed, use_afterstate, rsz_mode, simulations, max_game_steps = args
    board.lookup.init()
    agent = MCTSAgent(use_afterstate=use_afterstate, rsz_mode=rsz_mode, seed=game_seed + 91_337)

    def choose_action(b: board) -> int:
        action, _, _ = agent.get_best_action(b, num_simulations=simulations)
        return action

    return play_policy(game_seed, choose_action, max_steps=max_game_steps)


def evaluate_variant(config, rollout_index: int, simulations: int, use_afterstate: bool, rsz_mode: bool, variant_name: str, pressure_boards: list[int]):
    exp_id = planning_experiment_id(use_afterstate, rsz_mode, rollout_index)
    variant = f"{variant_name} MCTS ({simulations} rollouts)"
    
    board_args = [
        (raw, use_afterstate, rsz_mode, simulations, config.seed + rollout_index * 100_000 + i)
        for i, raw in enumerate(pressure_boards)
    ]
    single_records = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(run_single_board, args) for args in board_args]
        for future in progress(as_completed(futures), total=len(futures), desc=variant[:24], leave=False):
            single_records.append(future.result())

    stability_records = []
    for i, raw in enumerate(progress(pressure_boards, desc=f"{exp_id} stability", leave=False)):
        stability_records.append(
            run_stability_for_board(
                raw,
                use_afterstate,
                rsz_mode,
                simulations,
                config.seed + 2_000_000 + rollout_index * 100_000 + i,
                config.planning_stability_runs,
            )
        )

    game_args = [
        (config.seed + 3_000_000 + rollout_index * 10_000 + i, use_afterstate, rsz_mode, simulations, config.max_game_steps)
        for i in range(config.planning_eval_games)
    ]
    game_records = []
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(planning_game_worker, args) for args in game_args]
        for future in progress(as_completed(futures), total=len(futures), desc=f"{exp_id} games", leave=False):
            game_records.append(future.result())

    game_summary = summarize_games(game_records)
    row = {
        "experiment": exp_id,
        "variant": variant,
        "rollouts": simulations,
        **game_summary,
        "root_action_value_variance": safe_mean(r["root_action_value_variance"] for r in single_records),
        "visit_policy_entropy": safe_mean(r["visit_policy_entropy"] for r in single_records),
        "policy_recommendation_entropy": safe_mean(r["policy_recommendation_entropy"] for r in stability_records),
        "single_step_time_ms": safe_mean(r["time_per_decision_ms"] for r in single_records),
    }
    return row, {"single_step": single_records, "stability": stability_records, "games": game_records}


def run_experiment(config):
    pressure_boards = generate_pressure_boards(config.planning_boards, config.seed + 20_000)
    rows, details = [], {"boards": pressure_boards, "variants": {}}
    
    # 【核心：严谨的三组对比消融】
    # 1. (False, False) -> State MCTS
    # 2. (True, False)  -> 原生 Afterstate MCTS
    # 3. (True, True)   -> RSZ 论文版 Afterstate MCTS
    modes = [
        (False, False, "State"),
        (True, False, "Afterstate-Base"),
        (True, True, "RSZ-Afterstate")
    ]
    
    for rollout_index, simulations in enumerate(config.planning_rollouts):
        for use_afterstate, rsz_mode, variant_name in modes:
            row, detail = evaluate_variant(config, rollout_index, simulations, use_afterstate, rsz_mode, variant_name, pressure_boards)
            rows.append(row)
            details["variants"][row["experiment"]] = detail
            
            # 【核心修复】：使用安全的 get 方法提取字段打印，避免底层 summary 数据结构不兼容导致崩溃
            score = row.get('average_score', 0.0)
            time_ms = row.get('single_step_time_ms', 0.0)
            print(f"[{row['experiment']}] {row['variant']}: Score={score:.1f}, Time/Step={time_ms:.2f}ms")

    paths = write_result_bundle(config.output_dir, "planning", config, rows, details)
    print(f"Planning results saved: {paths['md']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run phase-1 RSZ Ablation experiments.")
    add_common_args_2(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()