# 文件路径: src/phase_3/planning/run_topology_experiment.py
import argparse
import os
import random
import sys
import time
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.environments.base_env import board
from src.common import (
    ACTION_NAMES, add_common_args_3, config_from_args, generate_pressure_boards,
    max_tile_value, popup_with_rng, progress, safe_mean, summarize_games, write_result_bundle
)
from src.phase_3.planning.mcts_topology_node import MCTSAgent

SIMULATIONS = 1000  

MCTS_CONFIGS = [
    ("3-H-S", "Heuristic + State", False, "heuristic", None, 5),
    ("3-H-A", "Heuristic + Afterstate", True, "heuristic", None, 5),
    ("3-N-S", "N-Tuple + State", False, "ntuple", "models/2048_state.bin", 0),
    ("3-N-A", "N-Tuple + Afterstate", True, "ntuple", "models/2048_afterstate.bin", 0)
]

# ==========================================
# 实验部分 1：宏观对局 (Macro Games)
# ==========================================
def macro_game_worker(args):
    game_seed, use_afterstate, eval_type, ntuple_path, rollout_limit = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    
    agent = MCTSAgent(
        use_afterstate=use_afterstate, 
        eval_type=eval_type,
        ntuple_path=ntuple_path,
        seed=game_seed + 1000, 
        p4_prob=0.1,
        rollout_limit=rollout_limit
    )
    
    b = board()
    popup_with_rng(b, rng, p4=0.1) 
    popup_with_rng(b, rng, p4=0.1)

    score, steps = 0, 0
    step_times, depths = [], []
    
    while True:
        start = time.perf_counter()
        
        # 【核心修复】：精准解包这 4 个返回值，提取正确的动作 action
        action, root_std, visit_entropy, step_max_depth = agent.get_best_action(b, num_simulations=SIMULATIONS)
        
        step_times.append(time.perf_counter() - start)
        depths.append(step_max_depth)
        
        next_b = board(b.raw)
        reward = next_b.move(action) # 现在 action 是纯粹的整数了，游戏正常进行！
        if reward == -1: break
        
        score += reward
        steps += 1
        popup_with_rng(next_b, rng, p4=0.1)
        b = next_b

    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(step_times),
        "macro_depth": float(safe_mean(depths))
    }

def run_macro_games(config):
    print(f"\n========== 【第一阶段】宏观对局测试 (Sims={SIMULATIONS}, 纯净得分) ==========")
    rows = []
    for cfg_id, name, use_after, eval_type, ntuple_path, r_limit in MCTS_CONFIGS:
        args_list = [(config.seed + i, use_after, eval_type, ntuple_path, r_limit) for i in range(config.search_games)]
        records = []
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(macro_game_worker, args) for args in args_list]
            for future in progress(as_completed(futures), total=len(futures), desc=f"{name[:15]}", leave=False):
                records.append(future.result())

        summary = summarize_games(records)
        avg_depth = safe_mean(r["macro_depth"] for r in records)
        row = {"experiment": cfg_id, "variant": name, **summary, "macro_depth": avg_depth}
        rows.append(row)
        
        print(f"[{cfg_id}] {name:<22} | 得分: {summary['average_score']:<7.0f} | 纵深: {avg_depth:<5.1f}层 | 耗时/步: {summary['time_per_step_ms']:.1f}ms")
    return rows


# ==========================================
# 实验部分 2：百盘静态剖面测试 (Static Topology Probe)
# ==========================================
def static_probe_worker(args):
    cfg_id, use_after, eval_type, path, r_limit, raw_board, seed = args
    board.lookup.init()
    agent = MCTSAgent(
        use_afterstate=use_after, eval_type=eval_type, ntuple_path=path,
        seed=seed, rollout_limit=r_limit, p4_prob=0.1
    )
    
    # 【核心修复】：接住函数返回的所有参数
    action, root_std, visit_entropy, step_max_depth = agent.get_best_action(board(raw_board), num_simulations=SIMULATIONS)
    
    return {
        "exp_id": cfg_id,
        "profile": agent.get_tree_profile(max_depth=12),
        "entropy": visit_entropy,       # 直接使用拿到的熵
        "max_depth": step_max_depth     # 直接使用拿到的深度
    }

def run_static_probe(config):
    print(f"\n========== 【第二阶段】100盘静态树剖面探针 (绝对公平横向对比) ==========")
    probe_boards = generate_pressure_boards(100, config.seed + 9999)
    
    results_by_exp = {cfg_id: {"profiles": [], "entropies": [], "depths": [], "name": name} 
                      for cfg_id, name, _, _, _, _ in MCTS_CONFIGS}
    
    tasks = []
    for cfg_id, name, use_after, eval_type, path, r_limit in MCTS_CONFIGS:
        for i, raw in enumerate(probe_boards):
            tasks.append((cfg_id, use_after, eval_type, path, r_limit, raw, config.seed + i * 1000))
            
    with ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(static_probe_worker, args) for args in tasks]
        for future in progress(as_completed(futures), total=len(futures), desc="Static Probing"):
            res = future.result()
            exp_id = res["exp_id"]
            results_by_exp[exp_id]["profiles"].append(res["profile"])
            results_by_exp[exp_id]["entropies"].append(res["entropy"])
            results_by_exp[exp_id]["depths"].append(res["max_depth"])

    rows = []
    for exp_id, data in results_by_exp.items():
        avg_entropy = safe_mean(data["entropies"])
        avg_depth = safe_mean(data["depths"])
        
        avg_profile = [0.0] * 13
        for d in range(13):
            avg_profile[d] = sum(p[d] for p in data["profiles"]) / len(data["profiles"])
            
        rows.append({
            "experiment": exp_id, "variant": data["name"],
            "probe_entropy": avg_entropy, "probe_depth": avg_depth, "layer_profile": avg_profile
        })
        
        print(f"\n>>> 剖面报告: [{exp_id}] {data['name']}")
        print(f"  先验熵 (宽度): {avg_entropy:.4f} | 穿透深度: {avg_depth:.2f}层")
        print("  每层真实节点分布 (Layer Nodes):")
        
        profile_str = " | ".join([f"L{d}:{avg_profile[d]:.1f}" for d in range(1, 9)])
        print(f"  {profile_str}")

    return rows

def main():
    parser = argparse.ArgumentParser(description="MCTS Topology & Macro Analysis")
    add_common_args_3(parser) 
    parser.add_argument("--games", type=int, default=50, help="宏观对局数")
    args = parser.parse_args()
    config = config_from_args(args)
    
    import dataclasses
    config = dataclasses.replace(
        config, search_games=args.games,
        output_dir=os.path.join("models", "phase_3", "results", "topology_tests")
    )
    
    board.lookup.init()
    
    macro_rows = run_macro_games(config)
    probe_rows = run_static_probe(config)
    
    final_rows = []
    for m in macro_rows:
        p = next(x for x in probe_rows if x["experiment"] == m["experiment"])
        merged = {**m, **p}
        final_rows.append(merged)
        
    paths = write_result_bundle(config.output_dir, "mcts_topology_analysis", config, final_rows, {})
    print(f"\n✅ 宏观与微观探测全部完成! 数据记录至: {paths['md']}")


if __name__ == "__main__":
    main()