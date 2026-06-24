# 文件路径: src/phase_2/run_drift_mcts.py
import argparse
import os
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.environments.base_env import board
from src.common import (
    ACTION_NAMES, add_common_args, config_from_args, generate_pressure_boards,
    max_tile_value, popup_with_rng, progress, safe_mean, summarize_games, write_result_bundle
)
# 导入我们刚刚创建的感知漂移的 MCTS 引擎
from src.phase_2.mcts3_new_node import MCTSAgent

# 漂移概率梯度
DRIFT_PROBS = [0.1, 0.3, 0.5, 0.7, 0.9]
SIMULATIONS = 2000      # 极度受限的算力预算
MACRO_GAMES = 30        # 宏观：每个梯度跑 30 局完整游戏
MICRO_BOARDS = 100      # 微观：每个梯度跑 100 个高压残局截面

MCTS_CONFIGS = [
    ("MCTS-State", "State MCTS", False),
    ("MCTS-After", "Afterstate MCTS", True),
]

# ==========================================
# 1. 微观解剖任务 (Micro Snapshot)
# ==========================================
def micro_snapshot_worker(args):
    raw, use_afterstate, p4_prob, seed = args
    board.lookup.init()
    agent = MCTSAgent(use_afterstate=use_afterstate, seed=seed, p4_prob=p4_prob)
    # 【接收 4 个返回值，包含了树的最大深度】
    action, root_variance, visit_entropy, max_depth = agent.get_best_action(board(raw), num_simulations=SIMULATIONS)
    return {
        "root_variance": root_variance,
        "visit_entropy": visit_entropy,
        "max_depth": max_depth
    }

# ==========================================
# 2. 宏观对局任务 (Macro Full-Game)
# ==========================================
def macro_game_worker(args):
    game_seed, use_afterstate, p4_prob = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    
    agent = MCTSAgent(use_afterstate=use_afterstate, seed=game_seed + 1000, p4_prob=p4_prob)
    b = board()
    popup_with_rng(b, rng, p4=p4_prob)
    popup_with_rng(b, rng, p4=p4_prob)

    score, steps = 0, 0
    step_times = []
    
    while True:
        start = time.perf_counter()
        action, _, _, _ = agent.get_best_action(b, num_simulations=SIMULATIONS)
        step_times.append(time.perf_counter() - start)
        
        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1: break
        
        score += reward
        steps += 1
        popup_with_rng(next_b, rng, p4=p4_prob)
        b = next_b

    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(step_times),
    }

# ==========================================
# 📊 自动绘图模块 (输出至 mcts 文件夹)
# ==========================================
def generate_mcts_plots(rows: list[dict], picture_dir: str):
    os.makedirs(picture_dir, exist_ok=True)
    models = ["MCTS-State", "MCTS-After"]
    colors = {"MCTS-State": "red", "MCTS-After": "blue"}

    # --- 1. 常规折线图 ---
    metrics = [
        ("average_score", "Average Score", "Macro: Score Degradation under Drift"),
        ("average_steps", "Survival Steps", "Macro: Game Lifespan under Drift"),
        ("micro_variance", "Root Q-Value Variance", "Micro: Vanishing Root Variance (Action Blindness)"),
        ("micro_depth", "Max Search Depth", "Micro: Tree Topology Collapse (Shallow Bush vs Deep Pine)")
    ]
    
    for key, y_label, title in metrics:
        plt.figure(figsize=(9, 6))
        for exp_id in models:
            m_rows = sorted([r for r in rows if r["experiment"] == exp_id], key=lambda x: x["p4_prob"])
            if not m_rows: continue
            x_vals = [r["p4_prob"] for r in m_rows]
            y_vals = [r[key] for r in m_rows]
            plt.plot(x_vals, y_vals, label=exp_id, color=colors[exp_id], marker='o', linewidth=2.5)
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("P(4) Spawn Probability", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, f"{key}_drift.png"), dpi=300)
        plt.close()

    # --- 2. 互斥堆叠柱状图 (Tile Rates) ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Max Tile Achievement Rates (MCTS Dual-Track)", fontsize=16, fontweight='bold')
    tile_levels = ["<1024", "1024", "2048", "4096", "8192", "16384+"]
    tile_colors = ["#e0e0e0", "#fbd390", "#f67c5f", "#f65e3b", "#edcf72", "#edc850"]
    
    for idx, exp_id in enumerate(models):
        ax = axes[idx]
        m_rows = sorted([r for r in rows if r["experiment"] == exp_id], key=lambda x: x["p4_prob"])
        if not m_rows: continue
        
        x_labels = [f"{r['p4_prob']:.1f}" for r in m_rows]
        x_pos = np.arange(len(x_labels))
        bottoms = np.zeros(len(x_labels))
        
        for t_idx, tile in enumerate(tile_levels):
            heights = []
            for r in m_rows:
                r_1024 = r.get("rate_1024", 0)
                r_2048 = r.get("rate_2048", 0)
                r_4096 = r.get("rate_4096", 0)
                r_8192 = r.get("rate_8192", 0)
                r_16384 = r.get("rate_16384", 0)
                r_32768 = r.get("rate_32768", 0)
                r_mega = r_16384 + r_32768
                
                if tile == "<1024": val = 1.0 - r_1024
                elif tile == "1024": val = r_1024 - r_2048
                elif tile == "2048": val = r_2048 - r_4096
                elif tile == "4096": val = r_4096 - r_8192
                elif tile == "8192": val = r_8192 - r_mega
                else: val = r_mega
                heights.append(max(0, val))
            
            ax.bar(x_pos, heights, bottom=bottoms, label=tile, color=tile_colors[t_idx], edgecolor='white', width=0.6)
            bottoms += heights
            
        ax.set_title(exp_id, fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels)
        ax.set_ylabel("Percentage", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        if idx == 1:
            ax.legend(title="Max Tile", loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.9, 0.95])
    plt.savefig(os.path.join(picture_dir, "tile_stacked_rates.png"), dpi=300)
    plt.close()


def run_experiment(config):
    rows = []
    print(f"========== MCTS Dual-Track Drift Test ==========")
    pressure_boards = generate_pressure_boards(MICRO_BOARDS, config.seed + 888)

    for p4 in DRIFT_PROBS:
        print(f"\n>>> Simulating Environment: P(4) = {p4:.2f} <<<")
        for exp_id, name, use_after in MCTS_CONFIGS:
            
            # 1. 运行微观解剖 (Micro)
            micro_args = [(raw, use_after, p4, config.seed + i) for i, raw in enumerate(pressure_boards)]
            micro_records = []
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                futures = [executor.submit(micro_snapshot_worker, args) for args in micro_args]
                for future in progress(as_completed(futures), total=len(futures), desc=f"Micro {exp_id}", leave=False):
                    micro_records.append(future.result())
                    
            avg_variance = safe_mean(r["root_variance"] for r in micro_records)
            avg_depth = safe_mean(r["max_depth"] for r in micro_records)

            # 2. 运行宏观整局 (Macro)
            macro_args = [(config.seed + int(p4*100) + i, use_after, p4) for i in range(MACRO_GAMES)]
            macro_records = []
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                futures = [executor.submit(macro_game_worker, args) for args in macro_args]
                for future in progress(as_completed(futures), total=len(futures), desc=f"Macro {exp_id}", leave=False):
                    macro_records.append(future.result())

            summary = summarize_games(macro_records)
            row = {
                "p4_prob": p4, "experiment": exp_id, "variant": name,
                **summary,
                "micro_variance": avg_variance,
                "micro_depth": avg_depth
            }
            rows.append(row)
            
            print(f"  [{exp_id}] Score: {summary['average_score']:.0f} | Steps: {summary['average_steps']:.0f}")
            print(f"  [Micro Anatomy] Q-Var: {avg_variance:.2f} | Max Depth: {avg_depth:.2f}")

    paths = write_result_bundle(config.output_dir, "mcts_drift_dual", config, rows, {})
    
    picture_dir = os.path.join("models", "phase2", "picture", "mcts")
    print(f"\n🎨 Generating Academic Plots in {picture_dir} ...")
    generate_mcts_plots(rows, picture_dir)
    print(f"✅ All Tests Completed! Pictures saved at: {picture_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser) 
    args = parser.parse_args()
    config = config_from_args(args)
    # 强制隔离输出目录
    config = config.__class__(**{**config.__dict__, "output_dir": os.path.join("models", "phase2", "results")})
    run_experiment(config)