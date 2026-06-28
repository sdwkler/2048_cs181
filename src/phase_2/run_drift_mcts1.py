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
from src.phase_2.mcts3_node1 import MCTSAgent

DRIFT_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
SIMULATIONS = 500      
MACRO_GAMES = 20

MCTS_CONFIGS = [
    ("MCTS-State", "State MCTS", False),
    ("MCTS-After", "Afterstate MCTS", True),
]

def drift_game_worker(args):
    game_seed, use_afterstate, env_p4_prob = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    
    agent = MCTSAgent(use_afterstate=use_afterstate, seed=game_seed + 1000, p4_prob=env_p4_prob)
    
    b = board()
    popup_with_rng(b, rng, p4=env_p4_prob) 
    popup_with_rng(b, rng, p4=env_p4_prob)

    score, steps = 0, 0
    step_times, q_vars, entropies, depths = [], [], [], []
    
    while True:
        start = time.perf_counter()
        
        # 完美对齐 MCTS 核心引擎的四元组返回！
        action, root_q_var, visit_entropy, step_max_depth = agent.get_best_action(b, num_simulations=SIMULATIONS)
        
        step_times.append(time.perf_counter() - start)
        q_vars.append(root_q_var)
        entropies.append(visit_entropy)
        depths.append(step_max_depth)
        
        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1: break
        
        score += reward
        steps += 1
        popup_with_rng(next_b, rng, p4=env_p4_prob)
        b = next_b

    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(step_times),
        "micro_variance": safe_mean(q_vars),
        "micro_entropy": safe_mean(entropies),
        "micro_depth": float(max(depths)) if depths else 0.0
    }

def generate_mcts_plots(rows: list[dict], picture_dir: str):
    os.makedirs(picture_dir, exist_ok=True)
    
    x_vals = sorted(list(set([r["p4_prob"] for r in rows])))
    models = ["MCTS-State", "MCTS-After"]
    
    def get_y_seq(exp_id, metric):
        m_rows = sorted([r for r in rows if r["experiment"] == exp_id], key=lambda x: x["p4_prob"])
        return np.array([r[metric] for r in m_rows])

    basic_metrics = [
        ("average_score", "Average Score", "Score Degradation under Prior Mismatch"),
        ("average_steps", "Survival Steps", "Game Lifespan under Prior Mismatch"),
        ("micro_depth", "Peak Tree Structure Depth", "Max Simulation Horizon Reached"), 
        ("micro_variance", "Action Discriminability (Root Q Std)", "Cognitive Clarity under Noise (Higher is Better)"),
        ("micro_entropy", "Visit Entropy (Chaos)", "Search Confusion (Lower is Better)"),
    ]
    
    for key, y_label, title in basic_metrics:
        plt.figure(figsize=(9, 6))
        y_state = get_y_seq("MCTS-State", key)
        y_after = get_y_seq("MCTS-After", key)
        
        plt.plot(x_vals, y_state, label="MCTS-State", color="red", marker='o', linewidth=2.5)
        plt.plot(x_vals, y_after, label="MCTS-After", color="blue", marker='^', linewidth=2.5)
        
        if key in ["micro_entropy"]:
            plt.fill_between(x_vals, y_after, y_state, where=(y_state > y_after), interpolate=True, color='red', alpha=0.15, label='State Chaos Penalty')
        else:
            plt.fill_between(x_vals, y_state, y_after, where=(y_after > y_state), interpolate=True, color='blue', alpha=0.1, label='Afterstate Advantage Margin')
        
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("Actual Environment P(4)", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, f"{key}_drift.png"), dpi=300)
        plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(18, 6))
    fig.suptitle("Max Tile Achievement Rates across Drift Environments", fontsize=18, fontweight='bold')
    
    tile_levels = ["<1024", "1024", "2048", "4096", "8192", "16384", "32768+"]
    tile_colors = ["#e0e0e0", "#fbd390", "#f67c5f", "#f65e3b", "#edcf72", "#edc850", "#edc53f"]
    axes_flat = axes.flatten()
    
    for idx, exp_id in enumerate(models):
        if idx >= len(axes_flat): break
        
        ax = axes_flat[idx]
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
                
                if tile == "<1024": val = 1.0 - r_1024
                elif tile == "1024": val = r_1024 - r_2048
                elif tile == "2048": val = r_2048 - r_4096
                elif tile == "4096": val = r_4096 - r_8192
                elif tile == "8192": val = r_8192 - r_16384
                elif tile == "16384": val = r_16384 - r_32768
                else: val = r_32768
                
                heights.append(max(0, val)) 
            
            ax.bar(x_pos, heights, bottom=bottoms, label=tile, color=tile_colors[t_idx], edgecolor='white', width=0.7)
            bottoms += heights
            
        ax.set_title(exp_id, fontsize=14)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_ylabel("Percentage of Games", fontsize=12)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        if idx == len(models) - 1:
            ax.legend(title="Max Tile", bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "tile_achievement_rates_drift.png"), dpi=300, bbox_inches='tight')
    plt.close()

def run_experiment(config):
    rows = []
    print(f"========== MCTS Prior Mismatch Robustness Test ==========")
    print(f"Agent Internal Belief Locked at: P(4) = 0.1")

    for env_p4 in DRIFT_PROBS:
        print(f"\n>>> Environment Drifting To: P(4) = {env_p4:.2f} <<<")
        for exp_id, name, use_after in MCTS_CONFIGS:
            
            args_list = [(config.seed + int(env_p4*100) + i, use_after, env_p4) for i in range(MACRO_GAMES)]
            records = []
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                futures = [executor.submit(drift_game_worker, args) for args in args_list]
                for future in progress(as_completed(futures), total=len(futures), desc=f"{exp_id}", leave=False):
                    records.append(future.result())

            summary = summarize_games(records)
            avg_variance = safe_mean(r["micro_variance"] for r in records)
            avg_entropy = safe_mean(r["micro_entropy"] for r in records)
            avg_depth = safe_mean(r["micro_depth"] for r in records)

            row = {
                "p4_prob": env_p4, "experiment": exp_id, "variant": name,
                **summary,
                "micro_variance": avg_variance,
                "micro_entropy": avg_entropy,
                "micro_depth": avg_depth
            }
            rows.append(row)
            
            print(f"  [{exp_id}] Score: {summary['average_score']:.0f} | Time: {summary['time_per_step_ms']:.1f}ms")
            print(f"  [Anatomy Tracker] Action Discriminability (Std): {avg_variance:.2f} | Entropy: {avg_entropy:.2f} | Tree Depth: {avg_depth:.0f}")

    paths = write_result_bundle(config.output_dir, "mcts_drift_robustness", config, rows, {})
    
    picture_dir = os.path.join("models", "phase2", "picture", "mcts")
    print(f"\n🎨 Generating Academic Plots in {picture_dir} ...")
    generate_mcts_plots(rows, picture_dir)
    print(f"✅ All Tests Completed! Pictures saved at: {picture_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_common_args(parser) 
    args = parser.parse_args()
    config = config_from_args(args)
    config = config.__class__(**{**config.__dict__, "output_dir": os.path.join("models", "phase2", "results")})
    run_experiment(config)