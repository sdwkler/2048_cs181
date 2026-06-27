import argparse
import os
import random
import sys
import time
import numpy as np
import matplotlib.pyplot as plt  # 【新增绘图库】
from concurrent.futures import ProcessPoolExecutor, as_completed

# 修复路径跳层问题，保证能找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.environments.base_env import board
from src.ntuple.feature_base import diff_pattern, feature, learning, pattern
from src.ntuple.loader import fast_mmap_load
from src.common import (
    ACTION_NAMES, add_common_args, config_from_args, generate_pressure_boards,
    max_tile_value, popup_with_rng, progress, safe_mean, summarize_games, write_result_bundle
)
from src.evaluators import HeuristicEvaluator, NTupleEvaluator
from src.phase_1.search.expectimax import ExpectimaxAgent 

_PROCESS_MODEL_CACHE = {}

MIDDLE_CONFIGS = [
    ("M1-State-Heur", "State + Heuristic", False, "heuristic", "state", 2),
    ("M2-After-Heur", "Afterstate + Heuristic", True, "heuristic", "afterstate", 2),

    ("M3-State-NT", "State + StateNTuple", False, "ntuple_state", "state", 2),
    # ("M4-State-ANT", "State + AfterstateNTuple", False, "ntuple_afterstate", "state", 2),
    # ("M5-After-NT", "Afterstate + StateNTuple", True, "ntuple_state", "afterstate", 2),
    ("M6-After-ANT", "Afterstate + AfterstateNTuple", True, "ntuple_afterstate", "afterstate", 2)
]

# 环境漂移梯度：10% 到 90%
DRIFT_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
GAMES_PER_ENV = 10

def build_ntuple(eval_type: str) -> learning:
    original_alloc = feature.alloc
    old_stdout = sys.stdout
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    try:
        feature.alloc = staticmethod(lambda num: range(num))
        tdl = learning()
        shapes = [[0, 1, 2, 3, 4, 5], [4, 5, 6, 7, 8, 9], [0, 1, 2, 4, 5, 6], [4, 5, 6, 8, 9, 10]]
        for shape in shapes: tdl.add_feature(pattern(shape))
        for shape in shapes: tdl.add_feature(diff_pattern(shape))
    finally:
        feature.alloc = original_alloc
        sys.stdout = old_stdout
        devnull.close()

    weight_file = "models/2048_afterstate.bin" if eval_type == "ntuple_afterstate" else "models/2048_state.bin"
    fast_mmap_load(tdl, weight_file)
    return tdl

def get_value_func(eval_type: str):
    if eval_type == "heuristic":
        return HeuristicEvaluator().evaluate
    if eval_type not in _PROCESS_MODEL_CACHE:
        _PROCESS_MODEL_CACHE[eval_type] = build_ntuple(eval_type)
    return NTupleEvaluator(_PROCESS_MODEL_CACHE[eval_type]).evaluate

def build_search_agent(use_afterstate: bool, eval_type: str, leaf_mode: str, p4_prob: float):
    return ExpectimaxAgent(
        use_afterstate=use_afterstate, 
        value_func=get_value_func(eval_type), 
        leaf_mode=leaf_mode,
        use_pruning=False, 
        p4_prob=p4_prob  # 核心死锁：AI 主观信仰锁死为 0.1
    )

def search_game_worker(args):
    game_seed, use_afterstate, eval_type, leaf_mode, search_depth, p4_prob = args
    board.lookup.init()
    rng = random.Random(game_seed)
    random.seed(game_seed)
    
    agent = build_search_agent(use_afterstate, eval_type, leaf_mode, p4_prob)
    b = board()
    popup_with_rng(b, rng, p4=p4_prob)
    popup_with_rng(b, rng, p4=p4_prob)

    score, steps = 0, 0
    step_times, compressions = [], []
    while True:
        start = time.perf_counter()
        action, comp_ratio, _, _ = agent.get_best_action(b, max_depth=search_depth)
        step_times.append(time.perf_counter() - start)
        compressions.append(comp_ratio)

        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1: break
        score += reward
        steps += 1
        popup_with_rng(next_b, rng, p4=p4_prob) # 真实环境概率
        b = next_b

    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(step_times),
        "compression_ratio": safe_mean(compressions),
    }

def compute_regret_drift(p4_prob: float, num_boards: int, seed: int, workers: int):
    boards = generate_pressure_boards(num_boards, seed)
    agent_state = build_search_agent(False, "ntuple_state", "state", p4_prob)
    agent_after = build_search_agent(True, "ntuple_afterstate", "afterstate", p4_prob)
    
    disagreements, regrets = 0, []
    for raw in boards:
        b = board(raw)
        action_s, _, _, _ = agent_state.get_best_action(b, max_depth=3)
        action_a, _, _, _ = agent_after.get_best_action(b, max_depth=3)
        if action_s != action_a:
            disagreements += 1
            regrets.append(1.0)
            
    return {"disagreement_rate": disagreements / len(boards) if boards else 0}


# ======================================================================
# 📊 【高能核心】：论文级自动绘图模块
# ======================================================================
def generate_plots(rows: list[dict], picture_dir: str):
    os.makedirs(picture_dir, exist_ok=True)
    
    # 包含了你所有的 6 个实验
    models = ["M1-State-Heur", "M2-After-Heur", "M3-State-NT", "M4-State-ANT", "M5-After-NT", "M6-After-ANT"]
    colors = {
        "M1-State-Heur": "gray", "M2-After-Heur": "orange", 
        "M3-State-NT": "red", "M4-State-ANT": "brown", 
        "M5-After-NT": "green", "M6-After-ANT": "purple"
    }

    # --- 1. 折线图 (Score, Time, Compression) ---
    metrics = [
        ("average_score", "Average Score", "Zero-Shot Score Degradation under Pressure"),
        ("time_per_step_ms", "Time per Step (ms) [Log Scale]", "Computational Breakdown Analysis"),
        ("compression_ratio", "Compression Ratio (Lower is Better)", "Topology Folding Rate vs Randomness")
    ]
    
    for key, y_label, title in metrics:
        plt.figure(figsize=(10, 6))
        for exp_id in models:
            m_rows = sorted([r for r in rows if r["experiment"] == exp_id], key=lambda x: x["p4_prob"])
            if not m_rows: continue
            x_vals = [r["p4_prob"] for r in m_rows]
            y_vals = [r[key] for r in m_rows]
            # 防御性字典获取，避免某些键拼写错误导致崩溃
            plt.plot(x_vals, y_vals, label=exp_id, color=colors.get(exp_id, "black"), marker='o', linewidth=2.5)
            
        plt.title(title, fontsize=14, fontweight='bold')
        plt.xlabel("P(4) Spawn Probability (Environment Drift)", fontsize=12)
        plt.ylabel(y_label, fontsize=12)
        if key == "time_per_step_ms":
            plt.yscale("log")  # 【学术细节】时间爆炸必须用对数坐标！
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, f"{key}_drift.png"), dpi=300)
        plt.close()

    # --- 2. 遗憾值 (Disagreement Rate) 折线图 ---
    regret_rows = sorted([r for r in rows if r["experiment"] == "M-Regret"], key=lambda x: x["p4_prob"])
    if regret_rows:
        x_vals = [r["p4_prob"] for r in regret_rows]
        y_vals = [r["disagreement_rate"] for r in regret_rows]
        plt.figure(figsize=(10, 6))
        plt.plot(x_vals, y_vals, label="Decision Divergence (State vs Afterstate)", color="purple", marker='s', linewidth=2.5)
        plt.title("Regret Amplification: Divergence in High-Risk Environments", fontsize=14, fontweight='bold')
        plt.xlabel("P(4) Spawn Probability", fontsize=12)
        plt.ylabel("Disagreement Rate (%)", fontsize=12)
        plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, "regret_divergence.png"), dpi=300)
        plt.close()

    # --- 3. 互斥堆叠柱状图 (Stacked Bar Chart for Tile Rates) ---
    # 【神级修复】：2 行 3 列的布局以兼容 6 个模型
    fig, axes = plt.subplots(2, 3, figsize=(18, 10)) 
    fig.suptitle("Max Tile Achievement Rates across Drift Environments", fontsize=18, fontweight='bold')
    
    tile_levels = ["<1024", "1024", "2048", "4096", "8192", "16384", "32768+"]
    tile_colors = ["#e0e0e0", "#fbd390", "#f67c5f", "#f65e3b", "#edcf72", "#edc850", "#edc53f"]
    
    # 展平 axes 数组，可以直接用 idx 0~5 进行无缝索引
    axes_flat = axes.flatten()
    
    for idx, exp_id in enumerate(models):
        if idx >= len(axes_flat): break  # 双重保护，防止数组越界
        
        ax = axes_flat[idx]
        m_rows = sorted([r for r in rows if r["experiment"] == exp_id], key=lambda x: x["p4_prob"])
        if not m_rows: continue
        
        x_labels = [f"{r['p4_prob']:.1f}" for r in m_rows]
        x_pos = np.arange(len(x_labels))
        bottoms = np.zeros(len(x_labels))
        
        for t_idx, tile in enumerate(tile_levels):
            heights = []
            for r in m_rows:
                # 兼容未包含更高方块统计的情况，兜底为 0
                r_1024 = r.get("rate_1024", 0)
                r_2048 = r.get("rate_2048", 0)
                r_4096 = r.get("rate_4096", 0)
                r_8192 = r.get("rate_8192", 0)
                r_16384 = r.get("rate_16384", 0)
                r_32768 = r.get("rate_32768", 0)
                
                # 【神级逻辑】：计算各层级的“净停留率”
                if tile == "<1024": val = 1.0 - r_1024
                elif tile == "1024": val = r_1024 - r_2048
                elif tile == "2048": val = r_2048 - r_4096
                elif tile == "4096": val = r_4096 - r_8192
                elif tile == "8192": val = r_8192 - r_16384
                elif tile == "16384": val = r_16384 - r_32768
                else: val = r_32768
                
                heights.append(max(0, val)) # 防止舍入误差导致负数
            
            ax.bar(x_pos, heights, bottom=bottoms, label=tile, color=tile_colors[t_idx], edgecolor='white', width=0.7)
            bottoms += heights
            
        ax.set_title(exp_id, fontsize=12)
        ax.set_xticks(x_pos)
        # 旋转横坐标标签以防拥挤
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_ylabel("Percentage of Games", fontsize=11)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0%}'.format(y)))
        
        # 只在第一张子图画总图例，保持画面整洁
        if idx == 0:
            ax.legend(title="Final Max Tile", loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.96, 0.96])
    plt.savefig(os.path.join(picture_dir, "tile_stacked_rates.png"), dpi=300)
    plt.close()


def run_experiment(config):
    rows = []
    print(f"========== Starting Middle Layer Environment Drift Test ==========")
    
    for p4 in DRIFT_PROBS:
        print(f"\n>>> Simulating Environment: P(4) = {p4:.2f} <<<")
        for exp_id, name, use_after, eval_type, leaf_mode, depth in MIDDLE_CONFIGS:
            args_list = [
                (config.seed + int(p4*100) + i, use_after, eval_type, leaf_mode, depth, p4)
                for i in range(GAMES_PER_ENV)
            ]
            records = []
            with ProcessPoolExecutor(max_workers=config.workers) as executor:
                futures = [executor.submit(search_game_worker, args) for args in args_list]
                for future in progress(as_completed(futures), total=len(futures), desc=name[:20], leave=False):
                    records.append(future.result())

            summary = summarize_games(records)
            row = {
                "p4_prob": p4, "experiment": exp_id, "variant": name,
                **summary, "compression_ratio": safe_mean(r["compression_ratio"] for r in records),
            }
            rows.append(row)
            print(f"  [{exp_id}] Score: {summary['average_score']:.0f} | Time/step: {summary['time_per_step_ms']:.2f}ms | Comp_Ratio: {row['compression_ratio']:.3f}")
        
        regret_stats = compute_regret_drift(p4, num_boards=100, seed=config.seed + int(p4*100), workers=config.workers)
        print(f"  [Regret Analysis] Disagreement Rate: {regret_stats['disagreement_rate']:.1%}")
        rows.append({
            "p4_prob": p4, "experiment": "M-Regret", "variant": "Disagreement Rate (State vs Afterstate)",
            "average_score": 0.0, "time_per_step_ms": 0.0, "compression_ratio": 0.0,
            "disagreement_rate": regret_stats["disagreement_rate"]
        })

    # 结果写入并立刻触发绘图
    paths = write_result_bundle(config.output_dir, "expectimax_drift", config, rows, {})
    
    # 调用绘图逻辑
    picture_dir = os.path.join("models", "phase2_new", "picture","search_drift")
    print(f"\n🎨 Generating Academic Plots in {picture_dir} ...")
    generate_plots(rows, picture_dir)
    
    print(f"✅ All Tests Completed! Data: {paths['md']}, Pictures: {picture_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 注意这里使用 common.py 里的 add_common_args
    add_common_args(parser) 
    args = parser.parse_args()
    config = config_from_args(args)
    
    # 【强制隔离输出目录】
    config = config.__class__(
        **{**config.__dict__, "output_dir": os.path.join("models", "phase2_new", "results","expectimax_drift")}
    )
    run_experiment(config)