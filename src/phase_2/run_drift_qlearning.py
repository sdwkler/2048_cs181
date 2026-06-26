# 文件路径: src/phase_2/run_drift_qlearning.py
from __future__ import annotations

import argparse
import glob
import math
import os
import pickle
import random
import sys
import time
import concurrent.futures

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from src.environments.base_env import board
from src.common import (
    add_common_args, config_from_args, apply_action, legal_actions, max_tile_value,
    popup_with_rng, progress, safe_mean, summarize_games, write_result_bundle, timestamp
)
# 直接引入你第一阶段定义的 Agent 和数据结构
from src.phase_1.learning.run_qlearning_new2 import QLearningAgent, StepResult, EXPERIMENTS

DRIFT_PROBS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
ZERO_SHOT_GAMES = 50
CONTINUAL_EPISODES = 20000
CONTINUAL_DRIFT_P4 = 0.5

# ======================================================================
# 核心工具：模型加载与继承
# ======================================================================
def get_latest_model_dir(base_dir="models/phrase_1/qlearning_runs"):
    """自动获取最新训练完成的第一阶段模型目录"""
    if not os.path.exists(base_dir):
        raise FileNotFoundError(f"Cannot find base directory: {base_dir}")
    dirs = [os.path.join(base_dir, d) for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    if not dirs:
        raise FileNotFoundError(f"No model directories found in {base_dir}")
    return max(dirs, key=os.path.getmtime)

def load_agent(path: str) -> QLearningAgent:
    """非侵入式反序列化第一阶段的 Agent"""
    with open(path, "rb") as f:
        payload = pickle.load(f)
    
    agent = QLearningAgent(payload["target_mode"], payload["feature_mode"], gamma=payload["gamma"])
    if agent.q_heads and payload.get("q_heads"):
        for head, state in zip(agent.q_heads, payload["q_heads"]):
            head.load_state_dict(state)
    if agent.v_head and payload.get("v_head"):
        agent.v_head.load_state_dict(payload["v_head"])
    if agent.m_head and payload.get("m_head"):
        agent.m_head.load_state_dict(payload["m_head"])
    return agent

class ContinualQLearningAgent(QLearningAgent):
    """
    优雅地继承以注入漂移环境 P(4) 和微小探索率，
    绝对不修改父类中严谨的特征提取和梯度更新逻辑。
    """
    def __init__(self, target_mode: str, feature_mode: str, p4_prob: float = 0.5):
        super().__init__(target_mode, feature_mode)
        self.p4_prob = p4_prob

    def choose_action(self, b: board, epsilon: float, rng: random.Random) -> tuple[int, bool]:
        """覆盖：引入 1% 探索率用于流形自适应"""
        actions = legal_actions(b)
        if not actions:
            return 0, True
        if rng.random() < epsilon:
            return rng.choice(actions), False
        # 兜底调用父类的绝对贪心决策
        return super().choose_action(b, 0.0, rng)[0], True

    def update_step(self, state_raw: int, action: int, rng: random.Random, td_lambda: float = 0.5) -> StepResult:
        """覆盖：仅为注入真实的 p4_prob 环境概率"""
        after_raw, reward = apply_action(state_raw, action)
        if reward == -1:
            return StepResult(0, 0.0, state_raw, True)

        next_b = board(after_raw)
        popup_with_rng(next_b, rng, p4=self.p4_prob) # <--- 核心注入点
        next_raw = next_b.raw

        feat = self.feature_board(state_raw, action, after_raw)

        # 后续严密复用原有逻辑
        if self.target_mode == "q":
            target = reward + self.gamma * self.best_action_value(next_raw)
            current = self.q_heads[action].estimate(feat)
            td_error = target - current
            self.q_heads[action].update(feat, td_error, self.gamma, td_lambda)
        elif self.target_mode == "v":
            target = self.gamma * self.best_action_value(next_raw)
            current = self.v_head.estimate(feat)
            td_error = target - current
            self.v_head.update(feat, td_error, self.gamma, td_lambda)
        elif self.target_mode == "mv":
            next_v = self.best_action_value(next_raw)
            current_v = self.v_head.estimate(feat)
            target_v = self.gamma * next_v
            td_error_v = target_v - current_v
            self.v_head.update(feat, td_error_v, self.gamma, td_lambda)

            scale = 0.01
            r_scaled = reward * scale
            next_v_scaled = next_v * scale 
            next_m = self.m_head.estimate(board(next_raw))
            current_m = self.m_head.estimate(feat)
            
            target_m = (r_scaled**2) + 2 * self.gamma * r_scaled * next_v_scaled + (self.gamma**2) * next_m
            td_error_m = target_m - current_m
            self.m_head.update(feat, td_error_m, self.gamma, td_lambda)
            td_error = td_error_v

        return StepResult(reward, td_error, next_raw, False)

# ======================================================================
# 实验 A: 零样本泛化 (Zero-Shot Robustness)
# ======================================================================
def zero_shot_worker(args):
    game_seed, model_path, env_p4 = args
    board.lookup.init()
    rng = random.Random(game_seed)
    
    agent = load_agent(model_path) # ε 默认为 0，纯推断
    
    b = board()
    popup_with_rng(b, rng, p4=env_p4)
    popup_with_rng(b, rng, p4=env_p4)

    score, steps = 0, 0
    step_times = [] # 【新增】用于记录每步耗时
    
    while True:
        start_t = time.perf_counter() # 【新增】开始计时
        action = agent.best_action(b)
        step_times.append(time.perf_counter() - start_t) # 【新增】记录决策耗时
        
        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1: break
        
        score += reward
        steps += 1
        popup_with_rng(next_b, rng, p4=env_p4)
        b = next_b

    return {
        "score": float(score),
        "steps": float(steps),
        "max_tile": float(max_tile_value(b)),
        "time_per_step_ms": 1000.0 * safe_mean(step_times) if step_times else 0.0 # 【修复】补上 summarize_games 需要的键
    }

def run_experiment_a_zero_shot(config, latest_model_dir: str):
    print(f"\n========== Experiment A: Zero-Shot Robustness (Cliff vs Smooth) ==========")
    rows = []
    
    for p4 in DRIFT_PROBS:
        print(f">>> Simulating Zero-Shot Drift: P(4) = {p4:.2f} <<<")
        for exp_id, variant, _, _ in EXPERIMENTS:
            model_path = os.path.join(latest_model_dir, f"qlearning_{exp_id.lower().replace('-', '')}.pkl")
            if not os.path.exists(model_path):
                continue
                
            args_list = [(config.seed + int(p4*100) + i, model_path, p4) for i in range(ZERO_SHOT_GAMES)]
            records = []
            with concurrent.futures.ProcessPoolExecutor(max_workers=config.workers) as executor:
                futures = [executor.submit(zero_shot_worker, args) for args in args_list]
                for future in progress(concurrent.futures.as_completed(futures), total=len(futures), desc=f"{exp_id}", leave=False):
                    records.append(future.result())

            summary = summarize_games(records)
            rows.append({
                "experiment": exp_id, "variant": variant, "p4_prob": p4, "phase": "zero-shot",
                **summary
            })
            print(f"  [{exp_id}] Avg Score: {summary['average_score']:.0f}")
            
    return rows

# ======================================================================
# 实验 B: 连续学习 (Continual Learning & Fast Adaptation)
# ======================================================================
def continual_worker(args):
    config, exp_id, variant, target_mode, feature_mode, is_pretrained, model_path, worker_id = args
    board.lookup.init()
    rng = random.Random(config.seed + int(exp_id[-1], 36) * 1000 + (100 if is_pretrained else 0))
    
    # 构建兼容漂移环境的 Agent
    agent = ContinualQLearningAgent(target_mode=target_mode, feature_mode=feature_mode, p4_prob=CONTINUAL_DRIFT_P4)
    
    # 注入预训练权重 (触发 Negative Transfer 或 Fast Adaptation)
    if is_pretrained and os.path.exists(model_path):
        base_agent = load_agent(model_path)
        if base_agent.q_heads: agent.q_heads = base_agent.q_heads
        if base_agent.v_head: agent.v_head = base_agent.v_head
        if base_agent.m_head: agent.m_head = base_agent.m_head
    
    agent.clear_traces()
    
    metrics = {"episodes": [], "td_rms": [], "train_scores": []}
    td_errors, window_scores = [], []
    td_lambda = getattr(config, 'q_td_lambda', 0.5)
    
    pbar_desc = f"{exp_id} {'FT' if is_pretrained else 'Scratch'}".ljust(15)
    
    for episode in tqdm(range(1, CONTINUAL_EPISODES + 1), desc=pbar_desc, position=worker_id, leave=True):
        agent.clear_traces()
        b = board()
        popup_with_rng(b, rng, p4=CONTINUAL_DRIFT_P4)
        popup_with_rng(b, rng, p4=CONTINUAL_DRIFT_P4)
        score = 0
        
        while True:
            actions = legal_actions(b)
            if not actions: break
                
            # 开启微小探索进行流形自适应
            action, _ = agent.choose_action(b, epsilon=0.01, rng=rng)
            step = agent.update_step(b.raw, action, rng, td_lambda)
            
            if step.terminal: break
            score += step.reward
            td_errors.append(abs(step.td_error))
            b = board(step.next_raw)
            
        window_scores.append(score)

        if len(td_errors) >= config.q_td_window:
            rms = math.sqrt(safe_mean(err * err for err in td_errors))
            metrics["episodes"].append(episode)
            metrics["td_rms"].append(rms)
            metrics["train_scores"].append(safe_mean(window_scores))
            td_errors.clear()
            window_scores.clear()
            
    final_score = safe_mean(metrics["train_scores"][-10:]) if metrics["train_scores"] else 0
    return exp_id, is_pretrained, metrics, final_score

def run_experiment_b_continual(config, latest_model_dir: str):
    print(f"\n========== Experiment B: Continual Learning @ P(4)=0.5 ==========")
    all_metrics = {}
    args_list = []
    
    worker_idx = 0
    for exp_id, variant, target_mode, feature_mode in EXPERIMENTS:
        model_path = os.path.join(latest_model_dir, f"qlearning_{exp_id.lower().replace('-', '')}.pkl")
        # 从零开始 (Tabula Rasa)
        args_list.append((config, exp_id, variant, target_mode, feature_mode, False, model_path, worker_idx))
        worker_idx += 1
        # 微调预训练 (Fine-Tuning)
        args_list.append((config, exp_id, variant, target_mode, feature_mode, True, model_path, worker_idx))
        worker_idx += 1

    with concurrent.futures.ProcessPoolExecutor(max_workers=config.workers) as executor:
        futures = [executor.submit(continual_worker, args) for args in args_list]
        for future in concurrent.futures.as_completed(futures):
            try:
                exp_id, is_pretrained, metrics, final_score = future.result()
                key = f"{exp_id}_{'FT' if is_pretrained else 'Scratch'}"
                all_metrics[key] = metrics
                print(f"✅ {key} completed. Final Window Score: {final_score:.1f}")
            except Exception as e:
                print(f"❌ Worker failed: {e}")
                
    return all_metrics

# ======================================================================
# 绘图逻辑 (严格复用你第一阶段的平滑风格与第二阶段的双层网格)
# ======================================================================
def smooth_curve(points, window=10):
    if len(points) < window: return points
    w = np.ones(window) / window
    return np.convolve(points, w, mode='valid')

def generate_drift_plots(zero_shot_rows, continual_metrics, picture_dir):
    os.makedirs(picture_dir, exist_ok=True)
    colors = {"3-A": "red", "3-B": "orange", "3-C": "gray", "3-D": "blue", "3-E": "green"}
    
    # --- 图 1: Zero-Shot 得分降级 (折线图) ---
    plt.figure(figsize=(10, 6))
    for exp_id, _, _, _ in EXPERIMENTS:
        m_rows = sorted([r for r in zero_shot_rows if r["experiment"] == exp_id], key=lambda x: x["p4_prob"])
        if not m_rows: continue
        x_vals = [r["p4_prob"] for r in m_rows]
        y_vals = [r["average_score"] for r in m_rows]
        plt.plot(x_vals, y_vals, label=exp_id, color=colors[exp_id], marker='o', linewidth=2.5)
        
    plt.title("Zero-Shot Score Degradation under Prior Mismatch", fontsize=14, fontweight='bold')
    plt.xlabel("P(4) Spawn Probability", fontsize=12)
    plt.ylabel("Average Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "zeroshot_score_drift.png"), dpi=300)
    plt.close()

    # --- 图 2: Zero-Shot 最大方块 (堆叠柱状图) ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Max Tile Achievement Rates across Drift Environments (Zero-Shot)", fontsize=18, fontweight='bold')
    tile_levels = ["<1024", "1024", "2048", "4096", "8192", "16384", "32768+"]
    tile_colors = ["#e0e0e0", "#fbd390", "#f67c5f", "#f65e3b", "#edcf72", "#edc850", "#edc53f"]
    axes_flat = axes.flatten()
    
    for idx, (exp_id, _, _, _) in enumerate(EXPERIMENTS):
        if idx >= len(axes_flat): break
        ax = axes_flat[idx]
        m_rows = sorted([r for r in zero_shot_rows if r["experiment"] == exp_id], key=lambda x: x["p4_prob"])
        if not m_rows: continue
        x_labels = [f"{r['p4_prob']:.1f}" for r in m_rows]
        x_pos = np.arange(len(x_labels))
        bottoms = np.zeros(len(x_labels))
        
        for t_idx, tile in enumerate(tile_levels):
            heights = []
            for r in m_rows:
                r_1024, r_2048 = r.get("rate_1024", 0), r.get("rate_2048", 0)
                r_4096, r_8192 = r.get("rate_4096", 0), r.get("rate_8192", 0)
                r_16384, r_32768 = r.get("rate_16384", 0), r.get("rate_32768", 0)
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
            
        ax.set_title(exp_id, fontsize=12)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(x_labels, rotation=45)
        ax.set_ylabel("Percentage" if idx % 3 == 0 else "", fontsize=11)
        if idx == 0: ax.legend(title="Max Tile", loc="lower left", bbox_to_anchor=(1.0, 0.0), fontsize=10)

    plt.tight_layout(rect=[0, 0, 0.96, 0.96])
    plt.savefig(os.path.join(picture_dir, "zeroshot_tile_stacked.png"), dpi=300)
    plt.close()

    # --- 图 3: 连续学习 (微调 vs 从零训练) 得分对比 ---
    plt.figure(figsize=(12, 7))
    for exp_id, _, _, _ in EXPERIMENTS:
        ft_key, sc_key = f"{exp_id}_FT", f"{exp_id}_Scratch"
        if ft_key in continual_metrics and sc_key in continual_metrics:
            ft_scores = continual_metrics[ft_key]["train_scores"]
            sc_scores = continual_metrics[sc_key]["train_scores"]
            episodes = continual_metrics[ft_key]["episodes"]
            
            w = max(1, len(ft_scores)//10)
            sm_ft = smooth_curve(ft_scores, w)
            sm_sc = smooth_curve(sc_scores, w)
            sm_ep = episodes[w-1:]
            
            # 使用虚线表示 Scratch，实线表示 Fine-Tuning
            plt.plot(sm_ep, sm_sc, linestyle="--", color=colors[exp_id], alpha=0.6)
            plt.plot(sm_ep, sm_ft, linestyle="-", color=colors[exp_id], linewidth=2.5, label=f"{exp_id} (Pretrained)")

    plt.title("Continual Learning @ P(4)=0.5: Negative Transfer vs Fast Adaptation", fontsize=14, fontweight='bold')
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "continual_learning_scores.png"), dpi=300)
    plt.close()
    
    # --- 图 4: 连续学习 TD-Error 冲击消散追踪 ---
    plt.figure(figsize=(12, 7))
    for exp_id, _, _, _ in EXPERIMENTS:
        ft_key = f"{exp_id}_FT"
        if ft_key in continual_metrics:
            td_rms = continual_metrics[ft_key]["td_rms"]
            episodes = continual_metrics[ft_key]["episodes"]
            w = max(1, len(td_rms)//10)
            sm_td = smooth_curve(td_rms, w)
            sm_ep = episodes[w-1:]
            
            plt.plot(sm_ep, sm_td, color=colors[exp_id], linewidth=2, label=f"{exp_id} FT Error")

    plt.title("TD-Error Shock Dissipation upon Environment Drift", fontsize=14, fontweight='bold')
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("TD-Error RMS (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "continual_learning_tderror.png"), dpi=300)
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Q-Learning Env Drift (Zero-Shot & Continual)")
    add_common_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    
    ts = timestamp()
    config = config.__class__(**{**config.__dict__, "output_dir": os.path.join("models", "phase2", "results", f"qlearning_drift_{ts}")})
    picture_dir = os.path.join("models", "phase2", "picture", "qlearning", ts)
    
    latest_model_dir = get_latest_model_dir("models/phrase_1/qlearning_runs")
    print(f"📦 Auto-loaded pre-trained weights from: {latest_model_dir}")

    # 执行实验 A 
    zero_shot_rows = run_experiment_a_zero_shot(config, latest_model_dir)
    
    # 执行实验 B
    continual_metrics = run_experiment_b_continual(config, latest_model_dir)
    
    # 保存结果与制图
    print(f"\n🎨 Generating Academic Plots in {picture_dir} ...")
    generate_drift_plots(zero_shot_rows, continual_metrics, picture_dir)
    paths = write_result_bundle(config.output_dir, "qlearning_drift_results", config, zero_shot_rows, {})
    
    print(f"✅ All Q-Learning Drift Tests Completed!")
    print(f"Data saved to: {paths['md']}")
    print(f"Pictures saved to: {picture_dir}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()