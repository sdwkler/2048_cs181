from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
import concurrent.futures
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.environments.base_env import board
from src.common import (
    ACTION_NAMES,
    add_common_args,
    apply_action,
    config_from_args,
    legal_actions,
    max_tile_value,
    play_policy,
    popup_with_rng,
    progress,
    safe_mean,
    summarize_games,
    timestamp,
    write_result_bundle,
)

SHAPES = (
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)

EXPERIMENTS = [
    ("3-A", "Q(s,a)+StateNTuple", "q", "state"),
    ("3-B", "Q(s,a)+AfterstateNTuple", "q", "afterstate"),
    ("3-C", "V(s')+StateNTuple", "v", "state"),
    ("3-D", "V(s')+AfterstateNTuple", "v", "afterstate"),
    ("3-E", "MV(s')+AfterstateNTuple", "mv", "afterstate"),
]


@dataclass
class StepResult:
    reward: int
    td_error: float
    next_raw: int
    terminal: bool


class SparseNTupleValue:
    def __init__(self, alpha: float = 0.05):
        self.alpha = alpha
        self.weights: dict[tuple[int, int], float] = {}
        self.traces: dict[tuple[int, int], float] = {}
        self.isom = self._build_isom()

    def clear_traces(self) -> None:
        self.traces.clear()

    def _build_isom(self) -> list[list[tuple[int, ...]]]:
        all_isom = []
        for shape in SHAPES:
            shape_isom = []
            for i in range(8):
                idx = board(0xFEDCBA9876543210)
                if i >= 4:
                    idx.mirror()
                idx.rotate(i)
                shape_isom.append(tuple(idx.at(t) for t in shape))
            all_isom.append(shape_isom)
        return all_isom

    def _indices(self, b: board):
        for feature_id, shape_isom in enumerate(self.isom):
            for shape in shape_isom:
                yield (feature_id, self._pattern_index(shape, b))

    @staticmethod
    def _pattern_index(shape: tuple[int, ...], b: board) -> int:
        index = 0
        for i, pos in enumerate(shape):
            index |= b.at(pos) << (4 * i)
        return index

    def estimate(self, b: board) -> float:
        return sum(self.weights.get(index, 0.0) for index in self._indices(b))

    def update(self, b: board, td_error: float, gamma: float = 1.0, td_lambda: float = 0.5) -> None:
        indices = list(self._indices(b))
        if not indices:
            return

        decay = gamma * td_lambda
        for k in list(self.traces.keys()):
            self.traces[k] *= decay
            if self.traces[k] < 1e-4:
                del self.traces[k]

        for index in indices:
            self.traces[index] = 1.0

        delta = self.alpha * td_error / len(indices)
        for index, trace_val in self.traces.items():
            self.weights[index] = self.weights.get(index, 0.0) + delta * trace_val

    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "weights": self.weights}

    def load_state_dict(self, payload: dict) -> None:
        self.alpha = payload["alpha"]
        self.weights = payload["weights"]


class QLearningAgent:
    def __init__(
        self,
        target_mode: str,
        feature_mode: str,
        alpha: float = 0.05,
        gamma: float = 1.0,
    ):
        self.target_mode = target_mode
        self.feature_mode = feature_mode
        self.gamma = gamma
        self.penalty_lambda = 0.001 

        if target_mode == "q":
            self.q_heads = [SparseNTupleValue(alpha=alpha) for _ in range(4)]
            self.v_head = None
            self.m_head = None
        elif target_mode == "v":
            self.q_heads = None
            self.v_head = SparseNTupleValue(alpha=alpha)
            self.m_head = None
        elif target_mode == "mv":
            self.q_heads = None
            self.v_head = SparseNTupleValue(alpha=alpha)
            self.m_head = SparseNTupleValue(alpha=alpha * 0.1)
    def set_alpha(self, new_alpha: float) -> None:
        """【新增】：动态更新所有评估头（Heads）的学习率"""
        if self.target_mode == "q":
            for head in self.q_heads:
                head.alpha = new_alpha
        elif self.target_mode == "v":
            self.v_head.alpha = new_alpha
        elif self.target_mode == "mv":
            self.v_head.alpha = new_alpha
            # 保持方差头（M-head）的学习率比例，原代码中是主学习率的 10%
            self.m_head.alpha = new_alpha * 0.1
    def clear_traces(self) -> None:
        if self.target_mode == "q":
            for head in self.q_heads:
                head.clear_traces()
        elif self.target_mode == "v":
            self.v_head.clear_traces()
        elif self.target_mode == "mv":
            self.v_head.clear_traces()
            self.m_head.clear_traces()

    def feature_board(self, state_raw: int, action: int | None = None, after_raw: int | None = None) -> board:
        if self.feature_mode == "afterstate" and after_raw is not None:
            return board(after_raw)
        return board(state_raw)

    def action_value(self, state_raw: int, action: int) -> float:
        after_raw, reward = apply_action(state_raw, action)
        if reward == -1:
            return -float("inf")
        feat = self.feature_board(state_raw, action, after_raw)

        if self.target_mode == "q":
            return self.q_heads[action].estimate(feat)
        elif self.target_mode == "v":
            return reward + self.v_head.estimate(feat)
        elif self.target_mode == "mv":
            v_val = self.v_head.estimate(feat)
            m_val = self.m_head.estimate(feat)
            
            scale = 0.01
            v_scaled = v_val * scale
            variance_scaled = max(0.0, m_val - (v_scaled * v_scaled))
            std_dev_unscaled = math.sqrt(variance_scaled) / scale
            
            return reward + v_val - self.penalty_lambda * std_dev_unscaled

    def best_action(self, b: board) -> int:
        actions = legal_actions(b)
        if not actions:
            return 0
        return max(actions, key=lambda action: self.action_value(b.raw, action))

    def best_action_value(self, state_raw: int) -> float:
        actions = legal_actions(board(state_raw))
        if not actions:
            return 0.0
        return max(self.action_value(state_raw, action) for action in actions)

    def choose_action(self, b: board, epsilon: float, rng: random.Random) -> tuple[int, bool]:
        actions = legal_actions(b)
        if not actions:
            return 0, True
            
        # 恢复 epsilon 掷骰子逻辑
        if rng.random() < epsilon:
            return rng.choice(actions), False
            
        best_a = max(actions, key=lambda action: self.action_value(b.raw, action))
        return best_a, True

    def update_step(self, state_raw: int, action: int, rng: random.Random, td_lambda: float = 0.5) -> StepResult:
        after_raw, reward = apply_action(state_raw, action)
        if reward == -1:
            return StepResult(0, 0.0, state_raw, True)

        next_b = board(after_raw)
        popup_with_rng(next_b, rng)
        next_raw = next_b.raw

        feat = self.feature_board(state_raw, action, after_raw)

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

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "target_mode": self.target_mode,
            "feature_mode": self.feature_mode,
            "gamma": self.gamma,
            "q_heads": [head.state_dict() for head in self.q_heads] if self.q_heads else None,
            "v_head": self.v_head.state_dict() if self.v_head else None,
            "m_head": self.m_head.state_dict() if self.m_head else None,
        }
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


def random_decision_board(seed: int, max_steps: int = 200) -> board:
    rng = random.Random(seed)
    b = board()
    popup_with_rng(b, rng)
    popup_with_rng(b, rng)
    for _ in range(rng.randint(10, max_steps)):
        actions = legal_actions(b)
        if not actions:
            break
        action = rng.choice(actions)
        reward = b.move(action)
        if reward == -1:
            break
        popup_with_rng(b, rng)
    return b

def policy_decision_board(agent: QLearningAgent, seed: int, max_warmup_steps: int = 200) -> board:
    rng = random.Random(seed)
    b = board()
    popup_with_rng(b, rng)
    popup_with_rng(b, rng)
    warmup_steps = rng.randint(0, max_warmup_steps)
    for _ in range(warmup_steps):
        actions = legal_actions(b)
        if not actions:
            break
        action = agent.best_action(b)
        reward = b.move(action)
        if reward == -1:
            break
        popup_with_rng(b, rng)
    return b

def rollout_return(agent: QLearningAgent, start_raw: int, seed: int, gamma: float = 1.0) -> float:
    rng = random.Random(seed)
    b = board(start_raw)
    total, discount = 0.0, 1.0
    while True:
        action = agent.best_action(b)
        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1:
            break
        total += discount * reward
        discount *= gamma
        popup_with_rng(next_b, rng)
        b = next_b
    return total

def collect_bias(agent: QLearningAgent, count: int, seed: int) -> dict:
    biases = []
    norm_biases = [] 
    for i in range(count):
        b = policy_decision_board(agent, seed + i)
        if not legal_actions(b):
            b = random_decision_board(seed + 100_000 + i)
        predicted = agent.best_action_value(b.raw)
        realized = rollout_return(agent, b.raw, seed + 10_000 + i, gamma=agent.gamma)
        
        bias = predicted - realized
        biases.append(bias)
        norm_biases.append(bias / (realized + 1.0)) 
        
    return {
        "samples": count,
        "average_bias": safe_mean(biases),
        "average_norm_bias": safe_mean(norm_biases), 
    }


def train_one(config, exp_id: str, variant: str, target_mode: str, feature_mode: str, model_dir: str, worker_id: int = 0):
    board.lookup.init()
    rng = random.Random(config.seed + int(exp_id[-1], 36) * 1000)
    agent = QLearningAgent(target_mode=target_mode, feature_mode=feature_mode)
    td_errors = []
    
    metrics = {"episodes": [], "td_rms": [], "normalized_td_rms": [], "train_scores": [], "bias": [], "norm_bias": []}
    window_scores = []
    td_lambda = getattr(config, 'q_td_lambda', 0.5)
    bias_interval = min(100, config.q_episodes // 100) 
    
    # 【新增：双重退火参数设置】
    # ==========================================
    # 1. 探索率 (Epsilon) 参数
    epsilon_start = 0.10
    epsilon_end = 0.0001
    epsilon_decay_cutoff = 0.80  # 在训练进行到 80% 时降到最低点
    
    # 2. 学习率 (Alpha) 参数
    alpha_start = 0.05
    alpha_end = 0.002            # 最终学习率，退火到底部精调
    # ==========================================

    start_time = time.perf_counter()
    pbar_desc = f"{exp_id} {variant}"[:22].ljust(22)
    
    for episode in tqdm(range(1, config.q_episodes + 1), desc=pbar_desc, position=worker_id, leave=True):
        agent.clear_traces()
        b = board()
        popup_with_rng(b, rng)
        popup_with_rng(b, rng)
        steps = 0
        score = 0
        # ==========================================
        # 【新增：计算当前 Episode 的退火状态】
        # ==========================================
        # 1. 计算 Epsilon (带截断)
        eps_progress = min(1.0, episode / (config.q_episodes * epsilon_decay_cutoff))
        current_epsilon = epsilon_start - (epsilon_start - epsilon_end) * eps_progress
        
        # 2. 计算 Alpha (全局线性退火)
        alpha_progress = episode / config.q_episodes
        current_alpha = alpha_start - (alpha_start - alpha_end) * alpha_progress
        
        # 3. 将新的学习率注入到 Agent 中
        agent.set_alpha(current_alpha)
        
        while config.max_game_steps is None or steps < config.max_game_steps:
            actions = legal_actions(b)
            if not actions:
                break
            action, _ = agent.choose_action(b, current_epsilon, rng)
            step = agent.update_step(b.raw, action, rng, td_lambda)
            
            if step.terminal:
                break
            
            score += step.reward
            td_errors.append(abs(step.td_error))
            b = board(step.next_raw)
            steps += 1
            
        window_scores.append(score)

        if len(td_errors) >= config.q_td_window:
            rms = math.sqrt(safe_mean(err * err for err in td_errors))
            current_score_avg = safe_mean(window_scores)
            
            normalized_rms = rms / (current_score_avg + 1.0)
            
            metrics["episodes"].append(episode)
            metrics["td_rms"].append(rms)
            metrics["normalized_td_rms"].append(normalized_rms)
            metrics["train_scores"].append(current_score_avg)
            
            td_errors.clear()
            window_scores.clear()
            
        if episode % bias_interval == 0:
            # 评估 Bias 时，为了获取绝对准确的预测能力，强行关闭 epsilon 干扰 (评估环境默认只按最佳动作走)
            bias_res = collect_bias(agent, 10, config.seed + 40_000 + episode)
            metrics["bias"].append((episode, bias_res["average_bias"]))
            metrics["norm_bias"].append((episode, bias_res["average_norm_bias"])) 

    model_path = os.path.join(model_dir, f"qlearning_{exp_id.lower().replace('-', '')}.pkl")
    agent.save(model_path)

    def choose_action(b: board) -> int:
        return agent.best_action(b)

    game_records = [
        play_policy(config.seed + 50_000 + i, choose_action, max_steps=config.max_game_steps)
        for i in range(config.q_eval_games)
    ]
    game_summary = summarize_games(game_records)
    elapsed = time.perf_counter() - start_time
    
    final_bias = collect_bias(agent, config.q_bias_samples, config.seed + 80_000)

    row = {
        "experiment": exp_id,
        "variant": variant,
        "target_mode": target_mode,
        "feature_mode": feature_mode,
        **game_summary,
        "td_error_rms_final": metrics["td_rms"][-1] if metrics["td_rms"] else 0.0,
        "td_error_rms_mean": safe_mean(metrics["td_rms"]),
        "normalized_td_rms_final": metrics["normalized_td_rms"][-1] if metrics["normalized_td_rms"] else 0.0,
        "average_bias": final_bias["average_bias"],
        "average_norm_bias": final_bias["average_norm_bias"],
        "model_path": model_path,
        "train_seconds": elapsed,
    }
    
    return row, metrics


def smooth_curve(points, window=10):
    if len(points) < window:
        return points
    w = np.ones(window) / window
    smoothed = np.convolve(points, w, mode='valid')
    return smoothed


def generate_plots(all_metrics, picture_dir):
    os.makedirs(picture_dir, exist_ok=True)
    colors = {"3-A": "red", "3-B": "orange", "3-C": "gray", "3-D": "blue", "3-E": "green"}
    
    sample_scores_len = len(next(iter(all_metrics.values()))["train_scores"])
    sample_bias_len = len(next(iter(all_metrics.values()))["bias"])
    
    SCORE_WINDOW = min(10, sample_scores_len // 10)
    BIAS_WINDOW = min(10, sample_bias_len // 10)
    
    # 图 1: 学习曲线
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"]:
            episodes = metrics["episodes"]
            scores = metrics["train_scores"]
            
            plt.plot(episodes, scores, color=colors[exp_id], alpha=0.15, linewidth=1)
            
            if len(scores) >= SCORE_WINDOW:
                smoothed = smooth_curve(scores, SCORE_WINDOW)
                sm_episodes = episodes[SCORE_WINDOW - 1:] 
                plt.plot(sm_episodes, smoothed, label=f"{exp_id}", color=colors[exp_id], linewidth=2.5)
            else:
                plt.plot(episodes, scores, label=exp_id, color=colors[exp_id], linewidth=2.5)

    plt.title(f"Learning Curve (Average Training Score)", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Score", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "learning_curve_smoothed.png"), dpi=300)
    plt.close()

    # 图 2: 绝对 TD-Error 波动
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"]:
            episodes = metrics["episodes"]
            td_rms = metrics["td_rms"]
            
            plt.plot(episodes, td_rms, color=colors[exp_id], alpha=0.15, linewidth=1)
            
            if len(td_rms) >= SCORE_WINDOW:
                smoothed = smooth_curve(td_rms, SCORE_WINDOW)
                sm_episodes = episodes[SCORE_WINDOW - 1:]
                plt.plot(sm_episodes, smoothed, label=exp_id, color=colors[exp_id], linewidth=2)
            else:
                plt.plot(episodes, td_rms, label=exp_id, color=colors[exp_id], linewidth=2)

    plt.title("Convergence Stability (Absolute TD-Error RMS)", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("TD-Error (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "td_error_volatility_smoothed.png"), dpi=300)
    plt.close()
    
    # 图 4: 相对 TD-Error (Normalized TD-Error)
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"] and "normalized_td_rms" in metrics:
            episodes = metrics["episodes"]
            norm_td_rms = metrics["normalized_td_rms"]
            
            plt.plot(episodes, norm_td_rms, color=colors[exp_id], alpha=0.15, linewidth=1)
            
            if len(norm_td_rms) >= SCORE_WINDOW:
                smoothed = smooth_curve(norm_td_rms, SCORE_WINDOW)
                sm_episodes = episodes[SCORE_WINDOW - 1:]
                plt.plot(sm_episodes, smoothed, label=exp_id, color=colors[exp_id], linewidth=2)
            else:
                plt.plot(episodes, norm_td_rms, label=exp_id, color=colors[exp_id], linewidth=2)

    plt.title("Normalized Convergence (TD-Error RMS / Avg Score)", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Relative TD-Error (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "normalized_td_error_smoothed.png"), dpi=300)
    plt.close()
    
    # 图 3: 相对过估计偏差 (Normalized Bias)
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if "norm_bias" in metrics and metrics["norm_bias"]:
            episodes, norm_biases = zip(*metrics["norm_bias"])
            
            plt.plot(episodes, norm_biases, color=colors[exp_id], alpha=0.15, linewidth=1, marker='.', markersize=4)
            
            if len(norm_biases) >= BIAS_WINDOW:
                smoothed = smooth_curve(norm_biases, BIAS_WINDOW)
                sm_episodes = episodes[BIAS_WINDOW - 1:]
                plt.plot(sm_episodes, smoothed, label=exp_id, color=colors[exp_id], linewidth=2.5)
            else:
                plt.plot(episodes, norm_biases, label=exp_id, color=colors[exp_id], linewidth=2.5, marker='o')
    
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Bias (Perfect Prediction)')
    plt.title("Empirical Overestimation Bias over Time (Normalized)", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Relative Prediction Bias (Bias / Realized Return)", fontsize=12) 
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "normalized_overestimation_bias_smoothed.png"), dpi=300)
    plt.close()


def run_experiment_parallel(config):
    timestamp_str = timestamp()
    model_dir = os.path.join("models","phrase_1","qlearning_runs", timestamp_str)
    picture_dir = os.path.join("models", "phrase_1", "picture", timestamp_str)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)

    rows = []
    all_metrics = {}
    
    num_experiments = len(EXPERIMENTS)
    print(f"===========================================================")
    print(f"Launching {num_experiments} experiments concurrently...")
    print(f"Your CPU usage will spike. Please wait...")
    print(f"===========================================================\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=num_experiments) as executor:
        futures = []
        for i, (exp_id, variant, target_mode, feature_mode) in enumerate(EXPERIMENTS):
            futures.append(
                executor.submit(train_one, config, exp_id, variant, target_mode, feature_mode, model_dir, i)
            )

        for future in concurrent.futures.as_completed(futures):
            try:
                row, metrics = future.result()
                exp_id = row["experiment"]
                rows.append(row)
                all_metrics[exp_id] = metrics
                tqdm.write(f"\n✅ {exp_id} Finished! Score: {row['average_score']:.0f}, Norm_TD: {row['normalized_td_rms_final']:.4f}, Norm_Bias: {row['average_norm_bias']:.4f}")
            except Exception as exc:
                tqdm.write(f"\n❌ An experiment generated an exception: {exc}")

    print(f"\nAll experiments completed! Generating combined plots...")
    generate_plots(all_metrics, picture_dir)
    
    rows = sorted(rows, key=lambda x: x["experiment"])
    paths = write_result_bundle(config.output_dir, "qlearning_parallel", config, rows, {})
    print(f"Q-learning results saved: {paths['md']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run phase-1 Q-learning ablation experiments.")
    add_common_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    
    run_experiment_parallel(config)


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    main()