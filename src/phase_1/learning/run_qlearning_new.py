from __future__ import annotations

import argparse
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

# 引入绘图与数值计算库
import numpy as np
import matplotlib.pyplot as plt

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.environments.base_env import board
from src.phase_1.common import (
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

# 包含了 3-E：均值-方差惩罚版 Afterstate
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
    def __init__(self, alpha: float = 0.0025):
        self.alpha = alpha
        self.weights: dict[tuple[int, int, int], float] = {}
        self.traces: dict[tuple[int, int, int], float] = {}
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
                yield (feature_id, 0, self._pattern_index(shape, b))
                yield (feature_id, 1, self._diff_index(shape, b))

    @staticmethod
    def _pattern_index(shape: tuple[int, ...], b: board) -> int:
        index = 0
        for i, pos in enumerate(shape):
            index |= b.at(pos) << (4 * i)
        return index

    @staticmethod
    def _diff_index(shape: tuple[int, ...], b: board) -> int:
        index = 0
        for i in range(1, len(shape)):
            index |= (b.at(shape[i]) - b.at(shape[i - 1]) + 15) << (5 * (i - 1))
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
            self.traces[index] = self.traces.get(index, 0.0) + 1.0

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
        alpha: float = 0.0025,
        gamma: float = 1.0,
    ):
        self.target_mode = target_mode
        self.feature_mode = feature_mode
        self.gamma = gamma
        self.penalty_lambda = 0.001  # 均值-方差惩罚系数

        if target_mode == "q":
            self.q_heads = [SparseNTupleValue(alpha=alpha) for _ in range(4)]
            self.v_head = None
            self.m_head = None
        elif target_mode == "v":
            self.q_heads = None
            self.v_head = SparseNTupleValue(alpha=alpha)
            self.m_head = None
        elif target_mode == "mv":
            # MV 模式：同时学习均值 V 和 平方均值 M
            self.q_heads = None
            self.v_head = SparseNTupleValue(alpha=alpha)
            # M值的量级较大，使用较小的学习率防止发散
            self.m_head = SparseNTupleValue(alpha=alpha * 0.1)

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
            # 计算方差 Var = M - V^2 (防止精度误差导致负数)
            variance = max(0.0, m_val - v_val * v_val)
            # 引入均值-方差惩罚
            return reward + v_val - self.penalty_lambda * math.sqrt(variance)

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

    def choose_action(self, b: board, epsilon: float, rng: random.Random) -> int:
        actions = legal_actions(b)
        if not actions:
            return 0
        if rng.random() < epsilon:
            return rng.choice(actions)
        return max(actions, key=lambda action: self.action_value(b.raw, action))

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
            # 1. 评估下一步的最优行动期望
            next_v = self.best_action_value(next_raw)
            current_v = self.v_head.estimate(feat)
            
            # V 头更新目标 (同 V 模式)
            target_v = self.gamma * next_v
            td_error_v = target_v - current_v
            self.v_head.update(feat, td_error_v, self.gamma, td_lambda)

            # 2. 评估下一步的平方期望 (为了稳定，Reward 缩放)
            scale = 0.01
            r_scaled = reward * scale
            next_m = self.m_head.estimate(board(next_raw))
            current_m = self.m_head.estimate(feat)
            
            # M 头更新目标: E[R^2 + 2 * R * gamma * V(s') + gamma^2 * M(s')]
            target_m = r_scaled**2 + 2 * self.gamma * r_scaled * (next_v * scale) + (self.gamma**2) * next_m
            td_error_m = target_m - current_m
            # M 头同样使用资格迹更新
            self.m_head.update(feat, td_error_m, self.gamma, td_lambda)
            
            td_error = td_error_v # 总体监控依然以 V 的误差为准

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


def epsilon_for_episode(episode: int, total_episodes: int) -> float:
    if total_episodes <= 1:
        return 0.01
    frac = (episode - 1) / (total_episodes - 1)
    return 1.0 + frac * (0.01 - 1.0)


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
    for i in range(count):
        b = policy_decision_board(agent, seed + i)
        if not legal_actions(b):
            b = random_decision_board(seed + 100_000 + i)
        predicted = agent.best_action_value(b.raw)
        realized = rollout_return(agent, b.raw, seed + 10_000 + i, gamma=agent.gamma)
        biases.append(predicted - realized)
    return {
        "samples": count,
        "average_bias": safe_mean(biases),
    }


def train_one(config, exp_id: str, variant: str, target_mode: str, feature_mode: str, model_dir: str):
    board.lookup.init()
    rng = random.Random(config.seed + int(exp_id[-1], 36) * 1000)
    agent = QLearningAgent(target_mode=target_mode, feature_mode=feature_mode)
    td_errors = []
    
    # 周期性监控数据存储
    metrics = {"episodes": [], "td_rms": [], "train_scores": [], "bias": []}
    window_scores = []
    
    td_lambda = getattr(config, 'q_td_lambda', 0.5)
    bias_interval = config.q_episodes // 5  # 每 20% 局数测一次 Bias

    start_time = time.perf_counter()

    for episode in progress(range(1, config.q_episodes + 1), desc=variant[:24], leave=False):
        agent.clear_traces()
        b = board()
        popup_with_rng(b, rng)
        popup_with_rng(b, rng)
        epsilon = epsilon_for_episode(episode, config.q_episodes)
        steps = 0
        score = 0
        
        while config.max_game_steps is None or steps < config.max_game_steps:
            actions = legal_actions(b)
            if not actions:
                break
            action = agent.choose_action(b, epsilon, rng)
            step = agent.update_step(b.raw, action, rng, td_lambda)
            if step.terminal:
                break
            
            score += step.reward
            td_errors.append(abs(step.td_error))
            b = board(step.next_raw)
            steps += 1
            
        window_scores.append(score)

        # 记录滑动窗口的 TD-Error 和 平均分
        if len(td_errors) >= config.q_td_window:
            rms = math.sqrt(safe_mean(err * err for err in td_errors))
            metrics["episodes"].append(episode)
            metrics["td_rms"].append(rms)
            metrics["train_scores"].append(safe_mean(window_scores))
            td_errors.clear()
            window_scores.clear()
            
        # 周期性收集 Bias 供画图使用
        if episode % bias_interval == 0:
            bias_res = collect_bias(agent, 20, config.seed + 40_000 + episode)
            metrics["bias"].append((episode, bias_res["average_bias"]))

    # 训练后保存模型
    model_path = os.path.join(model_dir, f"qlearning_{exp_id.lower().replace('-', '')}.pkl")
    agent.save(model_path)

    # 最终评估 10 局
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
        "average_bias": final_bias["average_bias"],
        "model_path": model_path,
        "train_seconds": elapsed,
    }
    
    return row, metrics


def smooth_curve(points, window=50):
    """
    使用一维卷积计算滑动平均。
    参数 window 决定了平滑的力度，越大越平滑。
    """
    if len(points) < window:
        return points
    w = np.ones(window) / window
    # mode='valid' 确保窗口完全覆盖数据点，避免边缘效应
    smoothed = np.convolve(points, w, mode='valid')
    return smoothed


def generate_plots(all_metrics, picture_dir):
    """根据收集的指标数据绘制顶级学术图表（带有滑动平均平滑效果）"""
    os.makedirs(picture_dir, exist_ok=True)
    colors = {"3-A": "red", "3-B": "orange", "3-C": "gray", "3-D": "blue", "3-E": "green"}
    
    # 根据跑的局数可以动态调整滑动窗口大小。如果跑 10 万局，可以将这里调成 500 或 1000。
    SMOOTH_WINDOW = 200
    
    # ---------------------------------------------------------
    # 图 1: 学习曲线 (Training Scores)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"]:
            episodes = metrics["episodes"]
            scores = metrics["train_scores"]
            
            # 画半透明的原始数据背景作为噪音底色
            plt.plot(episodes, scores, color=colors[exp_id], alpha=0.15, linewidth=1)
            
            # 画平滑后的主线
            if len(scores) >= SMOOTH_WINDOW:
                smoothed_scores = smooth_curve(scores, SMOOTH_WINDOW)
                # 对齐 X 轴：取对应的 episode 截断区间
                smooth_episodes = episodes[SMOOTH_WINDOW - 1:] 
                plt.plot(smooth_episodes, smoothed_scores, label=f"{exp_id} (MA={SMOOTH_WINDOW})", color=colors[exp_id], linewidth=2.5)
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

    # ---------------------------------------------------------
    # 图 2: TD-Error 波动 (对数坐标系)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"]:
            episodes = metrics["episodes"]
            td_rms = metrics["td_rms"]
            
            # 画半透明原始背景
            plt.plot(episodes, td_rms, color=colors[exp_id], alpha=0.15, linewidth=1)
            
            # 画平滑主线
            if len(td_rms) >= SMOOTH_WINDOW:
                smoothed_td = smooth_curve(td_rms, SMOOTH_WINDOW)
                smooth_episodes = episodes[SMOOTH_WINDOW - 1:]
                plt.plot(smooth_episodes, smoothed_td, label=exp_id, color=colors[exp_id], linewidth=2)
            else:
                plt.plot(episodes, td_rms, label=exp_id, color=colors[exp_id], linewidth=2)

    plt.title("Convergence Stability (TD-Error RMS)", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("TD-Error (Log Scale)", fontsize=12)
    plt.yscale("log")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "td_error_volatility_smoothed.png"), dpi=300)
    plt.close()
    
    # ---------------------------------------------------------
    # 图 3: 过估计偏差 (Overestimation Bias) 
    # 此项数据点较少，无需平滑，直接绘制
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["bias"]:
            episodes, biases = zip(*metrics["bias"])
            plt.plot(episodes, biases, marker='o', label=exp_id, color=colors[exp_id], linewidth=2.5, markersize=8)
    
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Bias (Ideal)')
    plt.title("Empirical Overestimation Bias over Time", fontsize=14)
    plt.xlabel("Training Episodes", fontsize=12)
    plt.ylabel("Prediction Bias", fontsize=12)
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "overestimation_bias.png"), dpi=300)
    plt.close()


def run_experiment(config):
    rows = []
    all_metrics = {}
    timestamp_str = timestamp()
    model_dir = os.path.join("models", "qlearning_runs", timestamp_str)
    picture_dir = os.path.join("models", "picture", timestamp_str)
    
    for exp_id, variant, target_mode, feature_mode in EXPERIMENTS:
        row, metrics = train_one(config, exp_id, variant, target_mode, feature_mode, model_dir)
        rows.append(row)
        all_metrics[exp_id] = metrics
        print(
            f"{exp_id} {variant}: score={row['average_score']:.1f}, "
            f"td_rms={row['td_error_rms_final']:.2f}, bias={row['average_bias']:.2f}"
        )
        
    # 生成带有平滑效果的高级折线图
    print(f"Generating smoothed performance plots in {picture_dir} ...")
    generate_plots(all_metrics, picture_dir)
    
    # 写入 Markdown 表格
    paths = write_result_bundle(config.output_dir, "qlearning", config, rows, {})
    print(f"Q-learning results saved: {paths['md']}")
    return paths


def main():
    parser = argparse.ArgumentParser(description="Run phase-1 Q-learning ablation experiments.")
    add_common_args(parser)
    args = parser.parse_args()
    config = config_from_args(args)
    run_experiment(config)


if __name__ == "__main__":
    main()