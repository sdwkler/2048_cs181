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

# 标准的 6-tuple 形状
SHAPES = (
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (0, 1, 2, 4, 5, 6),
    (4, 5, 6, 8, 9, 10),
)

# 剔除 3-C，保留 4 组核心对照实验
EXPERIMENTS = [
    ("3-D", "V(s')+Afterstate (Baseline)", "v", "afterstate"),
    ("3-E", "TDA-Full+Afterstate", "tda_full", "afterstate"),
    ("3-F", "TDA-2ply+Afterstate (Paper)", "tda_2ply", "afterstate"),
    ("3-G", "Dual-MV+Afterstate", "mv", "afterstate"),
]

@dataclass
class StepResult:
    reward: int
    td_error: float
    next_raw: int
    terminal: bool
    td_error_m_up: float = 0.0    # 正向方差 TD 误差
    td_error_m_down: float = 0.0  # 负向方差 TD 误差


# ============================================================================
# SparseNTupleValue：复用 DensePattern/DiffPattern 的硬编码索引逻辑，
# 但权重和资格迹改用 dict 稀疏存储，杜绝 768MB 稠密 numpy 的无意义零值搬运。
# estimate / update 的数学语义与旧版完全等价。
# ============================================================================
class SparseNTupleValue:
    """稀疏 N-Tuple 价值函数，同时包含绝对值特征和差分特征。"""

    def __init__(self, alpha: float = 0.05, shapes: tuple = SHAPES):
        self.alpha = alpha
        self.shapes = shapes
        self.num_shapes = len(shapes)

        # -------- 同构映射预计算（与 DensePattern / DenseDiffPattern 完全一致）--------
        self.isoms = []
        for p in shapes:
            iso = []
            for i in range(8):
                idx = board(0xFEDCBA9876543210)
                if i >= 4:
                    idx.mirror()
                idx.rotate(i)
                iso.append(tuple(idx.at(t) for t in p))
            self.isoms.append(iso)

        # 稀疏权重 + 资格迹：key = packed_int
        self.weights: dict[int, float] = {}
        self.traces: dict[int, float] = {}

    # -------- 索引计算（硬编码展开，与旧版完全一致）--------
    @staticmethod
    def _pattern_indices_inner(iso: tuple[int, ...], b_at) -> list[int]:
        """单个同构下的绝对值 6-tuple 索引（返回 1 个 int）。"""
        return [
            b_at(iso[0]) | (b_at(iso[1]) << 4) | (b_at(iso[2]) << 8)
            | (b_at(iso[3]) << 12) | (b_at(iso[4]) << 16) | (b_at(iso[5]) << 20)
        ]

    @staticmethod
    def _diff_indices_inner(iso: tuple[int, ...], b_at) -> list[int]:
        """单个同构下的差分 5-delta 索引（返回 1 个 int）。"""
        return [
            (b_at(iso[1]) - b_at(iso[0]) + 15)
            | ((b_at(iso[2]) - b_at(iso[1]) + 15) << 5)
            | ((b_at(iso[3]) - b_at(iso[2]) + 15) << 10)
            | ((b_at(iso[4]) - b_at(iso[3]) + 15) << 15)
            | ((b_at(iso[5]) - b_at(iso[4]) + 15) << 20)
        ]

    # -------- 复合 key：feature_type(1b) | shape_idx(2b) | raw_index(26b) --------
    @staticmethod
    def _key(feature_type: int, shape_idx: int, raw_index: int) -> int:
        """feature_type: 0=pattern, 1=diff; shape_idx: 0..3."""
        return (feature_type << 28) | (shape_idx << 26) | raw_index

    def clear_traces(self) -> None:
        self.traces.clear()

    # -------- 估值 --------
    def estimate(self, b: board) -> float:
        b_at = b.at
        val = 0.0
        w = self.weights
        for i in range(self.num_shapes):
            for iso in self.isoms[i]:
                idx = self._pattern_indices_inner(iso, b_at)[0]
                val += w.get(self._key(0, i, idx), 0.0)
            for iso in self.isoms[i]:
                idx = self._diff_indices_inner(iso, b_at)[0]
                val += w.get(self._key(1, i, idx), 0.0)
        return val

    # -------- TD(λ) 更新 --------
    def update(self, b: board, td_error: float, gamma: float = 1.0, td_lambda: float = 0.5) -> None:
        td_error = max(-50000.0, min(50000.0, td_error))

        # 迹衰减
        decay = gamma * td_lambda
        tr = self.traces
        for k in list(tr.keys()):
            tr[k] *= decay
            if tr[k] < 1e-4:
                del tr[k]

        # 当前状态激活
        b_at = b.at
        total_active = 0
        for i in range(self.num_shapes):
            for iso in self.isoms[i]:
                k = self._key(0, i, self._pattern_indices_inner(iso, b_at)[0])
                tr[k] = 1.0
                total_active += 1
            for iso in self.isoms[i]:
                k = self._key(1, i, self._diff_indices_inner(iso, b_at)[0])
                tr[k] = 1.0
                total_active += 1

        if total_active == 0:
            return

        delta = self.alpha * td_error / total_active
        w = self.weights
        for k, trace_val in tr.items():
            w[k] = w.get(k, 0.0) + delta * trace_val

    # -------- 序列化（numpy savez 替代 pickle，速度快 100x）--------
    def state_dict(self) -> dict:
        return {"alpha": self.alpha, "weights": dict(self.weights)}

    def load_state_dict(self, payload: dict) -> None:
        self.alpha = payload["alpha"]
        self.weights = payload["weights"]
# ============================================================================

class QLearningAgent:
    def __init__(self, target_mode: str, feature_mode: str, alpha: float = 0.05, gamma: float = 1.0):
        self.target_mode = target_mode
        self.feature_mode = feature_mode
        self.gamma = gamma

        # Dual-MV 参数设计
        self.lambda_up = 0.001    # 正向潜力奖励系数
        self.lambda_down = 0.002  # 负向风险惩罚系数

        self.q_heads = None
        self.v_head = SparseNTupleValue(alpha=alpha)

        if target_mode == "mv":
            self.m_up_head = SparseNTupleValue(alpha=alpha * 0.1)
            self.m_down_head = SparseNTupleValue(alpha=alpha * 0.1)
        else:
            self.m_up_head = None
            self.m_down_head = None

        # 步内缓存：_expected_best_action_value 在同一 update_step 内可能被多次调用
        self._step_eva_cache: dict[int, float] = {}
        self._step_pvv_cache: dict[int, float] = {}
        # board 复用池
        self._reuse_board = board()

    def set_alpha(self, new_alpha: float) -> None:
        self.v_head.alpha = new_alpha
        if self.target_mode == "mv":
            self.m_up_head.alpha = new_alpha * 0.1
            self.m_down_head.alpha = new_alpha * 0.1

    def clear_traces(self) -> None:
        self.v_head.clear_traces()
        if self.target_mode == "mv":
            self.m_up_head.clear_traces()
            self.m_down_head.clear_traces()

    def feature_board(self, state_raw: int, action: int | None = None, after_raw: int | None = None) -> board:
        return board(after_raw) if after_raw is not None else board(state_raw)

    def action_value(self, state_raw: int, action: int) -> float:
        after_raw, reward = apply_action(state_raw, action)
        if reward == -1:
            return -float("inf")
        feat = self.feature_board(state_raw, action, after_raw)

        if self.target_mode in ("v", "tda_full", "tda_2ply"):
            return reward + self.v_head.estimate(feat)
        elif self.target_mode == "mv":
            v_val = self.v_head.estimate(feat)
            m_up_val = self.m_up_head.estimate(feat)
            m_down_val = self.m_down_head.estimate(feat)
            
            scale = 0.01
            std_up = math.sqrt(max(0.0, m_up_val)) / scale
            std_down = math.sqrt(max(0.0, m_down_val)) / scale
            
            return reward + v_val + (self.lambda_up * std_up) - (self.lambda_down * std_down)

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

    def _pure_v_value(self, state_raw: int) -> float:
        """纯净期望值（步内缓存）。"""
        if state_raw in self._step_pvv_cache:
            return self._step_pvv_cache[state_raw]

        best = -float("inf")
        for action in range(4):
            after_raw, reward = apply_action(state_raw, action)
            if reward == -1:
                continue
            feat = self.feature_board(state_raw, action, after_raw)
            best = max(best, reward + self.v_head.estimate(feat))
        val = best if best != -float("inf") else 0.0
        self._step_pvv_cache[state_raw] = val
        return val

    def _expected_best_action_value(self, after_raw: int) -> float:
        """精确期望（步内缓存 + board 复用）。"""
        if after_raw in self._step_eva_cache:
            return self._step_eva_cache[after_raw]

        after = board(after_raw)
        empties = [i for i in range(16) if after.at(i) == 0]
        if not empties:
            val = self.best_action_value(after_raw)
            self._step_eva_cache[after_raw] = val
            return val

        weight = 1.0 / len(empties)
        expected_value = 0.0
        spawned = self._reuse_board  # 复用，避免反复 new board()
        for pos in empties:
            for tile, tile_prob in ((1, 0.9), (2, 0.1)):
                spawned.raw = after_raw
                spawned.set(pos, tile)
                expected_value += weight * tile_prob * self.best_action_value(spawned.raw)

        self._step_eva_cache[after_raw] = expected_value
        return expected_value

    def choose_action(self, b: board, epsilon: float, rng: random.Random) -> tuple[int, bool]:
        actions = legal_actions(b)
        if not actions: return 0, True
        if rng.random() < epsilon: return rng.choice(actions), False
        return max(actions, key=lambda action: self.action_value(b.raw, action)), True

    def update_step(self, state_raw: int, action: int, rng: random.Random, td_lambda: float = 0.5) -> StepResult:
        # 步间清空缓存（权重已更新，缓存值失效）
        self._step_eva_cache.clear()
        self._step_pvv_cache.clear()

        after_raw, reward = apply_action(state_raw, action)
        if reward == -1: return StepResult(0, 0.0, state_raw, True)

        next_b = board(after_raw)
        popup_with_rng(next_b, rng)
        next_raw = next_b.raw

        feat = self.feature_board(state_raw, action, after_raw)
        td_error_m_up = td_error_m_down = 0.0

        if self.target_mode == "v":
            target = self.gamma * self.best_action_value(next_raw)
            current = self.v_head.estimate(feat)
            td_error = target - current
            self.v_head.update(feat, td_error, self.gamma, td_lambda)

        elif self.target_mode == "tda_full":
            expected_next_v = self._expected_best_action_value(after_raw)
            target = self.gamma * expected_next_v
            current = self.v_head.estimate(feat)
            td_error = target - current
            self.v_head.update(feat, td_error, self.gamma, td_lambda)

        elif self.target_mode == "tda_2ply":
            actions_next = legal_actions(board(next_raw))
            if not actions_next:
                target = 0.0
            else:
                best_2ply_val = -float('inf')
                for a_next in actions_next:
                    after_next_raw, r_next = apply_action(next_raw, a_next)
                    if r_next != -1:
                        dv_val = self._expected_best_action_value(after_next_raw)
                        val = r_next + self.gamma * dv_val
                        if val > best_2ply_val: best_2ply_val = val
                target = self.gamma * best_2ply_val
            
            current = self.v_head.estimate(feat)
            td_error = target - current
            
            # TDA-2ply 极易过估计，此处将 td_lambda 强制归零（单步截断更新，阻断迹回响）
            self.v_head.update(feat, td_error, self.gamma, 0.0)

        elif self.target_mode == "mv":
            next_v = self._pure_v_value(next_raw)
            current_v = self.v_head.estimate(feat)
            
            target_v = self.gamma * next_v
            td_error_v = target_v - current_v
            self.v_head.update(feat, td_error_v, self.gamma, td_lambda)

            scale = 0.01
            r_v = reward + self.gamma * next_v
            delta_raw = (r_v - current_v) * scale
            
            delta_up = max(0.0, delta_raw)
            delta_down = max(0.0, -delta_raw)
            
            next_m_up = self.m_up_head.estimate(board(next_raw))
            next_m_down = self.m_down_head.estimate(board(next_raw))
            current_m_up = self.m_up_head.estimate(feat)
            current_m_down = self.m_down_head.estimate(feat)
            
            td_error_m_up = ((delta_up ** 2) + (self.gamma ** 2) * next_m_up) - current_m_up
            td_error_m_down = ((delta_down ** 2) + (self.gamma ** 2) * next_m_down) - current_m_down
            
            self.m_up_head.update(feat, td_error_m_up, self.gamma, td_lambda)
            self.m_down_head.update(feat, td_error_m_down, self.gamma, td_lambda)
            td_error = td_error_v

        return StepResult(reward, td_error, next_raw, False, td_error_m_up, td_error_m_down)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "target_mode": self.target_mode,
            "feature_mode": self.feature_mode,
            "gamma": self.gamma,
            "v_head": self.v_head.state_dict() if self.v_head else None,
            "m_up_head": self.m_up_head.state_dict() if self.m_up_head else None,
            "m_down_head": self.m_down_head.state_dict() if self.m_down_head else None,
        }
        # 稀疏字典 pickle 只需 KB 级，远小于稠密 numpy 的 GB 级
        with open(path, "wb") as f:
            pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)


# 后续工具类与数据收集函数均保持一致
def random_decision_board(seed: int, max_steps: int = 200) -> board:
    rng = random.Random(seed)
    b = board()
    popup_with_rng(b, rng)
    popup_with_rng(b, rng)
    for _ in range(rng.randint(10, max_steps)):
        actions = legal_actions(b)
        if not actions: break
        action = rng.choice(actions)
        reward = b.move(action)
        if reward == -1: break
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
        if not actions: break
        action = agent.best_action(b)
        reward = b.move(action)
        if reward == -1: break
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
        if reward == -1: break
        total += discount * reward
        discount *= gamma
        popup_with_rng(next_b, rng)
        b = next_b
    return total

def collect_bias(agent: QLearningAgent, count: int, seed: int) -> dict:
    biases, norm_biases, realized_values = [], [], []
    for i in range(count):
        b = policy_decision_board(agent, seed + i)
        if not legal_actions(b):
            b = random_decision_board(seed + 100_000 + i)
        predicted = agent.best_action_value(b.raw)
        realized = rollout_return(agent, b.raw, seed + 10_000 + i, gamma=agent.gamma)

        bias = predicted - realized
        biases.append(bias)
        norm_biases.append(bias / (realized + 1.0))
        realized_values.append(realized)

    sum_biases = sum(biases)
    sum_realized_plus_one = sum(realized_values) + len(realized_values)

    return {
        "samples": count,
        "average_bias": safe_mean(biases),
        "average_norm_bias": safe_mean(norm_biases),
        "norm_bias_rom": sum_biases / sum_realized_plus_one if sum_realized_plus_one != 0 else 0.0,
    }


def train_one(config, exp_id: str, variant: str, target_mode: str, feature_mode: str, model_dir: str, worker_id: int = 0):
    board.lookup.init()
    rng = random.Random(config.seed + int(exp_id[-1], 36) * 1000)
    agent = QLearningAgent(target_mode=target_mode, feature_mode=feature_mode)
    
    td_errors, td_errors_m_up, td_errors_m_down = [], [], []
    metrics = {"episodes": [], "td_rms": [], "normalized_td_rms": [], "train_scores": [], "bias": [], "norm_bias": [], "norm_bias_rom": [], "td_errors_m_up": [], "td_errors_m_down": []}
    window_scores = []
    
    td_lambda = getattr(config, 'q_td_lambda', 0.5)
    bias_interval = max(1, min(100, config.q_episodes // 100))
    
    epsilon_start, epsilon_end, epsilon_decay_cutoff = 0.10, 0.0001, 0.80 
    
    # 结合 Dense Matrix 更新较猛的特性，这里降低了初始学习率以防止过拟合
    alpha_start, alpha_end = 0.02, 0.001

    start_time = time.perf_counter()
    pbar_desc = f"{exp_id} {variant}"[:28].ljust(28)
    
    for episode in tqdm(range(1, config.q_episodes + 1), desc=pbar_desc, position=worker_id, leave=True):
        agent.clear_traces()
        b = board()
        popup_with_rng(b, rng)
        popup_with_rng(b, rng)
        steps, score = 0, 0
        
        eps_progress = min(1.0, episode / (config.q_episodes * epsilon_decay_cutoff))
        current_epsilon = epsilon_start - (epsilon_start - epsilon_end) * eps_progress
        
        alpha_progress = episode / config.q_episodes
        current_alpha = alpha_start - (alpha_start - alpha_end) * alpha_progress
        agent.set_alpha(current_alpha)
        
        while config.max_game_steps is None or steps < config.max_game_steps:
            actions = legal_actions(b)
            if not actions: break
            action, _ = agent.choose_action(b, current_epsilon, rng)
            step = agent.update_step(b.raw, action, rng, td_lambda)
            if step.terminal: break
            
            score += step.reward
            td_errors.append(abs(step.td_error))
            
            if target_mode == "mv":
                td_errors_m_up.append(abs(step.td_error_m_up))
                td_errors_m_down.append(abs(step.td_error_m_down))
                
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
            
            td_errors.clear(); window_scores.clear()

            if target_mode == "mv":
                if len(td_errors_m_up) >= config.q_td_window:
                    metrics["td_errors_m_up"].append((episode, math.sqrt(safe_mean(err * err for err in td_errors_m_up))))
                    metrics["td_errors_m_down"].append((episode, math.sqrt(safe_mean(err * err for err in td_errors_m_down))))
                td_errors_m_up.clear(); td_errors_m_down.clear()

        if episode % bias_interval == 0:
            bias_res = collect_bias(agent, 20, config.seed + 40_000 + episode)
            metrics["bias"].append((episode, bias_res["average_bias"]))
            metrics["norm_bias"].append((episode, bias_res["average_norm_bias"]))
            metrics["norm_bias_rom"].append((episode, bias_res["norm_bias_rom"]))

    model_path = os.path.join(model_dir, f"qlearning_{exp_id.lower().replace('-', '')}.pkl")
    agent.save(model_path)

    def choose_action(b: board) -> int: return agent.best_action(b)

    game_records = [
        play_policy(config.seed + 50_000 + i, choose_action, max_steps=config.max_game_steps)
        for i in range(config.q_eval_games)
    ]
    game_summary = summarize_games(game_records)
    elapsed = time.perf_counter() - start_time
    
    final_bias = collect_bias(agent, config.q_bias_samples, config.seed + 80_000)

    n_td = len(metrics["td_rms"])
    tail_n = max(1, n_td // 10) if n_td > 0 else 1
    td_error_rms_final = safe_mean(metrics["td_rms"][-tail_n:]) if metrics["td_rms"] else 0.0
    norm_td_rms_final = safe_mean(metrics["normalized_td_rms"][-tail_n:]) if metrics["normalized_td_rms"] else 0.0

    row = {
        "experiment": exp_id,
        "variant": variant,
        "target_mode": target_mode,
        "feature_mode": feature_mode,
        **game_summary,
        "td_error_rms_final": td_error_rms_final,
        "td_error_rms_mean": safe_mean(metrics["td_rms"]),
        "normalized_td_rms_final": norm_td_rms_final,
        "average_bias": final_bias["average_bias"],
        "average_norm_bias": final_bias["average_norm_bias"],
        "norm_bias_rom": final_bias["norm_bias_rom"],
        "model_path": model_path,
        "train_seconds": elapsed,
    }
    
    return row, metrics


def smooth_curve(points, window=10):
    if len(points) < window: return points
    return np.convolve(points, np.ones(window)/window, mode='valid')

def generate_plots(all_metrics, picture_dir):
    os.makedirs(picture_dir, exist_ok=True)
    colors = {"3-D": "red", "3-E": "blue", "3-F": "purple", "3-G": "green"}
    
    sample_scores_len = len(next(iter(all_metrics.values()))["train_scores"])
    sample_bias_len = len(next(iter(all_metrics.values()))["bias"])
    SCORE_WINDOW = max(1, min(10, sample_scores_len // 10))
    BIAS_WINDOW = max(1, min(10, sample_bias_len // 10))
    
    # 1. 学习曲线
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"]:
            episodes, scores = metrics["episodes"], metrics["train_scores"]
            plt.plot(episodes, scores, color=colors[exp_id], alpha=0.15, linewidth=1)
            if len(scores) >= SCORE_WINDOW:
                plt.plot(episodes[SCORE_WINDOW-1:], smooth_curve(scores, SCORE_WINDOW), label=exp_id, color=colors[exp_id], linewidth=2.5)
            else:
                plt.plot(episodes, scores, label=exp_id, color=colors[exp_id], linewidth=2.5)
    plt.title("Learning Curve (Average Training Score)", fontsize=14)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "learning_curve_smoothed.png"), dpi=300); plt.close()

    # 2. 绝对 TD-Error
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"]:
            episodes, td_rms = metrics["episodes"], metrics["td_rms"]
            plt.plot(episodes, td_rms, color=colors[exp_id], alpha=0.15, linewidth=1)
            if len(td_rms) >= SCORE_WINDOW:
                plt.plot(episodes[SCORE_WINDOW-1:], smooth_curve(td_rms, SCORE_WINDOW), label=exp_id, color=colors[exp_id], linewidth=2)
            else:
                plt.plot(episodes, td_rms, label=exp_id, color=colors[exp_id], linewidth=2)
    plt.title("Convergence Stability (Absolute TD-Error RMS)", fontsize=14)
    plt.yscale("log"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "td_error_volatility_smoothed.png"), dpi=300); plt.close()
    
    # 3. 相对 TD-Error
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if metrics["episodes"] and "normalized_td_rms" in metrics:
            episodes, norm_td_rms = metrics["episodes"], metrics["normalized_td_rms"]
            plt.plot(episodes, norm_td_rms, color=colors[exp_id], alpha=0.15, linewidth=1)
            if len(norm_td_rms) >= SCORE_WINDOW:
                plt.plot(episodes[SCORE_WINDOW-1:], smooth_curve(norm_td_rms, SCORE_WINDOW), label=exp_id, color=colors[exp_id], linewidth=2)
            else:
                plt.plot(episodes, norm_td_rms, label=exp_id, color=colors[exp_id], linewidth=2)
    plt.title("Normalized Convergence (TD-Error RMS / Avg Score)", fontsize=14)
    plt.yscale("log"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "normalized_td_error_smoothed.png"), dpi=300); plt.close()
    
    # 4. Dual M-Head 方差收敛性
    if any(metrics.get("td_errors_m_up") for metrics in all_metrics.values()):
        plt.figure(figsize=(10, 6))
        for exp_id, metrics in all_metrics.items():
            if metrics.get("td_errors_m_up"):
                episodes, m_up = zip(*metrics["td_errors_m_up"])
                plt.plot(episodes, m_up, color='blue', alpha=0.15, linewidth=1)
                if len(m_up) >= SCORE_WINDOW: plt.plot(episodes[SCORE_WINDOW-1:], smooth_curve(m_up, SCORE_WINDOW), label=f"{exp_id} Up-Var", color='blue', linewidth=2)
                
                _, m_down = zip(*metrics["td_errors_m_down"])
                plt.plot(episodes, m_down, color='red', alpha=0.15, linewidth=1)
                if len(m_down) >= SCORE_WINDOW: plt.plot(episodes[SCORE_WINDOW-1:], smooth_curve(m_down, SCORE_WINDOW), label=f"{exp_id} Down-Var", color='red', linewidth=2, linestyle='--')
        plt.title("Dual-Variance Convergence (M-Head TD-Error RMS)", fontsize=14)
        plt.yscale("log"); plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
        plt.savefig(os.path.join(picture_dir, "variance_head_td_error_smoothed.png"), dpi=300); plt.close()

    # 5. 归一化过估计偏差 ROM
    plt.figure(figsize=(10, 6))
    for exp_id, metrics in all_metrics.items():
        if "norm_bias_rom" in metrics and metrics["norm_bias_rom"]:
            episodes, rom_biases = zip(*metrics["norm_bias_rom"])
            plt.plot(episodes, rom_biases, color=colors[exp_id], alpha=0.15, linewidth=1, marker='.', markersize=4)
            if len(rom_biases) >= BIAS_WINDOW:
                plt.plot(episodes[BIAS_WINDOW-1:], smooth_curve(rom_biases, BIAS_WINDOW), label=exp_id, color=colors[exp_id], linewidth=2.5)
            else:
                plt.plot(episodes, rom_biases, label=exp_id, color=colors[exp_id], linewidth=2.5, marker='o')
    plt.axhline(0, color='black', linestyle='--', linewidth=2, label='Zero Bias')
    plt.title("Empirical Overestimation Bias (Ratio-of-Means)", fontsize=14)
    plt.legend(); plt.grid(True, linestyle="--", alpha=0.6); plt.tight_layout()
    plt.savefig(os.path.join(picture_dir, "overestimation_bias_rom_smoothed.png"), dpi=300); plt.close()

def run_experiment_parallel(config):
    timestamp_str = timestamp()
    model_dir = os.path.join("models", "phrase_1", "qlearning_runs", timestamp_str)
    picture_dir = os.path.join("models", "phrase_1", "picture", timestamp_str)
    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(picture_dir, exist_ok=True)

    rows, all_metrics = [], {}
    print(f"\n🚀 Launching {len(EXPERIMENTS)} heavy-duty N-tuple experiments concurrently...")

    with concurrent.futures.ProcessPoolExecutor(max_workers=len(EXPERIMENTS)) as executor:
        futures = [executor.submit(train_one, config, exp, var, tgt, feat, model_dir, i) for i, (exp, var, tgt, feat) in enumerate(EXPERIMENTS)]
        for future in concurrent.futures.as_completed(futures):
            try:
                row, metrics = future.result()
                rows.append(row)
                all_metrics[row["experiment"]] = metrics
                tqdm.write(f"\n✅ {row['experiment']} | Score: {row['average_score']:.0f} | Norm TD: {row['normalized_td_rms_final']:.4f} | ROM Bias: {row['norm_bias_rom']:.4f}")
            except Exception as exc:
                tqdm.write(f"\n❌ 错误: {exc}")

    generate_plots(all_metrics, picture_dir)
    paths = write_result_bundle(config.output_dir, "qlearning_parallel", config, sorted(rows, key=lambda x: x["experiment"]), {})
    print(f"✅ Q-learning 结果已保存: {paths['md']}")

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