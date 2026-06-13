import argparse
import math
import os
import pickle
import random
import sys
import time
from dataclasses import dataclass

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
]


@dataclass
class StepResult:
    reward: int
    td_error: float
    next_raw: int
    terminal: bool


class SparseNTupleValue:
    def __init__(self, alpha: float = 0.01):
        self.alpha = alpha
        self.weights: dict[tuple[int, int, int, int], float] = {}
        self.isom = self._build_isom()

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
            for iso_id, shape in enumerate(shape_isom):
                yield (feature_id, iso_id, 0, self._pattern_index(shape, b))
                yield (feature_id, iso_id, 1, self._diff_index(shape, b))

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

    def update(self, b: board, td_error: float) -> None:
        indices = list(self._indices(b))
        if not indices:
            return
        delta = self.alpha * td_error / len(indices)
        for index in indices:
            self.weights[index] = self.weights.get(index, 0.0) + delta

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
        alpha: float = 0.01,
        gamma: float = 1.0,
    ):
        self.target_mode = target_mode
        self.feature_mode = feature_mode
        self.gamma = gamma
        if target_mode == "q":
            self.q_heads = [SparseNTupleValue(alpha=alpha) for _ in range(4)]
            self.v_head = None
        else:
            self.q_heads = None
            self.v_head = SparseNTupleValue(alpha=alpha)

    def feature_board(self, state_raw: int, action: int | None = None, after_raw: int | None = None) -> board:
        if self.feature_mode == "afterstate" and after_raw is not None:
            return board(after_raw)
        return board(state_raw)

    def q_value(self, state_raw: int, action: int) -> float:
        after_raw, reward = apply_action(state_raw, action)
        if reward == -1:
            return -float("inf")
        feat = self.feature_board(state_raw, action, after_raw)
        return self.q_heads[action].estimate(feat)

    def v_action_value(self, state_raw: int, action: int) -> float:
        after_raw, reward = apply_action(state_raw, action)
        if reward == -1:
            return -float("inf")
        feat = self.feature_board(state_raw, action, after_raw)
        return reward + self.v_head.estimate(feat)

    def action_value(self, state_raw: int, action: int) -> float:
        if self.target_mode == "q":
            return self.q_value(state_raw, action)
        return self.v_action_value(state_raw, action)

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

    def update_step(self, state_raw: int, action: int, rng: random.Random) -> StepResult:
        after_raw, reward = apply_action(state_raw, action)
        if reward == -1:
            return StepResult(0, 0.0, state_raw, True)

        next_b = board(after_raw)
        popup_with_rng(next_b, rng)
        next_raw = next_b.raw

        if self.target_mode == "q":
            target = reward + self.gamma * self.best_action_value(next_raw)
            feat = self.feature_board(state_raw, action, after_raw)
            current = self.q_heads[action].estimate(feat)
            td_error = target - current
            self.q_heads[action].update(feat, td_error)
        else:
            # V(s') starts after the current slide reward has already been paid.
            # Its Bellman target is the best value available after the random tile.
            target = self.gamma * self.best_action_value(next_raw)
            feat = self.feature_board(state_raw, action, after_raw)
            current = self.v_head.estimate(feat)
            td_error = target - current
            self.v_head.update(feat, td_error)

        return StepResult(reward, td_error, next_raw, False)

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = {
            "target_mode": self.target_mode,
            "feature_mode": self.feature_mode,
            "gamma": self.gamma,
            "q_heads": [head.state_dict() for head in self.q_heads] if self.q_heads else None,
            "v_head": self.v_head.state_dict() if self.v_head else None,
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
        b = random_decision_board(seed + i)
        predicted = agent.best_action_value(b.raw)
        realized = rollout_return(agent, b.raw, seed + 10_000 + i, gamma=agent.gamma)
        biases.append(predicted - realized)
    return {
        "samples": count,
        "average_bias": safe_mean(biases),
        "max_bias": max(biases) if biases else 0.0,
        "min_bias": min(biases) if biases else 0.0,
    }


def train_one(config, exp_id: str, variant: str, target_mode: str, feature_mode: str):
    board.lookup.init()
    rng = random.Random(config.seed + int(exp_id[-1], 36) * 1000)
    agent = QLearningAgent(target_mode=target_mode, feature_mode=feature_mode)
    td_errors, td_windows = [], []
    bias = None
    start_time = time.perf_counter()

    for episode in progress(range(1, config.q_episodes + 1), desc=variant[:24], leave=False):
        b = board()
        popup_with_rng(b, rng)
        popup_with_rng(b, rng)
        epsilon = epsilon_for_episode(episode, config.q_episodes)
        steps = 0
        while config.max_game_steps is None or steps < config.max_game_steps:
            actions = legal_actions(b)
            if not actions:
                break
            action = agent.choose_action(b, epsilon, rng)
            step = agent.update_step(b.raw, action, rng)
            if step.terminal:
                break
            td_errors.append(abs(step.td_error))
            if len(td_errors) >= config.q_td_window:
                rms = math.sqrt(safe_mean(err * err for err in td_errors))
                td_windows.append({"episode": episode, "td_error_rms": rms})
                td_errors.clear()
            b = board(step.next_raw)
            steps += 1

        if bias is None and episode >= config.q_bias_episode:
            bias = collect_bias(agent, config.q_bias_samples, config.seed + 40_000 + episode)

    if td_errors:
        rms = math.sqrt(safe_mean(err * err for err in td_errors))
        td_windows.append({"episode": config.q_episodes, "td_error_rms": rms})
    if bias is None:
        bias = collect_bias(agent, config.q_bias_samples, config.seed + 40_000 + config.q_episodes)

    model_path = os.path.join("models", f"qlearning_{exp_id.lower().replace('-', '')}_{target_mode}_{feature_mode}.pkl")
    agent.save(model_path)

    def choose_action(b: board) -> int:
        return agent.best_action(b)

    game_records = [
        play_policy(config.seed + 50_000 + i, choose_action, max_steps=config.max_game_steps)
        for i in range(config.q_eval_games)
    ]
    game_summary = summarize_games(game_records)
    elapsed = time.perf_counter() - start_time
    row = {
        "experiment": exp_id,
        "variant": variant,
        "target_mode": target_mode,
        "feature_mode": feature_mode,
        **game_summary,
        "td_error_rms": td_windows[-1]["td_error_rms"] if td_windows else 0.0,
        "average_bias": bias["average_bias"],
        "model_path": model_path,
        "train_seconds": elapsed,
    }
    detail = {"td_windows": td_windows, "bias": bias, "games": game_records}
    return row, detail


def run_experiment(config):
    rows, details = [], {}
    for exp_id, variant, target_mode, feature_mode in EXPERIMENTS:
        row, detail = train_one(config, exp_id, variant, target_mode, feature_mode)
        rows.append(row)
        details[exp_id] = detail
        print(
            f"{exp_id} {variant}: score={row['average_score']:.1f}, "
            f"td_rms={row['td_error_rms']:.2f}, bias={row['average_bias']:.2f}"
        )
    paths = write_result_bundle(config.output_dir, "qlearning", config, rows, details)
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
