from __future__ import annotations

import math
import random

from src.environments.base_env import board


class MCTSAgent:
    """MCTS with terminal random rollouts and optional afterstate aggregation."""

    def __init__(
        self,
        use_afterstate: bool = False,
        exploration_c: float = 2000.0,
        seed: int | None = None,
        max_tree_depth: int = 64,
    ):
        self.use_afterstate = use_afterstate
        self.exploration_c = exploration_c
        self.max_tree_depth = max_tree_depth
        self.rng = random.Random(seed)
        self.Q: dict[tuple[int, int] | int, float] = {}
        self.N: dict[tuple[int, int] | int, int] = {}
        self.N_s: dict[int, int] = {}
        self.visited: set[tuple[int, int] | int] = set()
        self.last_root_action_values: dict[int, float] = {}
        self.last_root_action_visits: dict[int, int] = {}

    def get_best_action(self, b: board, num_simulations: int = 100) -> tuple[int, float, float]:
        self.Q.clear()
        self.N.clear()
        self.N_s.clear()
        self.visited.clear()
        self.last_root_action_values.clear()
        self.last_root_action_visits.clear()

        for _ in range(num_simulations):
            self._simulate_state(b.raw)

        legal_actions = self._legal_actions(b.raw)
        if not legal_actions:
            return 0, 0.0, 0.0

        action_scores, action_visits = [], []
        best_action, best_visits, best_score = legal_actions[0], -1, -math.inf
        for action in legal_actions:
            after_raw, reward = self._apply_action(b.raw, action)
            if self.use_afterstate:
                key = after_raw
                future = self.Q.get(key, 0.0) / max(1, self.N.get(key, 0))
                score = reward + future
            else:
                key = (b.raw, action)
                score = self.Q.get(key, 0.0) / max(1, self.N.get(key, 0))

            visits = self.N.get(key, 0)
            self.last_root_action_values[action] = score
            self.last_root_action_visits[action] = visits
            action_scores.append(score)
            action_visits.append(visits)
            if visits > best_visits or (visits == best_visits and score > best_score):
                best_visits, best_score, best_action = visits, score, action

        return best_action, self._std(action_scores), self._entropy(action_visits, len(legal_actions))

    def _simulate_state(self, state_raw: int, tree_depth: int = 0) -> float:
        legal_actions = self._legal_actions(state_raw)
        if not legal_actions:
            return 0.0
        if tree_depth >= self.max_tree_depth:
            return self._rollout_state(state_raw)

        self.N_s[state_raw] = self.N_s.get(state_raw, 0) + 1
        action, after_raw, reward = self._select_action(state_raw, legal_actions)
        key = after_raw if self.use_afterstate else (state_raw, action)

        if key not in self.visited:
            self.visited.add(key)
            future = self._rollout_afterstate(after_raw)
        else:
            next_raw = self._spawn(after_raw)
            future = self._simulate_state(next_raw, tree_depth + 1)

        if self.use_afterstate:
            stored_return = future
            total_return = reward + future
        else:
            stored_return = reward + future
            total_return = stored_return

        self.Q[key] = self.Q.get(key, 0.0) + stored_return
        self.N[key] = self.N.get(key, 0) + 1
        return total_return

    def _select_action(self, state_raw: int, legal_actions: list[int]) -> tuple[int, int, int]:
        n_state = max(1, self.N_s.get(state_raw, 1))
        best_action, best_after_raw, best_reward = legal_actions[0], state_raw, 0
        best_ucb = -math.inf
        for action in legal_actions:
            after_raw, reward = self._apply_action(state_raw, action)
            key = after_raw if self.use_afterstate else (state_raw, action)
            visits = self.N.get(key, 0)
            if visits == 0:
                ucb = math.inf
            else:
                mean = self.Q.get(key, 0.0) / visits
                if self.use_afterstate:
                    mean += reward
                ucb = mean + self.exploration_c * math.sqrt(math.log(n_state + 1) / visits)
            if ucb > best_ucb:
                best_ucb = ucb
                best_action, best_after_raw, best_reward = action, after_raw, reward
        return best_action, best_after_raw, best_reward

    def _rollout_afterstate(self, after_raw: int) -> float:
        return self._rollout_state(self._spawn(after_raw))

    def _rollout_state(self, state_raw: int) -> float:
        total = 0.0
        current = board(state_raw)
        while True:
            legal_actions = self._legal_actions(current.raw)
            if not legal_actions:
                return total
            action = self.rng.choice(legal_actions)
            reward = current.move(action)
            if reward == -1:
                return total
            total += reward
            self._spawn_in_place(current)

    def _spawn(self, raw: int) -> int:
        spawned = board(raw)
        self._spawn_in_place(spawned)
        return spawned.raw

    def _spawn_in_place(self, b: board) -> None:
        spaces = [i for i in range(16) if b.at(i) == 0]
        if not spaces:
            return
        b.set(self.rng.choice(spaces), 2 if self.rng.random() < 0.1 else 1)

    @staticmethod
    def _legal_actions(raw: int) -> list[int]:
        actions = []
        for action in range(4):
            trial = board(raw)
            if trial.move(action) != -1:
                actions.append(action)
        return actions

    @staticmethod
    def _apply_action(raw: int, action: int) -> tuple[int, int]:
        trial = board(raw)
        reward = trial.move(action)
        return trial.raw, reward

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))

    @staticmethod
    def _entropy(visits: list[int], action_count: int) -> float:
        total = sum(visits)
        if total <= 0:
            return math.log(action_count) if action_count > 0 else 0.0
        entropy = 0.0
        for visit in visits:
            if visit:
                p = visit / total
                entropy -= p * math.log(p)
        return entropy
