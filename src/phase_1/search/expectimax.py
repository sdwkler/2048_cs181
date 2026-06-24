from __future__ import annotations

import math
from collections.abc import Callable

from src.environments.base_env import board


class GreedyAgent:
    """One-ply baseline"""
    def __init__(self, value_func: Callable[[board], float]):
        self.value_func = value_func

    def get_best_action(self, b: board, max_depth: int = 1) -> tuple[int, float, float]:
        best_action, best_value = -1, -math.inf
        total, unique_afterstates = 0, set()
        for action in range(4):
            after = board(b.raw)
            reward = after.move(action)
            if reward == -1:
                continue
            total += 1
            unique_afterstates.add(after.raw)
            value = reward + self.value_func(after)
            if value > best_value:
                best_action, best_value = action, value

        if best_action == -1:
            return 0, 0.0, 0.0
        compression_ratio = len(unique_afterstates) / max(1, total)
        return best_action, compression_ratio, float(total)


class ExpectimaxAgent:
    def __init__(
        self,
        use_afterstate: bool = False,
        value_func: Callable[[board], float] | None = None,
        leaf_mode: str = "state",
        use_pruning: bool = False,   # 【新增：剪枝开关】
        prune_top_k: int = 2,        # 【新增：剪枝保留的顶级分支数】
        p4_prob: float = 0.1,        # 【中层接口：环境漂移概率】
    ):
        if value_func is None:
            raise ValueError("value_func is required")
        if leaf_mode not in {"state", "afterstate"}:
            raise ValueError("leaf_mode must be 'state' or 'afterstate'")

        self.use_afterstate = use_afterstate
        self.leaf_mode = leaf_mode
        self.value_func = value_func
        self.use_pruning = use_pruning
        self.prune_top_k = prune_top_k
        self.p4_prob = p4_prob
        self.p2_prob = 1.0 - p4_prob

        # 【幽灵缓存机制】：
        # 废弃直接阻断搜索的 dict 缓存。仅记录唯一状态数，从而无损计算压缩率，
        # 保证时间统计不被缓存命中污染！
        self.total_nodes_expanded = 0
        self.total_metric_visits = 0
        self.unique_metric_keys: set[tuple | int] = set()

    def get_best_action(self, b: board, max_depth: int = 3) -> tuple[int, float, float]:
        self.total_nodes_expanded = 0
        self.total_metric_visits = 0
        self.unique_metric_keys.clear()

        best_value, best_action = -math.inf, -1
        depth = max(1, max_depth)
        candidates = []

        for action in range(4):
            after = board(b.raw)
            reward = after.move(action)
            if reward == -1: continue
            
            # 如果开启剪枝，则利用打分器提前预判
            immediate_val = reward + self.value_func(after) if self.use_pruning else 0.0
            candidates.append((action, reward, after.raw, immediate_val))

        if not candidates:
            return 0, 0.0, 0.0

        # 执行前向动作剪枝
        if self.use_pruning:
            candidates.sort(key=lambda x: x[3], reverse=True)
            candidates = candidates[:self.prune_top_k]

        for action, reward, after_raw, _ in candidates:
            self._record_decision_child(b.raw, action, after_raw)
            value = reward + self._after_action_value(after_raw, depth - 1)
            if value > best_value:
                best_value, best_action = value, action

        compression_ratio = len(self.unique_metric_keys) / max(1, self.total_metric_visits)
        b_eff = self.total_nodes_expanded ** (1.0 / depth) if self.total_nodes_expanded else 0.0
        return best_action, compression_ratio, b_eff

    def _record_decision_child(self, state_raw: int, action: int, after_raw: int) -> None:
        self.total_metric_visits += 1
        # 【拓扑压缩的核心】：use_afterstate 决定了我们认为什么叫“同一个节点”
        if self.use_afterstate:
            self.unique_metric_keys.add(after_raw)
        else:
            self.unique_metric_keys.add((state_raw, action))

    def _after_action_value(self, after_raw: int, plies_remaining: int) -> float:
        self.total_nodes_expanded += 1
        if plies_remaining <= 0:
            return self._leaf_value_after_action(after_raw)
        
        # 幽灵往下走：彻底删除了 if key in table return 逻辑
        return self._chance_value(after_raw, plies_remaining)

    def _chance_value(self, after_raw: int, plies_remaining: int) -> float:
        self.total_nodes_expanded += 1
        after = board(after_raw)
        empties = [i for i in range(16) if after.at(i) == 0]
        if not empties:
            return self.value_func(after)

        weight = 1.0 / len(empties)
        expected_value = 0.0
        for pos in empties:
            for tile, tile_prob in ((1, self.p2_prob), (2, self.p4_prob)):
                spawned = board(after_raw)
                spawned.set(pos, tile)
                expected_value += weight * tile_prob * self._state_value(spawned.raw, plies_remaining - 1)
        return expected_value

    def _state_value(self, state_raw: int, plies_remaining: int) -> float:
        self.total_nodes_expanded += 1
        if plies_remaining <= 0:
            return self.value_func(board(state_raw))

        best_value, has_action = -math.inf, False
        candidates = []

        for action in range(4):
            after = board(state_raw)
            reward = after.move(action)
            if reward == -1: continue
            has_action = True
            
            immediate_val = reward + self.value_func(after) if self.use_pruning else 0.0
            candidates.append((action, reward, after.raw, immediate_val))

        if not has_action:
            return self.value_func(board(state_raw))

        # 执行前向动作剪枝
        if self.use_pruning:
            candidates.sort(key=lambda x: x[3], reverse=True)
            candidates = candidates[:self.prune_top_k]

        for action, reward, after_raw, _ in candidates:
            self._record_decision_child(state_raw, action, after_raw)
            value = reward + self._after_action_value(after_raw, plies_remaining - 1)
            if value > best_value:
                best_value = value

        return best_value

    def _leaf_value_after_action(self, after_raw: int) -> float:
        # 【算力斩断的核心】：leaf_mode 决定了我们在叶子节点用不用 for 循环！
        if self.leaf_mode == "afterstate":
            return self.value_func(board(after_raw))

        after = board(after_raw)
        empties = [i for i in range(16) if after.at(i) == 0]
        if not empties:
            return self.value_func(after)

        weight = 1.0 / len(empties)
        expected_value = 0.0
        for pos in empties:
            for tile, tile_prob in ((1, self.p2_prob), (2, self.p4_prob)):
                spawned = board(after_raw)
                spawned.set(pos, tile)
                self.total_nodes_expanded += 1
                expected_value += weight * tile_prob * self.value_func(spawned)
        return expected_value