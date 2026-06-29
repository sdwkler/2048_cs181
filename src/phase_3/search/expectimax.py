# src/phase_1/search/expectimax.py
from __future__ import annotations

import math
from collections.abc import Callable

from src.environments.base_env import board


class GreedyAgent:
    """One-ply baseline"""
    def __init__(self, value_func: Callable[[board, bool], float]):
        self.value_func = value_func

    def get_best_action(self, b: board, max_depth: int = 1) -> tuple[int, float, float, int, int]:
        best_action, best_value = -1, -math.inf
        total, unique_afterstates = 0, set()
        for action in range(4):
            after = board(b.raw)
            reward = after.move(action)
            if reward == -1:
                continue
            total += 1
            unique_afterstates.add(after.raw)
            value = reward + self.value_func(after, True)
            if value > best_value:
                best_action, best_value = action, value

        if best_action == -1:
            return 0, 0.0, 0.0, 0, 0
            
        compression_ratio = len(unique_afterstates) / max(1, total)
        # 返回: 动作, 压缩率, 分支因子, 展开节点数, 哈希命中数(贪心为0)
        return best_action, compression_ratio, float(total), total, 0


class ExpectimaxAgent:
    def __init__(
        self,
        value_func: Callable[[board, bool], float] | None = None,
        use_beam_search: bool = False,
        beam_width: int = 2,
        use_tt: bool = False,
        p4_prob: float = 0.1,
    ):
        if value_func is None:
            raise ValueError("value_func is required")

        # 默认全部使用 Afterstate 拓扑和公式 (符合当前消融实验需求)
        self.use_afterstate = True
        self.leaf_mode = "afterstate"
        self.value_func = value_func
        
        # 优化策略开关
        self.use_beam_search = use_beam_search
        self.beam_width = beam_width
        self.use_tt = use_tt
        
        self.p4_prob = p4_prob
        self.p2_prob = 1.0 - p4_prob

        # 统计仪盘表
        self.total_nodes_expanded = 0
        self.total_metric_visits = 0
        self.unique_metric_keys: set[int] = set()
        
        # DAG 图哈希缓存 (Transposition Table)
        self.tt: dict[tuple[int, int, str], float] = {}
        self.tt_hits = 0

    def get_best_action(self, b: board, max_depth: int = 3) -> tuple[int, float, float, int, int]:
        self.total_nodes_expanded = 0
        self.total_metric_visits = 0
        self.unique_metric_keys.clear()
        
        # 每次决策前清空哈希表，防止内存泄漏，且单步决策的 DAG 足够产生大量命中
        self.tt.clear()
        self.tt_hits = 0

        best_value, best_action = -math.inf, -1
        depth = max(1, max_depth)
        candidates = []

        for action in range(4):
            after = board(b.raw)
            reward = after.move(action)
            if reward == -1: continue
            
            # Beam Search 预评估
            immediate_val = reward + self.value_func(after, True) if self.use_beam_search else 0.0
            candidates.append((action, reward, after.raw, immediate_val))

        if not candidates:
            return 0, 0.0, 0.0, 0, 0

        # 执行 Beam Search 剪枝
        if self.use_beam_search:
            candidates.sort(key=lambda x: x[3], reverse=True)
            candidates = candidates[:self.beam_width]

        for action, reward, after_raw, _ in candidates:
            self._record_decision_child(after_raw)
            value = reward + self._after_action_value(after_raw, depth - 1)
            if value > best_value:
                best_value, best_action = value, action

        compression_ratio = len(self.unique_metric_keys) / max(1, self.total_metric_visits)
        b_eff = self.total_nodes_expanded ** (1.0 / depth) if self.total_nodes_expanded else 0.0
        
        return best_action, compression_ratio, b_eff, self.total_nodes_expanded, self.tt_hits

    def _record_decision_child(self, after_raw: int) -> None:
        self.total_metric_visits += 1
        self.unique_metric_keys.add(after_raw)

    def _after_action_value(self, after_raw: int, plies_remaining: int) -> float:
        self.total_nodes_expanded += 1
        
        if plies_remaining <= 0:
            return self.value_func(board(after_raw), True)
            
        # 哈希表读取
        if self.use_tt:
            tt_key = (after_raw, plies_remaining, 'chance')
            if tt_key in self.tt:
                self.tt_hits += 1
                return self.tt[tt_key]
        
        val = self._chance_value(after_raw, plies_remaining)
        
        # 哈希表写入
        if self.use_tt:
            self.tt[tt_key] = val
        return val

    def _chance_value(self, after_raw: int, plies_remaining: int) -> float:
        after = board(after_raw)
        empties = [i for i in range(16) if after.at(i) == 0]
        
        if not empties:
            return self.value_func(after, True)

        weight = 1.0 / len(empties)
        expected_value = 0.0
        for pos in empties:
            for tile, tile_prob in ((1, self.p2_prob), (2, self.p4_prob)):
                spawned = board(after_raw)
                spawned.set(pos, tile)
                expected_value += weight * tile_prob * self._state_value(spawned.raw, plies_remaining)
        return expected_value

    def _state_value(self, state_raw: int, plies_remaining: int) -> float:
        self.total_nodes_expanded += 1

        # 哈希表读取
        if self.use_tt:
            tt_key = (state_raw, plies_remaining, 'state')
            if tt_key in self.tt:
                self.tt_hits += 1
                return self.tt[tt_key]

        best_value, has_action = -math.inf, False
        candidates = []

        for action in range(4):
            after = board(state_raw)
            reward = after.move(action)
            if reward == -1: continue
            has_action = True
            
            immediate_val = reward + self.value_func(after, True) if self.use_beam_search else 0.0
            candidates.append((action, reward, after.raw, immediate_val))

        if not has_action:
            val = self.value_func(board(state_raw), False)
            if self.use_tt: self.tt[tt_key] = val
            return val

        if self.use_beam_search:
            candidates.sort(key=lambda x: x[3], reverse=True)
            candidates = candidates[:self.beam_width]

        for action, reward, after_raw, _ in candidates:
            self._record_decision_child(after_raw)
            value = reward + self._after_action_value(after_raw, plies_remaining - 1)
            if value > best_value:
                best_value = value

        # 哈希表写入
        if self.use_tt:
            self.tt[tt_key] = best_value
            
        return best_value