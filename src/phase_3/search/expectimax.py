from __future__ import annotations

import math
from collections.abc import Callable

from src.environments.base_env import board


class GreedyAgent:
    """One-ply baseline"""
    def __init__(self, value_func: Callable[[board, bool], float]):
        self.value_func = value_func

    def get_best_action(self, b: board, max_depth: int = 1) -> tuple[int, float, float, int]:
        best_action, best_value = -1, -math.inf
        total, unique_afterstates = 0, set()
        for action in range(4):
            after = board(b.raw)
            reward = after.move(action)
            if reward == -1:
                continue
            total += 1
            unique_afterstates.add(after.raw)
            # 【精确标识】：贪心算法评估的是滑动后的盘面，属于 Afterstate (True)
            value = reward + self.value_func(after, True)
            if value > best_value:
                best_action, best_value = action, value

        if best_action == -1:
            return 0, 0.0, 0.0, 0
        compression_ratio = len(unique_afterstates) / max(1, total)
        # 返回: 最佳动作, 压缩率, 分支效率(b_eff), 展开节点数(贪心为合法动作数)
        return best_action, compression_ratio, float(total), total


class ExpectimaxAgent:
    def __init__(
        self,
        use_afterstate: bool = False,
        value_func: Callable[[board, bool], float] | None = None,
        value_func_risk: Callable[[board, bool], float] | None = None,  # 【新增】：风险价值评估器
        risk_beta: float = 2.0,  # 【新增】：风险厌恶系数，控制对跌幅的恐惧程度
        leaf_mode: str = "state",
        use_pruning: bool = False,
        prune_top_k: int = 2,
        p4_prob: float = 0.1,
    ):
        if value_func is None:
            raise ValueError("value_func is required")
        if leaf_mode not in {"state", "afterstate", "afterstate_dual_risk"}:
            raise ValueError("leaf_mode must be 'state', 'afterstate', or 'afterstate_dual_risk'")

        self.use_afterstate = use_afterstate
        self.leaf_mode = leaf_mode
        self.value_func = value_func
        self.value_func_risk = value_func_risk
        self.risk_beta = risk_beta
        self.use_pruning = use_pruning
        self.prune_top_k = prune_top_k
        self.p4_prob = p4_prob
        self.p2_prob = 1.0 - p4_prob

        # 统计仪盘表
        self.total_nodes_expanded = 0
        self.total_metric_visits = 0
        self.unique_metric_keys: set[tuple | int] = set()

    def get_best_action(self, b: board, max_depth: int = 3) -> tuple[int, float, float, int]:
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
            
            # 【精确标识】：剪枝预判是在评估滑动后的盘面，属于 Afterstate (True)
            immediate_val = reward + self.value_func(after, True) if self.use_pruning else 0.0
            candidates.append((action, reward, after.raw, immediate_val))

        if not candidates:
            return 0, 0.0, 0.0, 0

        if self.use_pruning:
            candidates.sort(key=lambda x: x[3], reverse=True)
            candidates = candidates[:self.prune_top_k]

        for action, reward, after_raw, _ in candidates:
            self._record_decision_child(b.raw, action, after_raw)
            value = reward + self._after_action_value(after_raw, depth - 1)
            if value > best_value:
                best_value, best_action = value, action

        # 压缩率 = 唯一拓扑数 / 逻辑访问总数
        compression_ratio = len(self.unique_metric_keys) / max(1, self.total_metric_visits)
        b_eff = self.total_nodes_expanded ** (1.0 / depth) if self.total_nodes_expanded else 0.0
        
        return best_action, compression_ratio, b_eff, self.total_nodes_expanded

    def _record_decision_child(self, state_raw: int, action: int, after_raw: int) -> None:
        self.total_metric_visits += 1
        if self.use_afterstate:
            self.unique_metric_keys.add(after_raw)
        else:
            self.unique_metric_keys.add((state_raw, action))

    def _after_action_value(self, after_raw: int, plies_remaining: int) -> float:
        self.total_nodes_expanded += 1
        
        if plies_remaining <= 0:
            return self._leaf_value_after_action(after_raw)
        
        return self._chance_value(after_raw, plies_remaining)

    def _chance_value(self, after_raw: int, plies_remaining: int) -> float:
        self.total_nodes_expanded += 1
        after = board(after_raw)
        empties = [i for i in range(16) if after.at(i) == 0]
        
        if not empties:
            # 【精确标识】：没有空地无法发牌，此刻依然是 Afterstate，True
            # 注意：此处为死局防范，如果在 dual_risk 模式，也可仅返回 mean 或减去风险，这里简化为原价值
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

        best_value, has_action = -math.inf, False
        candidates = []

        for action in range(4):
            after = board(state_raw)
            reward = after.move(action)
            if reward == -1: continue
            has_action = True
            
            # 【精确标识】：剪枝预判，True
            immediate_val = reward + self.value_func(after, True) if self.use_pruning else 0.0
            candidates.append((action, reward, after.raw, immediate_val))

        if not has_action:
            # 【精确标识】：死局，此时传入的是发牌后的完整盘面，属于 State，False
            return self.value_func(board(state_raw), False)

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
        # 【架构级斩断 1】：传统 Afterstate 模式，直接打分
        if self.leaf_mode == "afterstate":
            return self.value_func(board(after_raw), True)

        # =========================================================
        # 【架构级斩断 2：完美映射 RSZ 理论的双轨制 Risk-Aware 模式】
        # O(1) 极速查询：不展开任何子节点，通过查表利用训练好的 V_risk 实现惩罚
        # =========================================================
        if self.leaf_mode == "afterstate_dual_risk":
            mean_val = self.value_func(board(after_raw), True)
            risk_val = self.value_func_risk(board(after_raw), True) if self.value_func_risk else 0.0
            
            # Risk 惩罚：利用风险厌恶系数 Beta 扣除预估跌幅
            return mean_val - self.risk_beta * risk_val

        # --- 传统 State 模式 (需要展开子节点算期望，计算量爆炸) ---
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
                self.total_nodes_expanded += 1
                # 【精确标识】：模拟发牌后的完整盘面，属于 State！False
                expected_value += weight * tile_prob * self.value_func(spawned, False)
        return expected_value