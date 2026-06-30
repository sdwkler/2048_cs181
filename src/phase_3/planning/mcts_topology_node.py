# 文件路径: src/phase_3/planning/mcts_topology_node.py
from __future__ import annotations

import math
import random
import sys
import os
from typing import Tuple, Dict, Any

from src.environments.base_env import board

# ==========================================================
# 【进程级模型缓存】：防止多进程环境下的 N-tuple 内存爆炸
# ==========================================================
_PROCESS_MODEL_CACHE = {}

def build_ntuple_safe(eval_type: str, ntuple_path: str):
    from src.ntuple.feature_base import feature, learning, pattern, diff_pattern
    from src.ntuple.loader import fast_mmap_load
    
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

    if not os.path.exists(ntuple_path):
        raise FileNotFoundError(f"missing N-tuple weight file: {ntuple_path}")
    fast_mmap_load(tdl, ntuple_path)
    return tdl


class FastHeuristic:
    _tables_initialized = False
    _empty_table = [0] * 65536
    _smooth_table = [0] * 65536
    _mono_l_table = [0] * 65536
    _mono_r_table = [0] * 65536
    _max_table = [0] * 65536

    @classmethod
    def _init_tables(cls):
        if cls._tables_initialized: return
        for i in range(65536):
            t0, t1, t2, t3 = i & 15, (i >> 4) & 15, (i >> 8) & 15, (i >> 12) & 15
            cls._empty_table[i] = (t0 == 0) + (t1 == 0) + (t2 == 0) + (t3 == 0)
            cls._smooth_table[i] = -(abs(t0 - t1) + abs(t1 - t2) + abs(t2 - t3))
            cls._mono_l_table[i] = (t0 >= t1) + (t1 >= t2) + (t2 >= t3)
            cls._mono_r_table[i] = (t0 <= t1) + (t1 <= t2) + (t2 <= t3)
            cls._max_table[i] = t0 if t0 >= t1 and t0 >= t2 and t0 >= t3 else max(t1, t2, t3)
        cls._tables_initialized = True

    def __init__(self, w_empty=270.0, w_mono=47.0, w_smooth=0.1, w_corner=500.0):
        self._init_tables()
        self.w_empty, self.w_mono = w_empty, w_mono
        self.w_smooth, self.w_corner = w_smooth, w_corner
        self._cache = {}

    def evaluate(self, raw: int, is_afterstate=False) -> float:
        cache_key = (raw, is_afterstate)
        if cache_key in self._cache: return self._cache[cache_key]

        r0, r1, r2, r3 = raw & 0xFFFF, (raw >> 16) & 0xFFFF, (raw >> 32) & 0xFFFF, (raw >> 48) & 0xFFFF
        c0 = (raw & 0xF) | ((raw >> 12) & 0xF0) | ((raw >> 24) & 0xF00) | ((raw >> 36) & 0xF000)
        c1 = ((raw >> 4) & 0xF) | ((raw >> 16) & 0xF0) | ((raw >> 28) & 0xF00) | ((raw >> 40) & 0xF000)
        c2 = ((raw >> 8) & 0xF) | ((raw >> 20) & 0xF0) | ((raw >> 32) & 0xF00) | ((raw >> 44) & 0xF000)
        c3 = ((raw >> 12) & 0xF) | ((raw >> 24) & 0xF0) | ((raw >> 36) & 0xF00) | ((raw >> 48) & 0xF000)

        empty = self._empty_table[r0] + self._empty_table[r1] + self._empty_table[r2] + self._empty_table[r3]
        if is_afterstate and empty > 0: empty -= 1

        smooth = sum(self._smooth_table[x] for x in (r0, r1, r2, r3, c0, c1, c2, c3))
        mono_l = sum(self._mono_l_table[x] for x in (r0, r1, r2, r3))
        mono_r = sum(self._mono_r_table[x] for x in (r0, r1, r2, r3))
        mono_u = sum(self._mono_l_table[x] for x in (c0, c1, c2, c3))
        mono_d = sum(self._mono_r_table[x] for x in (c0, c1, c2, c3))
        mono = max(mono_u, mono_d) + max(mono_l, mono_r)
        
        max_val = max(self._max_table[r0], self._max_table[r1], self._max_table[r2], self._max_table[r3])
        corner = 1 if max_val in (raw & 15, (raw >> 12) & 15, (raw >> 48) & 15, (raw >> 60) & 15) else 0

        score = self.w_empty * empty + self.w_mono * mono + self.w_smooth * smooth + self.w_corner * corner
        self._cache[cache_key] = score
        if len(self._cache) > 300000: self._cache.clear()
        return score


class UnifiedEvaluator:
    def __init__(self, eval_type: str = "heuristic", ntuple_path: str = None):
        self.eval_type = eval_type
        if self.eval_type == "heuristic":
            self.model = FastHeuristic()
        elif self.eval_type == "ntuple":
            cache_key = f"{eval_type}_{ntuple_path}"
            if cache_key not in _PROCESS_MODEL_CACHE:
                _PROCESS_MODEL_CACHE[cache_key] = build_ntuple_safe(self.eval_type, ntuple_path)
            self.model = _PROCESS_MODEL_CACHE[cache_key]
        else:
            raise ValueError(f"Unknown eval_type: {eval_type}")
        self.dummy_board = board()

    def evaluate(self, raw: int, is_afterstate: bool = False) -> float:
        if self.eval_type == "heuristic":
            return self.model.evaluate(raw, is_afterstate)
        else:
            self.dummy_board.raw = raw
            return self.model.estimate(self.dummy_board)


class Node:
    __slots__ = ("visit_count", "value_sum", "children", "is_chance", "is_evaluated", "prior", "last_action")
    def __init__(self, is_chance: bool = False, last_action: int = -1):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, "Node"] = {}
        self.is_chance = is_chance
        self.is_evaluated = False
        self.prior = 1.0
        self.last_action = last_action
        
    def expanded(self) -> bool: return len(self.children) > 0
    def value(self) -> float: return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0

class MCTSAgent:
    def __init__(self, use_afterstate: bool = False, eval_type: str = "heuristic", ntuple_path: str = None, 
                 seed: int | None = None, max_tree_depth: int = 64, rollout_limit: int = 5, p4_prob: float = 0.1):
        self.use_afterstate = use_afterstate
        self.max_tree_depth = max_tree_depth
        self.rollout_limit = rollout_limit
        self.rng = random.Random(seed)
        
        self.evaluator = UnifiedEvaluator(eval_type=eval_type, ntuple_path=ntuple_path)
        self.pb_c_base, self.pb_c_init = 19652.0, 1.25
        self.p4_prob = p4_prob 

        self._legal_actions_cache: dict[int, list[int]] = {}
        self._move_cache: dict[tuple[int, int], tuple[int, int]] = {}
        self.dummy_board = board()
        self.current_max_depth = 0 

    def get_best_action(self, b: board, num_simulations: int = 100) -> tuple[int, float, float, int]:
        self._legal_actions_cache.clear()
        self.current_max_depth = 0
        
        # 彻底删除了全局 MinMaxStats，这里没有任何相关残留

        self.root = Node(is_chance=False, last_action=-1)
        self.root.is_evaluated = True
        legal_actions = self._get_legal_actions_cached(b.raw)
        if not legal_actions: return 0, 0.0, 0.0, 0

        for _ in range(num_simulations):
            self._simulate(self.root, b.raw, depth=0)

        action_scores, action_visits_list = [], []
        best_a, best_visits, best_score = legal_actions[0], -1, -math.inf

        for a in legal_actions:
            if a not in self.root.children: continue
            child = self.root.children[a]
            visits = child.visit_count
            reward, _ = self._apply_move_cached(b.raw, a)
            score = reward + child.value()

            action_scores.append(score)
            action_visits_list.append(visits)

            if visits > best_visits or (visits == best_visits and score > best_score):
                best_visits, best_score, best_a = visits, score, a

        return best_a, self._std(action_scores), self._entropy(action_visits_list, len(legal_actions)), self.current_max_depth

    def _simulate(self, node: Node, current_raw: int, depth: int) -> float:
        self.current_max_depth = max(self.current_max_depth, depth)
        
        if depth >= self.max_tree_depth: return self._rollout(current_raw)

        if not node.is_chance:
            legal_actions = self._get_legal_actions_cached(current_raw)
            if not legal_actions: return 0.0
            
            # --- 1. 节点展开与您的规则一：混合均匀先验 (增加区分度且保底) ---
            if not node.expanded():
                action_scores = []
                for a in legal_actions:
                    r, after_raw = self._apply_move_cached(current_raw, a)
                    score = r + self.evaluator.evaluate(after_raw, is_afterstate=True)
                    action_scores.append(score)
                
                if action_scores:
                    max_score, min_score = max(action_scores), min(action_scores)
                    if max_score == min_score:
                        raw_priors = [1.0 / len(legal_actions)] * len(legal_actions)
                    else:
                        temp = 0.5 
                        exp_scores = [math.exp((s - min_score) / max((max_score - min_score), 1e-5) / temp) for s in action_scores]
                        sum_exp = sum(exp_scores)
                        raw_priors = [e / sum_exp for e in exp_scores]
                    
                    # 按照您的要求，强制注入 20% 均匀保底，防止极端贪婪
                    baseline_prob = 1.0 / len(legal_actions)
                    alpha = 0.20 
                    
                    for a, p in zip(legal_actions, raw_priors):
                        child = Node(is_chance=True, last_action=a)
                        child.prior = (1.0 - alpha) * p + alpha * baseline_prob
                        node.children[a] = child

            if not self.use_afterstate and not node.is_evaluated:
                value = self._rollout(current_raw)
                node.is_evaluated = True
                node.value_sum += value
                node.visit_count += 1
                return value
            elif self.use_afterstate and not node.is_evaluated:
                node.is_evaluated = True

            # ==========================================================
            # 【您的规则二：完全纯净的局部归一化，已彻底删除残余的全局对象】
            # ==========================================================
            
            # 第一步：收集当前已探索的子节点真实价值
            visited_qs = []
            for a in legal_actions:
                if node.children[a].visit_count > 0:
                    reward, _ = self._apply_move_cached(current_raw, a)
                    visited_qs.append(reward + node.children[a].value())
            
            # 第二步：计算局部极值 (如果全部为0/未探索，默认为0.0)
            if visited_qs:
                local_min_val = min(visited_qs)
                local_max_val = max(visited_qs)
            else:
                local_min_val = 0.0
                local_max_val = 0.0
                
            safe_range = max(local_max_val - local_min_val, 1e-5)

            # 第三步：利用您的局部规则分配分值并进行真正的局部归一化
            best_ucb, best_a = -math.inf, legal_actions[0]
            for a in legal_actions:
                child = node.children[a]
                
                if child.visit_count == 0:
                    # 【核心：您的设定】：未探索节点填充至当前最小非零期望
                    actual_val = local_min_val 
                else:
                    reward, _ = self._apply_move_cached(current_raw, a)
                    actual_val = reward + child.value()
                    
                # 【真正的局部归一化】(彻底废弃之前的 self.min_max_stats.normalize)
                val_score = (actual_val - local_min_val) / safe_range
                
                # UCB = 归一化得分 + 先验探索惩罚
                pb_c = math.log((node.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
                # 使用 max(1, node.visit_count) 确保在全为 0 的首发回合，纯靠 Prior 决定优先级
                exploration = (math.sqrt(max(1, node.visit_count)) / (child.visit_count + 1)) * child.prior
                ucb = val_score + pb_c * exploration
                
                if ucb > best_ucb: 
                    best_ucb, best_a = ucb, a

            # ==========================================================

            child_node = node.children[best_a]
            reward, after_raw = self._apply_move_cached(current_raw, best_a)
            total_return = reward + self._simulate(child_node, after_raw, depth + 1)
            
            node.value_sum += total_return
            node.visit_count += 1
            return total_return

        else:
            if self.use_afterstate and not node.is_evaluated:
                value = self._rollout(current_raw)
                node.is_evaluated = True
                node.value_sum += value
                node.visit_count += 1
                return value
            elif not self.use_afterstate and not node.is_evaluated:
                node.is_evaluated = True

            self.dummy_board.raw = current_raw
            self._spawn_in_place(self.dummy_board)
            spawned_raw = self.dummy_board.raw

            if spawned_raw not in node.children: 
                node.children[spawned_raw] = Node(is_chance=False)
            
            value = self._simulate(node.children[spawned_raw], spawned_raw, depth)
            
            node.value_sum += value
            node.visit_count += 1
            return value

    def _rollout(self, state_raw: int) -> float:
        anchor_value = self.evaluator.evaluate(state_raw, is_afterstate=self.use_afterstate)
        
        if self.rollout_limit <= 0:
            return anchor_value

        total = 0.0
        current_raw = state_raw
        
        if self.use_afterstate:
            self.dummy_board.raw = current_raw
            self._spawn_in_place(self.dummy_board)
            current_raw = self.dummy_board.raw

        steps = 0
        while steps < self.rollout_limit:
            legal_actions = self._get_legal_actions_cached(current_raw)
            if not legal_actions: return anchor_value + total

            if self.rng.random() < 0.1:
                action = self.rng.choice(legal_actions)
            else:
                best_a, best_score = legal_actions[0], -math.inf
                for a in legal_actions:
                    r, after_raw = self._apply_move_cached(current_raw, a)
                    score = r + self.evaluator.evaluate(after_raw, is_afterstate=True)
                    if score > best_score: best_score, best_a = score, a
                action = best_a

            reward, after_raw = self._apply_move_cached(current_raw, action)
            if reward == -1: return anchor_value + total
            total += reward

            self.dummy_board.raw = after_raw
            self._spawn_in_place(self.dummy_board)
            current_raw = self.dummy_board.raw
            steps += 1

        return anchor_value + total + self.evaluator.evaluate(current_raw, is_afterstate=False) * 0.1

    def _spawn_in_place(self, b: board) -> None:
        spaces = [i for i in range(16) if b.at(i) == 0]
        if not spaces: return
        b.set(self.rng.choice(spaces), 2 if self.rng.random() < self.p4_prob else 1)

    def _apply_move_cached(self, raw: int, action: int) -> Tuple[int, int]:
        key = (raw, action)
        if key in self._move_cache: return self._move_cache[key]
        self.dummy_board.raw = raw
        reward = self.dummy_board.move(action)
        after_raw = self.dummy_board.raw
        self._move_cache[key] = (reward, after_raw)
        if len(self._move_cache) > 500000: self._move_cache.clear()
        return reward, after_raw

    def _get_legal_actions_cached(self, raw: int) -> list[int]:
        if raw not in self._legal_actions_cache:
            actions = []
            for a in range(4):
                r, _ = self._apply_move_cached(raw, a)
                if r != -1: actions.append(a)
            self._legal_actions_cache[raw] = actions
        return self._legal_actions_cache[raw]

    @staticmethod
    def _std(values: list[float]) -> float:
        if not values: return 0.0
        mean = sum(values) / len(values)
        return math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))

    @staticmethod
    def _entropy(visits: list[int], num_actions: int) -> float:
        total = sum(visits)
        if total <= 0: return math.log(num_actions) if num_actions > 0 else 0.0
        return -sum((v / total) * math.log(v / total) for v in visits if v > 0)
        
    def get_tree_profile(self, max_depth: int = 15) -> list[int]:
        """BFS 遍历树：严格统计每一层(depth)真实展开的盘面数量"""
        profile = [0] * (max_depth + 1)
        queue = [(self.root, 0)]
        
        while queue:
            node, d = queue.pop(0)
            if not node.is_chance and node.visit_count > 0:
                profile[d] += 1
                
            for child in node.children.values():
                next_d = d + 1 if not node.is_chance else d
                if next_d <= max_depth:
                    queue.append((child, next_d))
                    
        return profile