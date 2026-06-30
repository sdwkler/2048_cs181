# 文件路径: src/phase_2/mcts_unified_node.py
from __future__ import annotations

import math
import random
from typing import Tuple, Dict, Any
from src.environments.base_env import board
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
# 【内存核武】：进程级别的全局缓存，确保每个 Worker 进程只加载一次模型
_PROCESS_MODEL_CACHE = {}

def build_ntuple_safe(eval_type: str, ntuple_path: str):
    """带 alloc 劫持的安全实例化，完美避开初始化的内存峰值"""
    from src.ntuple.feature_base import feature, learning, pattern, diff_pattern
    from src.ntuple.loader import fast_mmap_load
    
    original_alloc = feature.alloc
    old_stdout = sys.stdout
    devnull = open(os.devnull, "w")
    sys.stdout = devnull
    try:
        # 瞒天过海：骗过 Python 的内存分配
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


class UnifiedEvaluator:
    """统一评估适配器：融合了 Phase 1 极速加载与进程缓存"""
    def __init__(self, eval_type: str = "heuristic", ntuple_path: str = None):
        self.eval_type = eval_type
        
        if self.eval_type == "heuristic":
            from src.evaluators import FastHeuristic
            self.model = FastHeuristic()
        elif self.eval_type == "ntuple":
            # O(1) 拦截：如果当前进程已经加载过该模型，直接复用内存映射
            cache_key = f"{eval_type}_{ntuple_path}"
            if cache_key not in _PROCESS_MODEL_CACHE:
                _PROCESS_MODEL_CACHE[cache_key] = build_ntuple_safe(self.eval_type, ntuple_path)
                
            self.model = _PROCESS_MODEL_CACHE[cache_key]
        else:
            raise ValueError(f"未知的 eval_type: {eval_type}")
            
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
                 seed: int | None = None, max_tree_depth: int = 64, rollout_limit: int = 0, p4_prob: float = 0.1):
        self.use_afterstate = use_afterstate
        self.max_tree_depth = max_tree_depth
        
        # 将 rollout 限制暴露出来。设为 0 则代表【纯净的静态估值截断】，不往下模拟。
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

        root_entropy = self._entropy(action_visits_list, len(legal_actions))
        root_std = self._std(action_scores)
        return best_a, root_std, root_entropy, self.current_max_depth

    def _simulate(self, node: Node, current_raw: int, depth: int) -> float:
        self.current_max_depth = max(self.current_max_depth, depth)
        
        if depth >= self.max_tree_depth: return self._rollout(current_raw)

        if not node.is_chance:
            legal_actions = self._get_legal_actions_cached(current_raw)
            if not legal_actions: return 0.0
            
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
                        exp_scores = [math.exp((s - min_score) / (max_score - min_score) / temp) for s in action_scores]
                        sum_exp = sum(exp_scores)
                        raw_priors = [e / sum_exp for e in exp_scores]
                    
                    is_root = (depth == 0)
                    alpha = 0.25 if is_root else 0.0 
                    uniform_p = 1.0 / len(legal_actions)
                    
                    for a, p in zip(legal_actions, raw_priors):
                        child = Node(is_chance=True, last_action=a)
                        child.prior = (1.0 - alpha) * p + alpha * uniform_p
                        node.children[a] = child

            if not self.use_afterstate and not node.is_evaluated:
                value = self._rollout(current_raw)
                node.is_evaluated = True
                node.value_sum += value
                node.visit_count += 1
                return value
            elif self.use_afterstate and not node.is_evaluated:
                node.is_evaluated = True

            visited_qs = []
            for a in legal_actions:
                if node.children[a].visit_count > 0:
                    reward, _ = self._apply_move_cached(current_raw, a)
                    visited_qs.append(reward + node.children[a].value())
            
            local_max, local_min = (max(visited_qs), min(visited_qs)) if visited_qs else (0.0, 0.0)
            safe_range = max(local_max - local_min, 200.0)

            best_ucb, best_a = -math.inf, legal_actions[0]
            for a in legal_actions:
                child = node.children[a]
                if child.visit_count == 0:
                    val_score = 0.0 # FPU 保底，等于当前最差期望
                else:
                    reward, _ = self._apply_move_cached(current_raw, a)
                    actual_val = reward + child.value()
                    val_score = (actual_val - local_min) / safe_range
                    
                pb_c = math.log((node.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
                ucb = val_score + (pb_c * math.sqrt(node.visit_count) / (child.visit_count + 1)) * child.prior
                
                if ucb > best_ucb: best_ucb, best_a = ucb, a

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
        """
        完全独立的 Rollout 函数。你可以直接在这里修改逻辑：
        如果 rollout_limit == 0，相当于 AlphaZero 的直接截断估值。
        """
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

            # 简单的 e-greedy Rollout (10% 随机，90% 贪心估值)
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

        return anchor_value + total + (self.evaluator.evaluate(current_raw, is_afterstate=False) * 0.1)

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
        
    def export_tree_topology(self) -> Dict[str, Any]:
        """导出树结构，用于直观分析算力分布"""
        action_names = {0: "上(Up)", 1: "右(Right)", 2: "下(Down)", 3: "左(Left)"}
        root_data = {"visits": self.root.visit_count, "children": []}
        for a, child in self.root.children.items():
            root_data["children"].append({
                "action": action_names.get(a, str(a)),
                "visits": child.visit_count,
                "q_value": round(child.value(), 1),
                "prior": round(child.prior, 3)
            })
        # 按照访问量排序
        root_data["children"].sort(key=lambda x: x["visits"], reverse=True)
        return root_data