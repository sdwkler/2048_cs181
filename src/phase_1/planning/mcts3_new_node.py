# src/phase_1/planning/mcts_ultimate.py
from __future__ import annotations

import math
import random

from src.environments.base_env import board


# ==========================================================
# 【核武级优化】：O(1) 静态查表启发式
# 预先计算 65536 种行的属性，利用纯位移操作消灭所有 for 循环！
# ==========================================================
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
            cls._max_table[i] = max(t0, t1, t2, t3)
        cls._tables_initialized = True

    def __init__(self, w_empty=270.0, w_mono=47.0, w_smooth=0.1, w_corner=500.0):
        self._init_tables()
        self.w_empty, self.w_mono = w_empty, w_mono
        self.w_smooth, self.w_corner = w_smooth, w_corner
        self._cache = {}

    def evaluate(self, raw: int, is_afterstate=False) -> float:
        cache_key = (raw, is_afterstate)
        if cache_key in self._cache: return self._cache[cache_key]

        # 横行提取
        r0, r1 = raw & 0xFFFF, (raw >> 16) & 0xFFFF
        r2, r3 = (raw >> 32) & 0xFFFF, (raw >> 48) & 0xFFFF
        
        # 竖列提取 (纯位移操作，速度极快)
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

# ==========================================================
# 2. 核心架构: Stochastic MuZero 二分严格树
# ==========================================================
class MinMaxStats:
    def __init__(self):
        self.maximum = -float('inf')
        self.minimum = float('inf')
    def update(self, value: float):
        self.maximum = max(self.maximum, value)
        self.minimum = min(self.minimum, value)
    def normalize(self, value: float) -> float:
        if self.maximum > self.minimum:
            return (value - self.minimum) / (self.maximum - self.minimum)
        return value

class Node:
    def __init__(self, is_chance: bool = False):
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: dict[int, Node] = {}
        self.is_chance = is_chance
        self.is_evaluated = False
        self.prior = 1.0

    def expanded(self) -> bool:
        return len(self.children) > 0

    def value(self) -> float:
        return self.value_sum / self.visit_count if self.visit_count > 0 else 0.0


class MCTSAgent:
    def __init__(self, use_afterstate: bool = False, exploration_c: float = 2000.0, seed: int | None = None, max_tree_depth: int = 64, rollout_limit: int = 5):
        self.use_afterstate = use_afterstate
        self.max_tree_depth = max_tree_depth
        self.rollout_limit = rollout_limit
        self.rng = random.Random(seed)
        self.evaluator = FastHeuristic()
        self.pb_c_base, self.pb_c_init = 19652.0, 1.25
        
        self._legal_actions_cache: dict[int, list[int]] = {}
        # 【零对象优化】：全局唯一的试错棋盘共生体
        self.dummy_board = board()

        self.last_root_action_values: dict[int, float] = {}
        self.last_root_action_visits: dict[int, int] = {}

    def get_best_action(self, b: board, num_simulations: int = 100) -> tuple[int, float, float]:
        self._legal_actions_cache.clear()
        self.min_max_stats = MinMaxStats()

        self.last_root_action_values.clear()
        self.last_root_action_visits.clear()
        
        self.root = Node(is_chance=False)
        self.root.is_evaluated = True
        legal_actions = self._get_legal_actions_cached(b.raw)
        if not legal_actions: return 0, 0.0, 0.0
            
        for _ in range(num_simulations):
            self._simulate(self.root, b.raw, 0)

        action_scores, action_visits_list = [], []
        best_a, best_visits, best_score = legal_actions[0], -1, -math.inf

        for a in legal_actions:
            if a not in self.root.children: continue
            child = self.root.children[a]
            visits = child.visit_count
            
            self.dummy_board.raw = b.raw
            reward = self.dummy_board.move(a)
            score = reward + child.value()

            self.last_root_action_values[a] = score
            self.last_root_action_visits[a] = visits
            
            action_scores.append(score)
            action_visits_list.append(visits)

            if visits > best_visits or (visits == best_visits and score > best_score):
                best_visits, best_score, best_a = visits, score, a

        return best_a, self._std(action_scores), self._entropy(action_visits_list, len(legal_actions))

    def _simulate(self, node: Node, current_raw: int, depth: int) -> float:
        if depth >= self.max_tree_depth:
            return self._rollout(current_raw)

        # ==================================================
        # 决策节点 (Decision Node): 玩家做动作
        # ==================================================
        if not node.is_chance:
            legal_actions = self._get_legal_actions_cached(current_raw)
            if not legal_actions: return 0.0

            # --- 扩展逻辑 (防崩溃核心) ---
            if not node.expanded():
                for a in legal_actions:
                    node.children[a] = Node(is_chance=True) # 所有子节点都是机会节点
                    
            if not self.use_afterstate and not node.is_evaluated:
                # State 模式: 评估决策节点
                value = self._rollout(current_raw)
                node.is_evaluated = True
                node.value_sum += value
                node.visit_count += 1
                self.min_max_stats.update(value)
                return value
            elif self.use_afterstate and not node.is_evaluated:
                node.is_evaluated = True

            # --- PUCT 选择 ---
            best_ucb, best_a = -math.inf, legal_actions[0]
            for a in legal_actions:
                child = node.children[a]
                if child.visit_count == 0:
                    ucb = math.inf
                else:
                    self.dummy_board.raw = current_raw
                    reward = self.dummy_board.move(a)
                    val_score = self.min_max_stats.normalize(reward + child.value())
                    pb_c = math.log((node.visit_count + self.pb_c_base + 1) / self.pb_c_base) + self.pb_c_init
                    ucb = val_score + (pb_c * math.sqrt(node.visit_count) / (child.visit_count + 1)) * child.prior
                if ucb > best_ucb:
                    best_ucb, best_a = ucb, a
            
            child_node = node.children[best_a]
            self.dummy_board.raw = current_raw
            reward = self.dummy_board.move(best_a)
            after_raw = self.dummy_board.raw

            future_value = self._simulate(child_node, after_raw, depth)
            total_return = reward + future_value

            node.value_sum += total_return
            node.visit_count += 1
            self.min_max_stats.update(total_return)
            return total_return

        # ==================================================
        # 机会节点 (Chance Node): 环境生成方块
        # ==================================================
        else:
            if self.use_afterstate and not node.is_evaluated:
                # Afterstate 模式: 评估机会节点
                value = self._rollout(current_raw)
                node.is_evaluated = True
                node.value_sum += value
                node.visit_count += 1
                self.min_max_stats.update(value)
                return value
            elif not self.use_afterstate and not node.is_evaluated:
                node.is_evaluated = True

            # 模拟环境随机生成
            self.dummy_board.raw = current_raw
            self._spawn_in_place(self.dummy_board)
            spawned_raw = self.dummy_board.raw

            if spawned_raw not in node.children:
                node.children[spawned_raw] = Node(is_chance=False)
                
            value = self._simulate(node.children[spawned_raw], spawned_raw, depth + 1)
            node.value_sum += value
            node.visit_count += 1
            return value

    def _rollout(self, state_raw: int) -> float:
        total = 0.0
        current_raw = state_raw
        steps = 0

        while steps < self.rollout_limit:
            legal_actions = self._get_legal_actions_cached(current_raw)
            if not legal_actions: return total
                
            if self.rng.random() < 0.1: 
                action = self.rng.choice(legal_actions)
            else:
                best_a, best_score = legal_actions[0], -math.inf
                for a in legal_actions:
                    self.dummy_board.raw = current_raw
                    r = self.dummy_board.move(a)
                    score = r + self.evaluator.evaluate(self.dummy_board.raw, is_afterstate=True)
                    if score > best_score:
                        best_score, best_a = score, a
                action = best_a
                
            self.dummy_board.raw = current_raw
            reward = self.dummy_board.move(action)
            if reward == -1: return total
            total += reward
            self._spawn_in_place(self.dummy_board)
            current_raw = self.dummy_board.raw
            steps += 1

        return total + self.evaluator.evaluate(current_raw) * 0.1

    def _spawn_in_place(self, b: board) -> None:
        spaces = [i for i in range(16) if b.at(i) == 0]
        if not spaces: return
        b.set(self.rng.choice(spaces), 2 if self.rng.random() < 0.1 else 1)

    def _get_legal_actions_cached(self, raw: int) -> list[int]:
        if raw not in self._legal_actions_cache:
            actions = []
            for a in range(4):
                self.dummy_board.raw = raw
                if self.dummy_board.move(a) != -1:
                    actions.append(a)
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
        return -sum((v/total) * math.log(v/total) for v in visits if v > 0)