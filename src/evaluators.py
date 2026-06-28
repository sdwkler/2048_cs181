# 文件路径: src/phase_1/evaluators.py
from src.environments.base_env import board

class HeuristicEvaluator:
    """人工经验评估器 (带进程级永续高速缓存)"""
    
    # 【性能核武】：类变量！
    # 同一个进程内无论实例化多少个 HeuristicEvaluator，共享这同一个字典！
    # 彻底实现跨对局、跨 MCTS Agent 的记忆复用！
    _process_shared_cache = {}

    def __init__(self, w_empty=270.0, w_mono=47.0, w_smooth=0.1, w_corner=500.0):
        self.w_empty = w_empty
        self.w_mono = w_mono
        self.w_smooth = w_smooth
        self.w_corner = w_corner

    def evaluate(self, b: board, is_afterstate=False):
        cache_key = (b.raw, is_afterstate)
        
        # O(1) 高速缓存拦截
        if cache_key in HeuristicEvaluator._process_shared_cache:
            return HeuristicEvaluator._process_shared_cache[cache_key]

        grid = [b.at(i) for i in range(16)]
        empty_count = grid.count(0)
        
        if is_afterstate and empty_count > 0:
            empty_count -= 1
            
        max_val = max(grid)
        corner_max = 1 if max_val in (grid[0], grid[3], grid[12], grid[15]) else 0
        
        smooth = 0
        for i in range(4):
            for j in range(3):
                smooth -= abs(grid[i*4 + j] - grid[i*4 + j + 1])
                smooth -= abs(grid[j*4 + i] - grid[(j+1)*4 + i])
                
        mono_up = sum(1 for i in range(3) for j in range(4) if grid[i*4+j] >= grid[(i+1)*4+j])
        mono_down = sum(1 for i in range(3) for j in range(4) if grid[i*4+j] <= grid[(i+1)*4+j])
        mono_left = sum(1 for i in range(4) for j in range(3) if grid[i*4+j] >= grid[i*4+j+1])
        mono_right = sum(1 for i in range(4) for j in range(3) if grid[i*4+j] <= grid[i*4+j+1])
        mono = max(mono_up, mono_down) + max(mono_left, mono_right)

        score = (self.w_empty * empty_count) + (self.w_mono * mono) + \
               (self.w_smooth * smooth) + (self.w_corner * corner_max)
               
        # 写入类级别的共享缓存
        HeuristicEvaluator._process_shared_cache[cache_key] = score
        
        # OOM 保护：上限可以开到 50 万，单个进程占用约几十MB
        if len(HeuristicEvaluator._process_shared_cache) > 500000:
            HeuristicEvaluator._process_shared_cache.clear()
            
        return score
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
import math
from src.environments.base_env import board

class EvolutionaryEvaluator:
    """
    【进化算法终极评估器 (完美公平版)】
    修复了原始 LLM 代码对 Afterstate 的“空洞歧视”。
    通过赋予空地合理的接盘平滑度，确保在 P(4) 极端的环境中，
    精准引爆 State 架构的“期望悲观瘫痪”，而让 Afterstate 保持拓扑纯洁性。
    """
    _process_shared_cache = {}

    def __init__(self):
        self.w_empty = 0.30
        self.w_highest = 0.20
        self.w_corner = 0.15
        self.w_snake = 0.15
        self.w_smooth = 0.10
        self.w_merge = 0.10

        self.paths = [
            [0, 1, 2, 3, 7, 6, 5, 4, 8, 9, 10, 11, 15, 14, 13, 12],
            [0, 4, 8, 12, 13, 9, 5, 1, 2, 6, 10, 14, 15, 11, 7, 3],
            [3, 2, 1, 0, 4, 5, 6, 7, 11, 10, 9, 8, 12, 13, 14, 15],
            [3, 7, 11, 15, 14, 10, 6, 2, 1, 5, 9, 13, 12, 8, 4, 0],
            [12, 13, 14, 15, 11, 10, 9, 8, 4, 5, 6, 7, 3, 2, 1, 0],
            [12, 8, 4, 0, 1, 5, 9, 13, 14, 10, 6, 2, 3, 7, 11, 15],
            [15, 14, 13, 12, 8, 9, 10, 11, 7, 6, 5, 4, 3, 2, 1, 0],
            [15, 11, 7, 3, 2, 6, 10, 14, 13, 9, 5, 1, 0, 4, 8, 12]
        ]

    def evaluate(self, b: board, is_afterstate=False):
        cache_key = (b.raw, is_afterstate)
        if cache_key in EvolutionaryEvaluator._process_shared_cache:
            return EvolutionaryEvaluator._process_shared_cache[cache_key]

        grid = [b.at(i) for i in range(16)]
        vals = [(1 << p) if p > 0 else 0 for p in grid]
        
        # 1. 空地补偿
        empty_count = grid.count(0)
        if is_afterstate and empty_count > 0:
            empty_count -= 1 
        empty_ratio = empty_count / 16.0

        max_power = max(grid) if grid else 0
        highest_ratio = max_power / 16.0 

        # 2. 角落吸附距离
        max_indices = [i for i, x in enumerate(grid) if x == max_power]
        min_dist = float('inf')
        for idx in max_indices:
            row, col = divmod(idx, 4)
            dist = min(row, 3 - row) + min(col, 3 - col)
            if dist < min_dist: min_dist = dist
        corner_proximity = 1.0 - (min_dist / 6.0) if min_dist != float('inf') else 0.0

        smoothness = 0.0
        merge_count = 0
        for r in range(4):
            for c in range(4):
                idx = r * 4 + c
                v1, p1 = vals[idx], grid[idx]
                
                if v1 > 0:
                    # 向右检查
                    if c < 3:
                        p2 = grid[idx + 1]
                        # 【神级公平修正 1】：空地 (p2=0) 在平滑度中被视为最小的方块 2 (p=1)，不进行毁灭性惩罚
                        p2_adj = p2 if p2 > 0 else 1
                        smoothness += 1.0 / (1.0 + abs(p1 - p2_adj))
                        if p1 == p2_adj: merge_count += 1
                                
                    # 向下检查
                    if r < 3:
                        p3 = grid[idx + 4]
                        p3_adj = p3 if p3 > 0 else 1
                        smoothness += 1.0 / (1.0 + abs(p1 - p3_adj))
                        if p1 == p3_adj: merge_count += 1

        smoothness_ratio = smoothness / 24.0
        merge_ratio = merge_count / 24.0

        # 3. 蛇形链条 (最强杀器)
        best_snake_score = 0
        for path in self.paths:
            snake_score = 0
            for i in range(15):
                pos1, pos2 = path[i], path[i+1]
                p1, p2 = grid[pos1], grid[pos2]
                
                # 【神级公平修正 2】：只要前一个是有效数字，且【后一个是空地】或者【前一个 >= 后一个】，蛇形不断！
                if p1 > 0:
                    if p2 == 0 or p1 >= p2:
                        snake_score += 1
                        
            if snake_score > best_snake_score:
                best_snake_score = snake_score
        
        snake_ratio = best_snake_score / 15.0

        score = (self.w_empty * empty_ratio) + \
                (self.w_highest * highest_ratio) + \
                (self.w_corner * corner_proximity) + \
                (self.w_smooth * smoothness_ratio) + \
                (self.w_merge * merge_ratio) + \
                (self.w_snake * snake_ratio)

        EvolutionaryEvaluator._process_shared_cache[cache_key] = score
        if len(EvolutionaryEvaluator._process_shared_cache) > 500000:
            EvolutionaryEvaluator._process_shared_cache.clear()
            
        return score
    

class NTupleEvaluator:
    """封装好的 N-Tuple 评估器，统一接口"""
    def __init__(self, tdl_model):
        self.tdl = tdl_model
        
    def evaluate(self, b: board, is_afterstate: bool = False) -> float:
        return self.tdl.estimate(b)