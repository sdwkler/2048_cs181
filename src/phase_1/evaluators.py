# 文件路径: src/phase_1/evaluators.py
from src.environments.base_env import board
# src/phase_1/evaluators.py
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
    

# class HeuristicEvaluator:
#     """人工经验评估器"""
#     def __init__(self, w_empty=270.0, w_mono=47.0, w_smooth=0.1, w_corner=500.0):
#         self.w_empty, self.w_mono, self.w_smooth, self.w_corner = w_empty, w_mono, w_smooth, w_corner

#     def evaluate(self, b: board):
#         # 核心逻辑完全没变
#         grid = [b.at(i) for i in range(16)]
#         empty_count = grid.count(0)
#         max_val = max(grid)
#         corner_max = 1 if max_val in (grid[0], grid[3], grid[12], grid[15]) else 0
        
#         smooth = 0
#         for i in range(4):
#             for j in range(3):
#                 smooth -= abs(grid[i*4 + j] - grid[i*4 + j + 1])
#                 smooth -= abs(grid[j*4 + i] - grid[(j+1)*4 + i])
                
#         mono_up = sum(1 for i in range(3) for j in range(4) if grid[i*4+j] >= grid[(i+1)*4+j])
#         mono_down = sum(1 for i in range(3) for j in range(4) if grid[i*4+j] <= grid[(i+1)*4+j])
#         mono_left = sum(1 for i in range(4) for j in range(3) if grid[i*4+j] >= grid[i*4+j+1])
#         mono_right = sum(1 for i in range(4) for j in range(3) if grid[i*4+j] <= grid[i*4+j+1])
#         mono = max(mono_up, mono_down) + max(mono_left, mono_right)

#         return (self.w_empty * empty_count) + (self.w_mono * mono) + \
#                (self.w_smooth * smooth) + (self.w_corner * corner_max)
    
    
class NTupleEvaluator:
    """封装好的 N-Tuple 评估器，统一接口"""
    def __init__(self, tdl_model):
        self.tdl = tdl_model
        
    def evaluate(self, b: board) -> float:
        return self.tdl.estimate(b)