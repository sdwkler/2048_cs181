# 文件路径: src/phase_1/evaluators.py
from src.environments.base_env import board

class HeuristicEvaluator:
    """人工经验评估器"""
    def __init__(self, w_empty=270.0, w_mono=47.0, w_smooth=0.1, w_corner=500.0):
        self.w_empty, self.w_mono, self.w_smooth, self.w_corner = w_empty, w_mono, w_smooth, w_corner

    def evaluate(self, b: board):
        # 核心逻辑完全没变
        grid = [b.at(i) for i in range(16)]
        empty_count = grid.count(0)
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

        return (self.w_empty * empty_count) + (self.w_mono * mono) + \
               (self.w_smooth * smooth) + (self.w_corner * corner_max)
    
    
class NTupleEvaluator:
    """封装好的 N-Tuple 评估器，统一接口"""
    def __init__(self, tdl_model):
        self.tdl = tdl_model
        
    def evaluate(self, b: board) -> float:
        return self.tdl.estimate(b)