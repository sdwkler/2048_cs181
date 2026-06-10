# src/phase_1/search/run_search.py
import sys, os, time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.environments.base_env import board
# 【关键修改】：在这里必须把之前在 feature_base.py 里面新增的 diff_pattern 导进来！
from src.ntuple.feature_base import learning, pattern, diff_pattern
from src.ntuple.loader import fast_mmap_load
from src.phase_1.evaluators import HeuristicEvaluator, NTupleEvaluator
from src.phase_1.search.expectimax import ExpectimaxAgent

def _eval_worker(args):
    game_id, use_afterstate, eval_type, search_depth = args
    board.lookup.init()
    
    # 动态挂载评估器
    value_func, tdl = None, None
    if eval_type == "heuristic":
        value_func = HeuristicEvaluator().evaluate
    else:
        tdl = learning()
        
        shapes = [[0,1,2,3,4,5], [4,5,6,7,8,9], [0,1,2,4,5,6], [4,5,6,8,9,10]]
        
        # 1. 注册基础绝对值特征 (必须与 train_ntuple.py 顺序完全一致)
        for p in shapes:
            tdl.add_feature(pattern(p))
            
        # 2. 注册差分泛化特征 (【关键修改】：必须在这里同步挂载，保证 fast_mmap_load 能对齐二进制流)
        for p in shapes:
            tdl.add_feature(diff_pattern(p))
            
        weight_file = "models/2048_afterstate.bin" if eval_type == "ntuple_afterstate" else "models/2048_state.bin"
        
        old_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')
        if os.path.exists(weight_file): 
            fast_mmap_load(tdl, weight_file)
        sys.stdout = old_stdout
        value_func = NTupleEvaluator(tdl).evaluate
        
    agent = ExpectimaxAgent(use_afterstate=use_afterstate, value_func=value_func)
    
    b = board()
    b.init()
    score, steps, step_times, compressions = 0, 0, [], []
    
    while True:
        start_t = time.time()
        # 捕获压缩率探针
        action, comp_ratio = agent.get_best_action(b, max_depth=search_depth)
        step_times.append(time.time() - start_t)
        compressions.append(comp_ratio)
        
        test_b = board(b.raw)
        r = test_b.move(action)
        if r == -1: break 
            
        steps += 1
        score += r
        b = test_b
        b.popup()
        
    if tdl and hasattr(tdl, '_mmap_file'):
        tdl._mmap_handle.close()
        tdl._mmap_file.close()
        
    return game_id, steps, score, (1 << max(b.at(i) for i in range(16))) & ~1, np.mean(step_times), np.mean(compressions)

def run_experiment():
    configs = [
        ("1-B Standard+Heuristic", False, "heuristic"),
        ("1-C Afterstate+Heuristic", True, "heuristic"),
        ("1-D Standard+StateNTuple", False, "ntuple_state"),
        ("1-E Standard+AfterstateNTuple", False, "ntuple_afterstate"),
        ("1-F Afterstate+StateNTuple", True, "ntuple_state"),
        ("1-G Afterstate+AfterstateNTuple", True, "ntuple_afterstate")
    ]
    
    print("\n" + "="*110)
    print(f"| {'消融变体名称 (20局平均)':<35} | {'平均分':<8} | {'最高块':<8} | {'均耗时':<10} | {'均步数':<8} | {'置信表压缩率':<12} |")
    print("-" * 110)
    
    for name, use_aft, eval_type in configs:
        args_list = [(i, use_aft, eval_type, 3) for i in range(20)]
        scores, max_t, avg_times, steps_l, comps = [], [], [], [], []
        
        with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1)) as exc:
            for res in tqdm(as_completed([exc.submit(_eval_worker, a) for a in args_list]), total=20, desc=name[:20], leave=False):
                _, s, sc, mt, at, cp = res.result()
                scores.append(sc); max_t.append(mt); avg_times.append(at); steps_l.append(s); comps.append(cp)
                
        print(f"| {name:<36} | {np.mean(scores):<9.1f} | {np.max(max_t):<9} | {np.mean(avg_times):<9.4f}s | {np.mean(steps_l):<9.1f} | {np.mean(comps):<11.2%} |")
    print("="*110 + "\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_experiment()