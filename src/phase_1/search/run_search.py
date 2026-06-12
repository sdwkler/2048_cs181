# src/phase_1/search/run_search.py
import sys, os, time, datetime
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.environments.base_env import board
from src.ntuple.feature_base import learning, pattern, diff_pattern, feature
from src.ntuple.loader import fast_mmap_load
from src.phase_1.evaluators import HeuristicEvaluator, NTupleEvaluator
from src.phase_1.search.expectimax import ExpectimaxAgent

_process_model_cache = {}

def _eval_worker(args):
    game_id, use_afterstate, eval_type, search_depth = args
    board.lookup.init()
    
    value_func, tdl = None, None
    if eval_type == "heuristic":
        value_func = HeuristicEvaluator().evaluate
    else:
        global _process_model_cache
        if eval_type not in _process_model_cache:
            
            # 【内存保护锁 1】：拦截初始化的无用 0 列表分配，防止瞬间把 RAM 撑爆
            original_alloc = feature.alloc
            feature.alloc = staticmethod(lambda num: range(num))
            
            tdl = learning()
            shapes = [[0,1,2,3,4,5], [4,5,6,7,8,9], [0,1,2,4,5,6], [4,5,6,8,9,10]]
            
            for p in shapes:
                tdl.add_feature(pattern(p))
            for p in shapes:
                tdl.add_feature(diff_pattern(p))
                
            # 恢复分配器
            feature.alloc = original_alloc
            
            weight_file = "models/2048_afterstate.bin" if eval_type == "ntuple_afterstate" else "models/2048_state.bin"
            
            old_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')
            if os.path.exists(weight_file): 
                fast_mmap_load(tdl, weight_file)
            sys.stdout = old_stdout
            
            _process_model_cache[eval_type] = tdl
            
        tdl = _process_model_cache[eval_type]
        value_func = NTupleEvaluator(tdl).evaluate
        
    agent = ExpectimaxAgent(use_afterstate=use_afterstate, value_func=value_func)
    
    b = board()
    b.init()
    score, steps, step_times, compressions, b_effs = 0, 0, [], [], []
    
    while True:
        start_t = time.time()
        action, comp_ratio, b_eff = agent.get_best_action(b, max_depth=search_depth)
        step_times.append(time.time() - start_t)
        compressions.append(comp_ratio)
        b_effs.append(b_eff)
        
        test_b = board(b.raw)
        r = test_b.move(action)
        if r == -1: break 
            
        steps += 1
        score += r
        b = test_b
        b.popup()
        
    return game_id, steps, score, (1 << max(b.at(i) for i in range(16))) & ~1, np.mean(step_times), np.mean(compressions), np.mean(b_effs)

def run_experiment():
    configs = [
        ("1-A Greedy Baseline", False, "heuristic", 1),
        ("1-B Standard+Heuristic", False, "heuristic", 2),
        ("1-C Afterstate+Heuristic", True, "heuristic", 2),
        ("1-D Standard+StateNTuple", False, "ntuple_state", 2),
        ("1-E Standard+AfterstateNTuple", False, "ntuple_afterstate", 2),
        ("1-F Afterstate+StateNTuple", True, "ntuple_state", 2),
        ("1-G Afterstate+AfterstateNTuple", True, "ntuple_afterstate", 2)
    ]
    
    total_games = 100
    
    save_dir = os.path.join("models", "eval_results")
    os.makedirs(save_dir, exist_ok=True)
    
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"experiment_results_{timestamp}.md")
    
    header = f"| {'消融变体名称 ('+str(total_games)+'局平均)':<35} | {'平均分':<8} | {'1024率':<7} | {'2048率':<7} | {'4096率':<7} | {'均耗时':<9} | {'均步数':<7} | {'有效分支因子':<12} | {'置信表压缩率':<12} |"
    separator = "-" * 145
    
    print(f"\n>>> 实验结果将同步保存至: {save_path}")
    print("\n" + "="*145)
    print(header)
    print(separator)
    
    with open(save_path, "w", encoding="utf-8") as f:
        f.write(f"# 阶段一消融实验结果 ({timestamp})\n\n")
        f.write(header + "\n")
        f.write(separator + "\n")
        
        # 【内存保护锁 2】：不管你的 CPU 有多少核，强行限制最多同时启动 4 个游戏，将总内存占用死死压在安全线以下！
        safe_workers = max(1, multiprocessing.cpu_count() - 1)
        
        for name, use_aft, eval_type, depth in configs:
            args_list = [(i, use_aft, eval_type, depth) for i in range(total_games)]
            scores, max_t, avg_times, steps_l, comps, beffs = [], [], [], [], [], []
            
            with ProcessPoolExecutor(max_workers=max(1, safe_workers)) as exc:
                for res in tqdm(as_completed([exc.submit(_eval_worker, a) for a in args_list]), total=total_games, desc=name[:20], leave=False):
                    _, s, sc, mt, at, cp, bf = res.result()
                    scores.append(sc); max_t.append(mt); avg_times.append(at)
                    steps_l.append(s); comps.append(cp); beffs.append(bf)
                    
            rate_1024 = sum(1 for t in max_t if t >= 1024) / total_games
            rate_2048 = sum(1 for t in max_t if t >= 2048) / total_games
            rate_4096 = sum(1 for t in max_t if t >= 4096) / total_games
                    
            row_str = f"| {name:<36} | {np.mean(scores):<9.1f} | {rate_1024:<8.1%} | {rate_2048:<8.1%} | {rate_4096:<8.1%} | {np.mean(avg_times):<8.4f}s | {np.mean(steps_l):<8.1f} | {np.mean(beffs):<12.2f} | {np.mean(comps):<11.2%} |"
            
            print(row_str)
            f.write(row_str + "\n")
            f.flush() 
            
        print("="*145 + "\n")

if __name__ == "__main__":
    multiprocessing.freeze_support() 
    run_experiment()