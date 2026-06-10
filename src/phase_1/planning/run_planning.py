# src/phase_1/planning/run_planning.py
import sys, os, time
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
from src.environments.base_env import board
from src.phase_1.evaluators import HeuristicEvaluator
from src.phase_1.planning.mcts import MCTSAgent

def worker_init():
    board.lookup.init()

def _eval_worker(args):
    game_id, use_afterstate, sims = args
    # MCTS 在此阶段统一不使用预训练 N-Tuple，彻底探究纯粹方差
    evaluator = HeuristicEvaluator()
    agent = MCTSAgent(use_afterstate=use_afterstate, value_func=evaluator.evaluate)
    
    b = board()
    b.init()
    score, steps = 0, 0
    step_times, variances, entropies = [], [], []
    
    while True:
        start_t = time.time()
        # 获取动作与探针数据
        action, var, ent = agent.get_best_action(b, num_simulations=sims)
        step_times.append(time.time() - start_t)
        variances.append(var)
        entropies.append(ent)
        
        test_b = board(b.raw)
        r = test_b.move(action)
        if r == -1: break 
            
        steps += 1
        score += r
        b = test_b
        b.popup()
        
    return game_id, steps, score, (1<<max(b.at(i) for i in range(16))) & ~1, np.mean(step_times), np.mean(variances), np.mean(entropies)

def run_experiment():
    rollouts = [200, 500, 1000]
    results = []
    
    print("\n" + "="*115)
    print(f"| {'MCTS 算力缩放与方差探究阵列 (20局平均)':<35} | {'平均分':<8} | {'最高块':<8} | {'根估值方差':<11} | {'策略信息熵':<11} |")
    print("-" * 115)
    
    for sims in rollouts:
        for use_aft in [False, True]:
            name = f"{'Afterstate' if use_aft else 'Standard'} MCTS ({sims} Rolls)"
            args_list = [(i, use_aft, sims) for i in range(20)]
            scores, max_t, avg_times, steps_l, vars_l, ents_l = [], [], [], [], [], []
            
            with ProcessPoolExecutor(max_workers=max(1, multiprocessing.cpu_count()-1), initializer=worker_init) as exc:
                for res in tqdm(as_completed([exc.submit(_eval_worker, a) for a in args_list]), total=20, desc=name[:20], leave=False):
                    _, s, sc, mt, at, v, e = res.result()
                    scores.append(sc); max_t.append(mt); avg_times.append(at); steps_l.append(s); vars_l.append(v); ents_l.append(e)
                    
            print(f"| {name:<36} | {np.mean(scores):<9.1f} | {np.max(max_t):<9} | {np.mean(vars_l):<12.1f} | {np.mean(ents_l):<12.4f} |")
    print("="*115 + "\n")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    run_experiment()