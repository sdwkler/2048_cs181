import os
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.environments.base_env import board, info
from src.ntuple.feature_base import learning, pattern, diff_pattern

def train_network(target_type="afterstate", total_episodes=100000, alpha=0.1, seed=0):
    info(f"\n>>> 🚀 启动 {target_type.upper()} 差分联合 N-Tuple 训练流水线 (共 {total_episodes} 局)")
    board.lookup.init()
    random.seed(seed)
    
    tdl = learning()
    
    # 基础与差分特征注册
    shapes = [[0,1,2,3,4,5], [4,5,6,7,8,9], [0,1,2,4,5,6], [4,5,6,8,9,10]]
    for p in shapes: tdl.add_feature(pattern(p))
    for p in shapes: tdl.add_feature(diff_pattern(p))

    save_path = f"models/2048_{target_type}.bin"
    os.makedirs("models", exist_ok=True)
    tdl.load(save_path)

    for n in tqdm(range(1, total_episodes + 1), desc=f"Training {target_type.upper()}", dynamic_ncols=True, unit="ep"):
        path = []
        state = board()
        score = 0
        state.init()

        # 对局采样
        while True:
            best = tdl.select_best_move(state)
            path.append(best)
            if best.is_valid():
                score += best.reward()
                state = board(best.afterstate())
                state.popup()
            else:
                break
        
        # 回溯 TD-Learning
        target = 0
        path.pop() 
        
        while path:
            mv = path.pop()
            s_target = mv.afterstate() if target_type == "afterstate" else mv.state()
            error_val = target - tdl.estimate(s_target)
            target = mv.reward() + tdl.update(s_target, alpha * error_val)

        tdl.make_statistic(n, state, score, unit=10000)

    tdl.save(save_path)
    info(f">>> ✅ {target_type.upper()} 模型已保存至 {save_path}")

if __name__ == "__main__":
    train_network(target_type="afterstate", total_episodes=10000)
    train_network(target_type="state", total_episodes=10000)