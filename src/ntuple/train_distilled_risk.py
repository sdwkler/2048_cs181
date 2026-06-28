import os
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.environments.base_env import board, info
from src.ntuple.feature_base import learning, pattern, diff_pattern

def train_distilled_risk(total_episodes=50000, alpha=0.05, seed=42):
    info(f"\n>>> 🚀 启动 [冻结蒸馏] Risk N-Tuple 专属训练流水线 (共 {total_episodes} 局)")
    board.lookup.init()
    random.seed(seed)
    
    # 1. 实例化并加载极品 Mean 网络
    tdl_mean = learning()
    shapes = [[0,1,2,3,4,5], [4,5,6,7,8,9], [0,1,2,4,5,6], [4,5,6,8,9,10]]
    for p in shapes: tdl_mean.add_feature(pattern(p))
    for p in shapes: tdl_mean.add_feature(diff_pattern(p))
    
    mean_path = "models/2048_afterstate.bin"
    if not os.path.exists(mean_path):
        raise FileNotFoundError(f"找不到 {mean_path}，请确保原版 Afterstate 模型存在！")
    tdl_mean.load(mean_path)
    info(f">>> ✅ 成功加载极品基准模型: {mean_path} (权重已冻结)")

    # 2. 实例化空白的 Risk 网络
    tdl_risk = learning()
    for p in shapes: tdl_risk.add_feature(pattern(p))
    for p in shapes: tdl_risk.add_feature(diff_pattern(p))
    
    risk_path = "models/2048_distilled_risk.bin"
    os.makedirs("models", exist_ok=True)
    tdl_risk.load(risk_path) # 尝试加载已有进度

    for n in tqdm(range(1, total_episodes + 1), desc="Distilling Risk", dynamic_ncols=True, unit="ep"):
        path = []
        state = board()
        state.init()

        # ==========================================
        # 阶段 1：使用冻结的 Mean 网络进行采样
        # ==========================================
        while True:
            best = tdl_mean.select_best_move(state)
            path.append(best)
            if best.is_valid():
                state = board(best.afterstate())
                state.popup() # 正常随机发牌
            else:
                break
        
        # ==========================================
        # 阶段 2：计算跌幅 (Drop) 并单向蒸馏给 Risk
        # ==========================================
        target_mean = 0
        path.pop() # 弹出死局最后一步
        
        while path:
            mv = path.pop()
            s_after = mv.afterstate() 
            
            # 【核心】：向冻结的 Mean 网络查分
            curr_mean = tdl_mean.estimate(s_after)
            
            # 如果未来的 target_mean 比现在的 curr_mean 低，说明刚才的发牌是灾难
            drop = max(0.0, curr_mean - target_mean)
            
            # 【只更新 Risk 网络】
            curr_risk = tdl_risk.estimate(s_after)
            error_risk = drop - curr_risk
            tdl_risk.update(s_after, alpha * error_risk)
            
            # 传递 Target：当前步的 reward + 冻结的当前评估值
            target_mean = mv.reward() + curr_mean

        # 每 10000 局存一次档
        if n % 10000 == 0:
            tdl_risk.save(risk_path)

    tdl_risk.save(risk_path)
    info(f">>> ✅ 风险网络蒸馏完毕！保存至 {risk_path}")

if __name__ == "__main__":
    train_distilled_risk(total_episodes=50000)