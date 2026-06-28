import os
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.environments.base_env import board, info
from src.ntuple.feature_base import learning, pattern, diff_pattern

def train_dual_network(total_episodes=100000, alpha=0.1, seed=0):
    info(f"\n>>> 🚀 启动 [双轨制 Dual-Track] N-Tuple 联合训练流水线 (共 {total_episodes} 局)")
    info(f">>> 目标 1: V_mean (传统期望价值)")
    info(f">>> 目标 2: V_risk (RSZ 价值跌幅/灾难感知)")
    
    board.lookup.init()
    random.seed(seed)
    
    # 实例化两个完全独立的物理权重表
    tdl_mean = learning()
    tdl_risk = learning()
    
    # 基础与差分特征注册 (给两个模型都装上相同的感官)
    shapes = [[0,1,2,3,4,5], [4,5,6,7,8,9], [0,1,2,4,5,6], [4,5,6,8,9,10]]
    for p in shapes: 
        tdl_mean.add_feature(pattern(p))
        tdl_risk.add_feature(pattern(p))
    for p in shapes: 
        tdl_mean.add_feature(diff_pattern(p))
        tdl_risk.add_feature(diff_pattern(p))

    os.makedirs("models", exist_ok=True)
    mean_path = "models/2048_dual_mean.bin"
    risk_path = "models/2048_dual_risk.bin"
    tdl_mean.load(mean_path)
    tdl_risk.load(risk_path)

    for n in tqdm(range(1, total_episodes + 1), desc="Training Dual-Track", dynamic_ncols=True, unit="ep"):
        path = []
        state = board()
        score = 0
        state.init()

        # ==========================================
        # 1. 采样阶段 (Self-Play)
        # ==========================================
        while True:
            # 【关键】：在训练期，我们依然只使用 V_mean 来探索，
            # 这样才能保证我们以原生的随机动力学（不改变环境分布）去踩坑，从而学到真实的灾难。
            best = tdl_mean.select_best_move(state)
            path.append(best)
            if best.is_valid():
                score += best.reward()
                state = board(best.afterstate())
                state.popup() # 环境纯随机发牌
            else:
                break
        
        # ==========================================
        # 2. 联合回溯阶段 (Joint TD-Learning Backup)
        # ==========================================
        target_mean = 0
        path.pop() # 弹出死局最后一步
        
        while path:
            mv = path.pop()
            s_after = mv.afterstate() 
            
            # --- 评估当前的期望与风险 ---
            curr_mean = tdl_mean.estimate(s_after)
            curr_risk = tdl_risk.estimate(s_after)
            
            # --- 目标 1：更新 Mean 网络 ---
            error_mean = target_mean - curr_mean
            
            # --- 目标 2：计算 Drop 并更新 Risk 网络 (论文核心) ---
            # 如果真实的未来 target_mean 远远低于当前预估的 curr_mean，说明刚才发牌发生了灾难！
            # 我们提取这个跌幅，作为 risk 网络的监督目标。
            drop = max(0.0, curr_mean - target_mean)
            error_risk = drop - curr_risk
            
            # 执行更新
            # 更新 mean 并把真实的 reward 累加到 target 给上一层用
            target_mean = mv.reward() + tdl_mean.update(s_after, alpha * error_mean)
            
            # 风险网络的更新是独立的，它只拟合“跌幅”
            tdl_risk.update(s_after, alpha * error_risk)

        # 统计输出 (只打印 mean 网络的得分作为进度参考)
        tdl_mean.make_statistic(n, state, score, unit=10000)

    # 保存两个模型
    tdl_mean.save(mean_path)
    tdl_risk.save(risk_path)
    info(f">>> ✅ 双轨模型训练完毕！")
    info(f">>> Mean 保存至 {mean_path}")
    info(f">>> Risk 保存至 {risk_path}")

if __name__ == "__main__":
    train_dual_network(total_episodes=50000)