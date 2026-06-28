import os
import sys
import random
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
from src.environments.base_env import board, info

# 【核心：引入你在 MCTS 中的 O(1) 极速人工评估器】
from src.evaluators import FastHeuristic 
from src.ntuple.feature_base import learning, pattern, diff_pattern

def train_heuristic_based_risk_network(total_episodes=50000, alpha=0.05, seed=42):
    info(f"\n>>> 🚀 启动 [神经符号蒸馏] Heuristic-based Risk N-Tuple 训练 (共 {total_episodes} 局)")
    info(f">>> 基准 (V_mean): 固定的 FastHeuristic 人工函数 (绝对不更新)")
    info(f">>> 目标 (V_risk): N-Tuple 网络，纯粹拟合人工分数的跌幅 (Drop)")
    
    board.lookup.init()
    random.seed(seed)
    
    # 1. 实例化固定的人工基准函数
    heuristic_evaluator = FastHeuristic()
    
    # 2. 实例化空白的 N-Tuple 风险网络
    tdl_risk = learning()
    shapes = [[0,1,2,3,4,5], [4,5,6,7,8,9], [0,1,2,4,5,6], [4,5,6,8,9,10]]
    for p in shapes: tdl_risk.add_feature(pattern(p))
    for p in shapes: tdl_risk.add_feature(diff_pattern(p))

    risk_path = "models/2048_heuristic_risk.bin"
    os.makedirs("models", exist_ok=True)
    tdl_risk.load(risk_path)

    for n in tqdm(range(1, total_episodes + 1), desc="Training Risk N-Tuple", dynamic_ncols=True, unit="ep"):
        state = board()
        state.init()
        
        # 记录整局轨迹的 Afterstate 和 Next_State，用于事后计算跌幅
        trajectory = [] 

        # ==========================================
        # 阶段 1：数据采集 (Self-Play)
        # ==========================================
        while True:
            # 使用固定的人工函数进行 1-ply 的贪心走子，生成自然对局
            best_action, best_val = -1, -float('inf')
            best_after = None
            
            for a in range(4):
                temp_b = board(state.raw)
                reward = temp_b.move(a)
                if reward != -1:
                    # 贪心基准 = Reward + Heuristic(Afterstate)
                    val = reward + heuristic_evaluator.evaluate(temp_b.raw, is_afterstate=True)
                    if val > best_val:
                        best_val = val
                        best_action = a
                        best_after = board(temp_b.raw)
            
            if best_action == -1:
                break # 死局
                
            afterstate_raw = best_after.raw
            
            # 环境随机发牌
            best_after.popup() 
            next_state_raw = best_after.raw
            
            # 记录转移：(划动后的中间态, 发牌后的新状态)
            trajectory.append((afterstate_raw, next_state_raw))
            state = best_after

        # ==========================================
        # 阶段 2：风险网络监督学习 (Supervised Risk Distillation)
        # ==========================================
        # 注意：Risk 的学习不需要反向的 TD bootstrap，因为每次发牌的跌幅是独立的马尔可夫事件
        for after_raw, next_raw in trajectory:
            # 1. 查询基准函数在发牌前的乐观估计
            base_val_after = heuristic_evaluator.evaluate(after_raw, is_afterstate=True)
            
            # 2. 查询基准函数在发牌后的真实评价
            base_val_next = heuristic_evaluator.evaluate(next_raw, is_afterstate=False)
            
            # 3. 计算真实的价值跌幅 (Drop / Severity)
            # 如果分数变高或持平，drop 为 0。如果遭遇危机分数暴跌，drop 将是一个巨大的正数！
            real_drop = max(0.0, base_val_after - base_val_next)
            
            # 4. 让 N-Tuple 风险网络去拟合这个 Drop
            pred_drop = tdl_risk.estimate(board(after_raw))
            error = real_drop - pred_drop
            
            # 仅更新风险网络
            tdl_risk.update(board(after_raw), alpha * error)

        # 每 10000 局输出一次统计信息 (由于 tdl_risk 里面存的是惩罚分数，所以平均值不代表游戏胜率)
        if n % 10000 == 0:
            tdl_risk.save(risk_path)

    tdl_risk.save(risk_path)
    info(f">>> ✅ 神经符号 Risk N-Tuple 蒸馏完毕！保存至 {risk_path}")

if __name__ == "__main__":
    train_heuristic_based_risk_network(total_episodes=50000)