import os
import sys
import time
import random
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

# 确保路径可以找到 src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.environments.base_env import board
from src.phase_3.planning.mcts_topology_node import MCTSAgent

# 我们要测试的四种架构配置
MCTS_CONFIGS = [
    {"name": "H-State", "use_afterstate": False, "eval_type": "heuristic", "ntuple_path": None},
    {"name": "H-After", "use_afterstate": True,  "eval_type": "heuristic", "ntuple_path": None},
    {"name": "N-State", "use_afterstate": False, "eval_type": "ntuple",    "ntuple_path": "models/2048_state.bin"},
    {"name": "N-After", "use_afterstate": True,  "eval_type": "ntuple",    "ntuple_path": "models/2048_afterstate.bin"}
]

SIMULATIONS = 1000  # MCTS 的算力
ROLLOUT_LIMIT = 0   # 0 表示直接使用估值截断，你可以改成 5 来测试走子的影响

def build_custom_board(grid_list: list[int]) -> board:
    """辅助函数：将一维列表转换为棋盘"""
    b = board()
    for i, val in enumerate(grid_list):
        if val > 0:
            # 2048 环境里的值是 log2(val)，即 2->1, 4->2, 1024->10
            import math
            b.set(i, int(math.log2(val)))
        else:
            b.set(i, 0)
    return b

def run_micro_probe_test():
    """【微观探针】：在特定盘面下直接解剖树的拓扑结构"""
    print("="*60)
    print("🔬 启动微观树拓扑探测 (Simulations = 1000) 🔬")
    print("="*60)
    
    # 盘面 A: 完美的平滑黄金蛇 (Golden Snake)
    # 只能向左或向下，任何多余的探索都是浪费算力
    board_a = build_custom_board([
        1024, 512, 256, 128,
        16,   32,  64,  0,
        8,    4,   2,   0,
        0,    0,   0,   0
    ])
    
    # 盘面 B: 绝境求生 (Cluttered Survival)
    # 棋盘几乎满了，只能做特定的操作来苟活
    board_b = build_custom_board([
        2,    4,    8,    16,
        32,   64,   128,  256,
        1024, 512,  16,   8,
        2,    4,    8,    0
    ])

    test_cases = [
        ("The Golden Snake (平滑阵型)", board_a),
        ("Cluttered Survival (绝境求生)", board_b)
    ]

    for case_name, b in test_cases:
        print(f"\n\n{'#'*20} 探测盘面: {case_name} {'#'*20}")
        print(b)
        
        for cfg in MCTS_CONFIGS:
            # 初始化 Agent
            agent = MCTSAgent(
                use_afterstate=cfg["use_afterstate"], 
                eval_type=cfg["eval_type"], 
                ntuple_path=cfg["ntuple_path"],
                p4_prob=0.1,  # 无环境偏移
                rollout_limit=ROLLOUT_LIMIT 
            )
            
            # 强行下棋并记录
            action, q_std, entropy, max_depth = agent.get_best_action(board(b.raw), num_simulations=SIMULATIONS)
            tree_data = agent.export_tree_topology()
            
            print(f"\n>>> 代理: [{cfg['name']}]")
            print(f"    最大探索深度: {max_depth} 层")
            print(f"    访问信息熵 (树宽度): {entropy:.4f} (越低越窄，说明算力越集中)")
            
            # 打印第一层的算力分布
            print(f"    根节点算力分配情况:")
            for child in tree_data["children"]:
                bar = "█" * int(child["visits"] / SIMULATIONS * 20)
                print(f"      - {child['action']:<10} | 访问: {child['visits']:<4} | {bar} (Q: {child['q_value']:.0f})")

def macro_game_worker(args):
    """用于宏观对局并行运算的 Worker"""
    cfg, seed = args
    board.lookup.init()
    rng = random.Random(seed)
    
    agent = MCTSAgent(
        use_afterstate=cfg["use_afterstate"], 
        eval_type=cfg["eval_type"], 
        ntuple_path=cfg["ntuple_path"],
        p4_prob=0.1,
        seed=seed,
        rollout_limit=ROLLOUT_LIMIT
    )
    
    b = board()
    spaces = [i for i in range(16) if b.at(i) == 0]
    b.set(rng.choice(spaces), 2 if rng.random() < 0.1 else 1)
    spaces = [i for i in range(16) if b.at(i) == 0]
    b.set(rng.choice(spaces), 2 if rng.random() < 0.1 else 1)

    score, steps = 0, 0
    entropies, depths = [], []
    
    while True:
        action, _, visit_entropy, step_max_depth = agent.get_best_action(b, num_simulations=SIMULATIONS)
        entropies.append(visit_entropy)
        depths.append(step_max_depth)
        
        next_b = board(b.raw)
        reward = next_b.move(action)
        if reward == -1: break
        
        score += reward
        steps += 1
        
        spaces = [i for i in range(16) if next_b.at(i) == 0]
        if spaces: next_b.set(rng.choice(spaces), 2 if rng.random() < 0.1 else 1)
        b = next_b

    return {
        "score": score,
        "max_tile": max([b.at(i) for i in range(16)]),
        "avg_entropy": sum(entropies)/len(entropies) if entropies else 0,
        "avg_depth": sum(depths)/len(depths) if depths else 0
    }
from src.common import add_common_args_3, config_from_args, write_result_bundle
import argparse
def run_macro_benchmark(config):
    """【宏观基准测试】：在无环境偏移 (P(4)=0.1) 下对比全局树属性，并自动保存"""
    print("\n\n" + "="*60)
    print(f"📊 启动宏观基准测试 (无环境偏移 P4=0.1, 共 {config.search_games} 局)")
    print("="*60)
    
    results_summary = {}
    rows = [] # 用于写入 CSV 和 JSON
    
    for cfg in MCTS_CONFIGS:
        print(f"\n🚀 正在测试: {cfg['name']} ...")
        
        # 并行执行对局
        args_list = [(cfg, config.seed + 1000 + i) for i in range(config.search_games)]
        records = []
        with ProcessPoolExecutor(max_workers=config.workers) as executor:
            futures = [executor.submit(macro_game_worker, args) for args in args_list]
            for future in as_completed(futures):
                records.append(future.result())
                
        # 统计平均值
        avg_score = np.mean([r["score"] for r in records])
        avg_entropy = np.mean([r["avg_entropy"] for r in records])
        avg_depth = np.mean([r["avg_depth"] for r in records])
        
        # 构建用于保存的数据行
        row = {
            "experiment": cfg["name"],
            "use_afterstate": cfg["use_afterstate"],
            "eval_type": cfg["eval_type"],
            "average_score": avg_score,
            "tree_entropy": avg_entropy,
            "tree_depth": avg_depth,
            "simulations": SIMULATIONS,
            "rollout_limit": ROLLOUT_LIMIT
        }
        rows.append(row)
        
        results_summary[cfg["name"]] = {
            "Score": avg_score,
            "Tree Entropy (Width)": avg_entropy,
            "Tree Depth": avg_depth
        }
        
        print(f"   [完成] 平均得分: {avg_score:.0f} | 树平均宽度(熵): {avg_entropy:.3f} | 树平均深度: {avg_depth:.2f}")

    print("\n\n🏆 最终宏观树形总结 🏆")
    print(f"{'模型架构':<15} | {'平均得分':<10} | {'平均宽度(熵)':<15} | {'平均最大深度'}")
    print("-" * 65)
    for name, stats in results_summary.items():
        print(f"{name:<15} | {stats['Score']:<10.0f} | {stats['Tree Entropy (Width)']:<15.3f} | {stats['Tree Depth']:.2f} 层")
        
    print("\n💡 结论指标指引: 熵(Entropy)越低，证明 MCTS 没有浪费算力在低价值分支，树越'瘦'; Depth越大，证明搜索得越'深'。")
    
    # 【自动保存逻辑】
    paths = write_result_bundle(config.output_dir, "mcts_topology_analysis", config, rows, {})
    print(f"\n✅ 宏观测试数据已成功保存至:\n - Markdown: {paths['md']}\n - CSV: {paths['csv']}")

if __name__ == "__main__":
    from src.environments.base_env import board  # 确保导入了 board
    
    # 【修复核心】：在主进程执行任何棋盘操作前，先初始化全局高速查找表！
    board.lookup.init()
    parser = argparse.ArgumentParser(description="Run MCTS Topology Analysis")
    add_common_args_3(parser)
    # 开放局数自定义，方便快速测试
    parser.add_argument("--games", type=int, default=50, help="宏观测试对局数")
    args = parser.parse_args()
    
    config = config_from_args(args)
    
    import dataclasses
    # 强制覆盖默认的保存路径和对局数，保证文件存在专门的文件夹里
    config = dataclasses.replace(
        config, 
        search_games=args.games,
        output_dir=os.path.join("models", "phase_3", "results", "topology_tests")
    )
    
    # 1. 先跑微观探针，剖析单步决策 (直接打印到控制台，供人为观察)
    run_micro_probe_test()
    
    # 2. 再跑宏观比赛，拿全局统计学数据并保存到文件
    run_macro_benchmark(config)