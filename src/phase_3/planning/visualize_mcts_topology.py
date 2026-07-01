# 文件路径: src/phase_3/planning/visualize_mcts_topology.py
import os
import sys
import math
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import Patch

# 确保路径可以导入你的核心模块
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))

from src.environments.base_env import board
from src.phase_3.planning.mcts_topology_node import MCTSAgent

# ==========================================
# 1. 实验配置与初始化
# ==========================================
SIMULATIONS = 1000
SEED = 42

CONFIGS = [
    ("Heuristic + State", False, "heuristic", None),
    ("Heuristic + Afterstate", True, "heuristic", None),
    ("N-Tuple + State", False, "ntuple", "models/2048_state.bin"),
    ("N-Tuple + Afterstate", True, "ntuple", "models/2048_afterstate.bin")
]

# 构造一个高压差（需要深入探测）的复杂盘面
# 比如左上角有大数，但有些空隙需要填补
# 格式化为一个特定的 raw int (你可以根据你的 board 实现替换为一个真实的复杂 raw)
def get_complex_board():
    b = board()
    # 构造一个“诱惑与陷阱并存”的高压盘面
    # 1024 被夹在中间，512 和 256 在上面，底下有空位
    # Heuristic 极容易为了合并或者保持平滑而走错，导致 1024 沉底
    # N-tuple 有望通过深层前瞻，找到重整阵型的唯一路径
    
    b.set(0, 3)  # 8
    b.set(1, 9)  # 512
    b.set(2, 10) # 1024
    b.set(3, 2)  # 4

    b.set(4, 7)  # 128
    b.set(5, 8)  # 256
    b.set(6, 6)  # 64
    b.set(7, 4)  # 16

    b.set(8, 2)  # 4
    b.set(9, 3)  # 8
    b.set(10, 5) # 32
    b.set(11, 2) # 4

    # 最下面一行全空，留给环境随机生成
    b.set(12, 0)
    b.set(13, 0)
    b.set(14, 0)
    b.set(15, 0)
    return b

# ==========================================
# 2. 树结构解析与布局提取
# ==========================================
def extract_tree_graph(root_node, max_depth=10):
    """将 MCTS 树转化为 networkx 图，方便绘制"""
    G = nx.DiGraph()
    
    def traverse(node, current_id, depth):
        if depth > max_depth or node.visit_count == 0:
            return
        
        # 节点属性：是否为 chance 节点，访问次数
        G.add_node(current_id, is_chance=node.is_chance, visits=node.visit_count, depth=depth)
        
        for action_or_raw, child in node.children.items():
            if child.visit_count > 0:
                child_id = f"{current_id}_{action_or_raw}"
                G.add_edge(current_id, child_id)
                traverse(child, child_id, depth + 1)
                
    traverse(root_node, "root", 0)
    return G

def hierarchy_pos(G, root=None, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
    """自定义的树状图计算布局函数"""
    if root is None:
        root = "root"
    
    pos = {root: (xcenter, vert_loc)}
    children = list(G.neighbors(root))
    if not children:
        return pos
    
    dx = width / len(children) 
    nextx = xcenter - width/2 - dx/2
    for child in children:
        nextx += dx
        pos = {**pos, **hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, 
                                      vert_loc=vert_loc-vert_gap, xcenter=nextx)}
    return pos

# ==========================================
# 3. 核心动画与渲染逻辑
# ==========================================
def main():
    board.lookup.init()
    base_board = get_complex_board()
    
    print(f"正在初始化 4 个 MCTS 代理... (确保 N-Tuple 模型路径正确)")
    agents = []
    for name, use_after, eval_type, path in CONFIGS:
        agent = MCTSAgent(use_afterstate=use_after, eval_type=eval_type, ntuple_path=path, seed=SEED, rollout_limit=5)
        
        # 手动执行原版 get_best_action 中的初始化部分
        agent._legal_actions_cache.clear()
        agent.current_max_depth = 0
        from src.phase_3.planning.mcts_topology_node import Node
        agent.root = Node(is_chance=False, last_action=-1)
        agent.root.is_evaluated = True
        
        agents.append({
            "name": name,
            "agent": agent,
            "b_raw": base_board.raw,
            "sim_count": 0
        })

    # 创建 2x2 画布
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), facecolor='#1e1e1e')
    fig.suptitle("MCTS Search Topology Evolution (200 Simulations)", color='white', fontsize=20)
    axes = axes.flatten()
    
    def update(frame):
        # 每一帧，让每个 agent 跑 1 次 simulate，推进一步探测
        for ax, agent_data in zip(axes, agents):
            ax.clear()
            ax.set_facecolor('#1e1e1e')
            ax.axis('off')
            
            agent = agent_data["agent"]
            
            # 单步模拟 (Hook)
            if agent_data["sim_count"] < SIMULATIONS:
                agent._simulate(agent.root, agent_data["b_raw"], depth=0)
                agent_data["sim_count"] += 1
                
            # 提取图
            G = extract_tree_graph(agent.root, max_depth=12)
            if len(G.nodes) == 0:
                continue
                
            # 计算布局
            pos = hierarchy_pos(G, "root")
            
            # 区分节点类型与大小 (访问量越大，节点越大)
            node_colors = []
            node_sizes = []
            for n, d in G.nodes(data=True):
                visits = d.get('visits', 1)
                size = max(20, min(300, visits * 5))
                node_sizes.append(size)
                
                if n == "root":
                    node_colors.append('#ff5555') # 根节点红色
                elif d.get('is_chance'):
                    node_colors.append('#55aaff') # 环境节点蓝色
                else:
                    node_colors.append('#55ff55') # 决策节点绿色
            
            # 绘制边和节点
            nx.draw_networkx_edges(G, pos, ax=ax, edge_color='#666666', alpha=0.5, arrows=False)
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors, node_size=node_sizes, alpha=0.8)
            
            ax.set_title(f"{agent_data['name']}\nSims: {agent_data['sim_count']} | Nodes: {len(G.nodes)} | Max Depth: {agent.current_max_depth}", 
                         color='white', fontsize=12)

        return axes

    print(f"开始渲染动态图，共 {SIMULATIONS} 帧，可能需要几分钟...")
    ani = animation.FuncAnimation(fig, update, frames=SIMULATIONS, interval=100, blit=False, repeat=True)
    
    # 保存结果
    save_path = "mcts_topology_evolution.mp4" # 如果没装 ffmpeg，可以改成 .gif 并用 writer='pillow'
    try:
        ani.save(save_path, writer='ffmpeg', fps=15, dpi=120)
        print(f"\n✅ 渲染完成！动画已保存至: {os.path.abspath(save_path)}")
    except Exception as e:
        print(f"\n⚠️ ffmpeg 保存失败 (可能是没安装)，回退使用 GIF 保存: {e}")
        ani.save("mcts_topology_evolution.gif", writer='pillow', fps=15, dpi=120)
        print("✅ 已保存为 mcts_topology_evolution.gif")
        
if __name__ == "__main__":
    main()