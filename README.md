# 2048 随机序贯决策：Afterstate 解耦机制与多范式评估框架

本项目是一个基于 2048 游戏沙盒的强化学习与启发式搜索科研项目。旨在探究在“玩家动作确定、环境响应随机”的马尔可夫决策过程（MDP）中，**Afterstate（后状态）** 解耦机制如何在底层的代数估值、拓扑坍缩与采样方差上提升 AI 算法的泛化与生存能力。

## 📁 核心目录与文件结构说明

本项目采用高度解耦的模块化设计，目前已实现第一阶段（基础多范式机制剥离）的完整代码，具体目录功能如下：

```text
digital_defence_2048/
│
├── requirements.txt            # 项目 Python 依赖声明
├── README.md                   # 项目结构与运行说明文档
│
├── models/                     # 权重存储目录
│   ├── 2048_afterstate.bin     # 预训练好的解耦特征（V(s')）权重文件
│   └── 2048_state.bin          # 预训练好的普通特征（Q(s,a)）权重文件
│
└── src/                        # 核心源代码目录
    │
    ├── environments/           # 【环境与物理引擎层】
    │   └── base_env.py         # 极速 64 位 Bitboard 状态机引擎，处理 2048 核心滑动与生成逻辑
    │
    ├── ntuple/                 # 【表征学习与底层工具层】
    │   ├── feature_base.py     # N-Tuple 评估器基类、特征提取与 TD(0) 学习引擎
    │   ├── loader.py           # 零拷贝（Zero-Copy）内存映射加载器，极速并发读取网络权重
    │   └── train_ntuple.py     # N-Tuple 网络自动化训练流水线（支持 State 与 Afterstate 独立训练）
    │
    └── phase_1/                # 【第一阶段：基础机制解耦探究群】
        ├── evaluators.py       # 统一评估器中心（包含人类启发式 Heuristic 与 N-Tuple 封装）
        │
        ├── search/             # 1. 离线搜索范式（研究拓扑压缩与过估计偏差）
        │   ├── expectimax.py   # 纯净的 Expectimax 核心树展开算法，搭载置换表与压缩率探针
        │   └── run_search.py   # 启动器：执行 2x3 搜索消融实验矩阵并输出对齐面板
        │
        └── planning/           # 2. 在线规划范式（研究采样方差与算力等价性）
            ├── mcts.py         # 纯净的 MCTS 算法核心，搭载根节点动作估值方差与策略信息熵探针
            └── run_planning.py # 启动器：执行 MCTS 算力缩放（Rollouts）博弈实验并输出分析面板
```

```text
conda create -n pypy_env pypy -c conda-forge -y
conda activate pypy_env
pip install tqdm numpy
set PYTHONUTF8=1
chcp 65001
set PYTHONIOENCODING=utf-8