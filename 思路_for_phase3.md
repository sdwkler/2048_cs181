### 🌟 2048 Phase 3 (后续改进思路) 代码实现进度与映射总表

| 算法框架大类 | Phase 3 核心研究模块 | 状态建模 (State/After) | 设计思路与痛点解决 | 代码实现位置与核心变量/函数 | 当前完成状态 |
| :--- | :--- | :--- | :--- | :--- | :---: |
| **Q/V 强化学习<br>(无模型时序差分)** | **1. TDA-Full / TDA-2ply<br>(全宽期望目标)** | Afterstate | **痛点：** 单步采样导致 TD 目标方差极大。<br>**方案：** 对于TDA-Full遍历所有空格计算严格数学期望，消除环境白噪声；对于TDA-2ply，往深度探索，同样求取期望。 | **文件：** `run_qlearning.py`<br>**核心函数：** `_expected_best_action_value()`<br>**目标模式：** `tda_full`, `tda_2ply` | ⚠️ **代码已完成<br>正在运行** |
| | **2. Dual-MV<br>(双向下行方差约束)** | Afterstate | **痛点：** 原 MV 无法区分收益“暴涨”和“暴跌”。<br>**方案：** 设立双头网络，分离 `delta_up` 和 `delta_down`，仅惩罚下行风险，鼓励正向惊喜。 | **文件：** `run_qlearning.py`<br>**核心属性：** `m_up_head`, `m_down_head`<br>**逻辑：** 针对 `delta_up`, `delta_down` 分别计算 TD 误差并更新 | ⚠️ **代码已完成<br>正在运行** |
| **MCTS 树搜索<br>(蒙特卡洛树搜索)** | **1. 树拓扑结构微观剖析<br>(Tree Topology Probe)** | State vs Afterstate | **痛点：** 缺乏硬核证据证明 Afterstate 搜索效率高的内在机理。<br>**方案：** 测量树的信息熵与真实穿透深度，验证“深且瘦”的猜想。 | **文件 1：** `mcts_topology_node.py` <br>(含 `get_tree_profile()`, `_entropy()`)<br>**文件 2：** `run_topology_experiment.py` <br>(含 `run_static_probe()`) | ❌ **代码已完成<br>等待运行验证** |
| | **2. 引入风险评估网络<br>(双模块设计)** | Afterstate 主导 | **思路：** 利用神经网络或专门的 NTuple 单独拟合环境的随机生成风险。 | *(无对应代码)* |  |
| **Expectimax<br>(精确概率搜索)** | **1. 加速方式设计** | Afterstate 主导 | **思路：** 分别使用beam search以及TT缓存表，发现都可以大幅降低运行速度，并且在高层级的时候，beam search会出现较大的减损，而TT缓存表能够保持较好性质。 | *(无对应代码)* |  | 结果可见phrase_3

>在进行Q/V的运行中，有个极大的问题，我们如果使用原生的6N-tuple，无法突破我们16000的分数，因此我认为，我们或许需要将我们原本的差分Ntuple也纳入其中，这样才能得到更好的效果。我们的phrase3部分的Q/V学习则是应用了这个。