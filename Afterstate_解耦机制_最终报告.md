# Afterstate 解耦机制在 2048 序贯决策中的系统评估

> **核心命题：** 在 2048 这个「玩家动作确定、环境响应随机」的马尔可夫决策过程中，Afterstate（后状态）解耦机制究竟在哪些条件下能提升 AI 算法的决策质量与计算效率，又在哪些条件下失效？本文通过三类算法框架（Expectimax / MCTS / TD 学习）、三个阶段（标准环境基准消融 / 环境漂移鲁棒性测试 / 架构演进与深度优化）的系统实验，回答三个递进的研究问题。

---

## §一 引言

### 1.1 问题背景

2048 游戏的每一步可分解为两个物理性质截然不同的子过程：(1) 玩家滑动——方块合并，纯确定性；(2) 环境在空位随机生成 2 或 4——纯随机性。传统的 State 建模将这两者糅合在一起，从滑动前的完整盘面 S 直接跳转到下一回合的完整盘面 S_next。Afterstate 建模则显式插入一个中间节点 S'——滑动后、落子前的纯净盘面——将确定性过程与随机性过程强行解耦。

这种解耦在理论上能带来三个维度的优势：搜索树的拓扑折叠（不同动作路径收敛到同一个 S'）、采样方差的缩减（评估对象不含随机噪声）、以及 TD 学习梯度的稳定（Bellman 目标不掺杂环境随机性）。但这些理论声称在多大程度上能被实验验证？是否存在理论未能预见的失效模式？这是本文试图系统回答的问题。

### 1.2 三个研究问题

本文围绕三个递进的研究问题（RQ）组织全部实验与分析：

| RQ | 问题 | 对应的实验阶段 |
|---|---|---|
| **RQ1** | Afterstate 在标准环境下（P(4)=10%）是否具有一致的决策质量与计算效率优势？优势的两个维度（树结构分离 vs 评估对象纯净）是独立可拆分的还是必须联合解耦？ | Phase 1：标准环境基准消融 |
| **RQ2** | 当环境概率发生偏移（P(4) 从 10%→90%），Afterstate 的鲁棒性是否强于 State？「解耦环境」是否等同于「免疫环境变化」？ | Phase 2：环境漂移鲁棒性测试 |
| **RQ3** | 针对 RQ1 和 RQ2 暴露的缺陷——过估计结构性回归、Afterstate NT 过拟合环境先验——能否通过架构改进（TDA-Full 精确期望、Downside-MV 双头方差、HashDAG 去重、MCTS N-Tuple 接入）来修复？ | Phase 3：架构演进与深度优化 |

三个 RQ 的逻辑递进关系：RQ1 确立基线并发现矛盾 → RQ2 在非平稳条件下压力测试 → RQ3 尝试修复暴露的缺陷并评估修复效果。

### 1.3 主要贡献

1. **系统性地验证了 Afterstate 联合解耦的必要性。** Expectimax 的 2×2 消融矩阵揭示了树结构分离与评估对象纯净两个维度之间存在严重的交互效应：单独分离其中一个维度可能适得其反（1-E 错配崩塌 −92%），两个维度必须联合解耦才能释放全部优势。

2. **推翻了「解耦环境 = 免疫环境」的直觉假设。** Phase 2 环境漂移测试一致表明，Afterstate NT 的鲁棒性显著弱于 State——因为 N-Tuple 权重将 P(4)=10% 的先验概率隐式「过拟合」进了估值函数。State 的全宽展开机制反而提供了更强的抗偏移能力。

3. **发现了两类被寄予厚望的架构改进（TDA-Full 精确期望、Downside-MV 双头方差）在 100K 局尺度上均未超越简单的 V+Afterstate 采样版。** 这是一个重要的负结果：精确的数学期望在 TD(λ) 资格迹框架下会产生目标函数不一致；仅惩罚下行风险的反而不如不惩罚。这两项发现为后续研究划定了「什么方向可能无效」的边界。

4. **揭示了过估计在 ε+α 退火下只是被延缓而非被根治。** 25K 局时所有实验呈低估（bias 为负），100K 局时过估计结构性回归（norm_bias_rom 从 −0.194 飙升至 +1.976），说明 V(s') 架构的过估计倾向是结构性的。

5. **量化了 Afterstate 在 MCTS 中的拓扑质变。** 接入 N-Tuple 估值后，Afterstate MCTS 得分（136K）超越了 Expectimax 完全解耦（115K），且搜索树从 State 的「矮胖」形态扭转为「瘦深」形态——probe_entropy 更低（0.318 vs 0.330），macro_depth 更深（5.71 vs 5.20）。

---

## §二 背景与理论框架

### 2.1 Afterstate 的形式化定义

在 2048 的单步 MDP 中，存在三个严格区分的状态节点：

```
 S (State)           S' (Afterstate)        S_next (Next State)
┌─────────┐         ┌─────────┐            ┌─────────┐
│ 滑动前   │  玩家   │ 滑动后   │  环境随机  │ 下一轮   │
│ 完整盘面 │ ─────→ │ 中间盘面 │ ────────→ │ 完整盘面 │
│ 等待操作 │  动作a  │ 方块已合 │  生成2/4  │ 等待操作 │
│         │  得分R  │ 尚未生成 │            │         │
└─────────┘         └─────────┘            └─────────┘
```

- **S（State）**：方块已生成，轮到玩家滑动
- **S'（Afterstate）**：玩家刚滑完、方块已合并、环境尚未生成新方块
- **S_next（Next State）**：环境在 S' 空位随机生成 2 或 4 后，进入下一回合

Afterstate 解耦的核心思想是：将 S → S_next 的一步跨越，拆分为 S → S'（确定性）和 S' → S_next（随机性）两个子步骤。

### 2.2 Afterstate 的三个理论优势维度

这三个维度对应三类算法框架，构成全文的方法论基础：

| 理论维度 | 核心声称 | 需要的测量手段 | 对应算法 |
|---|---|---|---|
| **拓扑折叠** | 不同动作路径可收敛到同一个 S'，搜索树从树坍缩为 DAG | 精确计数节点、计算分支因子/压缩率 | Expectimax |
| **方差缩减** | 评估对象是 S'（纯净），不含随机落子噪声，估值方差更低 | 重复采样、统计估值分布 | MCTS |
| **梯度稳定** | Bellman 目标（max 操作作用于确定性 R+V(S') 而非含噪声的 S_next）使 TD 更新信号更平滑 | 逐步追踪 TD 误差、测量过估计偏差 | TD 学习 |

三种方法不是「都试试看哪个好」，而是**同一命题在三个维度上的独立证伪测试**——任何一个维度上的负面结果都会对 Afterstate 的理论完整性构成挑战。

### 2.3 Afterstate 的两个可分离维度

在实验操作层面，「Afterstate」并非不可分割的整体。它包含两个独立可切换的维度：

- **维度一：树结构的 Afterstate。** 搜索树是否在动作节点和随机落子之间显式插入 S' 机会节点。控制的是「树拓扑是否变瘦」。
- **维度二：评估对象的 Afterstate。** 打分器的输入是纯净的 S' 还是含噪声的 S_next。控制的是「看什么打分更准」。

这两个维度的独立性意味着可以构造 **2×2 消融矩阵**：树结构分离与否 × 评估对象纯净与否。如果某一维度切换后得分不变，说明该维度在给定条件下不贡献增益；如果同时切换后得分大幅跃升，说明两个维度存在协同效应；如果单独切换某一维度导致性能崩溃，说明两个维度存在非线性的交互效应。这一矩阵逻辑贯穿 Expectimax 和 TD 学习两组实验。

### 2.4 Bitboard 棋盘引擎

棋盘引擎是整个系统的基础——在 Expectimax 中每步可能评估数万个节点，在 MCTS 中单局可能执行数十万次 Rollout 滑动，在 TD 学习中十万局训练意味着数千万次棋盘操作。核心设计目标是将单次滑动降低至微秒量级。

**4-bit 编码。** 2048 棋盘的每格取值均为 2 的幂：空(0)、2、4、8、…、32768，共计 16 种状态，恰好可用 4 个二进制位编码（$2^4=16$），存储的是指数而非原值：

| 存储值（4 bit） | 0 (0000) | 1 (0001) | 2 (0010) | … | 15 (1111) |
|---|---|---|---|---|---|
| 方块值 | 空 | 2 | 4 | … | 32768 |

棋盘共 4×4=16 格，每格 4 bit，总计 64 bit，恰为一个 64 位整数的宽度。采用小端序布局——左上角对应最低 4 位、右下角对应最高 4 位。选择 4 bit（而非 3 或 5）的核心理由是对齐优势：棋盘的一行恰好是连续的 16 bit。

**查表法 O(1) 滑动。** Bitboard 编码的直接红利是查表法——棋盘一行包含 4 格 × 4 bit = 16 bit，全部可能取值仅 $2^{16}=65536$ 种。程序启动时穷举 0~65535 所有行，对每一行预计算左滑结果（方块合并 + 靠拢 + 得分），存入查表 `lookup.find`。运行时提取行 → 查表 → 拼接，每行 O(1)，所有合并决策逻辑在预计算阶段就已执行完毕，运行时被压缩为一次内存读取。

**方向映射。** 四个滑动方向共用一张查表。上下方向通过棋盘旋转转化为左右方向：`rotate_clockwise` = `transpose` + `mirror`，`rotate_counterclockwise` = `transpose` + `flip`。上移 = 顺时针旋转 → 查右滑 → 逆时针旋转回来；下移 = 顺时针旋转 → 查左滑 → 逆时针旋转回来。

### 2.5 N-Tuple 网络：学习驱动的表征引擎

2048 棋盘的状态空间极为庞大——16 格 × 每格 16 种取值，理论空间达 $16^{16} \approx 1.8 \times 10^{19}$，穷举存储不可行。N-Tuple 网络是 2048 AI 中经过验证的经典函数逼近器，相比深度神经网络有三个优势：线性可训（TD(0) 在线更新，无反向传播）；内存可控（权重表大小由 tuple 选取方案显式决定）；旋转不变性通过同构映射天然嵌入而非靠数据学习。

**特征构造。** N-Tuple 的核心机制与棋盘引擎的查表法完全同构——将局部编码为整数索引，直接查数组取值。区别在于棋盘引擎查的是「滑动后变成什么」（确定性预计算），N-Tuple 查的是「这个局部配置前景好不好」（从数据中学习）。

本项目采用**绝对值 + 差分联合方案**，共 4 种 shape × 2 种编码 = 8 个 feature：

- **4 个 `pattern`（绝对值）**：在 16 格棋盘上选定 6 个固定位置，将 6 格的方块指数（0~15, 4 bit/格）逐位移拼接为 24 bit 索引 → 权重表 $2^{24}=16\text{M}$ 条目。直接存储每格的绝对指数，见过的盘面估值精确。
- **4 个 `diff_pattern`（差分）**：存相邻位置的差值。相邻差范围 −15~+15（31 种可能，需 5 bit），加偏移 +15 映射到非负数后拼接为 25 bit 索引 → 权重表 $2^{25}\approx 33.5\text{M}$ 条目。存差值的好处是：形状相同而整体数值平移的盘面（如盘面上所有方块各升一级）编码完全一致、共享权重，从而将经验泛化到训练中从未出现的配置。

> 虽然 `diff_pattern` 只存 5 个差（比 6 个绝对值少一个数），但每个差 5 bit > 每个绝对值 4 bit，所以表反而更大。

**同构映射。** 为消除方向偏见，每个 feature 配有 8 组不同朝向的观察窗口（4 个旋转方向 × 2 种镜像），分别覆盖棋盘的不同角落。8 组窗口各自查表，结果累加作为该 feature 的输出——同一配置无论出现在棋盘的哪个方向贡献相同。

**三层估值流程。**
1. **窗口层**：每个 feature 的 8 组同构窗口各自查表取值，累加得到该 feature 的输出。
2. **编码层**：4 个绝对值 feature + 4 个差分 feature 各自独立完成窗口层求和，产生 8 个标量分值。
3. **汇总层**：8 个 feature 的分值相加，得到最终的盘面估值 V(S')。

一次完整估值仅需访问 8 × 8 = 64 个浮点数。权重更新时，TD 误差均分到 64 个被激活的权重上，每个各加 α × error / 64。

**离线预训练 vs 在线训练。** N-Tuple 权重在两组实验中有两种截然不同的使用方式。离线预训练（Expectimax）：以 S 和 S' 为不同输入对象各训练 50,000 局，产出两套冻结权重 `2048_state.bin` 和 `2048_afterstate.bin`——两套权重由完全相同的网络结构和训练算法产生，唯一差异在于训练时的观测对象，这为 2×2 消融矩阵中的「评估对象维度」提供了物理载体。在线训练（TD 学习）：权重从零初始化，使用稀疏字典（Python `dict`）按需存储，训练初期大量棋盘配置从未出现，避免预分配完整权重数组浪费内存。每步执行后立即进行 TD 更新，权重持续变化。

---

## §三 实验框架

### 3.1 共享基础设施

所有实验共享一套统一的底层系统，确保观测到的差异唯一归因于 Afterstate 机制本身：

- **棋盘引擎**：64 位 Bitboard 编码（详见 §2.4），单次滑动通过 65536 行查表法在微秒级完成。
- **随机控制**：所有实验共享 seed=181，方块生成序列完全一致。同一算力档位下 Standard 与 Afterstate 的 MCTS agent 使用完全相同的 RNG 初始状态，确保两组间唯一差异是树结构。
- **双模式验证**：`smoke`（1~3 局快速验证代码正确性）和 `full`（完整统计），两套模式共享相同的随机种子和算法参数，仅在数据规模上不同。smoke 的设计原则是覆盖所有代码路径但秒级完成，full 投入充分算力产出有统计意义的结论。

### 3.2 评估器体系

系统提供四类评估器，覆盖从人工预设到数据驱动的完整谱系：

| 评估器 | 评分依据 | 权重来源 | 主要使用场景 |
|---|---|---|---|
| `HeuristicEvaluator` | 空位数 + 单调性 + 平滑度 + 角落最大值 | 人工经验预设（w_empty=270, w_mono=47, w_smooth=0.1, w_corner=500） | Expectimax 无学习基线 |
| `NTupleEvaluator` | 封装预训练的 N-Tuple 网络，调用 `tdl.estimate(b)` | TD(0) 离线训练 | Expectimax 学习上限；TD 学习在线训练 |
| `FastHeuristic` | 与 HeuristicEvaluator 同一套评估逻辑 | 人工经验预设 | MCTS Rollout 专用——65536 行查表 + 位运算，O(1)，不涉及 N-Tuple |

评估器在运行时是决策的裁判（给搜索树叶子节点或 MCTS Rollout 终点打分），在实验结束后其估值与真实回报之间的偏差（过估计偏差）又成为被分析的对象。在 Expectimax 实验中 NTupleEvaluator 的权重是冻结的预训练快照（只读）；在 TD 学习实验中同一 N-Tuple 的权重随 TD 更新持续变化（在线更新）——接口相同，角色不同。

### 3.3 各 Phase 数据规模

| 实验 | Phase 1 full | Phase 1 扩展 / Phase 3 full | Phase 2 |
|---|---|---|---|
| Expectimax 对局数 | 100 | — | 50 / 概率点 |
| Expectimax 高压残局数 | 500 | — | — |
| MCTS 对局数 | 10 | 10（拓扑实验） | 10 / 概率点 |
| MCTS 高压残局数 | 100 | — | — |
| MCTS 稳定性重复次数 | 50 | — | — |
| TD 学习训练局数 | 25,000 | 100,000 | —（零样本） |
| TD 学习评估对局数 | 100 | 100 | smoke 级定性 |

### 3.4 通用与专属评价指标

**通用指标（全部实验）：**

| 指标 | 含义 |
|---|---|
| 平均得分 / 平均步数 | 终局总分和滑动次数的均值 |
| 各档方块达标率 | 终局最高方块达到 1024 / 2048 / 4096 / 8192 的局数占比 |
| 步均耗时 | 单步决策的平均计算时间 |

**Expectimax 专属指标：**

- **有效分支因子** $b_{\text{eff}} = N_{\text{nodes}}^{1/d}$：度量平均每加深一层树变胖多少倍。树越瘦（节点去重越充分），$b_{\text{eff}}$ 越低。
- **压缩率** $\text{CR} = N_{\text{unique}} / N_{\text{total}}$：直接量化有多少节点是重复展开的。架构不分离时 $\text{CR}\to 1.0$（几乎无折叠）；架构分离后 $\text{CR}\to 0.50\sim 0.55$（约一半节点被去重）。
- **决策分歧遗憾值**：在高压残局上分别让 State 方案和 Afterstate 方案各自决策，以完全解耦方案为基准计算分歧率和选错动作的估值代价，回答「估值更准是否真的转化为决策更好」。

$b_{\text{eff}}$ 和压缩率本质上是同一件事的两种度量——Afterstate 让不同路径合并到同一个 S'，搜索树变瘦。$b_{\text{eff}}$ 量的是后果（平均每层分岔数），压缩率直接数重复（多少节点白展开了）。

**MCTS 专属指标：**

- **根方差** $\sigma_{\text{root}} = \text{std}(Q_{\text{up}}, Q_{\text{right}}, Q_{\text{down}}, Q_{\text{left}})$：方差大 = AI 能清晰区分动作优劣；方差→0 = 四个方向估值趋同，在噪声中瞎蒙。
- **策略熵** $H(\pi) = -\sum_a p(a) \log p(a)$：同一盘面重复运行 50 次的首选动作分布熵。低熵 = 决策稳定；高熵 = 被 Rollout 噪声左右。直接量化「Afterstate 结构能否等效替代数倍的算力增长」。

**TD 学习专属指标：**

- **TD 误差 RMS** $\text{RMS}(\delta) = \sqrt{\frac{1}{n}\sum \delta_i^2}$，其中 $\delta = |R + \gamma V(s'_{\text{next}}) - V(s')|$，每 1000 步计算一次。同步计算**归一化 TD 误差 RMS** $= \text{RMS}(\delta) / (\text{average\_score} + 1)$，消除不同实验间绝对分数量级差异的影响。
- **过估计偏差** $\text{Bias} = V_{\text{predict}} - G_{\text{real}}$：正 Bias → 盲目乐观；负 Bias → 保守安全。采用双重归一化并行记录：**Mean-of-Ratios (MoR)** 对每个盘面等权计算 $\text{mean}_i(\text{bias}_i / (\text{realized}_i+1))$；**Ratio-of-Means (RoM)** $= \sum\text{bias}_i / \sum(\text{realized}_i+1)$，对大返回值盘面更鲁棒，减少离群点支配。

### 3.5 核心方法论：2×2 消融矩阵（重申）

§2.3 定义的两个独立维度——树结构分离与评估对象纯净——在 Expectimax 和 TD 学习实验中通过 2×2 消融矩阵进行独立操纵。Expectimax 的 8 组实验（1-A 至 1-H）和 TD 学习的 5 组实验（3-A 至 3-E）均嵌入这一矩阵逻辑。Phase 3 在此基础上增加了 TDA-Full 精确期望（3-F）和 Downside-MV 双头方差（3-G）两种目标函数变体。

---

## §四 RQ1：Afterstate 在标准环境下是否具有一致优势？

> 实验条件：P(4)=10%，标准 2048 规则，seed=181。

### 4.1 Expectimax：联合解耦是前提，错配代价不对称

**实验设计。** Expectimax 是完美信息下的最优搜索策略——假设叶子节点估值完全准确，深度优先穷举所有可能的动作序列和随机结果，取期望最大的动作。搜索深度统一为 2（玩家动→环境动），1-A（Greedy 基线）深度为 1。使用预训练 N-Tuple 冻结权重（分别以 S 和 S' 为输入各训练 50,000 局），通过 2×2 消融矩阵同时控制树结构（`use_afterstate`）和评估对象（`leaf_mode` + 打分器类型）。

**搜索树结构对比。** Afterstate 的介入改变了搜索树中决策节点和机会节点的连接方式：

```
【Standard Expectimax（不分离）】          【Afterstate Expectimax（分离）】
       S (决策节点)                              S (决策节点)
      / |  \                                  / | \
    a0 a1 a2 a3                             a0 a1 a2 a3
     |  |   |  |                              |  |  |  |
    [环境随机生成]                           S'0 S'1 S'2 S'3 (机会节点)
     |  |   |  |                              |  |  |  |
    S0 S1  S2  S3  (下一层决策节点)          [环境随机生成]
                                               |  |  |  |
                                              S0 S1 S2 S3

  ❌ 每个动作下挂 N 个随机子节点              ✅ S' 节点充当「拓扑漏斗」
  ❌ 两路径到同一 S_next 概率极低               ✅ 多路径高频命中同一 S'
  ❌ N_unique ≈ N_total                      ✅ N_unique << N_total (压缩率→0.5)
```

在 Standard 模式下，每个动作之后环境立即在所有空位上随机生成新方块，搜索树在机会节点处爆炸式分岔。在 Afterstate 模式下，动作之后首先抵达一个共同的 S' 节点（仅含合并结果），然后才分岔到不同的 S_next——由于不同动作可能产生相同或相似的 S'（例如左滑和右滑在对称盘面上产生镜像结果），搜索树从「每路径独立展开」坍缩为「多路径共享 S'」的 DAG 结构。

**8 组实验矩阵：**

| 实验 | 树结构 | 打分器 | 类别 |
|---|---|---|---|
| 1-A | Greedy（深度=1） | Heuristic | 绝对基线 |
| 1-B | Standard（不分离） | Heuristic（看 S_next 叶子） | 普通搜索 |
| 1-C | Afterstate（分离） | Heuristic（看 S' 叶子） | 架构解耦 |
| 1-D | Standard（不分离） | State N-Tuple（看 S） | 正确匹配 |
| 1-E | Standard（不分离） | Afterstate N-Tuple（看 S'） | **表征错配** |
| 1-F | Afterstate（分离） | State N-Tuple（看 S） | 反向错配 |
| 1-G | Afterstate（分离） | Afterstate N-Tuple（看 S'） | **完全解耦** |
| 1-H | Afterstate + Afterstate NT + 剪枝 | 同 1-G + Top-2 Beam Search | 终极拓展 |

1-D 和 1-G 是两组「正确匹配」的配置——打分器训练分布与叶子评估对象一致。1-E 和 1-F 是两组「错配」配置——通过对比可量化两种错配的代价和不对称性。

**结果。**

| 实验 | 树结构 | 评估对象 / 打分器 | 得分 | 2048率 | 4096率 | 耗时 | b_eff | 压缩率 |
|---|---|---|---|---|---|---|---|---|
| 1-A (Greedy) | — | Heuristic | 4,779 | 0% | 0% | 0.11ms | 3.49 | 0.997 |
| 1-B (Std+Heur) | Standard | State / Heur | 11,059 | 1% | 0% | 4.93ms | 47.23 | 0.996 |
| 1-C (After+Heur) | Afterstate | Afterstate / Heur | 11,952 | 4% | 0% | **0.64ms** | 14.45 | **0.504** |
| 1-D (Std+StateNT) | Standard | State / State NT | 64,740 | 92% | 72% | 26.81ms | 39.59 | 0.998 |
| 1-E (Std+AfterNT) | Standard | State / Afterstate NT | **5,066** | 0% | 0% | 34.47ms | 44.55 | 0.994 |
| 1-F (After+StateNT) | Afterstate | Afterstate / State NT | 60,126 | 93% | 67% | 2.56ms | 13.22 | 0.553 |
| 1-G (After+AfterNT) | Afterstate | Afterstate / Afterstate NT | **114,885** | **97%** | **91%** | 2.59ms | 13.04 | 0.553 |
| 1-H (+Pruning) | Afterstate | Afterstate / Afterstate NT | **125,721** | **100%** | **96%** | 2.31ms | **8.04** | 0.597 |

**回答 RQ1 的核心发现：**

**1. 联合解耦是前提，错配代价高度不对称。** 1-E（Standard 树 + Afterstate NT）——树不分离但打分器看 S'——得分仅 5,066，比纯贪心基线还差。Standard 树的 `leaf_mode="state"` 将 S_next（含随机落子噪声）喂给训练时只看过纯净 S' 的 Afterstate NT，输入分布彻底错位。反方向的 1-F（Afterstate 树 + State NT）仅下降 7%。错配成本的不对称性揭示了两类 N-Tuple 的本质差异：State NT 从 S 泛化到 S' 容易（少一个方块不影响盘面本质），Afterstate NT 从 S' 泛化到 S_next 困难（多一个随机方块污染了纯净信号）。

**2. 正确匹配时打分器质量主导得分。** 完全解耦的 1-G（115K）比 1-D（65K）高出 77%。通过 1-F vs 1-D 的对照（同 State NT、树不同）可知，树结构对得分影响很小（60K vs 65K），77% 的增幅主要来自打分器质量的提升：看 S' 学到的估值函数比看 S 学到的更精确。

**3. 树结构主导效率。** Afterstate 树的压缩率均为 0.50~0.60，Standard 树均接近 1.0——Afterstate 树约一半节点被去重合并。这直接解释了 10.3 倍的加速（1-D 26.81ms vs 1-G 2.59ms）。

**4. 剪枝锦上添花，但前提是评估精度。** 1-H 在 1-G 基础上加入 Top-2 Beam Search，得分再提 9% 至 126K，达到 100% 2048 和 96% 4096。

### 4.2 MCTS：优势随算力释放，但人工价值函数锁死上限

**实验设计。** MCTS 与 Expectimax 的根本不同在于：它不依赖任何预训练评估器，而是靠当场随机模拟（Rollout）判断动作好坏。本组实验完全使用 FastHeuristic（手工价值函数，与 HeuristicEvaluator 同一套评估逻辑，但通过 65536 行查表 + 纯位运算实现 O(1)）做 Rollout 终局估值代理。每次 Rollout 最多 5 步，滑行策略为 90% 启发式贪心 + 10% 真随机。四个方向的 Rollout 预算（200/500/1000/2000）由 PUCT 公式自动分配：

$$\text{PUCT} = V_{\text{score}} + c_{\text{pb}} \cdot \frac{\sqrt{N_{\text{parent}}}}{N_{\text{child}} + 1}, \quad c_{\text{pb}} = \log\left(\frac{N_{\text{parent}} + C_{\text{base}} + 1}{C_{\text{base}}}\right) + c_{\text{init}}$$

其中 $C_{\text{base}}=19652$，$c_{\text{init}}=1.25$，对齐 Stochastic MuZero 的探索调度策略。得分高的方向多试，得分少的少试；初始时四个方向估值相同，探索项迫使每个方向都先试几次。

**Afterstate 在 MCTS 中的噪声处理。** 与 Expectimax 不同，MCTS 中 Afterstate 的区别体现在随机噪声被固定在哪里：

- **Standard MCTS**：滑一下，系统立刻生成新方块，随机方块成为棋盘的一部分被固定下来。200 次 Rollout 的起点各不相同（方块掉在不同空格），噪声变成「事实」直接污染估值基线。
- **Afterstate MCTS**：滑一下，趁新方块还没生成，从干净的合并后盘面起步 Rollout。新方块不出现于起步时，而是被推进 Rollout 内部——成为 Rollout 里每一次滑行之后自然发生的随机事件，通过取平均消解掉。

Standard 与 Afterstate 的 agent 从同一 RNG 状态出发，Rollout 随机序列完全一致——两组之间的唯一差异是树结构。实验在三个层级同时进行：单步决策测试（100 个高压残局，记录根方差和策略熵）、稳定性测试（每个残局重复 50 次独立运行）、整局游戏测试（10 局完整游戏）。

**结果。**

| 实验 | Rollouts | 得分 | 2048率 | 4096率 | 根方差 | 策略熵 | 耗时 |
|---|---|---|---|---|---|---|---|
| 2-A (Std) | 200 | 48,548 | 100% | 50% | 25.0 | 0.342 | 50.2ms |
| 2-E (AS) | 200 | 45,414 | 100% | 40% | 22.7 | 0.339 | 56.0ms |
| 2-B (Std) | 500 | 58,056 | 100% | 70% | 23.2 | 0.239 | 125.8ms |
| 2-F (AS) | 500 | 54,280 | 100% | 50% | 22.5 | 0.263 | 116.1ms |
| 2-C (Std) | 1000 | 48,642 | 100% | 40% | 21.7 | 0.179 | 231.8ms |
| 2-G (AS) | 1000 | 60,497 | 100% | 70% | 21.0 | 0.200 | 220.6ms |
| 2-D (Std) | 2000 | 58,614 | 100% | 60% | 21.1 | 0.125 | 464.2ms |
| 2-H (AS) | 2000 | **65,934** | **100%** | **80%** | 20.4 | 0.146 | 443.6ms |

**回答 RQ1 的核心发现：**

**1. Afterstate 在高算力下占优，反超阈值为 1000 Rollouts。** 低算力时（200/500）Standard 略优，1000 Rollouts 时 Afterstate 首次反超（60.5K vs 48.6K），2000 时扩大领先（65.9K vs 58.6K）。

**2. Standard 在 500→1000 Rollouts 出现非单调下降。** Std 500: 58K → Std 1000: 48.6K（−16%）。可能原因：中间算力下 Rollout 噪声与 PUCT 探索-利用平衡进入不稳定区间。Afterstate 则平滑上升。

**3. 人工价值函数锁死上限在 ~66K。** 无 N-Tuple 的纯启发式 Rollout 无法达到 Expectimax N-Tuple 的 115K——但远高于 Heuristic Expectimax 的 12K，说明 MCTS 通过即时采样在一定程度上弥补了评估精度的不足。

### 4.3 TD 学习：25K 的乐观与 100K 的回归

**实验设计。** N-Tuple 网络从零初始化（稀疏字典 `SparseNTupleValue` 按需分配，训练初期大量配置从未出现）。Phase 1 基准 25,000 局，扩展至 100,000 局；Phase 3 全部 100,000 局。ε 截断退火 0.10→0.0001（训练进度 80% 触底，后 20% 纯贪心），α 线性退火 0.05→0.002（初期大步快速学习，末期小步精调防震荡）。γ=1.0（2048 为有限时域任务，终局必然来临），TD(λ)=0.5（平衡一步 TD 与蒙特卡洛回报，加速信用分配传播）。

**5 组消融矩阵：** 围绕更新目标（Q(s,a) vs V(s')）和 N-Tuple 特征输入（看 S vs 看 S'）两个维度组织。

| 实验 | 更新目标 | 特征输入 | 类别 |
|---|---|---|---|
| 3-A | Q(s,a) | State | 传统基线——教科书级 Q-Learning，4 个动作脑各学各的 |
| 3-B | Q(s,a) | Afterstate | 表征解耦——只换输入（看 S'），目标不变 |
| 3-C | V(s') | State | 逻辑证伪——V(S) 与动作无关，退化为只比即时奖励 |
| 3-D | V(s') | Afterstate | **完全解耦**——全局 V 共享经验 + 输入不含随机噪声 |
| 3-E | MV(s') | Afterstate | 风险控制——V-head 学均值，M-head 学平方期望，动作选择减 λ√Var |

Phase 3 在此基础上新增 3-F（TDA-Full 精确期望）和 3-G（Downside-MV 双头方差），详见 §六。

**3-E（MV）机制。** 在 V 之外增加一个 M，学习 $\mathbb{E}[G^2 \mid S']$（$G$ = 从当前往后直到终局的总得分），与 $V(S') \approx \mathbb{E}[G \mid S']$ 组合得到方差 $\text{Var}(S') = \max(0, M - V^2)$。动作选择时减去 $\lambda \cdot \sqrt{\text{Var}}$，惩罚高波动方向。M 的学习率设为 V 的 1/10——$G^2$ 比 $G$ 波动大得多，降低学习率防止 M 的训练不稳定干扰 V。

#### 25,000 局结果

| 实验 | 得分 | 2048率 | 4096率 | TD RMS(fin) | Norm Bias(RoM) | 训练时间 |
|---|---|---|---|---|---|---|
| 3-A Q+State | 3,672 | 0% | 0% | 105.6 | −0.524 | 919s |
| 3-B Q+After | 10,566 | 1% | 0% | 242.5 | −0.534 | 1,539s |
| 3-C V+State | 2,980 | 0% | 0% | 84.6 | −0.384 | 628s |
| 3-D V+After | **19,508** | **34%** | 0% | 526.9 | −0.194 | 1,705s |
| 3-E MV+After | **20,607** | **36%** | 0% | 538.0 | −0.142 | 3,315s |

**25K 层的核心发现：**

- 3-D 和 3-E 领跑，完全解耦方案确认优势（3-D vs 3-A: +431%）。3-E 在数值上略高于 3-D（+5.6%）但训练时间翻倍，方差惩罚增量有限。
- 3-C（V+State）得分 2,980——V(S) 对所有动作输出相同估值，退化为只比即时奖励，逻辑证伪成立。
- 所有实验 norm_bias_rom 为负（−0.14 到 −0.53）——ε+α 退火在 25K 尺度压制了过估计。

#### 100,000 局扩展：过估计结构性回归

> 将 3-D 和 3-E 延长至 100K 局，退火参数不变。

| 实验 | 得分 | 2048率 | 4096率 | Bias(绝对) | Norm Bias(RoM) | 25K→100K 变化 |
|---|---|---|---|---|---|---|
| 3-D (100K) | 18,731 | 34% | 2% | **+34,002** | **+1.976** | 得分 −4%，bias 从负翻正 |
| 3-E (100K) | 18,768 | 34% | 1% | **+34,314** | **+1.934** | 得分 −9%，3-E 优势消失 |

**100K 层的核心发现——这是 RQ1 最重要的矛盾发现：**

**1. 过估计爆炸回归。** norm_bias_rom 从 25K 时的 −0.194/~−0.142 飙升至 +1.98/+1.93——网络估值比实际回报高约 2 倍。ε+α 退火只是延缓而非根治了过估计。V(s') 架构的过估计倾向是结构性的：随着训练进行，TD(λ) 资格迹的长期回溯使估值被系统性抬高。

**2. 得分不升反降。** 3-D 从 19,508 降至 18,731；3-E 从 20,607 降至 18,768。3-E 在 25K 时 1,099 分的领先优势在 100K 时完全消失（差 36 分，0.2%）。MV 方差惩罚在长期训练中没有累积优势。

**3. 唯一的进步是 4096 率破零（2%）。** 但相比 Expectimax 同等 N-Tuple 权重达到的 91% 4096，差距仍然悬殊。100K 远非收敛终点。

### 4.4 RQ1 小结

**RQ1 的回答是「有条件的是」。** Afterstate 在标准环境下具有显著优势，但必须满足两个前提：(a) 树结构与评估对象两个维度联合解耦——单独分离可能适得其反（1-E: −92%）；(b) 评估器具有足够精度——手工启发式无法拉开性能差距。最重要的矛盾发现是：ε+α 退火在 25K 尺度压制了过估计，但在 100K 尺度过估计结构性回归。这个矛盾直接驱动了 RQ3 的 TDA-Full 精确期望实验——能否通过改造 TD 目标本身来根治过估计？

**遗留问题（输入 RQ2 和 RQ3）：**
1. Afterstate NT 将 P(4)=10% 的假设焊死在权重里——环境变化时会怎样？（→ RQ2）
2. 过估计在 100K 回归——改 TD 目标能否根治？（→ RQ3 TDA-Full）
3. MCTS 被人工价值函数锁死在 66K——接入 N-Tuple 后上限能提到多高？（→ RQ3 MCTS 拓扑）

---

## §五 RQ2：Afterstate 在环境漂移下是否仍然鲁棒？

> RQ1 发现 Afterstate NT 估值将 P(4)=10% 的先验假设隐式编码进了 N-Tuple 权重。RQ2 直接测试：如果环境本身变了（P(4) 从 10% 逐步提升至 90%），Afterstate 还能保持优势吗？一个直觉假设是：既然 Afterstate 解耦了环境，它应该在环境变化时更稳健。RQ2 的目的就是检验这个假设。

### 5.1 Expectimax 漂移：初始猜想被推翻

**实验设计。** 使用 Phase 1 训练好的 M3（State NT）和 M6（Afterstate NT）冻结权重，在 P(4) ∈ {0.1, 0.2, ..., 0.9} 的 9 个概率梯度上各跑 50 局，不做任何重训。引入 M-Regret 追踪：以完全解耦的 Afterstate NT 为基准，计算 State 与 Afterstate 决策分歧的盘面比例（Disagreement Rate）和这种不一致的实际估值代价（Regret）。

**完整漂移曲线：**

| P(4) | M3-State-NT | M6-After-ANT | M6 优势(分) | Disagreement Rate | Max Regret |
|---|---|---|---|---|---|
| 0.1 | 64,407 | **113,552** | +49,145 (+76%) | 0.26 | 15,464 |
| 0.3 | 54,239 | **69,746** | +15,507 (+29%) | 0.31 | 10,884 |
| 0.5 | 25,130 | **39,639** | +14,509 (+58%) | 0.31 | 18,679 |
| 0.7 | 18,767 | **26,216** | +7,449 (+40%) | 0.29 | 8,940 |
| 0.9 | 11,459 | **19,357** | +7,898 (+69%) | 0.33 | 13,839 |

**回答 RQ2 的核心发现：**

**1. 初始猜想被推翻——Afterstate 的鲁棒性弱于 State。** M6 的绝对优势从 P(4)=0.1 时的 +49,145 分急剧缩水至 P(4)=0.9 时的 +7,898 分。M6 得分下降 83%（114K→19K），M3 下降 82%（64K→11K）——衰减比例几乎一致，但 M6 的绝对落差更大。如果 Afterstate 真的「免疫环境变化」，它的衰减幅度应该远小于 State，但数据表明恰恰相反。

**2. 机理：Afterstate NT 将环境概率过拟合进了权重。** Afterstate 的 N-Tuple 训练时只见过 P(4)=10% 的环境，学到的 V(S') 估值隐式假设了「每个空格的出牌分布是固定的 90% 出 2、10% 出 4」。当这个分布偏移后，整套估值逻辑失真。State 每次决策全宽展开所有可能的落子——它不假设 P(4) 的取值，而是「算出」当前环境下的真实期望。在环境固定时这个「笨重但诚实」的机制是劣势（慢），但在环境偏移时反而是优势（准）。

**3. 决策分歧率随环境恶化单调上升。** Disagreement Rate 从 0.26 升至 0.33——环境越恶劣，State 和 Afterstate 给出的动作建议分歧越大。Max Regret 在 P(4)=0.4 时达到峰值 19,819——中等偏移下 Afterstate 的决策错判代价最高。

**4. 人工价值函数组（M1/M2）无区分度。** Heuristic State 和 Heuristic After 在任何 P(4) 下得分差距极小（<10%），两者都随环境恶化同步缓慢下降。低精度估值无法反映 N-Tuple 的过拟合效应。

### 5.2 MCTS 漂移：两者同步衰减，Afterstate 被反超

**实验设计。** 全部使用 FastHeuristic（手工价值函数），1,000 次模拟/步，10 局/概率点。追踪 micro_entropy（访问混乱度，反映算力是否分散）和 micro_depth（树深）。

| P(4) | MCTS-State | MCTS-After | After 优势 | State entropy | After entropy |
|---|---|---|---|---|---|
| 0.1 | 28,935 | **35,064** | +21% | 0.397 | 0.405 |
| 0.3 | 22,426 | **25,388** | +13% | 0.380 | 0.386 |
| 0.5 | 16,852 | **19,219** | +14% | 0.365 | 0.370 |
| 0.7 | 15,797 | 14,871 | **−6%** | 0.358 | 0.362 |
| 0.9 | 13,977 | **15,410** | +10% | 0.367 | 0.363 |

**回答 RQ2 的核心发现：**

**1. Afterstate 在高偏移时优势不再稳定。** P(4)=0.7 时 Afterstate 被 State 反超（14,871 < 15,797）。MCTS 的随机采样给了 State 一定的抗偏移能力——State 在搜索树中自然展开了更多随机分支，相当于为高 P(4) 环境做了被动适配。

**2. micro_entropy 在所有 P(4) 下维持在 0.36~0.40，没有明显收敛。** 在极高随机性干扰下，UCB 算力分配机制「雨露均沾」，无法锁定绝对优势分支。

**3. 人工价值函数再次锁死了上限。** P(4)=0.1 时最优仅 35K——虽然 MCTS 本身的随机采样提供了比 Expectimax（114K→19K）更平缓的衰减曲线，但绝对性能始终受限于启发式估值的精度。

### 5.3 Q-Learning 零样本迁移：全面崩溃

**实验设计。** 25K 局训练收敛后的 3-A 至 3-E 模型直接放入偏移环境，不做任何微调（Zero-Shot）。smoke 级定性验证，少量对局/概率点。

**核心数据（3-D vs 3-E）：**

| P(4) | 3-D (V+After) | 3-E (MV+After) |
|---|---|---|
| 0.1 | 19,767 | 17,957 |
| 0.3 | 10,215 | 10,733 |
| 0.5 | 7,809 | 7,556 |
| 0.7 | 4,644 | 5,991 |
| 0.9 | 5,242 | 5,141 |

**回答 RQ2 的核心发现：**

**1. 全部模型在 P(4)=0.9 时崩溃至 <6K。** 最优的 3-D 从标准环境的 19,767 暴跌至 5,242（−73%），3-E 从 17,957 跌至 5,141（−71%）。所有模型退化为仅能活数百步的随机策略级别。

**2. 3-E（MV）没有提供任何额外的鲁棒性。** 在绝大多数 P(4) 下 3-E 落后于 3-D。P(4)=0.7 时的 +29% 是孤立波动（smoke 模式下样本量小）。方差惩罚在环境恶化时反而进一步压低了探索积极性。

**3. 迁移学习（Continual Learning）未获得有效数据。** 在困难环境中继续训练试图恢复分数的实验（Phase 2 原计划）耗时极长且往往无法收敛回原水平（负迁移现象），未产出可用于统计分析的量化数据。

### 5.4 RQ2 小结

**RQ2 的回答是「否」——而且是反向的。** Afterstate 不仅没有更强的环境鲁棒性，反而因为将 P(4)=10% 的先验概率过拟合进了 N-Tuple 权重，在环境漂移时鲁棒性显著弱于 State。这个结论在 Expectimax（M6 优势缩水）、MCTS（P(4)=0.7 被反超）和 Q-Learning（全面崩溃）三个框架下高度一致。

**核心洞察「解耦环境 ≠ 免疫环境」：** Afterstate 通过状态截断分离了玩家的确定性动作和环境的随机响应——这确实带来了 Phase 1 中观察到的计算效率大幅提升。但在建模过程中，Afterstate 的价值函数不可避免地吸纳了环境当前的统计规律。当环境变化时，这个「过拟合」变成了致命的盲点。State 虽然低效，但它的全宽展开机制是「无模型」的——不对环境做任何假设，因此在环境突变时反而具备更强的泛化能力。

**遗留问题（输入 RQ3）：**
1. Afterstate NT 对 P(4) 的过拟合能否通过在 TD 目标中显式使用环境概率来消除？（→ TDA-Full）
2. 单步采样 TD Target 被高方差撕裂——改为精确期望能否改善？（→ TDA-Full）
3. 能否将「盘面价值」与「随机风险」拆分为两个独立网络？（→ Downside-MV 双头设计）

---

## §六 RQ3：能否通过架构改进修复 Afterstate 的短板？

> RQ1 暴露了过估计结构性回归；RQ2 暴露了 Afterstate NT 对环境概率的过拟合。RQ3 测试三类改进：(1) Q-Learning 层面——将 TD 目标从单步采样改为精确数学期望（TDA-Full），将 MV 方差重构为仅惩罚下行风险的 Downside-MV；(2) Expectimax 层面——利用 Afterstate 的确定性实现 HashDAG 去重和 Beam Search 剪枝；(3) MCTS 层面——接入 N-Tuple 估值，追踪树拓扑的微观变化。

### 6.1 Q-Learning：TDA-Full 与 Downside-MV 均未超越简单基线

> 全部使用 100,000 局训练，ε 截断退火 0.10→0.0001，α 线性退火 0.05→0.002，seed=181。模型从零初始化，使用稀疏字典 `SparseNTupleValue`。去掉了 Q-learning 基线 3-A/3-B，保留 3-C（逻辑证伪组）和 3-D（完全解耦基线），新增 3-F（TDA-Full）和 3-G（Downside-MV）。

**3-F（TDA-Full）的机制。** 标准 TD 学习使用单步采样计算 TD Target：$\text{target} = \gamma \cdot \max_a [R_a + V(S'_{\text{next}, a})]$，其中 $S'_{\text{next}}$ 是环境随机落子后的一个特定结果。TDA-Full 将此替换为对所有可能落子结果的精确数学期望：$\text{target} = \gamma \cdot \sum_{e} P(e) \cdot \max_a [R_a + V(S'_a \mid e)]$，遍历当前 afterstate 的所有空位 × 两种 tile 概率（90% 出 2，10% 出 4）。理论上这消除了环境采样的白噪声，但对每个 afterstate 需要至多 14 空位 × 2 tile × 4 动作 = 112 次 board 操作。

**3-G（Downside-MV）的机制。** 将 §4.3 中 3-E 的单 M-head 方差拆分为上行方差头（m_up_head，追踪正偏差——突然合成高阶方块的意外高分）和下行方差头（m_down_head，追踪负偏差——低于期望的意外低分）。动作选择公式变为 $a^* = \arg\max [R + V + \lambda_{\text{up}} \cdot \sigma_{\text{up}} - \lambda_{\text{down}} \cdot \sigma_{\text{down}}]$，仅对下行风险施加惩罚（$\lambda_{\text{down}}=0.002$），对上行潜力给予奖励（$\lambda_{\text{up}}=0.001$），期望实现「贪心且保守」的非对称行为。

| 实验 | 目标模式 | 特征模式 | 得分 | 2048率 | 4096率 | TD RMS(fin) | Norm Bias(RoM) | 训练时间 |
|---|---|---|---|---|---|---|---|---|
| 3-C | V+采样 | State | 2,980 | 0% | 0% | 88 | −0.241 | 4,032s |
| 3-D | V+采样 | Afterstate | **18,731** | **34%** | 2% | 1,651 | +1.976 | 13,115s |
| 3-E | TDA-Full | State | 2,980 | 0% | 0% | 158 | +0.502 | 12,434s |
| 3-F | TDA-Full | Afterstate | 17,767 | 20% | 2% | 1,314 | **+2.355** | 38,403s |
| 3-G | Downside-MV | Afterstate | 17,937 | 26% | 2% | 1,693 | +1.950 | 24,196s |

**回答 RQ3 的核心发现——这是本文最重要的负结果：**

**1. TDA-Full 没有实现预期的「降维打击」，反而劣于简单采样版。** 3-F（TDA-Full+Afterstate）得分 17,767，**低于** 3-D（V+采样）的 18,731。3-F 的 2048 率 20% < 3-D 的 34%。3-F 的过估计**更严重**（norm_bias_rom +2.355 > +1.976）。训练时间是 3-D 的 2.9 倍（38,403s vs 13,115s）。

**失败原因分析：** (a) TDA-Full 每步对 afterstate 做精确期望（遍历所有空格 × 两种 tile 概率 = 最多 28 次 board 操作）——这在数学上是「正确的 TD Target」。但 TD(λ) 资格迹将当前步的精确期望广播到历史数百步的特征，而历史那些步的更新目标仍是采样版——目标函数的不一致破坏了收敛方向。(b) 精确期望消除了采样噪声，也使模型丧失了探索多样性，策略过早锁定次优解。唯一正面信号：TD 误差 RMS 下降了 20%（1,314 vs 1,651），说明精确期望确实让局部更新信号更稳定——但稳定在了「精确但错误的方向」上。

**2. Downside-MV (3-G) 也未超越 V+Afterstate (3-D)。** 得分 17,937 < 18,731，2048 率 26% < 34%。将 MV 重构为分别追踪上行方差和下行方差（m_up_head / m_down_head），仅对下行风险施加惩罚——这个设计在理论上更精细，但在实践中下行惩罚压低了探索积极性，使策略过于保守。

**3. 3-E（TDA-Full+State）与 3-C（采样+State）得分完全相同（2,980）。** 再次确认了 State 特征下 V 学习的根本缺陷：V(S) 对所有动作输出相同估值，退化为只比即时奖励——换什么目标函数都没用。

**4. 所有 Phase 3 QL 实验仍无 4096 突破。** 最高 4096 率仅 2%（3-D, 3-F, 3-G）——100K 局训练仍无法接近 Expectimax 冻结 N-Tuple 权重的 91% 4096。

### 6.2 Expectimax：HashDAG 完美无损，BeamSearch 深度依赖

> 分别测试人工价值函数（Heuristic）组和预训练 N-Tuple 价值函数组，各含 Base（全展开）、BeamSearch（Top-2 剪枝）、HashDAG（哈希去重）三种变体。depth=2 和 depth=3 分别独立实验，100 局/配置。

| 配置 | Depth=2 得分 | Depth=3 得分 | Depth=3 4096率 |
|---|---|---|---|
| Heur + Base | 12,505 | 20,096 | 0% |
| Heur + BeamSearch | 8,996 (−28%) | 13,344 (−34%) | 0% |
| Heur + HashDAG | 12,505 (≡Base) | 20,096 (≡Base) | 0% |
| NTuple + Base | 113,399 | 144,703 | 84% |
| NTuple + BeamSearch | **120,719 (+6.5%)** | 130,123 (−10%) | 76% |
| NTuple + HashDAG | 113,399 (≡Base) | 144,703 (≡Base) | 84% |

**回答 RQ3 的核心发现：**

**1. HashDAG 在所有条件下完美无损。** 得分与 Base 完全一致（12,505≡12,505; 113,399≡113,399; 20,096≡20,096; 144,703≡144,703）。压缩率约 0.50~0.55，depth=3 时节点展开数减少 70%、时间缩短 53%。Afterstate 的确定性是 DAG 折叠的物理基础——不同动作路径高频收敛到同一个 S' 节点，哈希命中率极高。

**2. BeamSearch 效果高度依赖评估精度和搜索深度。** N-Tuple depth=2 时 BeamSearch 得分**反升 +6.5%**（113K→121K），2048 率从 98% 升至 100%；depth=3 时则 −10%（145K→130K）。浅层搜索时 Top-2 剪枝不仅没有错失关键分支，反而排除了估值噪声导致的分心路径（「去噪」效果）。深层搜索时这个红利消失。Heuristic 在任何深度都 −28%~−34%——高精度估值是 BeamSearch 从「豪赌」变为「自信」的前提。

### 6.3 MCTS：N-Tuple 接入后的拓扑质变

> 将 Phase 1 MCTS 中的人工价值函数替换为预训练 N-Tuple，在高压残局（空格 ≤ 3）上进行微观拓扑追踪。1,000 次模拟/步，10 局/配置。追踪 macro_depth（宏观平均树深）、probe_entropy（探针位置访问混乱度）、probe_depth（探针深度）、layer_profile（各层节点分布）。

| 配置 | 得分 | 2048率 | 4096率 | macro_depth | probe_entropy | probe_depth |
|---|---|---|---|---|---|---|
| Heuristic + State | 29,276 | 68% | 6% | 5.22 | 0.221 | 5.37 |
| Heuristic + Afterstate | 33,272 | 82% | 10% | 5.83 | 0.235 | 6.05 |
| N-Tuple + State | 90,636 | 100% | 24% | 5.20 | **0.330** | 5.69 |
| N-Tuple + Afterstate | **135,712** | **100%** | **70%** | **5.71** | 0.318 | **6.29** |

**回答 RQ3 的核心发现：**

**1. N-Tuple 接入后 MCTS 分数跃升 3~4 倍，超越 Expectimax 完全解耦。** Heuristic State 29K → N-Tuple State 91K（×3.1）；Heuristic After 33K → N-Tuple After 136K（×4.1）。N-Tuple Afterstate MCTS 的 136K 超越了 Phase 1 N-Tuple Expectimax 完全解耦的 115K——MCTS 的动态采样机制在获得高精度估值后，反而比固定深度的穷举搜索更有优势。

**2. 「State 矮胖，Afterstate 瘦深」的拓扑差异被精确量化。** State N-Tuple：probe_entropy=0.330（高混乱，算力分散），macro_depth=5.20（较浅）。Afterstate N-Tuple：probe_entropy=0.318（低混乱，算力聚焦），macro_depth=5.71（更深）。Afterstate 的 layer_profile 在深层（第 5 层以后）节点数远少于 State——搜索树从「矮胖的伞」扭转为「锐利的长剑」。

**3. 低精度估值无法发挥 Afterstate 的拓扑优势。** Heuristic 组的 entropy 差异仅 0.014 vs N-Tuple 组的 0.012，说明在评估精度不足时，树结构的差异被估值噪声淹没。只有当 UCB 获得了足够准确的动作价值信号，Afterstate 的低方差特性才能兑现为树的形态变化。

**4. Afterstate 在 MCTS 中的终极机制被揭示。** Afterstate 通过降低估值方差，欺骗了 UCB 公式中的探索项，强行将 MCTS 从「广度优先采样」扭转成「深度优先规划」。这就是它能在极度拥挤的盘面下规避低级随机死亡、将 90%+ 算力绝对聚焦于唯一正确分支的根本原因。

### 6.4 RQ3 小结

**RQ3 的回答是「部分可以，但最被寄予厚望的两个改进失败了」。**

- ✅ 成功的改进：HashDAG（无损去重）、BeamSearch（需高精度估值+浅深度）、MCTS N-Tuple 接入（拓扑质变 + 超越 Expectimax）
- 🔴 失败的改进：TDA-Full 精确期望（劣于采样版，过估计更严重）、Downside-MV 双头方差（劣于简单 V+Afterstate，策略过度保守）

这两个负结果是 RQ3 最有价值的知识贡献——它们为后续研究划定了「什么方向可能无效」的边界。TDA-Full 的失败指向一个深层问题：在 TD(λ) 资格迹框架下，「精确的数学期望」和「从采样历史中学习」这两个目标存在根本性的不一致。

---

## §七 综合讨论

### 7.1 Afterstate 的优势条件谱系

将三个 RQ 的全部证据汇总，Afterstate 并非在任何条件下都占优。下表给出它的优势条件谱系：

| 条件 | Afterstate 优势程度 | 关键证据 | 来源 |
|---|---|---|---|
| 预训练 N-Tuple + 树与评估联合解耦 | **极大** | Expectimax: +77% 得分, −90% 耗时 | RQ1 |
| 预训练 N-Tuple + 剪枝 | **极大** | Expectimax 1-H: 126K, 100% 2048 | RQ1 |
| N-Tuple MCTS 高算力 | **极大** | MCTS 136K vs 91K (+49%), 70% 4096 | RQ3 |
| TD 学习 25K ε+α 退火 | **显著** | 3-D 19.5K vs 3-A 3.7K (+431%) | RQ1 |
| HashDAG 去重 | **完美无损** | 所有配置下得分≡Base | RQ3 |
| BeamSearch + N-Tuple 浅深度 | **正向** | depth=2 +6.5%（「去噪」效应） | RQ3 |
| BeamSearch + N-Tuple 深深度 | **轻微负向** | depth=3 −10% | RQ3 |
| TD 学习 100K ε+α 退火 | **中等，过估计回归** | 3-D 18.7K, bias +1.98 | RQ1 |
| 低精度 Heuristic MCTS | **微弱** | After vs State 差异小（<10%） | RQ1/RQ2 |
| **TDA-Full + Afterstate 100K** | **无优势（负向）** | 3-F 17.8K < 采样版 18.7K | RQ3 |
| **Downside-MV + Afterstate 100K** | **无优势（负向）** | 3-G 17.9K < 18.7K, 策略过度保守 | RQ3 |
| 环境漂移 P(4)→0.9 | **Afterstate 鲁棒性弱于 State** | M6 优势 +49K→+8K; P=0.7 时 MCTS 被反超 | RQ2 |
| Zero-shot 迁移 | **全面崩溃** | 所有模型 P(4)=0.9 时 <6K | RQ2 |

### 7.2 被数据证伪的关键假设

以下列出了研究过程中提出的、被后续实验数据明确推翻的核心假设。这些负结果是本文的重要知识贡献。

| # | 原始假设 | 来源文档 | 实际结果 | 数据来源 |
|---|---|---|---|---|
| 1 | TDA-Full 精确期望「将从根源上消除白噪声」，成为降维打击 | 中层.md Phase 3 预测 | 3-F 得分 17,767 **<** 采样版 3-D 18,731；过估计更严重（+2.36 vs +1.98） | Phase 3 QL 100K |
| 2 | Downside-MV 将带来「策略质变」「生存步数显著提升」 | 中层.md / Qlearning问题.md | 3-G 得分 17,937 **<** 3-D 18,731；25K 时 MV 微弱优势在 100K 时消失 | Phase 3 QL 100K |
| 3 | ε+α 退火根治了过估计 | 浅层总结 §6.3 / §7.1 | 25K 时被压制，100K 时 norm_bias_rom 从 −0.19 飙升至 +1.98 | Phase 1 QL 100K |
| 4 | Afterstate 解耦环境 → 免疫环境变化 | Phase 1 实验后猜想 | Afterstate 鲁棒性**弱于** State（三个算法框架一致结论） | Phase 2 全模块 |
| 5 | Beam Search 「约 10% 的分数折损」 | 中层.md Phase 3 | depth=2 N-Tuple **+6.5%**；depth=3 才 −10%；Heuristic −28%~−34% | Phase 3 Expectimax |
| 6 | 100K 局训练接近收敛，得分应提升 | 浅层总结 §7.4 待办 | 3-D 得分从 19,508 **降至** 18,731；3-E 从 20,607 降至 18,768 | Phase 1/3 QL 100K |

### 7.3 三类算法的 Afterstate 增益机制对比

| 算法 | Afterstate 增益的主要来源 | 增益幅度 | 核心限制条件 |
|---|---|---|---|
| Expectimax | 树结构 DAG 折叠（−90% 耗时）+ 评估对象降噪（+77% 得分） | 极高 | 必须联合解耦；打分器训练分布必须与评估对象一致 |
| MCTS | 无评估器时增益微弱；接入 N-Tuple 后方差缩减触发拓扑质变（广→深） | 无 N-Tuple: 微弱；有 N-Tuple: 极高 | 需要高精度估值才能释放拓扑优势 |
| TD 学习 | Bellman 目标纯净（V(s') 共享经验）+ 特征输入降噪（看 S' 不看 S_next） | 25K: +431%；100K: 过估计回归 | 退火仅延缓过估计；TDA-Full 和 MV 均未解决结构性问题 |

### 7.4 关键局限性与开放问题

1. **TDA-Full 与 TD(λ) 的交互未充分探究。** Phase 3 代码中 TDA-Full 仍使用 TD(λ) 资格迹（λ=0.5），导致精确期望目标与采样历史之间的不一致。若改为 λ=0.0 或设计独立的更新机制，结果可能不同。

2. **QL 训练仍未接近收敛。** 100K 局时 TD 误差仍在下降趋势中，4096 率最高仅 2%，距 Expectimax 冻结权重的 91% 差距悬殊。更长的训练量（200K+）是否能使过估计最终收敛或继续膨胀？目前无法判断。

3. **Phase 2 QL 漂移为 smoke 数据。** Zero-Shot 迁移测试仅跑了定性验证，无法产出有统计显著性的量化结论。迁移学习（Continual Learning）实验未完成。

4. **梯度谱分析（方向二）未执行。** Phase 1 发现旧版 3-D 过估计 +193,795（纯贪心+大 α），提出「Afterstate 作为梯度预条件器」的假设。但新版 100K 数据表明过估计在退火下只是延迟而非消失，梯度病态可能是架构固有属性而非训练协议产物。此项验证可决定是否将 Afterstate 的优势归因于梯度层面的结构性改善。

5. **深层研究的 POMDP（战争迷雾）环境未启动。** 本文所有实验均建立在「AI 能清晰看到棋盘」的完全信息假设上。一旦引入部分可观测性，Afterstate 的所有已验证优势将面临根本性挑战。

---

## §八 结论

### 8.1 对三个 RQ 的最终回答

**RQ1：Afterstate 在标准环境下是否具有一致优势？**
有条件的是。优势依赖于两个前提的**同时满足**：(a) 树结构与评估对象联合解耦——单独分离可能适得其反；(b) 评估器具有足够精度。在完全解耦 + 高精度 N-Tuple 的条件下，Expectimax 得分提升 77%、耗时降低 90%，MCTS 在 1000+ Rollouts 占优，TD 学习 25K 完全解耦（3-D）为最优。但 100K 扩展实验揭示了重要的约束：ε+α 退火只是延缓而非根治过估计。

**RQ2：Afterstate 在环境漂移下是否仍然鲁棒？**
否——而且是反向的。Afterstate NT 将标准环境的概率先验过拟合进了 N-Tuple 权重，在环境漂移时鲁棒性显著弱于 State。这个结论在 Expectimax、MCTS 和 Q-Learning 三个框架下高度一致。核心洞察是：「解耦环境 ≠ 免疫环境」。

**RQ3：架构改进能否修复 Afterstate 的短板？**
部分可以，但两个最被寄予厚望的方案（TDA-Full 精确期望、Downside-MV 双头方差）失败了。成功的改进集中在利用 Afterstate 的确定性特性（HashDAG 无损去重）和接入更高精度的评估器（MCTS N-Tuple 拓扑质变）。TDA-Full 和 Downside-MV 的失败揭示了一个深层困境：在 TD(λ) 资格迹框架下，精确的数学期望与采样历史之间存在目标函数的不一致。

### 8.2 方法论启示

本文的一个方法论贡献是展示了**系统性负结果的科学价值**。TDA-Full 和 Downside-MV 的设计都有坚实的数学动机，如果只跑 25K 局（和大多数消融实验一样），很可能因为统计噪声得出「未见显著差异」的模糊结论。正是 100K 局的长尺度训练，才明确揭示了这两个方案的结构性缺陷——不是参数没调对，而是机制层面就不如简单基线。这为后续研究节省了大量试错成本。

---

## 附录 A：文件导航索引

| 文件 | 功能 |
|---|---|
| [src/environments/base_env.py](src/environments/base_env.py) | 64 位 Bitboard 2048 棋盘引擎 |
| [src/ntuple/feature_base.py](src/ntuple/feature_base.py) | N-Tuple 特征提取 + TD(0) 引擎 |
| [src/evaluators.py](src/evaluators.py) | Heuristic / NTuple / FastHeuristic 评估器 |
| [src/common.py](src/common.py) | 配置、指标、对局模拟、结果输出 |
| [src/phase_1/search/](src/phase_1/search/) | Phase 1 Expectimax 实验 |
| [src/phase_1/planning/](src/phase_1/planning/) | Phase 1 MCTS 实验 |
| [src/phase_1/learning/](src/phase_1/learning/) | Phase 1 Q-Learning 实验 |
| [src/phase_2/](src/phase_2/) | Phase 2 环境漂移实验 |
| [src/phase_3/](src/phase_3/) | Phase 3 架构演进实验 |
| [final_results/](final_results/) | 全部实验的最终 CSV/JSON/Markdown 数据 |

## 附录 B：TD 学习消融实验矩阵

| 实验 | 更新目标 | 特征输入 | Phase | 训练局数 | 说明 |
|---|---|---|---|---|---|
| 3-A | Q(s,a) | State | 1 | 25K | 传统 Q-Learning 基线 |
| 3-B | Q(s,a) | Afterstate | 1 | 25K | 表征解耦、目标不变 |
| 3-C | V(s') | State | 1+3 | 25K/100K | 逻辑证伪组 |
| 3-D | V(s') | Afterstate | 1+3 | 25K/100K | **完全解耦（性价比最优）** |
| 3-E | MV(s') | Afterstate | 1 | 25K/100K | 均值-方差风险控制 |
| 3-F | TDA-Full | Afterstate | 3 | 100K | 全宽期望目标（本文核心负结果） |
| 3-G | Downside-MV | Afterstate | 3 | 100K | 双头下行方差（本文核心负结果） |

## 附录 C：数据来源

| 数据 | 最终路径 |
|---|---|
| Phase 1 QL 25K | [final_results/phrase_1/eval_results/qlearning_parallel_full_20260627_235506.csv](final_results/phrase_1/eval_results/qlearning_parallel_full_20260627_235506.csv) |
| Phase 1 QL 100K | [final_results/phase_1_qlearning/results/qlearning_parallel_full_20260630_040128.csv](final_results/phase_1_qlearning/results/qlearning_parallel_full_20260630_040128.csv) |
| Phase 1 Expectimax | [final_results/phrase_1/eval_results/search_full_20260627_215003.csv](final_results/phrase_1/eval_results/search_full_20260627_215003.csv) |
| Phase 1 MCTS | [final_results/phrase_1/eval_results/planning_full_20260628_035707.csv](final_results/phrase_1/eval_results/planning_full_20260628_035707.csv) |
| Phase 2 Expectimax Drift | [final_results/phrase_2/expectimax/results/expectimax_drift_full_20260629_215319.csv](final_results/phrase_2/expectimax/results/expectimax_drift_full_20260629_215319.csv) |
| Phase 2 MCTS Drift | [final_results/phrase_2/mcts_drift/results/mcts_drift_robustness_full_20260630_002832.csv](final_results/phrase_2/mcts_drift/results/mcts_drift_robustness_full_20260630_002832.csv) |
| Phase 2 QL Drift | [final_results/phrase_2/Qlearning/qlearning_drift_20260626_140247/qlearning_drift_results_smoke_20260627_021156.csv](final_results/phrase_2/Qlearning/qlearning_drift_20260626_140247/qlearning_drift_results_smoke_20260627_021156.csv) |
| Phase 3 QL 100K | [final_results/phrase_3/eval_results/qlearning_parallel_full_20260701_144225.csv](final_results/phrase_3/eval_results/qlearning_parallel_full_20260701_144225.csv) |
| Phase 3 Expectimax d=2 | [final_results/phrase_3/expectimax_eval_results/depth2/search_optimizations_full_20260629_123634.csv](final_results/phrase_3/expectimax_eval_results/depth2/search_optimizations_full_20260629_123634.csv) |
| Phase 3 Expectimax d=3 | [final_results/phrase_3/expectimax_eval_results/depth3/search_optimizations_full_20260629_150446.csv](final_results/phrase_3/expectimax_eval_results/depth3/search_optimizations_full_20260629_150446.csv) |
| Phase 3 MCTS Topology | [final_results/phrase_3/topology_tests/mcts_topology_analysis_full_20260630_183925.csv](final_results/phrase_3/topology_tests/mcts_topology_analysis_full_20260630_183925.csv) |
