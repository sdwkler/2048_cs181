# 2048 MCTS 架构演进：微观流程与数学公式全解析

在 2048 游戏中，**State（决策节点）** 指的是“玩家回合，可滑动”（盘面已满载方块）；**Afterstate（机会节点）** 指的是“环境回合，等待生成方块”（玩家刚滑完，存在空位）。

---

## 阶段一：标准 MCTS (纯随机模拟)

这是最初始的 MCTS，只存在 State 节点，没有启发式打分，全靠“随机走到死”来评估。

### 1. 微观流程对比
* **State 模式 (`mcts.py` use_afterstate=False)**：
  1. **选择**：在 State 节点用 UCB 选动作。
  2. **扩展**：执行动作 $\to$ **系统立即随机生成方块** $\to$ 得到全新的 State 子节点。
  3. **评估**：从新 State 出发，**纯随机乱滑**直至 Game Over。
  4. **回溯**：将最终得分累加更新。

* **Afterstate 模式 (`mcts.py` use_afterstate=True，早期朴素版)**：
  1. **选择**：在 State 节点用 UCB 选动作。
  2. **扩展**：执行动作 $\to$ **拦截系统生成** $\to$ 得到 Afterstate 子节点。
  3. **评估**：从 Afterstate 出发，**第一步必须先由系统随机生成一个方块将其补全为 State**，随后纯随机乱滑直至 Game Over。
  4. **回溯**：将最终得分累加更新。

### 2. 计算公式
* **探索公式 (UCB1)**：
  $$UCB(s, a) = \frac{Q_{sum}(s, a)}{N(s, a)} + C \cdot \sqrt{\frac{\ln N(s)}{N(s, a)}}$$
* **价值更新**：
  $$Q_{sum}(s, a) \leftarrow Q_{sum}(s, a) + Reward_{total}$$

---

## 阶段二：DAG MCTS (全局图折叠 + 截断打分)

为了解决“走到死”的巨大噪音，本阶段引入了 7 步截断与启发式打分。同时引入**全局字典 (`self.Q`)** 进行状态复用。

### 1. 微观流程与字典碰撞
两者的 Rollout 流程完全一致（走 7 步后强制停止打分）。根本分歧在于**字典的缓存命中率**：

* **State DAG 模式 (`mcts3.py` use_afterstate=False)**：
  * **扩展**：动作 $\to$ 随机生成方块 $\to$ 新 State。
  * **查字典**：由于 2048 每次生成的方块位置和数值极具随机性，两个路径产生完全一模一样 State 的概率极低（< 1%）。
  * **结果**：极少发生图折叠，依然是一棵方差极大的庞大树。

* **Afterstate DAG 模式 (`mcts3.py` use_afterstate=True，引发严重坍塌)**：
  * **扩展**：动作 $\to$ 得到 Afterstate。
  * **查字典**：因为方块还未生成，只要滑动后的主体结构一致，瞬间命中字典！（命中率 > 80%）。
  * **结果（价值污染）**：一条绕了弯路的劣质路径和一条顶级路径，高频折叠到了同一个 Afterstate 上。劣质路径的 7 步打分拉低了该节点的总期望，导致 AI 认为好棋是烂棋，产生“悲观坍塌”。

### 2. 计算公式
由于图折叠导致子节点访问量 $N(s')$ 远大于父节点边 $N(s,a)$，UCB1 崩溃，被迫采用 UCT3 (双键分离)：
* **探索公式 (UCT3 / DP-UCB)**：
  $$UCB(s, a) = \frac{Q(s')}{N_{value}(s')} + C \cdot \sqrt{\frac{\ln N_{state}(s)}{N_{edge}(s, a) + \alpha \cdot N_{value}(s')}}$$
  *(注：用 $N_{edge}$ 保障探索分母不爆炸，用 $\alpha$ 引入图先验)*

---

## 阶段三：Stochastic MuZero 架构 (二分严格树 + 重度模拟)

为了彻底解决 DAG 的价值污染，废弃全局字典。树的物理结构变为**决策节点 (State)** 与 **机会节点 (Afterstate)** 严格交替。

在这个框架下，`use_afterstate` 控制的不再是树的结构，而是 **评估 (Heavy Rollout) 到底在哪一层拦截触发**！

### 1. 微观流程对比：评估拦截点的较量
* **State 评估模式 (`mcts3_new_node.py` use_afterstate=False)**：
  1. **选择**：在 Decision Node(State)，基于 PUCT 选动作。
  2. **扩展 1**：生成 Chance Node(Afterstate)，直接跳过评估。
  3. **扩展 2**：环境随机生成方块，生成新 Decision Node(State)。
  4. **【拦截评估】**：在新 State 触发 5 步 Heavy Rollout。
  * **缺点**：AI 评估的是“生成方块后”的盘面，包含了环境噪音，树极其宽泛。

* **Afterstate 评估模式 (`mcts3_new_node.py` use_afterstate=True，目前的最优解)**：
  1. **选择**：在 Decision Node(State)，基于 PUCT 选动作。
  2. **扩展 1**：生成 Chance Node(Afterstate)。
  3. **【拦截评估】**：**立刻在此时触发 5 步 Heavy Rollout！**
  * **优点**：AI 评估的是“方块生成前”的纯净盘面。屏蔽了所有环境随机性，算力 100% 聚焦在玩家的 4 个动作上。
  4. **推进**：评估完后，再让环境生成方块进入下一层 State。

### 2. 计算公式 (完全对齐 Stochastic MuZero)
为了解决后期高分值压垮探索系数的问题，引入了 `MinMaxStats`。
* **价值归一化**：
  $$ValueScore(s,a) = \frac{\frac{Q_{sum}}{N} - Min_{tree}}{Max_{tree} - Min_{tree}}$$
* **探索公式 (PUCT)**：
  $$PUCT(s,a) = ValueScore(s,a) + \left( \ln\frac{N(s) + C_{base} + 1}{C_{base}} + C_{init} \right) \cdot \frac{\sqrt{N(s)}}{N(s,a) + 1}$$
  *(注：$C_{base}=19652, C_{init}=1.25$)*
* **回溯更新**：
  $$Q_{sum} \leftarrow Q_{sum} + TotalReturn$$
  $$MinMaxStats.update(TotalReturn)$$