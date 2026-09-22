# 1. Executive summary

对你的 `3Chase1`，我认为最重要的结论不是“换一个更复杂的 MARL 网络”，而是**把协同决策、6-DOF 低层控制、推进器分配、柔性网捕获判据这四层彻底解耦**。现有文献能分别为这四层提供较强证据，但我没有找到一篇已经完整解决“3 个 6-DOF 水下机器人 + 异构 Herder/Netter 角色 + 两 Netter 共同携带柔性网 + 移动 Prey + CTDE MARL + 实际网面捕获”的成熟工作。最接近的是 2026 ICUAS 的 *Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning*，但它仍是空中多旋翼，而且每个 pursuer 携带的是自身刚性圆形捕获网，不是两机器人耦合的柔性网。([arXiv][1])

因此，你这个任务本身有明确的研究空隙。现有证据可以支持：

* **MAPPO 是合理主基线**。其在 cooperative MARL 上是经过系统 benchmark 的强 on-policy baseline；HAPPO/HARL 则特别适合作为“异构角色是否值得非同质更新/独立参数化”的对照。([NeurIPS 会议录][2])
* **30D 全局位置-free、body-relative actor observation 的总体方向是合理的**，而且和近期 pursuit 工作的 decentralized observation 很接近。但它严格来说不是完整 Markov state：推进器一阶状态、柔性网内部状态、PID 积分状态、可能的姿态/重力方向等都被隐藏了。
* 对你现在只有 3 个 Chaser、两个固定 teammate slot 的问题，**没有文献依据仅仅因为“30D”就上 Transformer**。优先级应当是：MLP → slot/shared teammate encoder → 对两个 Netter 做 permutation-invariant pooling → 如果确有 temporal aliasing 再上 GRU。MAT 证明 Transformer 可以用于多智能体序列建模，但不是“30D observation 应使用 Transformer”的证据。([NeurIPS 会议录][3])
* **不把 OBI mesh、网法向、四角点等放进执行期 actor 是合理而且我支持的。**真正应该加入的不是“更多 simulator truth”，而是只有在实验表明存在明显 perceptual aliasing 时，加入现实中也能稳定测量的网代理量，例如 attachment separation、separation rate、attachment tension/load、经过滞回处理的 slack/taut/contact 状态。
* 我最推荐作为论文主方法的是
  **4D body-frame subgoal + fixed PID + physically bounded wrench allocator**。这不是因为文献已经证明它一定比 direct thruster RL 好，而是因为已有工作共同支持“把任务级 RL 与低层稳定控制解耦”这一设计，而你的科学问题本来就是**异构协作和收网决策**，没有必要强迫 MAPPO 同时重新学习一遍 8 推进器分配。近期 6-DOF 水下 RL 已证明 direct-thruster 本身也可行，因此你的 8D direct-thruster MAPPO 仍然是很有价值的强 ablation，而不是应该删除的旧 baseline。([arXiv][4])
* **6D wrench RL + allocator 是非常重要的中间对照**。它能回答一个很有论文价值的问题：4D hierarchy 的提升到底来自“物理 allocator”，还是来自“更高层的动作抽象”。
* 你最终的核心 comparison 应该是：

[
\boxed{
8D;\text{thruster}
\quad\text{vs}\quad
6D;\text{wrench}+CA
\quad\text{vs}\quad
4D;\text{subgoal}+PID+CA
}
]

而不是只拿“你的方法 vs 某个传统 controller”。

控制分配方面，Johansen–Fossen 的 control-allocation 文献以及针对 over-actuated AUV 的 constrained allocation 都明确支持：**当执行器有幅值约束时，应显式处理可达 wrench 和 saturation，而不是把无约束伪逆当作最终实现。**([科学直接][5])

最后一个非常重要的信息论问题：如果你所谓的“收网区域”是**固定在世界中的区域**，当前 30D observation 中没有任何量告诉 Herder 该区域在哪里，那么问题本身是部分不可观的；网络再复杂也无法可靠解决。若“收网区域”实际上由 Netter/柔性网实时定义，则当前 teammate-relative observation 可以间接定义它，这时无需 world pose。

---

# 2. 文献表格

下面按“直接相关程度”而不是单纯年份排序。

**证据等级：**

* **A：直接证据**——水下协同追逐、6-DOF 水下控制或真正使用捕获网。
* **B：强近邻证据**——多机器人追逃/herding、柔性 tether/net、多机器人层级控制。
* **C：方法基础**——CTDE/MAPPO/HAPPO/control allocation/LOS 等。

## 2.1 Pursuit / encirclement / herding / underwater pursuit

| 文献                                                                                                                                         | 平台与验证                            | Observation / Action / 层级                                                                              | 异构 / 网                | 主要结果与本任务映射                                                                                                                         |
| ------------------------------------------------------------------------------------------------------------------------------------------ | -------------------------------- | ------------------------------------------------------------------------------------------------------ | --------------------- | ---------------------------------------------------------------------------------------------------------------------------------- |
| **Zhang et al., *Multi-robot Cooperative Pursuit via Potential Field-Enhanced Reinforcement Learning*, ICRA 2022**                         | 2D 移动机器人；仿真+实机                   | decentralized RL 与 APF 混合                                                                              | 同质；无网                 | 证明“几何先验/APF + RL”可作为纯 RL 的有力 pursuit baseline。不能证明 6-DOF 或柔性网性能。IEEE 官方页可核实。([IEEE Xplore][6])                                     |
| **Wang et al., *Encirclement Guaranteed Cooperative Pursuit with Robust Model Predictive Control*, IROS 2021**                             | 2D 多 pursuer/单 evader；仿真         | decentralized tube-MPC；显式保持 evader 位于 pursuer convex hull                                              | 同质；无网                 | 非常适合作为“几何包围 ≠ 捕获网”的传统强 baseline。其凸包约束不能直接作为你的成功定义。([arXiv][7])                                                                     |
| **Kouzeghar et al., *Multi-Target Pursuit by a Decentralized Heterogeneous UAV Swarm using Deep MARL*, ICRA 2023**                         | Crazyflie；仿真+实机                  | role-based MADDPG；distributed execution                                                                | **异构探索/追踪角色**；无网      | 是 Herder/Netter 固定功能角色的重要近邻证据；但其异构性是探索/跟踪，不涉及机械耦合。([arXiv][8])                                                                     |
| **Mohanty et al., *Distributed Multirobot Control for Non-Cooperative Herding*, DARS 2022/arXiv 2023**                                     | 地面 robots；仿真+最多 5 dog/5 sheep 实验 | CBF 约束 dog velocity；集中/分布版本                                                                            | 功能角色明确；无网             | **Herder 最有用的传统依据之一**：Herder 不必“追到 prey”，而应选择能诱导 prey 朝期望方向逃逸的位置。([arXiv][9])                                                      |
| **Chen et al., *A Dual Curriculum Learning Framework for Multi-UAV Pursuit-Evasion in Diverse Environments*, AAAI 2024**                   | 3D UAV；仿真并展示 sim-to-real         | MARL + curriculum；低层动力学控制接口                                                                            | pursuers 同类；无网        | 论文报告训练场景 >90% capture，并通过任务参数和环境参数双 curriculum 增强泛化。其 capture 是 proximity 类条件，**不能替代你的真实网捕获指标**。([arXiv][10])                      |
| **Feng, Wu, Tan, “基于MARL-MHSA架构的水下仿生机器人协同围捕策略: 数据驱动建模与分布式策略优化”, 自动化学报 2025**                                                               | **水下仿生机器鱼；仿真+水池实机**              | CTDE + multi-head self-attention；数据驱动 sim                                                              | 同类机器鱼；无网              | 中文直接相关工作。作者报告相较 MAPPO 平均围捕成功率提高 24.3%、围捕步长减少 30.9%。对象是仿生鱼围捕，不是 6-DOF thruster ROV。DOI 10.16383/j.aas.c250086。([AAS][11])           |
| **Feng et al., *Decentralized Multirobotic Fish Pursuit Control With Attraction-Enhanced Reinforcement Learning*, IEEE TIE 2025**          | 仿生机器鱼；仿真+实机                      | attraction prior + decentralized RL                                                                    | 同质；无网                 | 支持“pursuit 几何先验 + RL”以及真实水下 pursuit 可行性；与 FinsROV 六自由度差异大。([IEEE Xplore][12])                                                      |
| **Feng et al., *M²GRPO: Mamba-based Multi-Agent Group Relative Policy Optimization for Biomimetic Underwater Robots Pursuit*, arXiv 2026** | 仿生水下机器人；仿真+水池实验                  | history/Mamba + relational modeling                                                                    | 多 pursuer；无网          | 2026 预印本，对“历史是否能解决水下 pursuit POMDP”特别相关；但尚属预印本，不能作为 Transformer/RNN 必要性的普遍证明。([arXiv][13])                                         |
| **Gavin & Bronz, *Intercepting an Agile Target with Net-Carrying Drones using Competitive MARL*, ICUAS 2026**                              | 高保真 quadrotor simulator          | MAPPO + PFSP；actor local；action 为 collective thrust/body rates，再由低层 controller 执行                      | 多 pursuer；**真实捕获网概念** | **目前最接近你的 MARL+net 工作。**但每架 pursuer 自己携带刚性圆网，并非两个机器人连接一个柔性网，因此不能支持 OBI 网法向/面状态假设。DOI 10.1109/ICUAS69441.2026.11598610。([arXiv][1]) |
| **Masmitja et al., *Dynamic Robotic Tracking of Underwater Targets Using Reinforcement Learning*, Science Robotics 2023**                  | **真实海洋机器人**                      | RL 用于动态 target tracking                                                                                | 无机械网                  | 强证据表明 RL 可进入真实动态水下 target-tracking pipeline；任务是 tracking 而非 capture。DOI 10.1126/scirobotics.ade7811。([科学杂志][14])                   |
| **Zhu et al., *Task-Semantic Graph-Driven Distributed Agent Networking for Underwater Target Tracking*, arXiv 2026**                       | **6-DOF AUV simulator**；开源       | STG-MAPPO；semantic/local policy input；**velocity-level high-level actions → 6DOF executable controls** | 分布式 AUV swarm；无网      | 对你“不要让高层 RL 直接管推进器”的论证非常相关。开源 MARL-AUV。属 2026 预印本。([arXiv][15])                                                                    |

---

## 2.2 6-DOF AUV/ROV control、hierarchy 与 control allocation

| 文献                                                                                                                                                          | 模型/动作                                                              | 验证                                          | 对你的意义                                                                                                          |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------ | ------------------------------------------- | -------------------------------------------------------------------------------------------------------------- |
| **Cai, Chang, Girdhar, *Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-driven AUVs*, ICRA 2025**                                    | command-conditioned 6-DOF target → normalized individual thrusters | **zero-shot real AUV**；论文称性能与手调 PID 可比；域随机化 | 这是你 **8D direct-thruster baseline 最强的直接依据**。它证明 direct-thruster 不是不合理；但只证明单机控制，不证明它适合多机器人策略层。代码公开。([arXiv][4]) |
| **Johansen & Fossen, *Control Allocation—A Survey*, Automatica 2013**                                                                                       | generalized force → constrained actuators                          | 理论/综述                                       | 奠基文献：over-actuated 系统应把 motion control 与 actuator allocation 分层。Automatica 49(5):1087–1103。([科学直接][5])         |
| **Johansen et al., *Optimal Constrained Control Allocation in Marine Surface Vessels with Rudders*, Control Engineering Practice 2008**                     | constrained optimization                                           | marine control                              | 证明 marine control 中显式约束 allocator 是成熟路线；不是 AUV 特有。([科学直接][16])                                                 |
| **Yuan et al., *An Efficient Control Allocation Algorithm for Over-actuated AUVs Trajectory Tracking with Fault-Tolerant Control*, Ocean Engineering 2023** | saturation-constrained allocation / over-actuated AUV              | 仿真                                          | 与你的 8-thruster physical allocator 最直接。核心迁移：应该显式处理执行器 bounds、可实现 wrench 和残差。([科学直接][17])                        |
| **Fossen & Aguiar, *A Uniform Semiglobal Exponential Stable Adaptive LOS Guidance Law for 3-D Path Following*, Automatica 2024**                            | 3D LOS guidance                                                    | 理论+marine guidance                          | LOS 很适合作为“给定路径/几何目标后的 guidance 层”，不应该被当成一个完整 3-agent net capture algorithm。([科学直接][18])                        |
| **Yan et al., *Model Predictive Control of AUVs for Trajectory Tracking with External Disturbances*, Ocean Engineering 2020**                               | constrained trajectory MPC                                         | AUV simulation                              | 可作为强传统低层/trajectory baseline，但计算量和调参成本明显高于 PID。**不建议第一版拿 MPC 替换 PID，否则变量太多。**([科学直接][19])                      |
| **Havenstrøm et al., *PID Controller Assisted Reinforcement Learning for Path Following by AUVs*, arXiv 2020**                                              | PID 与 RL 混合                                                        | 6-DOF AUV simulation                        | 是“RL 不一定承担所有低层控制”的直接近邻。与你的 exact 4D subgoal architecture 不相同，因此只能作为概念证据。([arXiv][20])                          |

---

## 2.3 柔性网、缆绳、机械耦合的近邻文献

这里尤其要强调：**这些不是“OBI 柔性收网已有成熟解决方案”的证据。**

| 文献                                                                                                                                                                   | 柔性/机械关系                                | 实验            | 可迁移 / 不可迁移                                                                                                                                                           |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------- | ------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Klausen, Fossen, Johansen, *Autonomous Recovery of a Fixed-Wing UAV Using a Net Suspended by Two Multirotor UAVs*, Journal of Field Robotics 2018, 35(5):717–731** | **两个 carrier 共同悬挂一张 net**              | 实验验证          | 在拓扑上与你两个 Netter 带网最相似：两个机器人之间的相对几何本身就是捕获装置状态的一部分。但它是空中 suspended net，不能套用其几何假设到 OBI 水下网。DOI 10.1002/rob.21772。([Wiley Online Library][21])                           |
| **Li & Loianno, *Nonlinear MPC for Cooperative Transportation and Manipulation of Cable Suspended Payloads with Multiple Quadrotors*, IROS 2023**                    | 多机器人通过柔性 cable 共同操控 rigid payload      | 仿真+实验         | 强证据说明 mechanical coupling、actuator constraint 和 inter-robot constraint 不应被独立 agent controller 忽略；但 payload 不是网。([IEEE Xplore][22])                                   |
| **Yang et al., *Collaborative Navigation and Manipulation of a Cable-Towed Load by Multiple Quadrupedal Robots*, IEEE RA-L 2022**                                    | 多机器人 cable-towed load；存在 slack/taut 模式 | 实机            | 非常有价值的思想：柔性连接的“模式”可能比完整几何更有控制意义。支持你考虑稳定的 slack/tension proxy 而不是 mesh。([IEEE Xplore][23])                                                                            |
| **Novák, Báča, Saska, *Collaborative Object Manipulation on the Water Surface by a UAV-USV Team Using Tethers*, 2024**                                               | UAV+USV+tether+floating object         | Gazebo/VRX 仿真 | 是 marine-near-neighbor。作者明确把 tether tension constraint 纳入 MPC，并由上层产生 robot references、低层 controller 执行。非常支持你的 hierarchy 思路，但它不是水下 flexible-net capture。([arXiv][24]) |

---

## 2.4 CTDE / heterogeneous MARL / architecture

| 文献                                                                                                       | 核心结论                                                                             | 3Chase1 用法                                                                                                    |
| -------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------- |
| **Lowe et al., *Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments*, NeurIPS 2017** | MADDPG；centralized critic / decentralized policies 的经典工作                         | CTDE 奠基 baseline；连续 action 但比 MAPPO 老。([NeurIPS 会议录][25])                                                     |
| **Yu et al., *The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games*, NeurIPS 2022**      | 系统证明 MAPPO/IPPO 在多个 cooperative benchmarks 上很强                                   | **MAPPO 主 baseline 必须保留；IPPO 是判断 critic 是否真正有用的关键 ablation。**([NeurIPS 会议录][2])                               |
| **Kuba et al., *Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning*, ICLR 2022**     | HATRPO/HAPPO sequential update                                                   | 固定 Herder/Netter 异构角色下，可测试 separate/heterogeneous updates 是否优于共享 MAPPO。([OpenReview][26])                     |
| **Zhong et al., *Heterogeneous-Agent Reinforcement Learning*, JMLR 2024, 25(32):1–67**                   | HARL/HAPPO/HATRPO general heterogeneous-agent framework                          | **HAPPO 很适合作为你的 learned heterogeneous baseline。**JMLR 页面附公开代码。([机器学习研究杂志][27])                                |
| **Lyu et al., *A Deeper Understanding of State-Based Critics in MARL*, AAAI 2022, 36(9):9396–9404**      | state-based centralized critic 并非无条件优越，某些情形会引入 policy-gradient bias 或更高 variance | 很重要：不要默认“critic 塞越多 OBI truth 越好”。应做 critic-information ablation。DOI 10.1609/aaai.v36i9.21171。([AAAI 期刊][28]) |
| **Wen et al., *Multi-Agent Reinforcement Learning is a Sequence Modeling Problem*, NeurIPS 2022**        | MAT 把多 agent decision 建成 sequence                                                | 支持 Transformer 可作为 large/variable relational model；**不支持因为你的 obs=30D 就必须使用 Transformer。**([NeurIPS 会议录][3])   |

---

# 3. 本任务与文献的逐点映射

## 3.1 你的任务实际横跨四个不同问题

我建议论文中直接画成：

[
\text{MARL coordination}
\rightarrow
\text{motion command}
\rightarrow
\text{6DOF controller}
\rightarrow
\text{control allocation}
\rightarrow
\text{thrusters}
]

同时存在另一条机械链：

[
\text{Netter}_1
\longleftrightarrow
\text{flexible net}
\longleftrightarrow
\text{Netter}_2
\longleftrightarrow
\text{Prey contact}.
]

现有 pursuit MARL 大量研究第一条链的最左端；Fossen/control-allocation 研究最右端；tether/net manipulation 文献研究第二条链。**真正的研究空缺恰恰是这些链在一个任务里的组合。**

### 已有证据支持

MAPPO、HAPPO、body-relative decentralized observations、hierarchical control、bounded control allocation、role-aware pursuit、herding、tether constraints 分别都有已有依据。([NeurIPS 会议录][2])

### 尚无直接证据支持

以下如果最终实验成功，可以成为你的 contribution，但现在不应该作为预设“事实”：

> “4D subgoal 一定比 direct thruster 更好。”

没有这样的直接证据。

> “不观察 net shape 足以达到 optimal capture。”

也没有。

> “OBI 柔性网可以由两 Netter 间距完全表征。”

同样没有。

更准确的论文表达应该是：

> We hypothesize that stable, robot-centric proxies of the net configuration are sufficient for cooperative decision making, avoiding dependence on simulator-specific deformable-mesh states.

然后通过 ablation 证明。

---

# 4. Observation 与 action 设计评审

## 4.1 30D observation 是否 Markov？

### 严格意义：**不是。**

在完整 Unity dynamics 中，至少有以下 hidden state：

1. 8 个推进器的一阶响应内部状态；
2. OBI 网当前形状、速度、应力、接触状态；
3. 如果用了 PID，PID integrator / derivative filter state；
4. 海流/扰动状态，如果它们是时变而 actor 不观测；
5. Prey controller 内部状态，如果其逃逸策略含 memory/FSM；
6. 车辆的某些姿态信息。

其中第一点对 **8D direct-thruster RL 尤其重要**：

[
\dot T_i =
\frac{T_{i,\mathrm{cmd}}-T_i}{\tau_i}.
]

如果当前 observation 完全一样，但实际 (T_i) 不一样，则下一时刻 acceleration 不一样，因此状态 aliasing 是客观存在的。

### 但工程上可以是“足够好的近似 Markov observation”

尤其在 4D hierarchy 下：

[
a^{HL}
\rightarrow PID
\rightarrow\tau_d
\rightarrow allocator
\rightarrow T,
]

低层闭环会吸收大量未观测 actuator dynamics。

因此我的判断是：

**30D 非常适合作为你的 v1 baseline observation。不要因为严格不 Markov 就先把 observation 扩成 60–100D。**

---

## 4.2 当前 observation 每一类量怎么处理

### self linear/angular velocity：保留

必须保留。

不仅关系自身动力学，而且在没有完整 attitude/history 时提供短时动态状态。

---

### prey relative position：保留

这是 pursuit/herding 的核心状态。

body frame 表示非常合适：

[
{}^B r_P = R_{WB}^\top(p_P-p_i).
]

它天然消除了 global translation/yaw dependence。

---

### prey relative velocity：**强烈建议保留**

不要删。

因为它包含：

* closing velocity；
* lateral velocity；
* target lead；
* LOS rate 的关键信息。

例如 proportional navigation 核心本来就是 closing velocity 和 LOS angular rate。

如果只保留位置，你就是要求神经网络从时间历史自行求速度。

---

### `distance_to_prey`：数学冗余，但可以保留

因为

[
d_P=|r_P|
]

已经能从三个坐标算出来。

所以它：

* **不增加 Markov 信息**；
* 但提供非常好的 handcrafted feature。

30D 本来就很小，没有必要为了减一个维度删除。

可以做 feature-ablation，但不应作为论文重点。

---

### bearing：同样冗余，但建议暂时保留

[
\beta = \operatorname{atan2}(r_z,r_x).
]

它是 position 的确定函数，所以同样不提高 observability。

但 bearing 对 pursuit/herding 具有明确的任务语义。因此作为 inductive bias 可以保留。

真正的问题反而是 angle discontinuity。如果 bearing 允许跨 (-\pi,\pi)，更干净的是：

[
[\sin\beta,\cos\beta]
]

代替单一 angle，不过这会多 1D。

当前实现只要 normalized/wrapped 正确，也不是严重问题。

---

## 4.3 teammate role one-hot 是否保留？

**建议保留。**

即使 fixed slots 已经携带部分语义，它仍很便宜，而且：

* actor trunk 是 shared 的；
* Herder/Netter interaction semantics 不一样；
* 它允许同一个 teammate encoder 明确区分 role。

异构 MARL 本身也有充分理由显式处理 agent heterogeneity，而不是期待 shared policy 自动发现角色。([机器学习研究杂志][27])

不过你的 role 编码是：

```text
[is_herder, is_netter]
```

实际上只有二分类，所以数学上一个 bit 已经足够。

保留 one-hot 的理由主要是工程清晰性，不是信息量。

---

# 4.4 真正值得修改的是 teammate slot 的结构

这是我认为比 Transformer 更值得做的事情。

对一个 Netter：

```text
slot A = Herder
slot B = other Netter
```

有明确语义，concat 完全合理。

但是 Herder 看：

```text
Netter 1
Netter 2
```

这两个物理上应具有交换对称性。

如果仅仅：

[
[\mathrm{N1},\mathrm{N2}]
]

拼接，策略可能学出：

> “slot1 永远去左边，slot2 永远去右边”

而 slot swap 后突然行为变化。

### 推荐 v1.5 architecture

每个 teammate：

[
e_j =
\phi_{\rm team}
(
role_j,
r_j,
v_j
)
]

使用**同一个** MLP encoder。

然后：

Herder:

[
e_{\rm team}
============

\operatorname{mean}(e_{N_1},e_{N_2})
]

或者

[
e_{\rm team}
============

\rho(e_{N_1}+e_{N_2}).
]

这就是简化版 DeepSets。

Netter 因为两个 teammate role 不同，可以：

[
[e_H,e_{N_{\rm other}}].
]

这样比直接给 3-agent problem 上 Transformer 更有明确 inductive bias。

---

# 4.5 是否需要 Transformer？

### 目前：**没有必要作为默认方案。**

Transformer 更合理的触发条件是：

* team size 可变；
* 将来 3→5→10 台 robot；
* neighbour 数量动态变化；
* 多 prey / obstacles / targets；
* assignment 是关键问题；
* agent-agent higher-order relations 很复杂；
* DeepSets/shared-slot encoder 已明显成为瓶颈。

MAT 确实说明多智能体 joint decision 可转成序列建模，但不能从这个结果推导“30D observation 应该 Transformer”。([NeurIPS 会议录][3])

---

# 4.6 是否需要 LSTM/GRU？

这个问题比 Transformer 更值得认真测试。

因为你**明确存在 hidden physical states**。

我建议如下 ablation：

[
\boxed{
\text{MLP}
\quad vs\quad
\text{4-frame stack}
\quad vs\quad
\text{GRU}
}
]

而不是直接：

[
\text{MLP vs Transformer}.
]

### 上 GRU 的可验证触发条件

如果你发现：

> 两个几乎一样的 30D observation，在相同 action 下，经常产生明显不同的 next-state / reward。

而差异又能由过去几帧解释，那么就是典型 temporal partial observability。

尤其：

* net slack vs taut；
* thruster lag；
* prey 正在转向但瞬时状态接近；
* 网刚发生 contact；
* PID 积分接近 saturation。

这时 GRU 非常合理。

---

# 4.7 需不需要加 self orientation？

你的 30D 中有 angular velocity，但没有 roll/pitch orientation。

如果 FinsROV 姿态基本始终接近水平，由低层稳定器强约束，那么可以先不加。

但如果它真的有明显：

[
\phi,\theta
]

运动，那么相同 body velocity 下，由于重力/浮力恢复力不同，未来 dynamics 会不同。

此时最干净的可部署信息不是 world quaternion，而是：

[
\boxed{{}^B\hat g}
]

即 IMU 可得到的 body-frame gravity direction，3D。

它：

* 不泄露 world position；
* 不泄露 absolute yaw；
* sim-to-real 可实现；
* 给出 roll/pitch relative gravity。

我会把它放入 observation ablation，而不是立刻加进主 baseline。

---

# 4.8 不观察 OBI 网格：我支持

你的决定有两个很强的理由。

### 第一，OBI mesh truth 很可能是 simulator-specific privileged state

现实系统不可能轻易稳定获得：

* 每个 particle；
* triangulated mesh；
* instantaneous normal；
* 数十/数百节点速度。

让 actor 依赖这些量会制造很严重的 sim-to-real dependency。

### 第二，柔性网本来就不应该被强行约化成单一 rigid plane

你已经明确指出网：

* 会弯；
* 会折；
* 会波动；
* solver 可能不稳定。

所以类似：

[
\text{net center}+normal+width+height
]

可能根本不是一个稳定 physical state representation。

柔性 cable/tether 文献恰恰说明 slack、tension、attachment constraint 等 mechanical mode 很重要，而不是要求控制器知道整个柔性介质的每个节点。([IEEE Xplore][23])

---

# 4.9 只有在什么时候应该加入 net state？

当你观察到以下现象：

> 几乎完全相同的 robot/prey 30D state，因为 net deformation 不同，而需要完全不同的 action。

比如：

* 状态 A：网绷紧、张开，可以继续捕获；
* 状态 B：两个 Netter 位置完全相似，但是网已经缠绕/松弛；
* 当前 30D 看不出区别。

如果这样的 aliasing 经常发生，才应该加入 net proxy。

### 推荐优先级

**Level 1：无需新增 mesh state**

利用已有：

[
p_{N_2}-p_{N_1}
]

即可得到 Netter separation。

---

**Level 2：稳定的 attachment geometry**

例如：

[
d_A
===

|p^{attach}*{N_1}-p^{attach}*{N_2}|
]

以及

[
\dot d_A.
]

注意这里是 attachment point，不一定等于 vehicle COM。

这是非常好的低维 proxy。

---

**Level 3：attachment tension/load**

如果现实中也计划装 force/tension sensing：

[
F_{\mathrm{net},i}^{body}
]

或者更简单：

[
T_i=|F_{\mathrm{net},i}|.
]

进一步定义带滞回的：

[
z_{\rm net}
\in
{\text{slack},\text{taut},\text{high-load}}.
]

这种表示比网法向稳健得多。

---

**Level 4：contact state**

如果现实捕获系统能检测：

[
c_{\rm prey-net}\in{0,1},
]

可以进入 actor。

否则只应该作为 simulator evaluator。

---

### 不推荐给 actor

```text
OBI particle positions
mesh normals
mesh corners
instantaneous plane fit
prey-to-OBI-mesh perfect distance
OBI constraint internal states
```

除非明确做：

> privileged actor / teacher upper bound

并在论文中标记**不可直接 sim-to-real**。

---

# 4.10 一个更深的问题：真实 capture zone 在哪里？

如果：

> “Herder 把 Prey 赶往两 Netter 和网构成的动态区域”

那么没问题。

两 Netter 本身就在 observation 中。

但如果：

> “环境西侧有一个固定收网区域 X”

actor 不知道 (X)。

这时必须至少提供：

[
{}^B r_{\text{capture-zone}}
]

或其他现实可测等价物。

**这不属于 world-pose 泄漏。**

给 actor：

[
R^\top(p_{\rm goal}-p_i)
]

这样的局部 goal vector，与直接给：

[
p_i^W
]

完全不同。

前者可以由定位/感知模块产生，而且保持 translational invariance。

---

# 4.11 三种 action abstraction 的核心比较

## A. 8D direct thruster

[
\pi(o)
\rightarrow
u_{1:8}\in[-1,1]^8.
]

### 优点

* 最大 expressive power；
* 不依赖 PID；
* 可以学习 hydrodynamic coupling；
* 甚至能补偿 allocator/model error；
* *Learning to Swim* 已证明 thruster-level 6-DOF RL 可以 zero-shot 上真实 AUV。([arXiv][4])

### 缺点

策略同时要学习：

1. cooperative strategy；
2. vehicle stabilization；
3. 6DOF dynamics；
4. actuator allocation；
5. thruster lag；
6. saturation handling。

因此 credit assignment 很差。

Herder 学“站在 prey 后面”时，动作语义却是：

```text
V1 +0.31
V2 -0.14
...
H4 +0.67
```

这显然不利于角色-level strategy reuse。

### sim-to-real

也是三者中对：

* max thrust；
* reverse thrust；
* dead zone；
* thruster time constant；
* installation angle

最敏感的。

**但这些都是工程预期，不是已有 head-to-head 文献证明。**

---

# 4.12 B. 6D direct wrench + allocator

[
\pi(o)
\rightarrow
\tau_d
======

[F_x,F_y,F_z,M_x,M_y,M_z].
]

随后：

[
f^*
===

\arg\min_f
|W(Bf-\tau_d)|_2^2
+
\lambda|f|_2^2
]

subject to

[
f_i^{min}\le f_i\le f_i^{max}.
]

### 最大优点

把 actuator geometry 从 RL 中拿掉。

对于第 (i) 个推进器：

[
b_i
===

\begin{bmatrix}
d_i\
r_i\times d_i
\end{bmatrix},
]

于是

[
B=
[b_1,\ldots,b_8].
]

这是物理上清晰的 generalized-force mapping。

Control-allocation 文献明确支持这种 motion-control/allocation separation。([科学直接][5])

### 但它仍要求 RL 学低层 control

例如：

> 给多大 (F_x) 才会以合适速度接近 prey？

> 当前 yaw rate 应给多少 (M_y) 或相应 yaw torque？

RL 仍需学习水动力闭环。

所以我认为它是最有价值的**中间科学 baseline**。

---

## 必须记录 allocator residual

定义：

[
r_\tau
======

\tau_d-Bf^*.
]

报告：

[
e_\tau
======

\frac{|Wr_\tau|_2}
{|W\tau_d|_2+\epsilon}.
]

这是极其重要的指标。

因为：

> policy 命令了 wrench

不代表：

> robot 实现了这个 wrench。

特别当：

* 多个 thruster 饱和；
* wrench 方向物理不可达；
* (B) 条件数差；

二者差距会很大。

---

# 4.13 C. 4D body subgoal + PID + allocator

[
\pi(o)
\rightarrow
[
\Delta x_B,
\Delta y_B,
\Delta z_B,
\Delta\psi
].
]

然后：

[
\text{PID}
\rightarrow \tau_d
\rightarrow CA
\rightarrow f.
]

### 我推荐它作为论文主方法

原因是它把学习问题变成：

> **“我要去哪？”**

而不是：

> “此刻 H3 应该产生多少牛顿？”

Herder 可以学习：

> 去 prey 后侧 1.5 m。

Netter 可以学习：

> 向左展开 0.4 m，同时保持前方截获。

这些 action 在不同质量、不同推进器布置下仍保留相近任务语义。

这正是 hierarchical control 的价值。

近期水下 open-source 工作也开始显式使用 velocity-level action abstraction，说明高层 cooperative decision 与低层 6-DOF execution 解耦是活跃方向。([arXiv][15])

### 但不要把它写成“文献已证明更高效”

正确说法应是：

> **工程假设 H1：**更具任务语义且维度更低的 body-relative subgoal action 会降低 cooperative exploration 与 credit-assignment 难度。

然后用实验检验。

---

## 三者总结

| 属性                       | 8D thruster | 6D wrench+CA | 4D subgoal+PID+CA |
| ------------------------ | ----------: | -----------: | ----------------: |
| MARL action dim          |           8 |            6 |             **4** |
| 学 actuator geometry      |      **需要** |          不需要 |               不需要 |
| 学 vehicle stabilization  |      **需要** |       **需要** |            大部分不需要 |
| 动作任务语义                   |           低 |            中 |             **高** |
| allocator saturation 显式  |           否 |        **是** |             **是** |
| low-level 可解释性           |           低 |            高 |            **最高** |
| actuator-layout transfer |           差 |          中/好 |          **最好预期** |
| 最大 maneuver expressivity |      **最高** |            高 |                较低 |
| 对 PID tuning 依赖          |           无 |            无 |             **有** |
| net load 下适应潜力           |           高 |            高 |           取决于 PID |
| 作为协作论文主方案                |          一般 |  很好 ablation |           **最推荐** |

后三列中的相对优劣除明确引用文献的部分外，应当视为**待实验验证的工程假设**。

---

# 5. Baseline 与实验矩阵

我建议把实验拆成两个问题，避免审稿时解释不清。

---

## 5.1 Study A：协作算法比较

所有算法：

[
a^{HL}
======

[\Delta x,\Delta y,\Delta z,\Delta\psi]
]

然后统一：

[
PID+CA.
]

### Learned baselines

**B1. IPPO**

actor observation 相同。

无 centralized critic。

目的非常清楚：

[
\boxed{\text{CTDE 到底有没有贡献？}}
]

MAPPO 论文的系统 benchmark 也使 IPPO/MAPPO 成为自然 pair。([NeurIPS 会议录][2])

---

**B2. MAPPO**

你的主基线：

* shared trunk；
* role-aware；
* role-specific heads；
* centralized critic。

---

**B3. HAPPO**

我认为值得做。

尤其如果实现为：

* Herder policy；
* Netter policy；
* sequential heterogeneous update。

它直接测试：

> “角色差异是否足以使 homogeneous-style shared MAPPO 不合适？”

HAPPO/HARL 有明确的异构-agent 理论和 benchmark 背景。([OpenReview][26])

---

**B4. RMAPPO，可选**

只有在 MLP partial-observation diagnostics 明确显示 history 有价值后再升级为正式 baseline。

---

**MADDPG**

可以作为历史性 continuous-action MARL baseline，但如果计算预算有限，我把优先级排在：

[
IPPO,\ MAPPO,\ HAPPO
]

之后。MADDPG 的贡献主要是 CTDE 奠基，而不是今天最公平的 PPO-family comparator。([NeurIPS 会议录][25])

---

# 5.2 Study B：动作抽象比较

全部使用：

* 相同 MAPPO；
* 相同 observation；
* 相同 network capacity 尽可能匹配；
* 相同 training steps；
* 相同 curriculum；
* 相同 initial-state seeds。

比较：

### C1

[
MAPPO\rightarrow8D\ thrusters.
]

### C2

[
MAPPO\rightarrow6D\ wrench\rightarrow CA.
]

### C3

[
MAPPO
\rightarrow4D\ body\ subgoal
\rightarrow PID
\rightarrow CA.
]

这是我认为你的论文中**最有价值的一张主结果表**。

---

# 5.3 Traditional role-aware geometric baseline

建议不要做一个“所有机器人 pure pursuit”然后说传统算法很差。

应该做一个真正公平的：

[
\boxed{\text{Role-aware Geometric FSM}}
]

仍然输出同样：

[
[\Delta x_B,\Delta y_B,\Delta z_B,\Delta\psi].
]

---

## 状态机

```text
APPROACH
    ↓
SPREAD_NETTERS
    ↓
HERDER_DRIVE
    ↓
CAPTURE / HOLD
```

---

## APPROACH

令两个 Netter attachment/robot midpoint：

[
m_N=
\frac{p_{N_1}+p_{N_2}}{2}.
]

Prey 速度方向：

[
e_P=
\frac{v_P}{|v_P|+\epsilon}.
]

Netter 可选一个预测 intercept midpoint：

[
m_N^*
=====

p_P+
T_Iv_P.
]

不要在这里假设网有 rigid plane。

这里只定义：

> 两个 carrier 的 midpoint 去哪里。

---

# 5.4 SPREAD_NETTERS

定义 endpoint spacing：

[
d_N
===

|p_{N_1}-p_{N_2}|.
]

期望：

[
d_N^*.
]

然后 Netter controller 追踪：

[
e_d=d_N-d_N^*.
]

如果要生成左右展开方向，可以用 pursuit direction 与 gravity 构造：

[
e_l
===

\frac{\hat g\times e_{\rm approach}}
{|\hat g\times e_{\rm approach}|}.
]

于是：

[
p^*_{N_1}
=========

m_N+\frac{d_N^*}{2}e_l
]

[
p^*_{N_2}
=========

m_N-\frac{d_N^*}{2}e_l.
]

注意：这里定义的是 **Netter endpoint geometry**。

不是：

> “网是一个法向为 (n) 的刚性平面”。

两者区别很重要。

---

# 5.5 HERDER_DRIVE

如果希望 Prey 朝 Netter midpoint 逃：

[
e_{P\rightarrow N}
==================

\frac{m_N-p_P}
{|m_N-p_P|}.
]

Herder 应该去 Prey 的相反侧：

[
\boxed{
p_H^*
=====

p_P-d_H e_{P\rightarrow N}
}
]

因为：

* midpoint 在 prey 前面；
* Herder 在 prey 后面；
* prey 逃离 Herder；
* 因而大致朝 midpoint 运动。

这比：

[
p_H^*=p_P
]

即 pure pursuit 更符合 herding。

CBF herding 工作恰恰支持这种“通过 shepherd position 诱导非合作对象运动”的控制思想。([arXiv][9])

---

# 5.6 CAPTURE/HOLD

不要：

```text
if distance(prey, netter) < threshold:
    success
```

正式 baseline 和正式 eval 必须与 RL 一致。

例如：

[
c_{\rm contact}=1
]

持续 (T_h)，或者：

[
d_{\rm prey,actual\ deformable\ net}
<
d_{\rm cap}
]

持续 (T_h)。

即使 FSM 内部仍然只根据 robot geometry 决策，**终止条件必须是真网捕获**。

---

# 5.7 Pure pursuit / LOS / PN / APF 怎么放？

### Pure pursuit

作为 lower bound 很好：

[
p_i^*=p_P.
]

预期 Netter 会聚在一起、失去 spacing。

所以它回答的是：

> “没有角色协作会怎样？”

而不是最强传统 baseline。

---

### LOS

LOS 是**path guidance**，不是三机器人 cooperative capture strategy。

正确用法：

> FSM 产生 waypoint/path → LOS 执行该 path。

3D ALOS 有成熟理论依据。([科学直接][18])

---

### Proportional Navigation

适合作为单 pursuer/interceptor comparator。

经典形式可写成：

[
a_c
===

N V_c\dot\lambda.
]

因为你的 observation 有 relative velocity，所以能构造：

* closing speed；
* LOS rate。

不过它不解决：

* Netter spacing；
* Herder behavior；
* flexible net capture。

因此应定位为**interception primitive**，而不是完整 team baseline。

---

### APF

值得做。

例如：

[
F_i=
F_{\rm prey}
+
F_{\rm teammate}
+
F_{\rm boundary}.
]

Netter 可以额外加入 spacing potential：

[
U_s
===

\frac12k_s(d_N-d_N^*)^2.
]

Zhang ICRA 2022 已直接证明 APF-enhanced RL 对 cooperative pursuit 有意义。([IEEE Xplore][6])

---

### Formation control

这是 Netter 最自然的 classical component：

[
\text{midpoint tracking}
+
\text{relative distance regulation}.
]

我甚至认为：

> FSM + Herder geometric law + Netter formation controller

应该是你的 strongest classical baseline。

---

### CBF

CBF 最适合解决：

* inter-robot collision；
* net over-extension；
* workspace boundary；
* minimum/maximum spacing；

而不是直接完成捕获策略。

因此可以作为未来的 **safety filter**：

[
u^*
===

\arg\min_u
|u-u_{RL}|^2
]

subject to

[
\dot h+\alpha(h)\ge0.
]

但 v1 不建议同时加进去，否则贡献线太多。

---

# 6. Reward、课程、指标和统计设计

# 6.1 最终成功定义

你这里一定要非常严格。

推荐论文正式定义：

[
S_{\rm capture}
===============

S_{\rm collision}
\lor
S_{\rm sustained-net-contact}.
]

例如：

[
S_{\rm sustained-net-contact}
=============================

\mathbf 1
\left[
d(P,\mathcal N_t)<d_c
\quad
\forall t\in[t_0,t_0+T_h]
\right],
]

其中：

[
\mathcal N_t
]

是真实瞬时**变形**网面。

不是 nominal plane。

---

## 如果 closest surface distance 不稳定怎么办？

那么：

[
\boxed{\text{collision/contact event 优先}}
]

可以组合：

1. Prey 与 OBI collision/contact；
2. contact duration；
3. closest mesh distance 仅作诊断。

不要为了获得一个漂亮连续量，反过来把网拟合成刚体。

---

# 6.2 reward 最重要的原则

## 真成功奖励应 shared

[
r_i^{success}
=============

R_{cap},
\qquad
\forall i.
]

因为最终任务是 team capture。

---

## role reward 只负责 shaping

可以：

[
r_i=
r_{\rm team}
+
\lambda_{\rm role}r_i^{role}
----------------------------

r_{\rm control}.
]

不要让：

[
r_i^{role}
]

压倒真正捕获目标。

否则非常容易出现：

> Herder 永远保持漂亮的“behind position”，但没人真正收网。

---

# 6.3 Herder shaping

定义：

[
e_{PN}
======

\frac{m_N-p_P}
{|m_N-p_P|}.
]

Herder behind alignment：

[
q_H
===

\frac{p_P-p_H}
{|p_P-p_H|}
\cdot
e_{PN}.
]

理想：

[
q_H\rightarrow1.
]

于是可以：

[
r_H^{align}
===========

k_H q_H.
]

同时促进 prey toward net：

[
r^{progress}
============

k_p
[
d(P_t,m_{N,t})
--------------

d(P_{t+1},m_{N,t+1})
].
]

最好使用**progress difference**，而不是一直奖励“小距离”，否则容易 stationary exploitation。

---

# 6.4 Netter shaping

spacing：

[
r_N^{spacing}
=============

-k_s|d_N-d_N^*|.
]

relative-speed：

[
r_N^{relvel}
============

-k_v
|
v_{N_1}-v_{N_2}
|.
]

midpoint intercept：

[
r_N^{mid}
=========

-k_m
|m_N-m_N^*|.
]

这些都不需要任何 rigid net-plane assumption。

---

# 6.5 control reward

建议统一所有 learned policies：

[
r_u
===

-\lambda_u|u|^2
]

以及：

[
r_{\Delta u}
============

-\lambda_{\Delta u}
|u_t-u_{t-1}|^2.
]

不过三类 action abstraction 的 action norm 没法直接公平比较：

* thruster norm；
* wrench norm；
* subgoal norm；

语义完全不同。

因此**最终能耗必须在物理 thruster 输出层统一计算**：

[
J_E^{(1)}
=========

\int
\sum_i |T_i|,dt
]

或者：

[
J_E^{(2)}
=========

\int
\sum_i T_i^2,dt.
]

这两个只能称：

> thrust-effort / energy proxy

除非你有实际：

[
P_i(n_i,T_i)
]

电功率模型，才应该称 energy consumption。

---

# 6.6 allocator penalty 要慎用

可以记录：

[
e_\tau.
]

训练中也可以轻微 penalty：

[
r_\tau
======

-\lambda_\tau
|W(\tau_d-Bf)|^2.
]

但我建议第一版：

> **先只作为 metric，不加入 reward。**

否则 6D/4D policies 会被鼓励只请求小 wrench，产生“懒惰策略”。

---

# 6.7 Curriculum

我建议不是单纯调 capture radius，而是分物理能力阶段。

### Stage 0：低层系统验证

不做 MARL。

```text
dx step
dy step
dz step
yaw step
```

确认：

[
4D\ target
\rightarrow PID
\rightarrow wrench
\rightarrow allocator
\rightarrow motion
]

是稳定的。

---

### Stage 1：Netter formation

Prey stationary 或极慢。

任务：

* 两 Netter 展开；
* 保持 spacing；
* midpoint 接近 Prey。

不追求捕获。

---

### Stage 2：Herder geometry

固定 Netter。

Herder 学：

[
\text{get behind prey}
\rightarrow
\text{drive prey toward netter midpoint}.
]

---

### Stage 3：early capture proxy

允许你目前说的：

[
d(P,N_i)<d_{\rm proxy}
]

用于：

* curriculum promotion；
* 临时 dense reward；
* 或 early termination。

但：

[
\boxed{
\lambda_{\rm proxy}
\rightarrow0
}
]

必须逐渐 anneal。

它不进入最终论文主 metric。

---

### Stage 4：真实 flexible-net capture

切换为：

[
S_{\rm net}.
]

这时 proximity 不再算成功。

---

### Stage 5：更强 Prey

逐步提高：

* speed；
* turn rate；
* reactive evasiveness。

DualCL 的结果支持逐渐提高 pursuit 难度这一大方向。([arXiv][10])

---

### Stage 6：DR

建议随机化：

[
m,\ I,\ C_D,\ C_A,
]

[
T_{\max,i},
\quad
\tau_{\rm thruster},
]

[
CoB-CoM,
]

sensor：

[
noise,\ delay,
]

以及有实测依据时的：

[
k_{\rm net},c_{\rm net}.
]

但不要做：

> ±50% 所有参数

然后称“物理鲁棒”。

DR range 最好来自 calibration uncertainty。

*Learning to Swim* 已展示物理参数 DR 可提升 AUV policy 对模型变化的鲁棒性，但这不等于任意宽 DR 都合理。([arXiv][4])

---

# 6.8 主评价指标

我建议明确 primary/secondary。

## Primary metric 1

[
\boxed{
P_{\rm capture}
}
]

其中 capture 必须是实际 flexible net definition。

这是论文第一指标。

---

## Primary metric 2：capture time

不要只对成功 episode 求：

[
E[T|success].
]

因为这会产生 survivor bias。

一个算法：

* 只成功 30%；
* 但成功时 4 秒；

可能看起来比：

* 成功 95%；
* 平均 7 秒

“更快”。

更严谨的是把未捕获 episode 视作 right-censored data，并报告：

* Kaplan–Meier curve；
* 或 restricted mean capture time (RMST)。

如果不想把论文搞得太统计学，也至少同时报告：

[
(success\ rate,\ conditional\ capture\ time).
]

---

# 6.9 Net-specific metrics

至少：

### 最小真实网面距离

[
d_{\min}^{net}
==============

\min_t d(P_t,\mathcal N_t).
]

### capture/contact duration

[
T_{\rm contact}.
]

### Netter spacing error

[
E_s
===

\frac1T
\int
|d_N(t)-d_N^*|dt.
]

再报告最大 deviation：

[
E_s^{max}.
]

---

# 6.10 Role metrics

Herder：

[
q_H
]

behind alignment；

[
v_P\cdot e_{P\rightarrow N}
]

即 prey toward-net velocity。

Netter：

* spacing；
* midpoint-to-prey；
* endpoint relative velocity。

不要只报告 return。

这些 physical behavior metrics 才能解释：

> 为什么 algorithm capture 成功。

---

# 6.11 Low-level / allocator metrics

这是你的论文很容易比普通 MARL pursuit 做得更扎实的部分。

每个 episode 记录：

### Saturation rate

[
S_i
===

\frac{
#{|T_i|>0.99T_i^{max}}
}
{N}.
]

以及：

[
S_{\rm any}
===========

P(\exists i \text{ saturated}).
]

---

### Wrench residual

[
e_\tau.
]

---

### PID tracking

[
e_p
===

p_{goal}-p.
]

[
e_\psi.
]

---

### control smoothness

物理 thruster：

[
J_{\Delta T}
============

\frac1T
\sum_t
|T_t-T_{t-1}|^2.
]

最好不要直接比较 high-level (a_t) smoothness，因为三种 action 语义不同。

---

### thrust effort

[
J_T
===

\int\sum_i|T_i|dt
]

和/或：

[
J_{T^2}
=======

\int\sum_iT_i^2dt.
]

---

# 6.12 DR robustness

不要只给：

> nominal success / randomized success。

更好的图是：

[
P_{\rm capture}(\alpha)
]

其中：

[
\alpha
======

\text{perturbation severity}.
]

例如：

```text
0%
10%
20%
30%
```

分别对：

* mass；
* drag；
* max thrust；
* delay；
* net stiffness。

这样能看 robustness degradation curve。

---

# 6.13 统计设计

强化学习实验最常见的问题是：

> 跑了 3 个 seed，取最好的一个。

不要这样。

我的建议：

### 训练

如果算力允许：

[
\boxed{10\ independent\ training\ seeds}
]

是很好的目标。

至少：

[
5
]

但对于论文关键方法，10 更可信。

---

### Evaluation

每个 training seed 用**完全相同的一组 evaluation scenarios**。

例如：

[
500\sim1000
]

episode/seed。

所有算法共享：

```text
prey seed
initial poses
current realization
noise realization
DR realization
```

这样形成 paired comparison。

---

### 真正统计单位

不是每一个 episode 都等同一个独立训练样本。

应认识到：

[
episode
\subset
trained\ policy\ seed.
]

所以不要把：

[
10\times1000=10000
]

个 episode 当成 10000 个独立 policy samples，然后获得虚假的 (p<10^{-20})。

---

### 推荐报告

对 return/physical metric：

* median；
* mean；
* **IQM**；
* 95% bootstrap CI。

对 success：

* per-seed success；
* pooled paired/hierarchical bootstrap CI。

对 main method comparison 可以做 paired test。

多个 ablation 时用 Holm correction。

RL 领域对少 seed 和过度依赖单一平均值的统计问题已有系统讨论；Agarwal et al. 的 *Deep Reinforcement Learning at the Edge of the Statistical Precipice* 和 Henderson et al. 的 *Deep Reinforcement Learning That Matters* 是很值得遵循的实验规范。

---

# 6.14 我建议的 ablation 矩阵

不要全排列，否则爆炸。

## A. Action abstraction

| ID | Actor action        | PID | CA |
| -- | ------------------- | --- | -- |
| A1 | 8D thruster         | ×   | ×  |
| A2 | 6D wrench           | ×   | ✓  |
| A3 | **4D body subgoal** | ✓   | ✓  |

**最高优先级。**

---

## B. Temporal observation

| ID | obs                   |
| -- | --------------------- |
| B1 | 30D MLP               |
| B2 | 30D + previous action |
| B3 | 30D GRU               |

如果 thruster direct：

previous action 很重要。

如果 4D hierarchy：

GRU 可能收益较小，这本身就是有意义的结果。

---

## C. Net information

| ID | actor net state                    |
| -- | ---------------------------------- |
| C1 | **none**                           |
| C2 | attachment separation + rate       |
| C3 | + tension/slack proxy              |
| C4 | privileged OBI state，仅 upper-bound |

这组实验可以非常有说服力地支撑：

> dense mesh observation is unnecessary / unstable.

但只有结果真的支持时才能这么写。

---

## D. Representation

| ID | teammate handling                              |
| -- | ---------------------------------------------- |
| D1 | concat slots                                   |
| D2 | shared slot encoder                            |
| D3 | shared encoder + permutation-invariant pooling |

我把这个放在 Transformer 之前。

---

## E. CTDE/heterogeneity

| ID | training                     |
| -- | ---------------------------- |
| E1 | IPPO                         |
| E2 | MAPPO                        |
| E3 | role-head MAPPO              |
| E4 | HAPPO / separate-role policy |

---

# 7. 推荐实施路线与风险

## 7.1 v1：我认为最小、最有希望发表的方案

### Step 0：冻结 physics interface

在任何 MARL comparison 前完成：

[
\boxed{
B,\ PID,\ thruster\ calibration
}
]

审计。

---

### Step 1：建立 physical allocator

对每个 thruster：

[
B_i=
\begin{bmatrix}
d_i\
r_i\times d_i
\end{bmatrix}.
]

其中：

[
r_i
===

p_{thruster,i}-p_{CoM}.
]

必须按你 Unity 的：

```text
x forward
y up
z left
```

重新计算叉乘和 torque signs。

不要从“经验 mixer table”逆推出 (B)。

---

### Step 2：bounded least-squares allocator

第一版甚至不用复杂 QP。

做：

[
\min_f
|W(Bf-\tau_d)|_2^2
+
\lambda|f|_2^2
]

s.t.

[
f^{min}\le f\le f^{max}.
]

这是非常成熟的 control-allocation formulation；AUV 文献也明确考虑 saturation-constrained allocation。([科学直接][5])

1024 parallel environments 下，不必给每个 env 起通用 CPU optimization solver。

因为：

* 只有 8 variables；
* (B) 固定；
* bounds 固定或缓慢变化。

可以做 batched projected solver / bounded least squares / active-set specialization。后续再优化。

---

### Step 3：完成 4D PID interface

动作尺度固定且写进论文，例如：

[
\Delta x
========

a_x d_x^{max},
]

[
\Delta y
========

a_y d_y^{max},
]

[
\Delta z
========

a_z d_z^{max},
]

[
\Delta\psi
==========

a_\psi\psi^{max}.
]

并固定整个实验。

不要随着不同 algorithm 调不同：

[
d^{max}.
]

否则 action-space comparison 不公平。

---

### Step 4：30D MAPPO，不加 net state

这应该是你的 v1 主模型。

网络先用普通 MLP。

角色：

* shared backbone；
* Herder/Netter heads。

这是最容易解释的。

---

### Step 5：建立真正的网成功 evaluator

必须独立于 reward shaping。

输入：

[
Prey,\ actual\ Obi\ net.
]

输出至少：

```text
net_contact
closest_deformed_net_distance
contact_duration
capture_success
```

如果 closest distance numerical stability 不够：

> collision/contact 是 authoritative success；

distance 只做 auxiliary diagnostic。

---

### Step 6：三个主 action baseline

这是第一轮真正值得跑大规模 seed 的实验：

[
8D
\quad vs\quad
6D
\quad vs\quad
4D.
]

---

# 7.2 最值得做的五个 ablation

如果只能选五个：

### 1. **8D vs 6D vs 4D**

最重要。

直接回答：

> low-level abstraction 是否改善 cooperative MARL？

---

### 2. **MAPPO vs IPPO**

回答：

> centralized training 是否真正有贡献？

---

### 3. **30D MLP vs GRU**

回答：

> flexible-net/thruster dynamics 引起的 partial observability 是否需要 memory？

---

### 4. **no-net-state vs stable-net-proxy**

例如：

```text
none
vs
attachment separation/rate
vs
+tension
```

回答你最具特色的问题：

> 到底需要观察多少 flexible-net state？

---

### 5. **role-aware vs role-agnostic**

例如：

```text
shared single head
vs
role input
vs
role-specific heads
```

回答异构性究竟是不是性能来源。

如果还有余力，第六个再做：

> concat teammates vs permutation-invariant teammate encoder。

---

# 7.3 不建议 v1 做的事情

我暂时不会做：

* Transformer；
* graph neural network；
* learned low-level controller；
* CBF safety layer；
* MPC low-level；
* opponent self-play；
* online adaptation；
* world-coordinate critic/actor 大量 privileged features；
* OBI mesh encoder。

不是因为这些不好，而是一次加进去：

> 你再也不知道性能究竟来自哪里。

---

# 7.4 关键 failure modes 与诊断

## Failure 1：RL return 上升，但真实 capture 不升

首先查：

[
R_{\rm shaping}/R_{\rm success}.
]

很可能 reward hacking：

* Herder 保持漂亮位置；
* Netters spacing 很好；
* 但 Prey 从未接触网。

日志必须把：

```text
team return
true capture
proxy capture
```

完全分开。

---

## Failure 2：early curriculum 100%，full net 几乎 0%

这说明：

[
\text{UUV proximity}
\not\Rightarrow
\text{net capture}.
]

这是你尤其应该防止的。

建议记录 confusion matrix：

|               | true net capture | no net capture |
| ------------- | ---------------: | -------------: |
| proxy success |               TP |         **FP** |
| proxy failure |               FN |             TN |

重点看：

[
FP_{\rm proxy}.
]

如果很高，证明 early proxy 不能无限沿用。

---

## Failure 3：Netter 位置正确但网无法捕获

记录：

* attachment separation；
* net tension；
* solver stretch violation；
* max particle velocity；
* contact count；
* self-intersection 若可获得；
* attachment force；
* Netter relative angular orientation。

这通常说明：

> robot geometry 不等于 net configuration。

这正是 net-state ablation 的依据。

---

## Failure 4：4D policy 看似很差

不要马上归咎 MAPPO。

检查：

[
a^{HL}
\rightarrow goal
\rightarrow PID
\rightarrow\tau_d
\rightarrow\tau_{actual}.
]

逐层记录。

尤其：

[
e_{\rm PID},
\quad
e_\tau,
\quad
saturation.
]

否则可能其实是：

> RL 给的 goal 很合理，但 allocator 完全实现不了 PID wrench。

---

## Failure 5：6D wrench policy 一直要求不可能的 wrench

画：

[
\tau_d
\quad vs\quad
\tau_{achieved}.
]

如果 residual 大量集中在某些 DOF：

> 要么 action scaling 不合理；
> 要么 (B) 在该自由度 authority 很弱。

---

## Failure 6：8D direct thruster 高频抖动

记录：

[
\Delta T,
\quad
\Delta^2 T
]

以及 thrust first-order state。

可能是：

* action frequency 太高；
* reward 缺 smoothness；
* actuator lag 未观察；
* PPO 对延迟系统产生 oscillation。

---

## Failure 7：Herder 和 Netter 角色坍缩

记录 role-conditioned action distribution：

[
p(a|H),\quad p(a|N).
]

以及：

* average prey distance；
* behind score；
* spacing score；
* Netter midpoint behavior。

如果几乎相同，role heads 没真正产生 specialization。

---

# 7.5 强烈推荐做 observation aliasing diagnostic

这是一个很有研究味道、也很实用的实验。

从 replay/log 中找：

[
|o_i-o_j|<\epsilon
]

但：

[
|s'_{i}-s'_j|
]

非常大的 pair。

然后检查 divergent pair 是否对应：

* net tension 不同；
* thruster states 不同；
* PID integrator 不同；
* net contact 不同。

如果结果显示：

[
I(s_{hidden};s'|o)
]

很明显，那么你就有实验证据说明：

> 当前 observation 存在物理意义上的 perceptual aliasing。

这比：

> “POMDP 所以用了 LSTM”

要严谨很多。

---

# 7.6 在论文写作前必须完成的物理审计

## A. 坐标系审计

逐个 thruster 单独：

[
u_i=+1
]

和：

[
u_i=-1.
]

记录产生：

[
[F_x,F_y,F_z,M_x,M_y,M_z].
]

与：

[
B_i=
[d_i;r_i\times d_i]
]

逐列比较。

只有完成这个，你才能写：

> “physically calibrated thrust allocation matrix”。

---

## B. 单轴 wrench test

分别：

```text
+Fx / -Fx
+Fy / -Fy
+Fz / -Fz
+Mx / -Mx
+My / -My
+Mz / -Mz
```

记录：

* achieved wrench；
* acceleration；
* unintended cross-axis motion；
* saturation。

---

## C. PID step test

分别测试：

[
\Delta x,\Delta y,\Delta z,\Delta\psi.
]

报告：

* rise time；
* settling；
* overshoot；
* steady-state error；
* cross coupling。

还要分别：

1. 无网；
2. 网 attached；
3. 网明显有 tension。

如果第三种下 PID 崩掉，那么 4D hierarchy 的假设本身就不成立。

---

## D. Thruster identification

你论文里如果写“一阶 thruster response”：

[
\tau\dot T+T=K u,
]

就必须至少知道：

* (K)；
* (\tau)；
* forward/reverse asymmetry；
* deadzone；
* saturation。

否则只能称：

> simulated first-order thruster model

而不能声称：

> calibrated physical thruster dynamics。

---

## E. Net evaluator validation

人工制造至少：

1. 网完全张开；
2. 网弯曲；
3. 网严重下垂；
4. 网侧向卷曲；
5. Prey 接近 Netter 但未碰网；
6. Prey 穿过网；
7. Prey 在网旁擦过；
8. solver 瞬间振荡。

验证：

[
S_{\rm capture}
]

没有明显 false positives。

这一项对你的论文比再加一个 MARL algorithm 更重要。

---

# 7.7 我最终建议的论文主线

如果让我帮你把整个项目压缩成一条清晰 scientific story，我会定义成：

> **Can heterogeneous multi-robot capture with a deformable net be learned effectively using task-space decentralized policies, without exposing simulator-specific deformable-mesh states to the actors?**

然后三个子问题：

### RQ1 — Action abstraction

[
8D\ thruster
;vs;
6D\ wrench
;vs;
4D\ task\ space.
]

### RQ2 — Flexible-net observability

[
robot\ geometry
;vs;
stable\ net\ proxies
;vs;
privileged\ mesh.
]

### RQ3 — Heterogeneous coordination

[
IPPO
;vs;
MAPPO
;vs;
HAPPO/role-aware MAPPO.
]

这三条是彼此正交、且都能通过实验清楚回答的。

相比之下：

> “提出 Transformer + GRU + GNN + CBF + MPC + domain randomization 的新算法”

反而会削弱论文。

---

# 8. 完整参考文献列表

下面只列我本轮能够核实的核心条目；2026 arXiv 条目明确标为预印本。

### Pursuit / herding / underwater MARL

1. **Zhang, Z., Wang, X., Zhang, Q., Hu, T.** “Multi-robot Cooperative Pursuit via Potential Field-Enhanced Reinforcement Learning.” *2022 IEEE International Conference on Robotics and Automation (ICRA)*, 2022. DOI: **10.1109/ICRA46639.2022.9812083**. ([IEEE Xplore][6])

2. **Wang, C., Chen, H., Pan, J., Zhang, W.** “Encirclement Guaranteed Cooperative Pursuit with Robust Model Predictive Control.” *2021 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2021. arXiv:2108.07445. ([arXiv][7])

3. **Kouzeghar, M., Song, Y., Meghjani, M., Bouffanais, R.** “Multi-Target Pursuit by a Decentralized Heterogeneous UAV Swarm using Deep Multi-Agent Reinforcement Learning.” *ICRA 2023*, pp. 3289–3295. arXiv:2303.01799. ([arXiv][8])

4. **Mohanty, N., Grover, J., Liu, C., Sycara, K.** “Distributed Multirobot Control for Non-Cooperative Herding.” *International Symposium on Distributed Autonomous Robotic Systems (DARS)*, 2022, pp. 317–332; arXiv:2301.03293. ([arXiv][9])

5. **Chen, J., Li, G., Yu, C., Yang, X., Xu, B., Yang, H., Wang, Y.** “A Dual Curriculum Learning Framework for Multi-UAV Pursuit-Evasion in Diverse Environments.” *AAAI 2024*. arXiv:2312.12255. ([arXiv][10])

6. **冯育凯，吴正兴，谭民.** “基于MARL-MHSA架构的水下仿生机器人协同围捕策略: 数据驱动建模与分布式策略优化.” *自动化学报*, 2025, 51(10):2269–2282. DOI: **10.16383/j.aas.c250086**. ([AAS][11])

7. **Feng, Y., et al.** “Decentralized Multirobotic Fish Pursuit Control With Attraction-Enhanced Reinforcement Learning.” *IEEE Transactions on Industrial Electronics*, 72(8), 2025:8290–8300. ([IEEE Xplore][12])

8. **Feng, Y., et al.** “M²GRPO: Mamba-based Multi-Agent Group Relative Policy Optimization for Biomimetic Underwater Robots Pursuit.” arXiv:2604.19404, 2026. **预印本。** ([arXiv][13])

9. **Gavin, T., Bronz, M.** “Intercepting an Agile Target with Net-Carrying Drones using Competitive Multi-Agent Reinforcement Learning.” *2026 International Conference on Unmanned Aircraft Systems (ICUAS)*. DOI: **10.1109/ICUAS69441.2026.11598610**; arXiv:2607.05939. ([arXiv][1])

10. **Masmitja, I., Martin, M., O’Reilly, T., Kieft, B., Palomeras, N., Navarro, J., Katija, K.** “Dynamic Robotic Tracking of Underwater Targets Using Reinforcement Learning.” *Science Robotics*, 8(80), eade7811, 2023. DOI: **10.1126/scirobotics.ade7811**. ([科学杂志][14])

11. **Zhu, S., Han, G., Lin, C., He, Y.** “Task-Semantic Graph-Driven Distributed Agent Networking for Underwater Target Tracking.” arXiv:2605.15528, 2026. **预印本；代码公开于作者 MARL-AUV repository。** ([arXiv][15])

### 6-DOF control / allocation / guidance

12. **Cai, L., Chang, K., Girdhar, Y.** “Learning to Swim: Reinforcement Learning for 6-DOF Control of Thruster-driven Autonomous Underwater Vehicles.” *2025 IEEE International Conference on Robotics and Automation (ICRA)*, pp.11286–11293. DOI: **10.1109/ICRA55743.2025.11128688**; arXiv:2410.00120. ([arXiv][29])

13. **Johansen, T. A., Fossen, T. I.** “Control Allocation—A Survey.” *Automatica*, 49(5), 2013:1087–1103. DOI: **10.1016/j.automatica.2013.01.035**. ([科学直接][5])

14. **Johansen, T. A., Fuglseth, T. P., Tøndel, P., Fossen, T. I.** “Optimal Constrained Control Allocation in Marine Surface Vessels with Rudders.” *Control Engineering Practice*, 16(4), 2008:457–464. DOI: **10.1016/j.conengprac.2007.01.012**. ([科学直接][16])

15. **Yuan, C., Shuai, C., Ma, J., et al.** “An Efficient Control Allocation Algorithm for Over-actuated AUVs Trajectory Tracking with Fault-Tolerant Control.” *Ocean Engineering*, 273, 113976, 2023. DOI: **10.1016/j.oceaneng.2023.113976**. ([科学直接][17])

16. **Fossen, T. I., Aguiar, A. P.** “A Uniform Semiglobal Exponential Stable Adaptive Line-of-Sight (ALOS) Guidance Law for 3-D Path Following.” *Automatica*, 163, 111556, 2024. ([科学直接][18])

17. **Yan, Z., Gong, P., Zhang, W., Wu, W.** “Model Predictive Control of Autonomous Underwater Vehicles for Trajectory Tracking with External Disturbances.” *Ocean Engineering*, 217, 107884, 2020. DOI: **10.1016/j.oceaneng.2020.107884**. ([科学直接][19])

18. **Havenstrøm, S. T., Sterud, C., Rasheed, A., San, O.** “Proportional Integral Derivative Controller Assisted Reinforcement Learning for Path Following by Autonomous Underwater Vehicles.” arXiv:2002.01022, 2020. ([arXiv][20])

### Flexible net / tether / cooperative manipulation

19. **Klausen, K., Fossen, T. I., Johansen, T. A.** “Autonomous Recovery of a Fixed-Wing UAV Using a Net Suspended by Two Multirotor UAVs.” *Journal of Field Robotics*, 35(5), 2018:717–731. DOI: **10.1002/rob.21772**. ([Wiley Online Library][21])

20. **Li, G., Loianno, G.** “Nonlinear Model Predictive Control for Cooperative Transportation and Manipulation of Cable Suspended Payloads with Multiple Quadrotors.” *IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, 2023. ([IEEE Xplore][22])

21. **Yang, C., et al.** “Collaborative Navigation and Manipulation of a Cable-Towed Load by Multiple Quadrupedal Robots.” *IEEE Robotics and Automation Letters*, 7(4), 2022:10041–10048. ([IEEE Xplore][23])

22. **Novák, F., Báča, T., Saska, M.** “Collaborative Object Manipulation on the Water Surface by a UAV-USV Team Using Tethers.” arXiv:2407.08580, 2024. ([arXiv][24])

### MARL foundations

23. **Lowe, R., Wu, Y., Tamar, A., Harb, J., Abbeel, P., Mordatch, I.** “Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments.” *Advances in Neural Information Processing Systems 30*, 2017. ([NeurIPS 会议录][25])

24. **Yu, C., Velu, A., Vinitsky, E., Gao, J., Wang, Y., Bayen, A., Wu, Y.** “The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games.” *NeurIPS 35*, 2022:24611–24624. ([NeurIPS 会议录][2])

25. **Kuba, J. G., Chen, R., Wen, M., et al.** “Trust Region Policy Optimisation in Multi-Agent Reinforcement Learning.” *International Conference on Learning Representations (ICLR)*, 2022. ([OpenReview][26])

26. **Zhong, Y., Kuba, J. G., Feng, X., Hu, S., Ji, J., Yang, Y.** “Heterogeneous-Agent Reinforcement Learning.” *Journal of Machine Learning Research*, 25(32), 2024:1–67. ([机器学习研究杂志][27])

27. **Lyu, X., Baisero, A., Xiao, Y., Amato, C.** “A Deeper Understanding of State-Based Critics in Multi-Agent Reinforcement Learning.” *Proceedings of the AAAI Conference on Artificial Intelligence*, 36(9), 2022:9396–9404. DOI: **10.1609/aaai.v36i9.21171**. ([AAAI 期刊][28])

28. **Wen, M., Kuba, J., Lin, R., Zhang, W., Wen, Y., Wang, J., Yang, Y.** “Multi-Agent Reinforcement Learning is a Sequence Modeling Problem.” *NeurIPS 35*, 2022:16509–16521. ([NeurIPS 会议录][3])

---

## 最终优先级

如果你现在开始真正落地，我会按这个顺序：

[
\boxed{
\text{坐标/推力审计}

>

\text{bounded allocator}

>

\text{4D PID tracking}

>

\text{真实 net evaluator}

>

\text{30D MAPPO}

>

8D/6D/4D

>

\text{IPPO/MAPPO/HAPPO}

>

\text{net-state ablation}

>

\text{GRU}

>

\text{Transformer}
}
]

其中前四项甚至比再调 PPO hyperparameter 更优先。因为如果 (B)、PID 或真正的 capture evaluator 有物理错误，后面所有“MARL 提升多少”的数字都会失去意义。

尤其是你的 4D 分层设计，我认为非常值得做，但论文中最强的论证方式不是事先宣称它比 direct thruster 高级，而是用**同 observation、同 MAPPO、同训练预算、同 Prey、同最终 OBI 网捕获判据**的 8D→6D→4D controlled experiment，把“推进器分配”和“任务级协作抽象”的贡献真正拆出来。这会比单纯再提出一个更复杂的 MARL 网络更清晰，也更符合你这个问题最独特的科学价值。

[1]: https://arxiv.org/abs/2607.05939 "https://arxiv.org/abs/2607.05939"
[2]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html "https://proceedings.neurips.cc/paper_files/paper/2022/hash/9c1535a02f0ce079433344e14d910597-Abstract-Datasets_and_Benchmarks.html"
[3]: https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html "https://proceedings.neurips.cc/paper_files/paper/2022/hash/69413f87e5a34897cd010ca698097d0a-Abstract-Conference.html"
[4]: https://arxiv.org/abs/2410.00120 "https://arxiv.org/abs/2410.00120"
[5]: https://www.sciencedirect.com/science/article/abs/pii/S0005109813000368 "https://www.sciencedirect.com/science/article/abs/pii/S0005109813000368"
[6]: https://ieeexplore.ieee.org/document/9812083/ "https://ieeexplore.ieee.org/document/9812083/"
[7]: https://arxiv.org/abs/2108.07445 "https://arxiv.org/abs/2108.07445"
[8]: https://arxiv.org/abs/2303.01799 "https://arxiv.org/abs/2303.01799"
[9]: https://arxiv.org/abs/2301.03293 "https://arxiv.org/abs/2301.03293"
[10]: https://arxiv.org/abs/2312.12255 "https://arxiv.org/abs/2312.12255"
[11]: https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c250086 "https://www.aas.net.cn/cn/article/doi/10.16383/j.aas.c250086"
[12]: https://ieeexplore.ieee.org/document/10844952/ "https://ieeexplore.ieee.org/document/10844952/"
[13]: https://arxiv.org/html/2604.19404v1 "https://arxiv.org/html/2604.19404v1"
[14]: https://www.science.org/doi/10.1126/scirobotics.ade7811 "https://www.science.org/doi/10.1126/scirobotics.ade7811"
[15]: https://arxiv.org/abs/2605.15528 "https://arxiv.org/abs/2605.15528"
[16]: https://www.sciencedirect.com/science/article/abs/pii/S0967066107000329 "https://www.sciencedirect.com/science/article/abs/pii/S0967066107000329"
[17]: https://www.sciencedirect.com/science/article/pii/S0029801823003608 "https://www.sciencedirect.com/science/article/pii/S0029801823003608"
[18]: https://www.sciencedirect.com/science/article/pii/S0005109824000487 "https://www.sciencedirect.com/science/article/pii/S0005109824000487"
[19]: https://www.sciencedirect.com/science/article/abs/pii/S0029801820308490 "https://www.sciencedirect.com/science/article/abs/pii/S0029801820308490"
[20]: https://arxiv.org/abs/2002.01022 "https://arxiv.org/abs/2002.01022"
[21]: https://onlinelibrary.wiley.com/doi/full/10.1002/rob.21772 "https://onlinelibrary.wiley.com/doi/full/10.1002/rob.21772"
[22]: https://ieeexplore.ieee.org/document/10341785/ "https://ieeexplore.ieee.org/document/10341785/"
[23]: https://ieeexplore.ieee.org/document/9830869/ "https://ieeexplore.ieee.org/document/9830869/"
[24]: https://arxiv.org/abs/2407.08580 "https://arxiv.org/abs/2407.08580"
[25]: https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html "https://proceedings.neurips.cc/paper/2017/hash/68a9750337a418a86fe06c1991a1d64c-Abstract.html"
[26]: https://openreview.net/forum?id=EcGGFkNTxdJ "https://openreview.net/forum?id=EcGGFkNTxdJ"
[27]: https://jmlr.org/papers/v25/23-0488.html "https://jmlr.org/papers/v25/23-0488.html"
[28]: https://ojs.aaai.org/index.php/AAAI/article/view/21171 "https://ojs.aaai.org/index.php/AAAI/article/view/21171"
[29]: https://arxiv.org/html/2410.00120v2 "https://arxiv.org/html/2410.00120v2"
