# 3Chase1: 问题定义、当前接口与文献调研 Prompt

> 状态：2026-08-18 的实现审计与设计基线。Unity 端 Chaser observation 已收敛为 30D；完整的 4D subgoal + PID + physical wrench 架构尚未迁移进 `3Chase1` 训练链路。本文刻意区分“当前已实现”与“下一版目标架构”。

## 1. 目的与边界

`3Chase1` 是一个异构多潜器协同追逃/收网任务。本文用于：

- 给实现者固定问题、坐标、观测与动作的语义；
- 给后续实验定义可比较的 learned 和 traditional baselines；
- 为外部文献调研提供一份自包含且不会误导实现状态的 prompt。

本文不把 OBI 网格当作可信的精确几何传感器。网的主要作用是柔性物理约束、碰撞和捕获判定，而不是向策略提供稳定的“平面中心、法向、宽高”几何状态。

## 2. 场景与任务

### 2.1 参与者

一个 training area 内有四个 Agent：

| 集合 | 数量 | 角色 | 是否训练 |
|---|---:|---|---|
| Chasers | 1 | `Herder` | 是 |
| Chasers | 2 | `Netter1`、`Netter2` | 是 |
| Prey | 1 | 鱼 | 当前通常由固定逃逸策略控制，不更新 |

- `Herder` 的任务是把 prey 往两名 `Netter` 和渔网所在区域驱赶。
- 两名 `Netter` 要协作运动并维持适当的相对间距，以支持收网/捕获。
- Prey 的固定逃逸逻辑会远离最近的追方；它的 Unity 端动力学目前是受水柱边界约束的期望速度模型，而不是与 Chaser 完全同构的 FinsROV 动力学。

### 2.2 物理环境

- 每个 Chaser 是场景实例化的 `FinsROV_Fossen`，使用 6-DOF 水动力、浮力/阻力和 8 个推进器。
- `Assets/Scenes/3Chase1.unity` 已被做成多 Training Area 场景，并且每个 area 的场景实例包含 Fossen 兼容组件和 Domain Randomization；不能通过修改原始 `FinsROV_Fossen.prefab` 来改变该实验。
- 渔网为 Obi 柔性网。其形变受物理求解、锚点、接触和数值参数影响；网面可能弯曲，也未必存在对控制意义稳定的上下左右四个顶点。
- 因而网可视为环境动力学与终局约束的一部分。v1 actor 不应依赖网格顶点、网面法向、瞬时宽高或其他噪声很强的 OBI 内部量。

### 2.3 终局与捕获

`CatchAreaManager` 目前支持四种可配置捕获判据：

```text
NetSurfaceDistance          prey 到实际变形网面的距离在阈值内，且满足 hold time
NetCollision                prey 与网发生碰撞
UuvProximity                prey 距任一 Netter 小于阈值
NetCollisionOrUuvProximity  上述后二者之一
```

默认的真实任务语义应优先使用 `NetSurfaceDistance` 或 `NetCollision`；`UuvProximity` 是早期课程或诊断用的宽松替代目标，不能当成“收网成功”。捕获触发时三个 Chaser 各获得同一个终局奖励，Prey 获得终局惩罚，并结束四个 Agent 的 episode。当前默认值是 Chaser `+25`、Prey `-25`，但由环境参数配置覆盖。

## 3. 形式化问题

把任务建模为异构 cooperative-vs-fixed-opponent 的 Dec-POMDP：

```text
M = <S, {A_i}, P, {O_i}, O, R, gamma>
```

- 状态 `s in S`：三个 Chaser 与 Prey 的位姿、线/角速度，Fossen 水动力状态，推进器一阶响应，Obi 网粒子/约束状态，水流与随机化参数，以及捕获状态。
- 训练 Agent `i in {H, N1, N2}`：Herder 和两个 Netter。Prey 在当前实验中是环境的固定策略，不是共同训练的对手策略。
- 转移 `P(s' | s, a_H, a_N1, a_N2)`：Unity 的刚体、水动力、推进器、Obi 与 Prey 逃逸逻辑共同决定；它对 Chaser 来说部分可观测且含随机性。
- 局部观测 `o_i`：第 4 节定义的 30D、以当前 Chaser 为原点的 controller-body 特征。
- 执行策略：`a_i ~ pi_i(a_i | o_i, role_i)`；训练采用 CTDE，critic 可以使用训练期允许的集中信息，执行期不可依赖世界系真值或 OBI 内部状态。
- 优化目标：最大化 Chaser 团队折扣回报

```text
J(pi) = E_pi[sum_t gamma^t R_team,t]
```

其中 `R_team` 需要同时体现真实捕获成功、推进过程、角色协作与控制平稳性。由于各角色目前获得各自 shaping reward，训练中还应同时报告 team return、每角色 return 和 success rate，而不能只看 reward 总和。

## 4. 坐标与观测

### 4.1 控制坐标

所有 Chaser actor 的相对位置、相对速度、线速度和角速度都应在当前 Chaser 的 `controller_body` 坐标系表达：

```text
x: 前方 / surge
y: 上方 / heave
z: 左方 / sway
```

因此当前 Chaser 的局部信息对场景世界原点、出生位置和世界 yaw 不敏感。鱼的水平 bearing 定义为：

```text
bearing = atan2(fish_relative_position_body.z,
                fish_relative_position_body.x)

0:   prey 在正前方
> 0: prey 在左方
< 0: prey 在右方
```

Unity 的世界坐标系与 ROS 的坐标语义不同，但本任务的 actor/PID/allocator 接口必须只使用上述 controller-body 约定，不能在不同层各自添加未审计的符号翻转。

### 4.2 当前 actor observation：30D

每个 Chaser 的 actor observation 是 30D。Chaser 自己看到的两个队友槽位是固定排序的：

- Herder 看到 `[Netter1, Netter2]`；
- 每个 Netter 看到 `[Herder, 另一个 Netter]`。

| 索引 | 维度 | 字段 | 坐标/语义 |
|---|---:|---|---|
| `[0:3]` | 3 | `self_linear_velocity_body` | 自身线速度，controller-body |
| `[3:6]` | 3 | `self_angular_velocity_body` | 自身角速度，controller-body |
| `[6:9]` | 3 | `prey_relative_position_body` | Prey 相对自身的位置，controller-body |
| `[9:12]` | 3 | `prey_relative_velocity_body` | Prey 相对自身的速度，controller-body |
| `[12]` | 1 | `distance_to_prey` | `||prey_relative_position_body||` |
| `[13]` | 1 | `bearing_to_prey` | 上述水平 bearing，rad |
| `[14:16]` | 2 | `teammate_A_role_onehot` | `[is_herder, is_netter]` |
| `[16:19]` | 3 | `teammate_A_relative_position_body` | A 相对自身的位置 |
| `[19:22]` | 3 | `teammate_A_relative_velocity_body` | A 相对自身的速度 |
| `[22:24]` | 2 | `teammate_B_role_onehot` | `[is_herder, is_netter]` |
| `[24:27]` | 3 | `teammate_B_relative_position_body` | B 相对自身的位置 |
| `[27:30]` | 3 | `teammate_B_relative_velocity_body` | B 相对自身的速度 |

Python 的 multi-head actor 还接收 `role_ids`，注入自身角色 one-hot，并使用 Herder/Netter 的独立动作 head。因此 actor observation 内的队友 role one-hot 和 Python 注入的 self role 语义不同，二者都可保留。

这个 30D 是低维、局部、相对且适合 MLP 的输入。`64 x 2` 级别 MLP 处理 30D 没有容量问题；应优先关注特征尺度、坐标一致性和任务相关性，而不是为降低几维而删去速度或队友信息。

### 4.3 网的信息：当前不加入 actor obs

v1 不加入网的瞬时显式几何观察。理由：

1. OBI 柔性网不是可信的刚性平面，几何形状会随物理数值误差和接触而抖动。
2. 网的粗略位置已经可由两个 Netter 的相对位置间接推断。
3. 直接加入 `net center/right/up/normal/width/height` 会把不稳定且未必可实机获取的仿真特权信息泄漏给 actor。

仅当将来证明确有一个稳定、执行期可获得、且无法由 30D 推断的网状态时，才以单独消融加入。例如“有效捕获接触”若不是 episode 立即结束，才可能值得加一个 bit 或连续进度；若捕获直接终止，则不需要该 observation。

### 4.4 已移除的旧 13D controller suffix

旧版 Unity `ChaserAgent` 曾在 30D 后额外输出 13D：

```text
position - initial_controller_position: 3D world frame
rotation quaternion:                   4D world frame
linear velocity:                       3D world frame
angular velocity:                      3D world frame
```

该 suffix 是旧 position-controller 路径为重建 world-frame pose 而添加的兼容接口，不是高层策略的任务观测。现在 `ChaserAgent` 和 `3Chase1.unity` 的三台 Chaser 都只声明并输出 30D，Python 普通 MAPPO wrapper 也只消费这 30D。

旧 `MAPPO_POSITION_CONTROL` Python 路径已被替换为只消费 30D 的 body-frame 控制链路；不应重新加入 world-frame suffix。

## 5. 当前已实现动作链路

### 5.1 主训练路径：端到端 8 推动作

当前默认 `3Chase1` MAPPO 是端到端控制：

```text
30D local actor observation
  -> MAPPO actor
  -> 8D normalized action in [-1, 1]
  -> Unity FinsROVAgentRuntime
  -> each thruster's own MaxForward/MaxReverseForceN
  -> FinsROV_Fossen dynamics
```

推进器动作顺序固定为：

```text
[Vertical1, Vertical2, Vertical3, Vertical4,
 Horizontal1, Horizontal2, Horizontal3, Horizontal4]
```

在 `NormalizedMaxForceRequest` 模式下，动作 `+/-1` 分别映射为该推进器自身的正/反最大物理推力；当前 FinsROV 仿真基线通常是每路对称 `+/-7 N`。这条链路不经过 ROS2，也不经过 wrench allocator。

### 5.2 当前分层 position-control 路径

`MAPPO_POSITION_CONTROL` 已复用 `1Chase1` 的 local PID + physical wrench 语义。它不读取全局 position、quaternion 或旧 suffix，也不支持旧 `empirical_thruster_mixer`：

```text
30D local actor observation + self role
  -> MAPPO high-level actor
  -> a = [a_x, a_y, a_z, a_yaw] in [-1, 1]^4
  -> e_pos_body = [a_x, a_y, a_z] * [1.5, 0.5, 1.5] m
  -> e_yaw_deg = a_yaw * 90 deg
  -> fixed TraditionalPositionPID
  -> tau_body = [Fx, Fy, Fz, Mx, My, Mz]
  -> physical_wrench_allocator
  -> 8D normalized thruster action in [-1, 1]^8
  -> each Unity thruster maps to its own physical max force
```

这里的高层 action 是“当前机体系下的相对位移子目标和 yaw 误差”，不是全局位置，也不是推力。`target_body_delta_limits` 与 `yaw_error_limit_deg` 是有明确单位的配置参数，必须在训练配置、checkpoint 元数据和 eval 配置中保持一致。

### 5.3 PID 与 wrench allocator 的职责

- 高层 MAPPO：决定协作意图，例如 Herder 从哪侧逼近 Prey、Netter 如何拉开和前往捕获区域。
- PID：将局部子目标误差变成连续、平滑的 6D body wrench；它不需要 OBI 网状态。
- `physical_wrench_allocator`：在每路推进器物理边界内把 `tau_body` 分配为 8 路力。它使用经场景 FinsROV 几何和质心审计的物理 forward matrix `B`，目标为在推进器限幅内尽量实现所需 wrench，而非使用旧经验 mixer 的无量纲动作混合。
- Unity：将归一化八推输出按每个推进器的 max force 转成实际推力并施加，推进器自身保留一阶响应。

控制/分配器使用的机体系 wrench 顺序是：

```text
[Fx, Fy, Fz, Mx, My, Mz]
= [surge, heave, sway, roll, yaw, pitch]
```

仿真中每路推进器的基本物理边界为 `[-7, +7] N`。纯轴理论 wrench 可达边界、`B` 的定义、单位和计算方法见仓库根目录的 [Wrench 能力说明](../../../../docs/reference/thrusters/wrench-capability.zh-CN.md)。这些能力边界是推力可达集，不是速度上限。

### 5.4 已完成接口与验收条件

在把目标架构称作“实现完成”前，至少应验证：

1. 已完成：Unity 只输出并声明 30D Chaser actor observation，普通 MAPPO wrapper 无 suffix slicing。
2. 已完成：4D action 的每个维度有单位、范围和 body-frame 符号测试。
3. 待 Unity 场景实测：纯 `+Fx`、`+Fy`、`+Fz`、`+My` 指令的运动方向与约定一致。
4. 已完成：allocator 输出逐路不超过 `[-1, 1]`，换算后的力不超过逐推进器边界。
5. 已完成：执行期 policy 和 PID 不读取 world pose、世界系 quaternion、OBI 粒子/网格真值或 centralized critic 信息。
6. 已完成：新 checkpoint 从头训练，不将旧 8D-thruster 或旧 world-pose checkpoint 直接套到新动作语义。

## 7. 奖励、训练与评估要点

### 7.1 现有 reward 结构

当前 Unity 可切换两类 Chaser shaping：

- `HerdingNet`：Netter 有靠近 Prey、网宽/间距及进度 shaping；Herder 有靠近 Prey 和把 Prey 往网中心推进的 shaping。
- `SimpleChasePrey`：每个 Chaser 仅按自己接近 Prey 的距离与进度获得 shaping，常用于先验证移动/追逐能力。

无论使用哪一种 shaping，捕获终局奖励仍由 `CatchAreaManager` 的捕获判据结算。研究比较必须报告 success rate 和真实 `NetSurfaceDistance`/`NetCollision` 成功率，不能因 `UuvProximity` 或密集距离奖励较高就宣布收网成功。

### 7.2 建议的评估指标

对每个随机种子、每个 difficulty/DR 级别，至少统计：

- 真实收网成功率：按 `NetSurfaceDistance` 和/或 `NetCollision` 单独报告；
- 捕获时间、episode return、每角色 return；
- Prey 到网面最小距离、终止时到网面的距离；
- Herder/Netter 到 Prey 的最小距离与终止距离；
- Netter 相对间距的误差分布；
- 碰撞、越界、超时和 false-success 比例；
- 控制质量：平均/峰值线速度、角速度、动作变化率、推进器饱和率、wrench 残差和能耗代理；
- 对初始相对方位、Prey 速度、流场、推进器/水动力 DR、网物理参数扰动的鲁棒性。

### 7.3 应避免的实验混淆

- 不同 action abstraction（8 推、6D wrench、4D pose+yaw）不能共用 checkpoint，也不能只比较原始 return。
- 网面几何如果仅在仿真可用，不能只给 learned policy 而不说明它是 privileged observation。
- 不能把 Unity 的 `UuvProximity` 课程成功率与真实网捕获成功率混为一谈。
- PID、allocator、推进器边界、time scale、Prey policy 和 DR 分布必须随 run 一起记录。

## 8. 候选 baselines 与消融

下列 baseline 应在相同场景、捕获定义、训练预算和 DR 下比较：

| 类别 | 名称 | 高层输出 | 低层 | 价值 |
|---|---|---|---|---|
| Learned | End-to-end MAPPO | 8D thruster | 无 | 当前实现基线，学习控制和协作全部耦合 |
| Learned | Direct-wrench MAPPO/PPO | 6D normalized wrench | physical allocator | 分离低层推进器分配，但仍让 RL 学动力学控制 |
| Learned + classical | Hierarchical MAPPO + PID | 4D body subgoal pose+yaw | PID + physical allocator | 目标方案，降低高层动作维度并提供稳定的低层闭环 |
| Classical | Role-aware geometric pursuit | 目标航向/速度或 4D subgoal | 相同 PID + allocator | Pure pursuit、LOS/PN、距离保持、队形/间距控制组成的可解释基线 |
| Classical | Scripted finite-state netting | 阶段目标 | 相同 PID + allocator | `approach -> spread netters -> herder drive -> close/capture`，检验任务是否可由规则完成 |
| Ablation | 无 teammate 信息 | 14D self+prey | 同上 | 量化协作观测的必要性 |
| Ablation | 26D vs 30D | 去掉/保留队友 role one-hot | 同上 | 验证显式队友角色的价值 |
| Ablation | 30D vs 30D+可信网状态 | 仅在稳定、可部署的网状态存在时 | 同上 | 检验网信息是否真有增益 |
| Ablation | centralized critic 结构 | flat / role-aware / tokenized | 不变 | 检验 CTDE 表达而非控制接口的影响 |

传统基线不要直接输出八推，除非它本身就是推进器分配算法。为了公平解释“协作策略”与“低层控制品质”，传统和分层 learned baseline 应共享相同的 PID、physical wrench allocator、推力边界和坐标约定。

## 9. 相关实现入口

| 内容 | 路径 |
|---|---|
| Chaser observation、reward、8 推下发 | `UnityProject/marus-example/Assets/Scripts/ThreeChaseOne/ChaserAgent.cs` |
| 网与捕获判定 | `UnityProject/marus-example/Assets/Scripts/ThreeChaseOne/CatchAreaManager.cs` |
| OBI 网表面辅助模型 | `UnityProject/marus-example/Assets/Scripts/ThreeChaseOne/NetSurfaceModel.cs` |
| Prey 逃逸与观测 | `UnityProject/marus-example/Assets/Scripts/ThreeChaseOne/PreyAgent.cs` |
| Unity 归一化推力到物理力 | `UnityProject/marus-example/Assets/Scripts/RL/FinsROVAgentRuntime.cs` |
| 3Chase1 wrapper | `python/finssim_marl/src/finssim_marl/envs/unity/wrapper.py` |
| 当前 MAPPO config | `python/finssim_marl/src/finssim_marl/training/chasing_3_chase_1_config.py` |
| 当前 position-control 实现 | `python/finssim_marl/src/finssim_marl/algorithms/mappo_multihead_position_controller.py` |
| 当前 body PID+wrench backend | `python/finssim_marl/src/finssim_marl/algorithms/position_control_backends.py` |
| 1Chase1 的新 PID+wrench 参考 | `python/finssim_rl/src/finssim_rl/training/hierarchy_chase_config.py` |
| FinsROV wrench 能力与矩阵说明 | `docs/reference/thrusters/wrench-capability.zh-CN.md` |

## 10. 可直接发送给网页端 AI 的文献调研 Prompt

```markdown
请以“研究助理 + 严格文献综述”的方式，针对下面这个水下多机器人协同追逃/收网问题做调研。请使用中英文论文、顶会/期刊、公开技术报告与可复现实验资料；优先给出 2018-2026 的工作，同时补充真正奠基性的较早文献。不要编造论文、DOI、实验结果或代码链接。无法核实的条目请明确标注“待核实”。请用中文写作，但保留论文英文标题。

# 问题背景

我要研究一个 Unity 水下仿真任务 `3Chase1`：3 台 FinsROV 协作追逐 1 条 Prey。三台追方角色异构：

- 1 台 Herder：从合适方向逼近并把 Prey 往收网区域驱赶；
- 2 台 Netter：协作运动、保持适当相对间距，网是连接/附着于 Netter 系统的 Obi 柔性网；
- 1 条 Prey：当前由固定逃逸策略控制，不参与策略训练。

每台 Chaser 是 6-DOF FinsROV，带 Fossen 风格水动力、浮力/阻力、推进器一阶响应和 8 个推进器。网是 Obi 柔性物理对象，主要提供碰撞、力和最终捕获约束。网可能弯曲且数值上不稳定，不能假设其有可靠的刚性平面、稳定法向、稳定四角点，或可部署的高精度网格观测。

真实成功应以 Prey 到实际变形网面的距离/保持时间或 Prey 与网碰撞来定义。早期课程中可能临时使用“Prey 接近任一 Netter”，但它不是最终收网成功定义。

# 当前学习与控制接口

训练使用 CTDE 风格 MAPPO：执行时 actor 只用局部观测；训练期 critic 可以使用集中信息。Herder 和 Netter 是不同 role，有 role-aware shared trunk / role-specific heads。

每个 Chaser 的 actor observation 当前是 30D，全部在当前 Chaser 的 body frame 表达，约定 `x=forward/surge, y=up/heave, z=left/sway`：

```text
0:3    self linear velocity body
3:6    self angular velocity body
6:9    prey relative position body
9:12   prey relative velocity body
12     distance to prey
13     horizontal bearing to prey = atan2(rel_z, rel_x)
14:16  teammate A role one-hot [is_herder, is_netter]
16:19  teammate A relative position body
19:22  teammate A relative velocity body
22:24  teammate B role one-hot [is_herder, is_netter]
24:27  teammate B relative position body
27:30  teammate B relative velocity body
```

队友槽位是固定角色语义：Herder 看到两个 Netter；Netter 看到 Herder 和另一个 Netter。当前不想把不可靠的 OBI 网格形状、网法向、网宽高等直接加入 actor observation。请评估这个选择是否合理，并给出只有在什么条件下才应该加入哪类最小网状态。

当前主基线是 end-to-end MAPPO：actor 直接输出 8D 归一化推进器动作 `[-1,1]^8`，每路在 Unity 乘以自身物理最大推力。推进器顺序为：

```text
[Vertical1, Vertical2, Vertical3, Vertical4,
 Horizontal1, Horizontal2, Horizontal3, Horizontal4]
```

计划中的新分层方案是：

```text
30D local observation + role
  -> high-level MAPPO action [dx_body, dy_body, dz_body, yaw_error] in [-1,1]^4
  -> fixed traditional PID
  -> body wrench [Fx, Fy, Fz, Mx, My, Mz]
  -> physically calibrated bounded wrench allocator
  -> 8D normalized thruster action
```

其中 4D 高层动作表示本体坐标系下的相对位移子目标和相对 yaw 误差，不是 world-frame target，也不是直接力。allocator 要求每路推进器不超过物理边界，目标是尽量实现 6D wrench；其物理矩阵基于推进器位置、方向和质心审计，而不是经验混控表。

旧实现曾有一个 13D world-frame pose/quaternion/velocity suffix，专为旧 position controller 重建全局目标；它已从 Unity 端和 Python 执行路径删除。新方案不需要恢复它。请不要建议把 world position 或 OBI 内部真值泄漏给执行期 actor，除非明确作为 privileged-training ablation 并讨论 sim-to-real 风险。

# 需要回答的研究问题

1. 请给出与本任务最相关的文献分类和代表论文：
   - 多机器人 pursuit-evasion、encirclement、herding、capture；
   - 多机器人/多 AUV 协同捕获、拖网、柔性约束物体操控，或最接近的相关问题；
   - heterogeneous-role MARL、CTDE、MAPPO、centralized critic；
   - hierarchical RL / goal-conditioned RL 与固定经典低层控制器结合；
   - 6-DOF AUV/ROV 的传统追踪、路径跟随、队形、编队、pure pursuit、LOS、proportional navigation、MPC、CBF/安全约束控制。

2. 每篇推荐工作请给出：完整引文、发表 venue/年份、可访问链接（DOI/arXiv/出版社/代码）、研究问题、状态/观测、动作空间、控制层级、奖励或目标、是否异构、是否建模柔性物体/网、主要结果，以及它与本任务的可迁移性和不适用点。请区分真实 AUV 实验、纯仿真、2D 点质量模型与 6-DOF 水动力模型。

3. 请评审上述 30D 局部 observation：
   - 对 Herder/Netter 协作是否足够满足 Markov 性或近似 Markov 性？
   - 应否保留队友 role one-hot、相对速度和 bearing？
   - 在网几何不可信的前提下，什么最小、稳定、可部署的网/捕获状态可能有帮助？
   - actor 是否需要 LSTM/GRU、slot-structured MLP、DeepSets 或 Transformer？请不要仅凭维度 30 就推荐 Transformer；要给出可验证的触发条件和消融设计。

4. 请评审三个控制抽象的比较：
   - 8D direct-thruster RL；
   - 6D direct-wrench RL + physical allocator；
   - 4D body-subgoal pose+yaw RL + fixed PID + physical allocator。
   讨论学习难度、可解释性、样本效率、闭环稳定性、推进器饱和、wrench tracking residual、动作语义变化、sim-to-real、以及对异构协作高层策略的影响。请明确哪些结论有文献证据，哪些只是工程假设。

5. 请为本任务设计公平且可复现的 baseline 套件：
   - learned baselines（MAPPO/IPPO/HAPPO 等是否适合）；
   - 传统 role-aware geometric baseline；
   - pure pursuit / LOS / proportional navigation / potential-field / formation controller 是否适用；
   - 规则状态机 `approach -> spread netters -> herder drive -> capture` 应如何定义；
   - 所有高层 baseline 应复用同一 PID + physical allocator，以避免把协作决策和低层推进器分配混淆。

6. 请提出严谨的 reward、课程和评价设计。真实主指标必须是网面距离/网碰撞捕获成功率，而非仅距离 shaping 或 UUV proximity。请给出：success rate、capture time、角色 return、网捕获距离、Netter spacing、控制平稳性、饱和率、wrench residual、能耗代理、DR 鲁棒性、统计显著性和 ablation 矩阵。

7. 请最后给出一个优先级明确的实施建议：
   - v1 最小可发表/可验证方案；
   - 最值得做的 3-5 个 ablation；
   - 关键失败模式和对应诊断日志；
   - 哪些主张需要先做物理单轴实验或坐标审计才能写进论文。

请用如下结构输出：

1. Executive summary
2. 文献表格（按主题分组，至少 15 篇高相关且可核实条目）
3. 本任务与文献的逐点映射
4. Observation 与 action 设计评审
5. Baseline 与实验矩阵
6. Reward、课程、指标和统计设计
7. 推荐实施路线与风险
8. 完整参考文献列表

要求：给出事实与可验证来源，明确区分已证实结论、推测和工程建议；不要把 OBI 柔性网错误简化成可靠的刚性平面，也不要把 early-course 的 UUV proximity 当成最终收网成功。
```
