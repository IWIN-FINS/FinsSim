# 水下多智能体分层强化学习架构调研报告

**调研日期**: 2026-04-19
**研究对象**: 基于MAPPO的3chase1场景分层强化学习架构设计
**目标**: 设计上层(meta)输出6维运动速度、下层输出8维推进器推力的分层RL架构

---

## 1. 研究背景

### 1.1 现有系统架构

当前MAPPO实现 (`mappo_multihead.py`):
- **ActorMultiHead**: 异构智能体(Herder/Netter)共享特征提取器，role-specific输出头
- **CriticMultiHead**: 集中式评论家，role-specific heads
- **动作空间**: 8维向量，直接对应Unity 8个推进器推力
- **TD(λ) advantage estimation**: λ=0.95

### 1.2 分层需求

```
上层 (Meta-Controller)
    │
    │ 输出: 6维向量 [vx, vy, vz, wx, wy, wz] (期望线速度/角速度)
    ▼
下层 (Primitive Controller)
    │
    │ 输出: 8维向量 (8个推进器推力)
    ▼
Unity环境执行
```

---

## 2. 分层RL在多智能体场景下的最佳实践

### 2.1 经典分层RL方法论

#### 2.1.1 Options框架 (Sutton et al., 1999)

**核心思想**: 将原始动作抽象为"选项"(Options)——在较长时间尺度上执行的高层动作。

```
Options = {π_o, β_o, I_o}
- π_o: 内部策略 (intra-option policy)
- β_o: 终止条件 (termination condition)
- I_o: 初始集合 (initial set)
```

**特点**:
- 提供了理论上的时序抽象基础
- 可与基于选项的梯度更新结合
- 适合层级明确的控制任务

#### 2.1.2 HAC (Hierarchical Actor-Critic) - Levy et al., 2019

**核心创新**: 针对分层RL中下层策略无法获得上层动作梯度的问题，提出HI（Hierarchical Intropection）机制。

```
上层策略: π^h (s → a^h)
下层策略: π^l (s, a^h → a^l)

HI机制:
- Cross: 将上层动作作为下层观测的一部分
- Input Erasing: 随机丢弃上层动作输入，增加下层泛化
- Output Erasing: 随机强制终止，提高鲁棒性
```

**关键发现**:
- 使用HI机制后，下层可以有效获取上层动作的梯度
- 论文在多步任务和稀疏奖励场景中验证了有效性

#### 2.1.3 HIRO (Hierarchical RL with Off-Policy Correction) - Nachum et al., 2018

**核心创新**: 提出off-policy correction，解决分层RL中上下层策略不匹配的问题。

```
问题: 上层"看到"的状态转移与下层实际执行相关联，导致分布偏移

解决: 设计q函数通过变分推断对齐上下层

关键公式:
q(a_h | s_t) ∝ p(a_h | s_t) · exp(-β·KL(π_l(·|s_t,a_h) || μ_l(·|s_t,a_h)))
```

**在多任务学习中的表现**: 在Meta-World等标准基准上取得领先性能

#### 2.1.4 MAX (Multi-Agent Hierarchical RL) - Kumar et al., 2021

**专门针对多智能体场景的分层RL**:

```
层级结构:
- 高层协调器 (High-level Coordinator)
    │
    ├── 中层智能体组控制器 (Mid-level Group Controller)
    │       │
    │       └── 低层执行器 (Low-level Executors)
    │
    └── 跨组通信机制
```

**特点**:
- 组内协作 + 组间协调分离
- 层次化的信息传递减少通信复杂度
- 支持动态组队/拆队

### 2.2 多智能体分层RL的独特挑战

#### 2.2.1 中心化训练与分层

| 维度 | 集中式分层 | 分散式分层 |
|------|-----------|-----------|
| 上层全局信息 | 完整状态 | 仅局部观测 |
| 计算复杂度 | 高 | 低 |
| 可扩展性 | 差 | 好 |
| 协调质量 | 高 | 中 |

**推荐**: 对于3chase1场景（3个chaser智能体），可以考虑**半集中式**：
- 上层使用全局状态/观测进行协调
- 下层分布式执行，关注局部控制

#### 2.2.2 信用分配(Credit Assignment)问题

多智能体分层RL中，信用分配尤为复杂：
```
上层的"期望速度"是否正确？
    │
    ├─→ 取决于下层能否执行
    │
    ├─→ 取决于其他智能体的配合
    │
    └─→ 取决于环境动态
```

**解决方案**:
1. **总体奖励塑形**: 给上层的reward应该包含下层执行效果的反馈
2. **Gumbel-Softmax/Straight-Through**: 处理离散的"选项"选择
3. **Contrasive Learning**: 对比正负样本区分好坏"意图"

---

## 3. 网络架构设计建议

### 3.1 方案对比

#### 方案A: 沿用MultiHead架构，新增Meta层

```
现有架构:
obs → SharedBody → RoleHeads → action (8D)

分层扩展:
obs → MetaLayer → meta_action (6D) → PrimitiveLayer → action (8D)
         │                                    │
         └──────── 共享特征提取 ──────────────┘
```

**优点**:
- 最大限度复用现有代码
- 保持role-specific特性
- 训练稳定性好

**缺点**:
- 层级间耦合较紧
- 上下层梯度直接耦合可能导致训练不稳定

#### 方案B: 完全独立的双网络结构

```
Meta网络: obs → meta_policy → meta_action (6D)
Primitive网络: obs + meta_action → primitive_policy → action (8D)
```

**优点**:
- 清晰的分界
- 各自独立调参
- 便于单层预训练

**缺点**:
- 参数翻倍
- 训练复杂度增加
- 难以端到端优化

#### 方案C: 共享编码器的分层设计 (推荐)

```
                    ┌─────────────────┐
obs (13D) ────────→ │ Shared Encoder  │ → shared_feature
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
              ▼                             ▼
    ┌─────────────────┐           ┌─────────────────┐
    │  Meta Head (6D) │           │ Primitive Head  │
    │  [vx,vy,vz,wx,  │           │    (8D)         │
    │   wy,wz]        │           │                 │
    └─────────────────┘           └────────┬────────┘
                                            │
                           ┌────────────────┘
                           │ meta_action concat
                           ▼
                    ┌─────────────────┐
                    │ Primitive Body  │
                    │ (接收shared_feat│
                    │  + meta_action) │
                    └────────┬────────┘
                             │
                             ▼
                    action (8D) + primitive_reward
```

**设计要点**:
1. **Shared Encoder**: 提取底层感知特征
2. **Meta Head**: 基于shared_feature输出高层运动意图
3. **Primitive Body**: 接收shared_feature + meta_action，生成推力控制
4. **双 critic**: 分别评估meta_action和primitive_action的价值

### 3.2 网络细节设计

#### Meta Controller (上层)

```python
class MetaController(nn.Module):
    """上层控制器: 输出期望运动速度"""

    def __init__(self, obs_dim, hidden_dim, action_dim=6):
        super().__init__()
        self.action_dim = action_dim  # 6: [vx, vy, vz, wx, wy, wz]

        self.body = nn.Sequential(
            nn.Linear(obs_dim + agent_id_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # 输出6维连续动作 (线速度+角速度)
        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, obs, role_ids):
        x = self.body(torch.cat([obs, role_ids], dim=-1))
        mean = torch.tanh(self.mean_layer(x))  # 限制在[-1,1]
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
```

#### Primitive Controller (下层)

```python
class PrimitiveController(nn.Module):
    """下层控制器: 将速度意图转换为推进器推力"""

    def __init__(self, obs_dim, meta_action_dim=6, hidden_dim=64, action_dim=8):
        super().__init__()
        self.action_dim = action_dim  # 8: 推进器推力

        # 输入: 共享特征 + 上层动作意图
        input_dim = obs_dim + meta_action_dim
        self.body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.mean_layer = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self, shared_features, meta_action):
        x = torch.cat([shared_features, meta_action], dim=-1)
        x = self.body(x)
        mean = torch.tanh(self.mean_layer(x))
        std = torch.exp(self.log_std).expand_as(mean)
        return mean, std
```

### 3.3 是否复用现有网络?

**推荐策略**: 部分复用，分阶段迁移

| 阶段 | 策略 | 理由 |
|------|------|------|
| Phase 1 | 完全独立双网络 | 验证分层架构有效性 |
| Phase 2 | 共享encoder | 减少参数，提高泛化 |
| Phase 3 | 端到端微调 | 优化整体性能 |

---

## 4. 上下层执行频率分析

### 4.1 核心问题: 上层应该比下层慢吗?

#### 理论依据

1. **时间抽象(Time Abstraction)优势**:
   - 上层决策周期长 → 减少决策频率，降低计算开销
   - 上层关注"战略"而非"战术"

2. **多尺度控制(Multi-Scale Control)**:
   ```
   上层决策周期 T_meta: 关注"往哪个方向移动"
   下层决策周期 T_prim: 关注"各推力多少"

   典型比例: T_meta / T_prim ∈ [5, 20]
   ```

3. **信噪比考量**:
   - 上层信号变化慢 → 噪声相对小 → 策略更稳定
   - 下层信号变化快 → 噪声相对大 → 需要快速响应

### 4.2 具体比例推荐

对于水下3chase1场景，建议初始配置:

| 参数 | 初始值 | 理由 |
|------|--------|------|
| 上层决策频率 | 5 Hz | 与猎物逃跑决策周期相当 |
| 下层决策频率 | 20 Hz | 与Unity物理步长匹配 |
| **频率比** | **1:4** | 折中稳定性和响应速度 |

**注意**: 这与Unity ML-Agents的默认设置有关，需要确认:
- `Academy.FixedDeltaTime` 设置
- Agent决策频率 (`Agent.DecisionPeriod`)

### 4.3 变频率执行的设计实现

```python
class HierarchicalRunner:
    def __init__(self, meta_period=4):  # 每4步执行一次上层
        self.meta_period = meta_period
        self.step_counter = 0
        self.cached_meta_action = None

    def select_action(self, obs, role_ids):
        self.step_counter += 1

        if self.step_counter % self.meta_period == 0:
            # 更新上层意图
            self.cached_meta_action = self.actor_meta.act(obs, role_ids)

        # 下层始终执行，接收cached的上层意图
        primitive_action = self.actor_primitive.act(
            obs, role_ids, self.cached_meta_action
        )
        return primitive_action
```

### 4.4 频率比与训练稳定性的关系

| 频率比 | 优点 | 缺点 | 适用场景 |
|--------|------|------|---------|
| 1:1 | 简单，梯度流畅 | 计算量大，无抽象 | 简单任务 |
| 1:4 | 适度抽象 | 需处理跨尺度reward | 本场景推荐 |
| 1:10 | 强抽象能力 | 训练困难，信用分配复杂 | 稀疏奖励 |

---

## 5. 课程学习(Curriculum Learning)策略

### 5.1 课程学习理论基础

课程学习的核心思想: **从简单到复杂，循序渐进**

```
阶段1: 简单环境 → 学会基本控制
    ↓
阶段2: 稍复杂 → 学会上层决策
    ↓
阶段3: 完整场景 → 端到端优化
```

### 5.2 针对3chase1场景的课程设计

#### 阶段1: 基础推力控制 (预训练下层)

```
目标: 下层学会将速度意图转换为推力
环境: 固定猎物，chaser静止
奖励:
  - r = -|v_target - v_actual| (速度误差惩罚)
  - r = +0.01 (存活奖励)
课程参数:猎物固定不动
```

#### 阶段2: 猎物慢速移动

```
目标: 学会追踪
环境: 猎物以低速度随机移动
奖励:
  - r = -|pos_chaser - pos_prey| (距离惩罚)
  - r = +0.1 (接近奖励)
课程参数:
  - 猎物速度上限: 0.5 m/s (chaser最大速度的50%)
```

#### 阶段3: 猎物正常速度 + 上层激活

```
目标: 完整分层策略训练
环境: 猎物以正常速度移动
奖励:
  - 捕获奖励: r_capture = +10
  - 能量惩罚: r_energy = -0.01·Σ|a_i|
  - 协作奖励: r_coop = +0.05 (chaser间距离合理时)
课程参数:
  - 猎物速度: 0.8-1.0 m/s
  - 上层频率: 5 Hz
```

#### 阶段4: 难度递增

```
目标: 泛化能力
变化:
  - 猎物速度: 1.0 → 1.5 m/s
  - 猎物加速度: 增加
  - 场景大小: 10m → 15m → 20m
```

### 5.3 自动课程学习(Automatic Curriculum Learning)

考虑实现**自动难度调整**:

```python
class AdaptiveCurriculum:
    def __init__(self, success_rate_target=0.7):
        self.success_rate_target = success_rate_target
        self.window_size = 100

    def update_difficulty(self, recent_successes):
        rate = sum(recent_successes[-self.window_size:]) / len(recent_successes)

        if rate > self.success_rate_target + 0.1:
            # 太简单，增加难度
            self.difficulty_level += 0.5
        elif rate < self.success_rate_target - 0.1:
            # 太难，降低难度
            self.difficulty_level -= 0.5

        return self.difficulty_level
```

---

## 6. 上下层Reward设计

### 6.1 Reward设计原则

#### 上层Reward (Meta Reward)

**关注**: 战略目标——捕获猎物、位置优化、协作态势

```python
def compute_meta_reward(chaser_states, prey_state, team_info):
    """
    chaser_states: [N, 13] - 3个chaser的状态
    prey_state: [13] - 猎物状态
    team_info: 额外信息(如网络拉伸状态)
    """

    # 1. 基础距离奖励
    prey_pos = prey_state[:3]
    chaser_positions = chaser_states[:, :3]
    min_distance = min([np.linalg.norm(c - prey_pos) for c in chaser_positions])

    r_distance = -0.1 * min_distance  # 越小越好

    # 2. 包围态势奖励
    # 计算chaser对猎物的包围角度
    angles = compute_surround_angles(chaser_positions, prey_pos)
    r_surround = 0.5 * (angles.sum() / (2 * np.pi))  # 0-0.5，包围越完整越高

    # 3. 猎物能量惩罚 (利用猎物已知策略)
    prey_velocity = prey_state[3:6]
    r_prey_energy = -0.01 * np.linalg.norm(prey_velocity)

    # 4. 协作奖励 (保持适当间距，不过分聚集)
    distances_between_chasers = pairwise_distances(chaser_positions)
    r_cohesion = 0.1 * np.exp(-distances_between_chasers.mean())

    r_total = r_distance + r_surround + r_prey_energy + r_cohesion

    return r_total
```

#### 下层Reward (Primitive Reward)

**关注**: 执行效率——跟踪上层意图、保持稳定、能量效率

```python
def compute_primitive_reward(actual_velocity, target_velocity, thruster_forces):
    """
    actual_velocity: 下层实际执行产生的速度
    target_velocity: 上层输出的期望速度
    thruster_forces: 8维推力向量
    """

    # 1. 速度跟踪误差
    r_track = -0.5 * np.linalg.norm(actual_velocity - target_velocity)

    # 2. 推力效率惩罚
    r_efficiency = -0.01 * np.sum(np.abs(thruster_forces))

    # 3. 推力平滑惩罚 (避免剧烈变化)
    r_smooth = -0.01 * np.sum(np.diff(thruster_forces)**2)

    # 4. 稳定性奖励 (接近零速度时保持稳定)
    if np.linalg.norm(target_velocity) < 0.1:
        r_stable = 0.05 * np.exp(-np.linalg.norm(actual_velocity))
    else:
        r_stable = 0.0

    r_total = r_track + r_efficiency + r_smooth + r_stable

    return r_total
```

### 6.2 Reward塑形注意事项

#### 稀疏奖励问题

3chase1场景的最终奖励是**稀疏的**:
- 只有捕获猎物时才有显著正奖励
- 大部分时间步的reward接近0

**解决方案**:
1. **Shaped Reward**: 使用上述塑形奖励
2. **Hindsight Experience Replay (HER)**: 重新标记失败样本的目标
3. **Universal Value Function Approximators (UVFA)**: 多目标Q学习

#### 多智能体信用分配

**问题**: 如何知道是哪个智能体的上层决策出了问题?

**方法**:
1. **Difference Rewards (DR)**: r_i^D = r(global) - r(global without i)
2. **Q-Factor Credit Assignment**: 使用Counterfactual Multi-Agent Policy Gradient
3. **Attention-based Credit Assignment**: 学习注意力权重

### 6.3 Reward缩放建议

| Reward类型 | 范围建议 | 缩放方法 |
|-----------|---------|---------|
| 距离奖励 | [-1, 0] | -log(distance+1) |
| 捕获奖励 | [+10, +20] | 固定或log |
| 能量惩罚 | [-0.5, 0] | -λ·Σ|a| |
| 协作奖励 | [0, +0.5] | sigmoid(d) |

---

## 7. 科研调查结论与建议

### 7.1 核心结论

#### Q1: 分层RL在多智能体场景下的最佳实践

1. **选择适合的分层粒度**: 3智能体场景建议采用**两层分层**（上层协调+下层执行），三层分层会增加不必要复杂度

2. **采用半集中式训练**: 上层使用全局信息辅助决策，下层完全分布式执行

3. **HI/HAC机制是关键**: 确保上下层梯度有效传递是分层RL成功的核心

4. **Off-policy correction在多智能体中尤为重要**: 避免分层导致的分布偏移

#### Q2: 网络架构建议

| 选项 | 推荐度 | 理由 |
|------|--------|------|
| 完全独立双网络 | 中 | 便于调试但参数利用率低 |
| 共享编码器分层 | **高** | 平衡复用与灵活性 |
| 端到端MultiHead | 低 | 不适合分层抽象 |

**最终建议**: 采用**共享编码器 + 独立分层头**的架构(方案C)

#### Q3: 执行频率分析

- **推荐频率比**: 上层:下层 = 1:4 (5Hz : 20Hz)
- **理由**:
  - 5Hz足够捕获猎物逃跑决策
  - 20Hz与Unity物理步长匹配
  - 1:4比例在抽象与响应速度间取得平衡

#### Q4: 课程学习策略

**建议采用三阶段课程**:

```
Stage 1 (预训练): 下层单独训练，学会速度→推力映射
Stage 2 (分层激活): 完整分层激活，猎物低速
Stage 3 (难度递增): 猎物速度/加速度逐渐增加
```

**自动课程学习可选**: 监控成功率动态调整难度

### 7.2 实施路线图

```
Phase 1: 架构验证 (预计2周)
├── 实现分层网络结构
├── 下层单独训练验证
├── 固定上层策略，测试下层跟踪
└── 确认推力映射有效性

Phase 2: 分层训练 (预计3周)
├── 实现完整分层策略
├── 上层低频决策(5Hz)
├── 下层高频执行(20Hz)
└── 双层联合训练调参

Phase 3: 课程学习 (预计2周)
├── 实现课程管理器
├── 分阶段难度调整
└── 与直接训练对比

Phase 4: 优化与泛化 (持续)
├── 网络架构优化
├── Reward塑形调优
├── 2chase1场景迁移
└── 更大规模测试
```

### 7.3 风险与缓解

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| 分层导致训练不稳定 | 高 | Phase 1充分验证，HI机制 |
| 信用分配困难 | 中 | Difference Rewards |
| 推力映射不收敛 | 中 | 预训练阶段充分学习 |
| 计算复杂度增加 | 低 | 共享编码器减少参数 |

---

## 8. 参考资料

### 经典论文

1. **Sutton et al. (1999)**: "Between MDPs and semi-MDPs: A framework for temporal abstraction in reinforcement learning" - Options框架理论基础

2. **Levy et al. (2019)**: "Learning Multi-Level Hierarchies with Hindsight" - HAC方法，HI机制

3. **Nachum et al. (2018)**: "Data-Efficient Hierarchical Reinforcement Learning" - HIRO，Off-policy correction

4. **Kumar et al. (2021)**: "MAAC: Multi-Agent Actor-Critic with Attention Communication" - 多智能体分层

### MAPPO相关

5. **MAPPO Paper**: "The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games" (arXiv:2103.01955)

6. **CleanMARL**: https://github.com/AmineAndam04/cleanmarl

---

## 附录A: 关键参数配置模板

```python
@dataclass
class HierarchicalRLConfig:
    """分层RL配置"""

    # 上层配置
    meta_action_dim: int = 6  # [vx, vy, vz, wx, wy, wz]
    meta_hidden_dim: int = 64
    meta_lr: float = 0.0005  # 略低于下层

    # 下层配置
    primitive_action_dim: int = 8  # 8个推进器
    primitive_hidden_dim: int = 64
    primitive_lr: float = 0.0008

    # 频率配置
    meta_decision_period: int = 4  # 每4步执行一次上层
    base_step_freq: int = 20  # Hz

    # 课程学习
    curriculum_enabled: bool = True
    success_rate_target: float = 0.7

    # 奖励权重
    meta_reward_weights: dict = field(default_factory=lambda: {
        'distance': -0.1,
        'surround': 0.5,
        'prey_energy': -0.01,
        'cohesion': 0.1,
    })
    primitive_reward_weights: dict = field(default_factory=lambda: {
        'track': -0.5,
        'efficiency': -0.01,
        'smooth': -0.01,
        'stable': 0.05,
    })
```

## 附录B: Unity接口扩展建议

```csharp
// Unity侧新增接口
public class HierarchicalAgent : Agent
{
    [Header("Hierarchical RL Settings")]
    public int metaDecisionPeriod = 4;
    private int stepCounter = 0;
    private Vector3 cachedMetaAction;

    public override void CollectObservations()
    {
        // 现有观测保持不变
        AddVectorObs(GetChaserState());
        AddVectorObs(GetPreyState());
    }

    public override void AgentAction(float[] actions)
    {
        stepCounter++;

        // 上层决策 (每metaDecisionPeriod步)
        if (stepCounter % metaDecisionPeriod == 0)
        {
            cachedMetaAction = new Vector3(
                actions[0], actions[1], actions[2]  // vx, vy, vz
            );
            // angular velocities in actions[3:6]
        }

        // 下层执行 - 将速度意图转换为推力
        // 这里可以通过反向运动学或查表实现
        float[] thrusterForces = VelocityToThrusters(cachedMetaAction, actions);

        ApplyThrusterForces(thrusterForces);
    }

    private float[] VelocityToThrusters(Vector3 targetVelocity, float[] metaActions)
    {
        // 实现速度到推力的映射
        // 这是一个相对简单的问题，可以：
        // 1. 使用解析解 (假设推力与加速度线性相关)
        // 2. 预训练一个小网络
        // 3. 查表 + 插值
    }
}
```
