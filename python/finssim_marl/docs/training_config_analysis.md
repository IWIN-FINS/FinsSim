# 训练循环与频率参数指南

## 1. 核心概念：Training Step vs Environment Step

```
1 Training Step = 1次完整循环 = num_envs个episode全部完成
```

| 概念 | 计数器 | 含义 |
|------|--------|------|
| **Env Step** | `step` | 累计所有环境的总步数 |
| **Training Step** | `training_step` | 每完成一次"采集+更新"后+1 |


注意在强化学习中，完成一句游戏叫做**episode**，完成一步是**time_step**，完成一次训练是**epoch**

| 术语 | 定义 | 类比（以考试为例） |
| --- | --- | --- |
| Time-step | 智能体与环境的一次交互 | 考卷上做的每一道题 |
| Episode | 从开始到结束的完整序列 | 完整参加了一次考试 |
| Epoch | 神经网络参数更新的迭代次数 | 考完后把这卷子订正/复习了几遍 |

## 2. 训练循环流程

```
while step < total_timesteps:
    │
    ├─► collect_rollout()      # 采集 num_envs 个 episode
    │       step += 本次采集的环境步数总和
    │
    ├─► training_step++        # 训练步数 +1
    │
    ├─► algorithm.update()     # 用采集的数据更新策略
    │       └─► epochs 次 PPO 更新
    │
    ├─► 每 eval_steps 个 training_step:
    │       └─► evaluate()     # 运行 num_eval_ep 个评估episode
    │
    ├─► 每 save_freq 个环境步:
    │       └─► 保存 checkpoint
    │
    └─► 每 log_every 个 training_step:
            └─► 记录日志
```

## 3. 频率参数详解

### 3.1 `num_envs` - 并行环境数量

**作用**：决定每次 `collect_rollout()` 能采集多少个 episode。

**实际影响**：
- `num_envs` ↑ → 每 training_step 采集的数据量 ↑ → 梯度更稳定
- `num_envs` ↑ → 完成一个 training_step 需要的 Env Step ↑

### 3.2 `epochs` - PPO 更新轮数

**作用**：每次 `algorithm.update()` 对整个 batch 数据做多少次 PPO 更新。

**实际影响**：
- `epochs` ↑ → 每步更新次数 ↑ → 计算成本 ↑, 但数据利用更充分
- 与 `num_envs` 相关：`num_envs` 大时数据量大，`epochs` 可适当减小

### 3.3 `eval_steps` - 评估频率（以 training_step 为单位）

**作用**：每多少个 training_step 进行一次策略评估。

**实际公式**：
```
评估间隔(环境步) ≈ eval_steps × num_envs × 平均episode长度
```

**示例**（`num_envs=8`, `eval_steps=5`, 平均episode=500步）：
```
5 × 8 × 500 = 20,000 环境步进行一次评估
```

### 3.4 `num_eval_ep` - 每次评估的 episode 数

**作用**：每次调用 `evaluate()` 时运行的 episode 数量。

**实际影响**：
- `num_eval_ep` ↑ → 评估结果更稳定（均值方差更小），但评估更慢
- 通常设置 5~20 即可

### 3.5 `save_freq` - checkpoint 保存频率（以环境步为单位）

**作用**：每累计多少个环境步保存一次模型。

**注意**：`save_freq` 是按 `step` 计数器（环境步总数）计算的，不是按 training_step。

### 3.6 `log_every` - 日志记录频率（以 training_step 为单位）

**作用**：每多少个 training_step 记录一次训练指标（reward、loss等）。

## 4. 频率参数关系图

```
total_timesteps: 10,000,000 (目标环境步)
        │
        ▼
┌─────────────────────────────────────────────────────┐
│  Training Step (一次采集 + 一次更新)                  │
│                                                     │
│  采集: num_envs 个 episode                          │
│  更新: epochs 次 PPO passes                          │
│                                                     │
│  training_step++                                   │
└─────────────────────────────────────────────────────┘
        │
        ├─── eval_steps ───► evaluate() [num_eval_ep个episode]
        ├─── log_every ───► logger.log()
        │
        └─── save_freq 环境步累计 ───► checkpoint.save()
```

## 5. num_envs=32 时的推荐配置

| 参数 | 默认值 (num_envs=8) | 推荐值 (num_envs=32) | 原因 |
|------|---------------------|---------------------|------|
| `num_envs` | 8 | **32** | 更多并行环境，数据采集效率更高 |
| `epochs` | 10 | **8** | 数据量大了，减少更新次数省计算 |
| `eval_steps` | 5 | **3** | 每步环境步更多，eval更频繁一些 |
| `num_eval_ep` | 10 | **10** | 评估episode数，保持稳定即可 |
| `log_every` | 10 | **10** | 按training_step计，保持不变 |
| `save_freq` | 50,000 | **100,000** | 环境步累计更快，减少保存频率 |

### 推荐配置下的实际频率

```
eval_steps=3, num_envs=32, avg_ep=500:
  评估间隔 ≈ 3 × 32 × 500 = 48,000 环境步

save_freq=100,000:
  checkpoint间隔 = 100,000 环境步
```

## 6. 关键要点

1. **`num_envs` 是最核心的调节参数**：增大可以提高数据采集效率，但会增加内存占用。

2. **`epochs` 需要与 `num_envs` 配合调节**：`num_envs` 大时，数据量更大，可以适当减少 `epochs`。

3. **`eval_steps` 控制评估频率（按training_step）**，`save_freq` 控制保存频率（按环境步）：两者衡量的基准不同。

4. **实际评估间隔 = eval_steps × num_envs × avg_episode_length**，规划时请考虑实际的 episode 长度。
