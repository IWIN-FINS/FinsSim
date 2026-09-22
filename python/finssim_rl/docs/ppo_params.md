# PPO 关键参数说明

## 核心参数

| 参数 | 说明 | 推荐范围 |
|------|------|----------|
| `n_steps` | 每次更新前**每个环境**收集的步数 | 32 ~ 4096 |
| `batch_size` | 每次更新使用的 mini-batch 大小 | n_steps * num_envs 的因数 |
| `n_epochs` | 每次更新重复使用数据的 epoch 数 | 3 ~ 20 |
| `num_envs` | 并行环境数量 | 根据 CPU 资源调整 |
| `clip_range` | PPO 策略更新的裁剪范围 | 0.1 ~ 0.3 (默认 0.2) |
| `ent_coef` | 熵系数，鼓励探索 | 0.0 ~ 0.01 |
| `gamma` | 折扣因子 | 0.99 ~ 0.9999 |

## 参数关系

```
总采样步数 = n_steps × num_envs

每次更新样本数 = n_steps × num_envs

batch_size 必须是 n_steps × num_envs 的因数
```

## num_envs=64 下的推荐配置

### 配置 A：通用型（平衡训练速度与稳定性）
```python
training_config = PPOTrainingConfig(
    n_steps = 128,          # 每个环境128步
    batch_size = 512,       # 512 = 64 × 8，或 1024
    n_epochs = 10,
    gamma = 0.99,
    learning_rate = 3e-4,
    clip_range = 0.2,
)
# 总采样 = 128 × 64 = 8192 步/更新
```

### 配置 B：高频更新（适合简单任务）
```python
training_config = PPOTrainingConfig(
    n_steps = 64,           # 每个环境64步
    batch_size = 256,       # 256 = 64 × 4
    n_epochs = 10,
    gamma = 0.99,
    learning_rate = 3e-4,
)
# 总采样 = 64 × 64 = 4096 步/更新，更新更频繁
```

### 配置 C：低频更新（适合复杂任务）
```python
training_config = PPOTrainingConfig(
    n_steps = 256,          # 每个环境256步
    batch_size = 1024,      # 1024 = 64 × 16
    n_epochs = 10,
    gamma = 0.99,
    learning_rate = 3e-4,
)
# 总采样 = 256 × 64 = 16384 步/更新，样本利用率更高
```

## 当前项目配置参考

```python
# ppo_control_for_velocity（num_envs=64）
PPOTrainingConfig(
    n_steps = 512,
    batch_size = 1024,
    n_epochs = 10,
    gamma = 0.99,
    learning_rate = 1e-4,
)
# 总采样 = 512 × 64 = 32768 步/更新

# ppo_control_for_pose
PPOTrainingConfig(
    n_steps = 512,
    batch_size = 1024,
    n_epochs = 10,
    gamma = 0.99,
    learning_rate = linear_schedule(3e-4),
)
```

## 调参建议

1. **n_steps**：增大可以提高样本效率，但会限制策略更新频率
2. **batch_size**：通常设为总样本数的一部分，能被整除即可
3. **n_epochs**：增大可以更好利用数据，但可能导致过拟合
4. **num_envs**：增加可以加速采样，但每个环境的有效数据会减少
5. **clip_range**：增大允许更激进策略更新，减小使更新更保守
