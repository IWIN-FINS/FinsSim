# 配置系统架构详解

## 1. 三层配置结构

本项目采用三层配置结构，从上到下：

```
┌─────────────────────────────────────────────────────────┐
│  Layer 1: BaseConfig (场景配置类)                          │
│  例: Chasing3Chase1Config                                │
│                                                         │
│  - 包含 env_config (环境参数)                             │
│  - 包含 training_config (训练参数)                         │
│  - 提供 create_train_config() 工厂方法                     │
└─────────────────────────────────────────────────────────┘
                        │ create_train_config()
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 2: TrainConfig (扁平化最终配置)                    │
│                                                         │
│  - 所有参数平铺，无嵌套                                    │
│  - train.py 实际使用的配置                                 │
│  - 直接用于创建环境、算法、缓冲区等                         │
└─────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│  Layer 3: train.py 使用 TrainConfig 执行训练              │
└─────────────────────────────────────────────────────────┘
```

## 2. BaseConfig vs TrainConfig 关系

### BaseConfig（场景配置类）

```python
class Chasing3Chase1Config(BaseConfig):
    name = "chasing_3_chase_1"
    env_config: BaseEnvironmentConfig      # 嵌套：环境参数
    training_config: BaseTrainingConfig   # 嵌套：训练参数

    # 场景特有参数
    actor_hidden_dim: int = 64
    chaser_team_obs_dim: int = 13
    ...
```

**作用**：
- 定义一个**场景**的配置（3chase1、2chase1 等）
- 包含嵌套的 `env_config` 和 `training_config`
- 提供工厂方法 `create_train_config()` 返回 `TrainConfig`

### TrainConfig（扁平化配置）

```python
@dataclass
class TrainConfig:
    # 所有参数平铺在顶层
    num_envs: int = 8
    epochs: int = 8
    eval_steps: int = 5
    save_freq: int = 10
    ...
```

**作用**：
- 是 `BaseConfig.create_train_config()` 的返回值
- `train.py` 实际使用的配置对象
- 所有参数都是简单类型，没有嵌套

## 3. train.py 执行流程

```
train.py
  │
  ├─ 1. 解析 CLI 参数获取 config_name
  │
  ├─ 2. base_config = get_config(config_name)
  │      返回 Chasing3Chase1Config 实例
  │
  ├─ 3. config = base_config.create_train_config()
  │      返回 TrainConfig 实例（扁平化）
  │
  ├─ 4. CLI 参数覆盖 config 字段
  │
  └─ 5. main(args.config, base_config, config)
          │
          ├─ create_parallel_envs(config)     # 使用 config.num_envs
          ├─ algorithm = base_config.create_algorithm(...)  # 使用 config
          ├─ collect_rollout(mappo_conns, algorithm, config, rb)
          ├─ algorithm.update(batch)         # 使用 config.epochs
          ├─ evaluate(eval_env, algorithm, config)  # 使用 config.eval_steps
          └─ checkpoint_manager.save(...)     # 使用 config.save_freq
```

## 4. 创建 TrainConfig 的过程

`Chasing3Chase1Config.create_train_config()` 将嵌套配置展开为扁平结构：

```python
def create_train_config(self) -> TrainConfig:
    return TrainConfig(
        # 从 self.num_envs（BaseConfig 字段）获取
        num_envs=self.num_envs,

        # 从 self.training_config（BaseTrainingConfig）获取
        epochs=self.training_config.epochs,
        learning_rate_actor=self.training_config.learning_rate_actor,
        eval_steps=self.training_config.eval_steps,
        num_eval_ep=self.training_config.num_eval_ep,
        save_freq=self.training_config.save_freq,

        # 从 self.env_config（BaseEnvironmentConfig）获取
        env_base_port=self.env_config.env_base_port,
        unity_env_binary_path=self.env_config.env_path,
        ...
    )
```

## 5. 参数来源汇总

| TrainConfig 字段 | 来源 |
|-----------------|------|
| `num_envs`, `seed`, `device` 等通用字段 | `BaseConfig` 直接属性 |
| `epochs`, `learning_rate_actor`, `eval_steps` 等训练字段 | `BaseConfig.training_config` |
| `env_base_port`, `unity_env_binary_path` 等环境字段 | `BaseConfig.env_config` |
| `actor_hidden_dim`, `chaser_team_obs_dim` 等算法字段 | `BaseConfig` 直接属性 |

## 6. 为什么这样设计？

**BaseConfig 的嵌套结构**让相关参数自然分组：
- 环境相关 → `env_config`
- 训练相关 → `training_config`

**TrainConfig 扁平化**便于：
- `train.py` 直接访问所有字段 `config.xxx`
- CLI 参数覆盖 `config.xxx = value`
- 与 checkpoint 日志系统兼容

**工厂方法 `create_train_config()`**负责：
- 将嵌套配置"展开"为扁平结构
- 做默认值和类型转换
- 不同场景可以有不同的展开逻辑
