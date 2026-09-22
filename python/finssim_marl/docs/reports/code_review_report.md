# MAPPO 3 Chase 1 Unity 代码审查报告

**审查文件**: `cleanmarl/mappo_3chase1_unity.py`
**审查日期**: 2026-04-09
**审查人**: Claude Code
**最后更新**: 2026-04-09 (已修复 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3)

---

## 目录

1. [逻辑错误 (Logic Errors)](#1-逻辑错误-logic-errors) ✅ 已修复 1.1, 1.2, 1.3, 1.4
2. [潜在风险 (Potential Risks)](#2-潜在风险-potential-risks) ✅ 已修复 2.1, 2.2, 2.3
3. [可提升的地方 (Areas for Improvement)](#3-可提升的地方-areas-for-improvement)
4. [次要问题 (Minor Issues)](#4-次要问题-minor-issues)

---

## 1. 逻辑错误 (Logic Errors) ✅

### ✅ 1.1 `alive_envs.remove()` 在循环内移除元素 - 已修复

**位置**: `mappo_3chase1_unity.py` (原 line 1305)

**问题**: `alive_envs.remove(j)` 是按**值**移除，而不是按索引。当多个环境同时完成时，移除操作会导致错误的元素被删除。

**修复方案**: 使用独立的 `completed_envs` 列表收集需要移除的环境，然后在循环外统一移除：

```python
# 修复后代码
completed_envs = []
for i, j in enumerate(alive_envs):
    if done[i] or truncated[i]:
        completed_envs.append(j)
for j in completed_envs:
    alive_envs.remove(j)
    rb.add(episodes[j])
    episodes[j] = dict()
    if args.env_type == "smaclite":
        ep_stat[j] = infos[i]
# 收集未完成环境的数据
obs = []
state = []
for i, j in enumerate(alive_envs):
    if not done[i] and not truncated[i]:
        obs.append(next_obs[i])
        state.append(next_state[i])
```

---

### ✅ 1.2 `b_role_ids[:, :3]` 硬编码切片 - 已修复

**位置**: TD(lambda) 计算和训练循环部分

**问题**: 代码假设 `role_ids` 的排列顺序是 `[Herder, Netter, Netter, Prey]`，即 Chaser Team 在前3个位置。这个假设脆弱且容易出错。

**修复方案**: 引入 `chaser_team_order_tensor`，根据 agent_id 顺序动态计算 chaser team 的正确排列顺序：

```python
# 修复后：预计算 chaser team 的 role_ids 排列顺序
env_role_ids = role_ids_batch[0]  # 环境 agent 顺序的 role_ids
chaser_env_positions = [i for i, rid in enumerate(env_role_ids) if rid != ROLE_MAPPING['Prey']]
chaser_agent_ids = [env_role_ids[i] for i in chaser_env_positions]
herder_pos = chaser_env_positions[chaser_agent_ids.index(ROLE_MAPPING['Herder'])] if ROLE_MAPPING['Herder'] in chaser_agent_ids else None
netter_positions = [(chaser_env_positions[i], i) for i, rid in enumerate(chaser_agent_ids) if rid == ROLE_MAPPING['Netter']]
netter_positions.sort(key=lambda x: env_role_ids[x[0]])  # 按 agent_id 排序
chaser_team_order = [herder_pos] + [pos for pos, _ in netter_positions]] if herder_pos is not None else [pos for pos, _ in netter_positions]
chaser_team_order_tensor = torch.tensor(chaser_team_order, device=device).long()

# TD(lambda) 计算中使用 chaser_team_order_tensor
role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
next_role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)

# 训练循环中使用 actor_indices_tensor 提取 chaser obs
obs_actor_t = b_obs[:, t, actor_indices_tensor, :chaser_team_obs_dim]
role_ids_actor_t = b_role_ids[:, chaser_team_order_tensor]
```

---

### ✅ 1.3 `obs_chaser_team` 组装时缺少 None 检查 - 已修复

**位置**: `UnityEnvWrapper.step()` 方法

**问题**: 如果 `obs_dict` 中没有找到 Herder 或 Netter，对应的变量保持 `None`，`np.concatenate` 会失败或产生意外结果。

**修复方案**: 添加验证和错误处理：

```python
# 验证所有必需的 agent 都被找到
if herder_obs is None or netter1_obs is None or netter2_obs is None:
    raise ValueError(
        f"Missing agent observations in chaser team: "
        f"Herder={herder_obs is not None}, Netter1={netter1_obs is not None}, Netter2={netter2_obs is not None}. "
        f"Available agents: {list(obs_dict.keys())}"
    )
```

---

### ✅ 1.4 `get_role_from_agent_name` 的默认返回值问题 - 已修复

**位置**: `get_role_from_agent_name()` 函数

**问题**: 默认返回 `Netter(1)` 会静默地将未知 agent 误分类为 Netter，可能导致多个 agent 被当作 Netter。

**修复方案**: 改为抛出异常而不是静默失败：

```python
def get_role_from_agent_name(agent_name: str) -> int:
    """从agent名称中解析角色ID

    Args:
        agent_name: Unity环境的agent名称，格式如 'Herder?team=0?agent_id=3'

    Returns:
        role_id: 角色ID (0=Herder, 1=Netter, 2=Prey)

    Raises:
        ValueError: 如果无法从agent名称中解析出角色ID
    """
    for role_name, role_id in ROLE_MAPPING.items():
        if role_name in agent_name:
            return role_id
    raise ValueError(f"Cannot determine role for agent: '{agent_name}'. Available roles: {list(ROLE_MAPPING.keys())}")
```

---

### ✅ 1.5 评估间隔条件判断可能立即触发

**状态**: 已修复（手动）

**位置**: `mappo_3chase1_unity.py`

```python
if (training_step / args.epochs) % args.eval_steps == 0:
```

**问题**: 当 `training_step = 0` 时，`0 % eval_steps = 0`，条件成立，训练一开始就会立即执行评估。

**修复建议**:
```python
if training_step > 0 and (training_step / args.epochs) % args.eval_steps == 0:
```

---

### ✅ 1.6 `reward_weights` 定义但从未使用

**状态**: 已修复（手动）

**位置**: `UnityEnvWrapper.__init__()`

`reward_weights` 被定义但在整个代码中从未被引用，是死代码。

---

### ✅1.7 Critic 损失中 Herder/Netter 的权重硬编码

**状态**: 已修复（手动）

**位置**: 训练循环

```python
weight = torch.tensor([0.5, 0.3, 0.3], device=device)
value_loss = (((current_values - return_lambda_actor) ** 2) * weight).sum()
```

**问题**: 权重没有文档说明依据，且总和为 1.1 不是 1。

---

## 2. 潜在风险 (Potential Risks) ✅

### ✅ 2.1 Unity 进程端口冲突 - 已修复

**位置**: `get_unity_env()` 函数

**问题**: 如果 `worker_id` 超过 Unity 的端口偏移限制，或多个进程同时启动，可能发生端口冲突。

**修复方案**: 添加端口冲突重试机制：

```python
def get_unity_env(env_config: EnvConfig, seed, max_retries=3):
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=10.0)

    worker_id = env_config.worker_id
    env_base_port = env_config.env_base_port

    last_exception = None
    for attempt in range(max_retries):
        try:
            unity_env = UnityEnvironment(
                file_name=env_config.unity_env_binary_path,
                side_channels=[config_channel],
                no_graphics=True,
                seed=seed + worker_id,
                base_port=env_base_port + attempt,  # 尝试不同端口
                worker_id=worker_id
            )
            # ...
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Warning: Port {env_base_port + attempt} is in use, retrying...")
                config_channel = EngineConfigurationChannel()
                config_channel.set_configuration_parameters(time_scale=10.0)
            else:
                raise RuntimeError(f"Failed after {max_retries} attempts...")
```

---

### ✅ 2.2 评估环境没有设置 `base_port` - 已修复

**位置**: `get_unity_env_eval()` 函数

**问题**: 评估环境使用默认 `base_port=5005`，可能与训练worker的端口冲突。

**修复方案**: 添加 `base_port` 参数并提供默认值（与训练环境错开）：

```python
def get_unity_env_eval(seed, env_base_port=5004, worker_id=100):
    """创建评估用 Unity 环境

    Args:
        seed: 随机种子
        env_base_port: 基础端口（默认5004，与训练环境的基础端口5005错开）
        worker_id: Worker ID（默认100，确保与训练环境的worker不冲突）
    """
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=1.0)
    unity_env = UnityEnvironment(
        file_name="/RLChase/build/RLChase.x86_64",
        side_channels=[config_channel],
        no_graphics=True,
        seed=seed,
        base_port=env_base_port,
        worker_id=worker_id
    )
```

---

### ✅ 2.3 子进程的种子未正确设置 - 已修复

**位置**: `env_worker()` 函数

**问题**: Python 的 `random`、`np.random`、`torch.random` 的种子设置不会传递给子进程。

**修复方案**: 在子进程开始时设置随机种子：

```python
def env_worker(conn, env_config: EnvConfig, seed):
    """环境工作进程 - 在子进程内部创建环境"""
    # 设置子进程的随机种子，确保可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = None
    try:
        # ...
```

---

### 2.4 进度条更新计算不准确

**状态**: 未修复

**位置**: 训练循环

**问题**: `pbar.update(args.batch_size * np.mean(ep_length))` 不能准确反映实际执行的步数。

---

### 2.5 `_prey_escape` 策略的硬编码假设

**状态**: 未修复

**位置**: `UnityEnvWrapper._prey_escape()` 方法

**问题**: 硬编码了 3 个 chaser 和观察格式的假设。

---

### 2.6 `CloudpickleWrapper` 未被使用

**状态**: 未修复

`CloudpickleWrapper` 类被定义但未使用，是死代码。

---

### 2.7 缺少 CUDA 内存管理

**状态**: 未修复

代码没有对 GPU 内存进行任何管理，在长时间训练时可能导致 GPU 内存泄漏。

---

## 3. 可提升的地方 (Areas for Improvement)

### 3.1 使用配置文件或 dataclass 验证配置参数

建议添加配置验证逻辑，确保必需参数有效。

### 3.2 添加数据验证 (NaN/Inf 检测)

建议在关键位置添加 NaN/Inf 检测。

### 3.3 分离关注点 - 将 Unity 环境交互与训练逻辑解耦

`UnityEnvWrapper` 类承担了太多职责，建议分离。

### 3.4 添加更详细的日志和调试支持

建议使用 Python `logging` 模块添加日志。

### 3.5 将硬编码的超参数移到配置

| 硬编码值 | 位置 | 建议 |
|---------|------|------|
| `chaser_team_obs_dim = 13` | L1143 | 移到 `Args` 或从环境动态获取 |
| `chaser_team_action_dim = 8` | L1144 | 移到 `Args` |
| `weight = [0.5, 0.3, 0.3]` | L1464 | 移到 `Args` |
| `max_steps = 900` | L713, L1079 | 从 Unity 环境动态获取 |

### 3.6 优化 `_build_agent_id_onehot` 中的嵌套循环

双重嵌套循环对于大批量训练效率低，建议使用向量化操作。

### 3.7 添加类型注解

整个代码库缺少类型注解。

### 3.8 重构奖励计算逻辑

奖励分离逻辑分散在多处，建议集中管理。

---

## 4. 次要问题 (Minor Issues)

### 4.1 未使用的导入

- `base_events`, `base` - 未使用
- `Categorical` - 未使用
- `register_env` - 未使用
- `CloudpickleWrapper` 类 - 未使用

### 4.2 魔法数字 (Magic Numbers)

| 数字 | 位置 | 说明 |
|------|------|------|
| `13` | `chaser_team_obs_dim` | 追方团队观察维度 |
| `8` | `chaser_team_action_dim` | 追方团队动作维度 |
| `3` | 权重数组长度 | chaser team 成员数 |
| `900` | `max_steps` | 最大步数限制 |
| `10.0` | `time_scale` | 训练时 Unity 时间缩放 |
| `100` | eval worker_id | 评估环境 worker ID |

### 4.3 注释与代码不一致

**位置**: 原 line 916

```python
obs_array = np.array(obs_list) # todo: 修改先把obs_array和reward作为输入向量一一对齐！
```

这个 TODO 注释指出未完成的任务。

### 4.4 `_prey_escape` 中的冗余随机动作

`random_action` 定义在 try 块之前，但实际上只在 try 块失败时使用。

### 4.5 变量命名不一致

- `obs_chaser_team` 和 `obs_array` 命名风格不一致
- `global_reward_chaser` 在多处被重复覆盖使用

### 4.6 `get_state()` 未使用

`UnityEnvWrapper.get_state()` 方法定义但未被调用。

---

## 修复状态总结

### ✅ 已完成 (7项)

| 问题编号 | 描述 | 修复内容 |
|---------|------|---------|
| 1.1 | `alive_envs.remove()` Bug | 使用 `completed_envs` 列表先收集后移除 |
| 1.2 | `b_role_ids[:, :3]` 硬编码 | 引入 `chaser_team_order_tensor` 动态计算 |
| 1.3 | `obs_chaser_team` 缺少 None 检查 | 添加 `ValueError` 验证 |
| 1.4 | `get_role_from_agent_name` 默认返回 | 改为抛出 `ValueError` |
| 2.1 | Unity 端口冲突 | 添加 `max_retries` 重试机制 |
| 2.2 | 评估环境无 `base_port` | 添加 `env_base_port` 参数，默认5004 |
| 2.3 | 子进程种子未设置 | 在 `env_worker` 开始时设置种子 |

### ⏳ 未修复 (5项)

| 问题编号 | 描述 | 优先级 |
|---------|------|--------|
| 2.4 | 进度条更新计算不准确 | 低 |
| 2.5 | `_prey_escape` 硬编码假设 | 中 |
| 2.6 | `CloudpickleWrapper` 未使用 | 低 |
| 2.7 | 缺少 CUDA 内存管理 | 低 |
| 3.x | 各类优化建议 | 低 |

---

*报告生成时间: 2026-04-09*
*最后更新: 2026-04-09 (已修复 1.1, 1.2, 1.3, 1.4, 2.1, 2.2, 2.3)*
