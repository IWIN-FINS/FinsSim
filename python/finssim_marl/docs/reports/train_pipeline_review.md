# 训练流水线代码审查报告

## 概述

本报告对 `./python/finssim_marl/scripts/train.py` 及相关组件（MAPPO算法、环境worker、配置系统、RolloutBuffer）进行系统性审查，识别逻辑错误、潜在风险和bug。

**审查范围**：
- `scripts/train.py` - 主训练脚本
- `underwater_marl/algorithms/mappo_multihead.py` - MAPPO多头算法实现
- `underwater_marl/envs/unity/worker.py` - Unity环境worker进程
- `underwater_marl/training/config.py` - 配置管理系统
- `underwater_marl/training/chasing_3_chase_1_config.py` - 3chase1场景配置
- `underwater_marl/algorithms/networks/rollout_buffer.py` - Rollout缓冲区

---

## 严重问题 (Critical)

### 1. `evaluate()` 函数中奖励和长度累加索引错误 ✓ 已修复

**文件**: `scripts/train.py`
**行号**: 371-374

> **状态**: 2026/04/19 已修复。对应的内部修复记录未随公开版保留。

**修复内容**:
```python
# 修复后
for i, j in enumerate(alive_envs):
    ep_reward[j] += reward[i]   # j是环境ID，i是循环索引
    ep_length[j] += 1
```

---

### 2. `evaluate()` 函数中进度条重复关闭 ✓ 已修复

**文件**: `scripts/train.py`
**行号**: 原398行

> **状态**: 2026/04/19 已修复。对应的内部修复记录未随公开版保留。

**修复内容**: 移除评估主循环结束后的 `pbar.close()`，确保所有评估episode完成后再关闭进度条。

---

### 3. `evaluate()` 函数中reset后数据处理逻辑缺陷 ✓ 已修复

**文件**: `scripts/train.py`
**行号**: 405-432

> **状态**: 2026/04/19 已修复。对应的内部修复记录未随公开版保留。

**修复内容**: 通过移除提前的 `pbar.close()`，reset后的环境执行被纳入正常流程管理。

---

### 4. checkpoint恢复后环境状态未同步 ⚠ 未修复

**文件**: `scripts/train.py`
**行号**: 494-500

**问题分析**:
- checkpoint仅保存了算法（actor/critic网络）的状态
- Unity环境的内部状态（agent位置、速度等）没有保存
- resume后，算法权重与实际环境状态可能不匹配

**风险**: 训练初期可能不稳定；reward可能异常。

**修复建议**:
考虑在checkpoint中保存环境状态，或在resume后显式reset所有环境。

---

### 5. `chasing_3_chase_1_config.py` 中未使用的onnx导入 ⚠ 未修复

**文件**: `underwater_marl/training/chasing_3_chase_1_config.py`
**行号**: 15

**问题分析**:
- 导入了onnx模块但没有任何实际使用
- 这是死代码，降低代码可读性

**风险**: 低 - 不影响功能但影响代码质量。

---

## 高优先级问题 (High)

### 6. `mappo_multihead.py` 中TD(λ)实现中next_value的边界处理

**文件**: `underwater_marl/algorithms/mappo_multihead.py`
**行号**: 244-249

```python
if t == int(ep_len - 1):
    next_value = torch.zeros(3, device=device)
else:
    next_obs_chaser_t = b_obs_chaser_team[ep_idx, t + 1].reshape(3, config.chaser_team_obs_dim).unsqueeze(0)
    next_role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
    next_value = self.critic(obs=next_obs_chaser_t, role_ids=next_role_ids_t)
```

**问题分析**:
- 当`t`是最后一步时，`next_value`设为0，这符合TD(λ)对终止状态的处理
- 这是正确的实现方式，因为终止状态没有后续价值

**当前状态**: 此项经验证为正确实现，保留在此作为记录以便审查者理解。

---

### 7. `collect_rollout()` 中硬编码的3 agent假设

**文件**: `scripts/train.py`
**行号**: 171-177, 207-211

```python
while len(alive_envs) > 0:
    with torch.no_grad():
        num_envs = len(alive_envs)
        role_ids_current = role_ids_tensor[:num_envs]

        current_obs = torch.from_numpy(obs).float().to(algorithm.device)
        obs_actor = current_obs[:, algorithm.actor_indices, :config.chaser_team_obs_dim]
        role_ids_actor = role_ids_current[:, algorithm.actor_indices]
```

**问题分析**:
- 代码假设恰好3个chaser team agents（1 Herder + 2 Netter）
- `config.chaser_team_obs_dim` 硬编码为13
- 如果环境配置改变，代码不会动态适应

**风险**: 特定于3chase1场景，缺乏通用性。

---

### 8. RolloutBuffer中obs_chaser_team_dim维度不匹配

**文件**: `underwater_marl/training/chasing_3_chase_1_config.py`
**行号**: 119

```python
obs_chaser_team_dim=chaser_team_obs_dim * 3,  # 13 * 3 = 39
```

**问题分析**:
- `chaser_team_obs_dim = 13`（单个agent的obs维度）
- `obs_chaser_team_dim = 39`（3个agent的obs拼接）
- 但在 `collect_rollout` 中收集的 `obs_chaser_team` 是单个agent的obs，不是拼接的

**风险**: 如果环境实际返回的不是期望格式，可能导致形状不匹配错误。

---

### 9. 训练循环中的进度计算

**文件**: `scripts/train.py`
**行号**: 558

```python
step += int(np.sum(ep_length))
```

**问题分析**:
- `ep_length` 在 `collect_rollout` 完成后包含每个环境的episode长度
- `np.sum(ep_length)` 是正确的总计步数

**当前状态**: 验证为正确实现。

---

## 中优先级问题 (Medium)

### 10. 评估函数中的死代码

**文件**: `scripts/train.py`
**行号**: 409

```python
remaining_eps = (eval_ep_per_env + (1 if j < remainder else 0)) - eval_ep_count[j]
```

**问题分析**:
- 这个变量在第410行的for循环中使用
- 但外层if条件已经保证了 `remaining_eps >= 1`
- 代码可以简化

**风险**: 低 - 不影响功能但降低可读性。

---

### 11. critic loss中硬编码的权重

**文件**: `underwater_marl/algorithms/mappo_multihead.py`
**行号**: 326

```python
weight = torch.tensor([0.4, 0.3, 0.3], device=device)
```

**问题分析**:
- Herder权重0.4，Netter1权重0.3，Netter2权重0.3
- 权重硬编码，没有通过配置传递

**风险**: 缺乏灵活性；难以调优。

---

### 12. `evaluate()` 中 `done` 变量遮蔽

**文件**: `scripts/train.py`
**行号**: 414

```python
done = False
while not done and ep_l < 900:
    # ...
    done = content["done"] or content["truncated"]
```

**问题分析**:
- 在内部循环中重新定义了 `done` 变量
- 外层循环中也有 `done`（第367行），但这里是新作用域
- 可能造成混淆

**风险**: 低 - 变量作用域正确但可读性差。

---

### 13. 进度条更新与实际episode完成可能不同步

**文件**: `scripts/train.py`
**行号**: 381

```python
pbar.update(1)
```

**问题分析**:
- 当 `eval_ep_count[j]` 达到目标时，`pbar.update(1)` 被调用
- 但 `all_ep_rewards` 和 `all_ep_lengths` 的收集是在后续循环中
- 如果程序在两者之间崩溃，统计数据可能丢失

**风险**: 低 - 但可能导致日志不完整。

---

## 低优先级问题 (Low)

### 14. 进程清理顺序可能导致资源泄露

**文件**: `scripts/train.py`
**行号**: 601-611

```python
# Cleanup eval envs
for conn in eval_conns:
    conn.send(("close", None))
for process in eval_processes:
    process.join()

# Cleanup training envs
for conn in mappo_conns:
    conn.send(("close", None))
for process in processes:
    process.join()
```

**问题分析**:
- 等待每个eval进程结束后再清理training进程
- 如果进程没有正确响应close信号，会导致主进程hang

**风险**: 低 - 但可能导致训练无法正常退出。

---

### 15. daemon进程可能导致子进程被强制终止

**文件**: `scripts/train.py`
**行号**: 106-108, 286-288

```python
for process in processes:
    process.daemon = True
    process.start()
```

**问题分析**:
- `daemon=True` 意味着主进程退出时子进程会被强制终止
- 如果子进程正在执行关键清理操作，可能导致资源泄露

**风险**: 低 - 但可能导致Unity环境连接未正确关闭。

---

### 16. 错误处理不完整

**文件**: `underwater_marl/envs/unity/worker.py`
**行号**: 107-113

```python
except Exception as e:
    print(f"Error in env_worker: {e}")
    import traceback
    traceback.print_exc()
    if env is not None:
        env.close()
    conn.close()
```

**问题分析**:
- 捕获所有异常但只是打印
- 主进程无法感知worker的错误状态

**风险**: 中 - worker失败可能导致训练静默失败。

---

## 建议改进

### 1. 添加配置验证

在训练开始前验证配置的一致性：
- `chaser_team_obs_dim * 3` 与环境实际返回的 `obs_chaser_team` 维度是否匹配
- `num_trainable_roles` 与 `ROLE_MAPPING` 的一致性

### 2. 添加详细的训练诊断日志

- 每个rollout的episode数量
- 每个环境的完成率
- buffer的填充状态

### 3. 考虑添加环境健康检查

在训练循环中定期检查：
- 环境是否还在响应
- step的reward是否异常（NaN/Inf）
- 进程是否还存活

### 4. 重构评估逻辑

当前 `evaluate()` 函数过长（130+行），建议拆分为：
- `evaluate_single_env()` - 单个环境的评估
- `evaluate_all_envs()` - 管理多个环境的评估
- `collect_eval_results()` - 收集和汇总结果

---

## 验证正确的实现

以下代码经验证为正确实现：

1. **`collect_rollout()` 的主循环逻辑** - 环境完成时正确移除并保存数据
2. **TD(λ) 优势估计** - 边界条件处理正确（终止状态next_value=0）
3. **RolloutBuffer.get_batch()** - 正确重置buffer位置并返回batch
4. **role_ids 和 actor_indices 的对应关系** - 代码逻辑确保两者顺序一致

---

## 总结

| 严重程度 | 数量 | 已修复 | 未修复 |
|---------|------|--------|--------|
| Critical | 5 | 3 | 2 |
| High | 4 | 0 | 4 |
| Medium | 4 | 0 | 4 |
| Low | 3 | 0 | 3 |

**修复状态**:
- ✓ Critical #1, #2, #3: 2026/04/19 已修复
- ⚠ Critical #4: checkpoint恢复后环境状态未同步 - 需架构调整
- ⚠ Critical #5: 未使用的onnx导入 - 死代码清理

**最关键的问题**是 `evaluate()` 函数中的索引错误，这会直接导致评估指标不准确。已修复。

**说明**：对应的内部修复记录未随公开版保留；本报告仅保留问题背景和已修复状态。
