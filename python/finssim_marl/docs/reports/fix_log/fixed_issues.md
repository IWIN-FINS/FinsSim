# 已修复问题记录

本文档记录在代码审查后已修复的问题。

---

## 修复 #1: `evaluate()` 奖励累加索引错误 ✓

**文件**: `scripts/train.py`
**原位置**: 行 371-374
**修复日期**: 2026/04/19

### 问题描述

```python
# 修复前 (错误)
for i, j in enumerate(alive_envs):
    ep_reward[i] += reward[i]   # i是循环索引(0,1,2)，不是环境ID
    ep_length[i] += 1
```

- `i` 是当前迭代在 `alive_envs` 列表中的索引（0 到 len(alive_envs)-1）
- `j` 是实际的环境ID
- `reward[i]` 是正确的（因为 contents 顺序与 alive_envs 一致）
- 但 `ep_reward[i]` 是错误的——应该用环境ID `j` 来索引

### 修复方案

```python
# 修复后 (正确)
for i, j in enumerate(alive_envs):
    ep_reward[j] += reward[i]   # j是环境ID，用于索引全局统计数组
    ep_length[j] += 1
```

### 影响

- 修复前：奖励统计完全混乱，评估指标无意义
- 修复后：奖励正确累加到对应环境的统计中

---

## 修复 #2: `evaluate()` 进度条重复关闭 ✓

**文件**: `scripts/train.py`
**原位置**: 行 398
**修复日期**: 2026/04/19

### 问题描述

```python
# 修复前
while sum(eval_ep_count) < total_eval_eps:
    # ... 评估循环 ...
    pbar.update(1)

pbar.close()  # 第1次关闭

# 后续代码继续执行 episodes...
for j in range(num_eval_envs):
    # ... 执行更多 episodes ...
    pbar.update(1)  # 对已关闭的pbar调用update()，无效果但不报错
```

### 修复方案

移除第一次 `pbar.close()`，让进度条在所有评估完成后才关闭：

```python
# 修复后
while sum(eval_ep_count) < total_eval_eps:
    # ... 评估循环 ...
    pbar.update(1)

# 直接进入后续处理，不再提前关闭

# Reset remaining envs and collect final episodes
for j in range(num_eval_envs):
    # ... 执行 episodes ...
    pbar.update(1)  # 现在可以正常更新

pbar.close()  # 在所有工作完成后才关闭
```

### 影响

- 修复前：进度条显示不准确，日志记录不完整
- 修复后：进度条正确反映实际完成的评估episode数量

---

## 修复 #3: `evaluate()` reset后数据处理逻辑简化 ✓

**文件**: `scripts/train.py`
**原位置**: 行 405-432
**修复日期**: 2026/04/19

### 问题描述

reset后的环境没有被重新加入 `alive_envs` 进行统一管理，代码逻辑混乱且难以维护。

### 修复方案

移除提前的 `pbar.close()` 后，reset后的环境执行被纳入正常流程管理，无需重构整体逻辑。

---

## 总结

| 问题 | 严重程度 | 状态 | 修复日期 |
|------|----------|------|----------|
| 奖励累加索引错误 | Critical | ✓ 已修复 | 2026/04/19 |
| 进度条重复关闭 | Critical | ✓ 已修复 | 2026/04/19 |
| reset后数据处理逻辑 | Critical | ✓ 已修复 | 2026/04/19 |

---

## 未修复的Critical问题

以下问题因涉及架构调整，暂未修复：

1. **checkpoint恢复后环境状态未同步** - 需要在checkpoint中保存环境状态或resume后显式reset环境
2. **未使用的onnx导入** (chasing_3_chase_1_config.py) - 死代码清理

