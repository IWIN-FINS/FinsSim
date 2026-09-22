# HoldForPosition Fossen 与 Isaac WarpAUV 训练/DR 对比

## 目的

比较 Unity `HoldForPosition_Fossen` 的 Fossen 训练与参考项目
`reference/simulator/isaac-auv-env` 的 WarpAUV position-hold 训练。比较不使用
“回合数”作为主要指标，因为两个任务的回合上限不同：Unity 为 60 秒，Isaac 为
3 秒。

统一比较口径如下：

- **policy transitions**：策略与环境交互的决策步总数。
- **累计仿真时间**：所有并行环境的 `policy transitions * policy dt` 之和。
- **DR 抽样数下界**：按最大回合长度计算的最少 reset 次数。若发生越界早停，真实
  reset/DR 抽样数只会更高。
- **独立 DR 抽样**：不同 worker 真正使用不同随机种子的 DR 参数组合数量。

## 任务时间尺度

| 环境 | 物理步长 | 决策频率/周期 | policy dt | 单回合最大时长 | 单回合最大 policy steps |
|---|---:|---:|---:|---:|---:|
| Unity HoldForPosition Fossen | `0.02 s` | `DecisionPeriod=5` | `0.1 s` | `60 s` | `600` |
| Isaac WarpAUV | `1/120 s` | `decimation=2` | `1/60 s` | `3 s` | `180` |

Unity 的 `MaxStep=3000` 是物理 Academy step 上限；以 50 Hz Physics 和 10 Hz
决策频率换算后，对应最多 600 个 RL policy transitions。

## 可比训练量

下表只计算 HoldForPosition 任务期间产生的步数；`fossen_init` 从旧 checkpoint
warm-start 的历史步数不并入 Hold 任务经验。

| 训练/模型 | 并行环境 | Hold policy transitions | 累计仿真时间 | 每 worker DR reset 下界 | 全局 DR reset 下界 |
|---|---:|---:|---:|---:|---:|
| Unity `ppo_wrench_for_hold_position_fossen` 最新 checkpoint | 32 | 3,099,008 | 309,900.8 s / 86.1 h | 161 | 5,165 |
| Unity `ppo_wrench_for_hold_position_fossen_init` 最新 checkpoint | 32 | 3,049,024 | 304,902.4 s / 84.7 h | 159 | 5,082 |
| Isaac WarpAUV README 所称约 400 iteration 收敛量 | 2,048 | 19,660,800 | 327,680.0 s / 91.0 h | 53 | 109,227 |
| 提供的 Isaac `model_10000` checkpoint | 2,048* | 491,520,000 | 8,192,000.0 s / 2,275.6 h | 1,333 | 2,730,667 |

计算式：

```text
累计仿真时间 = policy transitions * policy dt
每 worker reset 下界 = policy transitions / (num_envs * max_policy_steps_per_episode)
全局 reset 下界 = policy transitions / max_policy_steps_per_episode
```

`model_10000` 的 iteration 编号来自 checkpoint 内的 `model_10000/...` 路径。其
`2,048` 个环境来自参考仓库 README 的训练命令；checkpoint 本身没有保存实际训练时
传入的 `num_envs`，因此该行是基于该仓库标准命令的估算。

## Domain Randomization 覆盖

### Unity 历史 run

两个已经完成的 Unity Hold run 都发生在 `-fins-dr-seed` 传递链修复之前。虽然场景中
`DomainRandomizationCoordinator` 配置为读取该命令行参数，但 FinsSim YAML 的
`experiment.seed` 没有传给 `finssim-rl`，所以 Unity 进程实际没有收到参数，所有
worker 都从 `baseSeed=12345` 的同一条确定性序列开始。

因此，若各 worker 的回合节奏近似同步：

| Run | 全局 reset 下界 | 历史 run 可视为的独立 Fossen DR 序列位置 | 修复后同等训练量的独立 DR 抽样 |
|---|---:|---:|---:|
| `ppo_wrench_for_hold_position_fossen` 最新 checkpoint | 5,165 | 约 161 | 约 5,165 |
| `ppo_wrench_for_hold_position_fossen_init` 最新 checkpoint | 5,082 | 约 159 | 约 5,082 |

若不同 worker 因越界早停而使 episode index 发生分叉，历史 run 的实际独立参数组合会
略高于该估计，但它不受控制、不可复现，也不能按 32 倍并行数计算。

### 修复后的 Unity 启动规则

RL launcher 现在将有效的 `experiment.seed` 传到 `finssim-rl`。每个 worker 的 Unity
命令行参数为：

```text
-fins-dr-seed (base_seed + worker_rank * 1,000,003)
```

例如 `base_seed=42` 时，worker 0 为 `42`，worker 1 为 `1000045`。这样每个 worker
都有不同、可复现且相互不重叠的 Fossen DR 序列。`unity_additional_args` 不应再手动
设置 `-fins-dr-seed`，该保留参数会被 launcher 拒绝以避免覆盖 worker seed。

相关实现：

- `python/finssim_cli/src/finssim_cli/main.py`
- `python/finssim_rl/src/finssim_rl/models/__init__.py`

## Isaac 与 Unity 的结论

1. 以累计物理仿真时间计，Unity 当前约 3.1M step 的训练（86.1 h）与 Isaac README
   声称约 400 iteration 的收敛训练（91.0 h）基本相当。
2. Isaac 的 2,048 个并行环境和 3 秒回合，使其在相近累计仿真时间内至少进行了约
   109,227 次 DR reset；修复后的 Unity 当前 run 对应约 5,165 次。这是约 21 倍的
   DR 参数抽样数量差。
3. 提供的 Isaac `model_10000` 若确实按 README 的 2,048 环境训练，则相对 Unity
   当前 3.1M-step run 多约 159 倍 policy transitions、26 倍累计仿真时间、529 倍独立
   DR reset 抽样。
4. Isaac 的 DR 维度较窄：代码中主要随机化 COM-to-COB 偏移（半径 0.05 m）与体积
   （`0.01975` 到 `0.02575 m^3`）。Unity Fossen profile 的参数维度更广，但需要依靠
   修复后的独立 worker seed 和足够的 reset 数，才能形成有效覆盖。

## 建议

- 不将历史 Hold run 的 DR 覆盖量按 32 倍并行数解释。
- 后续 Fossen DR 训练使用修复后的 launcher，并固定记录 `experiment.seed`、
  `num_envs`、`policy dt`、最大回合长度和实际 reset 计数。
- 若目标是接近 Isaac 的 DR 抽样量，优先缩短训练 episode 或增加并行环境；仅增加同一
  60 秒回合内的 policy steps 不会增加 DR 参数组合数量。
