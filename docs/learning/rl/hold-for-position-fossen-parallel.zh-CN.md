# HoldForPosition Fossen 单 Binary 并行训练

`HoldForPosition_Fossen_Parallel.unity` 使用 ML-Agents 的 `TrainingAreaReplicator`。
Python 的 `unity.num_areas` 是一个 Unity binary 内的 area 数；`unity.num_envs` 仍表示 binary 数。并行 area 模式要求 `num_envs: 1`。

## Unity 操作

在 Unity 中打开 MARUS 项目后，执行：

1. `FinsSim > RL > Prepare HoldForPosition Fossen Parallel Scene`。
2. 打开 `Assets/Scenes/HoldForPosition_Fossen_Parallel.unity`，执行 `FinsSim > RL > Validate HoldForPosition Fossen Parallel Scene`。
3. 执行静态方法 `FinsSimLinuxBuild.BuildHoldForPositionFossenParallelServer` 构建 Linux Server。

生成器从原 Fossen scene 创建副本，不会修改 `HoldForPosition_Fossen.unity`。每个复制 area 包含独立的 ROV、目标、水流提供器与域随机化协调器；area 间距为 32 m。

## 训练与验证

默认 2048-area 直接推力训练：

```bash
uv run --package finssim-cli finssim rl train \
  -c configs/rl/hold_for_position/ppo_control_for_hold_position_fossen_parallel.yaml
```

先进行扩容阶梯验证，而非直接加载 2048：

```bash
cd python/finssim_rl
uv run python scripts/smoke_training_areas.py \
  --env-path ../../artifacts/unity_builds/rl/linux/HoldForPosition_Fossen_Parallel_10Hz_Server/HoldForPosition_Parallel.x86_64 \
  --areas 4 256 512 1024 2048
```

任一层失败时脚本停止并输出启动时间、吞吐和 Unity RSS。4-area wrench PPO 烟测可附加 `--num-areas 4` 到 `scripts/smoke_wrench_training.py`。

单区可视化评估使用 parallel eval 配置的 `num_areas: 1`；无图形批量评估可在命令中加 `--num-areas 256`（或其他通过阶梯验证的规模）。

## 本机实测容量（2026-08-14）

以下数据来自 Linux Server binary 的无图形零动作步进；RSS 是 Unity Player
进程的 `psutil` resident set size，而非整机已用内存。吞吐的 `agent steps/s`
等于 Unity 每次向量步进速度乘以 area 数，因此是训练时更有参考价值的数字。

| `num_areas` | 启动时间 | 步进数 | agent steps/s | Unity RSS | 结果 |
| ---: | ---: | ---: | ---: | ---: | --- |
| 4 | 58.184 s | 8 | 401.798 | 294.7 MB | 通过 |
| 256 | 60.380 s | 8 | 5,437.512 | 4,145.3 MB | 通过 |
| 512 | 70.243 s | 4 | 3,902.114 | 8,127.9 MB | 通过 |
| 1024 | 196.214 s | 3 | 2,712.559 | 16,414.8 MB | 通过 |
| 2048 | — | — | — | — | 失败：Linux OOM killer 终止 Unity Player（SIGKILL） |

2048-area 尝试时，内核记录的 Unity Player `anon-rss` 为约 22.9 GiB
（`24,023,920 kB`），总虚拟地址空间约 34.2 GiB；当时 2 GiB swap 已用满。
因此，虽然正式配置保留 2048 作为目标默认值，本机实际训练请显式设为
`--num-areas 1024`（或配置中 `unity.num_areas: 1024`）。不要直接在本机启动
2048-area 训练；如更换内存更大的主机，应先重新运行上面的阶梯测试。

## 并行接口与旧多进程模式的区别

`unity.num_areas` 表示一个 Unity Player 内由 `TrainingAreaReplicator` 创建的
area/Agent 数，`unity.num_envs` 始终表示 Unity Player binary 的数量。
实现特意保留了两条互斥路径：

| 配置 | Unity 进程 | Python VecEnv | 适用场景 |
| --- | --- | --- | --- |
| `num_areas: 1`，`num_envs: N` | N 个 binary | 原有 `ManagedSubprocVecEnv` | 原有多 Unity 进程并行，完全保持兼容 |
| `num_areas: M (>1)`，`num_envs: 1` | 1 个 binary | `UnityTrainingAreasVecEnv` | 一个 binary 内 M 个复制 area |

第二种模式会把 ML-Agents 同一 behavior 下的 M 个 `AgentId` 固定映射成 SB3 的
M 个 VecEnv slot：动作按当前 `AgentId` 回填，决策包按固定 slot 重排；各 area
独立终止时会保留其 `terminal_observation` 与 `TimeLimit.truncated`，并等待所有
slot 获得下一次可训练的决策。`num_areas > 1` 且 `num_envs != 1` 会直接报错，避免
无意中启动 `num_envs × num_areas` 个环境。

## 推荐的本机 15 秒 DR 训练

本机实测 2048 area 会被 OOM killer 终止，因此正式训练使用
`ppo_wrench_for_hold_position_fossen_parallel_1024_15s.yaml` 与对应的 15 秒
Server binary。该场景把 `MaxStep` 设为 750 个 physics step（50 Hz），即 150 个
10 Hz policy transition；它由 Unity 场景生成器创建，不手改 scene YAML。

配置使用 `num_areas: 1024`、`n_steps: 64`、`batch_size: 4096`，所以每个 PPO
rollout 是 `65,536` 个聚合 transition、每 epoch 为 16 个 mini-batch。训练使用
1024 area；周期性评估另启动一个 Unity Player，`num_eval_areas: 1`，不会 reset
训练中的 rollout。`checkpoint_freq: 819200` 为总步数的 5%，`eval_freq: 327680`
为总步数的 2%；两者都由回调按训练 VecEnv 的 1024 slot 换算为向量步次数。
`total_timesteps: 16,384,000` 恰好是 250 个 rollout；在没有越界早停时对应约
109,227 次独立 DR episode 抽样，与 Isaac WarpAUV 对比报告的 400 iteration
目标相当。训练前仍应先运行 4 → 256 → 512 → 1024 的 binary 阶梯烟测。

### 15 秒 Server 复测（2026-08-15）

以下数据来自新构建的
`HoldForPosition_Fossen_Parallel_15s_10Hz_Server`。测试以零动作为输入、
`time_scale: 10`、8 个向量步进运行；场景的 `MaxStep` 已验证为 750（即 15 秒）。

| `num_areas` | 启动时间 | agent steps/s | Unity RSS | 结果 |
| ---: | ---: | ---: | ---: | --- |
| 4 | 126.331 s | 425.922 | 295.9 MiB | 通过 |
| 256 | 130.964 s | 2,341.158 | 4,150.7 MiB | 通过 |

256-area 测试结束后主机仍有约 88 GiB 可用内存；这证明新 15 秒场景、单 binary
多 Agent ID 的 VecEnv 映射以及无图形步进链路均可用。该结果只是当前二级阶梯，
1024-area 正式训练前仍需按 512、1024 顺序复测，不能以 256 的内存占用线性外推
为 2048-area 放行。
# 并行运行时配置（2026-08 重构）

并行规模只使用角色下的 `num_envs`：`env.unity.parallel_mode: multi_area`
表示单个 Unity Player 的 area 数，`multi_binary` 表示 Unity Player 数。训练与评估
分别位于 `env.train`、`env.eval`；评估默认 `asynchronous`，使用独立 Unity Player，按
FIFO 顺序评估模型快照且不暂停训练。两者的 worker ID 会从同一基址自动划分为不重叠区间。
