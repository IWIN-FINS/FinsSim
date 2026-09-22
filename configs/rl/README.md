# configs/rl — 强化学习实验配置

本目录存放所有 RL 实验（训练 / 评估 / 测试）的 YAML 配置文件。每个 YAML 是一份
**实验覆盖配置**（experiment override），它通过 `base_config` 字段引用注册在后端
Python 包中的基础配置，再用 `experiment` / `unity` / `trainer` 等字段覆盖具体参数。

## 目录布局

```
configs/rl/
├── README.md              ← 本文件
├── example.yaml           ← 通用入门示例（最小可运行配置）
│
├── pose_control/          ← 位姿控制：固定 / 重置目标点训练
├── moving_target/         ← 移动目标跟踪
├── 1chase1/               ← 1v1 追逐任务
│   └── hierarchy/         ← 分层策略（高层 PPO + 低层 PID）的奖励变体
└── legacy_pid/            ← 早期 PID 混合方案实验（存档）
```

### 按任务场景分目录

新增配置时，**先按任务场景归入对应子目录**，不要直接放在根目录下：

| 子目录 | 适用场景 | 典型 Unity Scene 前缀 |
|---|---|---|
| `pose_control/` | ROV 到达 / 保持固定目标位姿 | `ControlForPosition_*` |
| `moving_target/` | 跟踪平滑运动的移动目标 | `ControlForPosition_Dynamic_*` |
| `1chase1/` | 单 ROV 追逐单目标鱼 | `FinsROV/1Chase1_*` |
| `legacy_pid/` | PID / hybrid 残差控制（历史存档） | `ControlForPosition_SmoothNearTarget_Manual_*` |

如果引入了全新的任务场景，请新建一个子目录并在本表中登记。

## 文件命名规范

### 通用模式

```
<algo>_<action_space>_for_<task>[_<scene_variant>][_dr|_v2][_eval|_test].yaml
```

### 字段说明

| 字段 | 含义 | 示例 |
|---|---|---|
| `algo` | RL 算法 | `ppo` |
| `action_space` | 动作空间类型 | `control`（直接推力器）、`wrench`（6D 力矩分配） |
| `task` | 任务目标 | `pose`、`moving_target` |
| `scene_variant` | Unity 场景变体 | `smooth_near_target`、`angular_stability`、`yaw_torque` |
| `dr` | 域随机化（Domain Randomization） | `_dr`、`_force_coefficient_dr`、`_random_mass_dr` |
| `v2` | 场景 / 配置的第二代（如八推 ROV） | `_v2` |
| `eval` / `test` | 用途后缀 | `_eval` = 可视化评估；`_test` = 最小冒烟测试 |

### 示例

```
ppo_control_for_pose.yaml                         # PPO + 直接推力器，固定目标点位姿控制
ppo_control_for_pose_yaw_torque_dr_v2.yaml        # 同上 + yaw 力矩场景 + 域随机化 + v2
ppo_wrench_for_pose_smooth_near_target_v2.yaml    # PPO + wrench 动作空间 + 平滑接近目标 + v2
ppo_control_for_moving_target.yaml                # 移动目标跟踪
ppo_control_for_pose_test.yaml                    # 冒烟测试（num_envs=1, test=true）
ppo_control_for_pose_yaw_torque_eval.yaml         # 可视化评估（visual build, checkpoint 回放）
```

## YAML 结构

每个配置文件遵循统一的结构：

```yaml
base_config: ppo_control_for_pose     # 必填：后端注册的基础配置名（非文件路径）

experiment:                            # 实验元信息
  name: ppo_control_for_pose_10Hz      # 运行名（决定 artifacts/runs/rl/ 下的目录名）
  seed: 42
  tags: [ppo, pose_control, rot6d]     # 自由标签，用于检索
  output_dir: artifacts/runs/rl/...    # 可选：指定输出目录（相对路径基于仓库根）

unity:                                 # Unity 运行时参数
  env_path: ../../artifacts/.../ControlForPosition.x86_64
  num_envs: 32                         # 并行环境数
  env_base_port: 5005                  # 基础端口（多实例时需错开）
  time_scale: 10.0                     # Unity 时间加速倍率
  no_graphics: true                    # headless 训练建议 true
  timeout_wait: 240                    # 等待 Unity 启动的超时（秒）

trainer:                               # 后端训练器参数
  backend: rl                          # 后端类型
  overrides:                           # 透传给后端 CLI 的覆盖项
    device: cuda
    overwrite: true
```

### 关键字段说明

- **`base_config`**：注册在 `finssim_rl` / `finssim_marl` 包中的基础配置**名字**（不是文件路径）。
  YAML 文件本身的路径只给 `finssim-cli` 的 `-c` 参数使用，移动文件不影响 `base_config` 解析。
- **`output_dir`**：建议使用相对路径（基于仓库根），按 `<task_group>/<variant>` 分层，
  与本目录的子目录结构保持一致。
- **`env_base_port`**：同时跑多个训练时必须保证端口互不冲突。
- **`reward`**（可选）：部分任务（如 `1chase1/hierarchy/`）在 YAML 中内联了奖励协议参数，
  详见 [奖励协议](../../docs/learning/rewards/)。

## 训练 / 评估 / 测试

```bash
# 训练
uv run --package finssim-cli finssim rl train \
  -c configs/rl/pose_control/ppo_control_for_pose_v2.yaml

# 冒烟测试（最小环境，快速验证管线）
uv run --package finssim-cli finssim rl train \
  -c configs/rl/pose_control/ppo_control_for_pose_test.yaml

# 可视化评估（visual build + checkpoint 回放）
uv run --package finssim-cli finssim rl eval \
  -c configs/rl/pose_control/ppo_control_for_pose_yaw_torque_eval.yaml
```

## 新增配置 Checklist

1. **归入正确的子目录** — 按任务场景选择，不要放根目录。
2. **遵循命名规范** — 使用 `<algo>_<action_space>_for_<task>[_variant].yaml`。
3. **指定 `output_dir`** — 与子目录结构对齐，便于回溯。
4. **检查端口冲突** — `env_base_port` 与现有配置错开（参考同目录其他文件）。
5. **加注释** — 如果配置对应特定 Unity Scene build，在文件头部注明构建命令
   （`FinsSimLinuxBuild.Build*Server`）和运行命令。
6. **`eval` / `test` 后缀** — 评估配置用 `_eval`，冒烟测试用 `_test`，不要混用。
