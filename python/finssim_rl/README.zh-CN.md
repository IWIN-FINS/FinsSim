[English](README.md) | [中文](README.zh-CN.md)

# finssim_rl

`finssim_rl` 是 FinsSim 的单智能体 RL backend，负责与 Unity 联动的训练和评估工作流，底层主要基于 SB3 和若干控制器实验。

这个仓库有两种使用方式：

- 作为独立 backend，通过 `finssim-rl` 直接启动
- 作为 FinsSim 托管 backend，由顶层 `finssim rl ...` 启动

## 快速开始

```bash
cd /path/to/FinsSim/python/finssim_rl
uv sync

uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 5005
```

轻量测试启动可以直接加 `--test`，会强制：

```text
num_envs = 1
num_eval_envs = 1
exp_name 自动追加 _test 后缀
```

```bash
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev_test \
  --env-path /path/to/UnityBuild.x86_64 \
  --test
```

独立运行默认输出：

```text
artifacts/runs/rl/{exp_name}/
  logs/
  checkpoints/
  tensorboard/
```

从 FinsSim 顶层托管启动：

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --dry-run
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --test
```

推荐按任务成对维护 YAML：

- `configs/rl/<task>.yaml`：训练 YAML，通常指向 headless build
- `configs/rl/<task>_eval.yaml`：评估 YAML，通常指向 visual build，并显式指定要回放的单个 `checkpoint_path`

## 6D Wrench 策略训练

如果希望策略学习 6 个自由度上的合力/合矩，而不是直接学习 8 个推进器输出，可以使用 `ppo_wrench_for_pose_empirical_thruster_mixer` 配置。物理推力分配矩阵版本是 `ppo_wrench_for_pose_physical_wrench_allocator`：

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim rl train -c configs/rl/pose_control/ppo_wrench_for_pose.yaml --dry-run
uv run --package finssim-cli finssim rl train -c configs/rl/pose_control/ppo_wrench_for_pose.yaml
```

这个模式不会改动旧的 `ppo_control_for_pose` 八推力训练。训练时为了兼容当前 Unity 环境，Python 会把 6D 策略动作临时分配成 8D 推进器动作喂给仿真；保存和 ONNX 导出的策略头仍是 6D。

6D 策略输出是归一化动作：

```text
[surge, sway, heave, roll, pitch, yaw]
```

实机侧先裁剪到 `[-1, 1]`，再按训练配置中的 `virtual_control_limits` 缩放，并重排成解耦器输入：

```text
[Fx, Fy, Fz, Mx, My, Mz]
= [surge, heave, sway, roll, yaw, pitch]
```

默认限幅为：

```text
[surge, sway, heave, roll, pitch, yaw] = [200, 200, 200, 80, 80, 80]
```

ONNX 导出会在同目录生成 `*_info.txt`，其中也会记录这套轴顺序和换算关系。

## 文档入口

- Backend 使用说明：[English](docs/backend-usage.md) | [中文](docs/backend-usage.zh-CN.md)
- FinsSim 顶层工作流：[English](../../docs/learning/training/rl-backends.md) | [中文](../../docs/learning/training/rl-backends.zh-CN.md)

## 说明

- `finssim_rl` 当前使用自己独立的 Python 3.10 环境。
- 如果不传 `--output-dir`，输出会保留在 `python/finssim_rl/artifacts/` 下。
- 如果从 FinsSim 顶层启动，输出会被重定向到 `FinsSim/artifacts/`。
