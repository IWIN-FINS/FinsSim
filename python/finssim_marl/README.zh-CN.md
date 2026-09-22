[English](README.md) | [中文](README.zh-CN.md)

# finssim_marl

`finssim_marl` 是 FinsSim 的多智能体 RL backend，负责与 Unity 联动的 MARL 训练和评估工作流，当前主要围绕 MAPPO 风格训练栈。

这个仓库有两种使用方式：

- 作为独立 backend，通过 `finssim-marl` 直接启动
- 作为 FinsSim 托管 backend，由顶层 `finssim marl ...` 启动

## 快速开始

```bash
cd /path/to/FinsSim/python/finssim_marl
uv sync

uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 6000
```

轻量测试启动可以直接加 `--test`，会强制：

```text
num_envs = 1
num_eval_envs = 1
exp_name 自动追加 _test 后缀
```

```bash
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev_test \
  --env-path /path/to/UnityBuild.x86_64 \
  --test
```

独立运行默认输出：

```text
artifacts/runs/marl/{base_config}__{exp_name}/
  logs/
  checkpoints/
    {base_config}__{exp_name}/
  tensorboard/
```

从 FinsSim 顶层托管启动：

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --dry-run
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --test
```

## 文档入口

- Backend 使用说明：[English](docs/backend-usage.md) | [中文](docs/backend-usage.zh-CN.md)
- 3Chase1 课程学习与 reward 参数化：[中文](docs/3chase1_curriculum_environment_parameters.zh-CN.md)
- FinsSim 顶层工作流：[English](../../docs/learning/training/rl-backends.md) | [中文](../../docs/learning/training/rl-backends.zh-CN.md)

## 说明

- `finssim_marl` 当前使用自己独立的 Python 3.12 环境。
- 如果不传 `--output-dir`，输出会保留在 `python/finssim_marl/artifacts/` 下。
- MARL 的 checkpoint 默认会保留额外一层 `{base_config}__{exp_name}` 子目录，这是设计使然。
