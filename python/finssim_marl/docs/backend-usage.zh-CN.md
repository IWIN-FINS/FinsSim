[English](backend-usage.md) | [中文](backend-usage.zh-CN.md)

# finssim_marl 使用说明

`finssim_marl` 是 FinsSim 的多智能体 RL backend，负责 MAPPO rollout、进度条、checkpoint 和 logger 集成。它既可以独立运行，也可以由顶层 `finssim` CLI 托管启动。

## 环境

```bash
cd /path/to/FinsSim/python/finssim_marl
uv sync
```

`finssim_marl` 当前使用自己独立的 Python 3.12 环境。

## CLI

推荐入口：

```bash
uv run finssim-marl train --help
uv run finssim-marl eval --help
```

兼容脚本入口仍可使用：

```bash
uv run python scripts/train.py --help
uv run python scripts/eval.py --help
```

## 训练

最小训练命令：

```bash
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 6000
```

常用参数：

- `--config`：backend Python config 名称
- `--exp-name`：实验名称
- `--env-path`：Unity 二进制路径
- `--num-envs`：并行训练 Unity 环境数
- `--num-eval-envs`：并行评估 Unity 环境数
- `--env-base-port`：ML-Agents base port
- `--time-scale`：训练时的 Unity 仿真倍率
- `--show-graphics`：显示 Unity 窗口
- `--test`：轻量测试模式，强制 `num_envs=1`、`num_eval_envs=1`，并给 `exp_name` 自动追加 `_test` 后缀
- `--auto-resume`：从当前 run 的最新 checkpoint 恢复
- `--resume-from`：从指定 checkpoint 路径恢复
- `--overwrite`：允许复用已有 run 目录
- `--output-dir`：手动指定 run 根目录

测试模式示例：

```bash
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev_test \
  --env-path /path/to/UnityBuild.x86_64 \
  --test
```

## 内部 Run Name

MARL 保留了内部算法 run name 约定：

```text
{base_config}__{exp_name}
```

例如：

```text
chasing_3_chase_1__marl_dev
```

这个内部名字会用于 checkpoint 恢复，因此 MARL 的 checkpoint 会比 RL 多一层嵌套目录。

## 独立运行 Artifact 结构

如果不传 `--output-dir`：

```text
python/finssim_marl/artifacts/runs/marl/{base_config}__{exp_name}/
  logs/
  checkpoints/
    {base_config}__{exp_name}/
  tensorboard/
```

例如：

```bash
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64
```

默认输出：

```text
artifacts/runs/marl/chasing_3_chase_1__marl_dev/logs
artifacts/runs/marl/chasing_3_chase_1__marl_dev/checkpoints/chasing_3_chase_1__marl_dev
artifacts/runs/marl/chasing_3_chase_1__marl_dev/tensorboard
```

如果传入 `--output-dir /tmp/finsim-marl-dev`：

```text
/tmp/finsim-marl-dev/logs
/tmp/finsim-marl-dev/checkpoints/chasing_3_chase_1__marl_dev
/tmp/finsim-marl-dev/tensorboard
```

## 评估

### 训练中的异步评估

`env.eval.mode: asynchronous` 会让评估在后台线程中运行，同时训练线程继续
采样和更新策略。`multi_area` 下仍然只启动一个专用的 eval Unity binary，内部
包含 `env.eval.num_envs` 个 replicated area。

每次调度评估时，训练线程先保存不可变 checkpoint 到：

```text
checkpoints/{base_config}__{experiment.name}/eval_snapshots/
```

后台 worker 只评估该快照；训练线程收到结果后，才将**同一份**快照提升为
`best.pt`，不会把已继续训练的 live model 错当作最佳模型。默认会删除所有已
完成的非最佳快照。如需离线分析保留完整快照历史，才在 YAML 中显式设置：

```yaml
trainer:
  overrides:
    keep_eval_snapshots: true
```

`env.eval.mode: serial` 保留原有的同步评估行为。

示例：

```bash
uv run finssim-marl eval \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --checkpoint-dir artifacts/runs/marl/chasing_3_chase_1__marl_dev/checkpoints \
  --num-envs 1 \
  --num-episodes 10 \
  --env-base-port 6200
```

如果训练时使用了自定义 `--output-dir`，评估通常指向其中的 `checkpoints` 目录，再在其下按内部 run name 查找具体 checkpoint。

## FinsSim 顶层托管启动

在顶层工作区中：

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --dry-run
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --test
```

也可以在顶层 YAML 中统一写：

```yaml
trainer:
  backend: marl
  overrides:
    test: true
```

托管输出：

```text
FinsSim/artifacts/runs/marl/{experiment.name}/
  command.txt
  metadata.json
  resolved_config.yaml
  logs/
  checkpoints/
    {base_config}__{experiment.name}/
  tensorboard/
```

## 说明

- checkpoint 多出来的一层子目录是有意保留的，目前属于 MARL 的恢复训练语义。
- 多个任务并行时，优先错开 `--env-base-port`。
- 评估 worker 也会占用端口，因此需要为它们预留额外空间。
