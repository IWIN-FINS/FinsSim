[English](backend-usage.md) | [中文](backend-usage.zh-CN.md)

# finssim_rl 使用说明

`finssim_rl` 是 FinsSim 的单智能体 RL backend，负责基于 SB3 和控制器的单智能体训练逻辑。它既可以独立运行，也可以由顶层 `finssim` CLI 托管启动。

## 环境

```bash
cd /path/to/FinsSim/python/finssim_rl
uv sync
```

`finssim_rl` 当前使用自己独立的 Python 3.10 环境。

## CLI

推荐入口：

```bash
uv run finssim-rl train --help
uv run finssim-rl eval --help
uv run finssim-rl export --help
```

兼容脚本入口仍可使用：

```bash
uv run python scripts/train.py --help
uv run python scripts/eval.py --help
uv run python scripts/export_to_onnx.py --help
```

## 训练

最小训练命令：

```bash
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 5005
```

常用参数：

- `--config`：backend Python config 名称
- `--exp-name`：实验名，也是默认 artifact run 名
- `--env-path`：Unity 二进制路径
- `--num-envs`：并行 Unity 环境数
- `--env-base-port`：ML-Agents base port
- `--port-offset`：worker id / 端口偏移
- `--time-scale`：Unity 仿真时间倍率
- `--show-graphics`：显示 Unity 窗口
- `--test`：轻量测试模式，强制 `num_envs=1`、`num_eval_envs=1`，并给 `exp_name` 自动追加 `_test` 后缀
- `--resume`：从最新 checkpoint 恢复
- `--overwrite`：覆盖已有输出目录
- `--output-dir`：手动指定 run 根目录

测试模式示例：

```bash
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev_test \
  --env-path /path/to/UnityBuild.x86_64 \
  --test
```

## 独立运行 Artifact 结构

如果不传 `--output-dir`：

```text
python/finssim_rl/artifacts/runs/rl/{exp_name}/
  logs/
  checkpoints/
  tensorboard/
```

例如：

```bash
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64
```

默认输出：

```text
artifacts/runs/rl/rl_dev/logs
artifacts/runs/rl/rl_dev/checkpoints
artifacts/runs/rl/rl_dev/tensorboard
```

如果传入 `--output-dir /tmp/finsim-rl-dev`：

```text
/tmp/finsim-rl-dev/logs
/tmp/finsim-rl-dev/checkpoints
/tmp/finsim-rl-dev/tensorboard
```

## 评估

### 异步评估快照

异步评估会先将不可变模型快照写到 `checkpoints/eval_snapshots`，再由专用
Unity Player 从队列中读取并评估。默认在评估完成后删除该快照，仅保留当前
最优的 `checkpoints/best_model.zip`。如需为离线分析保留每一次已完成评估的
快照，在实验 YAML 中显式设置：

```yaml
trainer:
  overrides:
    keep_eval_snapshots: true
```

默认值为 `false`；正常回调退出时也会清理因中断或失败评估留下的队列快照。

示例：

```bash
uv run finssim-rl eval \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --checkpoint-path artifacts/runs/rl/rl_dev/checkpoints/best_model.zip \
  --num-episodes 5
```

`--checkpoint-path` 直接指向一个 checkpoint 文件，可以是
`best_model.zip`、`final_model.zip`，也可以是任意中途保存的 checkpoint。

## FinsSim 顶层托管启动

在顶层工作区中：

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --dry-run
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --test
```

也可以在顶层 YAML 中统一写：

```yaml
trainer:
  backend: rl
  overrides:
    test: true
```

托管输出：

```text
FinsSim/artifacts/runs/rl/{experiment.name}/
  command.txt
  metadata.json
  resolved_config.yaml
  logs/
  checkpoints/
  tensorboard/
```

## 说明

- 多个任务并行运行时，优先调整 `--env-base-port` 或 `--port-offset`。
- `--env-path` 可以来自 CLI，也可以放在 backend config 里，但显式 CLI 参数通常更利于复现实验。
