[English](rl-backends.md) | [中文](rl-backends.zh-CN.md)

# FinsSim RL Backend 使用说明

这份文档说明重构后的 Python RL 栈如何使用。核心思路是：

- FinsSim 顶层负责实验 YAML、artifact 目录和 metadata。
- `finssim_rl` 与 `finssim_marl` 保留各自的训练循环和算法实现。
- 顶层 `finssim` 是 orchestration CLI，不直接承担训练运行时。

## 初始化

在 FinsSim 根目录执行：

```bash
git clone https://github.com/IWIN-FINS/FinsSim.git
cd FinsSim
git submodule update --init
uv sync
uv run --package finssim-cli finssim info
```

顶层 CLI 保持轻量，不会直接 import SB3、Torch、ML-Agents 或 MARL 训练栈。

两个 backend 继续使用各自独立环境：

```bash
cd python/finssim_rl
uv sync

cd ../finssim_marl
uv sync
```

`finssim_rl` 当前使用 Python 3.10，`finssim_marl` 当前使用 Python 3.12。

## FinsSim 顶层托管启动

使用 dry-run 预览 backend 命令，而不启动 Unity：

```bash
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --dry-run
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --dry-run
```

真实训练：

```bash
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml
```

轻量测试训练：

```bash
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --test
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --test
```

`--test` 会统一透传到 backend，并强制：

```text
num_envs = 1
num_eval_envs = 1
experiment.name / --exp-name 自动追加 _test 后缀
```

如果实验名已经以 `_test` 结尾，不会重复追加。

真实评估：

```bash
uv run --package finssim-cli finssim rl eval -c configs/rl/example.yaml
uv run --package finssim-cli finssim marl eval -c configs/marl/example.yaml
```

真实运行前，需要在 YAML 中设置 `unity.env_path`。

## YAML 格式

标准实验 YAML：

```yaml
base_config: control_for_pose

experiment:
  name: rl_example
  seed: 42
  tags: []
  output_dir: null

unity:
  env_path: null
  num_envs: 1
  env_base_port: 5005
  port_offset: 0
  time_scale: 10.0
  no_graphics: true

trainer:
  backend: rl
  overrides: {}
```

测试 YAML 推荐统一写：

```yaml
trainer:
  backend: rl   # 或 marl
  overrides:
    test: true
```

这样不用在 RL/MARL 各自 YAML 中分别维护 `num_envs` / `num_eval_envs` 的不同覆盖写法。

主要字段说明：

- `base_config`：backend 内部已有的 Python config 名称
- `experiment.name`：FinsSim 顶层 run 名称
- `experiment.seed`：公共随机种子
- `experiment.output_dir`：可选输出目录；相对路径按 FinsSim workspace root 解析，绝对路径原样使用
- `unity.env_path`：Unity 二进制路径
- `unity.num_envs`：并行 Unity 环境数
- `unity.env_base_port`：ML-Agents base port
- `unity.port_offset`：主要用于单智能体 RL evaluation 或 worker id
- `unity.time_scale`：Unity 仿真时间倍率
- `unity.no_graphics`：默认 headless
- `trainer.backend`：只能是 `rl` 或 `marl`
- `trainer.overrides`：原样传递给 backend CLI

覆盖优先级：

```text
backend defaults < backend Python config < FinsSim YAML < CLI/backend overrides
```

## Artifact 目录

顶层托管运行默认写到：

```text
artifacts/runs/{backend}/{experiment.name}/
```

如果设置了 `experiment.output_dir`，则使用该目录替代默认目录。例如：

```yaml
experiment:
  name: ppo_control_for_pose
  output_dir: artifacts/runs/rl/new_reward/ppo_control_for_pose
```

典型结构：

```text
artifacts/runs/rl/rl_example/
  command.txt
  metadata.json
  resolved_config.yaml
  logs/
  checkpoints/
  tensorboard/

artifacts/runs/marl/marl_example/
  command.txt
  metadata.json
  resolved_config.yaml
  logs/
  checkpoints/
  tensorboard/
```

`metadata.json` 至少包含 backend、config path、run name、Python version、seed、Unity env path、num envs、base port 和 git commit。

`resolved_config.yaml` 会先记录顶层 FinsSim YAML 展开结果；真实训练启动后，backend 会刷新该文件并追加 `backend_resolved`，其中包含最终合并后的 backend 参数，例如 RL 的 `args` / `model_config`，或 MARL 的 `train_config` / `base_config` / `algorithm_config`。

### 路径对照

| 启动方式 | logs | checkpoints | tensorboard |
| --- | --- | --- | --- |
| `finssim rl train` | `FinsSim/artifacts/runs/rl/{experiment.name}/logs` | `FinsSim/artifacts/runs/rl/{experiment.name}/checkpoints` | `FinsSim/artifacts/runs/rl/{experiment.name}/tensorboard` |
| `finssim marl train` | `FinsSim/artifacts/runs/marl/{experiment.name}/logs` | `FinsSim/artifacts/runs/marl/{experiment.name}/checkpoints/{base_config}__{experiment.name}` | `FinsSim/artifacts/runs/marl/{experiment.name}/tensorboard` |

RL 和 MARL 现在已经统一了 `logs` 和 `tensorboard` 的约定。MARL 的 checkpoint 仍然多一层子目录，因为它的 checkpoint manager 是按内部 run name 恢复训练的。

## Backend 独立启动

也可以直接进入 backend 目录独立运行。

单智能体 RL：

```bash
cd FinsSim/python/finssim_rl
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 5005
```

多智能体 MARL：

```bash
cd FinsSim/python/finssim_marl
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 6000
```

独立运行默认输出：

```text
python/finssim_rl/artifacts/runs/rl/{exp_name}/
python/finssim_marl/artifacts/runs/marl/{base_config}__{exp_name}/
```

### 独立运行路径对照

| 启动方式 | 默认 output dir | logs | checkpoints | tensorboard |
| --- | --- | --- | --- | --- |
| `finssim-rl train --exp-name rl_dev` | `python/finssim_rl/artifacts/runs/rl/rl_dev` | `{output_dir}/logs` | `{output_dir}/checkpoints` | `{output_dir}/tensorboard` |
| `finssim-marl train --config chasing_3_chase_1 --exp-name marl_dev` | `python/finssim_marl/artifacts/runs/marl/chasing_3_chase_1__marl_dev` | `{output_dir}/logs` | `{output_dir}/checkpoints/chasing_3_chase_1__marl_dev` | `{output_dir}/tensorboard` |

如果传入 `--output-dir`，这些路径会整体迁移到指定目录下。

## Backend 细分文档

- `finssim_rl`：[README](../../../python/finssim_rl/README.zh-CN.md)，[Usage](../../../python/finssim_rl/docs/backend-usage.zh-CN.md)
- `finssim_marl`：[README](../../../python/finssim_marl/README.zh-CN.md)，[Usage](../../../python/finssim_marl/docs/backend-usage.zh-CN.md)
