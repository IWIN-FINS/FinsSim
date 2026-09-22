[English](rl-backends.md) | [中文](rl-backends.zh-CN.md)

# FinsSim RL Backend Usage

This document describes how the refactored Python RL stack is used. The core idea is:

- FinsSim top level manages experiment YAML, artifact directories, and metadata.
- `finssim_rl` and `finssim_marl` keep their own training loops and algorithm implementations.
- Top-level `finssim` is an orchestration CLI, not a trainer runtime.

## Initialization

At the FinsSim root:

```bash
git clone https://github.com/IWIN-FINS/FinsSim.git
cd FinsSim
git submodule update --init
uv sync
uv run --package finssim-cli finssim info
```

The top-level CLI is intentionally lightweight. It does not import SB3, Torch, ML-Agents, or the MARL training stack.

Backend environments stay separate:

```bash
cd python/finssim_rl
uv sync

cd ../finssim_marl
uv sync
```

`finssim_rl` currently uses Python 3.10, while `finssim_marl` uses Python 3.12.

## FinsSim-Managed Launch

Preview backend commands without launching Unity:

```bash
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --dry-run
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --dry-run
```

Real training:

```bash
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml
```

Lightweight test training:

```bash
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --test
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --test
```

`--test` is forwarded to the backend and forces:

```text
num_envs = 1
num_eval_envs = 1
experiment.name / --exp-name gets an automatic _test suffix
```

If the experiment name already ends with `_test`, the suffix is not duplicated.

Real evaluation:

```bash
uv run --package finssim-cli finssim rl eval -c configs/rl/example.yaml
uv run --package finssim-cli finssim marl eval -c configs/marl/example.yaml
```

Before real runs, set `unity.env_path` in the YAML config.

## YAML Schema

Standard experiment YAML:

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

Recommended test YAML form:

```yaml
trainer:
  backend: rl   # or marl
  overrides:
    test: true
```

This avoids keeping separate RL/MARL `num_envs` and `num_eval_envs` override patterns in test configs.

Meaning of the main fields:

- `base_config`: existing backend Python config name
- `experiment.name`: top-level FinsSim run name
- `experiment.seed`: shared random seed
- `experiment.output_dir`: optional output directory; relative paths are resolved from the FinsSim workspace root and absolute paths are used as-is
- `unity.env_path`: Unity binary path
- `unity.num_envs`: number of parallel Unity environments
- `unity.env_base_port`: ML-Agents base port
- `unity.port_offset`: primarily used by single-agent RL evaluation / worker ids
- `unity.time_scale`: Unity simulation time scale
- `unity.no_graphics`: headless by default
- `trainer.backend`: `rl` or `marl`
- `trainer.overrides`: passed through to the backend CLI as-is

Override priority:

```text
backend defaults < backend Python config < FinsSim YAML < CLI/backend overrides
```

## Artifact Layout

Top-level managed runs default to:

```text
artifacts/runs/{backend}/{experiment.name}/
```

When `experiment.output_dir` is set, that directory replaces the default. For example:

```yaml
experiment:
  name: ppo_control_for_pose
  output_dir: artifacts/runs/rl/new_reward/ppo_control_for_pose
```

Typical layout:

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

`metadata.json` includes backend, config path, run name, Python version, seed, Unity env path, num envs, base port, and git commit.

`resolved_config.yaml` first records the top-level FinsSim YAML expansion. When real training starts, the backend refreshes the same file and appends `backend_resolved`, which contains the final merged backend parameters, such as RL `args` / `model_config` or MARL `train_config` / `base_config` / `algorithm_config`.

### Path Reference

| Launch Path | logs | checkpoints | tensorboard |
| --- | --- | --- | --- |
| `finssim rl train` | `FinsSim/artifacts/runs/rl/{experiment.name}/logs` | `FinsSim/artifacts/runs/rl/{experiment.name}/checkpoints` | `FinsSim/artifacts/runs/rl/{experiment.name}/tensorboard` |
| `finssim marl train` | `FinsSim/artifacts/runs/marl/{experiment.name}/logs` | `FinsSim/artifacts/runs/marl/{experiment.name}/checkpoints/{base_config}__{experiment.name}` | `FinsSim/artifacts/runs/marl/{experiment.name}/tensorboard` |

RL and MARL now share the same `logs` and `tensorboard` convention. MARL still keeps one extra checkpoint subdirectory because its checkpoint manager resumes by internal run name.

## Backend Standalone Launch

You can also launch each backend directly.

Single-agent RL:

```bash
cd FinsSim/python/finssim_rl
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 5005
```

Multi-agent MARL:

```bash
cd FinsSim/python/finssim_marl
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 6000
```

Standalone default output:

```text
python/finssim_rl/artifacts/runs/rl/{exp_name}/
python/finssim_marl/artifacts/runs/marl/{base_config}__{exp_name}/
```

### Standalone Path Reference

| Launch Path | Default output dir | logs | checkpoints | tensorboard |
| --- | --- | --- | --- | --- |
| `finssim-rl train --exp-name rl_dev` | `python/finssim_rl/artifacts/runs/rl/rl_dev` | `{output_dir}/logs` | `{output_dir}/checkpoints` | `{output_dir}/tensorboard` |
| `finssim-marl train --config chasing_3_chase_1 --exp-name marl_dev` | `python/finssim_marl/artifacts/runs/marl/chasing_3_chase_1__marl_dev` | `{output_dir}/logs` | `{output_dir}/checkpoints/chasing_3_chase_1__marl_dev` | `{output_dir}/tensorboard` |

If `--output-dir` is provided, those paths are relocated under the chosen directory.

## Backend-Specific Docs

- `finssim_rl`: [README](../../../python/finssim_rl/README.md), [Usage](../../../python/finssim_rl/docs/backend-usage.md)
- `finssim_marl`: [README](../../../python/finssim_marl/README.md), [Usage](../../../python/finssim_marl/docs/backend-usage.md)
