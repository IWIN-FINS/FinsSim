[English](README.md) | [中文](README.zh-CN.md)

# finssim_marl

`finssim_marl` is the multi-agent RL backend for FinsSim. It provides Unity-connected MARL training and evaluation workflows built around the current MAPPO-style stack.

This repository can be used in two ways:

- as a standalone backend with `finssim-marl`
- as a FinsSim-managed backend launched by `finssim marl ...`

## Quick Start

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

Default standalone artifact layout:

```text
artifacts/runs/marl/{base_config}__{exp_name}/
  logs/
  checkpoints/
    {base_config}__{exp_name}/
  tensorboard/
```

FinsSim-managed launch:

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --dry-run
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml
```

## Documentation

- Backend usage: [English](docs/backend-usage.md) | [中文](docs/backend-usage.zh-CN.md)
- Top-level FinsSim workflow: [English](../../docs/learning/training/rl-backends.md) | [中文](../../docs/learning/training/rl-backends.zh-CN.md)

## Notes

- `finssim_marl` currently uses its own Python 3.12 environment.
- If `--output-dir` is not provided, outputs stay inside `python/finssim_marl/artifacts/`.
- MARL checkpoints keep an extra `{base_config}__{exp_name}` subdirectory by design.
