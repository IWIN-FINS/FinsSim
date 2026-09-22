[English](README.md) | [中文](README.zh-CN.md)

# finssim_rl

`finssim_rl` is the single-agent RL backend for FinsSim. It provides Unity-connected training and evaluation workflows built around SB3-based and controller-based experiments.

This repository can be used in two ways:

- as a standalone backend with `finssim-rl`
- as a FinsSim-managed backend launched by `finssim rl ...`

## Quick Start

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

Default standalone artifact layout:

```text
artifacts/runs/rl/{exp_name}/
  logs/
  checkpoints/
  tensorboard/
```

FinsSim-managed launch:

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --dry-run
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml
```

## Documentation

- Backend usage: [English](docs/backend-usage.md) | [中文](docs/backend-usage.zh-CN.md)
- Top-level FinsSim workflow: [English](../../docs/learning/training/rl-backends.md) | [中文](../../docs/learning/training/rl-backends.zh-CN.md)

## Notes

- `finssim_rl` currently uses its own Python 3.10 environment.
- If `--output-dir` is not provided, outputs stay inside `python/finssim_rl/artifacts/`.
- When launched from FinsSim top level, outputs are redirected to `FinsSim/artifacts/`.
