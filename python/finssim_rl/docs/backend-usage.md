[English](backend-usage.md) | [中文](backend-usage.zh-CN.md)

# finssim_rl Backend Usage

`finssim_rl` is the single-agent RL backend for FinsSim. It is responsible for SB3-based and controller-based single-agent training logic. It can run standalone or be launched by the top-level `finssim` CLI.

## Environment

```bash
cd /path/to/FinsSim/python/finssim_rl
uv sync
```

`finssim_rl` currently uses its own Python 3.10 environment.

## CLI

Recommended entrypoints:

```bash
uv run finssim-rl train --help
uv run finssim-rl eval --help
uv run finssim-rl export --help
```

Legacy script entrypoints still work:

```bash
uv run python scripts/train.py --help
uv run python scripts/eval.py --help
uv run python scripts/export_to_onnx.py --help
```

## Training

Minimal training command:

```bash
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 5005
```

Common flags:

- `--config`: backend Python config name
- `--exp-name`: experiment name and default artifact run name
- `--env-path`: Unity binary path
- `--num-envs`: number of parallel Unity environments
- `--env-base-port`: ML-Agents base port
- `--port-offset`: worker id / port offset
- `--time-scale`: Unity simulation time scale
- `--show-graphics`: show the Unity player window
- `--resume`: resume from the latest checkpoint
- `--overwrite`: overwrite an existing output directory
- `--output-dir`: choose a custom run root

## Standalone Artifact Layout

If `--output-dir` is not provided:

```text
python/finssim_rl/artifacts/runs/rl/{exp_name}/
  logs/
  checkpoints/
  tensorboard/
```

Example:

```bash
uv run finssim-rl train \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64
```

Default output:

```text
artifacts/runs/rl/rl_dev/logs
artifacts/runs/rl/rl_dev/checkpoints
artifacts/runs/rl/rl_dev/tensorboard
```

If `--output-dir /tmp/finsim-rl-dev` is provided:

```text
/tmp/finsim-rl-dev/logs
/tmp/finsim-rl-dev/checkpoints
/tmp/finsim-rl-dev/tensorboard
```

## Evaluation

### Asynchronous evaluation snapshots

Asynchronous evaluation queues immutable checkpoints in
`checkpoints/eval_snapshots` while a dedicated Unity Player evaluates them.
The default is to delete each snapshot after evaluation and retain only the
current `checkpoints/best_model.zip`. To retain every completed snapshot for
offline analysis, add the following to the experiment YAML:

```yaml
trainer:
  overrides:
    keep_eval_snapshots: true
```

With the default (`false`), residual queued snapshots are also removed during
normal callback shutdown after an interrupted or failed evaluation.

Example:

```bash
uv run finssim-rl eval \
  --config control_for_pose \
  --exp-name rl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --checkpoint-path artifacts/runs/rl/rl_dev/checkpoints/best_model.zip \
  --num-episodes 5
```

`--checkpoint-path` points to one checkpoint file directly. It may be
`best_model.zip`, `final_model.zip`, or any intermediate checkpoint file.

## FinsSim-Managed Launch

From the top-level workspace:

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml --dry-run
uv run --package finssim-cli finssim rl train -c configs/rl/example.yaml
```

Managed output:

```text
FinsSim/artifacts/runs/rl/{experiment.name}/
  command.txt
  metadata.json
  resolved_config.yaml
  logs/
  checkpoints/
  tensorboard/
```

## Notes

- Adjust `--env-base-port` or `--port-offset` when multiple jobs run in parallel.
- `--env-path` can come from either CLI or backend config, but explicit CLI values are usually easier to reproduce.
