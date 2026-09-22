[English](backend-usage.md) | [中文](backend-usage.zh-CN.md)

# finssim_marl Backend Usage

`finssim_marl` is the multi-agent RL backend for FinsSim. It is responsible for MAPPO rollout, progress bars, checkpointing, and logger integration. It can run standalone or be launched by the top-level `finssim` CLI.

## Environment

```bash
cd /path/to/FinsSim/python/finssim_marl
uv sync
```

`finssim_marl` currently uses its own Python 3.12 environment.

## CLI

Recommended entrypoints:

```bash
uv run finssim-marl train --help
uv run finssim-marl eval --help
```

Legacy script entrypoints still work:

```bash
uv run python scripts/train.py --help
uv run python scripts/eval.py --help
```

## Training

Minimal training command:

```bash
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64 \
  --num-envs 1 \
  --env-base-port 6000
```

Common flags:

- `--config`: backend Python config name
- `--exp-name`: experiment name
- `--env-path`: Unity binary path
- `--num-envs`: number of parallel training Unity environments
- `--num-eval-envs`: number of parallel evaluation Unity environments
- `--env-base-port`: ML-Agents base port
- `--time-scale`: training Unity simulation time scale
- `--show-graphics`: show the Unity player window
- `--auto-resume`: resume from the latest checkpoint for the current run
- `--resume-from`: resume from an explicit checkpoint path
- `--overwrite`: allow reuse of an existing run directory
- `--output-dir`: choose a custom run root

## Internal Run Name

MARL keeps the internal algorithm run naming convention:

```text
{base_config}__{exp_name}
```

Example:

```text
chasing_3_chase_1__marl_dev
```

That internal name is used for checkpoint resume, which is why MARL checkpoints keep one extra nested directory.

## Standalone Artifact Layout

If `--output-dir` is not provided:

```text
python/finssim_marl/artifacts/runs/marl/{base_config}__{exp_name}/
  logs/
  checkpoints/
    {base_config}__{exp_name}/
  tensorboard/
```

Example:

```bash
uv run finssim-marl train \
  --config chasing_3_chase_1 \
  --exp-name marl_dev \
  --env-path /path/to/UnityBuild.x86_64
```

Default output:

```text
artifacts/runs/marl/chasing_3_chase_1__marl_dev/logs
artifacts/runs/marl/chasing_3_chase_1__marl_dev/checkpoints/chasing_3_chase_1__marl_dev
artifacts/runs/marl/chasing_3_chase_1__marl_dev/tensorboard
```

If `--output-dir /tmp/finsim-marl-dev` is provided:

```text
/tmp/finsim-marl-dev/logs
/tmp/finsim-marl-dev/checkpoints/chasing_3_chase_1__marl_dev
/tmp/finsim-marl-dev/tensorboard
```

## Evaluation

### Asynchronous training evaluation

`env.eval.mode: asynchronous` runs evaluation in a background thread while the
training thread continues collecting rollouts and updating the policy. For
`multi_area`, this still launches exactly one dedicated evaluation Unity binary
containing `env.eval.num_envs` replicated areas.

Each scheduled evaluation first writes an immutable checkpoint under:

```text
checkpoints/{base_config}__{experiment.name}/eval_snapshots/
```

The worker evaluates that exact policy, and the training thread promotes the
same snapshot to `best.pt` only after its result is received. By default, all
completed non-best snapshots are deleted. Retain them for offline analysis only
when explicitly requested:

```yaml
trainer:
  overrides:
    keep_eval_snapshots: true
```

`env.eval.mode: serial` retains the historical synchronous evaluation behavior.

Example:

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

If a custom `--output-dir` was used for training, evaluation normally points to its `checkpoints` directory and then resolves the internal run name underneath it.

## FinsSim-Managed Launch

From the top-level workspace:

```bash
cd /path/to/FinsSim
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml --dry-run
uv run --package finssim-cli finssim marl train -c configs/marl/example.yaml
```

Managed output:

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

## Notes

- The extra checkpoint subdirectory is intentional and currently part of MARL resume semantics.
- When multiple jobs run in parallel, stagger `--env-base-port`.
- Evaluation workers also consume ports, so leave extra room for them.
