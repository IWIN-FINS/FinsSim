# MARLlib Integration Plan

This document records the FinsSim -> MARLlib benchmark boundary.

## File Responsibilities

- `third_party/MARLlib/marllib/envs/base_env/unity_3chase1.py`
  - Owns the RLlib `MultiAgentEnv` wrapper for Unity 3Chase1.
  - Converts ML-Agents `UnityParallelEnv` observations/actions to MARLlib/RLlib dictionaries.
  - Exposes Herder and Netter as learning agents.
  - Controls Prey through `prey_action_source`, defaulting to `wrapper_escape`.

- `third_party/MARLlib/marllib/envs/base_env/config/unity_3chase1.yaml`
  - Provides MARLlib's default environment registration config.
  - Keeps Unity runtime defaults and action-source defaults.

- `scripts/marllib_train.py`
  - FinsSim-owned launcher for MARLlib benchmarks.
  - Converts FinsSim CLI/YAML arguments into `marl.make_env`, `marl.algos.*`, `marl.build_model`, and `fit`.
  - Writes `marllib_result.json` into the FinsSim artifact directory.

- `configs/marl/marllib_3chase1_*.yaml`
  - User-facing benchmark configs for direct CLI runs.
  - Use `trainer.backend: marllib` so the top-level CLI launches the MARLlib path.

## Current Benchmark Caveat

The current MARLlib fork has a Ray-2 compatibility runner in `marllib/marl/algos/modern.py`.
At the moment, that runner provides a common rollout/smoke path and should not yet be
treated as definitive learning evidence for MAPPO/IPPO/HAPPO/MADDPG. The next integration
step is to replace or extend that runner with real RLlib `AlgorithmConfig` learners for
the algorithms we want to benchmark.

## CLI

Preview:

```bash
uv run --package finssim-cli finssim marllib train -c configs/marl/marllib_3chase1_smoke.yaml --dry-run
```

Run:

```bash
xvfb-run --auto-servernum --server-args='-screen 0 1280x1024x24' \
uv run --package finssim-cli finssim marllib train -c configs/marl/marllib_3chase1_mappo.yaml
```
