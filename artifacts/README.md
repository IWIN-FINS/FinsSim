# Artifacts Layout

This directory stores generated or runtime assets that should live outside the source trees.

Recommended usage:

- `unity_builds/`: exported Unity binary apps for training, evaluation, and demos.
- `runs/`: FinsSim-managed experiment outputs created by `finssim rl/marl ...`.
- `exports/`: exported deployable assets such as ONNX models.
- `reports/`: generated summaries, plots, and experiment reports.

Suggested Unity binary layout:

```text
artifacts/unity_builds/
  rl/
    linux/
    windows/
  marl/
    linux/
    windows/
  sandbox/
    linux/
```

Suggested experiment layout:

```text
artifacts/runs/
  rl/
  marl/
```

Real build outputs and checkpoints should remain ignored by git. Only this scaffold and `.gitkeep` files are tracked.
