[English](README.md) | [中文](README.zh-CN.md)

# FinsSim documentation

This index separates stable interfaces and operating procedures from dated
experiments and historical notes. Start with the category that matches your
task; documents under [`_archive/`](_archive/) are retained for traceability,
not as current instructions.

## Start here

- [Repository layout](overview/repository-layout.md)
- [RL backend usage](learning/training/rl-backends.md)
- [ROS 2 workspace](../ros2_ws/README.md)
- [ROS 2 package documentation](../ros2_ws/docs/README.md)
- [VS Code tasks (Chinese)](development/vscode-tasks.zh-CN.md)
- [FinsSimUnity](https://github.com/IWIN-FINS/FinsSimUnity)

## Reference contracts

- [Coordinate-frame convention audit](reference/coordinate-frames/convention-audit.zh-CN.md)
- [Coordinate-system unification](reference/coordinate-frames/system-unification.zh-CN.md)
- [Canonical thruster layout](reference/thrusters/canonical-layout.md)
- [Wrench capability](reference/thrusters/wrench-capability.zh-CN.md)
- [Chase body frame and observation](reference/observations/chase-body-frame-and-observation.zh-CN.md)

## Operations and perception

- [Calibration guide](operations/calibration/index.zh-CN.md)
- [Hardware bridge thruster mapping](operations/hardware/thruster-bridge-mapping.zh-CN.md)
- [AprilTag localization system](perception/apriltag-localization-system.zh-CN.md)
- [Underwater AprilTag research report](perception/research/underwater-apriltag-research-report.zh-CN.html)

## Simulation and learning

- [FinsROV Fossen profile identification](simulation/hydrodynamics/finsrov-fossen-profile-identification_20260812.zh-CN.md)
- [Unity–Fossen replay validator](simulation/validation/unity-fossen-replay-validator.zh-CN.md)
- [RL task documents](learning/rl/)
- [IRL pipeline](learning/irl/goal-yaw-pipeline.zh-CN.md)
- [MARL configuration loading](learning/marl/config-loading-instructions.html)
- [Reward protocols](learning/rewards/)
- [Water and wave domain randomization](learning/domain-randomization/water-and-wave.zh-CN.md)

## Experiments, governance, and archive

- [Hydrodynamics experiments](experiments/hydrodynamics/)
- [Hardware experiments](experiments/hardware/)
- [Training comparisons](experiments/training/)
- [Perception experiments](experiments/perception/)
- [License audit](governance/license-audit.zh-CN.md)
- [Historical command log](_archive/temp_command_line.md)

## Archive

[`_archive/`](_archive/) preserves historical command records and retired
migration pages. Its contents are not current operating instructions.
