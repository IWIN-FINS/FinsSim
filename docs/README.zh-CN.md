[English](README.md) | [中文](README.zh-CN.md)

# FinsSim 文档地图

本文档将稳定的接口规范、操作手册、带日期的实验结果和历史记录分开管理。请从与
当前任务对应的分类进入；[`_archive/`](_archive/) 下的内容为可追溯性保留，并非
当前操作说明。

## 建议起点

- [仓库结构](overview/repository-layout.md)
- [RL 后端使用说明](learning/training/rl-backends.zh-CN.md)
- [ROS 2 工作区](../ros2_ws/README.md)
- [ROS 2 package 文档入口](../ros2_ws/docs/README.md)
- [VS Code Tasks 使用说明](development/vscode-tasks.zh-CN.md)
- [FinsSimUnity](https://github.com/IWIN-FINS/FinsSimUnity)

## 接口规范

- [坐标系链路审计](reference/coordinate-frames/convention-audit.zh-CN.md)
- [坐标系统一报告](reference/coordinate-frames/system-unification.zh-CN.md)
- [Canonical 推进器布局](reference/thrusters/canonical-layout.md)
- [Wrench 能力](reference/thrusters/wrench-capability.zh-CN.md)
- [Chase Body Frame 与 Observation](reference/observations/chase-body-frame-and-observation.zh-CN.md)

## 操作与感知

- [标定总入口](operations/calibration/index.zh-CN.md)
- [硬件 Bridge 推进器映射](operations/hardware/thruster-bridge-mapping.zh-CN.md)
- [AprilTag 水下定位系统](perception/apriltag-localization-system.zh-CN.md)
- [水下 AprilTag 调研报告](perception/research/underwater-apriltag-research-report.zh-CN.html)

## 仿真与学习

- [FinsROV Fossen profile 标定基线](simulation/hydrodynamics/finsrov-fossen-profile-identification_20260812.zh-CN.md)
- [Unity–Fossen 回放验证器](simulation/validation/unity-fossen-replay-validator.zh-CN.md)
- [RL 任务文档](learning/rl/)
- [IRL Pipeline](learning/irl/goal-yaw-pipeline.zh-CN.md)
- [MARL 配置加载说明](learning/marl/config-loading-instructions.html)
- [Reward protocols](learning/rewards/)
- [水流与物理波浪域随机化](learning/domain-randomization/water-and-wave.zh-CN.md)

## 实验、治理与归档

- [水动力实验](experiments/hydrodynamics/)
- [硬件实验](experiments/hardware/)
- [训练对比实验](experiments/training/)
- [感知实验](experiments/perception/)
- [许可证审计](governance/license-audit.zh-CN.md)
- [历史命令记录](_archive/temp_command_line.md)

## 归档

[`_archive/`](_archive/) 保留历史命令记录和已废弃的迁移页；其中内容不是当前操作
说明。新建链接和文档更新请使用本页列出的正式路径。
