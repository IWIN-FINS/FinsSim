# FinsROV Physical Wrench 满量程速度实验

本文记录 `physical_wrench_allocator` 在已运行的 Unity Play 场景中，经 gRPC 和
ROS2 实际下发的满量程速度响应。它是运行时实验结果，不是理论 wrench 能力表。

## 实验范围

- 日期：2026-08-16
- Unity：前台 Play 中的 `FinsROV_Fossen`，不 reset 载具、不启停 Agent。
- 链路：`send_wrench_action` -> `/sim/finsrov/thrusters_out` -> gRPC adapter -> Unity。
- 观测：`/sim/finsrov/controller/dvl`、`/sim/finsrov/controller/imu`、
  `/sim/finsrov/controller/pose`。
- 分配配置：
  [ppo_wrench_for_pose_physical_wrench_allocator_sim.yaml](../../../ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator_sim.yaml)。
- policy action 顺序：`[surge, sway, heave, roll, pitch, yaw]`。
- `action=+/-1` 的物理 wrench 上限：
  `[19.528527 N, 18.415027 N, 22.501886 N, 3.863995 N*m, 3.079985 N*m, 7.080833 N*m]`。
- 每个推进器的最终 force-N 边界：`+/-7 N`。

每个方向以 1 秒从零升至满量程，保持约 8 秒；线速度和角速度均取命令保持阶段的
最后 2 秒中位数作为稳态值。实验退出时工具发布零推力，且确认
`/sim/finsrov/thrusters_out` 没有残留 publisher。

## 结果

| 自由度 | 峰值 | 最后 2 秒稳态 | 说明 |
|---|---:|---:|---|
| Surge `+` | `0.4998 m/s` | `0.4998 m/s` | 与负向对称 |
| Surge `-` | `0.4998 m/s` | `0.4998 m/s` | 与正向对称 |
| Sway `+` | `0.3520 m/s` | `0.3520 m/s` | 与负向对称 |
| Sway `-` | `0.3520 m/s` | `0.3520 m/s` | 与正向对称 |
| Heave `+` | `0.1325 m/s` | 约 `0 m/s` | 到达水面边界后的结果，不代表自由水体上限 |
| Heave `-` | `0.2196 m/s` | `0.2196 m/s` | 自由下潜方向 |
| Roll `+/-` | `0.3092/0.3066 rad/s` | 约 `0 rad/s` | 恢复力矩使姿态稳定，不是持续自转能力 |
| Pitch `+/-` | `0.2420/0.2378 rad/s` | 约 `0 rad/s` | 同上 |
| Yaw `+/-` | `3.5372 rad/s` | `3.5372 rad/s` | 约 `202.7 deg/s` |

满 `+surge` 的实际 allocator 输出为：

```text
target wrench: [Fx=+19.5285 N, Fy=0, Fz=0, Mx=0, My=0, Mz=0]
force_N: [+1.6497, -1.6497, -1.6497, +1.6497,
          +7.0000, +7.0000, -6.8087, -6.8087]
```

这表示水平推进器接近饱和，同时垂直推进器小幅参与以抵消实际质心和推进器位置造成的
非目标俯仰力矩。分配后满足纯 `Fx=19.5285 N`。

## 结论

1. 当前 physical allocator 没有 `0.4 m/s` 的速度限幅；它只限目标 wrench 和每路 `+/-7 N` 推力。
2. 旧 `+/-10 N` 虚拟 wrench 实验的 surge 稳态为 `0.1888 m/s`；更新为物理 wrench 上限后为
   `0.4998 m/s`。
3. 1Chase1 wrench baseline 在目标距离约 2 m 时，位置 P 项给出 `Fx=8*2=16 N`，理论上会低于满
   `19.5285 N` 的速度；同时航向误差超过 `15 deg` 时会先转向并把 `Fx` 置零。因此 chase 中约
   `0.4 m/s` 的 ROV 速度符合控制律，不是 allocator 的额外裁剪。
4. 实机辨识目录中的 `1.8` 是若干 CSV 行的 `phase_elapsed_sec`，不是速度。当前完整数据中最大
   `|nu_x|=0.5106 m/s`，三轴线速度范数最大 `0.5607 m/s`；本次 Unity surge `0.4998 m/s` 与其量级一致。

## 原始结果

本次 trial 的 `summary.json`、`summary.csv` 和运行日志不随公开仓库发布；上方表格
保留其可公开复核的汇总结果。复跑命令会在本地 `artifacts/` 下生成同类文件。
- `*_sender.log`：每个方向的 policy action、目标 wrench、归一化推进器命令和 force-N。
- `thruster_command.csv`：实验期间八推进器实际下发命令。

## 复跑

先在 Unity 启动目标场景并点击 Play，确认 VehicleROSBridge 和 gRPC adapter 已运行。随后执行：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh python ../tools/run_sim_wrench_speed_experiment.py \
  --axes all \
  --config src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator_sim.yaml \
  --output-dir ../artifacts/runs/diagnostics/sim_wrench_speed_physical
```

该工具不 reset Unity，也不激活或关闭任何场景脚本；它会依次执行全部 12 个单轴正负方向测试，
所以应在无其他 `/sim/finsrov/thrusters_out` publisher 的条件下运行。
