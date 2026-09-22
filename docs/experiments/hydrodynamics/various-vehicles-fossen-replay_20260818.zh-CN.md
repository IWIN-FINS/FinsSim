# VariousVehiclesNewHydrodynamics Fossen 实机回放对比

日期：2026-08-18

## 实验目的

在前台 Unity Editor 的 `Assets/Scenes/VariousVehiclesNewHydrodynamics.unity`
中，使用 `FinsROV_Fossen` 的 `Fossen6Dof` backend，直接回放实机辨识记录中的
8 路实际推进器力，比较 Unity 和实机的三轴平动、yaw rate，并检查 roll/pitch
响应。

本实验没有使用 `send_wrench_action`，也没有再次经过 ROS `ThrustAllocator`，因此
不会把 allocator 或 force-limit 配置差异混入水动力对比。

## 运行配置

- Unity 对象：`FinsROV_Fossen`
- 水动力：`HydrodynamicsController.mode = Fossen6Dof`
- DWP2 WaterObject：对本 parametric backend 禁用/置零，不作为第二个动态水动力源
- `VehicleRosBridge.thrusterCommandMode = ForceN`
- `ThrusterController.ForceResponseCompressionEnabled = false`
- Unity 内部 topic 仍为 `/finsrov/...`，gRPC adapter 映射为 `/sim/finsrov/...`
- 状态：`/sim/finsrov/controller/dvl`、`/sim/finsrov/controller/imu`
- 实际合力/力矩：`/sim/finsrov/debug/thruster_applied_wrench`

`thruster_applied_wrench` 的 `linear=[Fx,Fy,Fz]` 和 `angular=[Mx,My,Mz]`
是 Unity 从 8 个 `Thruster.GetAppliedWorldForce()` 汇总后的机体系量，才是本实验
用于动力学比较的 Unity 输入。`/sim/finsrov/thrusters_out` 只用于记录回放的
8 路请求力。

## 实验命令

先按 `marus-example/AGENTS.md` 启动 adapter，并等待 Unity 话题稳定：

```bash
cd ./ros2_ws
./scripts/launch_grpc_ros_adapter.sh topic_prefix:=/sim
./scripts/run_ros2_uv.sh ros2 topic info /sim/finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 topic hz /sim/finsrov/controller/imu --window 20
```

每个轴的采集由两个进程组成。先启动 recorder，再启动回放器；回放器每个 trial
前发送 `/sim/finsrov/reset`，按 CSV 原始时间间隔直接发布 8 路 `force_cmd_*_n`：

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control record_real_sim_diagnostics \
  --duration 105 \
  --output-dir ./ros2_ws/data/hydrodynamics/unity/various_vehicles_new_fossen_20260818/surge_x/surge_x_001

./scripts/run_ros2_uv.sh ros2 run motion_control replay_unity_hydrodynamics \
  --csv ./ros2_ws/data/hydrodynamics/real/hydrodynamic_identification/surge_x/surge_x_015/surge_x_015.csv \
  --output-dir ./ros2_ws/data/hydrodynamics/unity/various_vehicles_new_fossen_20260818/surge_x/surge_x_001 \
  --settle-sec 1 --between-trials-sec 1 --rate 50
```

其余轴使用同样命令，替换 CSV 和目录：

| 轴 | 实机参考 CSV | Unity sweep |
|---|---|---|
| `surge_x` | `real/.../surge_x/surge_x_015/surge_x_015.csv` | `+/-3,+/-6,+/-9,+/-12,+/-18 N` |
| `sway_z` | `real/.../sway_z/sway_z_004/sway_z_004.csv` | `+/-3,+/-6,+/-9,+/-12,+/-18 N` |
| `heave_y` | `real/.../heave_y/heave_y_015/heave_y_015.csv` | `-6,-9,-10,-12 N` |
| `yaw_y` | `real/.../yaw_y/yaw_y_007/yaw_y_007.csv` | `+/-1,+/-2,+/-4,+/-6,+/-8 N*m` |
| `roll_x` | `real/.../roll_x/roll_x_009/roll_x_009.csv` | `+/-0.5,+/-0.8,+/-1.0,+/-1.5 N*m` 的实机脉冲 |
| `pitch_z` | `real/.../pitch_z/pitch_z_003/pitch_z_003.csv` | `+/-1,+/-1.5,+/-2,+/-2.5,+/-3 N*m` 的实机脉冲 |

后处理命令：

```bash
./scripts/run_ros2_uv.sh python3 scripts/analyze_unity_fossen_hydrodynamics.py \
  --unity-root ./ros2_ws/data/hydrodynamics/unity/various_vehicles_new_fossen_20260818 \
  --real-root ./ros2_ws/data/hydrodynamics/real/hydrodynamic_identification
```

## 数据组织

```text
ros2_ws/data/hydrodynamics/unity/various_vehicles_new_fossen_20260818/
  surge_x/surge_x_001/
  sway_z/sway_z_001/
  heave_y/heave_y_001/
  roll_x/roll_x_001/
  pitch_z/pitch_z_001/
  yaw_y/yaw_y_001/
  unity_fossen_vs_real_summary.csv
  all_axes_wrench_and_response.png
  analysis_report.json
```

每个 trial 目录包含：

- `replay_commands.csv`：回放的 8 路实际力和 trial/phase
- `sim_thruster_applied_wrench__*.csv`：Unity 实际合力/力矩
- `sim_controller_dvl__*.csv`：Unity 机体系线速度
- `sim_controller_imu__*.csv`：Unity 机体系角速度和 IMU 加速度
- `unity_vs_real_trial_summary.csv`：每个幅值的实机/Unity 汇总
- `applied_wrench_vs_response.png`
- `velocity_acceleration_vs_time.png`

## 结果

### Surge

Unity 的 applied `Fx` 与回放目标基本一致，说明 horizontal thruster 的
`ForceN` 合成在 surge 方向是正确的。Unity 速度响应为：

- 低幅值约为实机的 `0.6~0.8`；
- `18 N` 时 Unity 约 `0.448 m/s`，实机约 `0.443 m/s`（正向），负向 Unity
  约 `0.449 m/s`、实机约 `0.395 m/s`。

当前 profile 的 surge 线性阻尼为 `39.073593`，与 Unity 自身 sweep 得到的
`Fx -> steady rate` 关系一致。低幅值差异更像实机 `tau_fit`/RPM 和短时间窗口
的估计误差，不应优先修改 Unity surge damping。

### Sway

Unity 的 applied `Fz` 与目标基本一致，方向也正确。Unity rate 大约是实机的
`0.77` 倍：例如 `18 N` 时 Unity 约 `0.330 m/s`，实机约 `0.357 m/s`。
当前 sway 线性阻尼为 `52.307030`，Unity 自身拟合约为 `54.5`，因此 Unity
内部的 Fossen damping 已经自洽；若要追实机，应优先复核实机 `tau_fit_fz_n`
和 DVL 的低幅值/稳态窗口，再小幅降低 sway damping，而不是修改 thruster
布局。

### Heave

Unity applied `Fy` 与目标接近，但速度明显大于实机：Unity 峰值约
`0.27~0.31 m/s`，实机约 `0.032~0.048 m/s`。这不是单纯的浮力静态配平问题：
profile 的 `displacedVolume=0.0118 m^3`、`waterDensity=1027 kg/m^3` 对应浮力
约 `118.8 N`，与 `12.11 kg` 的重力约相等，静态上是配平的。

更可能的原因是：

1. Fossen backend 使用显式有限差分 added-mass force；加速度滤波和初始加速度阶段
   尚未建立等效 added mass，短的 heave excitation 会先按接近刚体质量响应。
2. 实机 heave 测试是短脉冲加 coast，`tau_fit_fy_n`、DVL 速度和时间窗口并非
   与 Unity 的峰值定义完全同一物理量。
3. Unity 记录的 IMU 加速度是 specific force；分析图中已去掉静态 `+9.81 m/s^2`
   偏置，但实际拟合仍应使用统一的运动学加速度定义。

因此下一步应使用更长的 heave step 或把实机/Unity 都按同一初始加速度窗口比较，
再决定是调整 `addedMass.w`、`linear/quadratic w damping` 还是只修正回放测试窗口。

### Yaw

Yaw 是当前最接近的一轴：

- `1 N*m`：Unity applied 约 `0.962 N*m`，rate 约 `1.10 rad/s`，实机约
  `0.93 rad/s`；
- `8 N*m`：Unity applied 约 `7.08 N*m`，rate 约 `3.54 rad/s`，实机约
  `3.74 rad/s`。

Unity 的曲线随幅值上升与实机一致，当前 profile 的 yaw damping
`d_linear=0.3003185`、`d_quadratic=0.47763214` 已能产生接近的非线性趋势。
高幅值 Unity applied torque 小于命令/实机 `tau_fit`，主要来自 Unity
thruster 的实际限幅/几何力矩，不是额外叠加一个 DWP2 yaw 阻力。

### Roll

这是最重要的异常：用实机相同的垂向 8 路力回放时，Unity 产生的 applied `Mx`
符号与实机推力矩阵定义相反。例如正向 roll trial 中 Unity 约 `-0.442 N*m`
而实机矩阵对应正向 `Mx`。这说明当前不能直接用这组 roll 数据估计水动力阻尼，
因为输入 wrench 在推进器合成层已经不一致。

进一步检查发现，这不是 Rigidbody 质心造成的。`FinsROV_Fossen.prefab` 中四个
垂向推进器的正向实际力都汇成机体 `+Y`，其位置对应 Unity 几何得到的 `Mx`
系数约为：

```text
[V_LF, V_LB, V_RB, V_RF] -> [-0.133, -0.133, +0.143, +0.143] N*m/N
```

但辨识节点
`ros2_ws/src/hydrodynamic_identification/hydrodynamic_identification/hydrodynamic_identifier_node.py`
中的 `WRENCH_FROM_THRUST` 使用的是：

```text
[+0.156, +0.156, -0.156, -0.156] N*m/N
```

即同一 canonical force 列表的 roll 力矩符号相反；`python/finssim_rl` 的
`ThrustAllocator.B` 也采用 Unity 几何的负号版本。这是当前 roll 对比不能闭合的
坐标/矩阵契约问题，应先统一辨识侧的 `WRENCH_FROM_THRUST`、roll target allocator
和历史数据解释，再重做 roll sweep。不要通过修改 Unity 的 Fossen roll damping
或 Rigidbody `centerOfMass` 来掩盖这个输入矩阵符号错误。

Unity roll rate 峰值约 `0.15~0.42 rad/s`，实机记录约 `0.73~1.66 rad/s`，但
该差异首先应在统一 `Mx` 符号和大小后再判断。当前 Fossen profile 中 roll
参数为：

```text
linear damping p = 1
quadratic damping p = 2
added mass p = 0.1
```

零力 coast 的粗略回归只能说明实际衰减量级为数值上约 `1~数 N*m*s/rad`，受
恢复力、惯量和小角度噪声影响很大，不能把该回归当作最终标定值。

### Pitch

Pitch 的 Unity applied `Mz` 符号与实机一致，但 Unity 实际力矩略小：例如正向
`1 N*m` trial 约 `0.829 N*m`。Unity pitch rate 峰值约 `0.25~0.62 rad/s`，
实机约 `0.55~1.28 rad/s`。当前 profile 中 pitch 参数为：

```text
linear damping q = 1
quadratic damping q = 2
added mass q = 0.1
```

pitch 的响应差异可能来自实际 applied torque 大小、恢复力/惯量和实机 pulse
窗口；与 roll 不同，它没有首先暴露出符号相反问题。仍建议在统一 `Mz` 后重新做
更长的 pitch free-decay，再独立拟合 `I_eff、d_linear、d_quadratic、K_restore`。

## 结论和优先级

1. `yaw`：当前 Fossen 水动力曲线已经基本对齐，优先保留参数；不要再叠加 yaw
   torque overlay。
2. `surge/sway`：Unity 的 applied wrench 正确，主要差异是实机测得响应/推力曲线
   在低中幅值的非线性与辨识窗口，不是 DWP2 或重复 backend。
3. `heave`：需要单独重新设计同窗口实验，重点检查显式 added-mass 初始瞬态和
   heave damping；静态浮力本身不是首要嫌疑。
4. `roll`：先修正同一 8 路实机力在 Unity 中的 `Mx` 符号/推进器方向映射，之后
   才能谈 roll damping。
5. `pitch`：先校准 `Mz` 实际合成大小，再用 free-decay 同时估计惯量、恢复力和
   damping；不要直接把当前不稳定实机 roll/pitch fit 值写进 Fossen profile。

## 可复现工具

回放工具：

```text
ros2_ws/src/motion_control/motion_control/unity_hydrodynamic_replay.py
```

后处理工具：

```text
ros2_ws/scripts/analyze_unity_fossen_hydrodynamics.py
```

本次只新增 ROS2 记录/回放/分析工具，没有修改 DWP2 源码，也没有修改 Unity
scene 或 prefab。
