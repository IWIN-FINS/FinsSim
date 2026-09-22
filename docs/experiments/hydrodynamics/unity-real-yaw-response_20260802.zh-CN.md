# Unity DWP2 与实机 yaw 响应对比记录

日期：2026-08-02

场景：`marus-example/Assets/Scenes/VariousVehicles.unity`

目标：在 Unity 当前 DWP2 水动力配置下，通过 ROS2 向仿真发送和实机 yaw sweep 相同的力矩命令，采集 yaw rate，和已有实机大范围 yaw 数据对比。

## 数据位置

实机基准数据：

```text
./ros2_ws/data/hydrodynamics/real/yaw_response_real_large_sweep_005_run_only
```

Unity 有效采集数据：

```text
./ros2_ws/data/hydrodynamics/unity/identifier/variousvehicles_yaw_current_play_identity_axes_20260802
```

Unity 第一次错误坐标映射采集数据，仅作诊断证据，不用于拟合结论：

```text
./ros2_ws/data/hydrodynamics/unity/identifier/variousvehicles_yaw_current_play_wrong_axis_20260802
```

最终对比分析输出：

```text
./ros2_ws/data/hydrodynamics/analysis/yaw_response/real_yaw005_vs_unity_variousvehicles_identity_axes_20260802
```

关键文件：

```text
actual_tau_vs_steady_yaw_rate.png
actual_tau_vs_yaw_rate_damping_curve.png
yaw_response_metrics_comparison.png
unity_vs_real_yaw_trial_summary.csv
unity_vs_real_yaw_report.json
```

## gRPC adapter 启动与连接检查

Unity Editor 点 Play 后，`RosConnection` 和 `VehicleRosBridge` 不会瞬间完成连接。adapter 启动、Unity 重连、bridge 注册 sensor stream 和 thruster subscription 都需要几秒钟。

启动 adapter：

```bash
cd ./ros2_ws
./scripts/launch_grpc_ros_adapter.sh topic_prefix:=/sim
```

检查 topic：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic list | sort | grep /sim/finsrov
./scripts/run_ros2_uv.sh ros2 topic info /sim/finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 topic hz /sim/finsrov/controller/imu --window 20
```

本次有效连接状态：

```text
/sim/finsrov/thrusters_out: grpc_ros_adapter 是 subscriber
/sim/finsrov/controller/imu: grpc_ros_adapter 是 publisher
/sim/finsrov/controller/dvl: grpc_ros_adapter 是 publisher
IMU 频率约 25 Hz
```

Unity Editor log 中看到：

```text
Connected to the ROS Server
[VehicleRosBridge] Listening for thrusters on /finsrov/thrusters_out.
```

注意：Unity 内部 topic 仍是 `/finsrov/...`，`/sim` 前缀由 `grpc_ros_adapter` 在 ROS2 侧添加。

## probe 的作用

`FinsROVDwp2YawRuntimeProbe` 不是采集数据源，也不是必须打开才能得到 ROS `yaw_rate`。

它的作用只是 runtime 诊断：

- 找到当前 vehicle 下的 DWP2 `WaterObject`。
- 显示 `hydrodynamicAxisScalingEnabled` 是否启用。
- 显示 yaw damping mode、`hydrodynamicTorqueAxisScale`、`yawLinearDamping`、`yawQuadraticDamping`。
- 打印 DWP2 内部 raw/scaled/target/applied/final body yaw hydrodynamic torque。
- 打印推进器合成出来的 body yaw torque。

因此，之前 yaw_rate “没有数据/很小”的主要原因不是 probe 没开，而是 Unity controller IMU 的坐标映射被按实机映射读取了。

## 坐标映射问题

实机 hydrodynamic identifier 默认配置使用：

```yaml
ros_to_body_basis_indices: [0, 2, 1]
ros_to_body_basis_signs: [1.0, 1.0, 1.0]
```

这是给实机 ROS/FinsSim 链路准备的。

但是当前 Unity `VehicleRosBridge` 的 `/sim/finsrov/controller/imu` 发布的是 Unity local body frame 原始顺序：

```text
angular_velocity = transform.InverseTransformDirection(rigidbody.angularVelocity)
```

所以 Unity controller topic 做水动力辨识时应使用：

```bash
-p ros_to_body_basis_indices:="[0,1,2]" -p ros_to_body_basis_signs:="[1.0,1.0,1.0]"
```

错误映射表现：

```text
yaw 激励主要落到 nu_pitch_z_radps
nu_yaw_radps 只有很小响应
```

本次有效结论基于 identity axes 这组数据。

## Unity yaw sweep 采集命令

本次使用和实机大 sweep 对齐的 yaw torque levels：

```text
[0.3, 0.6, 0.9, 1.2, 1.5, 1.8, 2.4, 3.0, 3.6, 4.5] Nm
```

有效采集命令：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification hydrodynamic_identifier --ros-args \
  -p auto_start:=true \
  -p axes:="[yaw_y]" \
  -p amplitudes:="[0.3,0.6,0.9,1.2,1.5,1.8,2.4,3.0,3.6,4.5]" \
  -p include_negative:=true \
  -p hold_sec:=3.0 \
  -p baseline_sec:=0.5 \
  -p pre_zero_sec:=0.5 \
  -p rest_sec:=0.5 \
  -p post_zero_sec:=0.5 \
  -p sample_sec:=1.5 \
  -p output_directory:=./ros2_ws/data/hydrodynamics/unity/identifier/variousvehicles_yaw_current_play_identity_axes_20260802 \
  -p output_prefix:=unity_dwp2_yaw_current_identity_axes \
  -p thruster_topic:=/sim/finsrov/thrusters_out \
  -p imu_topic:=/sim/finsrov/controller/imu \
  -p dvl_topic:=/sim/finsrov/controller/dvl \
  -p motor_rpm_topic:=/sim/finsrov/hardware/motor_rpm_raw \
  -p ros_to_body_basis_indices:="[0,1,2]" \
  -p ros_to_body_basis_signs:="[1.0,1.0,1.0]" \
  -p min_fit_abs_tau:=0.01 \
  -p fit_max_abs_angular_accel_radps2:=50.0
```

第一次错误映射采集命令和上面基本一致，但没有传入 identity mapping，因此其结果只用于说明坐标问题。

## 分析方法

对每个 trial：

- 只取 `phase == excitation`。
- 稳态 yaw rate 使用 excitation 最后 `1.5 s` 的中位数。
- 横轴使用 `abs(actual_tau)`，Unity 没有 RPM 回读时 `tau_fit_my_yaw_nm` 等于命令重构的力矩。
- 拟合形式：

```text
abs(tau) = d_linear * abs(r) + d_quadratic * abs(r)^2
```

本次后处理使用 `uv`/ROS2 环境运行，避免系统 Python 的 NumPy/matplotlib ABI 问题。

后处理命令：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh python3 scripts/analyze_yaw_response_unity_identifier.py \
  --real-summary data/hydrodynamics/real/yaw_response_real_large_sweep_005_run_only/yaw_y_005_trial_summary.csv \
  --unity-csv data/hydrodynamics/unity/identifier/variousvehicles_yaw_current_play_identity_axes_20260802/yaw_y/yaw_y_001/yaw_y_001.csv \
  --output-dir data/hydrodynamics/analysis/yaw_response/real_yaw005_vs_unity_variousvehicles_identity_axes_20260802
```

## 当前结果

实机拟合：

```text
tau = 0.303 * |r| + 0.476 * |r|^2
R2 = 0.995
RMSE = 0.096 Nm
```

Unity DWP2 当前配置拟合：

```text
tau = 0.720 * |r| + 1.403 * |r|^2
R2 = 0.999
RMSE = 0.044 Nm
```

在重叠 yaw rate 区间：

```text
0.677 ~ 1.567 rad/s
```

Unity 当前等效 yaw 阻尼约为实机：

```text
2.74x
```

如果继续使用 DWP2 yaw hydrodynamic torque 轴向倍率补偿，按这组数据估计需要额外乘：

```text
0.36
```

这个值是从拟合曲线的重叠速度区间中位比值计算出来的，不是单个点拍脑袋。

## 解释

当前结果说明：

- DWP2 yaw 补丁已经参与 runtime 计算，probe 日志可看到 raw/scaled/target/applied yaw torque。
- 但当前 Unity yaw 阻尼仍偏大，导致相同 yaw torque 下 steady yaw rate 小于实机。
- 这次没有证明推进器语义/顺序/正负号错误；测试里正负 yaw 方向响应基本对称。
- 进一步调参应优先动 DWP2 yaw 动态阻尼相关参数或 yaw torque scale，而不是全局 `hydrodynamicForceCoefficient`。
