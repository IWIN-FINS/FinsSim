# FinsROV IMU 安装坐标校正

## 坐标链路

当前 H30 IMU 在潜器中的物理安装方向为右手系：

```text
sensor.x：朝后
sensor.y：朝右
sensor.z：朝上
```

ROS `finsrov_base_link` 使用标准 FLU 右手系：

```text
base.x：朝前
base.y：朝左
base.z：朝上
```

两者相差绕 `+z` 轴旋转 180 度。hardware bridge 使用：

```yaml
imu_sensor_to_base_quaternion_xyzw: [0.0, 0.0, 1.0, 0.0]
```

定义 `R_base_sensor` 为把 sensor-frame 向量变到 base frame 的旋转，则：

```text
a_base = R_base_sensor * a_sensor
omega_base = R_base_sensor * omega_sensor
R_world_base = R_world_sensor * transpose(R_base_sensor)
```

安装校正统一作用于 quaternion、angular velocity、linear acceleration 和协方差。
不能只给 `acceleration.x` 取反，否则会破坏 IMU 各字段之间的坐标一致性。

## Topic 语义

```text
MCU sensor frame: x后、y右、z上，右手系
  -> hardware_bridge 安装校正
/finsrov/hardware/imu_raw: finsrov_base_link，x前、y左、z上，右手系
/finsrov/hardware/telemetry: 同一校正后的 base_link IMU + 原子 RPM/时间戳
  -> state_fusion
/finsrov/imu_link: finsrov_base_link，右手系
  -> controller_state_adapter
/finsrov/controller/imu: controller_body，x前、y上、z左，Unity/FinsSim 左手语义
```

`imu_raw` 中的 `raw` 表示尚未经过 state fusion，并不表示未经物理安装外参校正。
当前不另外发布芯片原生 sensor-frame topic。

ROS 到 controller 的普通向量使用：

```text
[x, y, z]_controller = [x, z, y]_ros
```

该换轴矩阵行列式为 `-1`。角速度是轴向量，必须使用：

```text
omega_controller = det(B) * B * omega_ros
[wx, wy, wz]_controller = [-wx, -wz, -wy]_ros
```

四元数使用 `R_controller = B * R_ros * B^T`，不能直接交换四元数分量。

## 重力与 surge

静止时加速度计测量的是比力。ROS z 轴朝上时通常应看到：

```text
linear_acceleration ~= [0, 0, +9.81] m/s^2
```

这不表示重力方向朝上；物体的世界重力仍向下。水动力辨识会根据 quaternion
计算机体系重力投影并从 IMU 读数中扣除。潜器接近水平时，重力在 surge x 上的
投影接近零，因此正 surge 后的重力补偿加速度应为正。

`/finsrov/dvl_link` 当前不是独立 DVL 硬件数据，而是视觉/深度状态融合估计出的
body-frame 速度，不由 IMU 线加速度积分得到。因此 IMU x 符号错误时，DVL 速度仍
可能保持正确。

## 验收

启动 hardware bridge 后检查：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/imu_raw --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/telemetry --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/imu_link --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/controller/imu --once --full-length
```

验收要求：

1. status 中安装 quaternion 为 `[0,0,1,0]`，输出 frame 为 `finsrov_base_link`。
2. 静止水平时 `imu_raw.linear_acceleration.z` 接近 `+9.81 m/s^2`。
3. `imu_raw` 与同一 packet 的 `HardwareTelemetry` IMU 数值方向一致。
4. 短正 surge 时，重力补偿后的 base x 加速度与 DVL x 速度增量均为正；负 surge 反之。
5. controller yaw 角速度与 ROS z 角速度按轴向量规则符号相反。

历史 `surge_x_014` 是安装校正前数据，不应作为修复后 Fossen 参数的正式结果，也不会被
工具静默覆盖。应在修复后以新 run 编号重新采集。
