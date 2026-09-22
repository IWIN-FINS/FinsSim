# Hardware bridge telemetry

`hardware_bridge` 从一个 MCU telemetry packet 发布三条相关输出：

```text
/finsrov/hardware/telemetry     msgs/msg/HardwareTelemetry
/finsrov/hardware/imu_raw       sensor_msgs/msg/Imu
/finsrov/hardware/motor_rpm_raw std_msgs/msg/Float32MultiArray
```

`/finsrov/hardware/telemetry` 是标定专用的原子数据链路。它将 packet 内的 IMU、深度和
RPM 绑定到同一个 `mcu_time_ms`、`telemetry_sequence` 和映射后的 `header.stamp`。
`host_receive_time_ns` 是 bridge 收包时刻，用来计算链路延迟。

## IMU 安装校正

MCU 上报的 H30 sensor frame 是右手系 `x后、y右、z上`。bridge 在发布前使用：

```yaml
imu_sensor_to_base_quaternion_xyzw: [0.0, 0.0, 1.0, 0.0]
```

把 quaternion、angular velocity、linear acceleration 和协方差统一绕 z 轴旋转
180 度，输出为 `finsrov_base_link` 的 ROS FLU：`x前、y左、z上`。校正后的同一组
IMU 数值同时写入 `/finsrov/hardware/imu_raw` 和 `/finsrov/hardware/telemetry`。
该参数只能在启动时设置，运行中修改会被拒绝，避免姿态瞬间跳变。

`imu_raw` 中的 raw 表示未经过 state fusion，不表示未经安装外参校正。本项目不额外
发布芯片原生 sensor-frame topic。完整坐标说明见：

```text
./docs/calibration/imu-coordinate-calibration.zh-CN.md
```

bridge 内部只维护一个 `McuClockMapper`，会处理 MCU `uint32` 毫秒计数器溢出，并用近期
收包样本估计 MCU 到 ROS 的 offset 和慢速时钟漂移。状态 topic 的 JSON 中包含 `mcu_clock`。

本次 ROS2 改动不需要重新烧写下位机：继续使用现有 telemetry packet 的
`mcu_time_ms`。只有在后续确认该字段不是传感器采样/packet 生成时刻时，才需要另行修改
固件并安排烧写测试。

为保持现有融合链路行为，`/finsrov/hardware/imu_raw` 仍使用 host receive time；
`state_fusion_node`、`/finsrov/imu_link`、`/finsrov/dvl_link` 和 controller 契约均未改变。
`motor_rpm_raw` 暂时保留给旧外部工具，但新的水动力辨识和 RPM 一阶测量使用
`/finsrov/hardware/telemetry`，不再依赖无时间戳 RPM 与 IMU 拼接。

旧的 `/finsrov/hardware/motor_rpm_telemetry` 已移除。bridge 配置使用：

```yaml
hardware_telemetry_topic: /finsrov/hardware/telemetry
imu_sensor_to_base_quaternion_xyzw: [0.0, 0.0, 1.0, 0.0]
```
