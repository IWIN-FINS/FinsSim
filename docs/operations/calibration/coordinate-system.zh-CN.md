# 标定相关坐标系约定

本文是标定阶段的坐标系入口。完整审计和控制链路说明见：

```text
docs/coordinate_system_unification_zh-CN.md
docs/coordinate_convention_audit_zh-CN.md
ros2_ws/src/state_estimation/docs/fusion_pipeline.zh-CN.md
```

## 1. 关键坐标系

### pool_world

实船感知/融合侧的 ROS 物理世界系：

```text
x: 前向 / 池长方向
y: 左向 / 水平横向
z: 向上
z=0: 水池底部参考平面
water_surface_z_m: 水面高度
```

配置入口：

```text
ros2_ws/src/state_estimation/config/pool_world.yaml
```

### camera frame

相机坐标系由 OpenCV/PnP 使用。`T_camera_tag` 表示 tag 在相机坐标系下的 pose。

### tag frame

见 [apriltag-and-homography.zh-CN.md](apriltag-and-homography.zh-CN.md)：

```text
origin: tag 中心
+x: tag 左 -> 右
+y: tag 下 -> 上
+z: 右手系，垂直 tag 平面
```

### body frame

潜器 `finsrov_base_link`，ROS FLU：

```text
x: forward
y: left
z: up
```

### controller_world

控制器/RL 使用的 Unity/controller 坐标：

```text
x: 前向
y: 向上，水面附近为 0，水下为负
z: 左向
```

## 2. 位姿链路

视觉外参链：

```text
T_world_body = T_world_camera * T_camera_tag * inverse(T_body_tag)
```

深度修正：

```text
depth_m: 水压计从水面向下为正
pressure_sensor_z = water_surface_z_m - depth_m
body_z = pressure_sensor_z - pressure_sensor_offset_world_z
```

controller 入口换轴：

```text
controller.x = pool.x
controller.y = pool.z - water_surface_z_m
controller.z = pool.y
```

位置、线速度和线加速度等普通向量使用 basis `B`。注意默认 `B` 是一次轴
交换且 `det(B)=-1`，因此角速度属于轴向量，必须使用：

```text
A = det(B) * B = -B
omega_controller = A * omega_ros
[wx, wy, wz]_controller = [-wx, -wz, -wy]_ros
```

这意味着默认配置中 controller yaw 角速度 `wy` 与 ROS IMU 的 `wz` 符号
相反。角速度协方差同样使用 `A * C * A.T`，不能把角速度当普通线速度直接
交换轴。controller body 命令已经是 controller 坐标，不再转换。

姿态不能直接交换四元数分量，应使用矩阵换基：

```text
R_controller = B * R_ros * B.T
```

标定检查时，应同时观察原始 `/finsrov/imu_link.angular_velocity.z`、转换后的
`/finsrov/controller/imu.angular_velocity.y` 以及 controller pose 的 yaw
导数：前两个在默认 basis 下应符号相反，而后两个应符号一致。

## 3. 标定时最容易混淆的点

- `/finsrov/vision/pose_3d_camera` 的 `position.z` 是相机光轴方向距离，不是水深。
- `/finsrov/hardware/depth_raw` 是向下为正的 raw depth。
- `/finsrov/pose.position.z` 是 pool_world 下 body 原点高度，z-up。
- `send_position_goal --frame controller_world` 使用 controller/Unity 坐标，不是 `pool_world`。
- `motor_order`、`motor_signs` 只修推进器输出，不参与相机、位姿、速度、IMU 坐标变换。
