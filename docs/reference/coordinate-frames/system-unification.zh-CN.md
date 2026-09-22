# FinsROV 坐标系统一报告

标定相关坐标系入口见：

```text
docs/calibration/coordinate-system.zh-CN.md
docs/chase_body_frame_and_observation_zh-CN.md
```

## 结论

`motion_controller` 的入口统一为 Unity/controller 坐标约定：

- `controller_world.x`: 前向/池长方向
- `controller_world.y`: 向上，水面为 `0`，水下为负
- `controller_world.z`: 左向/水平横向
- `controller_body`: 同一轴约定下的自体坐标系，`x` 前、`y` 上、`z` 左

PID 内部使用的 `current_xyz_controller`、`target_xyz`、`error_world_xyz` 均为 `controller_world`。`error_body_xyz` 为 `controller_body`。

追逐类任务（`3Chase1` / `1Chase1`）的高层 observation 也统一使用这套 `controller_body` 约定：

- `body.x = forward`
- `body.y = up`
- `body.z = left`

对应维度规范：

- `3Chase1` chaser observation: `43D = 30D actor_obs + 13D controller_obs`
- `1Chase1` standard observation: `14D` local chase obs

详细字段和维度见：

```text
docs/chase_body_frame_and_observation_zh-CN.md
```

## Topic 边界

IMU 在进入这些边界前先由 hardware bridge 完成物理安装校正：芯片坐标
`x后、y右、z上` 绕 z 轴 180 度变为 `finsrov_base_link` 的 ROS FLU
`x前、y左、z上`。详细契约见 `docs/calibration/imu-coordinate-calibration.zh-CN.md`。

实机融合输出仍保留原始物理坐标：

- `/finsrov/pose`: `pool_world`
- `/finsrov/imu_link`: ROS/body 传感器约定
- `/finsrov/depth_link`: `pool_world` 下的 body 原点高度
- `/finsrov/dvl_link`: ROS/body 速度约定
- `/finsrov/state/status`: 融合状态

controller 只订阅 controller-native topic：

- `/finsrov/controller/pose`: `controller_world`
- `/finsrov/controller/imu`: `controller_body`
- `/finsrov/controller/depth`: `controller_world`，深度高度写在 `pose.position.y`
- `/finsrov/controller/dvl`: `controller_body`
- `/finsrov/controller/state/status`: controller-ready 状态

实机链路通过 `controller_state_adapter` 做唯一一次转换。Unity sim-truth 已经是 Unity/controller 坐标，必须直接发布 `/finsrov/controller/*`，不经过 adapter。

这里的“唯一一次转换”指 ROS FLU 到 controller 左手坐标的换基；它不包含更上游的
IMU sensor-to-base 物理安装旋转。两步不能合并，也不能在 fusion 或标定节点重复执行。

## 实机转换公式

`pool_world` 是 ROS 风格 z-up，水面高度由 `pool_world.yaml` 的 `water_surface_z_m` 指定。当前值为 `0.98`。

位置转换：

```text
controller.x = pool.x
controller.y = pool.z - water_surface_z_m
controller.z = pool.y
```

因此：

```text
pool.z = 0.98  -> controller.y = 0.00
pool.z = 0.68  -> controller.y = -0.30
```

位置、线速度和线加速度等普通向量使用同一个 basis：

```text
controller.x = ros.x
controller.y = ros.z
controller.z = ros.y
```

角速度是轴向量，不能在这个带反射的换轴矩阵上直接使用 `B`。令：

```text
B = [[1,0,0], [0,0,1], [0,1,0]]
A = det(B) * B = -B
```

则：

```text
omega_controller = A * omega_ros
[wx, wy, wz]_controller = [-wx, -wz, -wy]_ros
```

所以默认约定下 controller 的 yaw 角速度 `wy` 与 ROS 的 `wz` 符号相反。
角速度协方差也使用 `A * C * A.T`；混合 6D 协方差使用
`diag(B, A) * C * diag(B, A).T`。命令已经是 controller 坐标，不再重复做
轴向量变换。

四元数转换为：

```text
R_controller = B * R_ros * B.T
```

四元数不要直接交换分量。状态估计输出的 `/finsrov/imu_link` 和
`/finsrov/dvl_link` 仍是 ROS/body 物理坐标，只有
`controller_state_adapter` 将其转换成 `/finsrov/controller/imu` 和
`/finsrov/controller/dvl`。Unity sim-truth 已经是 controller 坐标，直接发布
`/finsrov/controller/*`，不能再转换一次。

## 命令接口

`send_position_goal` 不再接受 `--frame world` 或 `--frame body`。

有效命令：

```bash
ros2 run motion_control send_position_goal \
  --frame controller_world --x 0.0 --y -0.3 --z 0.0

ros2 run motion_control send_position_goal \
  --frame controller_body --x 0.1 --y 0.0 --z 0.0
```

默认 command topic：

- `/motion_controller/command/position_controller_world`
- `/motion_controller/command/position_controller_body`
- `/motion_controller/command/pose_controller_body`
- `/motion_controller/command/velocity_controller_body`

controller 接收端严格检查 `header.frame_id`。不是 `controller_world` 或 `controller_body` 的命令会被拒绝，不做隐式转换。

## Perception Topic

旧的平面 pose topic 已删除。当前 perception 侧有用输出是：

- `/finsrov/vision/refracted_pose_6d`
- `/finsrov/vision/refracted_pose_6d_pure`
- `/finsrov/vision/pose_3d_camera`
- `/finsrov/vision/refracted/status`

这些仍属于 perception/fusion 侧，不是 controller 直接输入。controller 只看 `/finsrov/controller/*`。
