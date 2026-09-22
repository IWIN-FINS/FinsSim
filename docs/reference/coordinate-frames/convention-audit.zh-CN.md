# FinsROV 坐标系链路审计

日期：2026-07-20

## 结论

当前系统里同时存在两套世界坐标约定：

1. `pool_world`：真实车感知/融合侧使用的 ROS 物理坐标，右手系，`x` 前，`y` 左，`z` 上，`z=0` 为池底，水面 `z=water_surface_z_m`。
2. `controller_world`：motion controller/PID/RL backend 使用的 Unity/controller 训练坐标，`x` 前，`y` 上，`z` 左，水面按 controller 约定应接近 `y=0`，水下为负 `y`。

当前 `motion_controller` 内部变量名 `position_world` 容易误导。它实际表示 controller world，不是 ROS `pool_world`。

如果目标是“进入 controller node 的所有 xyz、velocity、IMU 姿态、加速度都已经是 Unity/controller 坐标系”，建议把边界明确放在 controller node 的输入处或输入前的 adapter/fusion bridge 处。不要让 controller 一边接收 `pool_world`，一边接收 `controller_world`，再靠 topic 名和日志猜。

同样地，Unity 里的 chase 任务高层 observation 也统一按 `controller_body` 解释，不再把 prefab 自身的 local xyz 直接当成训练接口语义。追逐任务专项说明见：

```text
docs/chase_body_frame_and_observation_zh-CN.md
```

当前维度约定也一并统一为：

- `3Chase1` chaser: `30D actor_obs + 13D controller_obs = 43D`
- `1Chase1`: `14D` 高层局部追踪 observation

## 坐标定义

### pool_world

来源：

- `ros2_ws/src/state_estimation/config/pool_world.yaml`
- `ros2_ws/src/state_estimation/config/state_fusion.yaml`
- `ros2_ws/src/state_estimation/state_estimation/state_fusion_node.py`

定义：

- `x`：前向
- `y`：左向
- `z`：上向
- `z=0`：池底中心
- `water_surface_z_m=0.98`
- 硬件水深 `depth_raw` 为向下为正
- `depth_sign=-1.0`
- 水压计安装偏移 `pressure_sensor_offset_z_body=-0.115`

深度到 body 原点 z 的计算路线：

```text
pressure_sensor_z = water_surface_z_m + depth_sign * depth_raw.z
body_z = pressure_sensor_z - pressure_sensor_offset_z_body * cos(roll) * cos(pitch)
```

### controller_world

来源：

- `ros2_ws/src/motion_control/config/FinsROV/traditional_pid_position.yaml`
- `ros2_ws/src/motion_control/motion_control/math_utils.py`
- `ros2_ws/src/motion_control/motion_control/state_estimator.py`

定义：

- `x`：前向
- `y`：上向
- `z`：左向
- 控制器内部 target/current/error 使用该约定

当前 `motion_controller` 使用固定换轴：

```yaml
ros_to_controller_basis_indices: [0, 2, 1]
ros_to_controller_basis_signs: [1.0, 1.0, 1.0]
ros_to_controller_position_offset: [0.0, 0.0, 0.0]
```

即：

```text
controller.x = ros.x
controller.y = ros.z
controller.z = ros.y
```

四元数不是交换分量，而是矩阵换基：

```text
R_controller = B * R_ros * B^T
```

位置、线速度和线加速度等普通向量使用 `B`。角速度是轴向量；默认 `B` 的
行列式为 `-1`，必须使用带行列式修正的变换：

```text
A = det(B) * B = -B
omega_controller = A * omega_ros
[wx, wy, wz]_controller = [-wx, -wz, -wy]_ros
```

角速度协方差使用 `A * C * A.T`，混合线性/角速度的 6D 协方差使用
`diag(B, A)`。命令输入已经是 controller 坐标，不参与这一步。

## 真实车数据链路

### 硬件原始传感器

模块：

- `ros2_ws/src/hardware_bridge`

主要输出：

| Topic | 类型 | frame | 当前语义 |
| --- | --- | --- | --- |
| `/finsrov/hardware/imu_raw` | `sensor_msgs/Imu` | `finsrov_base_link` | MCU IMU 经 sensor-to-base 安装校正后的四元数、角速度、线加速度；raw 表示未融合 |
| `/finsrov/hardware/telemetry` | `msgs/HardwareTelemetry` | `finsrov_base_link` | 同一安装校正后的原子 IMU/RPM/深度与 MCU 映射时间 |
| `/finsrov/hardware/depth_raw` | `PoseWithCovarianceStamped` | `finsrov_depth_link` | `pose.position.z = depth_m`，向下为正 |
| `/finsrov/hardware/motor_rpm_raw` | `Float32MultiArray` | 无 header | canonical 8 路 RPM |
| `/finsrov/hardware/status` | `String` JSON | 无 header | 硬件桥状态、motor_order、motor_signs、last_mcu_command 等 |
| `/finsrov/hardware/thruster_cmd_echo` | `msgs/ThrusterCommandEcho` | `finsrov_base_link` | MCU 回显命令 |

注意：

- `motor_signs` 只作用在推进器命令/RPM 和 MCU 物理输出之间。
- `motor_signs` 不参与 pose、velocity、IMU、加速度坐标变换。
- IMU 芯片坐标为右手系 `x后、y右、z上`；bridge 使用 `[0,0,1,0]` 绕 z 轴 180 度校正到 ROS FLU。
- Pro1 当前配置文件是 `ros2_ws/src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml`。

### AprilTag / perception

模块：

- `ros2_ws/src/perception`
- 旧 Python perception 在 `ros2_ws/src/perception`，但当前主链路应使用 native refractive launch。

主要输出：

| Topic | 类型 | frame | 是否给控制主链路使用 | 当前语义 |
| --- | --- | --- | --- | --- |
| `/finsrov/vision/tag_detections_2d` | `AprilTagDetection2DArray` | camera frame | 是，给 refractive 节点 | 2D tag 角点 |
| `/finsrov/vision/pose_3d_camera` | `PoseWithCovarianceStamped` | `finsrov_overhead_camera` 或 IR camera | 否，调试/对照 | solvePnP 得到的 tag 在相机坐标系下的 3D 位姿，`position.z` 是相机光轴距离，不是水深 |
| `/finsrov/vision/tag_poses_3d_camera` | `AprilTagDetection3DArray` | camera frame | 通常否 | 多 tag camera-frame PnP 输出 |
| `/finsrov/vision/refracted_pose_6d` | `PoseWithCovarianceStamped` | `pool_world` | 是，state_fusion 主视觉输入 | 当前最佳可融合视觉 6D pose |
| `/finsrov/vision/refracted_pose_6d_pure` | `PoseWithCovarianceStamped` | `pool_world` | 否，debug/baseline | pinhole multi-tag PnP baseline，不依赖 depth/IMU/Snell |
| `/finsrov/vision/refracted/status` | `String` JSON | 无 header | 调试 | `mode`、`constrained_valid`、`pure_visual_valid`、深度和残差信息 |
| `/finsrov/vision/status` | `String` JSON | 无 header | 调试 | direct detector 状态，包含 `world_xy_yaw` 等调试字段 |

旧的平面 pose topic 已经不是当前主链路 topic。文档和 rqt 里如果还看到相关残留，应该删除。

`refracted_pose_6d` 的模式逻辑：

```text
depth_m < air_depth_threshold_m          -> mode=air
depth_m < underwater_depth_threshold_m   -> mode=surface_transition
otherwise                                -> mode=underwater_refraction
```

当前配置：

```yaml
air_depth_threshold_m: 0.02
underwater_depth_threshold_m: 0.08
publish_air_pose_on_constrained_topic: true
publish_surface_transition_pose_on_constrained_topic: true
```

含义：

- `mode=underwater_refraction` 且约束有效时，`/finsrov/vision/refracted_pose_6d` 发布 Snell + depth + IMU 混合约束结果。
- `mode=air` 或 `surface_transition` 且 pure PnP 有效时，也会在 constrained topic 上发布 pinhole fallback，`constrained_output_model=pinhole_air`。
- 所以 topic 名 `refracted_pose_6d` 有历史误导；实际语义是“当前最佳可融合视觉 6D pose”，输出 frame 仍是 `pool_world`。

### state_fusion

模块：

- `ros2_ws/src/state_estimation`

输入：

| Topic | 当前语义 |
| --- | --- |
| `/finsrov/vision/refracted_pose_6d` | `pool_world` 视觉位姿，默认只用 x/y 和 yaw，不用 vision z |
| `/finsrov/hardware/imu_raw` | body frame IMU |
| `/finsrov/hardware/depth_raw` | 向下为正的水深 |

输出：

| Topic | 类型 | frame | 当前语义 |
| --- | --- | --- | --- |
| `/finsrov/pose` | `PoseWithCovarianceStamped` | `pool_world` | 融合后的 body pose，`x/y/z` 是 ROS pool world |
| `/finsrov/imu_link` | `sensor_msgs/Imu` | `finsrov_base_link` | 融合/转发后的 IMU 姿态、角速度、加速度 |
| `/finsrov/depth_link` | `PoseWithCovarianceStamped` | `finsrov_depth_link` | `pose.position.z = body_z in pool_world`，不是 raw depth |
| `/finsrov/dvl_link` | `TwistWithCovarianceStamped` | `finsrov_base_link` | body-frame 速度 |
| `/finsrov/state/status` | `String` JSON | 无 header | fusion readiness、vision_mode、covariance、coordinate_convention |

重要事实：

- `/finsrov/pose` 当前发布的是 `pool_world`，不是 controller/Unity。
- `/finsrov/depth_link.pose.position.z` 当前是 pool_world body z，不是“向下为正水深”。
- `state_fusion` status 中已写明 `coordinate_convention=pool_world_z_up_bottom_origin_depth_corrected_to_body_origin`。

## motion_controller 链路

模块：

- `ros2_ws/src/motion_control`
- `python/finssim_rl/src/finssim_rl/models`

### controller 输入 topic

| Topic | 类型 | 当前处理 |
| --- | --- | --- |
| `/finsrov/controller/pose` | `PoseWithCovarianceStamped` | adapter 输出的 controller world 位姿 |
| `/finsrov/controller/imu` | `sensor_msgs/Imu` | adapter 输出的 controller body 姿态、线加速度和轴向量角速度 |
| `/finsrov/controller/depth` | `PoseWithCovarianceStamped` | adapter 输出的 controller y 高度 |
| `/finsrov/controller/dvl` | `TwistWithCovarianceStamped` | adapter 输出的 controller body 线速度/轴向量角速度 |
| `/finsrov/controller/state/status` | `String` JSON | adapter 输出的控制 gate 状态 |

原始 `/finsrov/pose`、`/finsrov/imu_link`、`/finsrov/depth_link` 和
`/finsrov/dvl_link` 仍由 state estimation 发布，保持 ROS/pool 物理坐标；
它们不应被 motion_controller 直接消费。

### controller 命令 topic

| 命令 | helper 行为 | 坐标语义 |
| --- | --- | --- |
| `ros2 run motion_control send_position_goal --frame controller_world ...` | 发布 `/motion_controller/command/position_controller_world`，`frame_id=controller_world` | 参数 `x/y/z` 是 controller/Unity world |
| `--frame controller_body` | 发布 body offset topic，`frame_id=controller_body` | 参数是 controller body offset |
| 手动发布 `frame_id=pool_world/map/odom/空` | controller 会按 `ros_to_controller_basis` 换轴 | 参数是 ROS/pool |
| 手动发布 `frame_id=controller_world/unity/model` | controller 不换轴 | 参数是 controller/Unity |

因此：

```text
send_position_goal --frame controller_world --x 2.0 --y -0.3 --z 0.0
```

表示：

```text
target_controller = [2.0, -2.0, 0.0]
```

不是：

```text
target_pool_world = [2.0, -2.0, 0.0]
```

### controller 内部状态

`VehicleState.position_world` 实际是 controller/world：

```text
state.position_world = reorder_vector(/finsrov/pose.position, B) + offset
```

当前默认：

```text
current_xyz_controller = [pose.x, pose.z, pose.y]
```

`VehicleState.position_ros` 是原始 `/finsrov/pose.position`，只用于日志追加：

```text
current_xyz_ros = [/finsrov/pose.position.x, y, z]
```

PID observation：

```text
[0:3]   target position in controller_world
[3:7]   target quaternion in controller_world
[7:10]  current position in controller_world
[10:14] current quaternion in controller_world
[14:17] linear velocity in controller_world
[17:20] angular velocity in controller_world
[20:23] optional raw current_xyz_ros for log only
```

### PID 内部

`traditional_position_pid.py` 假设输入已经是 Unity/controller 约定：

```text
target_xyz_unity = obs[0:3]
current_xyz_unity = obs[7:10]
pos_diff_unity = target - current
error_body = quat_inverse(current_quat) * pos_diff_unity
```

`PIDController` 的 6D 控制量：

```text
[Fx, Fy, Fz, Mx, My, Mz]
Fx: surge，向前
Fy: heave，向上
Fz: sway，向左
My: yaw
```

位置 PID 目前：

```text
u_surge = PID(error_body.x)
u_depth = PID(error_body.y)
u_sway  = PID(error_body.z)
control_6d = [u_surge, u_depth, u_sway, 0, u_yaw, 0]
```

当前 `traditional_pid_position.yaml` 里 `use_target_orientation=false`，而模型配置也可以关闭 yaw control。因此无 yaw 控制时只闭环平移，姿态会自由漂移。

### 推力分配

模块：

- `python/finssim_rl/src/finssim_rl/models/thrust_allocator.py`

推力分配矩阵输入：

```text
tau = [Fx, Fy, Fz, Mx, My, Mz]
```

输出：

```text
8 路 canonical thruster command, range [-1, 1]
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

`motion_controller` 再根据 `thruster_output_mode=force_n` 和 `thruster_force_limits_n` 把 [-1,1] action 变成每路 N，发布到 `/finsrov/thrusters_out`。

## 仿真 grpc_ros_adapter 链路

模块：

- `ros2_ws/src/grpc_ros_adapter`

当前 adapter 行为：

| gRPC stream | ROS output | 当前坐标处理 |
| --- | --- | --- |
| `StreamPoseSensor` | Unity sim-truth controller topic | position 按 Unity/controller 语义发送，不再交给 real adapter 换轴 |
| `StreamImuSensor` | Unity sim-truth controller topic | 姿态/线加速度/角速度按 Unity/controller 语义发送 |
| `StreamDvlSensor` | Unity sim-truth controller topic | body 线速度按 Unity/controller 语义发送 |
| `StreamDepthSensor` | Unity sim-truth controller depth topic | `pose.position.y` 使用 controller/world 高度语义 |

gRPC adapter 只负责 topic/address 和消息转发，不负责把 Unity/controller
坐标猜测或转换成 real 的 ROS/pool 坐标。坐标边界由 Unity bridge 的 topic
address 和 `controller_state_adapter` 的实机 raw 输入边界明确划分。

因此仿真链路必须明确使用 controller-native topic。Unity sim-truth 已经是
Unity/controller 坐标，不能先伪装成 `/finsrov/pose` 的 pool_world 再由
controller_state_adapter 转换，否则会发生第二次换轴。实机 raw topic 才经过
`controller_state_adapter`；该 adapter 对角速度使用 `det(B)*B`，而不是普通
向量的 `B`。

## 当前 rqt/topic 混乱点

### 可以作为控制主链路的 topic

| Topic | 用途 |
| --- | --- |
| `/finsrov/controller/pose` | motion_controller 主位置输入，`controller_world` |
| `/finsrov/controller/imu` | motion_controller 主姿态/角速度/加速度输入，`controller_body` |
| `/finsrov/controller/depth` | motion_controller 主垂向输入，`controller_world` |
| `/finsrov/controller/dvl` | motion_controller 主速度输入，`controller_body` |
| `/finsrov/controller/state/status` | motion_controller gate |
| `/motion_controller/command/position_controller_world` | controller_world 目标 |
| `/motion_controller/status/active_pose` | 当前 active target，controller_world |
| `/motion_controller/status/error_body` | controller_body error |
| `/motion_controller/status/reached` | reached 状态 |
| `/finsrov/thrusters_out` | 8 路 canonical 推进器命令 |

### 调试/中间 topic

| Topic | 不应直接理解成 controller current |
| --- | --- |
| `/finsrov/vision/pose_3d_camera` | 相机坐标下 PnP，`z` 是相机光轴距离 |
| `/finsrov/vision/tag_poses_3d_camera` | 相机坐标下多 tag PnP |
| `/finsrov/vision/refracted_pose_6d_pure` | pool_world debug/baseline |
| `/finsrov/vision/status` | direct detector JSON debug |
| `/finsrov/vision/refracted/status` | refractive pose JSON debug |
| `/finsrov/hardware/depth_raw` | 原始水深，向下为正 |
| `/finsrov/hardware/imu_raw` | 原始 IMU |

### 已删除或遗留

| Topic | 状态 |
| --- | --- |
| 旧平面 pose topic | 当前 native 链路已不再单独发布；如果 rqt 里还有，多半是旧节点、旧 monitor、旧文档或残留 publisher |

## 当前主要风险

1. `motion_controller` 内部变量名 `position_world` 和日志 `error_world_xyz` 没注明是 controller_world，会误导成 ROS pool_world。
2. `/finsrov/pose` 当前由真实车 `state_fusion` 发布为 `pool_world`；实机必须通过 `controller_state_adapter` 进入 `/finsrov/controller/*`，仿真则直接发布 controller-native topic。
3. 旧 `send_position_goal --frame world/body` 已删除，必须使用 `controller_world/controller_body`。
4. `refracted_pose_6d` 名字带 refracted，但在 air/surface_transition 下也可能发布 pinhole fallback。
5. `pose_3d_camera` 的 `z` 很容易被误认为深度，但它是相机光轴方向距离。
6. `ros2_ws/README.md` 里 canonical thruster 顺序有一处写成 `[V_LF, V_LB, V_RB, V_RF, H_LF, H_RF, H_RB, H_LB]`，而代码 `THRUSTER_NAMES` 和 Pro1 YAML 使用 `[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]`。

## 当前统一方案

进入 `motion_controller` 的所有状态统一成 `controller_world` /
`controller_body` 约定。

推荐边界：

```text
真实车硬件/perception/state_fusion 保持 ROS 物理 pool_world
                                      |
                                      v
                     controller_state_adapter
                                      |
                                      v
motion_controller 只消费 controller_world/controller_body
```

具体规则：

1. `state_estimation` 继续发布 `/finsrov/pose_raw_pool` 或 `/finsrov/pose` 为 `pool_world`，不要混入 Unity 语义。
2. `controller_state_adapter` 发布 `/finsrov/controller/*`，`frame_id` 明确为
   `controller_world` 或 `controller_body`。
3. `motion_controller` 只订阅 `/finsrov/controller/*`，不再对这些 topic 做
   第二次换轴。
4. 目标命令使用 `controller_world` 或 `controller_body`，例如
   `/motion_controller/command/position_controller_world`。
5. 日志字段统一命名：
   - `current_xyz_controller`
   - `current_xyz_pool`
   - `target_xyz_controller`
   - `error_controller_world`
   - `error_controller_body`
6. rqt 监控只展示主链路和明确 debug 分区，避免把 `pose_3d_camera`、`pure`、`refracted`、`pose` 混排成同一类位置。

`controller_state_adapter` 的变换规则为：

1. position: `[x,y,z]_pool -> [x,z,y]_controller`，再加水面 offset。
2. orientation: `R_controller = B * R_pool * B^T`。
3. position/linear velocity/linear acceleration: `B * v_pool`。
4. angular velocity/angular DVL: `det(B) * B * omega_pool`。
5. depth/body_z: pool z -> controller y。

Unity sim-truth 已经在 controller 坐标，应直接发布 controller topic，不能
再经过 adapter。原始 state estimation topic 只用于融合、诊断和 adapter 输入。
