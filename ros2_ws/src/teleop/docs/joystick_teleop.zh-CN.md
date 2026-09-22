# FinsROV 手柄 ROS2 Teleop

`teleop` 把普通游戏手柄接入当前实船 ROS2 控制链路。旧 joystick 的 TCP 协议不再直连下位机；手柄只提供摇杆和按钮输入，ROS2 节点负责转换成机体系力/力矩，再输出 8 路 canonical 推进器推力。

## 链路

```text
gamepad
  -> finsrov_joystick_teleop
  -> /finsrov/teleop/body_wrench_cmd
  -> /finsrov/thrusters_out  # Float32MultiArray[8], force_N
  -> hardware_bridge command_mode=force_n
  -> force_N -> RPM -> MCU DSHOT RPM closed-loop
```

`/finsrov/thrusters_out` 顺序固定为：

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

## 启动

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch teleop joystick_teleop.launch.py \
  params_file:=src/teleop/config/finsrov_gamepad_v4_pro1.yaml
```

启动前确认没有其它节点抢推进器：

```bash
./scripts/run_ros2_uv.sh ros2 topic info /finsrov/thrusters_out -v
```

## 默认按键

```text
左摇杆竖直: 前进/后退 surge
左摇杆水平: 左右平移 sway
右摇杆水平: 左转/右转 yaw
RT: 上浮 / +heave
LT: 下潜 / -heave
SELECT: 切换 wrench/direct 输出模式
START: 切换 teleop armed/disarmed
B: 急停，立即 disarm 并发布全 0
RB: 按住 Precision 档
X: 按住 Boost 档
```

启动时先按 START 进行控制。RB Precision 的优先级高于 X Boost；同时按下时仍是 Precision。

`SELECT` 以按下沿切换 `wrench <-> direct`，当前配置对应 XInput `BACK` 的 button 6。切换瞬间节点会发布一帧全 0；`armed/disarmed` 状态不变。

冰原狼 2 的 M1/M2 在标准 XInput 中通常不是独立输入。建议在飞智软件中配置：

```text
M1 -> X     # 后背中指按住 Boost，不需要放开右摇杆
M2 -> B     # 后背冗余急停
```

不同手柄的 pygame 轴和按钮编号可能不同。观察：

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/teleop/status --once --full-length
```

根据 `axes` 和 `buttons` 修改 `config/finsrov_gamepad_v4_pro1.yaml`。

当前 YAML 假定 PC XInput 下 `LT=axis 2`、`RT=axis 5`，静止为 `-1.0`、满按为 `+1.0`。切换手柄模式、重连或驱动更新后，先观察 status 中的 `axes`，再改 `left_trigger_axis`、`right_trigger_axis`、`trigger_rest_value` 与 `trigger_pressed_value`。未确认前不要启用推进器。

## Wrench 档位

所有值都在 ROS 手柄 wrench 语义下：surge/sway/heave 为 N，yaw 为 N*m。

| 档位      | 激活方式                |   surge |   sway |  heave |     yaw |
| --------- | ----------------------- | ------: | -----: | -----: | ------: |
| Cruise    | 默认                    |  +/-6.0 | +/-4.0 | +/-5.0 |  +/-0.8 |
| Precision | 按住 RB                 |  +/-2.0 | +/-1.5 | +/-1.5 | +/-0.25 |
| Boost     | 按住 X 或映射为 X 的 M1 | +/-10.0 | +/-7.0 | +/-8.0 |  +/-1.2 |

这三档四轴同时满量程都已按当前真实推进器正反向力界和物理矩阵 B 验算为可实现，不会触及推进器饱和。status 中的 `controller_body_wrench` 可用于现场复核请求值、实际 `B @ force`、残差和饱和通道。

## 两种模式

`mode: wrench` 是默认模式：

```text
ROS 手柄 wrench [surge, sway, heave, yaw_z]
  -> controller-body [Fx, Fy, Fz, Mx, My, Mz]
     = [surge, heave, sway, 0, yaw, 0]
  -> shared physical_wrench_allocator
  -> 8 路 canonical force_N
```

分配器和 RL/`motion_controller` 使用同一个 FinsROV 物理前向矩阵 `tau = B @ force`，并遵守每路推进器的正反向力界。`/finsrov/teleop/body_wrench_cmd` 仍按 ROS 机体系发布：`x=surge`、`y=sway`、`z=heave`、`torque.z=yaw`；只有进入 allocator 前才显式转换到 controller-body 的 `y-up/z-left` 约定。

`mode: direct` 是调试模式：

```text
左摇杆竖直 -> forward/backward 图案
LT/RT -> down/up 图案
右摇杆水平 -> turn_left/turn_right 图案
```

direct 模式用于快速检查电机顺序和方向，正常手动驾驶优先使用 wrench 模式。

## 安全行为

- 默认 `start_enabled=false`，启动不会自动给非零推力。
- 未连接手柄或读数异常时发布全 0。
- 节点退出时发布 3 帧全 0。
- 实船测试时 `hardware_bridge` 仍应承担最终 timeout 和 force_N -> RPM 转换。

## 调参重点

常用安全参数：

```yaml
max_surge_force_n: 6.0
max_sway_force_n: 4.0
max_heave_force_n: 5.0
max_yaw_moment_nm: 0.8
precision_surge_force_n: 2.0
precision_sway_force_n: 1.5
precision_heave_force_n: 1.5
precision_yaw_moment_nm: 0.25
boost_surge_force_n: 10.0
boost_sway_force_n: 7.0
boost_heave_force_n: 8.0
boost_yaw_moment_nm: 1.2
left_trigger_axis: 2
right_trigger_axis: 5
trigger_rest_value: -1.0
trigger_pressed_value: 1.0
mode_switch_button: 6
force_limits_positive_n: [...]
force_limits_negative_n: [...]
physical_wrench_limits_policy: [...]
```

`physical_wrench_limits_policy` 的顺序为 `[surge, sway, heave, roll, pitch, yaw]`，单位为 `[N, N, N, N*m, N*m, N*m]`。它应与 `ppo_wrench_for_pose_physical_wrench_allocator.yaml` 的 `wrench6d.wrench_limits` 保持一致；手柄仅使用其中 surge/sway/heave/yaw 四轴。

如果整体方向不符合预期，优先判断问题属于：

```text
手柄轴方向错       -> 改 *_axis_sign
teleop 的机体系映射错 -> 检查 ROS [x,y,z] 到 controller [x,y,z]=[forward,up,left] 的重排
单个物理电机方向错 -> 改 hardware_bridge motor_signs
单个物理电机顺序错 -> 改 hardware_bridge motor_order
```
