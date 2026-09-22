# FinsROV 硬件 Bridge 推进器映射说明

本文说明 `/finsrov/thrusters_out` 到下位机 MCU 8 路电机的顺序映射。当前约定是：

- 控制器、RL、仿真侧统一发布 Unity canonical 顺序。
- `hardware_bridge` 负责把 Unity canonical 顺序转换成 MCU 物理电机顺序。
- 实船 `force_n` 模式下，`hardware_bridge` 还负责把期望推力 `N` 映射成目标 RPM。
- MCU 只负责安全执行：enable 检查、超时停机、DSHOT telemetry RPM 闭环。

## Unity Canonical 输入顺序

`/finsrov/thrusters_out` 使用仿真文档 `docs/UUV_Thruster_Layout_Canonical.md` 中的 8 路顺序：

```text
index 0: V_LF = Vertical1   左前垂直
index 1: V_LB = Vertical2   左后垂直
index 2: V_RB = Vertical3   右后垂直
index 3: V_RF = Vertical4   右前垂直
index 4: H_LF = Horizontal1 左前水平
index 5: H_LB = Horizontal2 左后水平
index 6: H_RB = Horizontal3 右后水平
index 7: H_RF = Horizontal4 右前水平
```

因此控制器侧应始终按这个顺序发布：

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

## `/finsrov/thrusters_out` 的命令语义

同一个 topic 仍然保持 8 路 canonical 顺序，但具体数值由 `hardware_bridge.command_mode` 决定：

```text
force_n:
  输入单位是 N。
  bridge 根据实测推力曲线 force_N -> target_rpm -> normalized_rpm_command。
  这是实船闭环控制推荐模式。

normalized_rpm:
  输入是 [-1, 1]。
  bridge 只做顺序和符号映射，MCU 再把 +/-1.0 映射到固件中的 +/-3000 RPM。
  用于 motor_order/motor_signs、单电机方向、回包检查。

normalized_throttle:
  输入是旧的 [-1, 1] 直接油门语义。
  只作为旧固件/旧链路兼容入口，不建议作为实船默认控制。
```

当前 FineSUB 固件中：

```text
kDirectThrusterMaxRpm = 3000.0 RPM
```

所以 ROS2 配置中必须同步：

```yaml
firmware_max_rpm: 3000.0
```

`force_n` 模式的计算链路是：

```text
force_n = c1 * omega_rad_s * abs(omega_rad_s)
omega_rad_s = sign(force_n) * sqrt(abs(force_n) / abs(c1))
target_rpm = omega_rad_s * 60 / (2*pi)
normalized_rpm_command = target_rpm / firmware_max_rpm
```

注意：`M008` 的实测拟合 `c1` 是负数。bridge 反解时使用 `abs(c1)`，方向只由输入 force 符号和 `motor_signs` 决定。

## MCU 物理电机顺序

当前 `reference_code/FineSUB/Interface/Examples/TaskSUB.cpp` 中 MCU direct command 使用下面的物理顺序：

```text
MCU index 0: M1 motorLFLower 左前 lower，水平
MCU index 1: M2 motorLFUpper 左前 upper，垂直
MCU index 2: M3 motorLBUpper 左后 upper，垂直
MCU index 3: M4 motorLBLower 左后 lower，水平
MCU index 4: M5 motorRBLower 右后 lower，水平
MCU index 5: M6 motorRBUpper 右后 upper，垂直
MCU index 6: M7 motorRFUpper 右前 upper，垂直
MCU index 7: M8 motorRFLower 右前 lower，水平
```

也就是 MCU 期望收到：

```text
[H_LF, V_LF, V_LB, H_LB, H_RB, V_RB, V_RF, H_RF]
```

## 当前 Bridge 映射

`ros2_ws/src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml` 中当前默认配置暂定为：

```yaml
motor_order: [4, 5, 0, 1, 2, 7, 6, 3]
motor_signs: [-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0]
```

`motor_order` 的含义是：

```text
MCU output i = normalized_canonical_command[motor_order[i]] * motor_signs[i]
```

其中 `normalized_canonical_command` 在不同模式下来源不同：

```text
force_n:             force_N -> target_rpm / firmware_max_rpm
normalized_rpm:      /finsrov/thrusters_out 原值
normalized_throttle: /finsrov/thrusters_out 原值
```

所以当前映射表是：

```text
MCU M1 LF lower <- input 4 H_LF
MCU M2 LF upper <- input 5 H_LB
MCU M3 LB upper <- input 0 V_LF
MCU M4 LB lower <- input 1 V_LB
MCU M5 RB lower <- input 2 V_RB
MCU M6 RB upper <- input 7 H_RF
MCU M7 RF upper <- input 6 H_RB
MCU M8 RF lower <- input 3 V_RF
```

`motor_signs` 用于修正单个物理电机的正负方向。当前 `v4_pro1` 实测配置为 MCU output 0、1、2、4、6、7 反向，MCU output 3、5 保持正向。现场点动后如果发现某一路方向反了，把对应 MCU output index 的符号在 `1.0` 和 `-1.0` 之间切换。

## 电机转速回传

当前 FineSUB 固件的 telemetry 帧在 IMU/depth 数据后追加了 8 路 `motor_rpm`。ROS2 bridge 发布：

```text
/finsrov/hardware/motor_rpm_raw
```

消息类型是 `std_msgs/Float32MultiArray`。发布顺序和 `/finsrov/thrusters_out` 保持一致，是 Unity canonical 顺序：

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

MCU telemetry 原始数据仍是物理输出顺序 `[M1..M8]`，bridge 会按当前 `motor_order` / `motor_signs` 反向映射回 canonical 顺序。调试时可以一边点动某一路，一边看：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/motor_rpm_raw
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
```

其中 `/finsrov/hardware/status` 的 `last_motor_rpm` 是 canonical 顺序，`last_motor_rpm_mcu` 是 MCU 原始物理顺序。

如果使用旧固件，bridge 仍能解析 52B telemetry，但 `/finsrov/hardware/motor_rpm_raw` 会是 8 个 0。

## 推力曲线参数

`finsrov_hardware_bridge_v4_pro1.yaml` / `v4_pro2.yaml` 中的曲线配置使用 canonical 顺序：

```yaml
command_mode: force_n
firmware_max_rpm: 3000.0
thruster_curve:
  model: signed_quadratic
  c1_positive: [...]
  c1_negative: [...]
  rpm_min: [...]
  rpm_max: [...]
  force_deadband_n: [...]
  min_effective_rpm_positive: [...]
  min_effective_rpm_negative: [...]
```

V4 Pro1 的物理电机旋向按现场记录：

```text
CCW: M2, M3, M6, M8
CC:  M1, M4, M5, M7
```

`ros2_ws/data/thruster_curve/finsrov_v4_pro1` 目前有 4 组水平推进器 RPM sweep：

```text
M005_rpm_sweep_CC  -> H_LF, CC
M006_rpm_sweep_CCW -> H_RF, CCW
M007_rpm_sweep_CC  -> H_RB, CC
M008_rpm_sweep_CCW -> H_LB, CCW
```

由于 `positive/negative` 受 command 符号、接线顺序和传感器安装方向影响，V4 Pro1 配置不直接逐个照抄单次曲线，而是按同旋向取平均：

```text
CC:
  c1_positive = mean(H_LF positive, H_RB positive) = 9.387909395e-05
  c1_negative = mean(H_LF negative, H_RB negative) = 7.223931035e-05
  rpm_max = 2677.580078
  rpm_min = -2696.899414
  min_effective_rpm_positive = 476.0183565
  min_effective_rpm_negative = 342.577713

CCW:
  c1_positive = mean(H_RF positive, H_LB positive) = 1.118926888e-04
  c1_negative = mean(H_RF negative, H_LB negative) = 1.0372069015e-04
  rpm_max = 2628.0717775
  rpm_min = -2647.9121095
  min_effective_rpm_positive = 613.6658325
  min_effective_rpm_negative = 313.004013
```

根据当前 `motor_order`，物理电机旋向映射到 canonical 顺序为：

```text
V_LF <- M3 CCW
V_LB <- M4 CC
V_RB <- M5 CC
V_RF <- M8 CCW
H_LF <- M1 CC
H_LB <- M2 CCW
H_RB <- M7 CC
H_RF <- M6 CCW
```

所以 `v4_pro1.yaml` 中 8 路曲线数组按 `[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]` 填入：

```yaml
c1_positive: [CCW, CC, CC, CCW, CC, CCW, CC, CCW]
c1_negative: [CCW, CC, CC, CCW, CC, CCW, CC, CCW]
rpm_min: [CCW, CC, CC, CCW, CC, CCW, CC, CCW]
rpm_max: [CCW, CC, CC, CCW, CC, CCW, CC, CCW]
min_effective_rpm_positive: [CCW, CC, CC, CCW, CC, CCW, CC, CCW]
min_effective_rpm_negative: [CCW, CC, CC, CCW, CC, CCW, CC, CCW]
```

当前 `force_deadband_n` 先填 0，表示不在 force 层吞掉小指令；死区补偿主要由 `min_effective_rpm_*` 完成：非零 force 反解出来的 RPM 如果小于该方向最小有效 RPM，会被抬到最小有效 RPM。

### 和 `motion_controller` 的 force limits 的关系

当 `motion_controller` 使用 `thruster_output_mode: force_n` 时，`/finsrov/thrusters_out` 发布的是每个 canonical 推进器的期望推力，单位是 N，不是归一化油门。`hardware_bridge` 再用同一组 `thruster_curve` 把 force_N 反解成 target RPM，并最终发给 MCU。

因此 `motion_controller` 里的 `thruster_force_limits_n` 必须和 bridge 的曲线在同一套物理尺度上。V4 Pro1 当前写法是直接使用 command-limited 曲线的最大可用力：

```text
force_limit_positive[i] = c1_positive[i] * (rpm_max[i] * 2*pi / 60)^2
force_limit_negative[i] = c1_negative[i] * (abs(rpm_min[i]) * 2*pi / 60)^2
```

对应当前 V4 Pro1 配置：

```yaml
thruster_force_limits_n:
  positive: [8.4749, 7.3809, 7.3809, 8.4749, 7.3809, 8.4749, 7.3809, 8.4749]
  negative: [7.9750, 5.7618, 5.7618, 7.9750, 5.7618, 7.9750, 5.7618, 7.9750]
```

不要在这里再乘一次 `0.4`。`ros2_ws/data/thruster_curve/finsrov_v4_pro1` 里的数据已经是按上位机 command 合理范围采集/拟合出来的；下位机把 command `[-1, 1]` 映射到安全输出范围，`rpm_min/rpm_max` 已经代表这个安全范围内的实际 RPM 边界。再乘 `0.4` 会变成双重限幅，导致控制器认为最大推力只有约 1 N 量级。

## 运行时调整

启动 bridge：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py
```

如果要启动时直接选择配置文件、打开 debug echo、并 arm bridge，可以直接使用 launch arguments：

```bash
./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py \
  params_file:=./ros2_ws/src/hardware_bridge/config/finsrov_hardware_bridge_nature_test.yaml \
  debug_mask:=1 \
  debug_echo_decimation:=1 \
  enabled:=true
```

`debug_mask`、`debug_echo_decimation`、`enabled` 是启动时 parameter override。它们会覆盖 YAML 里的同名值；不传时就完全使用 YAML。

如果 `enabled:=true`，bridge 启动时的 startup zero frame 也会使用 `enabled=True`。如果 `enabled:=false`，startup zero frame 使用 `enabled=False`。

查看当前映射：

```bash
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge motor_order
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge motor_signs
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge command_mode
./scripts/run_ros2_uv.sh ros2 topic echo --once /finsrov/hardware/status
```

切换命令语义：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge command_mode normalized_rpm
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge command_mode force_n
```

切换 `command_mode` 时，bridge 会发送一帧 `enabled=false` 的 0 输出帧，避免运行中语义热切换保留旧输出。

运行时修改顺序：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge motor_order "[4, 5, 0, 1, 2, 7, 6, 3]"
```

运行时修改符号：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge motor_signs "[-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0, -1.0]"
```

修改 `motor_order` 或 `motor_signs` 时，bridge 会立即发送一帧 `enabled=false` 的 0 推力帧，避免热切换映射时保留旧输出。

如果运行时设置数组时报类似 `value must be a float array`，说明当前运行的 `/hardware_bridge` 还是旧代码。重新 build 并重启 bridge：

```bash
cd ros2_ws
./scripts/colcon_build_uv.sh --packages-select hardware_bridge
./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py
```

现场调试时仍建议先 disarm：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled false
```

确认映射无误后再 arm：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled true
```

## 单电机点动验证

打开 MCU echo：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge debug_mask 1
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge debug_echo_decimation 1
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/thruster_cmd_echo
```

先 disarm，确认不会动：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled false
./scripts/run_ros2_uv.sh ros2 topic pub --once /finsrov/thrusters_out std_msgs/msg/Float32MultiArray \
  "{data: [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

echo 里应该看到 `enabled: false`、`accepted: false`、`applied_thrust` 全 0。

确认安全后 arm，再点动某一路：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled true
./scripts/run_ros2_uv.sh ros2 topic pub --once /finsrov/thrusters_out std_msgs/msg/Float32MultiArray \
  "{data: [0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

如果 bridge 处于 `force_n` 模式，这条命令表示 `V_LF = +0.2 N`。如果 bridge 处于 `normalized_rpm` 模式，它表示 `V_LF = +0.2 * firmware_max_rpm`。

这条命令点动的是 Unity canonical 的 `V_LF`。按当前 V4 Pro1 映射，它会被 bridge 转到 MCU M3 `motorLBUpper`，并应用 MCU output 2 的 `-1.0` 符号。

如果点动 `V_LF` 时实际动的不是当前映射表对应的 MCU output / 物理电机，说明 `motor_order` 需要改。如果物理电机对了但方向反了，说明对应 MCU output index 的 `motor_signs` 需要在 `1.0` 和 `-1.0` 之间切换。

## 聚合动作验证

除了手写 `ros2 topic pub`，也可以使用 `hardware_bridge` 提供的测试发送器。它参照 Unity 文件：

```text
unity/RLControl/Assets/Scripts/ControlForPosition.cs
```

里的 `Heuristic()` 生成 `/finsrov/thrusters_out`，用于快速验证“前进/后退/转向/上浮/下潜”时电机顺序和符号是否整体正确。

`send_thruster_test` 是统一测试入口。默认 `--command-mode normalized_rpm`，适合测 `motor_order/motor_signs`。使用 normalized 测试前建议切到：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge command_mode normalized_rpm
```

如果要验证 `force_n -> RPM` 曲线映射，保持：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge command_mode force_n
```

单推进器 force_N 测试：

```bash
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test \
  index --index 4 --force 0.2 --rate 20 --duration 3
```

这里 `--index 4` 是 canonical `H_LF`，`--force 0.2` 表示期望推力 `+0.2 N`。

前进 force_N 测试：

```bash
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test \
  forward --command-mode force_n --force 0.6 --rate 20 --duration 3
```

这会发布 `[0, 0, 0, 0, +0.6, +0.6, -0.6, -0.6] N`。

启动 bridge 后，先打开 echo：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge debug_mask 1
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge debug_echo_decimation 1
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/thruster_cmd_echo --full-length
```

建议第一轮先 disarm，只检查 echo 中的 `received_thrust/applied_thrust/accepted/reject_flags`：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled false
```

打印某个模式但不发送：

```bash
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test forward --amplitude 0.2 --print-only
```

按 20Hz 发送 3 秒：

```bash
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test forward --amplitude 0.2 --rate 20 --duration 3
```

可用模式：

| mode           | Unity canonical 输出             |
| -------------- | -------------------------------- |
| `forward`    | `[0, 0, 0, 0, +a, +a, -a, -a]` |
| `backward`   | `[0, 0, 0, 0, -a, -a, +a, +a]` |
| `turn_left`  | `[0, 0, 0, 0, +t, +t, +t, +t]` |
| `turn_right` | `[0, 0, 0, 0, -t, -t, -t, -t]` |
| `up`         | `[+a, +a, +a, +a, 0, 0, 0, 0]` |
| `down`       | `[-a, -a, -a, -a, 0, 0, 0, 0]` |
| `stop`       | `[0, 0, 0, 0, 0, 0, 0, 0]`     |
| `index`      | 只点动一个 Unity canonical index |

其中 `a = --amplitude`，`t = clamp(--amplitude, -turn_limit, +turn_limit)`。默认 `--turn-limit 0.1`，和 Unity heuristic 里的转向限幅一致。

示例：

```bash
# 前进
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test forward --amplitude 0.2

# 后退
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test backward --amplitude 0.2

# 左转 / 右转
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test turn_left --amplitude 0.1
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test turn_right --amplitude 0.1

# 上浮 / 下潜
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test up --amplitude 0.15
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test down --amplitude 0.15

# 只点动 canonical index 7，即 H_RF
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test index --index 7 --amplitude 0.15
```

确认 echo 和物理响应都正确后，再 arm 小幅测试：

```bash
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled true
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test forward --amplitude 0.15 --duration 2
```

注意：`forward/backward/turn_left/turn_right` 的名字沿用 Unity heuristic 中正负输入的语义。实船实际表现如果相反，先不要改控制器算法，优先根据 echo 和物理电机响应判断是 `motor_order` 错、`motor_signs` 错，还是整体正方向定义需要调整。

## 持久化修改

运行时 `ros2 param set` 只影响当前运行的 bridge。确认现场映射后，把最终值写回：

```text
ros2_ws/src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml
```

然后重新 build 让 install 配置同步：

```bash
cd ros2_ws
./scripts/colcon_build_uv.sh --packages-select hardware_bridge
```

重启 bridge 后再次检查：

```bash
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge motor_order
./scripts/run_ros2_uv.sh ros2 param get /hardware_bridge motor_signs
```
