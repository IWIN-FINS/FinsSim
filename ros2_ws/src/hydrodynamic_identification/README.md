# FinsROV 实机水动力辨识到 Unity Fossen 参数迁移

这个 ROS2 包用于在实机水池中测量 FinsROV 的简化 6 自由度 Fossen 水动力参数，并把结果迁移到 `unity/marus-example` 的 `HydrodynamicsProfile`。

当前测量分成两类：

- `surge_x`、`sway_z`、`yaw_y`：常规单轴台阶激励。
- `heave_y`：默认使用下潜 + 零推力自然上浮的 `dive_and_coast`，适合初始浮在水面、无法安全主动上浮的水池。
- `roll_x`、`pitch_z`：推进器主动小角度力矩脉冲，不需要手动把潜器摆到 10 度。

对于 profile 冻结后的实机--Unity 回放验证，使用
`unity_fossen_replay_validator`。它将 held-out CSV、Unity profile hash 和
预注册指标写入 manifest，并且要求 Unity 记录实际 applied wrench；完整操作见
[`Unity–Fossen 回放验证器`](../../../docs/simulation/validation/unity-fossen-replay-validator.zh-CN.md)。
实机记录时指定 `run_role:=held_out_fossen_validation`，该模式不会对 CSV
重新拟合水动力系数。

坐标和 Unity Fossen 映射：

| ROS fit key | 实机语义           | Unity Fossen 字段 | 说明                                     |
| ----------- | ------------------ | ----------------- | ---------------------------------------- |
| `surge_x` | 体坐标 x 前进      | `u`             | 同向，系数取正                           |
| `sway_z`  | 体坐标 z 左向横移  | `v`             | Fossen`v` 是右向，符号相反但系数取正   |
| `heave_y` | 体坐标 y 上浮/下潜 | `w`             | Fossen`w` 是下向，符号相反但系数取正   |
| `roll_x`  | 绕体坐标 x 滚转    | `p`             | 同轴，系数取正                           |
| `pitch_z` | 绕体坐标 z 俯仰    | `q`             | Fossen`q` 是绕右向，符号相反但系数取正 |
| `yaw_y`   | 绕体坐标 y 偏航    | `r`             | Fossen`r` 是绕下向，符号相反但系数取正 |

## 输入状态坐标契约

辨识节点现在和 `motion_control.controller_state_adapter` 使用同一套 frame 语义：

- 如果输入消息 `header.frame_id` 是 raw real 语义，例如 `pool_world`、`finsrov_base_link` 或空字符串，节点会执行一次 raw real -> controller 转换。
- 如果输入消息 `header.frame_id` 已经是 `controller_world` 或 `controller_body`，节点直接透传，不再做第二次 `[0,2,1]` 重排。
- DVL 线速度和 IMU 线加速度是普通向量，使用 basis `B=[x,z,y]`；IMU 角速度是轴向量，使用 `det(B)*B=-B`，不能直接当普通向量换轴。
- DVL/IMU 的线速度、角速度、线加速度最终统一进入 controller body：`x forward, y up, z left`，并与 `controller_state_adapter` 的符号完全一致。
- IMU orientation 会先按 frame_id 转成 controller quaternion，再提取 `roll_x / pitch_z / yaw_y`，不会再把 Euler 角当普通向量粗暴重排。

### Roll 力矩符号约定

当前推力到机体系力矩统一使用右手定则：

```text
Mx = r_y * F_z - r_z * F_y
左侧垂向推进器 -> -Mx
右侧垂向推进器 -> +Mx
```

节点输出的 fit JSON 会写入 `wrench_matrix_version: right_hand_roll_v2`。旧版
CSV 的 `tau_fit_mx_nm` 是按相反 roll 行计算的，不能覆盖原始文件；离线分析工具
会根据该 metadata 自动对旧数据取反。

hardware bridge 已在上游把 H30 原生的右手系 `x后、y右、z上` 绕 z 轴 180 度校正为
`finsrov_base_link` 的 ROS FLU `x前、y左、z上`。因此 `HardwareTelemetry` 到达本节点时
已经是 base frame；辨识节点只做 ROS FLU 到 controller body 的换基，禁止再次应用
sensor-to-base 安装旋转。

## 原子 HardwareTelemetry 与时间同步

默认辨识输入是标定专用 topic：

```text
/finsrov/hardware/telemetry  msgs/msg/HardwareTelemetry
/finsrov/dvl_link            geometry_msgs/msg/TwistWithCovarianceStamped
```

`HardwareTelemetry` 是下位机同一个 telemetry packet 的原子快照，包含同一帧经
hardware bridge 安装校正后的
IMU quaternion、角速度、线加速度、深度、压力、8 路 RPM、`mcu_time_ms` 和
`telemetry_sequence`。bridge 内只有一个共享的 `McuClockMapper`，先将 32 位 MCU
毫秒时钟展开并映射为 ROS 时间，再把同一映射时间写入消息 header。消息中的
`host_receive_time_ns` 只用于诊断通信延迟。

辨识节点以 50 Hz 的统一时间轴缓存并插值 telemetry 和 `dvl_link`。默认数据质量门限为：

```yaml
hardware_telemetry_topic: /finsrov/hardware/telemetry
use_hardware_telemetry: true
sync_max_telemetry_age_sec: 0.02
sync_max_dvl_age_sec: 0.03
```

CSV 会保留 `sample_time_ros_sec`、`imu_mcu_time_ms`、`imu_ros_stamp_sec`、
`dvl_ros_stamp_sec`、`host_receive_time_ns`、`imu_age_sec`、`dvl_age_sec`、
`telemetry_sequence` 和 `synchronization_valid`。超出门限的行仍保留用于诊断，
但不会进入 Fossen 拟合。

这条 telemetry 链路只供水动力辨识、推力曲线和诊断使用，不替代 fusion/controller
输入。`/finsrov/hardware/imu_raw` 继续使用原来的 host receive stamp，
`/finsrov/hardware/motor_rpm_raw` 继续作为无时间戳兼容输出，但新的标定代码不再拼接使用。
旧的 `/finsrov/hardware/motor_rpm_telemetry` 和 `MotorRpmTelemetry.msg` 已删除。

旧参数名仍然保留：

```yaml
ros_to_body_basis_indices: [0, 2, 1]
ros_to_body_basis_signs: [1.0, 1.0, 1.0]
```

它们现在只表示 raw real 输入到 controller body 的 basis，不表示所有输入都必须重排。以后如果订阅 `/finsrov/controller/imu`、`/finsrov/controller/dvl`，只要这些消息的 `frame_id` 是 `controller_body`，辨识节点会直接使用其中的数值。

旧数据处理注意：

- `surge_x_014` 是 hardware bridge 安装校正前采集的数据，不作为修复后 Fossen 系数的正式结果，也不会被离线工具静默覆盖；应使用新 run 编号重测。
- 旧版 CSV 如果只有 `nu_yaw_radps`、`nu_roll_x_radps`、`nu_pitch_z_radps` 等处理后字段，没有 `imu_frame_id`、`imu_raw_angular_*`、`imu_controller_angular_*`，就不能无损离线重算。
- 这类旧数据可以作为历史参考，但不要继续用来拟合“已修正 IMU 坐标契约”后的 DWP2 yaw 曲线。
- 新版 CSV 会同时写出输入 frame/输入 vector/controller vector。列名 `imu_raw_*` 表示辨识节点收到的 ROS/base 数据，不表示未经 hardware bridge 安装校正的芯片原生数据。

## 拟合模型

常规台阶轴使用：

```text
tau_i = m_eff_i * nu_dot_i
      + d_linear_i * nu_i
      + d_quadratic_i * abs(nu_i) * nu_i
      + bias_i
```

台阶轴的拟合窗口默认从 excitation 一开始就打开：

```yaml
step_fit_start_sec: 0.0
```

这样起步阶段的加速度会参与 `m_eff` 拟合，后续速度较高的阶段继续参与线性/二次阻尼拟合。小水池里如果发现推进器刚启动的几十毫秒有明显延迟或尖峰，可以把 `step_fit_start_sec` 设成 `0.05` 到 `0.15`，只跳过最早的执行器瞬态。

除 `heave_y` 的 `dive_and_coast` 外，只有 `sample_window=1` 的 excitation 数据会进入拟合；baseline、rest、post_zero 只写入 CSV 和全量曲线图，不参与 `.fit.json` 的系数拟合。`dive_and_coast` 会先等待零推力状态下真实上浮，再把检测后的上浮样本纳入拟合。为了避免明显不符合潜器动力学连续性的高加速度尖峰或短脉冲污染数据，节点会对拟合样本做加速度异常值剔除：

```yaml
fit_max_abs_linear_accel_mps2: 5.0
fit_max_abs_angular_accel_radps2: 3.0
fit_accel_outlier_mad_threshold: 8.0
fit_accel_spike_local_mad_threshold: 6.0
fit_accel_spike_local_window: 5
```

`.fit.json` 中每个轴会记录 `raw_sample_count`、`sample_count`、`rejected_outlier_count`、`rejected_abs_accel_count`、`rejected_mad_accel_count` 和 `rejected_local_spike_count`。如果 `rejected_outlier_count` 很多，说明这次实验里存在较多非物理加速度尖峰，最好重测。

默认还会启用物理范围约束：

```yaml
fit_constrain_physical_coefficients: true
fit_max_effective_mass_kg: 100.0
fit_max_effective_inertia_kgm2: 20.0
fit_max_linear_damping: 500.0
fit_max_quadratic_damping: 1000.0
fit_max_restoring_stiffness_nm_per_rad: 100.0
fit_max_abs_bias: 100.0
```

约束含义：

- `m_eff/J_eff >= 0`
- `d_linear >= 0`
- `d_quadratic >= 0`
- roll/pitch 的 `restoring_stiffness >= 0`
- `bias` 允许正负，但限制绝对值

这能防止负质量、负阻尼这种物理上不合理的结果直接进入 `.fit.json`。但如果结果里 `active_bounds` 显示参数贴在 `lower` 或 `upper` 边界，尤其 `m_eff` 贴在 `0`，这通常不是好结果，而是数据动态不足、坐标/推力方向有误、或传感器不同步；应该重测或放大可观测动态。

roll/pitch 主动小角度力矩脉冲使用：

```text
tau_i = J_eff_i * angle_ddot_i
      + d_linear_i * angle_dot_i
      + d_quadratic_i * abs(angle_dot_i) * angle_dot_i
      + restoring_stiffness_i * sin(angle_i)
      + bias_i
```

这里 `J_eff` 是刚体转动惯量 + 附加转动惯量。和线速度轴一样，迁移到 Unity 时不要把 `J_eff` 直接填进 `addedMassDiagonal.p/q`，而要先减去 Unity `Rigidbody` 已经承担的刚体惯量。

## roll/pitch 主动小扰动流程

roll/pitch 不再要求手动摆初始角。每个 roll/pitch trial 中：

1. baseline：零推力，记录当前姿态和漂移。
2. pulse：节点自动给 `Mx_roll` 或 `Mz_pitch` 一个很小的目标力矩，默认 `0.6s`。
3. ringdown：节点自动发零推力，继续记录姿态回正，默认 `3.0s`。
4. rest：零推力等待回稳。

默认幅值适合在小水池中先做单轴中等强度测试：

```yaml
hold_sec: 9.0
surge_x_levels: [6.0, 12.0, 18.0]  # N
heave_y_levels: [4.5, 9.0, 13.5]   # N
sway_z_levels: [4.5, 9.0, 13.5]    # N
yaw_y_levels: [0.60, 1.20, 1.80]   # Nm
roll_x_levels: [0.18, 0.36]        # Nm
pitch_z_levels: [0.18, 0.36]       # Nm
attitude_pulse_sec: 0.6
attitude_ringdown_sec: 3.0
attitude_min_abs_tau_nm: 0.005
roll_x_max_abs_angle_deg: 5.0
roll_x_max_abs_rate_radps: 0.4
pitch_z_max_abs_angle_deg: 5.0
pitch_z_max_abs_rate_radps: 0.4
ignore_attitude_safety_checks: false
```

roll 和 pitch 使用互相独立的安全阈值，且默认相对于控制器世界姿态零点判断：

```yaml
roll_x_equilibrium_angle_rad: 0.0
pitch_z_equilibrium_angle_rad: 0.0
```

如果需要人为指定固定的机械配平偏置，可修改这两个 `*_equilibrium_angle_rad`；节点不会在各 trial 的 baseline 自动改变它们。

如果 roll/pitch trial 中姿态偏差或角速度超过安全限制，节点会立即切断推进器脉冲，但继续在同一个 `excitation` 窗口记录零推力 ringdown；因此自由衰减数据仍会进入拟合。注意这不是碰壁检测；小水池测试仍应从单轴、单幅值开始。

## heave_y 下潜 + 自然上浮

控制器机体系 `+y` 向上，所以 `dive_and_coast` 会把每个 `heave_y` 幅值解释为绝对值并自动下发 `-abs(amplitude)`：

```text
baseline -> controller -Fy 下潜 -> coast: 0 推力自然上浮 -> rest
```

`include_negative` 在此模式下不会生成主动上浮 trial；每个幅值只有一次下潜/上浮回合。coast 先只做诊断记录：必须先观察到 `v_y <= -threshold`，再观察到 `v_y >= +threshold`，才会开始后续的上浮拟合窗口。这样不会把仍在下潜的 coast 数据混入“自然上浮”数据。

拟合将下潜主动段和检测后的自然上浮段共同用于：

```text
tau_thruster_y = m_eff_y * y_ddot
                + d1_y * y_dot
                + d2_y * |y_dot| * y_dot
                + bias_y
```

其中 `tau_thruster_y` 来自同步 RPM 和推力曲线；coast 阶段的目标推力为零，但仍保留 RPM 残余推力。`bias_y` 近似为 controller `+y` 向上时的**负净浮力**，因此正浮力潜器通常得到负值。它用于配平诊断，不能直接填入 Unity 阻尼或 added mass。

YAML 默认：

```yaml
heave_y_identification_mode: dive_and_coast
heave_y_coast_sec: 8.0                    # 等待上浮的最长秒数
heave_y_ascent_velocity_threshold_mps: 0.01
heave_y_ascent_sample_sec: 2.0            # 检测上浮后进入拟合的秒数
```

如果在 `heave_y_coast_sec` 内未检测到上浮，节点会记录 `heave_y_ascent_events.detected=false`，该 trial 的 coast 不进入拟合，而不是伪造上浮数据。

推荐单轴命令：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run \
  hydrodynamic_identification hydrodynamic_identifier \
  --ros-args \
  --params-file src/hydrodynamic_identification/config/hydrodynamic_identification.yaml \
  -p auto_start:=true \
  -p pause_between_trials:=true \
  -p axes:="[heave_y]" \
  -p amplitudes:="[4.0,6.0,8.0]" \
  -p include_negative:=false \
  -p hold_sec:=1.0 \
  -p heave_y_coast_sec:=8.0 \
  -p heave_y_ascent_velocity_threshold_mps:=0.01 \
  -p heave_y_ascent_sample_sec:=2.0 \
  -p rest_sec:=2.0 \
  -p output_directory:=./ros2_ws/data/hydrodynamics/real/hydrodynamic_identification
```

先将潜器放到水面下约 `1-1.5 m`，每个 coast 结束后人工恢复到相近深度再按 Enter。若要使用历史的双向台阶流程，显式设置 `-p heave_y_identification_mode:=step`。

仅在潜器有人值守、实验区域清空且你明确要诊断保护逻辑时，才可临时忽略这两个检查：

```bash
-p ignore_attitude_safety_checks:=true
```

该开关默认 `false`，不会改变平动轴和 yaw 的保护/停止行为；正常实测不要开启。

## 总体 Pipeline

```text
推进器曲线确认
  -> 状态链路检查
  -> 单轴单幅值烟测
  -> surge/heave/sway/yaw 全 excitation 台阶实验
  -> roll/pitch 小力矩脉冲实验
  -> 生成 CSV 和 fit.json
  -> 检查拟合质量
  -> 把 m_eff/J_eff 拆成 Rigidbody 刚体项 + addedMass
  -> 把阻尼和恢复刚度迁移到 Unity
  -> Unity 单轴响应验证
```

最关键的一点：

```text
added_mass = max(m_eff - rigidbody_mass_or_inertia, 0)
```

如果直接把 `m_eff` 或 `J_eff` 填到 Unity `addedMassDiagonal`，刚体质量/惯量会被算两遍。

## 启动前检查

确认：

- 关闭其它会发布 `/finsrov/thrusters_out` 的节点。
- `hardware_bridge.command_mode=force_n`。
- `/finsrov/hardware/motor_rpm_raw` 有 8 路 canonical RPM。
- `/finsrov/dvl_link` 或 `/finsrov/controller/dvl` 有线速度，且 `frame_id` 符合上面的坐标契约。
- `/finsrov/imu_link` 或 `/finsrov/controller/imu` 有角速度和有效 orientation quaternion，且 `frame_id` 符合上面的坐标契约。
- 水池空间足够；线缆不会明显限制横移、yaw、roll/pitch 回正。

构建：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select hydrodynamic_identification
```

启动状态融合和硬件桥后，启动辨识节点：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch hydrodynamic_identification hydrodynamic_identification.launch.py \
  auto_start:=false
```

开始实验：

```bash
./scripts/run_ros2_uv.sh ros2 service call /hydrodynamic_identifier/start std_srvs/srv/Trigger {}
```

## 如何停止

节点有三种停止方式：

- 自动停止：所有 trial 跑完后进入 `post_zero_sec`，持续发零推力，然后写出 `.csv` 和 `.fit.json`。
- 手动停止：调用 `/hydrodynamic_identifier/stop`，立刻发 8 路零推力，停止采样，并用已有数据拟合。
- 进程退出：`Ctrl+C` 或节点销毁时会尝试发 8 路零推力。

### 幅值之间手动复位

小水池测试可以启用 `pause_between_trials:=true`。节点完成每一个有符号 trial
（例如 `+6 N` 或 `-6 N`）后会持续发送零推力并暂停，不会自动开始下一个幅值。
此时可以人工把潜器拉回安全位置，在运行标定节点的终端按 Enter 后继续。
暂停期间的传感器数据不会进入 excitation 拟合窗口。

```bash
./scripts/run_ros2_uv.sh ros2 run \
  hydrodynamic_identification hydrodynamic_identifier \
  --ros-args \
  --params-file src/hydrodynamic_identification/config/hydrodynamic_identification.yaml \
  -p auto_start:=true \
  -p pause_between_trials:=true \
  -p axes:="[surge_x]" \
  -p amplitudes:="[3.0,6.0]" \
  -p include_negative:=true \
  -p hold_sec:=1.0 \
  -p rest_sec:=3.0
```

如果终端没有可用 stdin，也可以在人工复位后从另一个终端继续：

```bash
./scripts/run_ros2_uv.sh ros2 service call \
  /hydrodynamic_identifier/continue std_srvs/srv/Trigger {}
```

`pause_between_trials` 默认是 `false`，不影响原有自动连续实验。

建议水池边第二个终端提前准备：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 service call /hydrodynamic_identifier/stop std_srvs/srv/Trigger {}
```

紧急停桨：

```bash
./scripts/run_ros2_uv.sh ros2 topic pub --once /finsrov/thrusters_out std_msgs/msg/Float32MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

## 小水池推荐命令

第一次不要跑完整默认配置。先测单轴、单幅值。下面的示例使用 `auto_start:=true`，命令启动后会先执行 `pre_zero_sec`，然后自动进入 trial。

单轴实验可以直接在命令行写幅值，不需要修改代码或 YAML：

```bash
-p axes:="[surge_x]" -p amplitude:=8.0
```

`amplitude` 的单位取决于自由度：

- `surge_x`、`heave_y`、`sway_z`：N。
- `roll_x`、`pitch_z`、`yaw_y`：Nm。

如果只想测一个方向，使用：

```bash
-p include_negative:=false -p amplitude:=8.0    # 正方向
-p include_negative:=false -p amplitude:=-8.0   # 反方向
```

如果希望同一个轴一次扫多个幅值，使用 `amplitudes`：

```bash
-p axes:="[surge_x]" -p amplitudes:="[6.0, 9.0, 12.0]"
```

`amplitude` / `amplitudes` 只允许配合单轴 `axes` 使用。多轴实验仍使用各轴自己的 `surge_x_levels`、`yaw_y_levels` 等参数，避免把 N 和 Nm 混在一起。

如果使用 launch，也可以直接写：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch hydrodynamic_identification hydrodynamic_identification.launch.py \
  auto_start:=true \
  axes:=surge_x \
  amplitude:=8.0 \
  include_negative:=false \
  hold_sec:=6.0
```

如果把 `auto_start` 改成 `false`，节点只会进入 ready 状态，不会发推进器命令；需要在另一个终端调用：

```bash
./scripts/run_ros2_uv.sh ros2 service call /hydrodynamic_identifier/start std_srvs/srv/Trigger "{}"
```

只测中等幅值 roll 脉冲：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification hydrodynamic_identifier \
  --ros-args \
  --params-file src/hydrodynamic_identification/config/hydrodynamic_identification.yaml \
  -p auto_start:=true \
  -p axes:="[roll_x]" \
  -p amplitude:=0.18 \
  -p include_negative:=false \
  -p attitude_pulse_sec:=0.4 \
  -p attitude_ringdown_sec:=2.0 \
  -p roll_x_max_abs_angle_deg:=3.0 \
  -p max_wrench_scale:=1.0
```

只测中等幅值 pitch 脉冲：

```bash
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification hydrodynamic_identifier \
  --ros-args \
  --params-file src/hydrodynamic_identification/config/hydrodynamic_identification.yaml \
  -p auto_start:=true \
  -p axes:="[pitch_z]" \
  -p amplitude:=0.18 \
  -p include_negative:=false \
  -p attitude_pulse_sec:=0.4 \
  -p attitude_ringdown_sec:=2.0 \
  -p pitch_z_max_abs_angle_deg:=3.0 \
  -p max_wrench_scale:=1.0
```

只测中等幅值 surge：

```bash
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification hydrodynamic_identifier \
  --ros-args \
  --params-file src/hydrodynamic_identification/config/hydrodynamic_identification.yaml \
  -p auto_start:=true \
  -p axes:="[surge_x]" \
  -p amplitude:=6.0 \
  -p include_negative:=false \
  -p hold_sec:=6.0 \
  -p step_fit_start_sec:=0.0 \
  -p rest_sec:=2.0 \
  -p max_wrench_scale:=1.0
```

## yaw 响应扫描和 Unity 对比

如果推进器语义、顺序、正负号已经排除，下一步不要先改 RL，而是测同一个 yaw 力矩下实机和 Unity 水动力环境的角速度响应曲线。

目标关系：

```text
输入: yaw 力矩 tau_yaw, Nm
输出: yaw 角速度 r(t), rad/s

稳态近似:
tau_yaw ~= d_linear * r + d_quadratic * |r| * r
```

这个实验关注三类量：

- `steady_abs_yaw_rate_radps`：excitation 最后 1.5s 的 yaw rate 绝对值中位数，主要用于调阻尼。
- `peak_abs_yaw_rate_radps`：整个 excitation 内的最大 yaw rate，反映小水池里是否容易突然打转。
- `initial_abs_yaw_accel_radps2`：excitation 前 0.5s 的角加速度中位数，主要用于判断转动惯量/added inertia 是否过大或过小。

实机测 yaw sweep，建议先单方向，从小到大，不要一次跑完整 60s：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification hydrodynamic_identifier \
  --ros-args \
  --params-file src/hydrodynamic_identification/config/hydrodynamic_identification.yaml \
  -p auto_start:=true \
  -p axes:="[yaw_y]" \
  -p amplitudes:="[0.30, 0.45, 0.60, 0.90, 1.20]" \
  -p include_negative:=false \
  -p baseline_sec:=1.0 \
  -p hold_sec:=5.0 \
  -p rest_sec:=2.0 \
  -p step_fit_start_sec:=0.0 \
  -p max_wrench_scale:=1.0
```

如果水池空间允许，再测反方向：

```bash
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification hydrodynamic_identifier \
  --ros-args \
  --params-file src/hydrodynamic_identification/config/hydrodynamic_identification.yaml \
  -p auto_start:=true \
  -p axes:="[yaw_y]" \
  -p amplitudes:="[-0.30, -0.45, -0.60, -0.90, -1.20]" \
  -p include_negative:=false \
  -p baseline_sec:=1.0 \
  -p hold_sec:=5.0 \
  -p rest_sec:=2.0 \
  -p step_fit_start_sec:=0.0 \
  -p max_wrench_scale:=1.0
```

Unity 侧使用已有的 `FinsROVDwp2WaterObjectMotionBenchmark`，不是 ROS controller。把它挂在潜器主 Rigidbody/root 上，保留 DWP2 WaterObject，关闭 RL controller 或 ROS bridge 对推进器的控制。配置：

- `Axis Convention`: `FinsRovXForwardYUpZLeft`
- `Auto Run On Start`: 可按需打开
- `Place At Benchmark Start`: 打开，位置建议 `(0, -3, 0)`
- `Trials`: 只启用 `YawR`
- 添加多个 `YawR` trial：`0.30, 0.45, 0.60, 0.90, 1.20`
- `Baseline Seconds`: `1`
- `Excitation Seconds`: `5`
- `Rest Seconds`: `2`

Unity 输出默认在：

```text
./ros2_ws/data/unity_dwp2_waterobject_benchmark/yaw_r
```

实机和 Unity 都测完后，运行离线汇总：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select hydrodynamic_identification
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification analyze_yaw_response
```

输出：

```text
./ros2_ws/data/yaw_response_comparison/yaw_response_summary.csv
./ros2_ws/data/yaw_response_comparison/yaw_response_comparison.png
./ros2_ws/data/yaw_response_comparison/yaw_response_report.json
```

`yaw_response_report.json` 会拟合：

```text
tau_abs ~= d_linear * abs(r) + d_quadratic * abs(r)^2
```

并给出：

```text
recommended_dwp2_body_yaw_torque_scale
```

如果 Unity 在同样 yaw 力矩下角速度明显小于实机，说明 Unity yaw 阻尼或 yaw 惯量偏大。此时优先修改 `Dwp2BodyYawTorqueScaler.yawTorqueScale`：

```text
yawTorqueScale = recommended_dwp2_body_yaw_torque_scale
```

例如推荐值约 `0.5`，就把 `Dwp2BodyYawTorqueScaler` 的固定 `yawTorqueScale` 设为 `0.5`，先关闭随机化，重新跑 Unity benchmark。曲线贴近后，再给训练用 DR 设一个窄范围，例如：

```text
yawTorqueScaleRange = [0.4, 0.7]
```

如果稳态 yaw rate 对上了，但 `initial_abs_yaw_accel_radps2` 仍明显小于实机，问题更像 Unity yaw 转动惯量或 added inertia 偏大；这时不要继续只降阻尼，应该检查 `Rigidbody.inertiaTensor`、质量分布、碰撞体简化导致的惯量，以及 Fossen/补偿脚本里是否额外启用了 yaw added mass。

## 输出结果

默认输出目录：

```text
./ros2_ws/data/hydrodynamic_identification
```

每次实验生成：

```text
<axis>/<axis>_NNN/<axis>_NNN.csv
<axis>/<axis>_NNN/<axis>_NNN.fit.json
<axis>/<axis>_NNN/<axis>_NNN_velocity_acceleration.png
<axis>/<axis>_NNN/<axis>_NNN_fit_velocity_acceleration.png
```

例如单独测三次 surge：

```text
data/hydrodynamic_identification/surge_x/surge_x_001/surge_x_001.csv
data/hydrodynamic_identification/surge_x/surge_x_001/surge_x_001_velocity_acceleration.png
data/hydrodynamic_identification/surge_x/surge_x_001/surge_x_001_fit_velocity_acceleration.png
data/hydrodynamic_identification/surge_x/surge_x_002/surge_x_002.csv
data/hydrodynamic_identification/surge_x/surge_x_003/surge_x_003.csv
```

如果一次配置多个轴，输出会放到 `multi_axis/multi_axis_NNN/`。实机标定建议每次只测一个轴，这样每个轴的数据天然分包，编号也连续。

CSV 关键列：

- `axis` / `phase` / `level`：当前实验轴、阶段和目标幅值；roll/pitch 的 `level` 是目标力矩 Nm。
- `tau_target_*`：期望目标力或力矩。
- `tau_fit_*`：根据 fresh RPM 和推进器曲线反算的实际力或力矩。
- `nu_x_mps` / `nu_y_mps` / `nu_z_mps` / `nu_yaw_radps`：线速度和 yaw 角速度旧列。
- `nu_roll_x_radps` / `nu_pitch_z_radps`：roll/pitch 角速度。
- `nudot_roll_x_radps2` / `nudot_pitch_z_radps2`：roll/pitch 差分滤波角加速度。
- `angle_roll_x_rad` / `angle_pitch_z_rad` / `orientation_valid`：主动小角度拟合用姿态角和 IMU orientation 有效标记。
- `dvl_frame_id` / `imu_frame_id`：本行样本使用的输入 frame。
- `dvl_raw_linear_*` / `dvl_controller_linear_*`：DVL 原始线速度和进入拟合的 controller body 线速度。
- `imu_raw_angular_*` / `imu_controller_angular_*`：IMU 原始角速度和进入拟合的 controller body 角速度。
- `imu_raw_linear_accel_*` / `imu_controller_linear_accel_*`：IMU 原始线加速度和进入拟合的 controller body 线加速度。
- `imu_raw_quat_*` / `imu_controller_quat_*`：IMU 原始 quaternion 和按 frame 契约转换后的 controller quaternion。

加速度的来源：

- 线速度来自 DVL；线加速度直接使用 IMU 消息里的 `linear_acceleration`，映射到 FinsSim 体坐标后再按 `acceleration_filter_alpha` 做一阶低通滤波。
- 角加速度不是直接使用 IMU orientation 差分，而是在 IMU callback 收到新 gyro 角速度时，按 IMU 的实际更新时间做差分并低通滤波得到。
- 如果 CSV 中线加速度仍然长期接近 0，优先直接检查 `/finsrov/imu_link.linear_acceleration` 是否本身接近 0 或未被驱动填充；此时问题在 IMU 驱动/滤波输出，而不是辨识节点的差分逻辑。

### IMU 重力补偿

`/finsrov/imu_link` 的 `linear_acceleration` 保留 IMU 的原始语义：潜器静止时仍包含重力在机体系中的投影，不能把静止时的约 `9.81 m/s^2` 直接当成潜器线加速度。水动力辨识节点内部默认开启重力补偿，不修改 `/finsrov/imu_link`、`/finsrov/hardware/imu_raw`、state-fusion 或 Unity 的 IMU topic。

补偿过程为：

1. 将 IMU quaternion 按当前 frame contract 转成 `controller` 坐标系，坐标约定为 `x forward, y up, z left`。
2. 用姿态把 controller 世界系重力 `[0, g, 0]` 旋转到 controller body 系。
3. 计算 `a_kinematic = a_imu - a_gravity_body`，再用 `a_kinematic` 形成 `nu_dot` 并进入 Fossen 拟合。

配置参数：

```yaml
gravity_compensation_enabled: true
gravity_mps2: 9.80665
```

CSV 同时保留原始和派生值：

- `imu_controller_linear_accel_*`：原始 IMU 线加速度经过坐标变换后的值，静止时通常包含重力。
- `imu_controller_gravity_*`：根据当前姿态计算出的机体系重力投影。
- `imu_controller_kinematic_accel_*`：真正进入辨识滤波和拟合的线加速度。

静止检查时，`imu_controller_gravity_*` 的模长应接近 `9.81`，而 `imu_controller_kinematic_accel_*` 应接近零。若需复现旧的未补偿行为，可显式设置：

```bash
-p gravity_compensation_enabled:=false
```

#### 对历史数据离线重算

历史测量不会因为修改节点参数而自动改变。使用 `refit_hydrodynamic` 可以不重新驱动潜器，直接从旧 CSV 的 controller quaternion 和 IMU 加速度生成补偿副本并重新拟合。原始 CSV/JSON 不会被覆盖：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select hydrodynamic_identification
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification refit_hydrodynamic \
  ./ros2_ws/data/hydrodynamics/real/hydrodynamic_identification/surge_x/surge_x_012/surge_x_012.csv \
  ./ros2_ws/data/hydrodynamics/real/hydrodynamic_identification/surge_x/surge_x_013/surge_x_013.csv
```

每个输入 run 目录下会新增：

```text
<run>_gravity_compensated.csv
<run>_gravity_compensated.fit.json
<run>_gravity_compensated_velocity_acceleration.png
<run>_gravity_compensated_fit_velocity_acceleration.png
```

离线工具只重算输入 CSV 中实际存在的轴；JSON 中会记录 `source_csv`、补偿重力常数和 `refit_method`，便于和原始结果逐项比较。

PNG 曲线图会画出本次轴向速度和加速度随时间变化的曲线：

- `surge_x/heave_y/sway_z`：线速度 `m/s` 和线加速度 `m/s^2`。
- `yaw_y/roll_x/pitch_z`：角速度 `rad/s` 和角加速度 `rad/s^2`。
- `<run_id>_velocity_acceleration.png` 是全量诊断图，包含 baseline、excitation、rest、post_zero。
- `<run_id>_fit_velocity_acceleration.png` 只包含进入拟合窗口的 excitation 样本；如果 rest 阶段有突变脉冲，它不会出现在这张图里。
- 如果一次跑多个轴，会为每个轴分别生成 `<run_id>_<axis>_velocity_acceleration.png` 和 `<run_id>_<axis>_fit_velocity_acceleration.png`。

### 对已完成 run 重画各力段 PNG

`plot_hydrodynamic_force_segments` 是离线工具，只读取 CSV 并在同目录下重画
每个正负 excitation 力段的速度/加速度图，不创建 ROS publisher，也不会下发推进器指令。
横轴优先使用新 CSV 的 `sample_time_ros_sec`；旧 CSV 会自动退回到 `time_sec`。

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select hydrodynamic_identification
./scripts/run_ros2_uv.sh ros2 run \
  hydrodynamic_identification plot_hydrodynamic_force_segments \
  ./ros2_ws/data/hydrodynamics/real/hydrodynamic_identification/surge_x/surge_x_014/surge_x_014.csv
```

输出包括：

```text
<run>_all_force_segments_velocity_acceleration.png
<run>_force_m18N_velocity_acceleration.png
<run>_force_p18N_velocity_acceleration.png
...
```

`.fit.json` 核心结构：

```json
{
  "model": "decoupled_fossen_6dof_with_roll_pitch_attitude_pulse",
  "step_axis_equation": "tau_i = m_eff_i * nu_dot_i + d_linear_i * nu_i + d_quadratic_i * abs(nu_i) * nu_i + bias_i",
  "step_fit_start_sec": 0.0,
  "fit_max_abs_linear_accel_mps2": 5.0,
  "fit_max_abs_angular_accel_radps2": 3.0,
  "fit_accel_outlier_mad_threshold": 8.0,
  "fit_accel_spike_local_mad_threshold": 6.0,
  "fit_accel_spike_local_window": 5,
  "fit_constrain_physical_coefficients": true,
  "fit_max_effective_mass_kg": 100.0,
  "fit_max_effective_inertia_kgm2": 20.0,
  "fit_max_linear_damping": 500.0,
  "fit_max_quadratic_damping": 1000.0,
  "fit_max_restoring_stiffness_nm_per_rad": 100.0,
  "fit_max_abs_bias": 100.0,
  "plot_files": [
    "./ros2_ws/data/hydrodynamic_identification/surge_x/surge_x_001/surge_x_001_velocity_acceleration.png",
    "./ros2_ws/data/hydrodynamic_identification/surge_x/surge_x_001/surge_x_001_fit_velocity_acceleration.png"
  ],
  "attitude_pulse_equation": "tau_i = m_eff_i * angle_ddot_i + d_linear_i * angle_dot_i + d_quadratic_i * abs(angle_dot_i) * angle_dot_i + restoring_stiffness_i * sin(angle_i) + bias_i",
  "fits": {
    "roll_x": {
      "fit_type": "attitude_pulse",
      "raw_sample_count": 0.0,
      "sample_count": 0.0,
      "rejected_outlier_count": 0.0,
      "rejected_local_spike_count": 0.0,
      "bounded_fit": true,
      "active_bounds": "",
      "m_eff": 0.0,
      "d_linear": 0.0,
      "d_quadratic": 0.0,
      "restoring_stiffness": 0.0,
      "bias": 0.0,
      "rmse": 0.0,
      "rmse_unit": "Nm"
    }
  }
}
```

查看最新结果：

```bash
cd ./ros2_ws
latest_fit=$(find data/hydrodynamic_identification -name '*.fit.json' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
uv run python -m json.tool "$latest_fit" | less
```

只看某个轴的最新结果，例如 surge：

```bash
latest_surge=$(find data/hydrodynamic_identification/surge_x -name '*.fit.json' -printf '%T@ %p\n' | sort -nr | head -n 1 | cut -d' ' -f2-)
uv run python -m json.tool "$latest_surge" | less
```

## 拟合质量检查

迁移到 Unity 前先筛掉明显坏数据：

- 每个轴 `sample_count` 是否足够。
- `rmse` 是否相对目标力/力矩足够小。
- `tau_fit_*` 是否主要来自 fresh RPM，而不是长期回退到命令推力。
- 加速度或角加速度是否噪声过大。
- `bias` 是否过大，过大通常意味着漂移、水流、线缆力、浮力配平或传感器零偏没有处理好。
- roll/pitch 中 `orientation_valid` 是否一直为 1。
- roll/pitch 的 `max_abs_angle_rad` 是否小于安全限制，正反向响应是否大致对称。
- `bounded_fit=true` 时 `m_eff`、`d_linear`、`d_quadratic` 或 `restoring_stiffness` 不应为负；如果 `active_bounds` 经常显示贴边，先查坐标符号、推进器方向、姿态映射和数据质量。

## 和原仿真动力学对照

判断“实测系数是否比原来的仿真动力学更好”，不要只看系数本身，要做同一输入下的曲线对照。建议每个轴保留一组不参与拟合的验证 trial，分别用：

- `baseline`：marus-example 原始水动力参数。
- `identified`：替换为本包测出的参数。
- `real`：实机 CSV 中的 DVL/IMU 曲线。

对照表：

| 项目             | 计算方式                                     | 越好方向     | 判定建议                                     | 说明                                                       |
| ---------------- | -------------------------------------------- | ------------ | -------------------------------------------- | ---------------------------------------------------------- |
| 速度 RMSE        | `sqrt(mean((v_sim - v_real)^2))`           | 越小越好     | `identified < baseline`，最好降低 `20%+` | 主指标；surge/sway/heave 用线速度，roll/pitch/yaw 用角速度 |
| 加速度 RMSE      | `sqrt(mean((a_sim - a_real)^2))`           | 越小越好     | 不应明显劣于 baseline                        | 加速度噪声大，作为辅助指标；先低通或剔除尖峰/脉冲          |
| 稳态速度误差     | `abs(v_ss_sim - v_ss_real)`                | 越小越好     | `identified < baseline`                    | 主要检验 damping 是否合理                                  |
| 初始加速斜率误差 | `abs(a_early_sim - a_early_real)`          | 越小越好     | `identified < baseline`                    | 主要检验`m_eff / addedMass` 是否合理                     |
| 上升时间误差     | `abs(t_63_sim - t_63_real)`                | 越小越好     | `identified < baseline`                    | 看整体时间常数，而不是单点拟合                             |
| 停推衰减误差     | rest 阶段速度包络 RMSE                       | 越小越好     | `identified < baseline`                    | 检验阻尼在无推进时是否自然                                 |
| 方向对称性       | 正向/反向同幅值参数差异                      | 越小越好     | 差异大于`30%` 要复查                       | 差异过大常见于推力曲线、线缆、水流或边界效应               |
| 物理可解释性     | 是否贴边、是否大 bias、是否大 rejected count | 贴边越少越好 | 频繁贴边不能算通过                           | 约束拟合只是防止荒唐结果，不代表数据可靠                   |
| 控制任务表现     | 同一控制器下轨迹误差、超调、稳定时间         | 越小越好     | identified 不应破坏控制稳定性                | 最终要回到任务闭环验证                                     |

结论可以按下面规则写：

```text
identified better:
  速度 RMSE、稳态误差、上升时间误差多数降低，且没有明显更差的控制表现。

identified inconclusive:
  只在拟合 trial 上更好，但验证 trial 没提升；或参数大量贴边。

identified worse:
  验证 trial 的速度 RMSE/稳态误差比 baseline 更大，或闭环控制更抖。
```

## 迁移到 Unity

阻尼字段：

```text
linearDamping.u      = abs(fits.surge_x.d_linear)
linearDamping.v      = abs(fits.sway_z.d_linear)
linearDamping.w      = abs(fits.heave_y.d_linear)
linearDamping.p      = abs(fits.roll_x.d_linear)
linearDamping.q      = abs(fits.pitch_z.d_linear)
linearDamping.r      = abs(fits.yaw_y.d_linear)

quadraticDamping.u   = abs(fits.surge_x.d_quadratic)
quadraticDamping.v   = abs(fits.sway_z.d_quadratic)
quadraticDamping.w   = abs(fits.heave_y.d_quadratic)
quadraticDamping.p   = abs(fits.roll_x.d_quadratic)
quadraticDamping.q   = abs(fits.pitch_z.d_quadratic)
quadraticDamping.r   = abs(fits.yaw_y.d_quadratic)
```

附加质量/惯量字段：

```text
addedMassDiagonal.u = max(fits.surge_x.m_eff - rb.mass, 0)
addedMassDiagonal.v = max(fits.sway_z.m_eff  - rb.mass, 0)
addedMassDiagonal.w = max(fits.heave_y.m_eff - rb.mass, 0)
addedMassDiagonal.p = max(fits.roll_x.m_eff  - rb_roll_inertia, 0)
addedMassDiagonal.q = max(fits.pitch_z.m_eff - rb_pitch_inertia, 0)
addedMassDiagonal.r = max(fits.yaw_y.m_eff   - rb_yaw_inertia, 0)
```

`restoring_stiffness` 不直接填入 `HydrodynamicsProfile` 的 damping 或 added mass。它用来校准 Unity 里的 hydrostatic 回正能力，主要对应：

- `HydrodynamicsProfile.displacedVolume`
- `HydrodynamicsProfile.centerOfMass`
- `HydrodynamicsProfile.centerOfBuoyancy`
- `Rigidbody.mass`

做实机参数对齐时先用：

```text
HydrodynamicsController.mode = Fossen6Dof
HydrodynamicsProfile.useFullAddedMassMatrix = false
```

先关闭 domain randomization 和复杂水流。等单轴响应对齐后，再打开随机化和 water provider。

## 注意事项

- roll/pitch 主动脉冲不需要手动摆角。
- roll/pitch 的 `*_levels` 是目标力矩 Nm，不是角度。
- 小水池优先把当前测试轴的 `roll_x_max_abs_angle_deg` 或 `pitch_z_max_abs_angle_deg` 降到 `3.0`，把 `attitude_pulse_sec` 降到 `0.3-0.4`。
- 节点能按角度/角速度超限提前结束当前 roll/pitch trial，但不能检测碰壁。
- `yaw_y` 迁移到 Unity 时填 `r`，不是 `q`。
- `pitch_z` 迁移到 Unity 时填 `q`，不是 `r`；`roll_x` 迁移到 `p`。
