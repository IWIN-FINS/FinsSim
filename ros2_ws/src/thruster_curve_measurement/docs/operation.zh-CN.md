# FinsROV 推进器推力曲线测定流程

本文说明如何测定推进器推力曲线。当前保留两条方法：

```text
独立 ThrusterBench 台架链路：
  上位机直接通过串口控制单推进器下位机固件，不依赖潜器、不启动 hardware_bridge。
  用于单个推进器拆出台架后的 command -> Force 曲线测定。

整机 / hardware_bridge 链路：
  上位机发布 ROS topic，hardware_bridge 再给潜器 MCU 发 8 路推进器指令。
  原有阶跃测试用于 command -> Force 曲线。
  新增 rpm-sweep 测试用于 RPM -> Force 曲线，并拟合静水模型 T = c1 * omega * abs(omega)。
```

推力曲线测量代码已经从 `hardware_bridge` 中拆出，单独放在 ROS2 package：

```text
ros2_ws/src/thruster_curve_measurement
```

整机 / hardware_bridge 链路中，`hardware_bridge` 仍只负责和 MCU 通信，测量 package 通过正常推进器输入 topic `/finsrov/thrusters_out` 给它输入推进器指令。独立台架链路则直接通过串口给 `FineSUB/Services/ThrusterBench` 固件发送单推进器命令。本文档也放在该 package 内：

```text
ros2_ws/src/thruster_curve_measurement/docs/operation.zh-CN.md
```

## 1. 基本原则

### 1.1 包结构与节点入口

`ros2_ws/src/thruster_curve_measurement` 是一个 ROS2 package，不是单个节点目录。实际可运行节点由 `setup.py` 中的 `console_scripts` 定义：

```text
thruster_bench_measure  独立台架测量：直接通过串口控制单推进器固件、读取 Modbus 力传感器、记录 CSV、稳态表格、离散图和拟合图
thruster_curve_measure  整机 / hardware_bridge 链路测量：发布 8 路 canonical 推进器 ROS 指令、读取 Modbus 力传感器和 RPM 回传、记录 CSV 并生成拟合结果
thruster_curve_sweep    hardware_bridge 链路辅助节点：只发布推进器 sweep 指令，不读取力传感器
```

目前 package 的关键文件为：

```text
thruster_curve_measurement/
  package.xml
  setup.py
  setup.cfg
  config/
    thruster_bench_measure_step.yaml
    thruster_curve_measure_step.yaml
  docs/
    operation.zh-CN.md
  launch/
    thruster_curve_measure.launch.py
  thruster_curve_measurement/
    force_sensor_modbus.py
    thruster_bench_driver.py
    thruster_bench_measure.py
    thruster_curve_measure.py
    thruster_curve_sweep.py
  tests/
    test_force_sensor_modbus.py
    test_thruster_bench_driver.py
    test_thruster_bench_measure.py
    test_thruster_curve_measure.py
    test_thruster_curve_sweep.py
```

这个 package 现在已经提供：

```text
config/   固化常用测量参数，例如阶跃范围、rpm-sweep 参数、串口和 Modbus 参数
launch/   hardware_bridge 链路的一键启动测量节点
```

当前 launch 只启动 hardware_bridge 链路的 `thruster_curve_measure` 测量节点，不会自动启动或使能 `hardware_bridge`。独立台架链路推荐直接使用 `thruster_bench_measure` CLI。

推力曲线测定也使用 hardware_bridge 的正常输入 topic：

```text
/finsrov/thrusters_out
```

测量时不要再把 `hardware_bridge` 临时切到其他输入 topic。为了避免抢控制入口，应停止其他会发布 `/finsrov/thrusters_out` 的控制器或遥控节点，只保留测量节点发布指令。

RPM 一阶测量节点 `thruster_rpm_first_order_identification` 默认订阅：

```text
/finsrov/hardware/telemetry  msgs/msg/HardwareTelemetry
```

它使用同一 telemetry packet 中的 `mcu_time_ms`、`telemetry_sequence` 和 `rpm[8]`，
因此 RPM 不再与无时间戳的 `/finsrov/hardware/motor_rpm_raw` 拼接。旧的
`motor_rpm_telemetry` 消息已删除；正式测量应使用原子 telemetry。


推进器 canonical 顺序为：

```text
index 0: V_LF 左前垂直
index 1: V_LB 左后垂直
index 2: V_RB 右后垂直
index 3: V_RF 右前垂直
index 4: H_LF 左前水平
index 5: H_LB 左后水平
index 6: H_RB 右后水平
index 7: H_RF 右前水平
```

### 1.2 对采样策略的批判性说明

当前测量节点发布的是归一化推进器指令：

```text
command in [-1, 1]
```

整机 / hardware_bridge 链路现在支持两种主要采样策略。

阶跃 `step` 模式沿用原有方法：把每个 command 当作一个离散点，等待局部稳态后统计末尾窗口，用于得到 `command -> Force` 曲线。采样点仍然采用“中间密、两头疏”的思想：

```text
高 command 区间：点更疏，例如 -1.0, -0.9, ..., -0.5 和 0.5, ..., 1.0
中等 command 区间：点适中，例如 -0.4, -0.35, ..., -0.15 和 0.15, ..., 0.45
零附近低速/死区区间：点最密，例如 -0.1, -0.08, -0.06, ..., 0.1
```

新增的 `rpm-sweep` 模式用于测定 `RPM -> Force` 曲线：command 从 `-amplitude` 连续慢速扫到 `+amplitude`，再从 `+amplitude` 扫回 `-amplitude`。由于下位机已经回传转速，最终拟合以 `target_rpm / target_omega_rad_s` 为横轴，而不是以 command 为横轴。

`rpm-sweep` 的核心思想是：

```text
1. command 连续慢速变化，不为每个 command 点单独等待稳态。
2. 每一行有效 `target_rpm + force_zeroed_n` 都可以成为一个 RPM-Force 拟合样本。
3. 拟合只使用连续扫描阶段：
   ramp_negative_to_positive
   ramp_positive_to_negative
4. hold、settle、pre_zero、post_zero 主要用于建立初始状态、扣零和检查残余水流，不混入 c1 拟合。
```

需要注意：

```text
1. 真正输出到推进器的符号、顺序、限幅仍由 hardware_bridge 和 MCU 决定。
2. step 模式更适合准稳态 `command -> Force`；rpm-sweep 更适合密集采样的 `RPM -> Force`。
3. `rpm-sweep` 的 `--ramp` 不能太短；扫得太快时，力传感器响应、水流建立和台架弹性会让同一 RPM 下的瞬时力产生偏差。
4. 空化/饱和最好结合电流、电压或 ESC 遥测判断；当前程序只采集力传感器数据和 RPM，不直接判断电流飙升。
```

## 2. 测量前检查

确认：

```text
1. 推进器固定可靠。
2. 力传感器已经机械调零，且受力方向正确。
3. 水池中没有人手靠近推进器。
4. 电源、电调、MCU、上位机通信正常。
5. 力传感器 Modbus 串口没有被厂家软件占用。
```

如果厂家软件正在占用同一个串口，ROS 侧无法同时读取该传感器。此时需要关闭厂家软件，或者让厂家软件导出数据后走离线对齐流程。

正式使用 `--mode rpm-sweep` 比较多个推进器的 `RPM -> Force` 曲线时，额外确认：

```text
1. 每个推进器固定到台架时，轴向推力方向相对力传感器完全一致。
2. 力传感器受力方向不能在不同推进器之间翻转；如果必须翻转，后处理时 force 符号也必须同步翻转。
3. 正桨 / 反桨要单独记录。它们外形尺寸相同，但正转和反转对应的推力方向、效率和符号可能不同。
4. 多个推进器必须使用完全相同的 amplitude、ramp、rate、供电电压、浸没深度和固定方式。
5. 每测完一路，等待水流衰减后再测下一路。连续扫太快时，水流惯性和传感器动态会污染 `RPM -> Force`。
```

`rpm-sweep` 的 `c1` 是带符号参数。若某个推进器出现 `rpm > 0` 时 `force < 0`，而其他推进器是 `rpm > 0` 时 `force > 0`，这通常首先说明坐标或受力方向没有统一，不应直接解释为推进器本体差异。

## 3. 构建 ROS 包

在仓库根目录执行：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select hardware_bridge thruster_curve_measurement
```

如果只想在源码态直接运行，也可以使用 `uv run python -m ...`，示例见下面独立台架流程。

## 4. 独立 ThrusterBench 台架流程（推荐用于单推进器台架）

这一流程用于“一个推进器 + 一个 ESC + 一个 MCU + 一个力传感器”的独立台架，不依赖潜器，也不需要启动 `hardware_bridge`。它保留原有的独立台架 command-force 测量方法。

### 4.1 下位机固件准备

下位机代码位于独立的 FineSUB 仓库：

```text
./reference_code/FineSUB
```

FinsSim 中的路径：

```text
./reference_code/FineSUB
```

只是符号链接，实际改动和提交都发生在 `./reference_code/FineSUB`。

台架固件模块为：

```text
FineSUB/Services/ThrusterBench/
  ThrusterBenchProtocol.hpp
  SingleThrusterBench.hpp

FineSUB/Interface/Examples/
  TaskThrusterBench.cpp
```

在 `FineSUB/Interface/ProjectConfig.h` 中启用：

```cpp
#define WITH_THRUSTER_BENCH_EXAMPLE
#define THRUSTER_BENCH_UART_ID 3
#define THRUSTER_BENCH_DSHOT_ID 4
#define THRUSTER_BENCH_OUTPUT_LIMIT 0.4f
#define THRUSTER_BENCH_TIMEOUT_MS 500u
#define THRUSTER_BENCH_STATUS_PERIOD_MS 200u
```

含义：

```text
WITH_THRUSTER_BENCH_EXAMPLE   启用独立台架任务；启用后 TaskSUB 不再导出，避免和潜器任务抢同一路 ESC
THRUSTER_BENCH_UART_ID        上位机连接下位机的 UART 编号，默认 3
THRUSTER_BENCH_DSHOT_ID       台架推进器使用的 DSHOT 输出编号，默认 4
THRUSTER_BENCH_OUTPUT_LIMIT   MCU 侧硬限幅；即使上位机要求更高，也会被 MCU 截断
THRUSTER_BENCH_TIMEOUT_MS     超过该时间未收到新命令，下位机自动 disable 并输出 0
```

然后用 CLion 构建并烧录对应 BSP，例如 `Robomaster_A`。烧录完成后，先不要安装推进器或先保持电源断开，确认串口能打开、命令帧和 status 帧正常。

### 4.2 上位机命令入口

独立台架使用：

```text
thruster_bench_measure
```

它不会发布 ROS 推进器 topic，而是直接打开两个串口：

```text
--thruster-port   下位机 MCU 串口，用于发送单推进器指令并读取 status
--sensor-port     力传感器 Modbus 串口
```

默认配置文件：

```text
src/thruster_curve_measurement/config/thruster_bench_measure_step.yaml
```

默认配置比较保守：

```text
output_limit: 0.1
step_commands: [-0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3]
```

注意这里有两层限制：

```text
上位机 output_limit：每次命令发给下位机的请求限幅
下位机 THRUSTER_BENCH_OUTPUT_LIMIT：固件硬限幅
```

实际输出近似为：

```text
applied = clamp(command, -1, 1) * min(output_limit, THRUSTER_BENCH_OUTPUT_LIMIT)
```

例如固件硬限幅是 `0.4`，上位机 `--output-limit 0.1`，则最大实际输出只到 `0.1`。

### 4.3 查询串口

Linux 下查看串口：

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

插拔 USB 转串口时观察内核日志：

```bash
dmesg -w
```

建议把下位机 MCU 和力传感器分清楚，例如：

```text
/dev/ttyUSB0  下位机 MCU
/dev/ttyUSB1  力传感器
```

实际编号以你的机器为准。

### 4.4 打印测量计划

源码态直接运行：

```bash
cd ./ros2_ws

PYTHONPATH=src/thruster_curve_measurement:src/hardware_bridge \
uv run python -m thruster_curve_measurement.thruster_bench_measure \
  --config src/thruster_curve_measurement/config/thruster_bench_measure_step.yaml \
  --thruster-port /dev/ttyUSB0 \
  --sensor-port /dev/ttyUSB1 \
  --csv data/thruster_bench/bench_plan.csv \
  --print-plan
```

如果已经 `colcon build` 并 `source install/setup.bash`，也可以运行：

```bash
ros2 run thruster_curve_measurement thruster_bench_measure \
  --config src/thruster_curve_measurement/config/thruster_bench_measure_step.yaml \
  --thruster-port /dev/ttyUSB0 \
  --sensor-port /dev/ttyUSB1 \
  --csv data/thruster_bench/bench_plan.csv \
  --print-plan
```

### 4.5 低功率冒烟测试

第一次测试不要直接扫完整曲线，先用很小命令确认：

```text
1. 下位机能收到命令。
2. 推进器方向符合预期。
3. disable / 超时会停机。
4. 力传感器读数方向正确。
5. CSV 能正常写入。
```

命令：

```bash
cd ./ros2_ws

PYTHONPATH=src/thruster_curve_measurement:src/hardware_bridge \
uv run python -m thruster_curve_measurement.thruster_bench_measure \
  --mode command \
  --thruster-port /dev/ttyUSB0 \
  --sensor-port /dev/ttyUSB1 \
  --csv data/thruster_bench/smoke.csv \
  --command 0.05 \
  --output-limit 0.1 \
  --duration 2 \
  --pre-zero 1 \
  --post-zero 1 \
  --zero-samples 5
```

如果方向反了，优先检查机械安装、力传感器受力方向、ESC/推进器方向定义。不要只靠后处理把符号改回来，否则后续控制曲线容易混乱。

### 4.6 执行独立台架阶跃测量

确认冒烟测试正常后，执行保守阶跃测量：

```bash
cd ./ros2_ws

PYTHONPATH=src/thruster_curve_measurement:src/hardware_bridge \
uv run python -m thruster_curve_measurement.thruster_bench_measure \
  --config src/thruster_curve_measurement/config/thruster_bench_measure_step.yaml \
  --thruster-port /dev/ttyUSB0 \
  --sensor-port /dev/ttyUSB1 \
  --csv data/thruster_bench/run001.csv \
  --summary-csv data/thruster_bench/run001_summary.csv \
  --plot data/thruster_bench/run001.png \
  --points-plot data/thruster_bench/run001_points.png \
  --fit-csv data/thruster_bench/run001_fit.csv \
  --fit-plot data/thruster_bench/run001_fit.png
```

想提高实际输出时，优先只提高上位机限幅，例如：

```bash
--output-limit 0.2
```

但它仍会被 MCU 侧 `THRUSTER_BENCH_OUTPUT_LIMIT` 限住。需要更高输出时，必须明确修改并重新烧录 FineSUB 固件。

如果要看连续正扫/反扫的 command-force 回环，可以使用 `loop` 模式。该模式连续执行 `-amplitude -> +amplitude -> -amplitude`，中间不回零、不阶跃停顿：

```bash
cd ./ros2_ws

PYTHONPATH=src/thruster_curve_measurement:src/hardware_bridge \
uv run python -m thruster_curve_measurement.thruster_bench_measure \
  --config src/thruster_curve_measurement/config/thruster_bench_measure_step.yaml \
  --thruster-port /dev/ttyUSB0 \
  --sensor-port /dev/ttyUSB1 \
  --mode loop \
  --amplitude 1.0 \
  --ramp 120 \
  --csv data/thruster_bench/M001_loop/raw.csv \
  --plot data/thruster_bench/M001_loop/curve.png
```

这里 `--ramp 120` 表示 `-1 -> +1` 用 120 秒，随后 `+1 -> -1` 再用 120 秒。`curve.png` 中会用不同颜色区分正扫和反扫。

### 4.7 独立台架输出 CSV 字段

独立台架 `--csv` 输出字段和 hardware_bridge 链路略有不同，不再包含 8 路 canonical 推进器列：

```text
wall_time_unix          系统 wall time
monotonic_time          单调时钟
elapsed_sec             本次测量开始后的时间
phase                   当前测量阶段
target_index            固定为 0，表示台架唯一推进器
target_name             固定为 single_thruster
target_command          当前目标 command
output_limit            上位机发送的限幅
sensor_ok               1 表示本行力传感器读取成功
sensor_error            Modbus 错误信息
raw_value               Modbus 原始寄存器值
signed_value            16 位补码恢复后的 signed 值
force_n                 按传感器公式换算得到的力，单位 N
zero_offset_n           测量前零点中位数
force_zeroed_n          扣零后的力，单位 N
sample_window           1 表示该行属于阶跃末尾稳态统计窗口
point_index             当前阶跃点序号
repeat_index            当前重复轮次
status_ok               1 表示本行收到下位机 status
status_enabled          下位机当前是否使能
status_accepted         下位机是否接受最近命令
status_reject_flags     下位机拒绝原因 bit mask
status_command          下位机回显的 command
status_applied          下位机实际 applied 输出
status_rpm              DSHOT 遥测 RPM；如果底层没有有效遥测，可能为 0
status_command_count    下位机累计接收命令数
```

`status_reject_flags` 常见值：

```text
0   正常
2   disabled
4   timeout
8   bad command
```

### 4.8 独立台架紧急停止

独立台架不经过 `hardware_bridge`，因此不能用 `ros2 param set /hardware_bridge enabled false` 停机。优先使用：

```text
1. 物理急停或切断 ESC 动力电。
2. Ctrl-C 停止 thruster_bench_measure；程序会在退出前发送多次 disable 命令。
3. 下位机 500 ms 收不到新命令会自动 disable。
```

正式测试时仍应准备硬件急停，因为软件停机依赖串口链路和 MCU 仍然正常工作。

## 5. hardware_bridge 链路：启动 hardware_bridge

启动时保持 bridge 的输入 topic 为正常推进器输入 `/finsrov/thrusters_out`，并先保持 `enabled=false`：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py \
  params_file:=src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  input_thruster_topic:=/finsrov/thrusters_out \
  debug_mask:=1 \
  debug_echo_decimation:=1 \
  enabled:=false
```

建议另开一个终端查看 MCU 回显：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/thruster_cmd_echo --full-length
```

确认回显正常后，再使能硬件：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled true
```

紧急停止：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled false
```

## 6. hardware_bridge 链路：先打印测量计划

以 `H_LF`，即 index 4 为例：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure \
  --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml \
  --index 4 \
  --sensor-port /dev/ttyUSB0 \
  --output-dir data/thruster_curve \
  --motor-name M001 \
  --print-plan
```

也可以通过 launch 打印测量计划：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 launch thruster_curve_measurement thruster_curve_measure.launch.py \
  index:=4 \
  sensor_port:=/dev/ttyUSB0 \
  output_dir:=data/thruster_curve \
  motor_name:=M001 \
  print_plan:=true
```

Windows 串口可以写成：

```bash
--sensor-port COM4
```

Linux 下常见串口是：

```bash
--sensor-port /dev/ttyUSB0
--sensor-port /dev/ttyUSB1
```

注意：串口名称取决于实际运行 `thruster_curve_measure` 的机器，而不是 SSH 客户端所在的机器。

```text
程序在 Windows 本机运行，且传感器插在 Windows 本机上：使用 COM4 这类名称
程序在 Linux 服务器运行，且传感器插在 Linux 服务器上：使用 /dev/ttyUSB0、/dev/ttyACM0 这类名称
```

如果通过 SSH 在服务器上执行测量命令，即使你的个人电脑上看到的是 `COM4`，服务器程序也不能直接访问这个 `COM4`。此时应在服务器上查询实际串口：

```bash
ls /dev/ttyUSB* /dev/ttyACM*
```

也可以插拔传感器时观察内核日志：

```bash
dmesg -w
```

## 7. 整机 / hardware_bridge 链路：执行测量

整机 / hardware_bridge 链路保留原有阶跃测试，同时新增 `rpm-sweep` 测试：

```text
step       原有方法：离散 command 阶跃，等待局部稳态，输出 command -> Force 曲线
rpm-sweep  新增方法：连续慢速扫 command，结合转速回传，输出 RPM -> Force 曲线和 c1
```

### 7.1 原有方法：阶跃 command-force 测量

测 index 4，即 `H_LF`：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure \
  --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml \
  --index 4 \
  --sensor-port /dev/ttyUSB0 \
  --output-dir data/thruster_curve \
  --motor-name M001
```

该命令会统一写入：

```text
data/thruster_curve/M001/raw.csv
data/thruster_curve/M001/summary.csv
data/thruster_curve/M001/curve.png
data/thruster_curve/M001/points.png
data/thruster_curve/M001/fit.csv
data/thruster_curve/M001/fit.png
data/thruster_curve/M001/rpm_fit.csv
data/thruster_curve/M001/rpm_fit.png
```

如果 `data/thruster_curve/M001/` 已经包含数据，程序会直接报错，避免覆盖已有测量结果。确认要覆盖时显式加：

```bash
--overwrite
```

等价的一键 launch 命令为：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 launch thruster_curve_measurement thruster_curve_measure.launch.py \
  index:=4 \
  sensor_port:=/dev/ttyUSB0 \
  output_dir:=data/thruster_curve \
  motor_name:=M001
```

`thruster_curve_measure.launch.py` 默认会读取已安装的参数文件：

```text
share/thruster_curve_measurement/config/thruster_curve_measure_step.yaml
```

如果要改用其他参数文件，可以覆盖：

```bash
config_file:=src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml
```

参数含义：

```text
--index 4              测第 4 路 canonical 推进器，即 H_LF
--sensor-port          力传感器 Modbus 串口
--mode                 测量模式；step 为稳态 command 曲线，rpm-sweep 为连续转速-推力曲线
--amplitude 1.0        备用阶跃范围；当 step_commands 为空时使用
--ramp                 连续扫 command 的时间；rpm-sweep 中表示半程时间，即 -A -> +A 的时间
--step 0.1             备用指令点间隔；当 step_commands 为空时使用
--step-hold 5          每个指令点保持 5 秒
--step-sample 2        只统计每个指令点最后 2 秒，避开刚切换时的瞬态
--step-rest 1          每个指令点后回零 1 秒，让水流和结构振动稍微衰减
--rate 10              10 Hz 发布推进器指令并读取力传感器
--zero-samples 30      测量前采 30 个零点样本，用中位数扣零
--output-dir           输出根目录，例如 data/thruster_curve
--motor-name           电机编号或样品名，例如 M001；最终目录为 <output-dir>/<motor-name>/
--overwrite            允许写入已有数据的输出目录；默认遇到非空目录会报错
raw.csv                完整时序数据
summary.csv            每个阶跃点稳态窗口的统计值
curve.png              完整时序中的 command 和 force 关系图
points.png             稳态采样点离散图，包含 mean +/- std
fit.csv                step 模式下，正向/反向分别拟合得到的 command-force 模型参数
fit.png                step 模式下，稳态离散点和 command-force 拟合曲线
rpm_fit.csv            基于转速的 T = c1 * omega * abs(omega) 拟合参数；rpm-sweep 直接使用 raw.csv 密集样本
rpm_fit.png            command-RPM 离散点和 omega-force 拟合曲线；rpm-sweep 中样本点会更多
--fit-min-abs-command  拟合时忽略 abs(command) 小于等于该值的点，默认 0
--rpm-fit-min-abs-rpm  拟合 c1 时忽略 abs(rpm) 小于等于该值的点，默认 0
--no-fit               只采集数据，不生成稳态点图和拟合结果
--force-topic          实时发布扣零后的力值，默认 /finsrov/thruster_curve/force_zeroed_n
--force-log-period     终端打印实时力值的周期，单位秒；设为 0 可关闭终端实时打印
--rpm-topic            订阅 hardware_bridge 发布的转速，默认 /finsrov/hardware/motor_rpm_raw
--rpm-stale-sec        转速数据最大允许延迟，默认 0.5 秒；设为 0 表示不过期
```

`--csv`、`--summary-csv`、`--plot`、`--points-plot`、`--fit-csv`、`--fit-plot`、`--rpm-fit-csv`、`--rpm-fit-plot` 仍可作为兼容参数覆盖单个输出文件，但常规测量不要再逐个传这些路径。

其中 `--config` 读取默认参数文件：

```text
src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml
```

这个文件里已经包含阶跃测量常用值：

```text
mode: step             默认阶跃模式
step_commands          非均匀指令点：两头疏、中间死区附近密
step_hold: 5.0         每个指令点保持 5 秒
step_sample: 2.0       每个点最后 2 秒进入稳态统计
step_rest: 1.0         每个点后回零 1 秒
repeat_rest: 90.0      repeat_count > 1 时，两轮完整扫描之间回零静置 90 秒
rate: 10.0             10 Hz 发布指令并读取力传感器
zero_samples: 30       测量前采 30 个零点样本
no_fit: false          默认生成稳态点图和拟合结果
fit_min_abs_command: 0 默认拟合时只排除 command = 0
force_topic            实时发布扣零后的力值
force_log_period: 1.0  每 1 秒在终端打印一次当前力值
rpm_topic              hardware_bridge 发布的 canonical 8 路转速
rpm_stale_sec: 0.5     转速样本超过 0.5 秒未更新则不写入 summary/拟合
```

命令行参数优先级高于 YAML。如果临时想把单点保持时间改成 8 秒，不需要改文件，可以直接覆盖：

```bash
--step-hold 8
```

阶跃测量流程为：

```text
1. 采集零点样本
2. 零推力等待
3. 对非均匀 step_commands 中的每个指令点执行：
   a. 保持该指令 5 秒
   b. 只把末尾 2 秒数据计入 summary
   c. 回零 1 秒
4. 零推力结束
5. 退出前自动发送全 0 推进器指令
```

默认 `step_commands` 的 command 分布为：

```text
[-1.0, -0.5] 和 [0.5, 1.0]：较疏
[-0.4, -0.15] 和 [0.15, 0.45]：中等
[-0.1, 0.1]：最密，用于捕捉零附近死区边界和低速启动临界点
```

如果要重复整组阶跃点，用：

```bash
--repeat-count 2
```

此时默认会在两轮完整扫描之间回零静置 `repeat_rest: 90.0` 秒，尽量让水池整体环流衰减。如果水池较小或最大推力较大，可以把 `repeat_rest` 改到 120 秒以上。

如果想手动指定更稀疏的指令点，用：

```bash
--step-commands=-1,-0.8,-0.6,-0.4,-0.2,0,0.2,0.4,0.6,0.8,1
```

### 7.2 新增方法：rpm-sweep 转速-推力测量

如果目标是辨识 `RPM -> Force`，使用 `rpm-sweep` 模式。它不等待每个 command 的稳态，也不生成 `summary.csv/points.png/fit.csv` 这套阶跃稳态产物；它会让 command 连续慢速变化，并直接用 `raw.csv` 中连续扫描阶段每一行有效的 `target_rpm + force_zeroed_n` 样本拟合：

```text
T = c1 * omega * abs(omega)
```

以 index 4，即 `H_LF` 为例：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure \
  --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml \
  --index 4 \
  --sensor-port /dev/ttyUSB0 \
  --mode rpm-sweep \
  --amplitude 1.0 \
  --ramp 180 \
  --rate 20 \
  --output-dir data/thruster_curve \
  --motor-name M001_rpm_sweep
```

该命令会统一写入：

```text
data/thruster_curve/M001_rpm_sweep/raw.csv
data/thruster_curve/M001_rpm_sweep/curve.png
data/thruster_curve/M001_rpm_sweep/rpm_fit.csv
data/thruster_curve/M001_rpm_sweep/rpm_fit.png
```

其中 `omega` 来自 `/finsrov/hardware/motor_rpm_raw` 中对应推进器的 RPM 回传。程序拟合 `c1` 时只使用以下两个连续扫描阶段：

```text
ramp_negative_to_positive
ramp_positive_to_negative
```

保持段和回零段会保存在 `raw.csv` 里，方便检查零点漂移、残余水流和台架状态，但不会进入 `rpm_fit.csv`。

这里 `--ramp 180` 表示 `-1 -> +1` 用 180 秒，随后 `+1 -> -1` 再用 180 秒。完整扫描时间约为：

```text
pre_zero + settle + 2 * ramp + post_zero
```

`--ramp` 越短，command 变化越猛烈，转速和力传感器之间的动态延迟越容易污染 `RPM -> Force` 拟合。第一次建议用 180 秒或更长；如果水池小、惯性大或转速回传抖动明显，可以加到 240 秒。

多个推进器对比时，必须使用同一套参数。例如四个水平推进器建议都用：

```text
--mode rpm-sweep
--amplitude 1.0
--ramp 180
--rate 20
```

不要把 `--ramp 30` 的结果和 `--ramp 120` / `--ramp 180` 的结果直接比较。虽然横轴是 RPM，但力传感器响应、水流建立和台架弹性仍有动态延迟，扫速不同会改变同一 RPM 下的瞬时力读数。

`rpm_fit.csv` 中三行含义：

```text
all       正反转一起拟合一个 c1；只有在正反转近似对称且符号统一时才适合作为单一模型
positive 只用 omega > 0 的点拟合；用于正转分支
negative 只用 omega < 0 的点拟合；用于反转分支
```

如果不同推进器的 `all` 差异很大，先不要直接判断为推进器差异，应先检查：

```text
1. force 符号是否统一：同一套坐标下，`rpm > 0` 时 force 应该具有一致符号。
2. 正桨/反桨是否混在一起比较：正桨和反桨应按对应方向分别比较。
3. positive / negative 分支是否不对称：单个推进器正反转本来可能不同，不应只看 all。
4. ramp/rate 是否一致：不同扫速的数据不能直接比较 c1。
5. 零点和回零段是否异常：post_zero 若仍有较大残余力，说明水流或台架还没有恢复。
```

对于单独拆下来的推进器台架测量，曲线差异不能归因于整机位置或整机流场。更常见的原因是测试坐标没有统一，或者测试工况没有统一。比较 M005、M006、M007、M008 这类单推进器数据时，建议先确认四次测量的 `amplitude/ramp/rate` 完全一致，再把 `rpm_fit.png` 中的符号方向统一。

如果只是为了控制分配或仿真，建议优先使用分段模型：

```text
omega > 0:  T = c1_positive * omega * abs(omega)
omega < 0:  T = c1_negative * omega * abs(omega)
```

只有在 `positive` 和 `negative` 的 c1 接近时，再考虑使用 `all` 这一行作为单一 `c1`。

### 7.3 辅助模式：ramp 和 loop

如果只是想快速看 command-force 连续趋势，可以使用 `ramp`；如果要观察连续正扫/反扫是否存在迟滞，可以使用 `loop`。`loop` 会连续执行：

```text
-amplitude -> +amplitude -> -amplitude
```

中间不回零、不做阶跃停顿。`curve.png` 的 command-force 散点会用不同颜色区分 `negative to positive` 和 `positive to negative`：

```bash
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure \
  --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml \
  --index 4 \
  --sensor-port /dev/ttyUSB0 \
  --mode loop \
  --amplitude 1.0 \
  --ramp 120 \
  --output-dir data/thruster_curve \
  --motor-name M001_loop
```

这里 `--ramp 120` 表示 `-1 -> +1` 用 120 秒，随后 `+1 -> -1` 再用 120 秒。完整连续扫描时间约为 `pre_zero + settle + 2 * ramp + post_zero`。

## 8. hardware_bridge 链路：实时查看力传感器读数

测量节点会在保存 CSV、图像和拟合结果的同时，实时发布扣零后的力值：

```text
/finsrov/thruster_curve/force_zeroed_n
```

另开一个终端即可查看：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/thruster_curve/force_zeroed_n
```

这个 topic 的类型是 `std_msgs/msg/Float32`，单位是 N。它对应 CSV 中的 `force_zeroed_n`，即已经扣除了测量开始前零点中位数的力值。

测量节点终端本身也会默认每 1 秒打印一次：

```text
force_zeroed=+1.234 N, raw_force=+1.567 N, command=+0.300, rpm=1234.5, phase=step_sample
```

在 `rpm-sweep` 模式下，`phase` 会显示为 `ramp_negative_to_positive` 或 `ramp_positive_to_negative` 等连续扫描阶段。

如果只想保留 topic，不想让终端刷屏，可以启动时加：

```bash
--force-log-period 0
```

如果完全不需要实时 topic，可以加：

```bash
--force-topic ""
```

## 9. hardware_bridge 链路：8 路推进器建议命令

建议逐个测，不要同时测多个推进器。下面是原有阶跃测量命令：

```bash
# 0 V_LF
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 0 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M001

# 1 V_LB
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 1 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M002

# 2 V_RB
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 2 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M003

# 3 V_RF
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 3 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M004

# 4 H_LF
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 4 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M005

# 5 H_RF
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 5 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M006

# 6 H_RB
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 6 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M007

# 7 H_LB
./scripts/run_ros2_uv.sh ros2 run thruster_curve_measurement thruster_curve_measure --config src/thruster_curve_measurement/config/thruster_curve_measure_step.yaml --index 7 --sensor-port /dev/ttyUSB0 --output-dir data/thruster_curve --motor-name M008
```

如果要对同一路执行 `rpm-sweep`，在对应命令中加入 `--mode rpm-sweep --amplitude 1.0 --ramp 180 --rate 20`，并把 `--motor-name` 改成带 `_rpm_sweep` 后缀，例如：

```bash
--mode rpm-sweep --amplitude 1.0 --ramp 180 --rate 20 --motor-name M005_rpm_sweep
```

每测完一路，建议等待水流衰减后再测下一路。

## 10. 迟滞与扫描方向

`rpm-sweep` 模式会执行双向连续扫描：

```text
-amplitude -> +amplitude -> -amplitude
```

因此同一次测量里已经包含正扫和反扫两条轨迹。它们的差异可能来自推进器/电调动态、力传感器响应、水流惯性和台架弹性，不应默认简单取平均。

推荐判断方式：

```text
1. 先看 rpm_fit.png 的 omega-force 散点是否大致落在同一条平方曲线上。
2. 如果正扫和反扫在同一 RPM 下明显分离，优先加长 --ramp，例如从 180 秒加到 240 秒。
3. 如果加长 --ramp 后仍明显分离，说明系统存在迟滞或动态滞后；此时应保留正扫/反扫信息，不要直接平均。
4. 如果只需要仿真或控制分配的准静态参数，优先看 positive / negative 分支的 c1，而不是只看 all。
```

如果专门想观察 command-force 的迟滞回环，可以额外使用 `--mode loop`；但正式 `RPM -> Force` 参数辨识仍推荐 `--mode rpm-sweep`。

## 11. 输出文件说明（整机 / hardware_bridge 链路）

`raw.csv` 输出完整时序数据，包含每一次发布指令和读取传感器的原始记录：

```text
wall_time_unix       系统 wall time
monotonic_time       单调时钟
elapsed_sec          本次测量开始后的时间
phase                当前测量阶段
target_index         当前测量推进器 index
target_name          当前测量推进器名称
target_command       当前目标推进器 command
V_LF...H_LB          8 路 canonical 推进器指令
sensor_ok            1 表示本行力传感器读取成功
sensor_error         Modbus 错误信息
raw_value            Modbus 原始寄存器值
signed_value         16 位补码恢复后的 signed 值
force_n              按传感器公式换算得到的力，单位 N
zero_offset_n        测量前零点中位数
force_zeroed_n       扣零后的力，单位 N
target_rpm           当前测量推进器的转速，来自 rpm_V_LF...rpm_H_LB 对应列
target_omega_rad_s   target_rpm 换算得到的角速度，单位 rad/s
rpm_age_sec          写入本行时，最近一次 RPM 消息的年龄
rpm_V_LF...rpm_H_LB  8 路 canonical 推进器转速，单位 RPM
sample_window        step 模式中表示该行属于阶跃末尾稳态统计窗口；rpm-sweep 中为 0
point_index          step 模式中的阶跃点序号；rpm-sweep 中为空
repeat_index         step 模式中的重复轮次；rpm-sweep 中为空
```

`step` 阶跃测量重点看：

```text
raw.csv       完整时序数据
summary.csv   每个阶跃点末尾稳态窗口的统计值
curve.png     完整时序中的 command-force 散点
points.png    稳态采样点离散图，包含 mean +/- std
fit.csv       command-force 拟合参数
fit.png       稳态离散点和 command-force 拟合曲线
rpm_fit.csv   如果 RPM 回传有效，也会基于 summary 拟合 omega-force 的 c1
rpm_fit.png   command-RPM 离散点和 omega-force 拟合曲线
```

`summary.csv` 只在 `--mode step` 下生成，按每个阶跃点汇总末尾稳态窗口：

```text
target_index         推进器 index
target_name          推进器名称
repeat_index         重复轮次
point_index          阶跃点序号
target_command       该阶跃点指令
sample_count         进入统计窗口的有效传感器样本数
force_mean_n         扣零后平均推力
force_median_n       扣零后中位推力
force_std_n          扣零后标准差
force_min_n          扣零后最小值
force_max_n          扣零后最大值
rpm_mean             该阶跃点稳态窗口内的平均转速，单位 RPM
rpm_median           该阶跃点稳态窗口内的中位转速，单位 RPM
rpm_std              该阶跃点稳态窗口内的转速标准差，单位 RPM
rpm_min              该阶跃点稳态窗口内的最小转速，单位 RPM
rpm_max              该阶跃点稳态窗口内的最大转速，单位 RPM
omega_mean_rad_s     rpm_mean 换算得到的平均角速度，单位 rad/s
```

`fit.csv` 输出 command-force 拟合参数。当前采用带符号的二次多项式，并且正向、反向分开拟合：

```text
force_n = quadratic_coeff * command * abs(command) + linear_coeff * command
```

`rpm-sweep` 测量重点看：

```text
raw.csv       完整密集采样，可用于后处理和排查异常点
curve.png     command-force 时序散点，用于快速检查力方向、异常跳变和回零段残余力
rpm_fit.csv   c1 拟合结果
rpm_fit.png   左图 command-RPM，右图 omega-force 与拟合曲线
```

`rpm_fit.csv` 输出基于转速的静水平方模型参数，模型对应：

```text
T = c1 * omega * abs(omega)
```

其中：

```text
T      推力，单位 N
omega  转速换算得到的角速度，单位 rad/s
c1     输出到 rpm_fit.csv 的 c1 列
```

`rpm-sweep` 拟合数据来源为 `raw.csv` 中连续扫描阶段的每一行有效 `target_rpm` 和 `force_zeroed_n`。程序只取：

```text
ramp_negative_to_positive
ramp_positive_to_negative
```

`pre_zero`、`settle_zero`、`hold_negative`、`hold_positive`、`post_zero` 等阶段会写入 `raw.csv`，但不进入 `rpm_fit.csv` 的 c1 拟合。

当前会分别输出三行：

```text
all       正反向一起拟合一个 c1，对应最简静水模型
positive 只用 omega > 0 的点拟合
negative 只用 omega < 0 的点拟合
```

这里的 `T` 和 `omega` 都是带符号量。若推进器固定方向或力传感器方向改变，`force_zeroed_n` 的符号也会改变，`c1` 的正负随之改变。因此跨推进器比较前必须统一坐标约定：

```text
统一约定示例：
  rpm > 0 表示该推进器的正转方向
  force_zeroed_n > 0 表示该推进器在台架定义的正推力方向

如果某一路 `rpm > 0` 时 force 与其他推进器符号相反，先检查台架安装方向、传感器受力方向、桨方向和 RPM 符号映射。
```

`rpm_fit.png` 左图为 `command -> RPM`，右图为 `omega -> force` 及 `T = c1 * omega * abs(omega)` 拟合曲线。实际推进器正反向常不完全对称，所以控制器或仿真初值可以先看 `all`，再根据误差决定是否采用 `positive/negative` 分段参数。

当前传感器换算公式为：

```text
signed_value = raw_value - 65536  if raw_value > 32767
signed_value = raw_value          otherwise

force_n = signed_value / 100 * 0.98
```

## 12. Modbus 参数调整（两条链路通用）

默认参数：

```text
baudrate: 115200
bytesize: 8
parity: N
stopbits: 1
timeout: 0.2
register address: 205
count: 1
device_id: 1
divisor: 100
scale: 0.98
```

如果传感器参数不同，可以这样覆盖：

```bash
--sensor-baudrate 115200
--sensor-bytesize 8
--sensor-parity N
--sensor-stopbits 1
--sensor-timeout 0.2
--sensor-address 205
--sensor-count 1
--sensor-device-id 1
--sensor-divisor 100
--sensor-scale 0.98
--sensor-offset-n 0.0
```

## 13. 安全建议

两条链路的停机方式不同，测试前必须明确自己正在使用哪条链路。

独立 ThrusterBench 台架链路：

```text
1. 首选物理急停或切断 ESC 动力电。
2. Ctrl-C 停止 thruster_bench_measure；程序退出前会发送多次 disable 命令。
3. 下位机固件默认 500 ms 收不到新命令会自动 disable。
4. 不能依赖 /hardware_bridge enabled false，因为独立台架根本不经过 hardware_bridge。
```

hardware_bridge 链路：

```text
1. 可以通过 ros2 param set /hardware_bridge enabled false 停止硬件输出。
2. 仍应准备物理急停，因为软件停机依赖 ROS、串口和 MCU 都正常。
```

hardware_bridge 链路推力曲线测定常使用全指令范围：

```bash
--amplitude 1.0
```

在 `step` 模式下，这表示阶跃点覆盖 `-1.0` 到 `+1.0`；在 `rpm-sweep` 模式下，这表示连续扫描覆盖 `-1.0` 到 `+1.0`。独立 ThrusterBench 台架链路也可以让 `step_commands` 覆盖 `-1.0` 到 `+1.0`，但实际输出还会受到 `--output-limit` 和下位机 `THRUSTER_BENCH_OUTPUT_LIMIT` 双重限制。

如果只是第一次验证接线方向、串口读数和数据记录流程，hardware_bridge 链路可以临时使用小幅值，例如：

```bash
--amplitude 0.1
```

独立台架链路建议优先保持默认保守配置，或显式设置：

```bash
--mode command --command 0.05 --output-limit 0.1 --duration 2
```

确认方向、传感器读数、MCU/status 回显都正确后，再逐步提高 `--amplitude` 或扩大独立台架链路的 `step_commands`。

hardware_bridge 链路如果出现异常，立即执行：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled false
```

并确认 `/finsrov/hardware/thruster_cmd_echo` 中 `applied_thrust` 已经回到 0。
