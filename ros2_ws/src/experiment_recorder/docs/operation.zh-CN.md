# FinsSim 实验数据采集包

`experiment_recorder` 是面向 FinsSim 论文实验的通用 ROS2 记录包。基础
`record_experiment` 命令只负责录包和分析，**不会**控制推进器或使能硬件；而
`run_t1_experiment`、`run_t2_experiment`、`run_t1_simulation` 与
`run_t2_simulation` 是建立在它之上的实验 runner，会按冻结 YAML 启动所选 controller、
录包并自动分析。实机 runner 只有在操作者明确提供 `--arm` 后才会请求使能 bridge。

本文默认的 T1/T2 协议、`/finsrov/...` topic、控制器和硬件检查构成 FinsROV profile；通用 recorder、event marker、manifest 和离线产物可供其他 vehicle profile 使用。

> AprilTag/Snell/fusion 运行时稳定性记录（独立只订阅节点）见
> [AprilTag 感知稳定性记录器](apriltag-stability-recorder.zh-CN.md)。它可以与 T1 同时运行，
> 不会改动 T1、控制器或硬件桥接。

基础 recorder 的职责是：

1. 按实验类型启动 `ros2 bag record`；
2. 记录统一的 experiment event marker；
3. 生成包含 git、配置 hash、checkpoint hash 和话题清单的 `manifest.json`；
4. 在实验结束后标记数据是否完整，供离线分析使用。

> 当前默认行为已扩展为：rosbag2 停止并完成 flush 后，recorder 自动执行一次
> `analyze_experiment`，不需要再手动运行 `finalize_experiment`。分析失败不会删除
> 原始 bag，而会在 manifest 中标记 `needs_review` 并保留错误原因。

实验协议正文位于：

```text
/figures/real_experiment_protocol.md
```

## 1. 数据目录

独立的 `record_experiment` session 默认输出到：

```text
./ros2_ws/data/experiments/<experiment-id>/<session-id>/
```

每个 session 包含：

```text
manifest.json          # 自动生成，记录版本和配置来源
raw/rosbag2/            # 原始 sqlite3 rosbag2
derived/                # 离线分析输出
external_truth/         # 仅提供 E1/E2 外部真值时创建
video/                  # 仅实际导入/录制视频时创建；当前 recorder 不自动录像
derived/analysis_report.json  # 自动生成的分析状态、指标和图表清单
derived/pose_samples.csv      # 从 rosbag 规范化导出的位姿样本
derived/command_samples.csv   # 从 rosbag 导出的控制/推进器数组
derived/position_goal_samples.csv # T1 controller_world 位置目标；旧 bag 缺 hold_start 时的恢复时基审计
derived/events.json            # 实验事件及其 metadata
derived/metrics.csv            # E1/E2 有外部真值时的逐话题水平误差
derived/tracking_metrics.csv    # T1 每个目标 hold 窗口的 x/y/z/yaw 误差
derived/target_point_samples.csv # 每个 T1 目标点 hold 窗口的 x/y/z/yaw 序列和基线
derived/plots/*.png            # 自动生成的轨迹、状态和命令图
derived/plots/targets/*_xyzyaw.png # 每个目标点的 x/y/z/yaw 四联图
```

原始 bag、视频和测力计数据不应直接提交 Git。论文仓库只提交分析脚本、汇总 CSV、manifest 和生成图表。

## 2. 构建和查看命令

在 ROS2 workspace 中使用仓库脚本：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select experiment_recorder
./scripts/run_ros2_uv.sh ros2 pkg executables experiment_recorder
```

### 冻结 T1 批次的 PID/PPO 汇总

单个 T1 trial 自动生成四自由度时序与指标；论文比较应再按**目标点/trial**汇总，不能把
ROS 帧当作独立样本。当前已完成的 2026-08-25 静水批次可用下列命令重建三种方法
`PID`、`PPO_WRENCH6`、`PPO_THRUSTER8` 的实机和 Unity 对比：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder compare_t1_experiments \
  --config src/experiment_recorder/config/t1_pid_ppo_comparison_20260825.yaml
```

输出在 `ros2_ws/data/experiments/analysis/T1_static_water_pid_ppo_20260825/`：
`trial_ledger.csv` 保存逐目标来源和 RMSE，`method_summary.csv` 保存均值/中位数，
`paired_deltas_vs_pid.csv` 保存相同目标点的 PPO--PID 差值，`plots/` 保存三方法图。
当旧仿真 bag 缺少 `hold_start`，分析器优先以与冻结目标完全匹配的
`position_controller_world` command 首次记录时间重建同一 hold 起点；若该一帧也在
rosbag 启动期丢失，则用同一 frozen run 中有 marker trial 标定的 `phase_end` 延迟反推。
两种恢复来源都会写入 `timing_provenance`，且均不修改原始 bag。

如果只需要查看将要记录的命令，不启动 bag：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_experiment \
  --experiment-id E1 --profile apriltag --session-id dry_run_001 --dry-run
```

可用 profile：

```text
apriltag  E1/E2，水平视觉/融合定位；不录制原始 IMU、depth、RPM、telemetry
t1        E4/E6/E7，真实 T1 和平面动力学控制话题
t2        E10，真实 T2 加 trajectory reference
t1_sim    E5，ROS2 仿真 T1
t2_sim    E9，ROS2 仿真 T2
full      同时记录 real/sim 话题，只有调试时使用
```

`apriltag` profile 的离线指标只读取 `pool_world.x/y`。ROS 的 `Pose` 消息无法在 rosbag2 中按字段切片，因此融合 pose 消息仍会以原始类型保存其完整字段，但 E1/E2 的分析和汇总不会读取或报告 yaw、深度和 z。

## T1/T2 快速使用指南（实机与仿真）

本节是 T1/T2 的**唯一快速入口**。所有命令均应从 `ros2_ws/` 目录执行；详细的
坐标、指标定义、安全状态机、轨迹参数和故障处理见后续对应章节。每次命令都会自动生成
run/session ID、rosbag、manifest 和 `derived/` 分析结果，正常结束后**不需要手动执行
`finalize_experiment`**。

### 运行阶段提示

所有 runner 使用 ROS2 默认的 `[INFO]` 输出显示当前阶段。实际运行时会依次显示：

1. run 初始化及总 trial 数；
2. `T1 acquisition i/N: target=<id>` 或 `T2 acquisition i/N: trajectory=<id>`，即当前目标点/轨迹及重复次数；
3. recorder 已启动、控制/轨迹执行进度、rosbag 收尾；
4. `automatic analysis started`、`automatic analysis complete`，以及本 trial 的 `analysis_report.json` 路径；
5. 两个 trial 间的等待时间和整批 run 的最终状态。

因此在运行时无需猜测程序是否卡在录包、控制还是绘图阶段。录包与自动分析的阶段消息会保存到
`trials/<trial-id>/logs/recorder.log`；runner 的终端消息则是操作者判断当前 trial 编号和整体进度的主来源。

### 任务、控制接口与默认规模

| 任务                           | 目标与受控自由度                                                                                 | 默认实验规模                                                                            | 可选 controller                                                                                              |
| ------------------------------ | ------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------ |
| **T1：定点保持**         | 在`controller_world` 中保持 `x`、深度 `y`、`z`、`yaw`；roll/pitch 仅监视，不作为目标   | 8 个目标点（4 个水平位置$\times$ 2 个深度）$\times$ 3 repeats；每 trial 60 s 控制段 | 仿真：`PID`、`PPO`-`wrench6`、`PPO`-`thruster8`；实机：`PID`、经过有界分配的 `PPO`-`wrench6` |
| **T2：三维平移轨迹跟踪** | 跟踪`x`、深度 `y`、`z`；roll/pitch/yaw 不跟踪。固定参考深度为 `y=-0.50 m`（水下 0.50 m） | 一次 CLI 只测一条轨迹，默认 3 repeats                                                   | 仿真：`PID_POSITION`、`PPO_WRENCH6`、`PPO_THRUSTER8`；实机：`PID_POSITION`、`PPO_WRENCH6`          |

T2 可选轨迹为 `straight`、`straight_pro`、`circle`、`ellipse`、`lemniscate` 和
`figure_eight`。实机 trial 的初始摆放深度记录为 `y=-0.20 m`，但没有自动预摆位或
实测起始 pose gate；操作者负责确认场地、可见性和安全净空。T2 的 reference 始终由
YAML 生成并写入 manifest，不能在试验后手工替换。

### 共同准备

首次使用、修改 Python entry point 或控制器代码后，先构建相关包：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select experiment_recorder motion_control grpc_ros_adapter
```

先用 `--dry-run` 检查 YAML、checkpoint、轨迹和输出目录。`--dry-run` 不启动 Unity、
controller、rosbag 或硬件。查看当前 CLI 的全部参数：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation --help
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_simulation --help
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment --help
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment --help
```

#### 仿真前提

- 默认使用 `simulators/unity/marus-example` 的
  `HoldForPosition_Fossen_Parallel_30s.unity` Fossen 场景。不要混用旧的
  `Code/UnityProject/...` 副本或在同一项目中打开另一个 scene。
- runner 会复用已经打开且场景匹配的 Unity Editor；否则启动配置中声明的 Unity 版本。
  它会临时切换到 ROS 评估运行时，在结束或 `Ctrl-C` 后退出 **Play Mode**，但不会关闭
  Editor。正常使用不要加 `--no-gui`。
- T1 的实验局部 `time_scale: 10.0` 加速 Unity/ROS2 仿真时钟；T2 使用
  ``ROS2 controller tick 完成后再推进 Unity 一步'' 的 lockstep，`time_scale: 10.0`
  只是墙钟速度上限。两者都不修改训练或全局评估的 `eval_time_scale`。
- 仿真 PPO 必须使用与 checkpoint 完全一致的 action ABI。`wrench6` checkpoint 不能按
  `thruster8` 加载，反之亦然。仿真评估保留训练时的原始动作语义：不附加实机 bridge 的
  推力曲线、安全 gain、后四推进器缩放或 wrench 再缩放。

#### 实机前提与启动顺序

实机 runner **不会**启动 hardware bridge、折射 AprilTag 或 fusion。请在三个独立终端
预先启动支持栈；下面以 RGB 配置为例，IR 相机时改用对应的 `refractive_apriltag_ir.launch.py`：

```bash
# 终端 A：先保持 disabled；正式 runner 的 --arm 才会在预检后使能 bridge。
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py \
  params_file:=src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  enabled:=false

# 终端 B：AprilTag 角点、折射约束位姿。
./scripts/run_ros2_uv.sh ros2 launch perception refractive_apriltag.launch.py \
  detector_params:=src/perception/config/direct_apriltag_rgb.yaml \
  refractive_params:=src/perception/config/refractive_apriltag_rgb.yaml \
  pool_world_params_file:=src/state_estimation/config/pool_world.yaml

# 终端 C：状态融合，发布 /finsrov/pose 等控制状态。
./scripts/run_ros2_uv.sh ros2 launch state_estimation state_fusion.launch.py \
  params_file:=src/state_estimation/config/state_fusion.yaml \
  pool_world_params_file:=src/state_estimation/config/pool_world.yaml
```

在执行任何带 `--arm` 的命令前，确认状态新鲜、无残留 controller，且 deadman/急停可用：

```bash
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/pose
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic info /finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 node list
```

不应同时手工启动另一个 `/motion_controller` 或其他 `/finsrov/thrusters_out` 发布者。
runner 只负责本次试验的 controller 和 recorder；`--arm` 不是启动支持栈的替代品。

### 最简命令与正式命令

以下 `/absolute/path/...` 必须替换为实际、已冻结的 SB3 `.zip` checkpoint。最简 PID
命令不需要 checkpoint；最简 PPO 命令必然需要 checkpoint 和正确的 action interface。

#### T1 仿真：定点保持

最简 PID 评估（默认全部 8 个目标点、每点 3 次）：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PID
```

PPO 6D wrench checkpoint：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PPO --ppo-action-interface wrench6 \
  --checkpoint /absolute/path/to/t1_wrench6_checkpoint.zip
```

PPO 原始 8-thruster checkpoint：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PPO --ppo-action-interface thruster8 \
  --checkpoint /absolute/path/to/t1_thruster8_checkpoint.zip
```

建议首先做一个 GUI pilot，而非直接跑完整 24 个 trial：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PPO --ppo-action-interface wrench6 \
  --checkpoint /absolute/path/to/t1_wrench6_checkpoint.zip \
  --setpoint P01_front_left_shallow --repeats 1
```

#### T2 仿真：固定参考轨迹跟踪

最简 PID pilot（只测一条 `straight`，一次重复）：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_simulation \
  --config src/experiment_recorder/config/t2_simulation_experiment.yaml \
  --controller PID_POSITION --trajectory straight --repeats 1
```

PPO 6D wrench 或原始 8-thruster 的完整示例（删除 `--repeats 1` 即按 YAML 运行 3 次）：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_simulation \
  --config src/experiment_recorder/config/t2_simulation_experiment.yaml \
  --controller PPO_WRENCH6 \
  --checkpoint /absolute/path/to/t2_wrench6_checkpoint.zip \
  --trajectory ellipse --repeats 1

./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_simulation \
  --config src/experiment_recorder/config/t2_simulation_experiment.yaml \
  --controller PPO_THRUSTER8 \
  --checkpoint /absolute/path/to/t2_thruster8_checkpoint.zip \
  --trajectory circle --repeats 1
```

#### T1 实机：PID 与 PPO 定点保持

先做最简预检（不会使能推进器）：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --controller PID \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml \
  --dry-run
```

PID 正式执行只需将末尾改为 `--arm`：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --controller PID \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml \
  --arm
```

PPO 正式执行（checkpoint 从冻结的 controller YAML 读取并写入 manifest）：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --controller PPO \
  --controller-config src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator.yaml \
  --arm
```

#### T2 实机：PID 与 PPO 轨迹跟踪

先选择**一种**轨迹并做预检；T2 的 `--trajectory` 是必填项：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PID_POSITION \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_trajectory_tracking.yaml \
  --trajectory straight --dry-run
```

正式 PID 与 PPO 示例：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PID_POSITION \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_trajectory_tracking.yaml \
  --trajectory ellipse --arm

./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PPO_WRENCH6 \
  --controller-config src/motion_control/config/FinsROV/ppo_trajectory_tracking_wrench6.yaml \
  --trajectory ellipse --arm
```

#### T2 顶视原始视频（可选）

`recording.top_camera_video.enabled` 默认是 `false`。将其改为 `true`，或只对一次
命令追加 `--record-video`，runner 会为**每一个** T2 trial 启动只读录像进程；
`--no-record-video` 可以临时压过 YAML 中的开启设置。录像订阅
`/finsrov/camera/raw/compressed`，这是 C++ detector 在 AprilTag 识别和 debug 绘制之前
按需编码的原始顶视图，因此 MP4 中不会出现 tag 框、角点、文字或 GUI 边框。

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PPO_WRENCH6 \
  --trajectory ellipse --record-video --arm
```

每个文件位于其 trial 自身目录，例如：

```text
ros2_ws/data/experiments/hardware/T2/PPO_WRENCH6/<run_id>/
  trials/<session_id>/video/top_camera_raw.mp4
  trials/<session_id>/video/top_camera_raw.metadata.json
```

sidecar JSON 记录原始话题、无标注来源声明、编码器、帧数、分辨率和源时间范围；run/trial
manifest 也会链接该文件。`direct_apriltag_{rgb,ir}.yaml` 的
`raw_image_rate_hz` 已设为 10 Hz，并与协议中的视频帧率一致。没有录像订阅者时 detector
不会 JPEG 编码原图；若使用自定义 detector YAML，必须使该参数与
`recording.top_camera_video.fps` 相匹配，并保证 `raw_image_enabled: true`。如果 detector
是在改动配置前启动的，应重启 perception；runner 会在下发轨迹前确认至少五帧且原图话题
达到 8 Hz，未通过时会中止该 trial，不会产出时间失真的录像。

实机 `PPO_THRUSTER8` 当前不在正式 T2 试验的启用列表中。T2 PPO 始终从所选
controller profile 的 `motion_controller.ros__parameters.checkpoint_path` 读取 checkpoint；
不支持用 CLI 覆盖它。训练产物必须采用
`<training-run>/checkpoints/<model>.zip` 与同级 `<training-run>/metadata.json` 的布局。
runner 会将 controller profile 和 metadata 复制到 run 的 `provenance/`，并在 run/trial
manifest 中记录两者及 checkpoint 的 SHA-256。因此修改 profile 后的新 run 可追溯，旧 run
仍保留当时的 profile 快照和模型哈希。

### 一次运行中实际发生什么

#### T1 流程

1. runner 解析 YAML、controller profile 和（PPO）checkpoint，并写入 run manifest；仿真还会检查/启动或复用 Unity Fossen Editor。
2. 仿真中每个 trial 会将 ROV 重置至 YAML 的 `controller_world` 原点，再静置 5 s；实机中操作者必须按 YAML 的起始净空要求摆放 ROV，runner 不会替代现场安全确认。
3. runner 启动 recorder，写入 `trial_start`/`hold_start` 事件，向 controller 下发当前目标。每个 trial 记录 5 s 前段、60 s 控制段和 5 s 后段。
4. T1 的主要稳态指标只从 `hold_start+40` 到 `hold_start+60 s` 的 20 s 窗口计算；完整 60 s 曲线仍用于显示响应与 settling time。成功要求 `x`、深度、`z`、yaw 与速度在 YAML 阈值内连续保持 10 s。
5. 每个 trial 结束后 recorder 停止 rosbag、自动分析，并将本 trial 的状态、命令、指标和图写入 `derived/`。

#### T2 流程

1. 每条命令只解析一种 YAML 中注册的 trajectory，并把实际发送的 10 Hz reference CSV 和 SHA-256 冻结到 run manifest。仿真每个 trial 自动 reset；实机由操作者在该轨迹首点上方 `y=-0.20 m` 摆放。
2. controller 接收 `MultiDOFJointTrajectory`，在整个路径中跟踪 `x/y/z`；没有 yaw 目标。实机每条路径默认 3 次，每两个 trial 之间至少等待 15 s；仿真 reset 后可立即开始下一次。
3. 指标从真正的 `trajectory_start`（lockstep 仿真从第一条 controller-stamped runtime reference）开始，覆盖完整的声明路径时长；不会为了跳过初始追赶阶段而裁掉前段。提前取消、安全异常或操作者终止会截断窗口并在 completion 中记录。
4. 已注册时长分别为：`straight`/`lemniscate` 30 s、`straight_pro` 90 s、`ellipse` 92.70 s、`figure_eight` 117.085700 s、`circle` 连续 3 圈共 137.142857 s。T2 不将 roll/pitch/yaw 伪造成跟踪误差，只保留其原始状态和 guard 审计。

运行期间，实机 runner 每秒显示 trajectory/trial 进度、`controller_world` 位置和状态健康度；仿真 runner 每 10 s 仿真时间打印一次状态。一次 `Ctrl-C` 的语义是提前结束当前 trial：仿真会 cancel、录完已有数据、自动分析并停止 Unity Play Mode；实机会执行取消、零推力和 bridge emergency-disable，然后等待 rosbag/分析收尾。两种情况都不要连续按第二次 `Ctrl-C`。

### 自动产生的数据、指标和图像

所有输出均位于：

```text
ros2_ws/data/experiments/hardware/T1/<controller>/<run_id>/        # T1 实机
ros2_ws/data/experiments/simulation/T1/<controller>/<run_id>/      # T1 仿真
ros2_ws/data/experiments/hardware/T2/<controller>/<run_id>/        # T2 实机
ros2_ws/data/experiments/simulation/T2/<controller>/<run_id>/      # T2 仿真
```

每个 run 都包含 `run_manifest.json`、冻结的 YAML/checkpoint SHA-256 与每个
trial 的 `raw/rosbag2/`。跨 trial 的 Unity、gRPC、reset service 或 lockstep controller 日志
按需位于 run 的 `logs/`；每个 trial 自己启动的 controller 与 recorder 日志位于该 trial 的
`logs/`。每个 session 的 `derived/analysis_report.json` 会列出实际参与
分析的话题、指标、图文件与异常；分析失败只会标记 `needs_review`，绝不会删除原始 bag。

| 任务         | 主要 CSV/JSON                                                                                                          | 自动图像                                                                                                                                                                                                                                    | 可用于论文的主要分析                                                                                                                                           |
| ------------ | ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **T1** | `pose_samples.csv`、`command_samples.csv`、`events.json`、`target_point_samples.csv`、`tracking_metrics.csv` | `plots/targets/<target>_xyzyaw.png`：每个 trial 的 `x/y/z/yaw` 四联图，红线为目标基线、绿色阴影为稳态评价窗；另有状态/命令诊断图                                                                                                        | 各自由度 RMSE、MAE/P95、稳态有符号/绝对误差、settling time、成功率、command effort、roll/pitch guard。默认 8 点$\times$ 3 repeats 会得到 24 张逐目标四联图。 |
| **T2** | `resolved_trajectories/<trajectory>.csv`、`reference_aligned_samples.csv`、`t2_metrics.json`/`.csv`            | 非 circle：`plots/t2/path_pool_xy.png`（实机）或 `path_controller_xz.png`（仿真）；`state_vs_reference_controller_xyz.png`、`tracking_error_controller_xyz.png`、命令图。circle 额外按 3 圈拆为 `circle_loop_01/02/03_path_*.png` | cross-track RMSE/P95、time-aligned`x/y/z` RMSE/MAE/P95、depth RMSE、completion、有效 reference 配对率、action energy/smoothness、状态新鲜度与 guard。        |

其中 `<controller>` 固定为 `PID`、`PPO_WRENCH6` 或 `PPO_THRUSTER8`。即使某一组当前
尚无对应策略试验，目录也会保留，避免后续原始数据混放。

T1/T2 分析只使用 rosbag 的原始时间戳和运行时 reference；不得手工修改 CSV 或把
`/finsrov/pose` 的 `pool_world` 轴直接与 `/finsrov/controller/pose` 的
`controller_world` 轴混合比较。T2 分析器会先使用冻结变换将状态与 reference 转到相同
`controller_world` 语义，再计算误差。若需要在旧 run 或补充外部真值后重跑分析，才执行：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder analyze_experiment \
  /absolute/path/to/ros2_ws/data/experiments/<experiment>/<run-or-session> --force
```

后续章节分别展开仿真运行时、T1 实机安全流程、T2 轨迹几何与 E1/E2 定位实验；日常执行应优先
遵循本节的命令和 YAML，不要复制旧 session 的手工 `record_experiment` 命令来代替 T1/T2 runner。

### 历史 T1/T2 数据状态

2026-08-28 前采集的 T1/T2 数据已规范化至当前 `hardware/` 与 `simulation/` 布局。迁移审计、
移动前 manifest 快照及 raw rosbag SHA-256 位于：

```text
ros2_ws/data/experiments/migrations/layout_v2_20260828/migration_manifest.json
```

没有完整 rosbag 的 trial、无法归属的旧 runtime log，以及完整失败的 run 位于：

```text
ros2_ws/data/experiments/archive/<reason>/...
```

手动 Ctrl-C 导致的 `status: interrupted` 不是异常判据；已完成且具备原始 bag 的 trial 保留为
正式数据。该迁移是一次性维护操作，源码中不保留可再次执行的迁移命令。

迁移 journal 位于 `ros2_ws/data/experiments/migrations/<migration-id>/`；每个 run/trial 的
`provenance/original_*manifest.json` 和 `relocation_history` 保留采集时的路径及校验和。迁移成功后
应重新运行 paper reanalysis，不应保留旧路径的符号链接或重复副本。

## 3. E1/E2 一键配置与 CLI

E1/E2 的默认配置位于：

```text
ros2_ws/src/experiment_recorder/config/e1_e2_apriltag.yaml
```

该 YAML 固化了 `apriltag` 录包 profile、E1 输出目录、需要 hash 的状态估计/外参/水池坐标配置，以及硬件桥、AprilTag 和 fusion 的启动清单。启动清单是审计用的声明，不会被 CLI 自动执行；特别是 `hardware_bridge.enabled` 保持 `false`，避免录包命令意外驱动推进器。

现场只需要输入独立测量得到的 `pool_world` 水平真值 `x,y`：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_apriltag_pose \
  --truth-x 0.250 --truth-y -0.400
```

这个命令会自动读取 YAML、生成不重复的 `YYYYMMDD_HHMMSS_E1` session ID，根据真值生成 position ID，并扫描已有真值 CSV 自动分配该位置的下一个 repeat 编号。随后它创建 rosbag、manifest 和一行 `truth_xy.csv`。默认持续记录，静态 pose 完成后按一次 `Ctrl-C`。

只检查解析后的 session、话题和输出路径而不创建 bag，可加 `--dry-run`：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_apriltag_pose \
  --truth-x 0.250 --truth-y -0.400 --dry-run
```

真值文件自动位于：

```text
ros2_ws/data/experiments/E1/<auto-session-id>/external_truth/truth_xy.csv
```

格式为：

```csv
position_id,repeat,truth_x_m,truth_y_m,truth_frame,measurement_method,uncertainty_m,notes
P_XP0p250_YN0p400,1,0.250000,-0.400000,pool_world,surveyed_body_reference,,
```

E2 不重复启动录包，也不重新测量真值；它读取 E1 session 的同一个 bag 和该 CSV，离线比较 pinhole、Snell/constrained 和 fusion 三种输出。正常采集不需要再输入 session ID、profile 或输出目录；如确有需要可用 `--duration`、`--max-bag-duration` 或 `--config-file` 覆盖默认值。

## 4. 通用 recorder 用法

示例：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_experiment \
  --experiment-id E1 \
  --profile apriltag \
  --session-id 20260822_pose_001 \
  --label fixed_pose_grid \
  --config ../src/state_estimation/config/state_fusion.yaml \
  --config ../src/state_estimation/config/state_fusion_extrinsics.yaml \
  --config ../src/state_estimation/config/pool_world.yaml \
  --metadata-json '{"operator":"NAME","truth_source":"surveyed_fixture","horizontal_pose_count":12,"truth_frame":"pool_world_xy"}'
```

默认会一直记录，使用一次 `Ctrl-C` 结束。recorder 会先发送 `session_stop`，再向 rosbag 发送一次停止信号并等待 bag flush 完成。不要连续按第二次 `Ctrl-C`，以免破坏 bag 收尾。

固定时长记录可以使用：

```bash
--duration 120
```

`--duration` 到期、操作者按一次 `Ctrl-C`、或 T1 runner 正常结束 trial 时，都会走同一
个自动收尾流程：发送 `session_stop` → 停止 rosbag → 写入 `ended_at` → 读取 bag →
生成 `derived/` → 更新 `manifest.json`。因此不需要再输入任何 `finalize` 命令。

如需排查录包本身而暂时跳过分析，可显式使用 `--no-auto-analysis`；这不是正式实验
建议流程。之后仍可手动补跑：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder analyze_experiment \
  /absolute/path/to/session --force
```

分析器按 profile 自动选择可识别的运行时数据：

- `apriltag`：输出 `pool_world` 水平轨迹；若存在 `external_truth/truth_xy.csv`，自动计算
  水平 RMSE、平均误差和 P95，并绘制 `trajectory_xy.png`。
- `t1`/`t1_sim`：输出位姿分量、控制/推进器命令和事件时间线；这些 CSV/PNG 是后续
  T1 误差、settling time、能量和安全审计脚本的输入；若事件 metadata 含
  `target_controller_world`，还会自动写入 `tracking_metrics.csv`（位置 RMSE/P95、
  yaw RMSE）和命令平方积分，不会凭空填入缺失指标。正式 T1 的主 RMSE/P95 只使用
  `hold_start + 40` 到 `+60 s` 的固定绿色窗口、以 10 Hz 重采样；完整 60 s 响应只用于
  轨迹展示和 settling time。每个目标会生成一张 `*_xyzyaw.png`，图中绿色阴影明确标出
  主评价窗口，红线为目标基线。
- `t2`/`t2_sim`：若录到 controller 的运行时 reference，会自动时间对齐 reference 与
  `controller_world` pose，输出 cross-track、深度、切线 yaw、状态新鲜度和完成判定；不会
  用发送端时间或手填轨迹替代控制器实际采用的 reference。

自动分析使用 ROS2 bag 中的原始消息时间戳，并将消息类型和识别到的话题列表写入
`analysis_report.json`。若机器没有 `rosbag2_py`、消息类型支持或 matplotlib，原始数据仍
然安全保留，report 会记录 `needs_review` 和具体依赖错误；安装依赖后用
`analyze_experiment --force` 即可重跑。

## Unity/MARUS 仿真实验：保留训练动作语义的 checkpoint 评估

`run_t1_simulation` 与 `run_t2_simulation` 用于论文的 ROS2 仿真实验，和
`run_t1_experiment` / `run_t2_experiment` 使用相同的 rosbag2、事件、manifest
及自动分析目录结构，但**不会启动或访问 hardware bridge**。

其运行时链路是：

```text
指定的 SB3 .zip checkpoint
  -> /sim motion_controller + selected action-interface adapter
  -> /sim/finsrov/thrusters_out
  -> grpc_ros_adapter
  -> Unity Fossen GUI
  -> /sim controller truth topics
  -> record_experiment -> rosbag2 -> automatic derived CSV/plots
```

启动前 runner 会检查是否已有同一 Unity project、同一 Fossen 场景且版本匹配的
Editor。若已有，会复用该 Editor，而**不会**启动第二个 Unity 进程；若同项目已开但
场景不同，runner 会拒绝运行并提示先切换场景，避免破坏操作者未保存的场景状态。
若没有匹配 Editor，默认才会启动 Unity Editor 并打开 Fossen 场景。进入 Play 前，Editor 中的
`ExternalPlayController` 会临时关闭 ML-Agent 行为、启用 `VehicleRosBridge`；
因此 checkpoint 只由 ROS2 controller 执行，不会与场景中的 Agent 抢推进器。
这不是把 SB3 `.zip` 转换成 Unity 内嵌 ONNX 推理。ROS2 只充当 checkpoint
推理宿主：仿真 profile 明确保留训练时的 action ABI，不导入实机的安全 gain、推力
曲线或重投影执行链。

两份仿真 protocol 都让 Unity/MARUS 发布 `/clock`，并让 ROS2 controller、
state-status 与事件记录使用该仿真时钟。T1 的 60 s 控制段按该时钟推进；T2 则以
controller 实际发布的 stamped runtime reference 作为起止依据，因此原轨迹时长绝不会被
runner 的墙钟或其滞后的 `/clock` 回调截短。

T1 仍使用独立的 `time_scale: 10.0` 自由运行 stepped clock。为避免 T2 的 10 Hz PPO
在高倍 Unity 下落后，T2 使用 `mode: ros2_control_lockstep`：每个 Unity physics step 发布
当前 `/clock` 后，gRPC adapter 等待 `/sim/motion_controller/debug/control_tick_complete`，只有
ROS2 controller 已完成该 tick（控制周期之间明确保持上一命令）才允许下一步物理积分。
`time_scale: 10.0` 是请求上限，实际墙钟速度取决于 Unity、gRPC 和策略推理吞吐；它不修改训练或
其他评估中的全局 `eval_time_scale: 1.0`。ack 超时会暂停 Unity 并使 runner 以错误退出，而不是
继续生成时间失配的轨迹数据。

每个 T2 trial 在发出 `trajectory_start` 和 reference 前，还会确认当前 rosbag2 已订阅
**事件与 stamped runtime-reference** 两个话题，并在 DDS 匹配后留出 writer 初始化时间。
正式计时从 controller 第一次实际采用 reference 的精确 `/clock` tick 开始；runner 等到该
reference 已推进完整声明时长才发送取消命令。事件 marker 仍写入 rosbag 并作为完整性审计，
但不会把外部 runner 的时钟传递延迟误算成 PPO/PID 的跟踪误差。

当前 Fossen scene 的固定物理步长为 0.02 s；因此 T2 仿真 PID profile 设为 50 Hz。PPO profile
保持训练时的 10 Hz，并在其余四个 physics step 明确保持上一输出，而不是被偷偷重采样到 50 Hz。

首次使用或改动 package/entry point 后，先构建：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select experiment_recorder motion_control
```

### T1：定点保持

先查看已冻结的计划而不启动任何进程：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PPO --ppo-action-interface wrench6 \
  --checkpoint /absolute/path/to/t1_checkpoint.zip \
  --dry-run
```

正式 GUI 评估（默认 8 个目标点、每点 3 次；每 trial 的控制段 60 s）：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PPO --ppo-action-interface wrench6 \
  --checkpoint /absolute/path/to/t1_checkpoint.zip
```

建议先做一个可视化 pilot：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PPO --ppo-action-interface wrench6 \
  --checkpoint /absolute/path/to/t1_checkpoint.zip \
  --setpoint P01_front_left_shallow \
  --repeats 1
```

`--controller PPO` 必须显式选择动作接口，避免把八推 checkpoint 按六维
wrench 网络加载：

```text
--ppo-action-interface wrench6
  6D PPO policy -> training-time physical allocator -> canonical 8-thruster command

--ppo-action-interface thruster8
  8D PPO policy -> raw canonical normalized 8-thruster vector -> Unity direct force request
```

八推 checkpoint 的原始动作仿真评估示例为：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PPO --ppo-action-interface thruster8 \
  --checkpoint ./artifacts/runs/rl/hold_for_position/eight_thruster/ppo_control_for_hold_position_fossen_parallel_1024_30s_learn_to_swim_dr_v2/checkpoints/best_model.zip \
  --setpoint P01_front_left_shallow --repeats 1
```

`thruster8` 直接发布 checkpoint 的 canonical 八路归一化 action。仿真 runner 将
`VehicleRosBridge` 保持在直接 force-request 模式：ROS2 的八个数值原样送入 Unity，
不会作 wrench 重投影、逐路力界映射、动作变化率限制、归一化到最大推力的映射或后四推进器缩放。
唯一保留的是训练时与 ML-Agents 相同的 `[-1,1]` action clip。

传统 PID 不需要 checkpoint：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_simulation \
  --config src/experiment_recorder/config/t1_simulation_experiment.yaml \
  --controller PID
```

T1 主指标仍严格使用 `hold_start + 40--60 s`，并输出每个目标点的
`x/y/z/yaw` state-target 图、`tracking_metrics.csv`、动作/推进器原始记录和
`analysis_report.json`。

### T2：固定参考轨迹跟踪

每次命令只允许一个 `--trajectory`。几何并不在 sim YAML 重复定义，而是从
`t2_hardware_experiment.yaml` 读取并把本次解析后的 reference CSV 复制进 run
目录；因此 `straight`、`circle`、`ellipse`、`figure_eight` 等轨迹和实机共享
同一来源。

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_simulation \
  --config src/experiment_recorder/config/t2_simulation_experiment.yaml \
  --controller PPO_WRENCH6 \
  --checkpoint /absolute/path/to/t2_wrench6_checkpoint.zip \
  --trajectory circle
```

T2 默认每条轨迹 3 次。它的指标只评估 `x/depth/z`，但仿真 profile 不会把策略的
roll/pitch/yaw wrench 分量归零或缩放。自动分析优先读取带 `/clock` Header 的
`/sim/motion_controller/debug/trajectory_reference_stamped`，并使用
`/sim/finsrov/controller/state/status_stamped` 审计状态新鲜度；无 Header 的
`trajectory_reference` 只保留作兼容性 debug。这样 Unity/ROS2 lockstep 仿真不会把
rosbag 墙钟与仿真时间混在一起。输出包括实际/reference 轨迹、
各自由度误差、cross-track RMSE/P95、depth RMSE、energy/smoothness 与
`t2_metrics.json`。仿真平面轨迹图固定为
`derived/plots/t2/path_controller_xz.png`（circle 则为三张
`circle_loop_*_path_controller_xz.png`）；它使用 Unity 的水平 `x-z` 平面，
不会把 `y` 深度轴误画成水平坐标。

### 运行、停止和输出

- 运行期间终端每 10 s 仿真时间显示一次 `/sim` 中的当前 controller-world 位置；T2 的
  ROS2 lockstep 下其墙钟间隔取决于实际策略推理吞吐。
- `Ctrl+C` 被视为当前 trial 的 `operator_abort`：runner 先 cancel/发送零推力，
  再等待 rosbag2 停止和自动分析完成；已录到的 partial trial 仍保留。
- 正常结束或 `Ctrl+C` 后，runner 会停止 Unity 的 Play Mode，但**不会关闭 Unity
  Editor**；下次 T1/T2 命令可复用仍打开的同项目同场景 Editor。
- 输出位于：

```text
ros2_ws/data/experiments/simulation/T1/<controller>/<run_id>/
ros2_ws/data/experiments/simulation/T2/<controller>/<run_id>/
  run_manifest.json
  resolved_trajectories/             # T2: frozen reference CSV
  trials/<session_id>/raw/rosbag2/
  trials/<session_id>/derived/
  trials/<session_id>/logs/        # trial-owned controller / recorder 日志
  derived/trial_plots/
  logs/
```

若 Unity 已由操作者启动并已打开正确 Fossen ROS 场景，可用 `--no-gui` 让
runner 只连接现有 `/sim` topics。此模式不会替你关闭场景内 ML-Agent；必须先
确认 Unity 只有 `VehicleRosBridge` 在订阅 `/finsrov/thrusters_out`。

覆盖已有 session 只能在确认目标目录正确后使用：

```bash
--overwrite
```

指定 checkpoint 时：

```bash
--checkpoint /absolute/path/to/model.zip
```

recorder 会把 checkpoint 路径和 SHA-256 写进 manifest，但不会复制 checkpoint。提交前应确保 checkpoint 路径在实验机器上可追溯。

## 5. 事件标记

事件标记使用已有的 `msgs/msg/TrajectoryEvent` 类型，但独立使用：

```text
/finsrov/experiment/event
```

例如：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder mark_experiment_event \
  --session-id 20260822_T1_pid_001 \
  --event trial_start \
  --label setpoint_S1 \
  --metadata-json '{"controller":"PID","setpoint_id":"S1","trial":1}'
```

建议事件顺序：

```text
trial_start
phase_start        # 例如 acquisition
hold_start
phase_end
trial_end
```

T1 的 `trial_start`/`hold_start` metadata 至少包含 controller、setpoint ID、
repeat、目标 `[x,y,z,yaw_deg]`、协议版本和 operator；`phase_end` 记录是否进入
steady-state 窗口。示例：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder mark_experiment_event \
  --session-id 20260901_T1_PID_S2_R01 \
  --event hold_start --label P02_front_right_shallow \
  --metadata-json '{"controller":"PID","setpoint_id":"P02_front_right_shallow","repeat":1,"target_controller_world":[1.0,-0.20,-0.50,45.0],"protocol_config":"t1_hardware_experiment.yaml"}'
```

发生故障时不要删除数据，记录：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder mark_experiment_event \
  --session-id 20260822_T1_ppo_001 \
  --event safety_abort \
  --label vision_lost \
  --metadata-json '{"vision_mode":"lost","operator_action":"disarm"}'
```

## 5.1 自动运行 T1 实机 PID/PPO

完整 T1 实验的公共协议只接收一个 YAML：

```text
ros2_ws/src/experiment_recorder/config/t1_hardware_experiment.yaml
```

该文件包含目标点序列、重复次数、控制时长、指标窗口、PID/PPO 控制器配置、
AprilTag/fusion/bridge 配置、输出目录和安全策略。每次 CLI 调用必须显式指定一个
`--controller-config`，因此一次运行只启动一个 PID 或 PPO 控制器。CLI 自动生成
`test_id`、`run_id` 和每个 trial 的 `session_id`。当前 YAML 将
`runner.auto_launch.start_support_stack` 设为 `false`：实验开始前由操作者启动并检查
hardware bridge、refractive AprilTag 和 state fusion；runner 只验证这些支持话题，随后按
trial 启动选中的 motion controller 和 recorder，并执行 24 个 trial。
比较 PID/PPO 时，用同一份公共协议分别运行两次，再按 setpoint/repeat 离线配对；
不会在同一次 CLI 调用中同时启动两个控制器。

正式实验前先检查解析后的计划，不启动任何 ROS2 节点：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --controller PID \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml \
  --dry-run
```

确认水池目标点、deadman、急停、PPO checkpoint 和相机/fusion 状态后，正式运行
必须显式提供 `--arm`；除 YAML 外不需要再输入测试 ID、目标点或时长：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --controller PID \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml \
  --arm
```

`--arm` 是唯一的硬件使能确认。没有该参数时 CLI 拒绝正式执行；YAML 中的
`enabled_without_arm` 永远为 `false`。即使 `hardware_bridge` 由操作者预启动，runner
也会在启动 controller 前通过 `/hardware_bridge/set_parameters` 设置 `enabled=true`，并等待
一条新的 status 确认；无法确认时不会启动 trial。运行中按一次 `Ctrl-C` 是紧急停止，不能把它
当作正常 trial 切换：runner 会立刻取消目标、向 `/finsrov/thrusters_out` 连续发送零
`force_N`、通过 `/hardware_bridge/set_parameters` 设置 `enabled=false`，并等待 bridge
status 在停止请求之后同时确认本地 disabled、MCU 已接收命令、direct-thruster disabled、
RPM/throttle target 为零；
随后会强制结束当前 controller、controller-state adapter 和 recorder 的进程组。该顺序
避免先杀 controller 后仍由 bridge/MCU 保留最后一帧非零命令。确认结果会写入
`run_manifest.json` 的 `emergency_stop` 字段。正式实验中不要连续按第二次 `Ctrl-C`，
以免在该安全序列完成前中断 runner 本身。
完整 run 正常结束时会取消目标并下发零推力帧，但会保留 bridge 的 `enabled=true` 状态；其余
支持栈进程仍由操作者按实验室流程保留或关闭。只有 `Ctrl-C` 或异常才会执行 bridge 禁用和
MCU 状态确认。

该确认依赖 MCU firmware diagnostics（`mcu_diagnostics_supported=true`）。若 telemetry
超时、诊断位未开启或 2 s 内没有收到停止后的状态帧，runner 仍会在超时后强制结束
controller/adapter/recorder，但会将 `emergency_stop.status` 标记为 `unconfirmed`；该 run
不能作为正式实验数据使用，必须先修复 telemetry/diagnostics 再重做。

注意：`--arm` 不是启动支持栈的开关。由于本协议的
`start_support_stack: false`，以下节点必须在执行 CLI 前已经运行且状态正常：

```text
hardware_bridge（可先以 `enabled:=false` 启动；正式 CLI 的 `--arm` 会在预检后使能）
refractive_apriltag
state_fusion
```

runner 不会重复启动或在退出时停止它们；它只在启动阶段等待配置中的支持话题，随后
为每个 trial 启动并停止一个 motion controller 和一个 `record_experiment` recorder。
因此不要同时手工启动第二个 motion controller，也不要让其他节点抢占
`/finsrov/thrusters_out`。支持栈异常或话题超时会使本次实验在发出目标前失败。

如果需要比较 `motion_control/config/FinsROV` 中的不同控制器 profile，
可以在 CLI 中覆盖当前控制器 YAML。公共目标点、时长、重复次数和指标仍来自同一份
T1 实验 YAML；每次覆盖运行只执行选中的 controller（当前协议为 8 个 setpoint
乘 3 次重复，即 24 个 trial），因此可以分别采集 PID 和 PPO，再按 setpoint/repeat
进行配对比较：

```bash
# 只采集传统 PID profile
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --controller PID \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml \
  --arm

# 只采集指定 PPO profile；checkpoint_path 从该 profile 读取
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_experiment \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --controller PPO \
  --controller-config src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator.yaml \
  --arm
```

正式运行前，将命令末尾的 `--arm` 替换为 `--dry-run`，可确认当前 profile、checkpoint
和 24 个 trial 计划。`--controller-config` 支持绝对路径；相对路径可从
`ros2_ws` 或仓库根目录执行。每次覆盖运行的 `run_manifest.json` 会记录实际使用的
controller YAML 和 checkpoint SHA-256，不会修改原始 T1 YAML。

每次运行的目录为：

```text
ros2_ws/data/experiments/hardware/T1/<controller>/<run_id>/
  run_manifest.json
  derived/target_plots/       # 完整 T1 run 汇总的每个目标点四联图
  logs/                         # 仅共享 support stack / runner 生命周期日志
  trials/<session_id>/raw/rosbag2/
  trials/<session_id>/logs/     # trial controller / recorder 日志
  trials/<session_id>/manifest.json
```

`session_id` 和对应的数据目录始终包含 CLI 指定的 controller YAML 文件名（去掉 `.yaml`），例如：

```text
20260822_120000_T1-HW-PID-PPO_traditional_pid_position_yaw_E6_traditional_pid_position_yaw_P01_front_left_shallow_R01
20260822_120000_T1-HW-PID-PPO_ppo_wrench_for_pose_physical_wrench_allocator_E7_ppo_wrench_for_pose_physical_wrench_allocator_P01_front_left_shallow_R01
```

`record_experiment` 也会把策略写入 `manifest.json` 的 `strategy` 字段；即使手工传入
不含策略名的 `--session-id`，录制器也会自动追加 `_<strategy>`，避免 PID/PPO 数据
在文件系统层面混淆。

PPO 的 `controllers.PPO.checkpoint_path` 必须在该 YAML 中填写为冻结 checkpoint
的绝对路径；runner 会同时把它作为 motion-controller 的运行时覆盖、写入 trial
manifest 并计算 SHA-256。这样正式试验不需要再修改 `motion_control` 的
控制器 profile。所有试验仍需在池体 survey sign-off 后执行；自动化只负责可重复的
启动、下发、录包和收尾，不替代现场安全确认。

## 6. E1/E2 AprilTag

E1 只需采集一次水平定位原始数据，E2 使用同一 session 离线重算，不新增 yaw 或深度采集。

### 6.1 真值输入位置

E1 的真值对象是带有刚性 AprilTag 板的 ROV body 参考点在 `pool_world` 水平面上的位置，不是单独某个 tag 中心。Tag 板安装位置在整个实验中必须保持不变；如果外部设备测到的是 tag 中心，需要先依据冻结的 `T_body_tag` 换算到 body 参考点。

启动 recorder 后，真值文件放在：

```text
./ros2_ws/data/experiments/E1/<session-id>/external_truth/truth_xy.csv
```

建议每个位置的每个重复各写一行：

```csv
position_id,repeat,truth_x_m,truth_y_m,truth_frame,measurement_method,uncertainty_m,notes
P01,1,0.250,-0.400,pool_world,survey_grid,0.005,center-left
P01,2,0.250,-0.400,pool_world,survey_grid,0.005,center-left
P01,3,0.250,-0.400,pool_world,survey_grid,0.005,center-left
```

这些数值必须来自独立测量（池底网格、测量夹具或独立定位设备），不能从 `/finsrov/pose`、`/finsrov/vision/refracted_pose_6d` 或 `/finsrov/vision/status` 复制。事件标记中的 `truth_xy` 用于和 rosbag 对齐，最终统计以该 CSV 为准。

启动顺序：

```bash
./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py \
  params_file:=src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  enabled:=false

./scripts/run_ros2_uv.sh ros2 launch perception refractive_apriltag.launch.py \
  detector_params:=src/perception/config/direct_apriltag_rgb.yaml \
  refractive_params:=src/perception/config/refractive_apriltag_rgb.yaml \
  pool_world_params_file:=src/state_estimation/config/pool_world.yaml
./scripts/run_ros2_uv.sh ros2 launch state_estimation state_fusion.launch.py \
  params_file:=src/state_estimation/config/state_fusion.yaml \
  pool_world_params_file:=src/state_estimation/config/pool_world.yaml
```

保持 bridge disabled，固定 ROV 在一个已经测量的 pose，启动 recorder：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_experiment \
  --experiment-id E1 --profile apriltag \
  --session-id 20260822_pose_grid_001 \
  --config src/state_estimation/config/state_fusion.yaml \
  --config src/state_estimation/config/state_fusion_extrinsics.yaml \
  --config src/state_estimation/config/pool_world.yaml \
  --metadata-json '{"horizontal_pose_count":12,"repeats_per_pose":3,"truth_frame":"pool_world_xy","truth_required":true}'
```

每个 pose 记录 15--30 s，并用 event 标出 pose 编号：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder mark_experiment_event \
  --session-id 20260822_pose_grid_001 \
  --event phase_start --label pose_01 \
  --metadata-json '{"pose_index":1,"truth_xy":[0.0,0.0],"truth_frame":"pool_world"}'
```

E2 离线比较：

```text
Snell/constrained pose
Pinhole baseline
Fusion output
```

三种方法必须使用同一 bag、同一真值和同一排除规则。

## 7. E4 平面动力学响应

E4 在静水中运行。每个方向使用正负两种符号、3 个非零等级、每个等级 3 次：

```text
+Fx / -Fx  surge
+Fz / -Fz  sway
+Fy / -Fy  depth/heave
+My / -My  yaw
```

推荐每个脉冲：

```text
0 command：2--3 s
目标 wrench：10--15 s
恢复 0 command：5 s
```

使用现有 wrench command 工具或受控测试脚本发送目标，不要手工修改 bridge 的 motor order/sign。实验时记录 profile `t1`，并将每个轴和等级写入 event metadata。

E4 的分析必须从相同的 body-wrench 定义计算实际分配 wrench、速度响应、稳态误差、settling time 和 cross-axis coupling。

## 8. T1：E5/E6/E7

### 7.1 T1 的性能维度

T1 实机不固定深度。正式指标包括：

```text
controller_world.x       surge position
controller_world.z       sway position
controller_world.y       depth/heave position
controller yaw           yaw
```

roll 和 pitch 不作为被控目标，不向它们报告跟踪性能；但必须记录并设置安全 guard。论文中应称为：

```text
4-DOF station keeping: surge, sway, depth, and yaw
```

不是完整 6-DOF 实机 station keeping。

### 7.2 T1 仿真 E5

每个 PPO 方法使用 3 个训练 seed；PID 不使用训练 seed。每个 seed 和每个测试层使用 100 个固定 episode：

```text
Standard
Disturbance
Disturbance+DR/OOD
```

T1 policy 可以保留 6D policy head，但必须在最终解析配置中明确：

```text
Fx：surge
Fy：depth/heave
Fz：sway
My：yaw
Mx/Mz：固定为 0，不控制 roll/pitch
```

不要使用一个 roll/pitch 仍然有效的 checkpoint，事后再把输出截断。需要 retrain 或确认训练时已经使用相同 action mask。

### 7.3 T1 实机 E6/E7

E4/E5/E6/E7 的统一 T1 协议文件为（其中本节重点是 E6/E7）：

```text
ros2_ws/src/experiment_recorder/config/t1_hardware_experiment.yaml
```

该 YAML 固定目标点、控制时长、成功阈值、指标窗口、话题、配置哈希清单和
安全门。`controller_world` 的 y 轴向上、水面为 0、负值表示水下。正式目标点为：

| ID                      | x (m) | y/depth (m) | z (m) | yaw (deg) |
| ----------------------- | ----: | ----------: | ----: | --------: |
| P01_front_left_shallow  |  1.00 |       -0.20 |  0.50 |         0 |
| P02_front_right_shallow |  1.00 |       -0.20 | -0.50 |        45 |
| P03_rear_right_shallow  | -1.00 |       -0.20 | -0.50 |        90 |
| P04_rear_left_shallow   | -1.00 |       -0.20 |  0.50 |       135 |
| P05_front_left_deep     |  1.00 |       -0.65 |  0.50 |       180 |
| P06_front_right_deep    |  1.00 |       -0.65 | -0.50 |      -135 |
| P07_rear_right_deep     | -1.00 |       -0.65 | -0.50 |       -90 |
| P08_rear_left_deep      | -1.00 |       -0.65 |  0.50 |       -45 |

目标点仍需在下水前完成池体 survey sign-off；若安全余量不足，必须新建协议版本
并重新冻结，不能直接编辑已经产生数据的 YAML。单次 controller 运行对每个
setpoint 做 3 次 trial，共 24 个 trial。PID 和 PPO 应分别运行两次，再按相同的
setpoint/repeat 编号离线配对，不在同一次启动中交错两个控制器。

单次 trial：

```text
手动放置到预注册初始误差范围
5 s 初始静止记录
等待 vision/IMU/depth/dvl ready（最多 15 s）
发送 controller_world goal，并标记 hold_start
控制和记录 60 s
第 40--60 s 为固定主评价/steady-state 窗口，最后 10 s 用于 success 判定
额外记录 5 s，发送 cancel、disarm，并写入 phase_end/trial_end
```

每个 trial 的时间预算为 90 s（不含重新摆放）。PID 和 PPO 必须交错执行，不能先
完成所有 PID 再完成所有 PPO。建议把 battery 状态、水温和操作员备注写入事件 metadata。
当前冻结 profile 的 PID/PPO 控制频率分别为 60/10 Hz；两者不是控制频率消融，
汇总表必须同时报告该差异，不能将结果解释为已隔离控制频率后的算法差异。

统一成功条件至少包括：

```text
平面位置误差
depth 误差
yaw 误差
速度阈值
连续保持时间
无安全事件
```

深度必须和 x/z/yaw 一样计算 RMSE、p95、steady-state error、settling time 和 success rate。

指标计算固定为：原始 bag 保留不变；主 RMSE、绝对误差 p95 和 signed/absolute
steady-state error 只在 `hold_start + 40` 到 `+60 s` 的固定 20 s 窗口计算，并以
10 Hz 重采样。这样从水面/公共起点驶向远处目标的瞬态不会污染定点保持指标。
完整 `hold_start` 后 60 s 仅用于 settling time：其定义为四个误差和速度同时进入阈值
并连续保持 10 s 的首次时间；未收敛记为 `null`。success rate 是
成功 trial 数除以该 controller 的 24 个尝试数；state timeout、workspace exit、
safety/operator abort、controller fault 和 roll/pitch guard failure 都计为失败，
不能删除失败 trial。

yaw 是环形变量。所有 yaw 指标均使用最短有符号角距离
`atan2(sin(yaw-target), cos(yaw-target))`，范围为 $[-\pi,\pi]$；因此测量值
`-179 deg` 相对目标 `+179 deg` 的误差是 `+2 deg`，而不是 `-358 deg`。
`target_point_samples.csv` 同时保留原始 `yaw_deg` 和该环形 `yaw_error_deg`，
目标响应图会以目标附近的连续分支显示 yaw。

统一离线 success 阈值写在 YAML 中：x/z/depth 误差 0.10 m、yaw 误差 10 deg、
线速度 0.05 m/s、yaw rate 5 deg/s、连续保持 10 s、state fresh fraction 至少 0.90。
roll/pitch 只作为 guard（20 deg，连续超过 1 s 判失败），不作为控制性能维度。

启动 recorder 时，必须把统一 YAML 和所有运行时配置都作为 `--config` 传入，以便
manifest 保存 SHA-256。例如 PID：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_experiment \
  --experiment-id E6 --profile t1 --session-id 20260901_T1_PID_S1_R01 \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --config src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml \
  --config src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  --config src/state_estimation/config/state_fusion.yaml \
  --config src/state_estimation/config/state_fusion_extrinsics.yaml \
  --config src/state_estimation/config/pool_world.yaml \
  --duration 90 \
  --metadata-json '{"controller":"PID","setpoint_id":"P01_front_left_shallow","repeat":1,"target_controller_world":[1.0,-0.20,0.50,0.0]}'
```

PPO 命令相同，但使用 E7、PPO 配置，并额外传入冻结 checkpoint：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_experiment \
  --experiment-id E7 --profile t1 --session-id 20260901_T1_PPO_S1_R01 \
  --config src/experiment_recorder/config/t1_hardware_experiment.yaml \
  --config src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator.yaml \
  --config src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  --config src/state_estimation/config/state_fusion.yaml \
  --config src/state_estimation/config/state_fusion_extrinsics.yaml \
  --config src/state_estimation/config/pool_world.yaml \
  --checkpoint /absolute/path/to/frozen_ppo_dr.zip --duration 90 \
  --metadata-json '{"controller":"PPO-DR","setpoint_id":"P01_front_left_shallow","repeat":1,"target_controller_world":[1.0,-0.20,0.50,0.0]}'
```

roll/pitch 只计算：

```text
最大倾角
超过 guard 的时间
是否触发安全终止
```

## 9. T2：E9/E10

T2 第一版只做位于 `controller_world.y=-0.50 m` 的固定深度平面轨迹。当前坐标约定为 y 轴向上，
因此该值表示水下 0.50 m：

```text
straight
straight_pro / 慢速四段折线
circle
ellipse / 顶点降速椭圆
lemniscate / 8 字
figure_eight / 大范围 8 字
```

不做 helix，直到 depth/heave 的独立响应证据完成。

### 9.1 T2 仿真 E9

必须保持 Unity 和 ROS2 30D observation 一致：

```text
4 个 preview position：12D
desired velocity：3D
body linear velocity：3D
body angular velocity：3D
up vector：3D
路径切线方向特征（sin/cos）：2D（仅为既有 30D policy observation，不是 yaw 目标）
progress encoding：4D
```

仿真主接口为：

```text
30D observation
→ 6D wrench policy
→ 训练时的 physical allocator
→ canonical 8 thruster command
```

direct-8 policy 可以作为补充，但不能和 wrench6 结果直接当作同一 action interface。

### 9.2 T2 实机 E10

T2 使用唯一实验配置：

```text
ros2_ws/src/experiment_recorder/config/t2_hardware_experiment.yaml
```

当前实机 T2 注册两类 translation-only controller：

| `--controller` | 执行链                                                                                                           | 移动参考的处理                                   | 姿态/学习边界                                    |
| ---------------- | ---------------------------------------------------------------------------------------------------------------- | ------------------------------------------------ | ------------------------------------------------ |
| `PID_POSITION` | `pose20` 位置 PID $\rightarrow$ empirical mixer $\rightarrow$ 8 路 canonical force                         | 在每个 60 Hz 控制周期采样当前 position reference | 不控制 roll/pitch/yaw；无 checkpoint、无速度前馈 |
| `PPO_WRENCH6`  | 30D PPO$\rightarrow$ `[F_x,F_y,F_z]` $\rightarrow$ physical allocator $\rightarrow$ 8 路 canonical force | 既有 trajectory30 observation                    | 不控制 roll/pitch/yaw；checkpoint 必须冻结并审计 |

因此移动目标点**可以直接用 PID 跟踪**：controller 每个 tick 都将轨迹的当前位置当作新的
position setpoint。`PID_POSITION` 是一个明确的纯位置 PID baseline，并不从轨迹的
`(v_x,v_y,v_z)` 使用速度前馈；相对于速度前馈或 RL 策略，它可能有正常的相位滞后。其候选 YAML 为：

```text
ros2_ws/src/motion_control/config/FinsROV/traditional_pid_trajectory_tracking.yaml
```

该 profile 禁用 `use_target_orientation` 和 `traditional_enable_yaw_control`，以免
`MultiDOFJointTrajectory` 为 ROS transport 使用的 identity quaternion 被误解为 yaw=0 目标；
其 `stop_on_goal_reached=false`，不会因到达早期参考点而停止。`PPO_THRUSTER8` 仍为 disabled，
直到 checkpoint 和动作契约审计完成。runner 会在 ROS2 启动前检查 PID/PPO profile 与 YAML 的
backend、observation、姿态边界及受限推力命令契约；不一致时直接拒绝运行。

YAML 中定义固定水下参考深度 `y=-0.50 m`，以及独立的初始摆放深度 `y=-0.20 m`；还定义
straight/straight_pro/circle/ellipse/lemniscate/figure_eight 六条位置--速度参数化轨迹、采样频率、工作空间、速度/加速度上限、
3 次重复、reference 对齐指标及完整安全策略。circle 每次 trial 连续完成三圈；`ellipse` 单圈从 `(1.0,0.0)`
出发，半长轴/半短轴分别为 `1.0/0.4 m`，并在四个顶点处降低参考速度；`survey_signoff` 内容会随 run manifest 保存，用于记录水池 survey
和 pilot 信息，但它是审计元数据，不再阻塞 `--arm`；硬件运行仍必须显式提供 `--arm`。

每次 T2 CLI 运行只允许选择一种轨迹；该轨迹按 YAML 重复 3 次。因此先检查 ellipse
的 3-trial PPO-Wrench6 计划、轨迹起点、峰值速度和加速度：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PPO_WRENCH6 \
  --trajectory ellipse \
  --dry-run
```

`--trajectory` 是必填项，且一次只能给一个 YAML 中已启用的 ID：`straight`、`straight_pro`、
`circle`、`ellipse`、`lemniscate` 或 `figure_eight`。它不会修改 YAML；更换轨迹时必须重新执行一条 CLI 命令，从而生成独立的
run ID 和 manifest。例如改测 straight：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PPO_WRENCH6 \
  --trajectory straight \
  --dry-run
```

当前候选路径的初始摆放点与首个发送 reference（`controller_world`，顺序均为 `[x,y,z]`）为：

| 轨迹             | 初始摆放点（记录，不验证） | 首个 reference（发送给 controller） |
| ---------------- | -------------------------- | ----------------------------------- |
| `straight`     | `[-1.00, -0.20, -0.40]`  | `[-1.00, -0.50, -0.40]`           |
| `straight_pro` | `[-1.00, -0.20, -0.40]`  | `[-1.00, -0.50, -0.40]`           |
| `circle`       | `[0.50, -0.20, 0.00]`    | `[0.50, -0.50, 0.00]`             |
| `ellipse`      | `[1.00, -0.20, 0.00]`    | `[1.00, -0.50, 0.00]`             |
| `lemniscate`   | `[0.00, -0.20, 0.00]`    | `[0.00, -0.50, 0.00]`             |
| `figure_eight` | `[0.00, -0.20, 0.00]`    | `[0.00, -0.50, 0.00]`             |

初始摆放点的 `y=-0.20 m` 表示水下 0.20 m；首个 reference 的 `y=-0.50 m` 表示水下 0.50 m。
两者都由 YAML 写入每个 trial 的 manifest，不通过实测 pose gate 校验。

`straight_pro` 的 x--z 路径按顺序为
`(-1.0,-0.4) → (1.0,0.4) → (1.0,-0.4) → (-1.0,0.4) → (-1.0,-0.4)`。
四段分别采用 32、13、32、13 s 的平滑 profile：每个拐点参考速度为零，随后再沿下一段加速，
而不是在转角处直接切换速度向量。

`ellipse` 是独立于既有 `circle` 的单圈轨迹，不会覆盖 circle 的三圈、半径或时长设置。它在
`controller_world` 平面中满足
`x=1.0\cos\theta`、`z=0.4\sin\theta`，逆时针从 `(1.0,0.0)` 出发，依次经过
`(0.0,0.4)`、`(-1.0,0.0)` 和 `(0.0,-0.4)`，最终回到起点。生成器采用弧长感知的相位调度，而非
直接匀速推进 `\theta`：中段的局部速度倍率为 1.0，四个顶点为 0.50，从而在转弯处连续降速、但不
停在中途顶点。全程仍采用统一的 smoothstep 起停，所以第一个和最后一个顶点的瞬时 reference
velocity 为零。ellipse 的 `92.70 s` 时长和 `0.103083509 m/s` 峰值目标是独立冻结的；在当前的
circle 更新后，两者峰值恰好相同。它不会因 circle 的半径、圈数或时长变化而被 runner 隐式重定时，
如需重新规定两条路径的相对速度，应显式修改 ellipse 的时长/目标速度并重新执行 dry-run。

`figure_eight` 同样独立于既有的小型 `lemniscate`。它采用
`x=1.0\sin\theta`、`z=0.8\sin\theta\cos\theta`：从中央交叉点 `(0,0)` 出发，先经过正 x/正 z 的
半环，随后穿过中央交叉点完成负 x/负 z 的半环；水平面分别达到 `(+/-1.0,0)` 与
`z=+/-0.40 m`。其单圈 `117.085700 s` 时长使峰值实际参考速度为 `0.103083509 m/s`，与当前
ellipse 的峰值完全一致；两者均保留统一的 smoothstep 起停。该八字没有额外的顶点减速要求，
其速度变化只来自曲线切向量和统一起停 profile。

`--controller` 选择实验 YAML 中登记的接口契约；`--controller-config` 则像 T1 一样指定
本次实际启动的 motion-controller profile。对于 PPO，runner 会读取该 profile 的 `checkpoint_path`，
并检查 backend、30D observation、6D wrench 接口和禁用全部旋转 wrench 的约束。对于 PID，runner
检查 `traditional_pid_position`/`pose20`、无 yaw 控制和 bounded force 输出，但不要求 checkpoint。
以下命令显式使用当前 PPO T2 profile：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PPO_WRENCH6 \
  --controller-config src/motion_control/config/FinsROV/ppo_trajectory_tracking_wrench6.yaml \
  --trajectory straight \
  --dry-run
```

以下命令检查 PID baseline；T2 不支持 `--checkpoint` 或 `--checkpoint-manifest`，PID 也不需要 checkpoint：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PID_POSITION \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_trajectory_tracking.yaml \
  --trajectory ellipse \
  --dry-run
```

T2 checkpoint 不在实验 YAML 中重复声明，也不能用 CLI 覆盖；它完全跟随 controller profile。
PPO 额外记录 checkpoint 与同一 training-run 下 `metadata.json` 的 SHA-256，并将 profile 和
metadata 快照写入 run 的 `provenance/`；PID 则明确记录 checkpoint 为 `null`。每次 T2 run 的根目录名
也固定包含轨迹 ID：

```text
<timestamp>_<experiment_id>_<strategy>_<trajectory>/
```

例如 `..._E10-T2-HARDWARE_candidate_ppo_wrench6_trajectory30_ellipse/`。该 run 下的三个 trial 使用
`..._ellipse_R01`、`..._ellipse_R02`、`..._ellipse_R03`，因此即使单独复制 rosbag、图或 manifest，
也能从路径名看出其对应轨迹。

T2 runner 启动时会查询 ROS graph。若发现已经存在全局 `/motion_controller`，终端会输出
`[WARN]`，并在 `run_manifest.json` 中登记 `startup_warnings`；它不会自动杀掉该节点，也不会
阻塞本次启动。该情况通常意味着上一次中断的 controller 尚未退出，可能竞争
`/finsrov/thrusters_out`，应先停止旧 controller 后再执行 `--arm`。

PPO 运行前需核对 controller profile 的 checkpoint 及同一 training-run 下 `metadata.json`；runner
会把它们的 SHA-256 和 profile/metadata 快照写入 run。PID 运行前需完成低推力 pilot 并冻结使用的
PID YAML。随后由操作者预启动 bridge、AprilTag 和 fusion，并执行。
下例以 PPO 为例；将 controller/profile 替换为上方 `PID_POSITION` 即可运行 PID：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t2_experiment \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --controller PPO_WRENCH6 \
  --controller-config src/motion_control/config/FinsROV/ppo_trajectory_tracking_wrench6.yaml \
  --trajectory ellipse \
  --arm
```

每个 trial 自动执行：

```text
从 YAML 读取初始摆放点 `[x, y=-0.20 m, z]`，以及首个 reference
`[x, y=-0.50 m, z]`
确认至少收到一帧类型正确的 `/finsrov/controller/pose`（`controller_world` 状态流可用），
但不判断它是否位于首个 reference，也不执行自动预摆位
trajectory_start 事件 + 发送 controller_world MultiDOFJointTrajectory
straight/lemniscate 运行 30 s；straight_pro 按四段折线运行 90 s；ellipse 单圈运行 92.70 s；
figure_eight 单圈运行 117.085700 s；circle 连续转 3 圈并运行 137.142857 s
trajectory_end + cancel
结束后继续记录 7 s
停止本 trial recorder 与 controller；距下一 trial 至少等待 15 s
完整 run 退出时取消轨迹、下发零推力，但保持 bridge enabled；Ctrl-C/异常才禁用 bridge
```

运行中按一次 `Ctrl-C` 的语义是“在当前位置提前结束本 trial”。runner 会先记录
`operator_abort`、取消轨迹并下发零推力，然后最多等待
`runner.auto_launch.recorder_finalize_timeout_sec`（当前 90 s）让 rosbag flush 和自动分析完成；
该部分轨迹仍会生成 `derived/` 下的 CSV、指标和图，但 completion 会因
`operator_abort` 为 0。随后 bridge 按紧急路径禁用，run 的状态为 `interrupted`。请不要连续按第二次
`Ctrl-C`，否则操作系统仍可能在 recorder 完成分析前终止进程。

运行轨迹期间，runner 按 `runner.runtime_status.log_interval_sec`（当前 1 Hz）在终端输出
`trajectory`、重复编号、进度、`/finsrov/controller/pose` 的 `[x,y,z]` 和状态健康度。缺失或超过
`pose_stale_after_sec` 的 pose/status、未 ready/initialized、IMU/depth 不新鲜或不允许的视觉模式会
显示为 `status=ABNORMAL(...)`，并在状态改变时额外打印 `[WARN]`；恢复后打印恢复提示。这些是实时
操作者提示，正式完成判定仍以 rosbag 的冻结后处理协议为准。若本 trial 的 controller 或 recorder
进程异常退出，runner 会立即打印 `[ERROR]` 并进入既有的安全中止路径。

straight、straight_pro、circle、ellipse、lemniscate 和 figure_eight 的起点并不相同。当前 trajectory30 PPO backend 只接收测试
轨迹，不能也不会用一个未注册的 point-to-point/PID 控制器把艇自动移到起点；因此
`start_placement_mode` 固定为 `declared_initial_placement_no_pose_gate`。所有 YAML 轨迹首个
reference 均为 `[x,-0.50,z]`，初始摆放点均为 `[x,-0.20,z]`；二者写入
`MultiDOFJointTrajectory`/trial manifest。该消息使用 identity quaternion 作为 ROS Transform
运输字段，并不下发姿态目标。这是**参考轨迹和摆放声明**，不是对 ROV 实测起始位置或姿态的验证；
操作者仍须按水池安全规程摆放 ROV，但该摆放过程不参与 runner
的通过/失败判定。

每个 trial 结束后，runner 先停止 recorder（并触发自动分析）和该 trial 的 controller，再在
启动下一 trial 前等待至少 15 s。进程停止、自动分析或下一 controller 启动可能使实际墙钟间隔
更长，但绝不会短于 15 s；这一最小间隔的开始/结束时间写入 run-level
`run_manifest.json` 的 `inter_trial_intervals`。间隔和任何人工摆放时间均不计入对应轨迹的跟踪指标。

轨迹点由 runner 生成，均包含位置和速度。在当前固定 `y=-0.50 m` 的参考平面中，
`controller_world.y` 和 `vy` 固定为 YAML 注册值。controller 的既有 30D observation 仍从
reference `(vx,vz)` 导出一个路径方向特征，但 runner 不发送 yaw reference，分析器也不生成
yaw 跟踪误差。

每个 trial 的 rosbag 保留以下原始数据；自动分析只从这些原始时间戳重建指标，不手工导入
CSV 数值：

```text
状态与感知：/finsrov/controller/pose、/finsrov/pose、controller IMU/depth/DVL、
            controller/fusion/vision status、折射/纯 PnP AprilTag 输出
运行时参考：/motion_controller/debug/trajectory_reference
控制链路：  /motion_controller/command/trajectory、cancel、goal/status，
            observation、policy action、wrench6d、thruster command、/finsrov/thrusters_out
硬件审计：  MCU telemetry、raw IMU/depth/RPM、thruster command echo、hardware status
事件与来源：/finsrov/experiment/event、trial metadata、YAML/config/checkpoint SHA-256
```

主要指标：

- cross-track RMSE/p95；
- time-aligned position error；
- 完成率；
- 越界率；
- depth guard；
- roll/pitch guard；
- command effort；
- action smoothness；
- saturation 和 allocation residual。

其中 cross-track、time-aligned position/depth、x/y/z 分量误差、命令 effort/smoothness、
状态新鲜度和 completion 已由当前分析器生成。`thruster_saturation_fraction` 与
`allocation_residual` 的字段也会保留，但在 bridge echo 与 allocator 前向模型使用同一
时间基准前会明确写为 `null`，不能作为论文结果。

**T2 计分时间窗口。** T2 是轨迹跟踪而不是定点保持，因此 RMSE 不跳过起始瞬态。实机和非锁步
旧记录从 `trajectory_start` 事件开始；`ros2_control_lockstep` 仿真从第一条 controller-stamped
runtime reference 的精确 tick 开始（同时仍要求存在 `trajectory_start` 审计事件）。两者都按 YAML
冻结的 10 Hz 时间网格计算到完整声明时长结束（straight/lemniscate 为 30 s，四段 straight_pro 为
90 s，单圈 ellipse 为 92.70 s，单圈 figure_eight 为 117.085700 s，三圈 circle 为 137.142857 s）；若
发生 `trajectory_end`、安全终止或操作者终止，则在对应终止事件提前截断。参考和实测 pose
仅在同一网格点可由运行时 reference 与相同坐标语义的状态在不超过 0.25 s 的相邻采样间隔内
插值时计入；缺失点不补造，并降低 `reference_valid_fraction`。因此该完整路径时长覆盖“从实际
初始位置开始跟上路径”的全过程，也与当前不做起始 pose gate 的协议一致。实际采用的起止
时间、采样数和缺失情况均写入 `t2_metrics.json.evaluation_window`。

误差定义为 `actual-reference`：x、y/depth 和 z 为有符号米误差。物理融合输出 `/finsrov/pose`
使用 `pool_world=[x,y,z]`（z-up）；T2 reference 使用
`controller_world=[x,z-water_surface_z_m,y]`。分析器先按 trial metadata 中冻结的变换将实测 pose
转为 `controller_world`，再计算误差，绝不直接把 pool y/z 与 controller y/z 相比。每个受控自由度同时输出 RMSE、
MAE 和绝对误差 P95，保存在 `t2_metrics.json.per_dof_error`。roll/pitch/yaw 都不是 T2 跟踪目标；
原始 IMU 与 guard 状态会保留，但不会伪造其“跟踪误差”。

自动分析的 T2 专用输出位于每个 session 的 `derived/`：

```text
reference_aligned_samples.csv  # runtime reference、controller 坐标实测和 pool 坐标实测的时间对齐样本
t2_metrics.json / .csv        # cross-track、time-aligned、depth、completion
plots/t2/path_pool_xy.png                    # 非 circle 轨迹：物理水平面 pool x-y
plots/t2/circle_loop_01_path_pool_xy.png     # circle 第 1 圈的物理实测/参考对比
plots/t2/circle_loop_02_path_pool_xy.png     # circle 第 2 圈的物理实测/参考对比
plots/t2/circle_loop_03_path_pool_xy.png     # circle 第 3 圈的物理实测/参考对比
plots/t2/state_vs_reference_controller_xyz.png
plots/t2/tracking_error_controller_xyz.png   # controller x/y/z 有符号误差，零线为基线
plots/t2/tracking_error.png
```

非 circle 的 `path_pool_xy.png` 是潜器实际 `pool x-y` 水平轨迹与由 controller runtime reference
逆变换得到的物理参考轨迹叠加图。circle 不再把三圈叠在同一张路径图中；每个 trial 自动按参考相位拆成
`circle_loop_01/02/03_path_pool_xy.png` 三张图，每张只比较对应一圈；
`state_vs_reference_controller_xyz.png` 显示三个受控自由度在同一 `controller_world` 中的状态与目标基线；
`tracking_error_controller_xyz.png` 则单独显示三个受控自由度的误差随时间变化。每个 session 自动
生成一组以上图表，run 级目录另复制其索引，便于比较三个重复 trial。

run 级目录的 `resolved_trajectories/*.csv` 是正式发送给 controller 的 reference；每个
trial manifest 保存其 SHA-256。T2 completion 同时要求：reference progress、reference--pose
有效配对率、`/finsrov/controller/state/status` 的 `ready/initialized/imu_fresh/depth_fresh`
比例和允许的 `vision_mode`（默认 `fresh` 或 `coast`）均满足 YAML 中冻结的阈值，且没有
failure event。状态 JSON 没有 header 时，分析器只使用 reference 时刻之前且不超过
`status_max_age_sec` 的最近状态，避免用未来消息或 stale 状态填补数据。

## 10. E8：中心点人工扰动恢复（独立于正式 T1）

E8 不是 E6/E7 静水定点保持的附加 phase，也不使用八个 T1 target 或其
`hold_start+40--60 s` 统计窗口。每个 E8 trial 只控制唯一中心目标：

```text
controller_world = [x=0.0, y(depth)=-0.5, z=0.0, yaw=0.0 deg]
```

它评估的是**人工施加、以状态观测确认的扰动后恢复**，不测量也不推断外力大小，
因此论文中不能称作“相同外力”或“定量抗流”实验。协议、数据和分析结果均与静态 T1
完全分离：

```text
protocol: ros2_ws/src/experiment_recorder/config/t1_disturbance_recovery_hardware.yaml
raw data: ros2_ws/data/experiments/hardware/T1_disturbance/<controller>/<run_id>/
command:  run_t1_disturbance_experiment
```

若融合报告 `vision_mode:hold` 或 `vision_mode:lost`，E8 不会因该视觉状态单独中止；运动控制器自身会关闭这些状态下的推力输出，运行器持续打印状态并等待 `fresh`/`coast` 恢复。失定位期间不能进入“可以施加扰动”、不能确认扰动、也不能确认恢复。话题陈旧、IMU/深度异常或其他状态错误仍按安全策略中止。

### 10.1 固定流程

```text
发布中心目标
→ 连续控制至少 10 s（每秒打印位置、误差、速度和 health）
→ 若尚未稳定则继续控制，不能扰动
→ x/z 误差不超过 0.10 m、深度误差不超过 0.20 m、yaw 误差不超过 10 deg，连续满足 readiness 2 s 后打印“可施加一次扰动”（不以速度/yaw-rate 作起扰门限）
→ 操作员只施加一次短促推/拉/拉索释放
→ 自动确认 x 或 z 水平偏差持续越过 0.20 m
→ 检测到后至少继续控制 10 s；若要获得严格 settling time，则继续至连续稳定 5 s，最长 45 s
→ cancel goal、零推力、停止 recorder 并自动分析
```

没有检测到扰动时 controller 不会停止，会持续打印状态直到冻结的 90 s timeout；此 trial
写入 `disturbance_not_detected`，保留为无有效扰动的记录，不能悄悄删除。`Ctrl-C` 或任一
runner 异常均走硬件 bridge 的零推力/禁用安全路径。

检测只基于 controller-world 状态。对目标点，上述误差为：

```text
ex = x
edepth = y + 0.5
ez = z
eyaw = wrap(yaw)
```

只有 x 或 z 的绝对误差达到 `0.20 m`，并持续 `persistence_sec`（当前 `0.30 s`），且
AprilTag/fusion/controller health 为 `fresh` 或 `coast` 时，才发布 `disturbance_detected`。
深度、yaw、线速度和角速度均不参与扰动判定；持续时间只抑制单帧视觉跳变，它检测的是
**可观测水平偏离**，而非未知外力本身。

起扰前的 `WAIT_READY` 与恢复完成均只检查四个位置/姿态误差，不检查线速度或 yaw rate；速度仍被记录为诊断数据，但不参与 E8 的任何通过/完成判定。

### 10.2 启动

先按常规顺序启动 hardware bridge（首次建议 `enabled:=false`）、折射 AprilTag 和 state fusion；
检查状态新鲜，再使用 dry-run 核对冻结目标和 controller：

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_disturbance_experiment \
  --config src/experiment_recorder/config/t1_disturbance_recovery_hardware.yaml \
  --controller PID \
  --controller-config src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml \
  --dry-run
```

完成 dry-run、池体 clearance、deadman/e-stop 和单推进器安全检查后，才加 `--arm`。PPO 示例：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_t1_disturbance_experiment \
  --config src/experiment_recorder/config/t1_disturbance_recovery_hardware.yaml \
  --controller PPO \
  --controller-config src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator_maxforce.yaml \
  --arm
```

默认每次命令只执行一个人工扰动 trial。pilot 后若已冻结物理拉索位置、方向、阈值与重复数，
可显式使用 `--trials 5`。不要在已经产生正式数据后直接编辑 YAML；复制为新版本并重新
记录 config hash。

### 10.3 自动产物和解释

每个 E8 session 在 recorder 结束时自动生成：

```text
derived/disturbance_samples.csv
  # controller-world 状态、x/depth/z/yaw 误差、诊断性有限差分速度和归一化误差
derived/disturbance_metrics.csv
  # detection、峰值偏差、从峰值到恢复的时间、恢复确认、IAE
derived/disturbance_analysis_report.json
  # 事件/时间戳、冻结阈值、warnings；不把未知外力伪造成牛顿值
derived/plots/disturbance_recovery_xyzyaw.png
  # 检测前 5 s 及检测后的四自由度误差；紫色 t=0 线为 disturbance_detected，
  # 绿色带为恢复阈值，虚线标出峰值/恢复进入时刻
```

离线分析以 raw rosbag 的 `disturbance_detected` event 和 controller-world pose 为准。
最大偏差在检测后 `peak_search_sec` 内确定；恢复时间从该实测峰值起算。若未能连续满足
四个位置/姿态误差阈值 `required_stable_hold_sec`（当前 5 s），该 trial 记为未恢复，而不是删掉。

## 11. 结束和归档

停止 recorder 后，分析会自动执行并写入 `derived/analysis_report.json` 和
`derived/plots/`。只有在旧目录需要补跑、或补齐外部真值后才使用：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder analyze_experiment \
  ./ros2_ws/data/experiments/E6/20260822_T1_pid_001 \
  --force
```

`finalize_experiment` 只作为旧流程兼容命令保留。

只有满足以下条件才把结果复制到论文 artifacts：

```text
bag metadata 存在
manifest 完整
配置 hash 存在
checkpoint hash 存在（RL）
trial event 完整
外部真值/测力计文件已登记
没有手工修改 CSV 数值
```
