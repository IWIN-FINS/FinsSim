# AprilTag 感知稳定性记录器

`record_apriltag_stability` 是独立、只订阅的 ROS2 节点，用于记录生产链路中
AprilTag detector、Snell/折射约束位姿和 state fusion 的运行时可用性。它不改动
`run_t1_experiment` 或 `run_t2_experiment`，不启动或停止 controller、hardware bridge、perception/fusion，
也绝不发布 `/finsrov/thrusters_out`、goal 或参数服务请求。因此可以在正在运行的 T1
试验旁路启动；停止记录器不会停止 T1。

## 记录层级

| 层级 | 话题 | 统计定义 |
|---|---|---|
| 基础 detector | `/finsrov/vision/status` | 每帧 `detected=true` 比例与连续 `false` 时段 |
| 折射约束输出 | `/finsrov/vision/refracted/status` | `constrained_valid=true`；另保留严格 `snell_valid=true` 与 fallback |
| 融合 | `/finsrov/state/status` | `vision_fresh=true`、`ready=true` 与连续不可用时段 |
| 输出轨迹 | Snell、pinhole、fusion pose | 水平 `x,y` 时间线与相邻输出增量（不是绝对精度） |

可用率是状态样本比例；自由漂移的一次记录或 event 划分出的 T1 session/phase 才是统计单元，不能把帧当成独立实验重复。该实验没有外部真值，不能替代 E2 的定位精度实验。

## A. 无推力自由漂移，60 s

先正常启动相机、折射节点和 state fusion。保持 bridge 禁用或无控制器输出，ROV 在水面自由漂移。随后启动：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_apriltag_stability \
  --config src/experiment_recorder/config/apriltag_stability.yaml \
  --mode free_drift --duration-sec 60 --label surface_unforced
```

60 秒后自动结束、分析和出图。若提前 Ctrl-C，记录器仍会写出已有数据并自动生成报告；它不会向 T1/bridge 发送任何信号。

## B. 与 T1 同时记录

先在**终端 A**启动记录器，它会等待 T1 的 `TrajectoryEvent`：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_apriltag_stability \
  --config src/experiment_recorder/config/apriltag_stability.yaml \
  --mode t1 --label t1_perception_availability
```

再在**终端 B**按现有流程启动 `run_t1_experiment`。不指定 `--linked-t1-session` 时，记录器会自动收集一组 T1 的全部八个目标点 session。对每一个点，只保留从 `hold_start`（controller 已收到目标并开始控制）到 `phase_end` 的 detector/Snell/fusion/pose 样本；`trial_start` 到 `hold_start` 的采集/复位段，以及 `phase_end` 到下一目标点的间隔均不会写入观测数据。四类 event marker 仍会保存，用于审计边界。60 秒 hold 再登记为 `hold_transient`（0--40 s）和 `late_hold`（40--60 s），但不猜测何时到达目标点。

T1 全部结束后，在终端 A 按 Ctrl-C。该 Ctrl-C 只终止记录器本身，随后自动分析；它不会取消 T1 goal 或关闭使能。

若只观察一个已知 `TrajectoryEvent.session_id`，可增加：

```bash
  --linked-t1-session <session_id>
```

## C. 与 T2 同时记录

先在独立终端启动：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run experiment_recorder record_apriltag_stability \
  --config src/experiment_recorder/config/apriltag_stability.yaml \
  --mode t2 --label t2_perception_availability
```

再按既有方式运行一次 `run_t2_experiment`。记录器对每个 repeat 独立识别 T2 的
`trajectory_start` 和 `trajectory_end`：只保存这段真实轨迹跟踪时间内的感知稳定性样本。
`trial_start` 前的准备段、`trajectory_end` 后的 7 秒 post-record 段，以及两个 repeat
之间 15 秒的复位间隔均不会写入观测数据；相应 event marker 仍会保存以审计边界。

## 输出

```text
ros2_ws/data/experiments/E1/<timestamp>_E1-APRILTAG-AVAILABILITY_Stability/
  manifest.json
  resolved_config.yaml
  raw/stream.ndjson
  derived/phase_summary.csv
  derived/dropout_intervals.csv
  derived/analysis_report.json
  derived/plots/
    availability_by_phase.png
    dropout_duration_histogram.png
    horizontal_pose_timeline.png
```

`phase_summary.csv` 记录 detector、约束输出、严格 Snell 物理解和 fusion 的可用率/速率，以及相邻水平输出增量。T1 会按每个 hold，T2 会按每条轨迹 repeat 汇总。`dropout_intervals.csv` 列出每层连续 false 的开始、结束和长度。水平时间线是 `pool_world` 输出，不是真实误差图。

## 开始前检查

```bash
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/vision/status
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/vision/refracted/status
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/state/status
```

若 `/finsrov/vision/refracted/status` 未发布，报告会明确显示折射层无样本；不要把它解释为零可用率。
