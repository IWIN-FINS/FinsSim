# FinsSim 手柄轨迹采集

`trajectory_data` 用于采集自由手柄操作轨迹。它不发布推进器命令，也不启动或使能硬件桥；运行时只启动一个 `ros2 bag record` 子进程和事件标记发布者。

本文的 topic、池体范围和配置文件是当前 FinsROV profile。包提供的 session、导出、对齐与质量检查流程可复用于其他 ROV；接入时应以新的 profile 替换这些 vehicle-specific 参数。

每次录制保存在 `ros2_ws/data/teleop_trajectories/<session_id>/`。目录包括：

- `rosbag/`：原始 sqlite3 rosbag2，所有话题保持其发布频率和消息 header 时间戳。
- `manifest.json`：话题清单、Git revision、手柄/bridge/fusion 配置的 SHA-256。
- `raw_streams/*.parquet`：逐话题解码后的原始频率数据，保留 `source_time_sec` 和 `bag_time_sec`。
- `trajectory.parquet`：按指定频率对齐后的训练表，包含每个输入的 source stamp、age 和 validity。
- `quality_report.json`、`jump_events.parquet`、`out_of_bounds_events.parquet`、`plots/`：采样率、候选跳变、遥测序号丢包、池边界统计和对比图。

`plots/trajectory_xyz.png` 展示融合 pose 的三个位置分量随时间变化，并标出池体范围；`plots/trajectory_3d_pool.png` 在 `x[-2,2]`、`y[-1,0]`、`z[-1,1]` 的半透明池体中绘制融合轨迹。越界点不会被裁掉，用于发现定位漂移或坐标系问题。

不录制图像或相机帧。只记录视觉节点已经输出的 6D pose 与状态，因此数据体积主要由 IMU、遥测和控制命令组成。

## 记录的话题

动作链路：

```text
/finsrov/teleop/body_wrench_cmd
/finsrov/thrusters_out
/finsrov/hardware/thruster_cmd_echo
/finsrov/hardware/motor_rpm_raw
```

原始硬件与融合状态：

```text
/finsrov/hardware/telemetry
/finsrov/hardware/imu_raw
/finsrov/hardware/depth_raw
/finsrov/pose
/finsrov/imu_link
/finsrov/depth_link
/finsrov/dvl_link
/finsrov/state/status
```

`/finsrov/hardware/imu_raw` 是 hardware bridge 完成安装位姿/坐标变换后发布给 fusion 的 IMU，不记录 MCU 变换前的裸字节。它与 `/finsrov/imu_link` 同时保留，故可以检查原始输入是否跳变、该跳变是否进入融合状态，或由 status 的 `reject_reason`、freshness 与协方差解释。

## 实船采集

先按常规顺序启动硬件 bridge、视觉/fusion 和手柄 teleop。确认状态健康、且 `/finsrov/thrusters_out` 只有手柄发布者后，再开录制终端：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run trajectory_data record_teleop_trajectory \
  --session-id 20260818_free_drive_001 \
  --label free_drive
```

已有同名 session 时，默认会拒绝启动以避免误删旧数据。确认需要重录同一个 ID 时，显式传入 `--overwrite`；它会先清空该 session 目录中的 bag、导出和分析产物，再开始新录制。`--overwrite` 只能删除输出根目录下的单个 session，`--dry-run` 不会删除任何文件：

```bash
./scripts/run_ros2_uv.sh ros2 run trajectory_data record_teleop_trajectory \
  --session-id 20260818_free_drive_001 --label free_drive --overwrite
```

用一次 `Ctrl-C` 停止。录制器会把 rosbag 放在独立进程组中，收到 Ctrl-C 后只由录制器向 rosbag 发送一次停止请求并等待其写完数据库和 metadata；不要额外对同一个终端发送第二次 Ctrl-C。录制器会发送 `session_start` 和 `session_stop` 事件。每段自由轨迹都可在另一个终端标记，不需要先定义 goal：

```bash
./scripts/run_ros2_uv.sh ros2 run trajectory_data mark_teleop_trajectory \
  --session-id 20260818_free_drive_001 \
  --event phase_start --label straight_surge

./scripts/run_ros2_uv.sh ros2 run trajectory_data mark_teleop_trajectory \
  --session-id 20260818_free_drive_001 \
  --event phase_end --label straight_surge
```

建议每段先静止约 3 秒，录制稳态、缓慢直行/横移/升沉/转向、组合运动、松杆滑行和停止。不同操作风格应分 session，异常撞击、断连和视觉丢失不删除，使用 `operator_mark` 标记，后续由质量字段筛选。

## 离线导出与质量检查

录制完成后再导出；这里才做统一频率对齐，默认 30 Hz：

```bash
cd ./ros2_ws
SESSION=data/teleop_trajectories/20260818_free_drive_001

./scripts/run_ros2_uv.sh ros2 run trajectory_data export_teleop_trajectory \
  "$SESSION" --rate 30

./scripts/run_ros2_uv.sh ros2 run trajectory_data analyze_teleop_trajectory \
  "$SESSION"
```

新录制不使用文件级压缩，优先保证一次 Ctrl-C 停止后 SQLite bag 可直接读取和恢复。对于旧的 `.db3.zstd` session，导出器会通过 ROS2 compression reader 自动解压读取，无需手动操作。若 session 中存在人工恢复的 `rosbag_recovered/`，导出器会自动优先使用，并在 `export_summary.json` 的 `bag_source` 标记为 `recovered`；也可通过 `--bag-dir` 明确指定来源。

连续信号（pose、IMU、速度）在相邻源样本之间线性插值，四元数使用 SLERP；手柄 wrench、推进器命令和 MCU echo 使用前向零阶保持，绝不引用未来动作。`trajectory.parquet` 中的 `source_stamp_*_sec` 与 `*_age_sec` 使每个对齐样本都可回溯到源数据，`synchronization_valid` 只在硬件遥测、原始 IMU、pose、DVL、动作和 fusion 的 IMU/depth freshness 同时满足要求时为真。

跳变检测只在 `raw_streams/` 原始频率数据上运行，不会把插值造成的平滑误判为传感器行为。它输出视觉 pose、融合 pose、原始/融合深度、原始/融合 IMU 角速度与加速度、融合速度的候选跳变，并在事件旁附加最近的 fusion 状态。`jump_events.parquet` 是筛查入口，最终判断仍应结合 `raw_streams`、`state/status` 和 bag 回放。

质量报告还检查 `pool_world` 池边界：`x` 在 `[-2, 2] m`、`z` 在 `[-1, 1] m`、`y` 只检查最深边界 `y >= -1 m`，不假设 y 上界。`pool_bounds` 给出融合 pose 与视觉 pose 的分轴越界次数、最大越界距离和连续越界片段数；`out_of_bounds_events.parquet` 可按时间戳定位每个越界样本。
