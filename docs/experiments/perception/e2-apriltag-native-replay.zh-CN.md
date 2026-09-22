# E2 AprilTag 定位消融：原生 ROS2 链路回放评估

本实验使用归档的水下底部 AprilTag 图像集，量化 `pool_world` 的水平
位置误差，并直接比较同一运行时姿态节点的两条输出：

- `pinhole`：`refractive_apriltag_pose_node` 发布的纯视觉 pinhole PnP 基线；
- `snell`：同一节点发布的平面水面 Snell 折射、深度和 IMU 姿态约束结果。

这是 **E2 的归档 PnP--Snell 消融**，不是在线 E1 的 FinsROV 固定点绝对定位记录，
也不使用旧的“按数据集真值重新拟合外参”做法。由于归档中没有同步 IMU、深度或
完整 state-fusion 输入，它也不是论文协议中完整的 E2 `PnP / Snell / fusion` 三方法比较。

## 运行时链路

```text
归档 images/raw/*.png
  -> direct_apriltag_node (C++ 原生 detector, replay_image_list_file)
  -> /apriltag_localization_eval/tag_detections_2d
  -> refractive_apriltag_pose_node (C++ 原生姿态节点)
  -> pinhole_pose / snell_pose
  -> recorder: 逐图像、逐物理真值点聚合、指标和图表
```

`T_world_camera` 直接来自
`state_estimation/config/state_fusion_extrinsics.yaml`。为让固定在池底的
历史 marker 可通过该节点的 `T_body_tag` 契约，runner 仅在结果目录创建
`config/virtual_bottom_tag_extrinsics.yaml`：它原样复制 `T_world_camera`，并为
archive 的 tag 1 设置 identity `T_body_tag`。这不是对相机外参的拟合或修改。

历史图像没有同步的 IMU、pressure 数据。runner 会在独立 namespace 发布明确声明的
oracle 输入：水面 `z=1 m`、池底 tag 平面 `z=0 m`、压力深度 `1 m`、水平 IMU。
因此此实验仅能支持 **Snell + 已知深度/水平姿态约束下的 x/y 几何精度**，不支持
end-to-end EKF、深度、姿态或连续相机流可用性结论。

归档 `samples.csv` 的 `truth_x_m/y_m` 是 AprilTag **外白边左下角**的采集网格坐标；它既不是
当前相机外参使用的 `pool_world` 原点，也不是原生 PnP/Snell 输出的 tag 中心。配置固定记录
完整的物理转换：`pool_x = truth_x - 2.0 + 0.06`、`pool_y = truth_y - 1.0 + 0.06`。
其中 `0.06 m` 是 12 cm 外框的一半；Unity 网格的 `(x,z)` 分别对应 ROS `pool_world` 的
水平 `(x,y)`。逐图像 CSV 同时保留 annotation 坐标和转换后的中心真值，不能在分析时临时改原点。

E2 会明确覆写两个原生节点的 `marker_length_m=0.098`（归档 tag 黑框边长），不使用当前
RGB 实机配置中为新 ROV tag 设置的 `0.093 m`。它还显式传入全数据集拟合的
`calibration/pool_homography.yaml`。该 homography 仅用于 native detector 的二维中心像素诊断映射；
本实验报告的 PnP/Snell 位姿误差由相机内参/畸变、`T_world_camera` 和折射模型计算，因此不能把
homography 拟合误差误写成 PnP 或 Snell 的定位精度。

## 启动

先构建改动过的原生 detector 和 recorder：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select perception experiment_recorder
```

先用少量图像验证链路；此运行不作为论文结果：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_apriltag_localization_evaluation \
  --config src/experiment_recorder/config/e2_apriltag_localization_dataset.yaml \
  --session-id smoke_20 \
  --limit 20
```

完整重放：

```bash
./scripts/run_ros2_uv.sh ros2 run experiment_recorder run_apriltag_localization_evaluation \
  --config src/experiment_recorder/config/e2_apriltag_localization_dataset.yaml
```

默认 2 Hz 重放完整的 720 张归档图像，预计约 6--8 分钟；具体进度会输出
`detector statuses`、`pinhole` 和 `snell` 消息数量。运行中不启动 hardware bridge、
camera 或 state-fusion EKF，所有话题位于 `/apriltag_localization_eval/...`。回放节点还会被
重命名为 `e2_direct_apriltag_node` 与 `e2_refractive_apriltag_pose_node`，因此不会与正在运行的
实机感知节点发生 ROS 节点名冲突。

## 输出

每次运行产生一个新目录：

```text
ros2_ws/data/experiments/E2/<timestamp>_E2_NativeReplay/
  input/                         # 重放列表和不可变 source samples.csv 副本
  config/                        # 实际传入两条 C++ 节点的参数和虚拟 tag 外参
  logs/                          # 两个 C++ 节点 stdout/stderr
  raw/ros_events.ndjson          # detector、pose、status、replay 审计事件
  derived/frame_predictions.csv  # 每张图像的 native output 与 x/y 误差
  derived/point_predictions.csv  # 同一物理真值点的重复图像均值
  derived/summary_metrics.csv    # frame 与 truth-position-mean 指标
  derived/analysis_report.json   # 哈希、链路、约束、范围和结果
  derived/plots/                 # CDF、热图、误差向量、可用性图
```

论文的主指标应取 `truth_position_mean`：先对同一测量位置的多帧结果求均值，再在物理
位置间计算 horizontal RMSE、平均误差和 P95。`saved_image_frame` 仅作为辅助诊断指标。
归档样本是保存后的图像，不能把其中的成功比例称为实时相机检测率。
