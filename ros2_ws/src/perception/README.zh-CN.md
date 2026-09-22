# FinsROV 感知与 AprilTag 定位

`perception` 是 FinsROV 唯一的 ROS2 感知包。实时定位链路使用 C++ 实现；Python 仅用于现场监视、相机标定和离线校验，不进入检测关键路径。

```text
V4L2 相机
  -> direct_apriltag_node (C++ / AprilTag 3)
  -> /finsrov/vision/tag_detections_2d
  -> refractive_apriltag_pose_node (C++) + IMU + depth
  -> /finsrov/vision/refracted_pose_6d
  -> state_estimation

perception_monitor (Python)
  <- /finsrov/camera/debug/compressed 和视觉/融合状态
```

检测节点不发布高频 `/finsrov/camera/image_raw`。monitor 只订阅低频 JPEG 调试图，因而不会阻塞相机读取和 AprilTag 检测。

## 实机启动

RGB 相机的标准折射定位链路：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch perception refractive_apriltag.launch.py
```

红外相机：

```bash
./scripts/run_ros2_uv.sh ros2 launch perception refractive_apriltag_ir.launch.py
```

只运行 C++ 检测/PnP debug：

```bash
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag.launch.py
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag_ir.launch.py
```

现场监视 GUI 单独启动：

```bash
./scripts/run_ros2_uv.sh ros2 run perception perception_monitor
```

## 配置与输出

- `config/direct_apriltag_rgb.yaml`、`config/direct_apriltag_ir.yaml`：相机、AprilTag detector、PnP 和 debug 图参数。
- `config/refractive_apriltag_rgb.yaml`、`config/refractive_apriltag_ir.yaml`：Snell 折射、depth/IMU 约束与多 Tag 刚体参数。
- `calibration/`：相机内参、畸变与水池平面标定的唯一权威副本。

实时主输出为 `/finsrov/vision/refracted_pose_6d`：水下时使用 AprilTag 角点、Snell 折射、depth 与 IMU 的混合约束；空气或水面过渡状态使用安全的 pinhole 输出。`/finsrov/vision/refracted_pose_6d_pure` 始终是普通 pinhole multi-tag PnP baseline，仅供对照，不是纯视觉 Snell 求解。

完整模型、坐标系与拒绝条件见 [折射 AprilTag 算法说明](docs/refractive_apriltag_pose_algorithm.zh-CN.md)。相机标定和辅助工具见 [标定工具说明](docs/rgb_camera_apriltag_calibration.zh-CN.md)。

## Python 工具

以下工具继续由 `ros2 run perception` 提供：

```text
preview_stereo_udp              record_stereo_udp_video
capture_chessboard_images       capture_stereo_chessboard_images
calibrate_camera_chessboard     calibrate_stereo_chessboard
collect_homography_points       evaluate_homography_yaml
scan_fiducial_tags
```

旧 Python 在线节点 `camera_capture`、`apriltag_tracker`、`direct_perception` 及其 launch 已删除；它们不能与 C++ detector 混用。
