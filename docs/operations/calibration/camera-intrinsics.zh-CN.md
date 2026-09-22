# 相机设备与内参标定

本文只说明“标定顺序、产物和应该写到哪里”。具体脚本参数见包内文档：

```text
ros2_ws/src/perception/docs/rgb_camera_apriltag_calibration.zh-CN.md
```

## 1. 先固定运行配置

相机内参必须和运行时的成像配置一致，至少要固定：

```text
device
width
height
fps
fourcc
focus/exposure/IR illumination
```

优先使用稳定设备路径：

```text
/dev/v4l/by-id/...
```

不要依赖 `/dev/video0`、`/dev/video1` 这类编号；插拔 USB 后编号可能变化。

检查设备：

```bash
v4l2-ctl --list-devices
v4l2-ctl -d /dev/video0 --list-formats-ext
```

## 2. 内参标定产物

相机内参文件至少包含：

```text
image_width
image_height
camera_matrix: fx, fy, cx, cy
distortion_coefficients: k1, k2, p1, p2, k3
```

推荐保存位置：

```text
ros2_ws/src/perception/calibration/rgb_camera.yaml
ros2_ws/src/perception/calibration/ir_apriltag_camera.yaml
```

native C++ AprilTag / refractive 节点的配置文件应引用同一组内参，不要复制出多个互相漂移的数值。

## 3. 棋盘格流程

采集棋盘格：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run perception capture_chessboard_images --help
```

求解内参：

```bash
./scripts/run_ros2_uv.sh ros2 run perception calibrate_camera_chessboard --help
```

更完整的采图要求、棋盘格尺寸、质量判断见：

```text
ros2_ws/src/perception/docs/rgb_camera_apriltag_calibration.zh-CN.md
```

## 4. 什么时候必须重新标定

以下情况应重新标定或确认内参是否可按比例缩放：

- 更换相机。
- 更换镜头或焦距。
- 更换运行分辨率，且没有按比例正确缩放内参。
- 改变对焦位置。
- 图像裁剪、缩放、去畸变流程发生变化。
- 红外相机和 RGB 相机互换。

## 5. 验收标准

最低要求：

```text
重投影误差稳定且足够小
角点覆盖画面中心和边缘
标定分辨率和运行分辨率一致
AprilTag PnP 的 reprojection_error_px 不长期偏大
```

实船融合验收继续看：

```text
docs/calibration/validation-checklist.zh-CN.md
```
