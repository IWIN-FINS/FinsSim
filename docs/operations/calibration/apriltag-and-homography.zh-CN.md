# AprilTag 参数、ID 与 Homography

本文统一说明 AprilTag 基础参数和旧 2D homography 标定。当前实船 6D 主链路不再依赖 homography 作为位置主源，但 homography 仍可用于 GUI/debug 对照。

包内详细流程见：

```text
ros2_ws/src/perception/docs/rgb_camera_apriltag_calibration.zh-CN.md
ros2_ws/src/perception/README.zh-CN.md
```

## 1. 必须确认的 AprilTag 参数

每个现场 tag 至少要确认：

```text
tag_family
tag_id
marker_length_m
tag frame 方向
```

`marker_length_m` 是 AprilTag 黑白编码区域边长，不是纸张外框，也不是防水膜外框。

## 2. 扫描 tag family 和 id

不要靠猜。先使用扫描工具：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run perception scan_fiducial_tags --help
```

示例：

```bash
./scripts/run_ros2_uv.sh ros2 run perception scan_fiducial_tags \
  --device /dev/v4l/by-id/usb-DSJ_USB_Camera_200901010001-video-index0 \
  --width 1280 \
  --height 720 \
  --fps 30 \
  --fourcc MJPG
```

如果输出：

```text
detected: DICT_APRILTAG_36H11: ids=[3]
```

对应配置就是：

```yaml
tag_family: DICT_APRILTAG_36H11
target_tag_id: 3
```

## 3. tag frame 约定

当前代码中的 AprilTag 局部坐标系：

```text
tag origin: tag 正方形中心
tag +x: 从 tag 左边指向右边
tag +y: 从 tag 下边指向上边
tag +z: 按右手系 x cross y 得到，垂直 tag 平面
```

从 tag 正面看：

```text
        +y
         ^
         |
  p0 ----+---- p1
   |     |     |
   |   origin  |  -> +x
   |     |     |
  p3 ----+---- p2
```

角点顺序：

```text
p0 = (-half, +half, 0)  左上
p1 = (+half, +half, 0)  右上
p2 = (+half, -half, 0)  右下
p3 = (-half, -half, 0)  左下
```

`T_world_tag`、`T_body_tag`、`world-rpy-deg` 都按这个 tag frame 理解。

## 4. 旧 Homography 标定

旧 Python `apriltag_tracker` 中的 `world_xy` debug 字段使用：

```text
tag 中心像素 [u, v] -> pool_homography.yaml -> 水池平面 [x, y]
```

标定文件：

```text
ros2_ws/src/perception/calibration/pool_homography.yaml
```

这个流程适合固定俯视相机 + 单平面近似，主要用于调试，不作为当前 6D 状态融合主输入。

采集工具：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run perception collect_homography_points --help
./scripts/run_ros2_uv.sh ros2 run perception evaluate_homography_yaml \
  src/perception/calibration/pool_homography.yaml
```

更详细按键和点位要求见包内文档：

```text
ros2_ws/src/perception/docs/rgb_camera_apriltag_calibration.zh-CN.md
```

## 5. 检测不到 tag 时先查

按这个顺序排：

```text
1. tag 是否完整入镜
2. tag family/id 是否配置错误
3. marker_length_m 是否离谱
4. 图像是否过暗、过曝、失焦、运动模糊
5. 红外相机下打印材料是否对比度不足
6. tag 白边是否被裁掉
7. native/Python 节点是否使用了错误相机
```
