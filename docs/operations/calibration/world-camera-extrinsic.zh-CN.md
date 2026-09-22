# T_world_camera 外参标定

`T_world_camera` 是固定相机到水池世界坐标 `pool_world` 的外参：

```text
T_world_camera: camera frame -> pool_world
p_world = T_world_camera * p_camera
```

完整工具文档见：

```text
ros2_ws/src/state_estimation/docs/world_camera_extrinsic_calibration.zh-CN.md
```

## 1. 它解决什么问题

AprilTag PnP 首先得到的是：

```text
T_camera_tag
```

也就是 tag 在相机坐标系下的位置和姿态。状态融合需要的是潜器在水池世界系下的 pose，因此需要：

```text
T_world_body = T_world_camera * T_camera_tag * inverse(T_body_tag)
```

其中：

- `T_world_camera`：固定相机外参，本步骤标定。
- `T_camera_tag`：AprilTag PnP 输出。
- `T_body_tag`：tag 安装在潜器上的外参，人工测量，见 [tag-body-extrinsic.zh-CN.md](tag-body-extrinsic.zh-CN.md)。

## 2. 两种采集方案

### 方案 A：多个固定 tag

把多个 AprilTag 固定在水池中已知世界坐标的位置，同时采集。适合已有标定板或固定参考点的场景。

配置文件示例：

```text
ros2_ws/src/state_estimation/config/world_camera_calibration_tags.yaml
```

### 方案 B：单个 tag 移动采点

一次只摆一个 AprilTag 到已知世界坐标点，采集一批样本；然后移动到下一点继续采集。

这是当前更推荐的现场流程，因为点位覆盖更自由。即使画面里有多个 tag，也可以通过 `--tag-id` 指定本次只使用哪一个。

## 3. 采集和求解命令

启动 AprilTag PnP/native 视觉节点后，查看工具帮助：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera --help
```

单点采集示例：

```bash
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz 0.842 -0.540 0.000 \
  --world-rpy-deg 0.0 0.0 0.0 \
  --point-name p01 \
  --count 40
```

求解：

```bash
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera solve \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --output ./tmp/finsrov_T_world_camera.yaml \
  --update-extrinsics src/state_estimation/config/state_fusion_extrinsics.yaml \
  --ransac
```

完整参数、RANSAC、残差分组排查见包内工具文档。

## 4. 产物写入位置

最终写入：

```text
ros2_ws/src/state_estimation/config/state_fusion_extrinsics.yaml
```

只更新：

```yaml
T_world_camera:
```

不要在这个步骤顺手修改 `T_body_tag`，除非你正在做 tag 安装外参标定。

## 5. 质量判断

经验值：

```text
translation_rmse_m < 0.03 m   比较好
translation_rmse_m 0.03~0.08  可先调试，但要谨慎
translation_rmse_m > 0.08 m   通常说明点位、tag 尺寸、内参或 PnP 有问题
```

如果 RANSAC 剔除整组点，优先检查该 `point_name` 的 `--world-xyz`、`--world-rpy-deg` 和现场摆放，而不是盲目放宽阈值。
