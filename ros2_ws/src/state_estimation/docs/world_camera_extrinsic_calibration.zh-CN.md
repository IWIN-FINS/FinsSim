# T_world_camera 外参标定工具说明

这个工具用于标定固定相机到水池世界系的外参：

```text
T_world_camera
```

标定总入口见：

```text
docs/calibration/index.zh-CN.md
docs/calibration/world-camera-extrinsic.zh-CN.md
```

状态融合节点使用的完整链路是：

```text
T_world_body = T_world_camera * T_camera_tag * inverse(T_body_tag)
```

本文使用 `T_A_B` 表示：

```text
把 B 坐标系下的点变换到 A 坐标系
p_A = T_A_B * p_B
```

因此链路中的三个量分别是：

```text
T_world_camera: camera 坐标系 -> pool_world 坐标系
T_camera_tag: tag 坐标系 -> camera 坐标系，由 AprilTag PnP 输出
T_body_tag: tag 坐标系 -> body 坐标系，由潜器上 tag 安装外参提供
inverse(T_body_tag): body 坐标系 -> tag 坐标系，状态融合内部使用
```

注意：潜器上每个 tag 到 body 的安装外参不是本工具求解的内容。这个工具只求固定相机外参 `T_world_camera`。潜器上每个 tag 到 body 的安装外参需要单独测量并写入 `state_fusion_extrinsics.yaml`。

当前 `state_fusion_extrinsics.yaml` 只应填写现场测量方向的 `T_body_tag`。旧字段 `T_tag_body` 不再作为现场配置入口使用；如果需要理解历史内部方向，请看专门文档。

`T_body_tag` 的详细定义和人工测量方法见：

```text
ros2_ws/src/state_estimation/docs/t_body_tag_extrinsic.zh-CN.md
```

本文档中的 `world` 指 `pool_world`，约定为右手系、`z` 轴向上，**原点在水池中心底部**。水面高度用参数 `water_surface_z_m` 表示，默认 1.0 m。水压计深度 `depth_m` 是从水面向下为正，且当前水压计位于 body 原点下方 0.0678 m，因此状态融合节点默认使用：

```text
pressure_sensor_z = water_surface_z_m - depth_m
pressure_sensor_offset_world_z = pressure_sensor_offset_z_body * cos(roll) * cos(pitch)
body_z = pressure_sensor_z - pressure_sensor_offset_world_z
```

也就是 `water_surface_z_m=1.0`、水平姿态、水压计深度为 1.0 m 时，水压计位于池底 `z=0`，body 原点位于水压计上方 0.0678 m，因此 `/finsrov/pose.pose.pose.position.z = 0.0678`。Unity/训练模型的 `y-up` 坐标转换不在状态估计层完成，而是在 `motion_controller` 的 `ros_to_controller_basis_*` 边界完成。

其中 `T_camera_tag` 来自 AprilTag PnP topic：

```text
/finsrov/vision/tag_poses_3d_camera
```

## 1. 两种采集方案

这个工具支持两种外参采集流程：

```text
方案 A: 多个 AprilTag 固定在水池已知位置，同时采集
方案 B: 只移动一个 AprilTag，每次放到一个已知世界坐标点，逐点采集
```

方案 B 更适合现场没有多块 tag 板、或者想用同一张 tag 覆盖更多水池区域的情况。即使摄像头画面里同时出现多个 tag，`capture-point` 也只会使用 `--tag-id` 指定的那一个，并在终端打印：

```text
seen_ids=[2, 3, 4]; using tag_id=4
```

这行表示当前画面里检测到了 2、3、4，但本次采样只使用 tag 4。

## 2. 准备已知世界坐标的 tag

### 方案 A：多个固定 tag

在水池中固定几个 AprilTag，测量它们在 `pool_world` 下的位姿，写成一个 YAML，例如：

```yaml
T_world_tag:
  2:
    translation_xyz: [0.50, 0.20, 0.00]
    rotation_rpy_deg: [0.0, 0.0, 0.0]
  3:
    translation_xyz: [1.20, 0.20, 0.00]
    rotation_rpy_deg: [0.0, 0.0, 0.0]
  4:
    translation_xyz: [0.50, 0.90, 0.00]
    rotation_rpy_deg: [0.0, 0.0, 0.0]
```

推荐保存为：

```text
ros2_ws/src/state_estimation/config/world_camera_calibration_tags.yaml
```

字段含义：

```text
translation_xyz: tag 原点在 pool_world 下的位置，单位 m
rotation_rpy_deg: tag frame 相对 pool_world 的 roll/pitch/yaw，单位 deg
```

这里的矩阵语义是：

```text
T_world_tag
p_world = T_world_tag * p_tag
```

所以 `rotation_rpy_deg` 描述的是 tag 自己的局部坐标轴在 `pool_world` 下的方向，不是相机姿态，也不是潜器姿态。

欧拉角顺序为：

```text
R_world_tag = Rz(yaw) * Ry(pitch) * Rx(roll)
```

`pool_world` 使用 z-up 右手系：

```text
x: 水池世界 x 正方向
y: 水池世界 y 正方向
z: 向上，z=0 为水池中心底部平面
```

tag frame 在当前代码里的约定是：

```text
tag origin: tag 中心
tag +x: 从 tag 左边指向右边
tag +y: 从 tag 下边指向上边
tag +z: 按右手系 x cross y 得到，垂直 tag 平面
corner order: 使用 AprilTag detector canonical order，不按图像左上角重排
```

当前 native AprilTag PnP 和 `/finsrov/vision/tag_detections_2d` 使用同一套 canonical corner order：

```text
0: (-x, +y)
1: (+x, +y)
2: (+x, -y)
3: (-x, -y)
```

如果你之前用旧版本节点采集过 samples，旧版本曾按图像左上角重排角点，会导致 `T_camera_tag` 的 yaw 不代表真实 tag frame。升级后需要重新采集用于 6D full-pose 标定的 samples；旧 samples 只能作为 `--point-cloud` 位置标定参考。

如果 tag 平放在水底，图案正面朝上，通常 `roll=0`、`pitch=0`，只需要根据贴纸在水池平面内的朝向填写 `yaw`：

```text
tag +x 指向 pool_world +x:  --world-rpy-deg 0 0 0
tag +x 指向 pool_world +y:  --world-rpy-deg 0 0 90
tag +x 指向 pool_world -x:  --world-rpy-deg 0 0 180
tag +x 指向 pool_world -y:  --world-rpy-deg 0 0 -90
```

注意：如果求解时使用 `--point-cloud`，求解只使用 tag 中心点的 3D 对应关系：

```text
world_tag_position ~= T_world_camera * camera_tag_position
```

这种模式会忽略 `world-rpy-deg`，因此更适合“只知道 tag 中心位置，不完全确定 tag 局部坐标轴方向”的现场标定。只有不加 `--point-cloud`、使用完整 6D 模式时，`world-rpy-deg` 才必须准确。

也可以直接写四元数：

```yaml
rotation_xyzw: [0.0, 0.0, 0.0, 1.0]
```

或者写 4x4 矩阵：

```yaml
matrix:
  - 1.0
  - 0.0
  - 0.0
  - 0.50
  - 0.0
  - 1.0
  - 0.0
  - 0.20
  - 0.0
  - 0.0
  - 1.0
  - 0.00
  - 0.0
  - 0.0
  - 0.0
  - 1.0
```

### 方案 B：单个 tag 移动采点

这种方案不需要提前写 `world-tags` YAML。每摆一个点，就在命令行里直接告诉工具当前 tag 的世界坐标。

比如你使用 tag 4，把它放在水池世界坐标：

```text
x = 0.50 m
y = 0.20 m
z = 0.00 m  # tag 中心在水池底部平面；如果在水面上，z 应填 water_surface_z_m
yaw = 0 deg
```

采集这个点时直接运行：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 4 \
  --world-xyz 0.50 0.20 0.00 \
  --world-rpy-deg 0.0 0.0 -90.0 \
  --point-name p01 \
  --count 40
```

采完后，把同一张 AprilTag 移到下一个点，例如：

```text
x = 1.20 m
y = 0.20 m
z = 0.00 m  # 水池底部平面
yaw = 0 deg
```

再运行：

```bash
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz 0.842 -0.54 0.00 \
  --world-rpy-deg 0.0 0.0 0.0 \
  --point-name p01 \
  --count 40

./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz 0.65 -0.275 0.00 \
  --world-rpy-deg 0.0 0.0 90.0 \
  --point-name p02 \
  --count 40

./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz -1.08 -0.626 0.00 \
  --world-rpy-deg 0.0 0.0 180.0 \
  --point-name p03 \
  --count 40

./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz 1.19 -0.711 0.00 \
  --world-rpy-deg 0.0 0.0 -90.0 \
  --point-name p04 \
  --count 40

cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz 0.60 -0.6 1.055 \
  --world-rpy-deg 0.0 0.0 -90.0 \
  --point-name p01 \
  --count 40

cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz -0.6 -0.2 1.055 \
  --world-rpy-deg 0.0 0.0 -90.0 \
  --point-name p02 \
  --count 40

./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz -0.8 -0.6 1.055 \
  --world-rpy-deg 0.0 0.0 -90.0 \
  --point-name p03 \
  --count 40

./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture-point \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --tag-id 1 \
  --world-xyz 1.2 -0.2 1.055 \
  --world-rpy-deg 0.0 0.0 -90.0 \
  --point-name p04 \
  --count 40
```

`capture-point` 默认会追加到同一个 samples 文件。只有想从头开始清空样本时才加：

```bash
--overwrite
```

建议至少采集：

```text
4 个以上不共线点
每个点 30 到 60 帧
点位覆盖相机视野的中心和四周
```

如果 tag 在水池平面上旋转了，需要把 yaw 写对：

```bash
--world-rpy-deg 0.0 0.0 90.0
```

如果 tag 不是水平贴在水池平面，而是有 roll/pitch，也要把 `world-rpy-deg` 写成真实姿态。PnP 6D 外参标定对 tag 姿态比较敏感，不能只填位置不管姿态。

## 3. 启动 AprilTag PnP

先启动 native AprilTag 节点，确保它能发布 PnP 检测：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag.launch.py
```

检查是否有数据：

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/tag_poses_3d_camera --once
```

如果没有数据，先检查相机、tag family、tag id、相机内参和 `marker_length_m`。

## 4. 采集外参样本

### 方案 A：多个固定 tag 同时采集

运行：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera capture \
  --world-tags src/state_estimation/config/world_camera_calibration_tags.yaml \
  --samples /tmp/finsrov_world_camera_samples.yaml \
  --count 120 \
  --max-reprojection-error-px 4.0
```

工具会订阅：

```text
/finsrov/vision/tag_poses_3d_camera
```

并把已知 tag 的 `T_camera_tag` 和对应 `T_world_tag` 写入：

```text
/tmp/finsrov_world_camera_samples.yaml
```

建议：

```text
至少使用 3 个不共线 tag
tag 覆盖相机画面的不同区域
每个 tag 采集几十帧
reprojection_error_px 越小越好，通常建议小于 4 px
```

### 方案 B：单个 tag 逐点移动采集

按前面“方案 B”的方式，每放一个点运行一次 `capture-point`。终端输出类似：

```text
mode: movable single tag point; using tag_id=4
world_xyz: [0.5, 0.2, 0.0]
world_rpy_deg: [0.0, 0.0, 0.0]
existing_samples: 80; overwrite=False
point_sample 1/40: seen_ids=[2, 4]; using tag_id=4; reproj=0.531px; total=81
```

这里要关注三件事：

```text
using tag_id=4      当前实际用于采样的 tag
seen_ids=[2, 4]     当前画面里检测到的所有 tag
reproj=0.531px      当前 PnP 重投影误差
```

如果画面里有多个 tag，不需要遮挡其他 tag；只要 `--tag-id` 指定正确，工具会忽略其他 tag。

如果终端一直显示：

```text
waiting for target tag
```

说明当前画面没有检测到 `--tag-id` 指定的 tag，检查 tag id、tag family、遮挡、曝光和距离。

## 5. 求解 T_world_camera

运行：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run state_estimation calibrate_world_camera solve \
  --samples ./tmp/finsrov_world_camera_samples.yaml \
  --output ./tmp/finsrov_T_world_camera.yaml \
  --update-extrinsics src/state_estimation/config/state_fusion_extrinsics.yaml \
  --ransac \
  --point-cloud
```

对于“单个 tag 逐点移动采集”，推荐加 `--point-cloud`。这个模式不是“只求位置、不求 yaw”，而是用每个采样点的 tag 中心 3D 坐标做刚体点云配准：

```text
world_tag_position ~= T_world_camera * camera_tag_position
```

它会输出完整的 `T_world_camera`，包括相机 roll/pitch/yaw。例如输出里会有：

```text
rotation_rpy_deg: [...]
```

它只是不会使用每个点里填写的 tag 局部姿态。原因是水平摆放单张 tag 时，tag 平面法向、正反面和 AprilTag 局部坐标系方向很容易和 `pool_world` 定义不一致；如果只是为了求固定相机外参，用中心点云配准通常更稳。

如果你已经准确知道每个 tag 的完整 6D 姿态，包括 tag 局部坐标系 x/y/z 轴在 `pool_world` 中的方向，可以不加 `--point-cloud`，使用完整 6D 模式。完整 6D 模式的公式是：

```text
T_world_camera_i = T_world_tag_i * inverse(T_camera_tag_i)
```

多个样本会对平移和旋转做平均或点云配准。`--ransac` 会先剔除明显离群的样本。

求解结果会写到：

```text
./tmp/finsrov_T_world_camera.yaml
```

同时 `--update-extrinsics` 会直接更新：

```text
src/state_estimation/config/state_fusion_extrinsics.yaml
```

只会替换其中的：

```yaml
T_world_camera:
```

不会改 `T_body_tag`。

## 5.1 如何填写潜器上 tag 的 T_body_tag

`calibrate_world_camera` 只负责求解固定相机外参 `T_world_camera`。潜器上每个 AprilTag 的安装外参 `T_body_tag` 需要人工测量，并写入同一个配置文件：

```text
ros2_ws/src/state_estimation/config/state_fusion_extrinsics.yaml
```

不要在本文档里维护另一套 `T_body_tag` 教程，避免和代码实际约定分叉。统一说明见：

```text
docs/calibration/tag-body-extrinsic.zh-CN.md
ros2_ws/src/state_estimation/docs/t_body_tag_extrinsic.zh-CN.md
```

## 6. 看误差是否合理

命令结束后会打印：

```text
translation_rmse_m
translation_max_error_m
rotation_rmse_deg
rotation_max_error_deg
```

经验判断：

```text
translation_rmse_m < 0.03 m   比较好
translation_rmse_m 0.03~0.08  可先调试，但要谨慎
translation_rmse_m > 0.08 m   通常说明 tag 世界坐标、tag 尺寸、相机内参或 PnP 检测有问题
```

如果误差很大，优先检查：

```text
1. world tag 的 translation_xyz 是否量错
2. world tag 的 yaw 方向是否写反
3. AprilTag marker_length_m 是否是编码区域边长，不是纸张外框
4. 相机内参是否对应当前分辨率
5. /finsrov/vision/tag_poses_3d_camera 的 reprojection_error_px 是否过大
```

如果 `--ransac --point-cloud` 后 `inlier_count` 少于总样本数，说明有些采样点和其它点不一致。常见情况是某一个 `point_name` 整组被剔除，这通常表示这个点的 `--world-xyz` 量错、符号写反、点位没有放准，或者测量参考点不是 tag 中心。

可以用下面的命令查看每组点的残差：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh python - <<'PY'
from pathlib import Path
import yaml, numpy as np
from collections import defaultdict
from state_estimation.calibration_tools.calibrate_world_camera import (
    sample_position_arrays,
    rigid_transform_from_points,
    position_residuals,
)

samples = (yaml.safe_load(Path("tmp/finsrov_world_camera_samples.yaml").read_text()) or {}).get("samples", [])
camera_points, world_points = sample_position_arrays(samples)
initial = rigid_transform_from_points(camera_points, world_points)
initial_residuals = position_residuals(initial, camera_points, world_points)
inliers = np.where(initial_residuals <= 0.08)[0]
solved = rigid_transform_from_points(camera_points[inliers], world_points[inliers])
residuals = position_residuals(solved, camera_points, world_points)

by_point = defaultdict(list)
for index, sample in enumerate(samples):
    by_point[sample.get("point_name", "")].append((index, residuals[index], index in set(inliers)))

for point_name in sorted(by_point):
    values = by_point[point_name]
    errors = np.array([item[1] for item in values])
    inlier_count = sum(1 for item in values if item[2])
    print(
        f"{point_name}: inliers={inlier_count}/{len(values)} "
        f"residual mean/min/max={errors.mean():.4f}/{errors.min():.4f}/{errors.max():.4f} m"
    )
PY
```

如果某组显示：

```text
p02: inliers=0/40 residual mean=0.1360 m
```

说明 p02 这一组整体和其它点不一致，优先检查 p02 的 `--world-xyz`，而不是调整 RANSAC 阈值硬算。

## 7. 标定完成后验证

重新 build 或使用 symlink install 后启动状态融合：

```bash
cd ros2_ws
./scripts/colcon_build_uv.sh --packages-select state_estimation
./scripts/run_ros2_uv.sh ros2 launch state_estimation state_fusion.launch.py
```

查看融合状态：

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/pose --once
```

如果 `/finsrov/state/status` 里出现：

```text
unknown_tag_*
```

说明检测到了没有写入 `state_fusion_extrinsics.yaml` 的 tag，需要补 `T_body_tag`。

如果出现：

```text
vision_mahalanobis_gate
```

说明视觉观测相对 EKF 当前状态跳变过大，可能是外参、tag id 或坐标方向有问题。

## 8. 用带真值的数据集验证 `T_world_camera`

如果已经采集了带真值的 AprilTag PnP 数据集，可以用下面的工具检查当前外参是否和数据集一致：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 run state_estimation evaluate_pnp_truth_dataset \
  --dataset /artifacts/datasets/apriltag_pnp_truth/sessions/20260701_190100 \
  --extrinsics src/state_estimation/config/state_fusion_extrinsics.yaml \
  --origin-xy 2.0 1.0 \
  --truth-z-mode depth_from_surface \
  --water-surface-z-m 1.0 \
  --max-reprojection-error-px 4.0
```

参数含义：

```text
--origin-xy 2.0 1.0
  数据集标定坐标系里的 (2.0, 1.0) 对应当前 ROS/pool_world 的 x/y 原点。

--truth-z-mode depth_from_surface
  数据集 truth_z_m 表示从水面向下的深度，而不是 z-up 高度。

--water-surface-z-m 1.0
  当前 ROS/pool_world 中水面高度为 z=1.0，因此 truth_z_m=1.0 的水底点会转换为 ros_z=0.0。
```

如果你的真值文件已经直接使用底部原点、z-up 坐标，则改成：

```bash
--truth-z-mode z_up
```

不要再使用旧的 `ros_z=-truth_z_m` 方式，除非你明确在复现旧坐标系误差。
