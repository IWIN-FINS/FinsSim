# T_body_tag 外参填写说明

本文档只说明 `state_fusion_extrinsics.yaml` 里的 AprilTag 机体系外参应该怎么填。

标定总入口见：

```text
docs/calibration/index.zh-CN.md
docs/calibration/tag-body-extrinsic.zh-CN.md
```

配置文件位置：

```text
ros2_ws/src/state_estimation/config/state_fusion_extrinsics.yaml
```

## 1. 只允许写 T_body_tag

当前代码只接受：

```yaml
T_body_tag:
```

禁止写：

```yaml
T_tag_body:
```

如果配置里出现 `T_tag_body`，`refractive_apriltag_pose_node` 和 `state_fusion_node` 都会直接启动失败。

## 2. T_body_tag 的定义

本文使用：

```text
T_A_B
```

表示把 B 坐标系下的点变换到 A 坐标系。

因此：

```text
T_body_tag
```

表示：

```text
tag frame -> body frame
p_body = T_body_tag * p_tag
```

也就是说，`T_body_tag` 就是“AprilTag 在潜器 body frame 下的位置和姿态”。这是人工测量最直观的方向。

## 3. body frame 怎么定义

`body frame` 指 `finsrov_base_link`，采用 ROS FLU 右手系：

```text
x: forward，潜器前方
y: left，潜器左侧
z: up，潜器上方
```

机体系原点应该选在控制和状态估计共同使用的潜器中心点。这个点应尽量和 Unity 训练模型里的 ROV root/control center 对应同一个物理点。

## 4. tag frame 怎么定义

当前代码中 AprilTag 的局部角点定义是：

```text
tag origin: tag 中心
tag +x: 从 tag 左边指向右边
tag +y: 从 tag 下边指向上边
tag +z: 按右手系 x cross y 得到，垂直 tag 平面
```

如果 tag 贴在潜器上表面，图案正面朝上，通常有：

```text
tag +z ~= body +z
```

平面内的方向由贴纸旋转角决定。

## 5. translation_xyz 怎么填

`translation_xyz` 就是 tag 中心在 `finsrov_base_link` 下的位置：

```yaml
translation_xyz: [x_body, y_body, z_body]
```

单位是米。

例子：

```text
tag 在机体原点前方 6.5 cm、左侧 0.5 cm、上方 17 cm
```

填写：

```yaml
translation_xyz: [0.065, 0.005, 0.17]
```

不要取负号，不要写反向外参。

## 6. rotation_rpy_deg 怎么填

`rotation_rpy_deg` 是 tag frame 相对 body frame 的姿态：

```text
R_body_tag = Rz(yaw) * Ry(pitch) * Rx(roll)
```

填写格式：

```yaml
rotation_rpy_deg: [roll_deg, pitch_deg, yaw_deg]
```

如果 tag 贴在上表面，且 tag 正面朝上，一般只需要调 `yaw_deg`。

常见情况：

```text
tag +x 指向潜器前方:
rotation_rpy_deg: [0.0, 0.0, 0.0]

tag +x 指向潜器左侧:
rotation_rpy_deg: [0.0, 0.0, 90.0]

tag +x 指向潜器后方:
rotation_rpy_deg: [0.0, 0.0, 180.0]

tag +x 指向潜器右侧:
rotation_rpy_deg: [0.0, 0.0, -90.0]
```

## 7. 两个 tag 沿 body x 轴排布的例子

假设：

```text
tag0 在前方
tag9 在后方
两个 tag 基本沿 body x 轴排布
两个 tag 都贴在上表面
两个 tag 的 +x 都指向潜器前方
```

可以写成：

```yaml
T_body_tag:
  0:
    translation_xyz: [0.065, 0.005, 0.17]
    rotation_rpy_deg: [0.0, 0.0, 0.0]
  9:
    translation_xyz: [-0.065, 0.0, 0.17]
    rotation_rpy_deg: [0.0, 0.0, 0.0]
```

如果实际是 tag9 在前方、tag0 在后方，就交换两个 tag 的 x 坐标。

如果两个 tag 的贴纸图案方向不同，必须分别给每个 tag 填自己的 `rotation_rpy_deg`。

## 8. 如何验证填得对不对

启动视觉节点后看：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted/status
```

重点看：

```text
visible_tag_ids
known_tag_ids
constrained_valid
constrained_residual_m
constrained_reject_reason
```

如果多 tag 外参一致，`constrained_residual_m` 应该明显下降。默认阈值是：

```text
marker_length_m * max_geometry_error_ratio
```

当前 RGB 配置默认：

```text
0.098 * 0.20 = 0.0196 m
```

如果 `constrained_reject_reason=geometry_error`，优先检查：

```text
1. tag id 是否写对
2. translation_xyz 是否是 tag center in body
3. rotation_rpy_deg 是否表达 tag frame 相对 body frame
4. marker_length_m 是否等于实际 tag 边长
5. T_world_camera 是否正确
```
