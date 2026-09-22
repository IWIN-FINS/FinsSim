# 标定验收 Checklist

标定完成后不要直接上控制器。先按本 checklist 验证 perception、fusion、controller 三层坐标一致。

## 1. 相机和 tag

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/camera/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted/status --once --full-length
```

检查：

- 相机实际分辨率、fps、fourcc 和标定使用的一致。
- `visible_tag_ids` 是预期的 tag。
- 没有长期 `unknown_tag_*`。
- `reprojection_error_px` 没有长期偏大。
- `constrained_reject_reason` 不是长期 `geometry_error`。

## 2. T_world_camera

移动 tag 或潜器，观察 `/finsrov/vision/refracted_pose_6d`：

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted_pose_6d --once --full-length
```

检查：

```text
向 pool_world +x 移动 -> pose.x 增大
向 pool_world +y 移动 -> pose.y 增大
向上靠近水面 -> pose.z 增大
绕 z 正向旋转 -> yaw 正方向符合约定
```

## 3. T_body_tag

多 tag 场景：

```text
固定潜器不动
分别只露出一个 tag
比较每个 tag 推出的 body pose
```

如果不同 tag 估计差异大，先查：

- tag id 是否写对。
- `translation_xyz` 是否是 tag center in body。
- `rotation_rpy_deg` 是否表示 tag frame 相对 body frame。
- `marker_length_m` 是否是编码区域边长。

## 4. depth 和 body z

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/depth_raw --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/depth_link --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/pose --once --full-length
```

检查：

- `depth_raw` 向下为正。
- `/finsrov/pose.position.z` 是 pool_world z-up 下 body 原点高度。
- 下潜时 `depth_raw` 增大，`/finsrov/pose.position.z` 减小。

## 5. state_fusion

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/pose
```

检查：

- `ready=true`
- `initialized=true`
- `vision_fresh=true` 或可接受的 coast 状态
- `imu_fresh=true`
- `depth_fresh=true`
- covariance 没有持续发散
- 视觉丢失时状态进入 coast/stale，不出现大跳变

## 6. controller 输入

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/controller/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/controller/pose --once --full-length
```

检查：

```text
pool_world.z 接近水面高度 -> controller.y 接近 0
潜器下潜 -> controller.y 变负
pool_world.x 增大 -> controller.x 增大
pool_world.y 增大 -> controller.z 增大
```

确认这些都正确之后，再启动 `motion_controller` 和推进器输出。

