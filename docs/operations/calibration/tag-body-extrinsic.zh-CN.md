# T_body_tag 安装外参标定

`T_body_tag` 描述 AprilTag 安装在潜器上的位置和方向。

包内详细说明见：

```text
ros2_ws/src/state_estimation/docs/t_body_tag_extrinsic.zh-CN.md
```

## 1. 只写 T_body_tag

当前配置推荐且只应写：

```yaml
T_body_tag:
```

不要写旧字段：

```yaml
T_tag_body:
```

`T_body_tag` 的语义：

```text
tag frame -> body frame
p_body = T_body_tag * p_tag
```

也就是“tag 中心和 tag 坐标轴在潜器 body frame 下的位置和姿态”。

## 2. body frame

当前 `finsrov_base_link` 使用 ROS FLU 右手系：

```text
body x: forward，潜器前方
body y: left，潜器左侧
body z: up，潜器上方
```

body 原点应尽量对应控制中心、状态估计中心和 Unity 训练模型的 ROV root/control center。

## 3. 怎么人工测量

对每个 tag 记录：

```text
translation_xyz = tag 中心在 body frame 下的位置，单位 m
rotation_rpy_deg = tag frame 相对 body frame 的姿态，单位 deg
```

例子：

```yaml
T_body_tag:
  4:
    translation_xyz: [0.065, 0.005, 0.170]
    rotation_rpy_deg: [0.0, 0.0, 90.0]
```

如果 tag 贴在潜器上表面且正面朝上，通常 roll/pitch 接近 0，主要测 yaw：

```text
tag +x 指向潜器前方: yaw = 0 deg
tag +x 指向潜器左侧: yaw = 90 deg
tag +x 指向潜器后方: yaw = 180 deg
tag +x 指向潜器右侧: yaw = -90 deg
```

## 4. 多 tag 验证

固定潜器不动，分别遮挡其它 tag，只保留一个 tag 可见，比较 `/finsrov/vision/refracted_pose_6d` 或 `/finsrov/pose`：

```text
只看 tag A -> body pose
只看 tag B -> body pose
只看 tag C -> body pose
```

如果 `T_body_tag` 正确，不同 tag 反推的 body pose 应基本一致。

常见错误：

```text
位置相差接近 tag 安装间距: translation 方向写反
yaw 差 90/180 deg: tag frame yaw 写错
z 方向反: tag 正面方向或 body z 定义错
unknown_tag_*: 检测到了没有写入 T_body_tag 的 tag id
```
