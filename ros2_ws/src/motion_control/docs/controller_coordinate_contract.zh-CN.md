# Controller 坐标契约

本文档把 `motion_control` 相关话题的坐标约定写死，避免 real、sim、analysis、identifier 之间再次发生二次坐标转换。

## 1. 两层状态话题

系统里明确分成两层：

1. raw 物理/融合层
   - `/finsrov/pose`
   - `/finsrov/imu_link`
   - `/finsrov/depth_link`
   - `/finsrov/dvl_link`
2. controller 输入层
   - `/finsrov/controller/pose`
   - `/finsrov/controller/imu`
   - `/finsrov/controller/depth`
   - `/finsrov/controller/dvl`
   - `/finsrov/controller/state/status`

`motion_controller`、RL policy、controller diagnostics 默认都应该消费第二层。

## 2. raw 层约定

real `state_estimation` 输出的 raw 层约定是：

- `/finsrov/pose`
  - `frame_id = pool_world`
  - `pool_world` 为 `z-up`
- `/finsrov/imu_link`
  - `frame_id = finsrov_base_link`
  - body 轴为 ROS FLU:
    - `x` forward
    - `y` left
    - `z` up
- `/finsrov/dvl_link`
  - `frame_id = finsrov_base_link`
  - body 速度同样使用 ROS FLU

## 3. controller 层约定

controller 层统一采用 Unity / 训练模型约定：

- `controller_world`
  - `x` forward-like horizontal axis
  - `y` up, 且 `y=0` 表示水面
  - `z` left-like horizontal axis
- `controller_body`
  - `x` forward
  - `y` up
  - `z` left

对应分量映射写死为：

```text
controller.x = ros.x
controller.y = ros.z
controller.z = ros.y
```

也就是：

```text
B =
[ 1  0  0 ]
[ 0  0  1 ]
[ 0  1  0 ]
```

位置、线速度和线加速度等普通向量使用：

```text
p_controller = B * p_ros + [0, -water_surface_z_m, 0]
v_controller = B * v_ros
a_controller = B * a_ros
R_controller = B * R_ros * B^T
```

角速度是轴向量。由于默认 `det(B)=-1`，必须使用：

```text
A = det(B) * B = -B
omega_controller = A * omega_ros
[wx, wy, wz]_controller = [-wx, -wz, -wy]_ros
```

角速度协方差使用 `A * C * A.T`；混合线性/角速度的 6D 协方差使用
`diag(B, A)`。controller body 命令已经是 controller 坐标，不再转换。

其中水面 offset 只用于 real raw world position 转换到 `controller_world`。
姿态使用旋转矩阵换基，不要直接交换 quaternion 分量。

## 4. `controller_state_adapter` 的职责

`controller_state_adapter` 是 real 链路进入 controller 层的唯一边界。

它的输入默认是：

- `/finsrov/pose`
- `/finsrov/imu_link`
- `/finsrov/depth_link`
- `/finsrov/dvl_link`
- `/finsrov/state/status`

它的输出默认是：

- `/finsrov/controller/pose`
- `/finsrov/controller/imu`
- `/finsrov/controller/depth`
- `/finsrov/controller/dvl`
- `/finsrov/controller/state/status`

从 2026-08-02 开始，adapter 的行为明确为：

1. 如果输入 `frame_id` 是 raw 约定
   - `pool_world`
   - `finsrov_base_link`
   - 其他非 `controller_*` frame

   那么执行一次 raw -> controller 变换。

2. 如果输入 `frame_id` 已经是
   - `controller_world`
   - `controller_body`

   那么直接透传，不再二次变换。

这条规则是为了防止同一份 controller-native 数据被再次按 ROS FLU / pool_world 解释。

## 5. Unity sim-truth 约定

Unity `VehicleRosBridge` 直接发布 `/finsrov/controller/*`，这些话题必须已经满足 controller 层约定。

因此：

- 消费 `/finsrov/controller/imu`、`/finsrov/controller/dvl` 的节点，不应该再按 raw FLU 做 `[0,2,1]` remap。
- 如果一个分析节点要读 `/finsrov/imu_link`、`/finsrov/dvl_link`，它才应该做 raw -> controller 的 basis 变换。

## 6. 排障规则

遇到 real/sim 坐标异常时，先判断读的是哪一层 topic：

1. 读的是 `/finsrov/imu_link` / `/finsrov/dvl_link`
   - 这是 raw 层
   - 可以做 raw -> controller 变换
2. 读的是 `/finsrov/controller/imu` / `/finsrov/controller/dvl`
   - 这是 controller 层
   - 不能再做第二次变换

最常见错误就是把 `/finsrov/controller/*` 当成 raw 层再映射一遍。

## 7. 推荐检查命令

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/imu_link --once
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/controller/imu --once
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/dvl_link --once
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/controller/dvl --once
```

如果 real 链路正常，应满足：

- `/finsrov/imu_link` 是 raw FLU
- `/finsrov/controller/imu` 是 controller body
- `/finsrov/dvl_link` 是 raw FLU
- `/finsrov/controller/dvl` 是 controller body
