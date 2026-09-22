# FinsROV 水面折射 AprilTag 6D 定位与状态融合算法说明

本文以算法说明为主，描述固定俯视相机透过水面观察水下 FinsROV 顶部 AprilTag 刚性板时，如何利用 AprilTag 角点、水压计深度和 IMU 信息估计潜器的 6D pose，并进一步通过 Python EKF 融合发布控制器/RL 可用状态。

工程入口见：

```text
ros2_ws/src/perception/src/refractive_apriltag_pose_node.cpp
ros2_ws/src/state_estimation/state_estimation/state_fusion_node.py
ros2_ws/src/state_estimation/config/state_fusion.yaml
```

## 摘要

系统目标是在水面上方固定相机、潜器位于水下的条件下，估计潜器在 `pool_world` 中的位姿：

```text
T_world_body = [R_world_body, t_world_body]
```

其中：

```text
t_world_body = [x, y, z]^T
R_world_body = R(roll, pitch, yaw)
```

由于水面折射、顶部 tag 共面、俯视视角和水下图像质量等因素，纯视觉直接优化完整 6D pose 容易病态。因此当前控制主链路采用约束式 6D measurement：

```text
x, y, yaw 由 AprilTag 多角点 + Snell 折射几何优化得到
z          由水压计深度强约束
roll,pitch 由 IMU quaternion 强约束
```

之后 Python fusion 层用：

```text
PositionVelocityEKF: [x, y, z, vx, vy, vz]
YawEKF: yaw
```

融合异步视觉、深度和 IMU 信息，输出 `/finsrov/pose`、`/finsrov/imu_link`、`/finsrov/depth_link`、`/finsrov/dvl_link`。

## 1. 问题定义

### 1.1 已知量

设相机固定在水面上方，潜器顶部安装多个 AprilTag，所有 tag 和潜器刚性连接。

已知标定量：

```text
K, D
  相机内参和畸变参数。

T_world_camera
  相机坐标系到 pool_world 的外参。

T_body_tag_j
  第 j 个 AprilTag 坐标系到潜器 body 坐标系的外参。

L
  AprilTag 编码区域边长 marker_length_m。

Π_water
  水面平面。

n_air, n_water
  空气和水的折射率。
```

其中 `K` 是 3x3 pinhole camera matrix：

```text
K =
[ fx   s  cx
   0  fy  cy
   0   0   1 ]
```

含义：

```text
fx, fy:
  以 pixel 为单位的焦距。它们决定像素偏移对应的视线角度。

cx, cy:
  主点坐标，通常接近图像中心，但不应假设一定等于 width/2、height/2。

s:
  skew，普通 USB 相机通常接近 0。
```

`D` 是镜头畸变参数。当前 OpenCV 标定文件通常使用 plumb_bob / Brown-Conrady 模型：

```text
D = [k1, k2, p1, p2, k3]
```

含义：

```text
k1, k2, k3:
  径向畸变参数，修正广角或镜头边缘带来的桶形/枕形畸变。

p1, p2:
  切向畸变参数，修正镜头光轴和成像平面不完全平行导致的偏移。
```

在算法中，`K,D` 的作用是把 AprilTag 角点像素 `u_jk=[u,v]` 变成相机坐标系下的单位视线。如果 `K,D` 和当前相机分辨率、焦距、裁剪方式不一致，后面的 Snell 折射和 PnP 都会系统性偏移。

传感器观测：

```text
u_jk
  第 j 个 AprilTag 第 k 个角点的图像像素坐标。

d
  水压计深度，水面向下为正。

q_imu, omega_imu
  IMU quaternion 和角速度。
```

### 1.2 AprilTag 位置观测是怎么来的

相机本身并不直接输出 AprilTag 的 3D 位置。当前链路先由 C++ AprilTag 3 detector 在图像里识别 tag：

```text
image frame
  -> 灰度/阈值/边缘与四边形候选
  -> 解码 AprilTag family 和 id
  -> 输出 tag 中心像素和四个角点像素
```

工程 topic：

```text
/finsrov/vision/tag_detections_2d
```

每个检测包含：

```text
tag_id
family
decision_margin
center_px
corner_pixels_xy = [x0,y0, x1,y1, x2,y2, x3,y3]
mean_edge_px
```

这里得到的是 **2D 像素角点**，不是潜器 3D pose。后续算法利用：

```text
1. tag id -> 查 T_body_tag_j，知道这个 tag 在潜器上的安装位置
2. corner_pixels_xy -> 作为相机观测 u_jk
3. marker_length_m -> 构造 tag 四个角点在 tag frame 下的 3D 坐标
4. K,D,T_world_camera,Snell,depth,IMU -> 反推出潜器 pose
```

所以“识别到 AprilTag 的位置”严格分两层：

```text
AprilTag detector:
  只识别图像中的 tag id 和角点像素。

refractive_apriltag_pose_node:
  才根据几何模型、水压计和 IMU 解算 pool_world 下的 6D pose。
```

### 1.3 待估计量

理论完整目标是：

```text
X_6d = [x, y, z, roll, pitch, yaw]^T
```

但当前主算法不直接优化完整 6D，而是求解：

```text
theta = [x, y, yaw]^T
```

并由其它传感器给出：

```text
z     = f_depth(d, roll_imu, pitch_imu)
roll  = roll_imu
pitch = pitch_imu
```

最终输出仍是完整 6D pose：

```text
t_world_body = [x, y, z]^T
q_world_body = quat(roll_imu, pitch_imu, yaw)
```

### 1.4 为什么不直接纯视觉 6D

如果只用顶部共面 AprilTag 角点做完整 6D 优化，存在几个问题：

```text
1. 顶部 tag 大多近似共面，z/roll/pitch 可观性较弱。
2. 固定俯视相机下，x/y/yaw 最稳定，z/roll/pitch 更容易被噪声放大。
3. 水面折射模型对水面平整度、折射率、相机外参敏感。
4. 水压计对 z 的观测更直接，IMU 对 roll/pitch 的短期观测更稳定。
```

因此主链路采用：

```text
视觉主约束 x/y/yaw
水压计主约束 z
IMU 主约束 roll/pitch
```

## 2. 坐标系与刚体模型

### 2.1 坐标系

`pool_world`：

```text
x: 前向/池长方向
y: 左向/水平横向
z: 向上
z=0: 水池底部参考平面
z=water_surface_z_m: 水面
```

`body = finsrov_base_link`：

```text
x: forward
y: left
z: up
```

`tag_j`：

```text
origin: tag 中心
x: tag 左边 -> 右边
y: tag 下边 -> 上边
z: 右手系垂直 tag 平面
```

### 2.2 AprilTag 角点刚体坐标

设 tag 边长为 `L`，`h = L/2`。单个 tag 的四个角点在 tag frame 下定义为：

```text
q_0 = [-h,  h, 0]^T
q_1 = [ h,  h, 0]^T
q_2 = [ h, -h, 0]^T
q_3 = [-h, -h, 0]^T
```

第 `j` 个 tag 的第 `k` 个角点转换到 body frame：

```text
P_body_jk = T_body_tag_j * q_k
```

如果一帧中检测到多个 tag，则所有角点组成同一个刚体观测集合：

```text
S = { (j, k, P_body_jk, u_jk) }
```

这一步是多 tag 融合的基础：所有 tag 都不是独立目标，而是同一个潜器刚体上的不同 3D 点。

## 3. 深度与 IMU 约束

### 3.1 水压计深度到 body z

硬件深度 `d` 定义为水面向下为正。`pool_world` 是 z-up，因此压力传感器位置的 world z 为：

```text
z_pressure = water_surface_z_m + depth_sign * d
```

当前：

```text
depth_sign = -1
```

压力传感器不一定在 body 原点，设其在 body frame 下的 z 偏移为：

```text
p_sensor_body = [0, 0, z_offset]^T
```

只考虑 roll/pitch 对垂向投影的影响，当前实现使用：

```text
z_body = z_pressure - z_offset * cos(roll_imu) * cos(pitch_imu)
```

也就是：

```text
z = z_body
```

由水压计强约束。

### 3.2 IMU 姿态先验

IMU quaternion 转换为：

```text
(roll_imu, pitch_imu, yaw_imu)
```

当前 constrained 视觉解算使用：

```text
roll  = roll_imu
pitch = pitch_imu
```

但不使用 `yaw_imu` 作为绝对世界 yaw。原因是 IMU yaw 的零点可能不与 `pool_world` 对齐，且存在漂移。

角速度：

```text
omega_z = imu.angular_velocity.z
```

用于 Python `YawEKF` 的 yaw 预测。

## 4. 水面折射成像模型

### 4.1 水面平面

设水面为平面：

```text
Π_water: n^T X + b = 0
```

当前第一版通常使用水平水面：

```text
n = [0, 0, 1]^T
z = water_surface_z_m
```

### 4.2 像素反投影为空气 ray

给定像素：

```text
u = [u, v]^T
```

使用相机内参和畸变去畸变，得到相机坐标系下的单位视线：

```text
r_camera = normalize(K^-1 * [u, v, 1]^T)
```

用相机外参转到 world：

```text
C_world = t_world_camera
r_air_world = R_world_camera * r_camera
```

空气中光线：

```text
X_air(s) = C_world + s * r_air_world
```

### 4.3 与水面相交

求交点 `S`：

```text
n^T (C_world + s * r_air_world) + b = 0
```

得到：

```text
s = -(n^T C_world + b) / (n^T r_air_world)
S = C_world + s * r_air_world
```

如果分母接近 0，说明 ray 与水面近似平行，应 reject。

### 4.4 Snell 折射

Snell 定律：

```text
n_air * sin(alpha_air) = n_water * sin(alpha_water)
```

其中 `alpha_air` 和 `alpha_water` 是光线与水面法线的夹角。

空气入水时：

```text
n_water > n_air
=> alpha_water < alpha_air
```

即水中光线向法线偏折。

向量形式中，给定入射方向 `r_air`、界面法线 `n`、折射率比：

```text
eta = n_air / n_water
```

可以求得折射方向：

```text
r_water = refract(r_air, n, eta)
```

若发生无效几何或数值异常，则该角点 residual 置为大残差或 reject。

### 4.5 反投影到目标深度平面

给定候选 pose 下某个角点预测世界坐标 `P_world_jk(theta)`，取其 z 值：

```text
z_target = P_world_jk(theta).z
```

水中折射 ray：

```text
X_water(t) = S + t * r_water
```

与水平面 `z = z_target` 相交：

```text
t = (z_target - S_z) / r_water_z
Q_world_jk = S + t * r_water
```

`Q_world_jk` 表示：在当前预测角点深度平面上，这个像素观测根据 Snell 模型对应的水下世界点。

### 4.6 为什么 Snell 模型不能直接确定潜器位置

Snell 模型可以把 **一个像素观测** 转换成 **一条经过水面折射后的水下 ray**。如果再给定一个目标深度平面 `z=z_target`，就能求出这条 ray 在该平面上的交点：

```text
u -> refracted ray -> Q_world(z_target)
```

但潜器 pose 不是单个角点的位置。未知量包括：

```text
x, y, yaw
```

而每个角点的真实 world z 又依赖潜器姿态和该角点在 body 上的位置：

```text
P_world_jk(theta).z =
  [Rz(yaw) * Ry(pitch_imu) * Rx(roll_imu) * P_body_jk + t_world_body].z
```

也就是说：

```text
要反投影像素，需要知道角点所在的 z 平面；
角点所在的 z 平面，又依赖当前候选 yaw 和 body 刚体布局。
```

此外，多角点之间必须满足刚体约束：

```text
同一个 theta 必须同时解释两个 tag 的所有角点。
```

所以单个 Snell 反投影只能给出“某个深度平面上的观测点”，不能直接给出唯一的潜器 `[x,y,yaw]`。优化问题的作用就是寻找一个潜器 pose，使所有角点的 Snell 反投影观测和刚体预测位置整体最一致。

## 5. Constrained 6D 优化

### 5.1 状态变量

定义优化变量：

```text
theta = [x, y, psi]^T
```

其中 `psi` 是 yaw。

由 depth/IMU 补全完整 pose：

```text
t_world_body(theta) = [x, y, z_body]^T
R_world_body(theta) = Rz(psi) * Ry(pitch_imu) * Rx(roll_imu)
```

### 5.2 角点预测

对每个刚体角点：

```text
P_world_jk(theta) = R_world_body(theta) * P_body_jk + t_world_body(theta)
```

### 5.3 残差定义

像素观测通过 Snell 反投影到 `P_world_jk(theta).z` 平面：

```text
Q_world_jk(theta) = RefractBackProject(u_jk, P_world_jk(theta).z)
```

残差：

```text
r_jk(theta) = Q_world_jk(theta) - P_world_jk(theta)
```

注意 `Q_world_jk` 也依赖 `theta`，因为它的目标 z 平面来自 `P_world_jk(theta).z`。

### 5.4 优化目标

全部 tag、全部角点联合优化：

```text
theta* = argmin_theta Σ_(j,k) || r_jk(theta) ||²
```

展开为：

```text
theta* = argmin_[x,y,psi] Σ_(j,k)
  || RefractBackProject(u_jk, P_world_jk([x,y,psi]).z)
     - P_world_jk([x,y,psi]) ||²
```

当前实现使用数值 Jacobian 和小规模 Levenberg-Marquardt：

```text
J = ∂r / ∂theta
H = J^T J + λI
δ = solve(H, J^T (-r))
theta_next = theta + δ
```

接受准则：

```text
if cost(theta_next) < cost(theta):
  theta = theta_next
  λ = λ * 0.5
else:
  λ = λ * 5.0
```

终止条件：

```text
|δ_x|, |δ_y|, |δ_yaw| 足够小
或达到 max_iterations
```

### 5.5 优化目标函数的意义

这个目标函数的核心意义是：寻找一个潜器位姿 `theta=[x,y,yaw]`，使 **相机看到的 AprilTag 角点**、**Snell 折射几何**、**水压计深度**、**IMU roll/pitch**、**tag 在潜器上的刚体布局** 同时尽可能一致。

对每一个 tag 角点 `(j,k)`，目标函数比较的是两个点：

```text
Q_world_jk(theta)
```

和：

```text
P_world_jk(theta)
```

二者含义不同：

```text
Q_world_jk(theta):
  从相机像素 u_jk 出发，经过 K,D 去畸变和反投影，
  再经过空气/水界面的 Snell 折射，
  最后落到当前预测角点深度平面上的“观测点”。

P_world_jk(theta):
  根据假设的潜器位姿 theta、IMU roll/pitch、水压计 z、
  以及已知的 T_body_tag 和 tag 尺寸，
  由刚体模型推算出来的“该角点理论上应该在世界中的位置”。
```

残差：

```text
r_jk(theta) = Q_world_jk(theta) - P_world_jk(theta)
```

就是在问：

```text
如果潜器真的处在 theta 这个位置和 yaw，
那么该角点的刚体预测位置，
是否和相机像素经过折射模型反推出的位置一致？
```

所有角点的残差平方和：

```text
Σ_(j,k) ||r_jk(theta)||²
```

就是整体几何不一致程度。优化器最小化它，相当于在找一个让所有角点“共同解释得最好”的潜器 `[x,y,yaw]`。

这不是在单纯拟合一个 tag 中心点，而是在同时使用：

```text
1. 每个 tag 的 4 个角点像素；
2. 每个 tag 在潜器 body 下的固定外参 T_body_tag；
3. 相机外参 T_world_camera；
4. 水压计给出的 body z；
5. IMU 给出的 roll/pitch；
6. 水面折射几何。
```

因此，目标函数的物理含义是一个 **多传感器几何一致性误差**。

如果目标函数最小值很小，说明当前视觉角点、深度、IMU、外参和折射模型彼此一致，可以把结果作为有效视觉 measurement。  
如果目标函数最小值很大，通常说明存在以下问题之一：

```text
1. tag 检测角点错误；
2. tag_id 或 T_body_tag 配错；
3. marker_length_m 配错；
4. camera K,D 或 T_world_camera 配错；
5. 水面扰动导致 Snell 平面假设失效；
6. depth 或 IMU 数据和图像时间不同步；
7. 潜器处在水面过渡区，折射模型本身不稳定。
```

所以目标函数还有第二个作用：它不仅给出最优位姿，还提供一个 reject/gating 指标。当前系统会用 residual、reprojection error、geometry error 等量判断这帧视觉是否可信，避免错误视觉测量把 EKF 拉炸。

从数学上看，Snell 模型只解决“像素光线经过水面后往哪里走”的局部光线问题；优化目标函数解决的是“哪个潜器整体位姿能同时解释所有这些折射光线”的全局刚体定位问题。这两个层次不能互相替代。

### 5.6 yaw 是怎么计算的

当前 yaw 不是简单地“识别两个 AprilTag 在 `pool_world` 里的位置，然后取连线角度”。

原因有两点：

```text
1. AprilTag detector 直接给的是图像 2D 角点，不是每个 tag 的可靠 world 3D 中心。
2. 即使有两个 tag 的中心，中心连线只利用了很少的信息，无法利用每个 tag 的四个角点、tag 自身朝向和折射几何残差。
```

当前 yaw 是优化变量：

```text
theta = [x, y, yaw]
```

在每次迭代中，yaw 会改变所有 body 角点在 `pool_world` 中的预测位置：

```text
P_world_jk(yaw) = Rz(yaw) * Ry(pitch_imu) * Rx(roll_imu) * P_body_jk + [x,y,z]^T
```

优化器寻找一个 yaw，使得所有可见 tag 的所有角点同时满足：

```text
Snell 反投影得到的 Q_world_jk
≈
刚体模型预测的 P_world_jk
```

如果潜器上只有两个 tag，它们确实会给 yaw 提供强约束，但约束来源不是“两个中心点连线”这一条量，而是：

```text
两个 tag 的 8 个角点
+ 每个 tag 的 T_body_tag
+ 每个 tag 自身平面方向
+ Snell 折射反投影残差
```

因此它比中心连线角更充分，也能在只看到一个 tag 时仍然利用该 tag 四个角点的方向信息估计 yaw。

### 5.7 输出位姿

优化结束后：

```text
x*   = theta*[0]
y*   = theta*[1]
psi* = theta*[2]
z*   = z_body
```

输出：

```text
position = [x*, y*, z*]
quaternion = quat(roll_imu, pitch_imu, psi*)
```

发布：

```text
/finsrov/vision/refracted_pose_6d
```

如果几何误差过大：

```text
constrained_residual_m / marker_length_m > max_geometry_error_ratio
```

则拒绝该 measurement：

```text
constrained_reject_reason = geometry_error
```

### 5.8 水面附近是否使用折射模型

不是所有深度都使用水下折射模型。当前按水压计深度分模式：

```text
depth_m < air_depth_threshold_m
  mode = air
  潜器/tag 近似在空气中或刚接触水面，不使用 Snell underwater constrained 模型。

air_depth_threshold_m <= depth_m < underwater_depth_threshold_m
  mode = surface_transition
  tag 可能部分出水、部分入水，水面遮挡/反光/折射状态不稳定。

depth_m >= underwater_depth_threshold_m
  mode = underwater_refraction
  使用 Snell + depth + IMU constrained 优化。
```

当前默认阈值：

```yaml
air_depth_threshold_m: 0.02
underwater_depth_threshold_m: 0.08
```

也就是：

```text
depth < 2 cm:
  空气/近水面模式。

2 cm <= depth < 8 cm:
  水面过渡区。

depth >= 8 cm:
  水下折射模式。
```

在 `air` 或 `surface_transition` 时，如果配置允许：

```yaml
publish_air_pose_on_constrained_topic: true
publish_surface_transition_pose_on_constrained_topic: true
```

节点可能在 constrained topic 上发布 pinhole baseline，并用 pressure depth 替换 z：

```text
constrained_output_model = pinhole_air
```

这不是 Snell constrained 结果。控制侧和调试时必须看：

```text
/finsrov/vision/refracted/status
  mode
  constrained_output_model
  constrained_valid
  constrained_reject_reason
```

## 6. 多 tag 融合算法

### 6.1 主链路：角点级联合优化

多 tag 融合发生在 residual 层，而不是 pose 层。

对所有可见 tag：

```text
S = union_j { four corners of tag j }
```

每个角点都转换到同一个 body frame：

```text
P_body_jk = T_body_tag_j * q_tag_k
```

然后使用同一个 `theta = [x,y,yaw]` 解释所有角点：

```text
P_world_jk(theta) = R(theta) * P_body_jk + t(theta)
```

联合目标：

```text
theta* = argmin_theta Σ_(j,k in all visible tags) ||r_jk(theta)||²
```

因此：

- 多 tag 数量越多，残差约束越多。
- tag 之间的相对位置由 `T_body_tag` 刚体外参固定。
- 某个 tag 外参错误会表现为整体残差上升或 `geometry_error`。
- 不存在主链路中“tag A pose 和 tag B pose 做平均”的步骤。

如果当前潜器上贴了两个 AprilTag，例如 tag 15 和 tag 16，那么一帧中理想情况下会形成：

```text
tag 15:
  corner 0,1,2,3 -> 4 个 residual

tag 16:
  corner 0,1,2,3 -> 4 个 residual
```

每个 residual 是 3 维：

```text
r_jk = [r_x, r_y, r_z]^T
```

所以两个 tag 同时可见时，优化器实际最小化的是 8 个角点、24 个标量残差共同组成的目标函数：

```text
theta* = argmin_theta (
  ||r_15,0||² + ||r_15,1||² + ||r_15,2||² + ||r_15,3||²
 +||r_16,0||² + ||r_16,1||² + ||r_16,2||² + ||r_16,3||²
)
```

两个 tag 的作用：

```text
1. 增加角点数量，提高抗噪声能力。
2. 扩大刚体点分布范围，提高 yaw 约束。
3. 当其中一个 tag 角点质量差时，另一个 tag 仍提供约束。
4. 通过 residual 一致性暴露 T_body_tag 或 marker_length_m 错误。
```

如果只有一个 tag 可见，算法仍可运行，但 yaw 和位置对噪声更敏感；如果两个 tag 都可见，联合优化会更稳定。

### 6.2 旧 PnP fallback：pose 级选择/融合

旧 `/finsrov/vision/tag_poses_3d_camera` 入口仍保留，但当前默认：

```yaml
use_vision_pnp: false
```

如果启用 fallback，Python fusion 对每个 tag 先算：

```text
T_world_body_j = T_world_camera * T_camera_tag_j * inverse(T_body_tag_j)
```

然后：

```text
if max distance between candidates > multi_tag_conflict_distance_m:
  choose candidate with lowest reprojection_error
else:
  weighted average position and yaw
```

这是调试/回退逻辑，不是当前实船默认控制主链路。

“PnP fallback”的含义是：不考虑水面折射，直接使用普通 pinhole 相机模型和 OpenCV `solvePnP` 从 tag 角点求解相机坐标系下的 tag pose：

```text
object points: tag/body 上已知 3D 角点
image points:  图像中的 2D 角点
K,D:           相机内参和畸变
solvePnP ->    T_camera_tag 或 T_camera_body
```

再通过外参转换到 `pool_world`：

```text
T_world_body = T_world_camera * T_camera_tag * inverse(T_body_tag)
```

它适合：

```text
空气中调试
折射模型关闭时的 baseline
检查相机内参、tag id、T_body_tag 是否大致正确
```

它不适合作为水下默认控制输入，因为它假设光线在同一种介质中直线传播，没有处理空气-水界面的折射。

## 7. Python EKF 状态融合

Python fusion 节点：

```text
ros2_ws/src/state_estimation/state_estimation/state_fusion_node.py
```

当前默认输入：

```yaml
vision_6d_topic: /finsrov/vision/refracted_pose_6d
use_vision_6d: true
use_vision_6d_z: false
use_vision_yaw: true

vision_pnp_topic: /finsrov/vision/tag_poses_3d_camera
use_vision_pnp: false

imu_raw_topic: /finsrov/hardware/imu_raw
depth_raw_topic: /finsrov/hardware/depth_raw
```

### 7.1 PositionVelocityEKF

状态：

```text
x = [p_x, p_y, p_z, v_x, v_y, v_z]^T
```

预测模型是常速度模型：

```text
p_k^- = p_{k-1} + v_{k-1} Δt
v_k^- = v_{k-1}
```

矩阵：

```text
x_k^- = F x_{k-1}

F =
[1 0 0 Δt 0  0 ]
[0 1 0 0  Δt 0 ]
[0 0 1 0  0  Δt]
[0 0 0 1  0  0 ]
[0 0 0 0  1  0 ]
[0 0 0 0  0  1 ]
```

协方差预测：

```text
P_k^- = F P_{k-1} F^T + Q
```

视觉更新默认只更新 x/y：

```text
z_vision = [x_vision, y_vision]^T

H_vision =
[1 0 0 0 0 0
 0 1 0 0 0 0]
```

深度更新只更新 z：

```text
z_depth = z_body

H_depth =
[0 0 1 0 0 0]
```

标准更新：

```text
y = z - H x^-
S = H P^- H^T + R
K = P^- H^T S^-1
x = x^- + K y
P = (I - K H) P^- (I - K H)^T + K R K^T
```

### 7.2 YawEKF

Yaw 状态：

```text
ψ = yaw
```

预测使用 IMU gyro：

```text
ψ_k^- = wrap(ψ_{k-1} + ω_z Δt)
P_k^- = P_{k-1} + q_yaw Δt
```

视觉更新使用 AprilTag/Snell yaw：

```text
z_yaw = yaw(refracted_pose_6d.orientation)
e = wrap(z_yaw - ψ^-)
S = P^- + R_yaw
K = P^- / S
ψ = wrap(ψ^- + K e)
P = (1 - K) P^-
```

这就是当前对 IMU yaw 漂移的主要修正机制：

```text
短期: IMU gyro 提供平滑高速 yaw 变化
长期: AprilTag yaw 把漂移拉回 pool_world 绝对方向
```

这里的 `IMU gyro` 指 IMU 陀螺仪测得的角速度：

```text
imu.angular_velocity = [omega_x, omega_y, omega_z]
```

其中 `omega_z` 是绕 body z 轴的角速度，单位通常是 rad/s。对于 yaw 来说，它提供短时间内的转动增量：

```text
delta_yaw ≈ omega_z * Δt
```

IMU yaw 漂移的来源主要是：

```text
1. gyro 存在零偏 bias，即静止时 omega_z 不完全为 0。
2. yaw 由角速度积分得到，微小 bias 会随时间累积成角度误差。
3. 磁力计在室内/水下/电机附近容易受干扰，未必能稳定提供绝对航向。
4. IMU quaternion 的 yaw 零点不一定和 pool_world 的 yaw=0 对齐。
```

因此，纯靠 gyro 积分：

```text
yaw(t) = yaw(0) + ∫ omega_z dt
```

会随时间慢慢偏离真实 `pool_world` yaw。

AprilTag/Snell 视觉给出的 yaw 来自几何外参和角点观测：

```text
T_world_camera
T_body_tag
AprilTag corner pixels
Snell/depth/IMU roll-pitch constrained optimization
```

它是 `pool_world` 下的低频绝对 yaw measurement。因此 YawEKF 用：

```text
gyro_z 预测
vision_yaw 更新
```

来实现：

```text
保留 IMU 高频平滑性
同时抑制长期 yaw 漂移
```

当前没有用 AprilTag 修正 roll/pitch。roll/pitch 漂移如果成为问题，需要后续引入更完整的姿态融合设计，例如视觉 roll/pitch 可观性评估、IMU bias 状态、或外部姿态约束。

### 7.3 Mahalanobis gate

每次观测更新前计算：

```text
d² = y^T S^-1 y
```

如果：

```text
d² > measurement_gate_mahalanobis
```

拒绝更新。默认：

```yaml
measurement_gate_mahalanobis: 9.0
```

常见拒绝原因：

```text
vision_6d_mahalanobis_gate
vision_6d_yaw_mahalanobis_gate
depth_mahalanobis_gate
```

这个 gate 防止 tag 重现、水面扰动、外参错误或 depth 跳变导致 EKF 突跳。

## 8. 视觉丢失状态机

异步传感器频率不同，且潜器可能跑出相机视野。因此 fusion 不能在视觉丢失后无限常速度漂移。

当前状态：

```text
uninitialized:
  还没有有效视觉。

fresh:
  最近视觉更新仍在 vision_timeout_sec 内。

coast:
  视觉短时丢失，但未超过 vision_coast_timeout_sec。
  EKF 继续常速度预测。

hold:
  视觉丢失太久。
  x/y 固定到最后一次有效视觉位置。
  x/y 速度衰减/归零。
```

典型参数：

```yaml
vision_timeout_sec: 0.25
vision_coast_timeout_sec: 0.75
imu_timeout_sec: 0.10
depth_timeout_sec: 0.30
```

这样设计是为了避免：

```text
AprilTag 长时间不可见
-> EKF 只靠旧速度预测
-> pose 漂移到相机外或错误位置
-> controller/RL 得到危险 observation
```

## 9. 输出语义

### 9.1 `/finsrov/pose`

```text
frame_id = pool_world
position = [p_x, p_y, p_z] from PositionVelocityEKF
orientation = quat(roll_imu, pitch_imu, yaw_fused)
```

这是状态融合后的潜器 body 原点在 `pool_world` 下的完整位姿，是控制器/RL 理解“潜器在哪里、朝向如何”的主 pose。

具体含义：

```text
position.x:
  pool_world x，来自视觉 x 和 EKF 预测/更新。

position.y:
  pool_world y，来自视觉 y 和 EKF 预测/更新。

position.z:
  pool_world z-up 下的 body 原点高度，主要由水压计 depth 更新。

orientation.roll/pitch:
  来自 IMU quaternion。

orientation.yaw:
  来自 YawEKF，短期由 gyro 预测，长期由 AprilTag/Snell yaw 修正。
```

它不是 raw vision pose，也不是 raw IMU pose，而是 fusion 后的导航状态。

### 9.2 `/finsrov/imu_link`

```text
frame_id = finsrov_base_link
orientation = quat(roll_imu, pitch_imu, yaw_fused)
angular_velocity = raw IMU angular_velocity
linear_acceleration = raw IMU linear_acceleration
```

这是 state estimation 输出的 ROS/body IMU 语义 topic。它保留 IMU 的角速度
和线加速度，但 orientation 中的 yaw 已经替换成 fusion 后的 yaw。进入
controller/RL 前由 `controller_state_adapter` 转成 controller body 坐标；
不要把这个 raw topic 直接当成 controller topic：

```text
roll/pitch:
  IMU 原始姿态中的 roll/pitch。

yaw:
  fusion yaw，不是直接使用 IMU yaw。

angular_velocity:
  MCU/IMU 回传的原始角速度，单位 rad/s。
  转 controller 时使用 A = det(B) * B，而不是普通向量的 B。

linear_acceleration:
  MCU/IMU 回传的原始线加速度，单位 m/s^2。
```

这样做的目的：控制器仍然从 IMU topic 获得姿态和角速度，但不会直接继承 IMU yaw 的长期漂移。

### 9.3 `/finsrov/depth_link`

```text
frame_id = finsrov_depth_link
pose.position.z = fused body z in pool_world
```

注意它不是 raw pressure depth。

区别：

```text
/finsrov/hardware/depth_raw:
  水压计原始深度，水面向下为正。

/finsrov/depth_link:
  fusion 后的 body 原点 z，高度向上为正，属于 pool_world 语义。
```

如果潜器下潜：

```text
hardware depth_raw 增大
/finsrov/depth_link.pose.position.z 减小
```

### 9.4 `/finsrov/dvl_link`

当前没有真实 DVL 时，发布伪 DVL：

```text
v_world = [v_x, v_y, v_z] from PositionVelocityEKF
v_body = R_world_body^T * v_world
```

输出：

```text
frame_id = finsrov_base_link
twist.linear = v_body
```

这里的 DVL 是接口语义，不代表当前一定有真实 DVL 硬件。当前速度来源是 PositionVelocityEKF：

```text
视觉 x/y 更新 + 常速度模型 -> vx/vy
depth z 更新 + 常速度模型 -> vz
```

然后把 world velocity 转到 body frame：

```text
v_body = R_world_body^T * v_world
```

控制器/RL 侧可以像使用 DVL 一样读取 body-frame 线速度：

```text
twist.linear.x: body forward velocity
twist.linear.y: body left velocity
twist.linear.z: body up velocity
```

限制：

```text
1. 它是由位姿差分/EKF 推出的伪速度，不是真实水体相对速度。
2. 视觉丢失时 x/y 速度会进入 coast/hold 逻辑。
3. 如果视觉频率低或外参错误，伪 DVL 速度也会受影响。
```

## 10. 当前实现和原始计划的差异

| 原始计划 | 当前实现 |
| --- | --- |
| constrained 6D 默认控制输入 | 已实现：`/finsrov/vision/refracted_pose_6d` |
| constrained 优化 `[x,y,yaw]` | 已实现 |
| z 由 depth 强约束 | 已实现，C++ measurement 和 Python EKF 都使用 depth |
| roll/pitch 由 IMU 强约束 | 已实现 |
| yaw 由视觉修正 IMU 漂移 | 已实现，`YawEKF` 用 gyro predict + vision yaw update |
| 多 tag 融合 | 主链路为角点级联合优化 |
| pure visual Snell 6D | 未实现；当前 `refracted_pose_6d_pure` 是 pinhole multi-tag baseline |
| Ceres Solver | 未引入；当前是手写小规模 LM |

## 11. 常用检查

检查折射视觉 measurement：

```bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted_pose_6d --once --full-length
```

检查 Python fusion：

```bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/pose
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/pose --once --full-length
```

典型异常解释：

```text
missing_depth_or_imu:
  refractive 节点还没收到 depth 或 IMU。

stale_depth_or_imu:
  depth/IMU 时间戳距离 detection 太远。

geometry_error:
  T_body_tag、T_world_camera、marker_length_m、相机内参或水面模型可能不一致。

vision_6d_mahalanobis_gate:
  Python EKF 认为视觉 measurement 相对当前状态跳变过大。

hold:
  视觉已经长时间丢失，fusion 固定 x/y 并衰减速度。
```
