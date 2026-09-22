# FinsROV\_AprilTag水下定位系统

# 基于水面折射 AprilTag 与状态融合的 FinsROV 水池定位系统说明

## 摘要

本文总结 FinsROV 在水池实验中的位置定位系统与潜器状态融合算法。系统采用固定俯视相机观测潜器顶部 AprilTag 刚性板，并结合水压计和 IMU，实现潜器在 `pool_world` 坐标系下的位姿估计。由于水面折射、顶部 AprilTag 共面、俯视视角以及水下图像质量等因素，系统没有使用纯视觉直接求解完整 6D 位姿，而是采用约束式 6D measurement：视觉主要估计水平位置和 yaw，水压计强约束深度，IMU 强约束 roll/pitch，最后由 Python 状态融合节点输出控制器和强化学习可用状态。该主链路在原始材料中明确描述为：\(x/y/yaw\) 由 AprilTag 多角点、Snell 折射和优化得到，\(z\) 由水压计强约束，roll/pitch 由 IMU quaternion 强约束，速度由 PositionVelocityEKF 估计。

## 关键词

FinsROV；AprilTag；水面折射；Snell 定律；水下定位；状态融合；EKF；水压计；IMU

---

# 系统总体结构

## 1\.1 系统目标

系统目标是在水面上方固定相机、潜器位于水下的条件下，估计潜器在 `pool_world` 中的位姿。位姿可以写为：

$T_{wb}=\left[R_{wb},,t_{wb}\right]$

其中，下标 \(w\) 表示 `pool_world`，下标 \(b\) 表示潜器 body frame。

位置向量为：

$t_{wb}=
\begin{bmatrix}
x\
y\
z
\end{bmatrix}$

姿态由 roll、pitch、yaw 表示：

$\phi=\operatorname{roll},\qquad
\theta=\operatorname{pitch},\qquad
\psi=\operatorname{yaw}$

完整 6D 状态为：

$X_{\mathrm{6D}}=
\begin{bmatrix}
x&
y&
z&
\phi&
\theta&
\psi
\end{bmatrix}^{T}$

但当前主算法并不直接优化完整 6D 状态，而是优化水平位置和 yaw，并由水压计与 IMU 补全其余自由度。原始材料中说明，当前控制主链路采用约束式 6D measurement：\(x,y,\\psi\) 由 AprilTag 多角点和 Snell 折射几何优化得到，\(z\) 由水压计深度强约束，roll/pitch 由 IMU quaternion 强约束。

---

## 1\.2 主链路

系统主链路如下：

```Plain Text
固定俯视相机
  ↓
direct_apriltag_node
  ↓
/finsrov/vision/tag_detections_2d
  ↓
refractive_apriltag_pose_node
  ↑                         ↑
/finsrov/hardware/depth_raw  /finsrov/hardware/imu_raw
  ↓
/finsrov/vision/refracted_pose_6d
  ↓
state_fusion_node.py
  ↓
/finsrov/pose
/finsrov/imu_link
/finsrov/depth_link
/finsrov/dvl_link
/finsrov/state/status
```

其中，`direct_apriltag_node` 负责从图像中检测 AprilTag 的 ID 和角点像素；`refractive_apriltag_pose_node` 负责结合折射模型、水压计和 IMU 解算视觉 6D measurement；`state_fusion_node.py` 负责 EKF 状态融合。原始材料中的流程图也明确给出了该链路。

---

# 坐标系与符号约定

## 2\.1 坐标系

本文使用三个主要坐标系：

`pool_world` 中，\(x\) 为池长方向，\(y\) 为水平横向，\(z\) 向上。水池底部参考平面为 \(z=0\)，水面高度记为：

$z=z_w$

潜器 body frame 使用 ROS FLU 约定，即 body x forward，body y left，body z up。原始材料中也明确给出了 `pool_world`、`finsrov_base_link` 和 `tag frame` 的定义。

---

## 2\.2 工程变量到数学符号的映射

为保证飞书公式导入稳定，本文公式中不直接使用工程字段名，而采用下表数学符号。

---

# AprilTag 刚体观测模型

## 3\.1 AprilTag detector 输出

相机本身不直接输出潜器 3D 位姿。当前链路先由 AprilTag 3 detector 在图像中识别 tag，输出 tag ID、family、decision margin、中心像素、角点像素和边长像素等信息。材料中强调，AprilTag detector 输出的是 2D 像素角点，不是潜器 3D pose；后续节点才根据 tag ID、角点像素、tag 尺寸、相机内外参、Snell 模型、depth 和 IMU 解算潜器 pose。

设第 \(j\) 个 tag 的第 \(k\) 个角点像素为：

$u_{jk}=
\begin{bmatrix}
u_{jk}\
v_{jk}
\end{bmatrix}$

---

## 3\.2 Tag 角点局部坐标

设 AprilTag 编码区域边长为 \(L\)，则：

$h=\frac{L}{2}$

四个角点在 tag 坐标系下定义为：

$q_0=
\begin{bmatrix}
-h\
h\
0
\end{bmatrix},
\quad
q_1=
\begin{bmatrix}
h\
h\
0
\end{bmatrix},
\quad
q_2=
\begin{bmatrix}
h\
-h\
0
\end{bmatrix},
\quad
q_3=
\begin{bmatrix}
-h\
-h\
0
\end{bmatrix}$

第 \(j\) 个 AprilTag 到潜器 body frame 的安装外参记为：

$T_{bt_j}$

则第 \(j\) 个 tag 的第 \(k\) 个角点在 body frame 下的位置为：

$P_{b,jk}=T_{bt_j}q_k$

如果一帧中检测到多个 tag，则所有 tag 的所有角点共同组成刚体观测集合：

$\mathcal{S} = \left\{ (j,k,P_{b,jk},u_{jk}) \right\} $

该集合是多 tag 融合的基础。所有 tag 不是独立目标，而是同一个潜器刚体上的不同 3D 点。

---

# 深度与 IMU 约束

## 4\.1 水压计深度约束

水压计深度 \(d\) 定义为水面向下为正。由于 `pool_world` 为 z\-up，压力传感器在世界系中的高度为：

$z_p=z_w+s_d d$

其中 \(s\_d\) 为深度方向符号。当前常用配置中：

$s_d=-1$

若水压计相对 body 原点的 z 偏移为 \(z\_o\)，则当前实现使用 IMU roll/pitch 对垂向投影进行修正：

$z_b=z_p-z_o\cos\phi_{\mathrm{imu}}\cos\theta_{\mathrm{imu}}$

最终，\(z\_b\) 作为潜器 body 原点在 `pool_world` 中的高度。材料中给出的工程公式与该表达一致。

---

## 4\.2 IMU 姿态约束

IMU quaternion 提供：

$\phi_{\mathrm{imu}},\quad
\theta_{\mathrm{imu}},\quad
\psi_{\mathrm{imu}}$

当前 constrained 视觉解算中采用：

$\phi=\phi_{\mathrm{imu}}$

$\theta=\theta_{\mathrm{imu}}$

但不使用 \(\\psi\_\{\\mathrm\{imu\}\}\) 作为 `pool_world` 下的绝对 yaw。原因是 IMU yaw 的零点不一定和 `pool_world` 对齐，并且 yaw 积分会漂移。材料中也明确说明，系统使用 IMU angular velocity\.z 做 yaw 高频预测，使用视觉 yaw 做低频绝对修正，而不使用 IMU quaternion 自带 yaw 作为绝对 yaw 观测。

---

# 水面折射成像模型

## 5\.1 水面平面

设水面平面为：

$\Pi_w:\quad n_{\Pi}^{T}X+b_{\Pi}=0$

当前第一版通常采用水平水面假设：

$n_{\Pi}=
\begin{bmatrix}
0\
0\
1
\end{bmatrix}$

若水面高度为 \(z\_w\)，则对应水面为：

$z=z_w$

该假设适用于水面较平稳的情况。若水面波动明显，实际折射路径会偏离模型，可能导致 residual 增大或被 gate 拒绝。

---

## 5\.2 像素反投影为空气 ray

相机内参矩阵为：

$K=
\begin{bmatrix}
f_x&s&c_x\
0&f_y&c_y\
0&0&1
\end{bmatrix}$

畸变参数记为：

$D=
\begin{bmatrix}
k_1&
k_2&
p_1&
p_2&
k_3
\end{bmatrix}$

给定角点像素齐次坐标： 

$
\tilde{u}_{jk}=
\begin{bmatrix}
u_{jk}\
v_{jk}\
1
\end{bmatrix}
$

去畸变后，像素反投影得到相机坐标系下的单位视线： 

$r_c=
\frac{K^{-1}\tilde{u}_{jk}}
{\left|K^{-1}\tilde{u}_{jk}\right|_2}$

设相机到 `pool_world` 的外参为：

$T_{wc}=
\left[
R_{wc},t_{wc}
\right]$

则相机中心和空气中视线方向为：

$C_w=t_{wc}$

$r_a=R_{wc}r_c$

空气中 ray 为：

$X_a(s)=C_w+s r_a$

材料中说明，\(K,D\) 的作用是将 AprilTag 角点像素变成相机坐标系下的单位视线；若 \(K,D\) 与当前相机分辨率、焦距或裁剪方式不一致，Snell 折射和 PnP 都会系统性偏移。

---

## 5\.3 空气 ray 与水面求交

将空气 ray 代入水面平面：



$ n_{\Pi}^{T} \left( C_w+s r_a \right) + b_{\Pi} = 0$



$求得交点参数：$

$s^{*}=
-\frac{n_{\Pi}^{T}C_w+b_{\Pi}}
{n_{\Pi}^{T}r_a}$



$水面入射点为：$

$S=C_w+s^{*}r_a
$



$若分母 (n_{\Pi}^{T}r_a) 接近 0，则表示 ray 与水面近似平行，应拒绝该角点或设置为大残差。材料中也明确提到，若分母接近 0，应 reject。$

---

## 5\.4 Snell 折射

空气和水的折射率分别记为：

$n_{\mathrm{air}},\qquad n_{\mathrm{water}}$



Snell 定律为：

$n_{\mathrm{air}}\sin\alpha_{\mathrm{air}} = n_{\mathrm{water}}\sin\alpha_{\mathrm{water}}$


折射率比为：

$\eta=
\frac{n_{\mathrm{air}}}{n_{\mathrm{water}}}$



$给定空气中入射方向、水面法向量和折射率比，可计算水中折射方向：$

$ r_{\mathrm{water}} = \operatorname{refract} \left( r_a, n_{\Pi}, \eta \right)$



$水中 ray 为：$

$ X_{\mathrm{water}}(\tau) = S+\tau r_{\mathrm{water}} $

$这里用 (n_{\Pi}) 表示水面法向量，用 (n_{\mathrm{water}}) 表示水的折射率，避免符号混淆。材料中同样指出，空气入水时 (n_{\mathrm{water}}>n_{\mathrm{air}})，水中光线向法线偏折。$

---

## 5\.5 反投影到目标深度平面

定义竖直方向选择向量：

$e_3=
\begin{bmatrix}
0\
0\
1
\end{bmatrix}$

对于候选位姿下的角点世界坐标 \(P\_\{w,jk\}\)，目标深度平面为：

$z_{\mathrm{target}} = e_3^TP_{w,jk}$

$水中 ray 与该深度平面相交时：$

$\tau^{*} = \frac{ z_{\mathrm{target}}-e_3^TS }{ e_3^Tr_{\mathrm{water}} } $



$于是像素观测经过 Snell 折射模型反投影得到的水下世界点为：$

$ Q_{w,jk} = S+\tau^{*}r_{\mathrm{water}}$

$材料中该步骤被描述为：折射 ray 与目标深度平面相交，得到 (Q_{\mathrm{world},jk})。$



**为什么 Snell 模型不能直接确定潜器位置：**

Snell 模型可以把 一个像素观测 转换成 一条经过水面折射后的水下 ray。如果再给定一个目标深度平面 \`z=z\_target\`，就能求出这条 ray 在该平面上的交点：



```Plain Text
u -> refracted ray -> Q_world(z_target)
```



但潜器 pose 不是单个角点的位置。未知量包括：x, y, yaw



而每个角点的真实 world z 又依赖潜器姿态和该角点在 body 上的位置：

```Plain Text
P_world_jk(theta).z =
  [Rz(yaw) * Ry(pitch_imu) * Rx(roll_imu) * P_body_jk + t_world_body].z
```



也就是说：

要反投影像素，需要知道角点所在的 z 平面；
角点所在的 z 平面，又依赖当前候选 yaw 和 body 刚体布局。



此外，多角点之间必须满足刚体约束：同一个 theta 必须同时解释两个 tag 的所有角点。

所以单个 Snell 反投影只能给出“某个深度平面上的观测点”，不能直接给出唯一的潜器 `[x,y,yaw]`。优化问题的作用就是寻找一个潜器 pose，使所有角点的 Snell 反投影观测和刚体预测位置整体最一致。

---

# Constrained 6D 几何优化

## 6\.1 优化变量

当前系统不是直接优化完整 6D 状态，而是只优化：

$\boldsymbol{\xi}=
\begin{bmatrix}
x\
y\
\psi
\end{bmatrix}$

$其中 (\psi) 为 yaw。这里使用 (\boldsymbol{\xi}) 表示优化变量，避免与 pitch 符号 (\theta) 混淆。$

由水压计和 IMU 补全完整位姿：

$ t_{wb}(\boldsymbol{\xi}) = \begin{bmatrix} x,y, z_b \end{bmatrix}$

$ R_{wb}(\boldsymbol{\xi}) = R_z(\psi)R_y(\theta_{\mathrm{imu}})R_x(\phi_{\mathrm{imu}})$



$原始材料中说明，当前 constrained 解只优化 (x,y,yaw)，其它自由度由 pressure depth 和 IMU roll/pitch 强约束。$

---

## 6\.2 角点世界坐标预测

根据候选位姿，角点在 `pool_world` 中的预测位置为：

$ P_{w,jk}(\boldsymbol{\xi}) = R_{wb}(\boldsymbol{\xi})P_{b,jk} + t_{wb}(\boldsymbol{\xi})$

$这个点表示：如果潜器真的处于当前候选状态 (\boldsymbol{\xi})，则第 (j) 个 tag 的第 (k) 个角点理论上应该出现在世界坐标系中的位置。材料中给出的预测公式与该表达一致。$

---

## 6\.3 残差定义

同一个角点有两个世界位置：

$(P_{w,jk}(\boldsymbol{\xi}))：由潜器刚体模型预测的角点位置；$

$(Q_{w,jk}(\boldsymbol{\xi}))：由像素观测经过 Snell 折射反投影得到的角点位置$



定义残差：

$r_{jk}(\boldsymbol{\xi}) = Q_{w,jk}(\boldsymbol{\xi}) - P_{w,jk}(\boldsymbol{\xi})$

$其中 (Q_{w,jk}) 也依赖 (\boldsymbol{\xi})，因为它反投影到的目标深度平面由 (P_{w,jk}(\boldsymbol{\xi})) 的 z 坐标决定。因此，Snell 模型不是一次正向公式直接解出潜器位姿，而是一个非线性优化问题。$

---

## 6\.4 优化目标函数

对所有可见 tag 的所有角点联合优化：

$ \boldsymbol{\xi}^{*} = \arg\min_{\boldsymbol{\xi}} \sum_{j}\sum_{k=0}^{3} \left| r_{jk} (\boldsymbol{\xi}) \right|_2^2$

$展开为：$

$ \boldsymbol{\xi}^{*} = \arg\min_{x,y,\psi} \sum_{j}\sum_{k=0}^{3} \left| Q_{w,jk}(x,y,\psi) - P_{w,jk}(x,y,\psi) \right|_2^2$

该目标函数的物理意义是：寻找一个潜器整体位姿，使相机角点观测、Snell 折射几何、水压计深度、IMU roll/pitch、tag 在潜器上的刚体布局同时尽可能一致。材料中也将其解释为多传感器几何一致性误差。

Q:从相机像素 u\_jk 出发，经过 K,D 去畸变和反投影，

再经过空气/水界面的 Snell 折射，

最后落到当前预测角点深度平面上的“观测点。



P:根据假设的潜器位姿 theta、IMU roll/pitch、水压计 z、

以及已知的 T\_body\_tag 和 tag 尺寸，

由刚体模型推算出来的“该角点理论上应该在世界中的位置。



---

## 6\.5 Levenberg\-Marquardt 求解

将所有残差堆叠为：

$r(\boldsymbol{\xi})=
\begin{bmatrix}
r_1^T&
r_2^T&
\cdots&
r_N^T
\end{bmatrix}^{T}$

数值 Jacobian 为：

$J_r=
\frac{\partial r}{\partial \boldsymbol{\xi}}$

近似 Hessian 为：

$H=
J_r^TJ_r+\lambda I$

增量为：

$ \delta = H^{-1}J_r^T(-r) $



$状态更新为：$

$ \boldsymbol{\xi}_{\mathrm{next}} = \boldsymbol{\xi} + \delta$

$若下一步代价降低，则接受更新并减小 (\lambda)；否则拒绝更新并增大 (\lambda)。当前 C++ 实现使用的是手写小规模 LM，而不是 Ceres Solver。$

---

## 6\.6 视觉位姿输出

优化完成后：

$ x^{}=\xi_1^{} $ $ y^{}=\xi_2^{} $ $ \psi^{}=\xi_3^{} $



视觉位置 measurement 为：

$p_{\mathrm{vis}} = \begin{bmatrix} x^{}, y^{}, z_b \end{bmatrix} $



视觉姿态 measurement 为：

$q_{\mathrm{vis}} = \operatorname{quat} \left( \phi_{\mathrm{imu}}, \theta_{\mathrm{imu}}, \psi^{*} \right) $



$该结果发布到：$

```Plain Text
/finsrov/vision/refracted_pose_6d
```

$材料中也明确写到，优化完成后 position 为 ([x^,y^,z_{\mathrm{body}}])，orientation 为 IMU roll/pitch 和优化 yaw 组成的 quaternion。$

---

## 6\.7 几何误差拒绝

若几何残差相对 tag 尺寸过大，则拒绝该视觉 measurement：

$\frac{e_g}{L}>\gamma_g$

其中，\(e\_g\) 表示 constrained 几何残差，\(L\) 表示 tag 编码区域边长，\(\\gamma\_g\) 表示几何误差阈值。对应工程字段见附录。

常见拒绝原因：

```Plain Text
constrained_reject_reason = geometry_error
```

材料中说明，该阈值按 tag 尺寸归一化，即 constrained residual 与 marker length 的比值超过阈值时，触发 geometry error。

---

# 多 AprilTag 融合

## 7\.1 角点级联合优化

当前主链路中的多 tag 融合发生在 residual 层，而不是 pose 层。系统不会先对每个 tag 单独求一个 pose 再平均，而是把所有可见 tag 的角点一起放入同一个优化问题中。

观测集合为：

$\mathcal{S}
=
\bigcup_j
\left\{
(j,k,P_{b,jk},u_{jk})
\mid
k=0,1,2,3
\right\}$

$统一优化：$

$\arg\min_{\boldsymbol{\xi}}
\sum_{(j,k)\in\mathcal{S}}
\left|
r_{jk}(\boldsymbol{\xi})
\right|_2^2$

$若当前 tag 15 和 tag 16 同时可见，则目标函数可写为：$

$\boldsymbol{\xi}^{*} = \arg\min_{\boldsymbol{\xi}} \left( \sum_{k=0}^{3} \left| r_{15,k} (\boldsymbol{\xi}) \right|_2^2 + \sum_{k=0}^{3} \left| r_{16,k}(\boldsymbol{\xi}) \right|_2^2 \right)$

多 tag 的作用包括：

1. 增加角点数量，提高抗噪声能力；

2. 扩大刚体点分布范围，提高 yaw 约束；

3. 当某个 tag 角点质量差时，其他 tag 仍可提供约束；

4. 通过 residual 一致性暴露 tag 外参或 tag 尺寸错误。

材料中明确指出，当前 constrained Snell 主链路中的多 tag 不是“每个 tag 算一个 pose 然后平均”，而是角点级联合优化。

---

## 7\.2 与 PnP fallback 的区别

旧 PnP fallback 的逻辑是每个 tag 先得到一个候选位姿：

$T_{wb}^{(j)} = T_{wc}T_{ct_j}T_{bt_j}^{-1} $

若候选之间冲突较大，则选择 reprojection error 最低的 tag；若候选一致，则按 covariance 或 reprojection error 加权融合 position/yaw。

但当前默认主链路不是该 fallback。材料中明确给出当前默认：

```Plain Text
use_vision_pnp: false
```

并说明旧 PnP fallback 只是 baseline 或回退调试逻辑，不是实船默认主链路。

---

# 水面附近模式切换

系统不是在所有深度下都使用水下折射模型，而是根据深度分模式。定义模式变量：

$m(d)=
\begin{cases}
0, & d<d_a\
1, & d_a\le d<d_u\
2, & d\ge d_u
\end{cases}$

其中：

$d_a=0.02\ \mathrm{m}$

$d_u=0.08\ \mathrm{m}$

模式解释如下：

在 `underwater_refraction` 模式且 tag 足够时，系统发布 constrained `snell_depth_imu` 结果。在 `air` 或 `surface_transition` 模式下，如果配置允许，constrained topic 上可能发布 pinhole baseline 并用 pressure depth 替换 z。原始材料中给出了这三个模式和默认阈值。

---

# Python 状态融合

## 9\.1 PositionVelocityEKF

位置速度 EKF 状态为：

$x_k=
\begin{bmatrix}
p_x&
p_y&
p_z&
v_x&
v_y&
v_z
\end{bmatrix}^{T}$

采用常速度预测模型：

$x_k^{-}=Fx_{k-1}$

其中：

$F=
\begin{bmatrix}
1&0&0&\Delta t&0&0\
0&1&0&0&\Delta t&0\
0&0&1&0&0&\Delta t\
0&0&0&1&0&0\
0&0&0&0&1&0\
0&0&0&0&0&1
\end{bmatrix}$

协方差预测为：

$P_k^{-}=FP_{k-1}F^T+Q$

视觉 measurement 默认只更新 \(x,y\)：

$z_{\mathrm{vis}}=
\begin{bmatrix}
x_{\mathrm{vis}}\
y_{\mathrm{vis}}
\end{bmatrix}$

$H_{\mathrm{vis}}=
\begin{bmatrix}
1&0&0&0&0&0\
0&1&0&0&0&0
\end{bmatrix}$

深度 measurement 只更新 \(z\)：

$z_{\mathrm{dep}}=z_b$

$H_{\mathrm{dep}}=
\begin{bmatrix}
0&0&1&0&0&0
\end{bmatrix}$

标准 EKF 更新为：

$y_k=z_k-Hx_k^{-}$

$S_k=HP_k^{-}H^T+R$

$K_k=P_k^{-}H^TS_k^{-1}$

$x_k=x_k^{-}+K_ky_k$

$材料中明确指出，PositionVelocityEKF 的状态为 ([p_x,p_y,p_z,v_x,v_y,v_z]^T)，预测模型为常速度模型，视觉 6D 默认只更新 (x/y)，深度只更新 (z)。$

---

## 9\.2 YawEKF

YawEKF 状态只有 yaw：$\psi_k$



预测使用 IMU gyro：

$\psi_k^{-} = \operatorname{wrap} \left( \psi_{k-1} + \omega_z\Delta t \right) $

$P_{\psi,k}^{-} = P_{\psi,k-1} + q_{\psi}\Delta t $



$视觉 yaw 更新误差为：$

$e_{\psi} = \operatorname{wrap} \left( \psi_{\mathrm{vis}} - \psi_k^{-} \right) $



$innovation covariance 为：$

$S_{\psi} = P_{\psi,k}^{-} + R_{\psi} $



$Kalman gain 为：$

$K_{\psi} = \frac{P_{\psi,k}^{-}}{S_{\psi}} $



$yaw 更新为：$

$\psi_k = \operatorname{wrap} \left( \psi_k^{-} + K_{\psi}e_{\psi} \right) $



$协方差更新为：$

$P_{\psi,k} = (1-K_{\psi})P_{\psi,k}^{-} $



该设计表示：短期由 IMU gyro 提供平滑 yaw 预测，长期由视觉 yaw 把漂移拉回 `pool_world` 绝对方向。材料中明确说明，当前默认使用 vision yaw 更新，不使用 IMU quaternion yaw 作为绝对观测。

---

# 传感器分工

当前系统中各传感器的分工如下：

当前明确不做：

```Plain Text
AprilTag 修正 roll/pitch
IMU yaw 作为 pool_world 绝对 yaw
IMU acceleration 积分成速度/位置
```

材料中也明确说明，视觉默认不直接更新 z、roll、pitch。

---

# 异常观测处理

## 11\.1 Mahalanobis gate

对任意 measurement，innovation 为：

$y_k=z_k-Hx_k^{-}$

innovation covariance 为：

$S_k=HP_k^{-}H^T+R$

Mahalanobis 距离为：



$ d_M^2 = y_k^TS_k^{-1}y_k $ 



若： $ d_M^2>\gamma_m $

$则拒绝该次观测。
常见拒绝状态包括：$

```Plain Text
vision_6d_mahalanobis_gate
vision_6d_yaw_mahalanobis_gate
depth_mahalanobis_gate
```

该机制用于防止 AprilTag 重新出现时 pose 突跳、外参或 tag ID 错误、水面扰动导致视觉 measurement 离谱，以及 depth 瞬时异常。材料中给出的默认 Mahalanobis gate 阈值为 9\.0。

---

## 11\.2 视觉丢失状态机

视觉状态机包括：

其作用是避免潜器跑出相机视野或 tag 被遮挡后，EKF 继续依靠旧速度无限漂移。材料中说明，进入 `hold` 后 x/y 固定到最后一次有效视觉位置，x/y 速度衰减或归零。

---

# 输出语义

## 12\.1 `/finsrov/pose`

输出坐标系为 `pool_world`，位置来自 PositionVelocityEKF，姿态由 IMU roll/pitch 和融合 yaw 组成：

$p_{\mathrm{out}}=
\begin{bmatrix}
p_x\
p_y\
p_z
\end{bmatrix}$

$ q_{\mathrm{out}} = \operatorname{quat} \left( \phi_{\mathrm{imu}}, \theta_{\mathrm{imu}}, \psi_{\mathrm{fused}} \right) $ 

---

## 12\.2 `/finsrov/imu_link`

该 topic 中，orientation 同样使用融合 yaw，而 angular velocity 和 linear acceleration 原样来自 IMU。

---

## 12\.3 `/finsrov/depth_link`

该 topic 的 z 不是 raw pressure depth，而是融合链路中的 body z：

$z=z_b$

---

## 12\.4 `/finsrov/dvl_link`

当前没有真实 DVL 时，系统发布伪 DVL。世界系速度为：

$v_w=
\begin{bmatrix}
v_x\
v_y\
v_z
\end{bmatrix}$

body 系速度为：

$v_b=
R_{wb}^{T}v_w$

材料中明确说明，伪 DVL 的 `twist.linear` 为 body frame 下速度，并且视觉进入 `hold` 时 x/y velocity 会被置 0，避免错误速度继续给控制器。

---

# 完整实验流程

## 13\.1 实验前准备

1. 固定俯视相机，保证实验过程中相机不移动。

2. 检查相机内参 \(K,D\)。

3. 检查相机外参 \(T\_\{wc\}\)。

4. 检查 AprilTag 板是否牢固。

5. 检查每个 tag 的安装外参 \(T\_\{bt\_j\}\)。

6. 检查 tag 编码区域边长 \(L\)。

7. 测量当前水面高度 \(z\_w\)。

8. 检查水压计零点、深度方向和安装偏移 \(z\_o\)。

9. 检查 IMU quaternion 与 gyro 数据。

10. 检查 ROS topic 频率。

---

## 13\.2 节点启动顺序

```Plain Text
1. 启动相机
2. 启动 direct_apriltag_node
3. 启动硬件桥接节点，发布 depth_raw 和 imu_raw
4. 启动 refractive_apriltag_pose_node
5. 启动 state_fusion_node.py
6. 检查 /finsrov/vision/refracted/status
7. 检查 /finsrov/state/status
8. 检查 /finsrov/pose
```

---

## 13\.3 单帧视觉定位流程

```Plain Text
图像
  ↓
AprilTag detector
  ↓
tag_id + corner_pixels_xy
  ↓
根据 tag_id 查询 T_body_tag
  ↓
构造角点 P_b,jk
  ↓
读取 depth_raw，计算 z_b
  ↓
读取 IMU，得到 roll_imu / pitch_imu
  ↓
执行 Snell 折射反投影
  ↓
构造 residual r_jk
  ↓
LM 优化 ξ = [x, y, yaw]^T
  ↓
输出 /finsrov/vision/refracted_pose_6d
```

---

## 13\.4 状态融合流程

```Plain Text
收到 IMU:
  使用 gyro_z 预测 yaw
  使用 roll/pitch 组成最终 quaternion

收到 depth:
  更新 PositionVelocityEKF 的 pz

收到 vision_6d:
  更新 PositionVelocityEKF 的 px / py
  更新 YawEKF 的 yaw

周期输出:
  /finsrov/pose
  /finsrov/imu_link
  /finsrov/depth_link
  /finsrov/dvl_link
  /finsrov/state/status
```

---

# 常见异常与排查

## 14\.1 `missing_depth_or_imu`

含义：折射节点还没有收到 depth 或 IMU。

重点检查：

```Bash
ros2 topic hz /finsrov/hardware/depth_raw
ros2 topic hz /finsrov/hardware/imu_raw
```

---

## 14\.2 `stale_depth_or_imu`

含义：depth 或 IMU 时间戳距离当前 AprilTag detection 太远。

重点检查：

1. depth 时间戳；

2. IMU 时间戳；

3. camera image 时间戳；

4. ROS 节点时钟；

5. topic 发布频率。

---

## 14\.3 `geometry_error`

含义：Snell 几何优化后的 residual 过大。

可能原因：

1. tag 安装外参错误；

2. 相机外参错误；

3. tag 编码区域边长错误；

4. 相机内参或畸变参数错误；

5. 水面高度配置错误；

6. 水面波动明显；

7. AprilTag 角点检测质量差；

8. depth 或 IMU 与图像不同步。

---

## 14\.4 `vision_6d_mahalanobis_gate`

含义：Python EKF 认为视觉 measurement 相对当前预测状态跳变过大。

可能原因：

1. AprilTag 刚重新进入视野；

2. 水面扰动导致视觉异常；

3. tag ID 或 tag 外参错误；

4. EKF 当前状态已漂移；

5. measurement noise 或 process noise 参数不合适。

---

## 14\.5 `hold`

含义：视觉已经长时间丢失，fusion 固定 x/y 并衰减或归零 x/y 速度。

这不是 bug，而是防止视觉丢失后状态继续漂移的安全策略。材料中的典型异常解释也指出，`hold` 表示视觉已长时间丢失，fusion 固定 x/y 并衰减速度。

---

# 结论

当前 FinsROV 水池定位系统可以概括为：

$x,y,\psi
\leftarrow
\operatorname{LM}
\left(
\operatorname{AprilTag},
\operatorname{Snell},
d,
q_{\mathrm{imu}}
\right)$

$z
\leftarrow
\operatorname{Depth}$

$\phi,\theta
\leftarrow
\operatorname{IMU}$

$\psi_{\mathrm{fused}}
\leftarrow
\operatorname{YawEKF}
\left(
\omega_z,
\psi_{\mathrm{vis}}
\right)$

$p,v
\leftarrow
\operatorname{PositionVelocityEKF}
\left(
p_{\mathrm{vis}},
z_b
\right)$

该方案的核心是：视觉只承担它在当前几何条件下最可靠的部分，即 \(x,y,\\psi\)；水压计承担 \(z\)；IMU 承担 roll/pitch 和 yaw 高频预测；视觉 yaw 对 IMU yaw 漂移进行长期修正。这样可以避免纯视觉完整 6D 在水下俯视、共面 tag 和水面折射条件下的不稳定问题。当前实现已经完成 constrained 6D 默认控制输入、\(\[x,y,yaw\]\) 优化、depth 强约束 z、IMU 强约束 roll/pitch、视觉修正 yaw 漂移、多 tag 角点级联合优化、Python EKF 融合和视觉丢失状态机；但 pure visual Snell 6D 尚未实现，`refracted_pose_6d_pure` 当前仍是 pinhole multi\-tag baseline。

---

# 附录 A：工程入口

```Plain Text
ros2_ws/src/perception/src/refractive_apriltag_pose_node.cpp
ros2_ws/src/state_estimation/state_estimation/state_fusion_node.py
ros2_ws/src/state_estimation/config/state_fusion.yaml
```

---

# 附录 B：主要 ROS Topic

---

# 附录 C：参数表

## C\.1 水池与深度参数

对应公式：

$z_p=z_w+s_d d$

$z_b=z_p-z_o\cos\phi_{\mathrm{imu}}\cos\theta_{\mathrm{imu}}$

---

## C\.2 折射模型参数

对应公式：

$\eta=
\frac{n_{\mathrm{air}}}{n_{\mathrm{water}}}$

---

## C\.3 模式切换参数

对应公式：

$m(d)=
\begin{cases}
0, & d<d_a\
1, & d_a\le d<d_u\
2, & d\ge d_u
\end{cases}$

---

## C\.4 EKF 参数

---

## C\.5 视觉丢失状态机参数

---

## C\.6 默认融合开关

```YAML
vision_6d_topic: /finsrov/vision/refracted_pose_6d
use_vision_6d: true
use_vision_6d_z: false
use_vision_yaw: true

vision_pnp_topic: /finsrov/vision/tag_poses_3d_camera
use_vision_pnp: false

use_imu_yaw_measurement: false
```

---

# 附录 D：常用检查命令

## D\.1 检查折射视觉节点

```Bash
cd ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted_pose_6d --once --full-length
```

## D\.2 检查 Python fusion

```Bash
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status --once --full-length
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/pose
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/pose --once --full-length
```

## D\.3 检查输入频率

```Bash
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/hardware/imu_raw
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/hardware/depth_raw
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/vision/refracted_pose_6d
```

材料中也给出了这些检查命令，用于检查 refractive 节点、fusion 节点和输入 topic 频率。
