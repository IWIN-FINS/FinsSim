# FinsROV 六自由度 Wrench 限幅计算工具

本目录中的 `compute_finsrov_wrench_limits.py` 用于根据八个推进器的最大实际推力和推进器几何布局，计算潜器可以实现的六自由度纯轴向 wrench 上限。

脚本输出的是物理可达 wrench 边界，用于审计推进器布局、估计饱和和设计实验；它不能直接写入 PPO-wrench 的 `action_gains`。后者是无量纲的矩阵输入增益。

## 1. 动作和坐标顺序

PPO 的归一化动作顺序为：

```text
[surge, sway, heave, roll, pitch, yaw]
```

脚本内部计算的 Unity/FinsROV 机体系 wrench 顺序为：

```text
[Fx, Fy, Fz, Mx, My, Mz]
```

坐标约定：

```text
x：向前
y：向上
z：向左
```

脚本最后还会输出用户要求的顺序：

```text
[Fx_max, Fz_max, Fy_max, Mx_max, Mz_max, My_max]
```

这个顺序对应：

```text
[surge, sway, heave, roll, pitch, yaw]
```

## 2. 参数含义

### `thruster_force_limits_n`

这是每个推进器的实际推力边界，单位是 N：

```yaml
positive: [F1+, F2+, ..., F8+]
negative: [F1-, F2-, ..., F8-]
```

其中 `negative` 使用正数表示反向推力的绝对值。例如，当前 Unity `FinsROV_Fossen`
仿真推进器是双向 `7 N`：

```yaml
positive: [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
negative: [7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]
```

它不是推进器曲线系数 `c1`。

三者区别如下：

```text
c1                     RPM 到推力的曲线系数
force_n                ROS 命令的单位/模式，表示消息数值单位为 N
thruster_force_limits_n 推进器允许接收的最大正向/反向推力，单位 N
```

如果使用二次推力曲线，可以通过以下公式估计最大推力：

```text
omega_max = rpm_max * 2*pi / 60
F_max     = abs(c1) * omega_max^2
```

优先使用推力台实测或已经验证的推力曲线端点。

### `L`

`L` 是整个八推进器系统的六自由度 wrench 控制范围，不是单个推进器的最大推力。

脚本对每个自由度分别求解：

```text
只产生目标自由度的力或力矩
其他五个 wrench 分量约束为 0
每个推进器不能超过正向/反向推力边界
```

因此计算出的结果不会把一个 surge 能力误认为同时包含 yaw 力矩的能力。

## 3. 推进器顺序

几何文件和推力边界必须使用统一顺序：

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

含义为：

```text
V_LF：Vertical Left Front
V_LB：Vertical Left Back
V_RB：Vertical Right Back
V_RF：Vertical Right Front
H_LF：Horizontal Left Front
H_LB：Horizontal Left Back
H_RB：Horizontal Right Back
H_RF：Horizontal Right Front
```

## 4. 几何配置

默认几何配置位于：

```text
tools/finsrov_wrench_geometry.yaml
```

每个推进器需要配置：

```yaml
position: [x, y, z]
direction: [dx, dy, dz]
```

单位分别为：

```text
position：m
direction：无量纲方向向量
```

脚本会自动将方向向量归一化，并根据以下公式构造 wrench 矩阵：

```text
B_i = [d_i, r_i × d_i]
```

其中：

```text
r_i：推进器相对于 reference_point 的位置
d_i：推进器正向 force_N 的机体系方向
```

`reference_point` 应设置为计算力矩时使用的参考点。最终建议设置为 Unity `Rigidbody.centerOfMass`：

```yaml
reference_point: [com_x, com_y, com_z]
```

如果 Unity scene 对 `centerOfMass` 有 override，不能继续使用几何文件中的默认原点。

## 5. 使用示例

### 5.1 统一的仿真推进器推力上限

假设八个推进器都使用：

```text
正向最大推力：7.0 N
反向最大推力：7.0 N
```

运行：

```bash
cd .

uv run --project python/finssim_rl \
  python tools/compute_finsrov_wrench_limits.py \
  --geometry tools/finsrov_wrench_geometry.yaml \
  --positive 7 7 7 7 7 7 7 7 \
  --negative 7 7 7 7 7 7 7 7 \
  --safety-factor 0.7
```

### 5.2 使用实机每个推进器不同的推力上限

```bash
uv run --project python/finssim_rl \
  python tools/compute_finsrov_wrench_limits.py \
  --geometry tools/finsrov_wrench_geometry.yaml \
  --positive 8.4749 7.3809 7.3809 8.4749 7.3809 8.4749 7.3809 8.4749 \
  --negative 7.9750 5.7618 5.7618 7.9750 5.7618 7.9750 5.7618 7.9750 \
  --safety-factor 0.7
```

### 5.3 计算理论最大值

如果不想使用安全系数：

```bash
--safety-factor 1.0
```

实际训练不建议直接使用理论最大值，因为推进器会频繁饱和，且没有控制余量。建议先使用 `0.6` 到 `0.8`，再根据饱和率调整。

## 6. 输出说明

脚本会输出：

```text
Positive pure-axis capability
Negative pure-axis capability
Symmetric capability before safety factor
L in [Fx,Fy,Fz,Mx,My,Mz] order
L in requested [Fx,Fz,Fy,Mx,Mz,My] order
```

例如统一推力上限、`safety-factor=0.7` 时，输出为：

```text
L in [Fx,Fy,Fz,Mx,My,Mz] order:
[13.669969, 15.751320, 12.890519, 2.704797, 4.956583, 2.155989]

L in requested [Fx,Fz,Fy,Mx,Mz,My] order:
[13.669969, 12.890519, 15.751320, 2.704797, 2.155989, 4.956583]
```

对于当前 PPO-wrench policy，动作顺序为：

```yaml
action_gains: [surge, sway, heave, roll, pitch, yaw]
```

`action_gains` 无单位，不能把脚本输出的 `[Fx,Fy,Fz,Mx,My,Mz]` 物理上限直接复制进去。

## 7. 结果验证

计算结果不能替代 Unity 和实机验证。建议对每个自由度发送：

```text
action = ±0.25
action = ±0.50
action = ±0.75
action = ±1.00
```

并记录：

```text
目标 wrench
8 个推进器 force_N
实际合成 wrench = B @ force_N
非目标自由度残差
推进器饱和率
速度或角速度响应
```

重点检查：

1. surge 不应产生明显 yaw torque；
2. sway 不应产生明显 yaw torque；
3. pitch/roll 的正负方向应与 Unity 和实机一致；
4. Unity 和实机使用相同 force_N 时，推进器目标力应一致；
5. `reference_point` 必须与实际 Rigidbody 质心一致。

如果结果显示某个平动动作伴随明显 yaw，优先检查推进器位置、方向、canonical order 和 motor sign，不要先修改 `action_gains`。
