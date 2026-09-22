# FinsROV Fossen 6DOF 标定基线

## 适用范围

本文记录 `FinsROV_HydrodynamicsProfile.asset` 的当前 Fossen 6DOF 基线。它用于 Unity 场景 `VariousVehiclesNewHydrodynamics`：该场景的 `HydrodynamicsController` 为 `Fossen6Dof`，并会禁用 DWP2 的参数化水动力，不能与 DWP2 的 `WaterObject` 系数混用。

坐标和参数顺序固定为 FinsSim controller body：

```text
[u, v, w, p, q, r]
= [surge_x, sway_z, heave_y, roll_x, pitch_z, yaw_y]
= [x forward, z left, y up, roll about x, pitch about z, yaw about y]
```

阻尼模型为：

```text
tau_D,i = -(d1_i + d2_i * abs(nu_i)) * nu_i
```

Unity `Rigidbody` 负责刚体质量、刚体惯量和重力；profile 的 `addedMassDiagonal` 仅表示附加质量/附加转动惯量。完整非对角 added-mass 矩阵当前关闭并保持为零。

## 最终写入值

| 项 | u / surge | v / sway | w / heave | p / roll | q / pitch | r / yaw |
|---|---:|---:|---:|---:|---:|---:|
| `linearDamping` | 39.0736 | 52.3070 | 80.0968 | 1.0 | 1.0 | 0.3003 |
| `quadraticDamping` | 0.0 | 0.0 | 100.0 | 2.0 | 2.0 | 0.4776 |
| `addedMassDiagonal` | 15.0347 | 28.5059 | 23.3544 | 0.10 | 0.10 | 0.5904 |

线性/二次平动阻尼的单位分别为 `N/(m/s)`、`N/(m/s)^2`；角阻尼的单位分别为 `N*m/(rad/s)`、`N*m/(rad/s)^2`。平动 added mass 单位为 `kg`，角向 added inertia 单位为 `kg*m^2`。

## 实测来源与取值规则

| 自由度 | 最新数据 | 实测 m_eff | 实测 d1 | 实测 d2 | profile 处理 |
|---|---|---:|---:|---:|---|
| surge | `surge_x_015` | 27.1447 kg | 39.0736 | 0 | 采用；二次项命中非负下界，设 0 |
| sway | `sway_z_004` | 40.6159 kg | 52.3070 | 0 | 采用；二次项命中非负下界，设 0 |
| heave | `heave_y_015` | 35.4644 kg | 80.0968 | 483.0870 | 采用 `m_eff`、`d1`；`d2` 仅在约 0.01-0.02 m/s 上浮速度内辨识，不稳定，保守维持 100 |
| yaw | `yaw_y_007` | 0.8369 kg*m^2 | 0.3003 | 0.4776 | 采用 |
| roll | `roll_x_009` | 0.0650 kg*m^2 | 0 | 0 | 不采用：试验达到约 45 deg / 1.66 rad/s，非小角度且拟合全部命中下界；使用 RL 稳定基线 |
| pitch | `pitch_z_003` | 0.6735 kg*m^2 | 0.1974 | 0.7303 | 不采用：最大约 76 deg / 1.28 rad/s，超出小角度 Fossen 标定范围；使用 RL 稳定基线 |

数据目录：

```text
ros2_ws/data/hydrodynamics/real/hydrodynamic_identification/
```

其中所有正式平动/yaw 结果均使用同步后的 `HardwareTelemetry` 时间戳和 DVL 状态时间；拟合已剔除局部加速度脉冲和 MAD 异常点。

## Added Mass 拆分

本 profile 对应校准 prefab 的刚体物理基线：

```text
Rigidbody mass = 12.11 kg
Unity local inertia = (Ix, Iy, Iz)
                      = (0.131548, 0.246513, 0.231859) kg*m^2
```

因此：

```text
Ma_u = 27.144699 - 12.11     = 15.034699 kg
Ma_v = 40.615948 - 12.11     = 28.505948 kg
Ma_w = 35.464355 - 12.11     = 23.354355 kg
Ma_r =  0.836910 - Iy        =  0.590397 kg*m^2
```

FinsSim 的 yaw `r` 映射 Unity local `y`，故这里使用 `Iy`。若之后改变 `Rigidbody.mass` 或 inertia tensor，必须按同一公式重新计算 added mass/inertia，不能保留当前数值。

## 未写入 profile 的量

- 试验 bias：它混合了静态浮力失配、轻微水流、推进器曲线误差与传感器偏置，不作为阻尼项写入。
- pitch 试验的 restoring stiffness：profile 没有独立刚度字段；姿态恢复由 `centerOfMass`、`centerOfBuoyancy`、重力和浮力产生。
- 非对角 added mass、交叉阻尼、完整 Coriolis 项：没有可靠的多轴辨识数据，保持零/禁用。

## RL 使用建议

roll/pitch 的 `d1=(1,1)`、`d2=(2,2)`、`Ma=(0.1,0.1)` 是刻意保守的 RL 稳定基线，而非实测真值。训练时优先通过 domain randomization 扰动这两轴的阻尼和 CoM/CoB，而不是将大角度试验的拟合值直接当作物理参数。对 heave 的二次阻尼也应围绕 100 做随机化，待获得更大上浮速度范围的数据后再更新。
