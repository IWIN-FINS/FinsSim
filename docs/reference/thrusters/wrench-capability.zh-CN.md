# FinsROV 六自由度 Wrench 纯轴能力

本文记录当前 FinsROV 八推进器在两套推力边界下的理论纯轴 wrench 能力：

1. Unity 仿真：`FinsROV_Fossen.prefab` 八个推进器的对称 `+/-7 N` 物理限制。
2. 实机 V4 Pro1：`ppo_wrench_for_pose.yaml` 与 hardware bridge 曲线端点对应的逐路正、反向限制。

这不是速度上限。速度还取决于质量、附加质量、水动力、浮力、姿态和场景边界；本文件只说明推进器在不引入其余五个 wrench 分量时能合成的力/力矩范围。

## 坐标、顺序与求解定义

机体系 wrench 顺序为：

```text
[Fx, Fy, Fz, Mx, My, Mz]
```

其中 `x=前/surge`、`y=上/heave`、`z=左/sway`；旋转分别绕 `x/y/z` 轴。
推进器顺序固定为：

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

计算使用 [compute_finsrov_wrench_limits.py](../../../tools/compute_finsrov_wrench_limits.py)
和 [finsrov_wrench_geometry.yaml](../../../tools/finsrov_wrench_geometry.yaml)。对每个轴分别求解线性规划：

```text
max/min B @ force_N 的目标分量
subject to 其余五个 wrench 分量 = 0
           -negative[i] <= force_N[i] <= positive[i]
```

因此表中的值是纯轴能力，不是把非目标力矩也算进去后的合力。当前仿真几何文件以
`FinsROV_Fossen.prefab` 的 `Rigidbody.centerOfMass=[0,-0.08,0] m` 为力矩参考点，
与 Unity 的 `AddForceAtPosition` 一致。实船或其他 prefab 使用前，必须将该
reference point 改成实际质心后重算。

## 输入推力边界

| 场景 | 正向最大力 `N` | 反向最大力绝对值 `N` |
|---|---|---|
| 仿真 | `[7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]` | `[7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0, 7.0]` |
| 实机 V4 Pro1 | `[8.4749, 7.3809, 7.3809, 8.4749, 7.3809, 8.4749, 7.3809, 8.4749]` | `[7.9750, 5.7618, 5.7618, 7.9750, 5.7618, 7.9750, 5.7618, 7.9750]` |

实机数组来自 [ppo_wrench_for_pose.yaml](../../../ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose.yaml)
的 `thruster_force_limits_n`，其来源与说明见
[硬件 Bridge 推进器映射说明](../../operations/hardware/thruster-bridge-mapping.zh-CN.md)。
仿真数组来自 `FinsROV_Fossen.prefab` 的八个 `MaxForwardForceN: 7`、
`MaxReverseForceN: -7`，并与
[ppo_wrench_for_pose_sim.yaml](../../../ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose_sim.yaml)
保持一致。

## 满推理论能力

下表的 `+` 表示沿该轴正方向的最大纯轴能力；`-` 列记录沿负方向的最大幅值。
`对称可用值` 为两者较小者，可用作正负动作共用的物理上界。计算安全系数为 `1.0`。

| Wrench | 单位 | 仿真 `+` | 仿真 `-` | 仿真对称可用值 | 实机 `+` | 实机 `-` | 实机对称可用值 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `Fx` surge | N | 19.528527 | 19.528527 | 19.528527 | 18.586585 | 18.586585 | 18.586585 |
| `Fy` heave | N | 22.501886 | 22.501886 | 22.501886 | 27.243033 | 25.636077 | 25.636077 |
| `Fz` sway | N | 18.415027 | 18.415027 | 18.415027 | 18.068824 | 18.586585 | 18.068824 |
| `Mx` roll | N*m | 3.863995 | 3.863995 | 3.863995 | 3.791354 | 3.791353 | 3.791353 |
| `My` yaw | N*m | 7.080833 | 7.080833 | 7.080833 | 8.019448 | 6.947713 | 6.947713 |
| `Mz` pitch | N*m | 3.079985 | 3.079985 | 3.079985 | 2.535180 | 3.247581 | 2.535180 |

实机的方向不对称来自各通道的正/反向 RPM 曲线端点，不应被仿真的对称限幅掩盖。

## PPO 动作顺序

PPO wrench meta-action 的物理顺序是：

```text
[surge, sway, heave, roll, pitch, yaw]
=> [Fx, Fz, Fy, Mx, Mz, My]
```

满推理论对称可用值按该顺序重排后为：

```text
simulation: [19.528527, 18.415027, 22.501886, 3.863995, 3.079985, 7.080833]
hardware:   [18.586585, 18.068824, 25.636077, 3.791353, 2.535180, 6.947713]
```

为保留 `30%` 控制余量，将上面每项乘以 `0.7`：

```text
simulation: [13.669969, 12.890519, 15.751320, 2.704797, 2.155989, 4.956583]
hardware:   [13.010609, 12.648177, 17.945254, 2.653947, 1.774626, 4.863399]
```

这些是根据推进器可达集导出的物理能力边界。历史 `matrix` 模式的
`wrench6d.action_gains` 是无量纲增益，不能把本表直接写入其中。新的
`physical_wrench_allocator` 模式则明确使用 `wrench6d.wrench_limits`，其单位和顺序就是本表的
PPO 动作顺序；它会先产生目标 wrench，再在每路推力边界内求解 `B @ force_n`。
两种模式的 checkpoint 不兼容，物理模式必须从头训练。

当前物理模式训练入口为
[ppo_wrench_for_pose_fossen_physical_wrench_allocator.yaml](../../../configs/rl/pose_control/ppo_wrench_for_pose_fossen_physical_wrench_allocator.yaml)，
训练后 ROS2 仿真部署使用
[ppo_wrench_for_pose_physical_wrench_allocator_sim.yaml](../../../ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator_sim.yaml)。
实机部署使用独立的
[ppo_wrench_for_pose_physical_wrench_allocator.yaml](../../../ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator.yaml)。
它以实机逐推进器推力边界和本表的实机对称可用值进行分配；不能将历史
`ppo_wrench_for_pose.yaml` 的 checkpoint 改接到此配置，因为该 checkpoint 的动作
是在 `empirical_thruster_mixer` 契约下训练的。

## 复现命令

在仓库根目录运行：

```bash
# Unity 仿真：FinsROV_Fossen prefab 的全部推进器双向 7 N。
uv run --project python/finssim_rl python tools/compute_finsrov_wrench_limits.py \
  --geometry tools/finsrov_wrench_geometry.yaml \
  --positive 7 7 7 7 7 7 7 7 \
  --negative 7 7 7 7 7 7 7 7 \
  --safety-factor 1.0

# 实机 V4 Pro1：使用当前 curve/RPM 边界导出的逐路 force_N 限制。
uv run --project python/finssim_rl python tools/compute_finsrov_wrench_limits.py \
  --geometry tools/finsrov_wrench_geometry.yaml \
  --positive 8.4749 7.3809 7.3809 8.4749 7.3809 8.4749 7.3809 8.4749 \
  --negative 7.9750 5.7618 5.7618 7.9750 5.7618 7.9750 5.7618 7.9750 \
  --safety-factor 1.0
```

实际控制验证仍应记录 `B @ force_N` 的非目标轴残差和推进器饱和率。仿真中还应使用
ROS2/gRPC 实测速度响应；实船则必须在安全限幅和单轴点动验证后再提高到本表边界。
当前 `physical_wrench_allocator` 的 Unity Play 实测速度、完整链路和复跑方法见
[physical-wrench-speed_20260816.zh-CN.md](../../experiments/hardware/physical-wrench-speed_20260816.zh-CN.md)。
