# Chase Body Frame 与 Observation 约定

本文是 FinsSim 中追逐类任务的顶层权威说明，适用于：

- `3Chase1`
- `1Chase1`
- 后续复用同类“追目标 / 协同追捕”接口的任务

## 1. 统一机体系

追逐类任务统一使用 `controller_body` 语义：

```text
body.x = forward
body.y = up
body.z = left
```

也就是说：

- `x > 0`：目标在前方
- `y > 0`：目标在上方
- `z > 0`：目标在左侧

这套约定与 `controller_world / controller_body` 文档保持一致，不再把 Unity prefab 的原始 local axis 直接当成接口定义。

## 2. 与 pool_world / controller_world 的关系

- `pool_world`：ROS 实船物理坐标，`x` 前、`y` 左、`z` 上
- `controller_world`：训练/控制坐标，`x` 前、`y` 上、`z` 左
- `controller_body`：机体系，`x` 前、`y` 上、`z` 左

追逐任务里的高层 observation 一律在 `controller_body` 下表达。普通 position controller 任务仍可走现有 `controller_world + controller_body` 组合接口；当前 `1Chase1` hierarchy 路径中，低层 PID 直接接收高层 subgoal 解码出的 `controller_body` 位置误差。

## 3. 3Chase1 高层 Observation

每个 Chaser 的高层 `actor_obs` 为 `30D`：

| 字段 | 维度 | 语义 |
| --- | --- | --- |
| `self_linear_velocity_body` | 3 | 自身体线速度，`controller_body` |
| `self_angular_velocity_body` | 3 | 自身体角速度，`controller_body` |
| `fish_relative_position_body` | 3 | prey 相对自身的位置，`controller_body` |
| `fish_relative_velocity_body` | 3 | prey 相对自身的速度，`controller_body` |
| `distance_to_fish` | 1 | `||fish_relative_position_body||` |
| `bearing_to_fish` | 1 | 水平面方位角，定义见下 |
| `teammate_slot_a` | 8 | 队友 one-hot + 相对位置 + 相对速度 |
| `teammate_slot_b` | 8 | 队友 one-hot + 相对位置 + 相对速度 |

总计：

```text
3 + 3 + 3 + 3 + 1 + 1 + 8 + 8 = 30D
```

`teammate_slot_*` 的 8 维展开为：

```text
teammate_type_onehot(2) + teammate_relative_position_body(3) + teammate_relative_velocity_body(3)
```

### 低层 controller suffix

`3Chase1` 里 Chaser 仍保留 `13D controller_obs` 后缀，仅供低层 position controller 使用：

```text
43D total = 30D actor_obs + 13D controller_obs
```

其中：

- 高层 MARL actor 只消费前 `30D`
- 低层 controller backend 只消费后 `13D`

## 4. 1Chase1 高层 Observation

`1Chase1` 当前统一为纯高层追踪 observation，不混入 controller suffix，总计 `14D`：

| 字段 | 维度 | 语义 |
| --- | --- | --- |
| `self_linear_velocity_body` | 3 | 自身体线速度 |
| `self_angular_velocity_body` | 3 | 自身体角速度 |
| `fish_relative_position_body` | 3 | prey 相对自身的位置 |
| `fish_relative_velocity_body` | 3 | prey 相对自身的速度 |
| `distance_to_fish` | 1 | 距离 |
| `bearing_to_fish` | 1 | 水平面方位角 |

总计：

```text
14D
```

说明：

- 这是 `1Chase1` 的**标准高层 observation 规范**。
- hierarchy 训练也使用同一套纯 `14D` observation；旧 `HierarchyPose20` / `pose20` 入口已废弃。
- 低层 traditional PID 不从 observation suffix 取控制目标，而是直接接收高层输出的 `3D body-frame subgoal` 作为 `controller_body` 位置误差。

## 5. bearing 定义

`bearing_to_fish` 统一定义为：

```text
bearing_to_fish = atan2(fish_relative_position_body.z, fish_relative_position_body.x)
```

语义固定为：

- `0`：鱼在正前方
- `> 0`：鱼在左侧
- `< 0`：鱼在右侧

注意这里是**水平面方位角**，只描述左右偏差，不编码俯仰角。

## 6. 为什么显式加入 distance / bearing

新增这两个量的目的是：

- 减轻策略网络自己从 3D 相对位置里再隐式拟合距离/方位的负担
- 让追踪奖励和策略输入的几何量更对齐
- 在 1Chase1 和 3Chase1 之间保持统一的“追鱼”接口

## 7. 哪些量给高层，哪些量给低层

高层 actor：

- 使用局部、相对、追踪语义强的 `actor_obs`
- 不直接依赖 world 原点或绝对位姿

低层 controller：

- Python backend 将高层输出的 body-frame subgoal 直接作为低层 position controller 的 `controller_body` 位置误差。
- Unity 可视化通过 side channel 接收该局部 subgoal，并在 Unity 层用当前 chaser transform 转成 world marker；不需要把可视化位姿塞进 observation。

因此当前推荐理解是：

```text
高层：决定“往哪里追”
低层：负责“怎么稳定地到这个局部目标”
```

## 8. 1Chase1 静态追踪 Baseline

为排查高层策略、低层控制律和推进器分配链路，提供三种无 checkpoint 的静态 baseline。三者都读取同一份 14D observation，并输出相同 canonical 顺序的 8D 推进器动作：

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]
```

| 配置 | 控制链 |
| --- | --- |
| `configs/rl/1chase1/baseline/pid_eval.yaml` | 位置 PID + yaw PID -> 6D wrench -> thrust allocator -> 8 推 |
| `configs/rl/1chase1/baseline/wrench_eval.yaml` | 代数 wrench PD -> thrust allocator -> 8 推 |
| `configs/rl/1chase1/baseline/thruster_eval.yaml` | 代数追踪律 -> 显式 canonical 8 推 mixer |

三者共享的追踪误差是：

```text
e_body = fish_relative_position_body
e_guided = clamp(e_body + 0.30 * fish_relative_velocity_body)
yaw_error = -atan2(e_body.z, e_body.x)
```

`e_body.z > 0` 表示 prey 在左侧。Unity 中正 yaw 向右，因此 controller 输出采用相反的
`yaw_error < 0` 使潜器向左转。1Chase1 的 controller-body 与 FinsROV 的 canonical
Unity local 轴对齐：`x=+X surge`、`y=+Y heave`、`z=+Z left/sway`。水平 4 推的纯前进
命令为 `[+,+,-,-]`，纯左移命令为 `[-,+,+,-]`。

运行示例：

```bash
cd .
uv run --package finssim-cli finssim rl eval \
  -c configs/rl/1chase1/baseline/pid_eval.yaml --use-editor
```

将 YAML 中的文件名替换为 `wrench_eval.yaml` 或 `thruster_eval.yaml` 即可。三个 YAML 默认跑 10 个 episode，并每 100 步打印一次 body error、yaw、wrench（PID 除外）和 8 推动作平均幅度；它们不读取或写入 RL checkpoint。

Unity 的 `OneChaseOnePoseAgent` 同时记录 `terminal_success`、`terminal_out_of_bounds`、`terminal_timeout`、终止距离、相对速度和动作幅度。查看 baseline 时优先比较 capture rate、终止原因、最小距离和动作幅度，不要只比较 dense reward。
