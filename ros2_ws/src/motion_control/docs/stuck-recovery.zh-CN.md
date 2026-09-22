# Motion Controller Stuck Recovery

## 背景

实机位置控制时，目标点靠近池壁。目标点本身可达，但潜器偶然贴到墙后，普通位置 PID 会继续按 `target - current_pose` 直接施加水平力。

如果这部分力把潜器压向墙面，墙体反作用力会抵消运动，潜器速度接近 0，位置误差又不会明显下降。此时 PID 积分会继续累积，推力分配器和硬件桥再经过限幅、最小有效 RPM、重排和正负号映射后，输出可能长期停在几个固定值附近。现场表现就是：

- 潜器贴墙后不再离开墙面。
- 误差可能越来越大。
- 推进器输入看起来长时间几乎不变。

这不是 reached 停控问题。如果 reached 生效，`/finsrov/thrusters_out` 会变成 0；而贴墙 stuck 是非零推力持续存在。

## 当前机制

stuck recovery 加在 `motion_control/controller_node.py` 的 ROS 节点外层，只影响 position command，不影响 velocity command。

检测条件同时满足时，进入 stuck candidate：

- 三维位置误差大于 `stuck_recovery.min_position_error_m`。
- 机体系线速度范数小于 `stuck_recovery.velocity_threshold_mps`，或者误差比上一帧继续变大。
- 水平推进器动作幅值大于 `stuck_recovery.horizontal_effort_threshold`。
- 误差在 `stuck_recovery.progress_epsilon_m` 以内没有明显改善。

candidate 持续超过 `stuck_recovery.trigger_duration_sec` 后触发恢复：

- reset 后端 PID，清掉积分状态。
- 保存当前推进器动作。
- 垂推按 `stuck_recovery.vertical_hold_scale` 缩放。
- 水平推进器按 `-stuck_recovery.backoff_horizontal_scale` 反向输出。
- 该退墙动作持续 `stuck_recovery.escape_duration_sec`。
- 之后进入 `stuck_recovery.cooldown_sec` 冷却期，再恢复普通位置 PID。

## 当前 FinsROV 位置 PID 参数

配置文件：

```text
ros2_ws/src/motion_control/config/FinsROV/traditional_pid_position.yaml
```

当前默认启用：

```yaml
stuck_recovery:
  enabled: true
  min_position_error_m: 0.12
  velocity_threshold_mps: 0.03
  horizontal_effort_threshold: 0.18
  progress_epsilon_m: 0.02
  trigger_duration_sec: 4.0
  escape_duration_sec: 1.0
  backoff_horizontal_scale: 0.35
  vertical_hold_scale: 1.0
  cooldown_sec: 1.0
  status_topic: /motion_controller/status/stuck
```

通用 `controller.yaml` 中默认关闭，避免影响其他控制配置。

## 现场观察

查看 stuck 状态：

```bash
ros2 topic echo /motion_controller/status/stuck
```

典型输出字段：

```json
{"active":true,"phase":"escape","position_error_m":0.18,"speed_norm_mps":0.01,"horizontal_effort":0.32,"held_duration_sec":2.1}
```

`phase` 含义：

- `idle`：未检测到 stuck。
- `candidate`：满足卡住条件，但持续时间还没到。
- `escape`：正在执行退墙反向水平推力。
- `cooldown`：退墙后冷却，暂不再次触发。
- `navigation_not_ready`：融合状态不满足控制条件。

## 调参建议

如果正常慢速靠近目标也误触发：

- 增大 `trigger_duration_sec`。
- 减小 `velocity_threshold_mps`。
- 增大 `horizontal_effort_threshold`。
- 增大 `min_position_error_m`。
- 减小 `backoff_horizontal_scale`，降低恢复动作对正常 PID 的打断。

如果贴墙后仍然退不出来：

- 增大 `escape_duration_sec`。
- 增大 `backoff_horizontal_scale`。
- 减小水平 PID 的积分增益或增大积分衰减。

如果退墙时深度变化太明显：

- 保持 `vertical_hold_scale: 1.0`，让退墙阶段沿用触发前的垂直推力。

## 注意事项

这个机制不是避障算法，也不知道墙的精确法向。它假设“当前水平 PID 正在把艇压向障碍物”，所以用当前水平动作的反方向短时间退让。目标点仍应尽量离墙留出艇体半宽以上的安全距离。
