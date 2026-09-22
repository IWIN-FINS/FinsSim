# HoldForPosition Learn To Swim Goal-Hold Reward

`HoldForPosition` 支持用 ML-Agents `EnvironmentParametersChannel` 选择密集奖励。

| `finsim_hold.reward.mode` | 枚举 | 含义 |
| --- | --- | --- |
| `0.0` | `InspectorDefault` | 使用场景 Inspector 的原有奖励配置。 |
| `1.0` | `LearnToSwimGoalHold` | FinsSim 扩展版：在指数位置、姿态、动作能量奖励之外，增加显式跟踪/姿态/动作变化惩罚、近目标角速度惩罚和稳定驻留奖励。 |
| `2.0` | `IsaacPositionHold` | Isaac WarpAUV 对应的基础 position-hold 结构：仅位置、姿态、动作能量三项指数奖励。Unity 仍以 8 个推进器动作计算动作能量，而 Isaac 使用 6 维动作。 |
| `3.0` | `IsaacPositionHoldNearGoalDamping` | 保持 mode 2 的 Isaac 三项基础奖励，仅在近目标区额外加入低权重动作能耗、动作变化率和角速度惩罚，以抑制满推高频振荡。 |

本文说明 `mode: 1.0` 的 FinsSim 扩展奖励；它不是原始 Isaac 奖励。

## Mode 1 公式

```text
r = w_p exp(-k_p ||p_goal - p||^2)
  + w_q exp(-k_q theta(q_goal, q))
  + w_a exp(-k_a ||a||^2)
  - w_d ||p_goal - p||
  - w_q_abs theta(q_goal, q)
  - w_delta mean((a_t - a_(t-1))^2)
  - w_a_near near_goal_weight mean(a_t^2)
  - w_omega near_goal_weight ||omega_body||^2
  + w_hold I(position, attitude, angular velocity are inside hold bounds)
```

- `theta` 是完整 quaternion 姿态误差（roll、pitch、yaw 都受约束）。
- `a` 是当前 8 路推进器连续 action；这里使用平方范数而非均方，匹配
  WarpAUV 的 action-energy 写法。
- 前三项沿用 Learning to Swim 的稠密指数结构。额外的绝对误差惩罚保证在远离目标
  时仍有非零梯度，避免 action-energy 项造成静差。
- 近目标角速度惩罚仅在 `near_goal_radius` 内按线性权重启用，并截断最大角速度，避免
  高角速度异常值主导整个 reward。
- 近目标动作能耗惩罚同样只在 `near_goal_radius` 内按线性权重启用。它直接扣除 8 路
  动作的均方值，因此接近目标时持续 98% 满推会产生约 `-0.60` 的额外单步 reward，
  而不会削弱远距离接近目标所需的推进力。
- `hold_reward` 只有位置、姿态和角速度同时处于阈值内才发放，训练目标从“接近”变为
  “停住并保持”。
- 任务仍使用当前随机 cube 目标，而不是 WarpAUV 的固定原点目标；超界仍以 `-1`
  结束 episode，防止策略利用开放水域逃逸。

## 训练配置

示例配置：

```text
configs/rl/hold_for_position/ppo_wrench_for_hold_position_fossen_parallel_2048_15s_learn_to_swim_dense.yaml
```

关键透传参数位于 `env.unity.environment_parameters`：

```yaml
finsim_hold.reward.mode: 1.0
finsim_hold.reward.position_scale: 0.2
finsim_hold.reward.position_error_exponent: 1.0
finsim_hold.reward.attitude_scale: 0.5
finsim_hold.reward.attitude_error_exponent: 1.0
finsim_hold.reward.action_scale: 0.2
finsim_hold.reward.action_energy_exponent: 1.0
finsim_hold.reward.distance_penalty_scale: 0.15
finsim_hold.reward.attitude_penalty_scale: 0.05
finsim_hold.reward.action_delta_penalty_scale: 0.02
finsim_hold.reward.near_goal_radius: 0.75
finsim_hold.reward.near_goal_action_energy_penalty_scale: 0.60
finsim_hold.reward.near_goal_angular_velocity_penalty_scale: 0.03
finsim_hold.reward.near_goal_angular_velocity_penalty_max_radps: 5.0
finsim_hold.reward.hold_position_threshold: 0.20
finsim_hold.reward.hold_attitude_threshold_deg: 10.0
finsim_hold.reward.hold_angular_velocity_threshold_radps: 0.25
finsim_hold.reward.hold_reward_scale: 0.10
```

使用 Isaac/WarpAUV 基础结构时，只设置：

```yaml
# No distance/attitude/action-delta/near-goal/hold additions.
finsim_hold.reward.mode: 2.0
```

## Mode 3：Isaac + 近目标阻尼

mode 3 与 mode 2 使用相同的基础项，且不包含 mode 1 的距离绝对惩罚、姿态绝对惩罚、
动作变化惩罚或稳定驻留奖励；只增加：

```text
r = 0.2 exp(-||position_error||^2)
  + 0.5 exp(-attitude_error)
  + 0.2 exp(-||action||^2)
  - w_a_near near_goal_weight mean(action^2)
  - w_delta_near near_goal_weight mean((action_t - action_(t-1))^2)
  - w_omega near_goal_weight min(||omega_body||^2, omega_max^2)
```

mode 3 的动作变化率项只惩罚推进器换向/快速改变，不惩罚为抵消浮力或稳态扰动而维持的
固定配平推力。绝对动作能耗项应保持较小：若过大，策略可能宁可停在近目标区边缘，也不愿
付出必要推力到达目标中心。

推荐作为抑制目标点高频振荡、同时保留目标可达性的初始配置：

```yaml
finsim_hold.reward.mode: 3.0
finsim_hold.reward.near_goal_radius: 0.75
finsim_hold.reward.near_goal_action_energy_penalty_scale: 0.15
finsim_hold.reward.near_goal_action_delta_penalty_scale: 0.05
finsim_hold.reward.near_goal_angular_velocity_penalty_scale: 0.01
finsim_hold.reward.near_goal_angular_velocity_penalty_max_radps: 5.0
```

这些值在每个 `OnEpisodeBegin()` 读取。因此可以用不同 YAML 重复构建同一个 Unity
Player，而无需为了 reward 系数重新修改 scene 或 C#。

## 诊断

启用 `HoldForPosition.enableStatsRecorder` 后，TensorBoard 会记录：

- `FinsROV/hold_position_reward`
- `FinsROV/hold_attitude_reward`
- `FinsROV/hold_action_reward`
- `FinsROV/hold_distance_penalty`
- `FinsROV/hold_attitude_penalty`
- `FinsROV/hold_action_delta_penalty`
- `FinsROV/hold_action_delta_rms_mean`
- `FinsROV/hold_action_delta_rms_p95`
- `FinsROV/hold_near_goal_angular_velocity_penalty`
- `FinsROV/hold_reward` (stable-hold bonus)
- `FinsROV/hold_total_reward`

正常学习时，绝对距离和姿态 penalty 的绝对值应下降，`hold_target_window` 和
`hold_reward` 应上升。若 `hold_action_delta_penalty` 很大，先降低
`action_delta_penalty_scale`，不要通过改动水动力学参数掩盖控制抖动。
