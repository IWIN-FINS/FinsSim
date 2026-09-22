# 3Chase1 课程学习与 Reward 参数化

本文说明多 Training Area 的 `3Chase1` / `3Chase1_headless` 如何用同一个 Unity build，通过 FinsSim YAML 动态控制 reward、捕获条件和 prey 难度。

## 1. 设计原则

当前方案使用 ML-Agents 内置 `EnvironmentParametersChannel`：

```text
FinsSim YAML
  -> finssim-cli 写入 resolved_config.yaml
  -> finssim_marl 训练进程读取 curriculum
  -> 每个 Unity worker 的 EnvironmentParametersChannel
  -> Unity ThreeChaseOneCurriculumController
  -> CatchAreaManager / ChaserAgent / PreyAgent 字段
```

这意味着：

- 不需要为每一种 reward 重新建 scene。
- 不需要为每一个 lesson 重新 build Unity。
- observation/action 规格不变，checkpoint 和算法结构不受影响。
- 课程切换按训练全局 step 发生，不依赖 Unity 场景内计数。

## 2. Unity 端入口

`3Chase1_new` 由 `ThreeChaseOneFinsRovSceneBuilder` 修复或生成 headless scene 时，会自动在 `CatchAreaManager` 所在物体上挂载：

```text
Assets/Scripts/ThreeChaseOne/Training/ThreeChaseOneCurriculumController.cs
```

该脚本在运行时读取 ML-Agents environment parameters，并写入：

- `CatchAreaManager`：捕获判定、捕获阈值、终局 reward。
- `ChaserAgent`：追方 step reward、靠近、网宽、速度/动作惩罚等系数。
- `PreyAgent`：鱼的生存 reward、逃逸 reward、速度、垂直速度比例。

## 3. YAML 结构

示例文件：

```text
configs/marl/3chase1/curriculum/finsrov_curriculum_v1.yaml
```

核心结构：

```yaml
unity:
  env_path: artifacts/unity_builds/marl/linux/FinsROV/3Chase1_headless/3Chase1.x86_64
  environment_parameters:
    finsim_3c1.lesson_id: 0
    finsim_3c1.capture.criterion: 2
    finsim_3c1.reward.mode: 1

curriculum:
  enabled: true
  update_interval_steps: 1000
  lessons:
    - name: uuv_proximity_bootstrap
      start_step: 0
      parameters:
        finsim_3c1.lesson_id: 0
        finsim_3c1.capture.criterion: 2
```

`unity.environment_parameters` 是基础参数。每个 lesson 的 `parameters` 会覆盖基础参数。未覆盖的 key 保持基础值。

## 4. Reward 模式

`finsim_3c1.reward.mode` 是 float 参数，Unity 端会四舍五入到整数：

```text
0 = HerdingNet
1 = SimpleChasePrey
```

`HerdingNet` 是原始协同收网 shaping reward：Netter 关注靠近鱼、网宽、网面形状，Herder 关注赶鱼到网中心。

`SimpleChasePrey` 是简单追逐 reward：每个追方都只按“自己是否接近 prey”得到单步 shaping reward。捕获成功条件仍由 `CatchAreaManager` 的 `capture.*` 参数决定，终局 reward 仍由 `reward.chaser_capture` 和 `reward.fish_capture` 决定。

Simple chase 相关参数：

```text
finsim_3c1.reward.chaser.simple_chase_distance
finsim_3c1.reward.chaser.simple_chase_progress
finsim_3c1.reward.chaser.simple_chase_distance_range
```

对应公式：

```text
simple_reward =
  simple_chase_distance * clamp01(1 - distance_to_prey / simple_chase_distance_range)
  + simple_chase_progress * clamp(previous_distance_to_prey - distance_to_prey, -1, 1)
```

## 5. 捕获判定枚举

`finsim_3c1.capture.criterion` 是 float 参数，Unity 端会四舍五入到整数：

```text
0 = NetSurfaceDistance
1 = NetCollision
2 = UuvProximity
3 = NetCollisionOrUuvProximity
```

第一版课程建议：

```text
lesson 0: UuvProximity，用更宽松距离快速学会接近鱼。
lesson 1: NetSurfaceDistance，切到真实收网目标。
lesson 2: Strict NetSurfaceDistance，缩小网面距离并增加 hold time。
```

## 6. 支持的参数 key

捕获与终局：

```text
finsim_3c1.lesson_id
finsim_3c1.capture.criterion
finsim_3c1.capture.net_surface_distance
finsim_3c1.capture.net_surface_hold_time
finsim_3c1.capture.uuv_distance
finsim_3c1.reward.mode
finsim_3c1.reward.chaser_capture
finsim_3c1.reward.fish_capture
```

追方 reward：

```text
finsim_3c1.reward.chaser.step_penalty
finsim_3c1.reward.chaser.simple_chase_distance
finsim_3c1.reward.chaser.simple_chase_progress
finsim_3c1.reward.chaser.simple_chase_distance_range
finsim_3c1.reward.chaser.fish_closing
finsim_3c1.reward.chaser.netter_spacing
finsim_3c1.reward.chaser.netter_spacing_progress
finsim_3c1.reward.chaser.herder_fish_closing
finsim_3c1.reward.chaser.herding_progress
finsim_3c1.reward.chaser.angular_velocity_penalty
finsim_3c1.reward.chaser.linear_velocity_penalty
finsim_3c1.reward.chaser.action_magnitude_penalty
finsim_3c1.reward.chaser.action_delta_penalty
```

鱼的 reward 与难度：

```text
finsim_3c1.reward.prey.survival
finsim_3c1.reward.prey.average_separation_progress
finsim_3c1.reward.prey.nearest_threat_progress
finsim_3c1.prey.move_speed
finsim_3c1.prey.vertical_speed_scale
```

## 7. 启动训练

先确认 Unity build 存在：

```bash
ls -lh artifacts/unity_builds/marl/linux/FinsROV/3Chase1_headless/3Chase1.x86_64
```

dry-run：

```bash
cd .
uv run --package finssim-cli finssim marl train \
  -c configs/marl/3chase1/curriculum/finsrov_curriculum_v1.yaml \
  --dry-run
```

正式训练：

```bash
cd .
uv run --package finssim-cli finssim marl train \
  -c configs/marl/3chase1/curriculum/finsrov_curriculum_v1.yaml
```

训练日志里应该能看到：

```text
[INFO] Initial Unity environment parameters prepared: ...
[INFO] Curriculum lesson active: index=0, name=uuv_proximity_bootstrap, ...
```

Unity player log 里应该能看到：

```text
[ThreeChaseOneCurriculumController] lesson_id=0
[ThreeChaseOneCurriculumController] Environment parameters initialized.
```

## 8. 注意事项

- 课程切换只改变运行时字段，不改变 observation/action 维度。
- `reward.mode=1` 只改变追方 step shaping reward，不改变捕获成功条件。
- 参数值必须是数字；字符串、列表、字典不会被接受。
- 如果 Unity worker 崩溃并被 Python 重启，当前 lesson 参数会重新传给新 worker。
- `update_interval_steps` 不是每一步都强制发参数；lesson 改变时会立即发，间隔到达时会重发当前参数，防止 worker 重启或通信恢复后状态不一致。
