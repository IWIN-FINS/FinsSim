# T2 MarineGym 风格三维轨迹跟踪

T2 使用一个 Unity scene 和一个固定的 8D Unity action 合约：
`[Vertical1..4, Horizontal1..4]`。8D checkpoint 直接输出推进器 action；
6D `wrench6` checkpoint 输出
`[Fx,Fy,Fz,Mx,My,Mz] = [forward,up,left,roll,yaw,pitch]`，再由 Python
`ThrustAllocator` 转为 8D 后送入同一个 Unity scene。两种 checkpoint 不能交叉加载。

两种变体的 observation 都为固定 30D：4 个 body-frame preview offset（12）、
参考线速度（3）、当前 body 线速度（3）、body 角速度（3）、body up direction
（3）、相对切线 yaw 的 sin/cos（2）、时间编码（4）。坐标一律是
`controller_body: x forward, y up, z left`。

传统基线（无 checkpoint）使用同一份 30D observation：它对当前 preview offset
和参考/实际 body velocity 做位置-速度 PD，对轨迹切线 yaw 与 body-up 做姿态
恢复，再通过 Python `ThrustAllocator` 输出同一 scene 的 8 路推进器 action。用于
先验证轨迹任务是否可控以及 reward 是否给出合理排序：

```bash
cd .
uv run --package finssim-cli finssim rl eval \
  -c configs/rl/trajectory_tracking/baseline/wrench_pd_physical_allocator_eval.yaml
```

训练 build：

```bash
cd /marus-example
./tools/unity/build_in_worktree.sh --execute-method FinsSimLinuxBuild.BuildTrajectoryTrackingFossenParallel30sServer --log-file /tmp/trajectory-tracking-server.log --verbose
```

部署 profile：real 使用 `ppo_trajectory_tracking_{thruster8,wrench6}.yaml`，Unity
sim 使用同名 `_sim.yaml`。先将 `checkpoint_path` 替换为对应训练产物；sim profile
始终使用 `/sim` command 与 thruster topics。

示例轨迹命令：

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run motion_control send_trajectory \
  --topic /sim/motion_controller/command/trajectory \
  --points-json '[[0,0,-2,0],[5,2,-2,0],[10,4,-2,0]]'
```

控制器只在收到整条 `trajectory_msgs/MultiDOFJointTrajectory` 时 reset policy 一次；
之后按时间插值 current/preview reference，不会把每个 preview 当成独立 position goal。

## 30 s 可达轨迹约束

T2 不要求随机抽到的几何图形一定在一个 episode 内走完整圈。每次 reset 先抽取
圆、8 字或螺旋的几何尺度，再以物理测定的满量程速度为上界生成 reference：

```text
measured full-wrench steady-state: surge 0.4998, sway 0.3520, heave 0.2196 m/s
training reference limits:         surge 0.30,   sway 0.20,   heave 0.10 m/s
```

参考上限保留了推进器一阶响应、曲率、初始误差和 Fossen/DR 扰动的跟踪余量，不是改变
推进器的 `+/-7 N` 力限幅。对周期曲线，使用：

```text
omega = sign * min(2*pi*requested_cycles / 30 s,
                   min_phase_axis(v_axis_limit / |d(position_axis)/d(phase)|))
```

因此当大圆或大 8 字的一整圈在 30 秒内不可达时，reference 仍以可跟踪速度连续前进，
episode 在 30 秒结束时自然停在该图形的一段，例如约半圈，而不是加速到不可跟踪的速度。
Inspector 的 `selectedTrajectoryCycles` 显示该 episode 实际走过的圈数。直线也限制为
`span / 30 s <= maxReferenceSurgeSpeedMps`；过长时只生成可完成的有限线段。

可在 `TrajectoryTrackingAgent` 组件调整以下字段：

- `trajectoryScaleXRange`、`trajectoryScaleYRange`、`trajectoryScaleZRange`：几何范围。当前
  `X=0.5..2.5 m`、`Z=0.25..1.25 m` 已覆盖 4 m x 2 m 水池，也包含更大的 OOD 轨迹。
- `requestedTrajectoryCycles`：希望 30 秒完成的圈数。实际圈数会受速度上限裁剪。
- `maxReferenceSurgeSpeedMps`、`maxReferenceSwaySpeedMps`、`maxReferenceHeaveSpeedMps`：参考
  的 body-axis 速度约束。只有重新完成针对该轴的速度辨识后才应上调。

### Curriculum stage

同一个 T2 binary 可由训练 YAML 选择阶段，无须重新 build：

```yaml
env:
  unity:
    environment_parameters:
      finsim_trajectory_curriculum_stage: 0 # 0: line/circle, 1: circle/8, 2: circle/8/helix
```

该值通过 ML-Agents `EnvironmentParametersChannel` 下发，在每次 episode reset 生效；训练和
异步 eval 都使用各自 YAML 中的相同值。Player log 会输出
`[TrajectoryTrackingAgent] curriculum stage=<N> source=mlagents_parameter`。用于直接启动 binary
的调试参数是 `-fins-trajectory-curriculum-stage N`，其优先级高于 YAML parameter；两者均未
提供时保留 scene Inspector 的 `curriculumStage`。

## TensorBoard metrics

Unity 每个统计窗口通过 `StatsRecorder` 回传 `FinsROV/trajectory_tracking/*`。Python 的通用
Unity stats adapter 位于 `finssim_rl/envs/unity_stats.py`，会把任意 Unity metrics 写入
TensorBoard，不依赖 task-specific key 前缀：

- train: `tensorboard/unity_train_metrics`
- asynchronous eval: `tensorboard/async_eval`
- synchronous eval: `tensorboard/unity_eval_metrics`

T2 记录 `tracking_error_mean_m`、`tracking_error_p95_m`、`tracking_error_max_m`、角速度
mean/P95、policy action RMS、相邻决策 action-delta RMS mean/P95、轨迹进度以及 8 路推进器
实际饱和率。每累计 16384 个 decision samples 发送一次；在 episode reset 前也会 flush，因此
短评测不会因未达到窗口阈值而缺失指标。
