# GoalYaw 单智能体 AIRL/GAIL Pipeline

## 范围与边界

本链路新增 `python/finssim_irl`，不替换 `python/finssim_rl` 的 PPO、PID、Unity
训练或已有 ROS 控制功能。Unity 的新任务只包含每回合一个“目标位置 + 目标 yaw”，
成功后结束回合；它不复现旧任务的奖励、目标刷新或 dropout 语义。

Unity 与数据集的固定 ABI 为：

- observation：16 维，`target_local(3) + relative_rotation_6d(6) + local_linear_velocity(3) + local_angular_velocity(3) + normalized_distance(1)`；
- 原生动作：8 个直接推进器，顺序为 `Vertical1..4, Horizontal1..4`；
- IRL 生成场景使用 `NormalizedMaxForceRequest`：每个动作 `a∈[-1,1]` 按对应推进器当前的
  `MaxForwardForceN/MaxReverseForceN` 映射为物理推力；它不是固定 50N，也不是固定 7N 的
  全局缩放。推进器启用 DR 时，该上限会随当前 episode 的推进器参数一起变化；
- 可选策略动作：`wrench4_v1=[surge_fx, heave_fy, sway_fz, yaw_my]`，固定嵌入为 `[Fx,Fy,Fz,0,My,0]` 后再由现有物理分配器输出 8 推进器。它不是现有 6D wrench ABI 的截断或别名。

任务阈值、归一化尺度和时间步由
`configs/irl/goal_yaw/task_v1.yaml` 统一定义。成功必须连续保持 **5 s** 同时满足：
位置误差不大于 0.15 m、yaw 误差不大于 10 deg、线速度模长不大于 0.20 m/s、角速度模长
不大于 20 deg/s。该保持时间按 Unity 物理仿真时间累计，不受运行 `time_scale` 影响。Unity 的
90 s 上限对应 `DecisionPeriod=5`、`MaxStep=4500`（0.02 s 物理步长；策略仍以 10 Hz 决策）。

所有采集数据统一存于 `/data/UnderwaterSim/datasets/irl/goal_yaw`，不写入代码仓库的
`artifacts/`。其中 `raw/` 保存 ROS/rosbag 与导出表，`*_npz/` 保存可恢复采集分片，
`*_rlds/` 保存供训练读取的权威 RLDS TFRecord 数据集。

## 1. 准备环境和 Unity binary

```bash
cd ./python/finssim_irl
uv sync --all-groups

cd /marus-example
./tools/unity/build_in_worktree.sh \
  --execute-method GoalYawIrlLinuxBuild.BuildGoalYawIrlFossenServer \
  --log-file /tmp/goal-yaw-irl-server.log \
  --verbose
```

正式批量采集/训练使用 Server executable：

```text
./artifacts/unity_builds/irl/linux/GoalYawIRL_Fossen_10Hz_Server/GoalYawIRL.x86_64
```

如需可视化检查，另构建 Linux Player：

```bash
./tools/unity/build_in_worktree.sh \
  --execute-method GoalYawIrlLinuxBuild.BuildGoalYawIrlFossenPlayer \
  --log-file /tmp/goal-yaw-irl-player.log \
  --verbose
```

并使用 `configs/irl/goal_yaw/demos_pid_thruster8_debug.yaml`；它固定为 GUI 与
`time_scale=1.0`，仅用于 3--5 回合人工检查，不可用于正式数据集。

构建过程在 Unity worktree 内从 `ControlForPosition_Fossen.unity` 生成
`Assets/Scenes/IRL/GoalYawIRL_Fossen_10Hz.unity`，不会修改源 scene。它要求场景中
只有一个 `GoalYawIrlAgent`，并验证 obs16、8 action、行为名 `GoalYawIRL` 和 10 Hz
decision period。

## 2. 采集 Unity PID 专家数据

先固定一种动作契约；不能把 8D 和 4D 数据混写到同一数据集。
`demos_pid_*.yaml` 会加载同目录的 `pid_expert_sim.yaml`。该文件独立固定了 T1 仿真 PID
的有效控制频率、位置/yaw PID 参数、经验推进器混控、轴限幅和 7 N 推力边界；它不启动、
引用或改写 `traditional_pid_position_yaw_sim.yaml`。采集配置还开启 24 个 reset strata：3 个
目标距离区间（0.5–1、1–2、2–3 m）× 4 个初始 yaw 误差区间（0–45、45–90、90–135、135–180 deg）
× 2 个高度差类别（近水平/明显高度差）。采集器仅写入通过上述成功门限的回合；超时或越界回合会丢弃，
默认最多尝试所需成功轨迹数的 10 倍。采集先写每回合独立的 NPZ 分片；随后在同一命令中
写入权威归档格式：使用 RLDS episode/step 语义的分片 TFRecord。NPZ 仅用于可恢复采集与
审计，AIRL/GAIL 训练应读取 RLDS 目录。

```bash
cd ./python/finssim_irl

uv run finssim-irl collect-unity-pid \
  --runtime-config ./configs/irl/goal_yaw/demos_pid_thruster8.yaml \
  --task-config ./configs/irl/goal_yaw/task_v1.yaml \
  --action-contract thruster8_v1 \
  --episodes 200 \
  --max-attempts 2000 \
  --output /data/UnderwaterSim/datasets/irl/goal_yaw/unity_pid_thruster8_v1_npz \
  --rlds-output /data/UnderwaterSim/datasets/irl/goal_yaw/unity_pid_thruster8_v1_rlds

uv run finssim-irl validate-demos \
  --task-config ./configs/irl/goal_yaw/task_v1.yaml \
  --action-contract thruster8_v1 \
  --input /data/UnderwaterSim/datasets/irl/goal_yaw/unity_pid_thruster8_v1_npz

uv run finssim-irl validate-rlds \
  --task-config ./configs/irl/goal_yaw/task_v1.yaml \
  --action-contract thruster8_v1 \
  --input /data/UnderwaterSim/datasets/irl/goal_yaw/unity_pid_thruster8_v1_rlds
```

若使用 4D wrench，替换为 `demos_pid_wrench4.yaml` 和 `wrench4_v1`，并使用独立的
输出目录。NPZ 分片包含 `manifest.json` 与 `episodes/episode_*.npz`；RLDS 目录包含
`finsim_rlds_manifest.json`、`rlds_schema.json` 与 `train-*.tfrecord`。每个 RLDS episode 是
一个 `tensorflow.SequenceExample`：context 中保存 episode ID/JSON 元数据，step 中保存
obs16、归一化 policy action、时间戳、可选 native reward，以及 `is_first/is_last/is_terminal`。
最终观测单独作为最后一个 step，且 `action_valid=false`；90 s 超时为 `is_last=true` 但
`is_terminal=false`。任务 SHA-256、obs16 和动作契约会在导入训练前再次校验。

## 3. 采集实机手柄专家数据

`irl_data` 只发布目标位姿和轨迹事件，**不**启用 teleop、不订阅手柄、也不
写推进器。现有 `teleop`、安全/使能链和 `trajectory_data` 仍是唯一
的实机控制与记录链路。

先构建并启动既有 passive recorder：

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select irl_data
source /opt/ros/humble/setup.bash
source .venv/bin/activate
# 在只选择构建一个包的工作区中，显式加载它及其消息/记录依赖；完整工作区
# 构建后的 install/setup.bash 已包含相同的 package hooks。
source install/msgs/share/msgs/package.bash
source install/trajectory_data/share/trajectory_data/package.bash
source install/irl_data/share/irl_data/package.bash

# 另一个终端：以既有流程开始 rosbag 录制；此命令本身不控制潜器。
record_teleop_trajectory --session-id 20260902_goal_yaw --label goal_yaw_irl
```

每一个人工驾驶回合，先显式写入真实目标，再用已有的受控手柄链路完成该回合：

```bash
publish_goal_yaw_irl_event \
  --session-id 20260902_goal_yaw --episode-id ep-001 --event episode_start \
  --goal-x 1.0 --goal-y 0.0 --goal-z -2.0 --goal-yaw-deg 90 --frame-id map

# 使用既有 teleop 链路人工驾驶至目标；保持其现有安全检查和 emergency-stop 流程。

publish_goal_yaw_irl_event \
  --session-id 20260902_goal_yaw --episode-id ep-001 --event episode_stop --outcome success
```

录制完成后用既有 exporter 生成对齐表，并将 raw event 转为显式 episode manifest：

```bash
export_teleop_trajectory /data/UnderwaterSim/datasets/irl/goal_yaw/raw/real/20260902_goal_yaw
build_goal_yaw_irl_manifest \
  --events-parquet /data/UnderwaterSim/datasets/irl/goal_yaw/raw/real/20260902_goal_yaw/raw_streams/trajectory_event.parquet \
  --session-id 20260902_goal_yaw \
  --output /data/UnderwaterSim/datasets/irl/goal_yaw/raw/real/20260902_goal_yaw/goal_yaw_episode_manifest.jsonl
```

导入只接受 `synchronization_valid=true` 的采样。必须在一次标定后显式提供：

- `policy-from-ros`：ROS 到 policy 坐标的 3×3 正交基变换；
- `policy-action-from-ros`：录制动作顺序/符号到 policy 动作顺序的 signed-permutation。

工具不猜测 ROS/Unity 坐标轴、yaw 轴或推进器顺序。下例中的单位矩阵只能在实机
frame 与动作顺序已被验证完全相同的情况下使用；否则必须换成经标定的矩阵。

```bash
cd ./python/finssim_irl
uv run finssim-irl import-ros-demos \
  --task-config ./configs/irl/goal_yaw/task_v1.yaml \
  --action-contract wrench4_v1 \
  --trajectory-parquet /data/UnderwaterSim/datasets/irl/goal_yaw/raw/real/20260902_goal_yaw/trajectory.parquet \
  --episode-manifest /data/UnderwaterSim/datasets/irl/goal_yaw/raw/real/20260902_goal_yaw/goal_yaw_episode_manifest.jsonl \
  --policy-from-ros-json '[[1,0,0],[0,1,0],[0,0,1]]' \
  --policy-action-from-ros-json '[[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]' \
  --output /data/UnderwaterSim/datasets/irl/goal_yaw/real_teleop_wrench4_v1_npz \
  --rlds-output /data/UnderwaterSim/datasets/irl/goal_yaw/real_teleop_wrench4_v1_rlds
```

实机数据进入训练集前，应人工审查每回合的目标 frame、起止时间、时间同步、动作饱和、
急停/异常和成功标签。`episode_stop` 的 `outcome` 是审计元数据，不替代数据质量审查。

## 4. 训练、评估和部署导出

训练时可以混合多个同一任务 SHA-256、同一动作契约的 RLDS 数据集。下面是 8D GAIL；将
config 换成 `airl_thruster8.yaml`、`gail_wrench4.yaml` 或 `airl_wrench4.yaml` 即可。

```bash
cd ./python/finssim_irl
uv run finssim-irl train \
  --config ./configs/irl/goal_yaw/gail_thruster8.yaml \
  --demos /data/UnderwaterSim/datasets/irl/goal_yaw/unity_pid_thruster8_v1_rlds \
  --demos /data/UnderwaterSim/datasets/irl/goal_yaw/real_teleop_thruster8_v1_rlds

uv run finssim-irl evaluate \
  --config ./configs/irl/goal_yaw/gail_thruster8.yaml \
  --checkpoint ./artifacts/runs/irl/goal_yaw/gail_thruster8_v1/generator_policy.zip \
  --episodes 200 \
  --output ./artifacts/evaluations/irl/goal_yaw/gail_thruster8_v1.json

uv run finssim-irl export \
  --task-config ./configs/irl/goal_yaw/task_v1.yaml \
  --action-contract thruster8_v1 \
  --checkpoint ./artifacts/runs/irl/goal_yaw/gail_thruster8_v1/generator_policy.zip \
  --output ./artifacts/deployments/irl/goal_yaw/gail_thruster8_v1
```

`evaluate` 使用 Unity 原生 GoalYaw 指标（位置误差、yaw 误差、成功率和 episode
native return），不将 AIRL/GAIL 的 learned reward 当作部署成功指标。`export` 只导出
generator PPO policy 和任务/动作契约；奖励网络仅用于训练与诊断，不应被部署控制链路
依赖。
