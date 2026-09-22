# Unity Fossen Held-out Replay Validator

`unity_fossen_replay_validator` 用于验证已经冻结的 FinsROV 对角简化
Fossen profile 是否能预测新的实机单轴响应。它不重新拟合参数，也不评价
控制器、wrench allocator 或 RPM 控制器。

验证对象是静水、小速度、单轴条件下的 `surge_x`、`sway_z`、`heave_y` 或
`yaw_y`。不应把通过其中某几轴的结果写成“完整 6-DOF 数字孪生”。

## 试验顺序

1. **冻结 profile。** 记录 Unity 的 `FinsROV_HydrodynamicsProfile.asset`、
   代码 commit 和硬件桥配置。开始 held-out 试验后，不得基于该批 CSV 修改
   added mass、阻尼、推力曲线或 allocator。
2. **采集新的实机 held-out CSV。** 使用已有 identifier 的同一坐标约定、
   RPM 反馈和推力曲线，但指定 `run_role:=held_out_fossen_validation`。
   该模式只写 CSV、运行记录和全量运动图，不会生成 `.fit.json` 或调用
   参数拟合。
3. **冻结 manifest。** `prepare` 将 CSV 与 Unity profile 的 SHA-256 写入
   `validator_manifest.json`。后续 CSV/profile 被改动时，`analyze` 默认拒绝运行。
4. **Unity 回放。** 用 RPM 反馈恢复的八路实际力直接回放；禁用 domain
   randomization 和第二套水动力 source，记录 Unity 实际 applied wrench、DVL
   和 IMU。
5. **离线分析。** `analyze` 按 trial 与 phase 对齐，输出输入审计、受迫响应、
   coast 衰减、耦合泄漏和 trial-level 统计量。

## 实机采集

以下示例只运行一个 surge 幅值及正反方向各三次。实机开始前仍须按
`AGENTS.md` 的顺序确认硬件 bridge、传感器状态、单一 `/finsrov/thrusters_out`
发布者和安全水域。

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 launch hydrodynamic_identification \
  hydrodynamic_identification.launch.py \
  auto_start:=true \
  run_role:=held_out_fossen_validation \
  axes:=surge_x amplitude:=9.0 include_negative:=true \
  repeat_per_trial:=3 \
  hold_sec:=9.0 rest_sec:=4.0
```

每条 trial 使用“baseline → excitation → coast → rest”。`surge_x`、
`sway_z`、`yaw_y` 建议分别选择两个没有用于 profile 拟合的安全幅值；每个幅值
正反方向各三次。`heave_y` 保持 `dive_and_coast`，只使用安全下潜与自然上浮，
不把净浮力 bias 直接解释成阻尼。

默认 `pause_between_trials:=false`：每条 trial 的 `rest_sec` 零推力结束后，
节点立即开始下一条 trial 的 baseline，不需要按 Enter 或调用 continue service。

该运行结束后，输出目录中的 CSV 旁会有同名 `.validation.json`。例如：

```text
.../data/hydrodynamic_identification/surge_x/surge_x_024/
  surge_x_024.csv
  surge_x_024.validation.json
```

不要对这一批 CSV 运行 `refit_hydrodynamic`，也不要将它加入 profile 的训练/辨识集。

## 创建冻结 manifest

```bash
REAL_CSV=/absolute/path/to/surge_x_024.csv
PROFILE=./simulators/unity/marus-example/Assets/Models/FinsROV/Hydrodynamics/FinsROV_HydrodynamicsProfile.asset
RUN=./ros2_ws/data/hydrodynamics/validation/20260903_surge_held_out

./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification \
  unity_fossen_replay_validator prepare \
  --real-csv "$REAL_CSV" --profile "$PROFILE" --axis surge_x \
  --output-dir "$RUN"
```

默认门限被写入 manifest，因而必须在看结果前确定。它们是项目的预注册工程门限，
不是通用水动力学定律：输入 wrench 相对误差 (p95\leq 0.05)，excitation
NRMSE \(\leq 0.20\)，coast NRMSE \(\leq 0.30\)，稳态误差 \(\leq 20\%\)，
衰减时间误差 \(\leq 25\%\)。如需调整，应在 `prepare` 时传入相应参数，并在
实验 manifest 中保留理由；不得在看过结果后修改。

## Unity 回放和记录

在 `HoldForPosition_Fossen_Parallel_30s_NoDR.unity` 或等价的 Fossen 场景中确认：

- 使用同一个冻结 `HydrodynamicsProfile`；
- `Fossen6Dof` backend 已启用；
- domain randomization 关闭；
- DWP2/mesh 等第二水动力力源关闭；
- `/sim/finsrov/thrusters_out` 接受单位为 N 的 canonical 八推进器顺序；
- `/sim/finsrov/reset`、`/sim/finsrov/controller/dvl`、
  `/sim/finsrov/controller/imu` 与
  `/sim/finsrov/debug/thruster_applied_wrench` 均已可用。

先启动 gRPC--ROS2 adapter 和 Unity 场景。然后先录制、后回放：

```bash
SIM_DIR="$RUN/unity"
mkdir -p "$SIM_DIR"

./scripts/run_ros2_uv.sh ros2 run motion_control \
  record_real_sim_diagnostics \
  --duration 180 --output-dir "$SIM_DIR"

# 在另一个终端执行；所有 trial 完成后回放器会持续发送零力。
./scripts/run_ros2_uv.sh ros2 run motion_control \
  replay_unity_hydrodynamics \
  --csv "$REAL_CSV" --output-dir "$SIM_DIR" \
  --force-source force_from_rpm \
  --settle-sec 1 --between-trials-sec 1 --rate 50
```

`--force-source force_from_rpm` 是必须项：它回放由实机 RPM 反馈和已标定推力
曲线恢复的实际力。回放器不会经过 allocator 或再次通过推力曲线。分析器以
`/sim/finsrov/debug/thruster_applied_wrench` 为 Unity 输入真值；如果该话题缺失，
不能声称完成动力学验证。

## 分析与判读

```bash
./scripts/run_ros2_uv.sh ros2 run hydrodynamic_identification \
  unity_fossen_replay_validator analyze \
  --manifest "$RUN/validator_manifest.json" \
  --sim-dir "$SIM_DIR"
```

输出位于 `$RUN/analysis/`：

```text
trial_summary.csv          # 每 trial 的输入、响应、coast 与泄漏指标
analysis_report.json       # 门限、分类数量、trial-level bootstrap CI
summary_metrics.png        # NRMSE 与输入一致性汇总
trial_XXX_response.png     # applied wrench 与真实/Unity 速度响应
```

`classification` 的含义如下：

- `consistent_with_frozen_profile`：输入审计和所有已登记 phase 指标均通过；
- `mismatch`：输入一致，但 Fossen 受迫或自由衰减响应没有通过门限；
- `incomplete`：缺少 Unity topic、phase、有效输入或 coast 数据，不能用于模型结论。

统计单位始终是 **trial**，不是高频采样点。论文应报告每轴 trial 指标的中位数和
bootstrap 95% CI；只对通过该协议的轴作“held-out replay-consistent”表述。若想
证明辨识 profile 的价值，可将同一份 held-out CSV 再回放到冻结的经验阻尼 baseline，
比较每条 trial 的 NRMSE 差值；这不需要额外实机试验。
