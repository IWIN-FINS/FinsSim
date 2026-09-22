# motion_control

ROS2 package for terminal-driven motion commands that are converted into
8-thruster outputs for FinsSim vehicles through a selectable backend from
`python/finssim_rl`.

## What It Provides

Long-running node:

* `motion_controller`
  * subscribes to vehicle observations
  * accepts position, pose, and velocity commands
  * builds the backend observation vector
  * runs the selected controller or RL policy
  * publishes 8 thruster values to `/thrusters_out`
* `reset_service`
  * exposes the ROS2 reset service
  * publishes cancel, zero-thruster, and Unity reset messages

Helper CLIs:

* `send_position_goal`
* `send_velocity_command`
* `cancel_goal`
* `reset_vehicle`
* `record_real_sim_diagnostics`
* `analyze_real_sim_diagnostics`

Offline Real/Simulation Diagnostics
-----------------------------------

After recording topics with `record_real_sim_diagnostics`, regenerate the
action, state, observation, pose, thruster and comparison plots with:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control \
  analyze_real_sim_diagnostics \
  --real-dir ./ros2_ws/data/rl_real_sim_diagnostics/20260810_real_goal_0_-0p5_0_yaw90 \
  --sim-dir ./ros2_ws/data/rl_real_sim_diagnostics/20260810_sim_goal_0_-0p5_0_yaw90 \
  --checkpoint ./artifacts/runs/rl/wrench/ppo_wrench_for_pose_v2_0809/checkpoints/best_model.zip
```

The script writes generated files to `--real-dir` by default. The logged
`debug_action` is the controller's actual action after clipping and rate
limiting. The raw action is reconstructed by deterministic inference from
the recorded `debug_observation` and the supplied checkpoint. Use
`--no-raw-action` when a checkpoint is unavailable.

Pose plots and pose CSV files use the controller-native orientation convention,
not the conventional ROS ZYX Euler extraction: `roll_x` is rotation about the
controller X axis, `pitch_z` is rotation about the controller Z axis, and
`yaw_y` is rotation about the controller Y axis. This matches
`x=forward, y=up, z=left`; the output column order remains
`[roll_x, pitch_z, yaw_y]`.

Offline Fossen Hydrodynamic Replay
----------------------------------

`replay_unity_hydrodynamics` replays the measured real-vehicle CSV directly to
the Unity bridge. It publishes the eight `force_cmd_*_n` columns to
`/sim/finsrov/thrusters_out`; it does not run `ThrustAllocator`, the ROS
thruster force curve, or another command conversion. Start the Unity Editor
scene `VariousVehiclesNewHydrodynamics` with the Fossen vehicle and the `/sim`
gRPC adapter first, then run:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run motion_control replay_unity_hydrodynamics \
  --csv data/hydrodynamics/real/hydrodynamic_identification/yaw_y/yaw_y_007/yaw_y_007.csv \
  --output-dir data/hydrodynamics/unity/various_vehicles_new_fossen_20260818/yaw_y/yaw_y_001 \
  --settle-sec 1 --between-trials-sec 1 --rate 50
```

Record the Unity topics before starting the replay:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control record_real_sim_diagnostics \
  --duration 105 \
  --output-dir data/hydrodynamics/unity/various_vehicles_new_fossen_20260818/yaw_y/yaw_y_001
```

Regenerate all axis summaries and plots with:

```bash
./scripts/run_ros2_uv.sh python3 scripts/analyze_unity_fossen_hydrodynamics.py \
  --unity-root data/hydrodynamics/unity/various_vehicles_new_fossen_20260818 \
  --real-root data/hydrodynamics/real/hydrodynamic_identification
```

For this comparison, `/sim/finsrov/debug/thruster_applied_wrench` is the
authoritative Unity input. It is the body-frame sum of the forces actually
applied by the eight Unity thrusters, not merely the requested ROS array. The
analysis compares that wrench with the real identifier's `tau_fit_*` fields.
The generated dataset is under
`data/hydrodynamics/unity/various_vehicles_new_fossen_20260818/`.

Do not use the current `roll_x` trials to fit roll damping: the replayed
vertical-thruster forces produce an applied Unity `Mx` with the opposite sign
to the real identifier's positive roll convention. First correct the Unity
vertical-thruster direction/moment mapping, then repeat the roll sweep. The
current heave pulse is also not a steady-step identification; use a longer,
matched excitation and coast window before changing heave added mass or
damping. The yaw sweep is already sufficiently consistent to serve as the
baseline for the Fossen yaw profile.

## Supported Backends

Built-in backend names:

* `traditional_pid_position`
* `traditional_ff_velocity`
* `traditional_pid_velocity`
* `traditional_pid_ff_velocity`
* `traditional_pid_ff_eso_velocity`
* `ppo_control_for_pose`
* `ppo_control_for_moving_target_dynamic`
* `ppo_control_for_moving_target_dynamic_smoke`
* `ppo_control_for_velocity`
* `ppo_control_for_velocity_no_rotation`
* `ppo_control_v2_for_pose`
* `ppo_control_v2_for_velocity`
* `hybrid_pid_pose_v3`
* `hybrid_pid_pose_v3_random_tau`
* `hybrid_pid_frozen_v3`
* `hybrid_pid_pose_residual`
* `hybrid_pid_pose_residual_frozen`

Notes:

* pose backends consume the 20D pose observation and are suitable for point-to-point control
* velocity backends accept direct velocity commands
* some pose backends can also use a target orientation quaternion
* if a command contains orientation but the current backend or YAML disables it, the node prints a warning and ignores that orientation target

## Workspace And `.venv`

This workspace is expected to run inside `ros2_ws/.venv`.

First-time setup:

```bash
cd ./ros2_ws
./scripts/bootstrap_uv_env.sh
./scripts/colcon_build_uv.sh --packages-select motion_control
```

Recommended ways to run commands after that:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run motion_control motion_controller --ros-args \
  --params-file ./ros2_ws/src/motion_control/config/FinsROV/traditional_pid_position.yaml
```

Or source once and keep the shell:

```bash
cd ./ros2_ws
source ./scripts/source_ros2_uv.sh
```

After sourcing, plain `ros2 run ...` works in that shell.

## Controller Profiles And Startup

The controller profile YAML is the single source of truth for the selected
controller backend and its runtime model configuration. In particular,
`backend_name` and `checkpoint_path` must be defined in the selected YAML under
`motion_controller.ros__parameters` (or under the file's wildcard namespace for
the sim-only profile). The launch file no longer accepts command-line overrides
for either parameter.

To switch between PID, PPO, or wrench control, select a different profile file:

```bash
./scripts/run_ros2_uv.sh ros2 launch motion_control motion_controller.launch.py \
  params_file:=src/motion_control/config/FinsROV/traditional_pid_position.yaml
```

For an RL profile, the checkpoint is read from that profile as well:

```bash
./scripts/run_ros2_uv.sh ros2 launch motion_control motion_controller.launch.py \
  params_file:=src/motion_control/config/FinsROV/ppo_control_for_pose_dr.yaml
```

The remaining launch arguments, such as `pool_world_params_file` and
`cuda_visible_devices`, control runtime topology or the execution environment.
`state_input_mode` defaults to `auto`: launch reads it from the selected YAML
profile and falls back to `raw_fusion` when the profile does not set it.

In VS Code, `Start All (with debug)` and `Start Control Only` use a dropdown
containing the YAML files currently under `config/`. The `run/controller-node`
task accepts a custom basename or path, while `run/controller-node-profile`
provides the same built-in profile dropdown for a standalone controller.

## State Inputs

Preferred mode:

* `odom_topic`
  * if provided, the node uses `nav_msgs/msg/Odometry` directly
* `pose_topic`
  * if provided and `odom_topic` is empty, the node uses pose plus IMU/DVL-derived velocities

Current FinsROV prototype mode:

* `pose_topic`
* `imu_topic`
* `depth_topic`
* `dvl_topic`
* `state_status_topic`

For real-vehicle runs, `state_status_topic` should normally be
`/finsrov/controller/state/status`, published by `controller_state_adapter`. The
controller uses this topic as a navigation safety gate before calling any RL or
traditional backend.

Default safety parameters:

```yaml
state_status_topic: /finsrov/controller/state/status
require_fusion_ready: true
state_timeout_sec: 0.5
allowed_vision_modes: [fresh, coast]
```

With `require_fusion_ready: true`, the controller publishes zero thrust and does
not build RL/PID observations when:

* `/finsrov/controller/pose`, `/finsrov/controller/imu`, or
  `/finsrov/controller/dvl` is missing or stale
* `/finsrov/controller/state/status` is missing or stale
* fusion reports `initialized=false`
* fusion reports `ready=false`
* fusion reports a `vision_mode` outside `allowed_vision_modes`

This means all position/RL/velocity backends consume the same validated state.
Global position commands can still be accepted before the vehicle is ready, but
no thrust is produced until the navigation gate opens. Body-frame goals and
world-frame velocity commands need the current pose/orientation for conversion,
so they are rejected while the gate is closed.

The current Unity bridge can publish these streams directly from one script
without requiring a physically mounted IMU first. For Unity simulation, keep
`require_fusion_ready: true` and use a sim profile that sets:

```yaml
state_input_mode: sim_truth
```

Then start the normal controller launch:

```bash
./scripts/run_ros2_uv.sh ros2 launch motion_control motion_controller.launch.py \
  params_file:=src/motion_control/config/FinsROV/ppo_wrench_for_pose_sim.yaml
```

This starts `sim_truth_state_status`, which watches
`/finsrov/controller/pose`, `/finsrov/controller/imu`,
`/finsrov/controller/depth`, and `/finsrov/controller/dvl`, then publishes
controller-compatible JSON on `/finsrov/controller/state/status`. You can still
override the profile explicitly with `state_input_mode:=raw_fusion` or
`state_input_mode:=sim_truth` when debugging launch topology.

## Coordinate Frames

`motion_controller` consumes controller-compatible coordinates. For Unity
simulation these are produced directly by `VehicleRosBridge`; for the real
vehicle, `controller_state_adapter` converts physical pool/depth and body FLU
state into the same controller-native contract before publishing
`/finsrov/controller/*`.

The real-vehicle stack may still use this physical convention upstream:

* `pool_world` is right-handed and z-up
* `pool_world.z=0` is the pool-bottom center, while pressure depth is positive
  downward from the water surface
* state estimation can compute body-origin height as:
  `pose.position.z = water_surface_z_m - depth_m - pressure_sensor_offset_world_z`
* body vectors use ROS-style FLU physical axes before conversion: `x` forward,
  `y` left, and `z` up

The Unity-trained controllers keep their original model convention:

* controller/model `x`: forward
* controller/model `y`: up
* controller/model `z`: left

For real-vehicle raw topics, the conversion contract is:

```yaml
ros_to_controller_basis_indices: [0, 2, 1]
ros_to_controller_basis_signs: [1.0, 1.0, 1.0]
ros_to_controller_position_offset: [0.0, 0.0, 0.0]
```

With the default mapping:

```text
controller.x = ros.x
controller.y = ros.z
controller.z = ros.y
```

For positions, the default offset is zero:

```text
p_controller = B * p_ros
```

This keeps controller topics in the Unity-trained convention: `controller.y=0`
is the water surface and `controller.y=-2` is 2 m below it. For the real
vehicle, `controller_state_adapter` applies this conversion exactly once before
publishing `/finsrov/controller/*`. `motion_controller` itself expects these
topics to already be controller-native.

Vectors are not translated. However, ordinary (polar) vectors and angular
velocity (an axial/pseudovector) must not use the same transform when the basis
contains a reflection. Let `B` be the configured ROS -> controller basis. Use:

```text
position, linear velocity, linear acceleration: v_controller = B * v_ros
angular velocity and angular-rate covariance:    w_controller = A * w_ros
                                                A = det(B) * B
```

For the default basis, `det(B)=-1`, so:

```text
[wx, wy, wz]_controller = [-wx, -wz, -wy]_ros
```

In particular, controller yaw rate is `wy_controller = -wz_ros`. The same
`A` rule applies to angular covariance; for a mixed 6D covariance use
`diag(B, A)` rather than `diag(B, B)`. Body-frame command vectors are already
expressed in controller coordinates and are passed through unchanged.

Orientation is still transformed as a rotation matrix:

```text
R_controller = B * R_ros * B^T
```

Do not swap quaternion components by hand. The raw state-estimation topics
(`/finsrov/pose`, `/finsrov/imu_link`, `/finsrov/dvl_link`) remain in the ROS
physical convention. `controller_state_adapter` applies the conversion exactly
once and publishes controller-native `/finsrov/controller/*` topics. Unity
sim-truth should publish the latter directly and must not pass through this
adapter a second time.

Command `header.frame_id` is strict:

* absolute commands must use `controller_world`
* relative/body commands must use `controller_body`
* `world`, `body`, `pool_world`, empty frame, `unity_*`, and `model_*` are rejected

The controller subscribes controller-native state only. Real raw fusion topics
must pass through `controller_state_adapter`; Unity sim-truth must publish
`/finsrov/controller/*` directly. The adapter is frame-aware:

* raw `pool_world` / `finsrov_base_link` inputs are remapped into controller convention
* `controller_world` / `controller_body` inputs are passed through unchanged

See [Controller Coordinate Contract](docs/controller_coordinate_contract.zh-CN.md)
for the full topic-by-topic convention.

## Command Topics

Default input topics:

* `/motion_controller/command/position_controller_world`
* `/motion_controller/command/position_controller_body`
* `/motion_controller/command/pose_controller_body`
* `/motion_controller/command/velocity_controller_body`
* `/motion_controller/command/cancel`

Default status topics:

* `/motion_controller/status/active_pose`
* `/motion_controller/status/error_body`
* `/motion_controller/status/reached`

Position goals are marked reached only after the position/velocity/orientation
tolerances remain satisfied for `goal_reach_hold_time_sec`. With
`stop_on_goal_reached: true`, the controller then clears the active command and
publishes zero thrusters. With `stop_on_goal_reached: false`, it keeps the
active command and continues closed-loop hold control while publishing
`reached=true`.

Default reset interfaces:

* service: `/motion_controller/reset_vehicle`
* topic pulse forwarded to Unity: `/finsrov/reset`

## Position And Pose Commands

`send_position_goal` always publishes once per call for `count` times, then exits.
It does not keep a node alive forever.

Controller-world absolute position. `--frame controller_world` publishes
`header.frame_id=controller_world`, so `x/y/z` are Unity-trained controller
coordinates:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_position_goal \
  --frame controller_world --x 1.5 --y -0.5 --z 0.0
```

This means:

* `x=1.5`: controller/Unity forward position
* `y=-0.5`: 0.5 m below the controller water-surface origin
* `z=0.0`: controller/Unity lateral position

### Manually send a registered T2 trajectory

`send_position_goal` remains the static point-goal tool above.  To manually
exercise a named T2 trajectory without starting the recorder, use the separate
`send_t2_trajectory_goal` executable.  It reads the same experiment YAML as
`run_t2_experiment`, validates workspace/speed/acceleration limits, and sends
one complete `MultiDOFJointTrajectory` in `controller_world`.  It does not
enable the hardware bridge, start a recorder, or create an experiment manifest.

```bash
cd ./ros2_ws

./scripts/run_ros2_uv.sh ros2 run motion_control send_t2_trajectory_goal \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --trajectory circle \
  --dry-run
```

After the controller/state stack is running and the normal hardware safety
checks are complete, omit `--dry-run` to publish the trajectory.  `--count`
repeats the complete ROS trajectory message for delivery reliability; it does
not repeat the physical circle/ellipse.  The controller executes the declared
duration/turn count once.  T2 currently carries an identity quaternion only as
a ROS transport field: the PID T2 profile controls `x/y(depth)/z` and does not
control roll, pitch, or yaw.

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_t2_trajectory_goal \
  --config src/experiment_recorder/config/t2_hardware_experiment.yaml \
  --trajectory circle
```

Controller-body relative position. `--frame controller_body` publishes
`header.frame_id=controller_body`, so `x/y/z` are offsets from the current
vehicle pose in the controller body convention:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_position_goal \
  --frame controller_body --x 2.0 --y 0.0 --z 0.0
```

Body-frame axis convention used by the helper CLI/controller model:

* `x`: forward
* `y`: up
* `z`: left

This is the Unity-trained controller/model body convention, not the raw ROS
`finsrov_base_link` convention. ROS `finsrov_base_link` is FLU:
`x` forward, `y` left, `z` up. The adapter remaps ROS vectors with
`controller.x = ros.x`, `controller.y = ros.z`, `controller.z = ros.y`.

So:

* `x > 0` means move forward
* `y > 0` means move up
* `z < 0` means move right

## Yaw Pitch Roll For Position Goals

`send_position_goal` now supports:

* `--yaw`
* `--pitch`
* `--roll`

These are angles in degrees, not angular rates.

Controller-world absolute pose target example:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_position_goal \
  --frame controller_world --x 1.5 --y -0.5 --z 0.0 \
  --yaw 90 --pitch 0 --roll 0
```

For traditional PID position plus yaw evaluation, start the controller with:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control motion_controller --ros-args \
  --params-file ./ros2_ws/src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml
```

Then send manual references with either `send_position_goal` or the clearer alias:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_position_yaw_goal \
  --frame controller_world --x 1.5 --y -0.5 --z 0.0 --yaw 90
```

Semantics:

* `--frame controller_world`: position is absolute in `controller_world` / Unity coordinates; `yaw/pitch/roll` are absolute target orientation in controller world frame
* `--frame controller_body`: position is a relative offset in `controller_body`; `yaw/pitch/roll` are relative body-frame orientation offsets from the current vehicle attitude

For the common real-boat command:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_position_goal \
  --frame controller_world --x 0.0 --y -0.5 --z 0.0 --yaw 90
```

the controller stores this exact target:

```text
target_controller = [0.0, -0.5, 0.0]
target_yaw_controller = 90 deg
```

It does not reinterpret that command as ROS `pool_world`. Real `/finsrov/pose`
is converted by `controller_state_adapter` into `/finsrov/controller/pose`
before the error is computed.

Body-frame relative pose example:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_position_goal \
  --frame controller_body --x 2.0 --y 0.0 --z 0.0 \
  --yaw 45 --pitch 0 --roll 0
```

Orientation targets are only used when both are true:

* YAML sets `use_target_orientation: true`
* the selected backend is marked as supporting target orientation

At the moment, ROS2-side target-orientation support is enabled for:

* `traditional_pid_position`
* `ppo_control_for_pose`
* `ppo_control_v2_for_pose`

Backend-specific behavior:

* `traditional_pid_position`: only yaw is used; pitch and roll are ignored with a warning
* `ppo_control_for_pose`: full target quaternion is used
* `ppo_control_v2_for_pose`: full target quaternion is used

For other backends, the command is still accepted but the node warns and ignores the orientation part.

Goal completion then checks:

* position error
* linear speed magnitude
* orientation error in degrees if orientation targeting is enabled

The tolerance is controlled by `orientation_tolerance_deg`.

## Velocity Commands

`motion_controller` can be switched between two control-source modes:

* `ros_manual`
  * ROS2 velocity/pose commands are accepted
  * the controller publishes thrusters on `thruster_topic`
* `unity_random`
  * ROS2 velocity/pose commands are ignored
  * the controller stops publishing thrusters so Unity-side random-reference control can own the vehicle

Switch modes:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control set_control_mode ros_manual
./scripts/run_ros2_uv.sh ros2 run motion_control set_control_mode unity_random
```

Unity must match the selected mode. In `ros_manual`, disable Unity-side random
reference/control for the same vehicle and let Unity only publish sensors and
consume `/finsrov/thrusters_out`. In `unity_random`, let Unity own its random
reference/control path and use ROS2 only for monitoring topics.

Velocity command example:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_velocity_command \
  --frame controller_body --vx 0.20 --vy 0.0 --vz 0.0
```

Velocity plus angular-rate example:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_velocity_command \
  --frame controller_body --vx 0.20 --vy 0.0 --vz 0.0 \
  --wy 0.30 --wp 0.0 --wr 0.0
```

Angular CLI parameters are rates:

* `--wy`: yaw rate
* `--wp`: pitch rate
* `--wr`: roll rate

These are not target angles.

If you want terminal velocity commands to be respected directly, choose a
velocity backend such as:

* `traditional_pid_velocity`
* `traditional_pid_ff_eso_velocity`
* `ppo_control_v2_for_velocity`

Position-only backends reject direct velocity commands on purpose.

## Cancel And Reset

Cancel the active motion command:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control cancel_goal
```

`cancel_goal` cancels whichever command is active, whether it came from
`send_position_goal` or `send_velocity_command`. The old
`cancel_position_goal` command is still installed as a compatibility alias.

Call the reset service with the helper CLI:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control reset_service --ros-args \
  --params-file ./ros2_ws/src/motion_control/config/FinsROV/traditional_pid_position.yaml
```

Then call it:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control reset_vehicle
```

Or call the ROS2 service directly:

```bash
./scripts/run_ros2_uv.sh ros2 service call /motion_controller/reset_vehicle std_srvs/srv/Trigger "{}"
```

Reset behavior:

* `reset_service` publishes cancel to `/motion_controller/command/cancel`
* `reset_service` publishes zero thrusters to the configured `thruster_topic`
* `reset_service` publishes a reset pulse on the configured `reset_topic`
* Unity-side `FinsROVResetRosBridge` receives that pulse and restores the vehicle to its initial position, rotation, linear velocity, and angular velocity

## Direct 6D Wrench Meta-Action Test

For a hardware test that bypasses policy inference but uses the same deployment
path as the wrench controller, use `send_wrench_action`. The six values are a
normalized meta-action in this order:

```text
[surge, sway, heave, roll, pitch, yaw]
```

在当前 FinsROV body frame 下：

```text
local X = forward
local Y = up
local Z = left
```

因此 6D action 的方向是：

| Action | Unity 机体轴 | 正方向 |
|---|---|---|
| `surge` | `+X` | 向前 |
| `sway` | `+Z` | 向左 |
| `heave` | `+Y` | 向上 |
| `roll` | 绕 `X` 轴 | 滚转 |
| `pitch` | 绕 `Z` 轴 | 俯仰 |
| `yaw` | 绕 `Y` 轴 | 偏航 |

映射关系：

```text
[surge, sway, heave, roll, pitch, yaw]
= [Fx,    Fz,   Fy,    Mx,  Mz,   My]
```

这是**机体系**，不是 `controller_world`。正负方向均随潜器自身姿态变化。

注意：标准 Fossen 中 `w/heave` 通常定义为向下，而当前 RL action 的 `heave` 定义为 Unity `+Y` 向上，所以：

```text
heave +0.2 -> 上浮
heave -0.2 -> 下潜
```

The sender applies the same sequence as `motion_controller`:

```text
6D action -> FinsROV allocator-body wrench
           [Fx, Fy, Fz, Mx, My, Mz] = [forward, up, left, roll, yaw, pitch]
           -> ThrustAllocator(matrix)
           -> per-thruster force-N limits
           -> /finsrov/thrusters_out
```

The default limits match `config/FinsROV/ppo_wrench_for_pose.yaml`. A config
can be supplied explicitly to prevent the test profile from drifting:

```bash
./scripts/run_ros2_uv.sh ros2 run motion_control send_wrench_action \
  --action 0.20 0 0 0 0 0 \
  --duration 5 --ramp-time 0.5 \
  --config src/motion_control/config/FinsROV/ppo_wrench_for_pose.yaml
```

Before running it, stop `motion_controller` or make sure it is not running.
The controller publishes zero output while idle, so leaving it alive creates
two publishers competing for `/finsrov/thrusters_out`. The sender publishes at
20 Hz by default and sends several zero-force messages on normal exit or
`Ctrl-C`. `--ramp-time 0` gives an immediate step; keep the default ramp for a
first real-vehicle test.

Examples for individual axes:

```bash
# Forward surge, normalized action +0.20
./scripts/run_ros2_uv.sh ros2 run motion_control send_wrench_action \
  --action 0.20 0 0 0 0 0 --duration 5

# Positive yaw, normalized action +0.20
./scripts/run_ros2_uv.sh ros2 run motion_control send_wrench_action \
  --action 0 0 0 0 0 0.20 --duration 5

# Simultaneous six-axis action
./scripts/run_ros2_uv.sh ros2 run motion_control send_wrench_action \
  --action 0.10 -0.10 0.05 0 0 0.15 --duration 3
```

The wrench is always in the vehicle body frame. It is not
`controller_world`, and it is not the standard Fossen/SNAME tuple
`[X,Y,Z,K,M,N]`. With the current wrench profile, `--action ... 0.20` on yaw
means a target body yaw moment `My=+0.020 Nm` before allocation. The node
prints the exact allocator-body wrench, normalized eight-thruster output, and
final force-N command before starting the test. This is a force command test,
not a direct PWM test.

## Key YAML Parameters

Common parameters you are likely to edit:

* `backend_name`
* `checkpoint_path`
* `control_rate_hz`
* `policy_action_rate_limit_per_sec`: optional normalized policy-action rate
  limit. A positive value ramps each new command from zero and limits changes
  before wrench allocation; `0` preserves the historical unrestricted output.
* `thruster_topic`: preferred output topic for 8-value thruster commands; `pwm_topic` remains a compatibility alias
* `thruster_output_mode`: `normalized_direct` keeps the old action output; `force_n` maps controller/RL action `[-1,1]` to per-thruster force in N before publishing `thruster_topic`
* `thruster_force_limits_n`: positive/negative force limits in canonical thruster order. For V4 Pro1 these limits should come from the command-limited thruster curve in `ros2_ws/data/thruster_curve/finsrov_v4_pro1` / `finsrov_hardware_bridge_v4_pro1.yaml`; do not multiply by another `0.4`, because the fitted RPM range already represents the safe command range.
* `pose_topic`
* `imu_topic`
* `depth_topic`
* `dvl_topic`
* `use_target_orientation`
* `orientation_tolerance_deg`
* `reset_topic`
* `reset_service_name`

FinsROV examples live in:

* [ppo_control_for_pose.yaml](config/FinsROV/ppo_control_for_pose.yaml)
* [traditional_pid_position.yaml](config/FinsROV/traditional_pid_position.yaml)
* [traditional_pid_velocity.yaml](config/FinsROV/traditional_pid_velocity.yaml)

Each profile keeps its backend and checkpoint together. Do not select a backend
or checkpoint separately from the YAML file, because doing so can make the
loaded policy inconsistent with the rest of the profile parameters.

## Runtime Notes

RL and hybrid backends are loaded lazily from `python/finssim_rl`.

If your runtime environment has `stable-baselines3` compiled against NumPy 1.x
but the shell is using NumPy 2.x, those backends will fail to load. The
provided `.venv` scripts are there to avoid that mismatch and keep ROS2 and the
controller package on the same Python interpreter.

`motion_controller` catches runtime exceptions inside the periodic control
loop. If inference or observation construction fails after the node has started,
it clears the active command, publishes zero thrusters, logs the traceback, and
keeps the node alive so you can send a corrected command without restarting the
terminal. Startup failures such as an invalid YAML, missing backend dependency,
or missing checkpoint still stop the node because the controller cannot run in
that state.
