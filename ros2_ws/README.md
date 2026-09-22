# ros2_ws

Standard ROS 2 workspace for communication across the complete FinsSim stack:
Unity/Isaac simulation, learning and control, perception and state estimation,
experiment tooling, and real-vehicle hardware integration. `grpc_ros_adapter`
is one bridge in this workspace, not its sole purpose.

> **Python environment note:** FinsSim uses ROS 2 Humble for middleware and
> system tooling, but it does **not** run workspace Python nodes in Humble's
> system Python environment. Python dependencies are managed by this
> workspace's internal `uv` `.venv`. Therefore, do not use bare `ros2`,
> `colcon`, or Python commands for this workspace. Run commands through
> `./scripts/run_ros2_uv.sh ...` and builds through
> `./scripts/colcon_build_uv.sh`; for an interactive shell, first source
> `./scripts/source_ros2_uv.sh`.

## Layout

```text
ros2_ws/
  src/
    msgs/
    motion_control/
    hardware_bridge/
    perception/      # C++ runtime + Python monitor/calibration tools
    state_estimation/
    teleop/
    thruster_bridge/
    grpc_ros_adapter/
    uuv_sensor_msgs/
```

`msgs` contains custom ROS2 interfaces used by the real-vehicle debug
and hardware bridge layers. `motion_control` is the controller/RL
inference package that publishes 8 thruster commands; in the real-vehicle
`force_n` configuration those commands are per-thruster force requests in N.
`hardware_bridge` is the real-vehicle full-duplex MCU bridge.
`perception` is the unified real-vehicle perception package: its C++
AprilTag 3 pipeline is the only supported online detector, while Python keeps
the GUI monitor and calibration tools.
`state_estimation` fuses vision, IMU, and depth into
the ROS topics consumed by the controller. `teleop` reads a normal
gamepad with `pygame` and publishes manual per-thruster force commands for
real-vehicle teleoperation. `grpc_ros_adapter` is the ROS2 node and gRPC server
that Unity connects to.
`uuv_sensor_msgs` is the ROS2 port of the underwater sensor message package used by the adapter.

## Documentation map

Use this file for workspace setup, build, and the complete bring-up chain. Each
package owns its own executable, parameter, and troubleshooting documentation:

- [`perception`](src/perception/README.md) and
  [`state_estimation`](src/state_estimation/README.md): vision,
  calibration, and fusion.
- [`motion_control`](src/motion_control/README.md),
  [`teleop`](src/teleop/README.md), and
  [`hardware_bridge`](src/hardware_bridge/README.md): control,
  manual operation, and real-vehicle I/O.
- [`experiment_recorder`](src/experiment_recorder/README.md),
  [`trajectory_data`](src/trajectory_data/README.md), and
  [`apriltag_dataset_collector`](src/apriltag_dataset_collector/README.md):
  experiment and dataset collection.
- [`hydrodynamic_identification`](src/hydrodynamic_identification/README.md)
  and [`thruster_curve_measurement`](src/thruster_curve_measurement/README.md):
  vehicle identification and measurement.
- [`grpc_ros_adapter`](src/grpc_ros_adapter/README.md): Unity–ROS2 connection.

Cross-package contracts, real-hardware procedures, and experiment results live
in the repository-level [`docs/`](../docs/README.md). See
[`ros2_ws/docs/`](docs/README.md) for the boundary between workspace and
project documentation.

## Submodules

This workspace keeps the same upstream dependency model as the original
`grpc_ros_adapter` repository:

```text
src/grpc_ros_adapter                          -> MARUSimulator/grpc_ros_adapter
src/grpc_ros_adapter/grpc_ros_adapter/protobuf -> labust/labustsim-proto
src/uuv_sensor_msgs                            -> labust/uuv_sensor_msgs
```

After cloning the workspace repository, initialize them with:

```bash
git submodule update --init --recursive
```

`src/grpc_ros_adapter` is the upstream MARUS adapter submodule with local ROS2
packaging/import fixes in its worktree. `uuv_sensor_msgs` is based on the
upstream `noetic` branch, with local ROS2 Humble port changes in the submodule
worktree so it can be built by `colcon`.

## Python Environment

`ros2_ws` uses its own `uv`-managed `.venv` and should be treated as the only
Python runtime for this workspace.

Why this matters:

* ROS2 codegen uses the Python interpreter seen by `colcon` and CMake.
* `ros2 run` executes generated entry scripts, so a wrong shebang can silently
  jump back to `/usr/bin/python3`.
* user-site packages such as `~/.local/lib/python3.10/site-packages` can leak
  incompatible `numpy`, `matplotlib`, `stable-baselines3`, or `torch`.

The workspace scripts now solve this in one place:

* `.venv` is created by `uv`
* `PYTHONNOUSERSITE=1` is exported at runtime
* `COLCON_PYTHON_EXECUTABLE` and `Python3_EXECUTABLE` are forced to `.venv`
* generated ROS2 entry scripts are rewritten to use `.venv/bin/python`

If you already had an older `.venv` created with `--system-site-packages`,
`./scripts/bootstrap_uv_env.sh` will recreate it automatically.

## One-Time Setup

```bash
cd ./ros2_ws
./scripts/bootstrap_uv_env.sh
./scripts/colcon_build_uv.sh --cmake-clean-cache
```

After that, use one of these two entry patterns.

## Recommended Usage

Open a shell with the workspace environment loaded:

```bash
cd ./ros2_ws
source ./scripts/source_ros2_uv.sh
```

Or run a single command inside the correct environment:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic list
```

## Build

Always prefer the wrapper:

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh
```

This wrapper loads ROS Humble, activates `.venv`, disables user-site imports,
forces the package build interpreter to `.venv/bin/python`, keeps `colcon`
itself on the ROS system toolchain for Humble compatibility, and patches
installed ROS2 entry scripts so `ros2 run` stays inside the workspace
environment.

## Native Toolchain And OpenCV

This machine has multiple native toolchains and OpenCV installations:

```text
/usr/local/lib/libopencv_*.so.4.12.0      # desired OpenCV for this workspace
/usr/lib/x86_64-linux-gnu/libopencv_*.so  # Ubuntu/OpenCV 4.5.x
/home/linuxbrew/.linuxbrew/bin/ld         # Homebrew linker, not suitable here
/usr/bin/ld                               # Ubuntu linker, expected
```

For this workspace we intentionally use `/usr/local` OpenCV 4.12, but the
linker must be the Ubuntu linker. If Homebrew `ld` is first in `PATH`, ROS2 C++
packages may fail with errors such as:

```text
undefined reference to `avcodec_send_frame@LIBAVCODEC_58'
undefined reference to `TIFFReadRGBATile@LIBTIFF_4.0'
undefined reference to `spdlog::...'
```

Those messages do not mean OpenCV 4.12 is unusable. They mean the wrong linker
is resolving the transitive Ubuntu/ROS libraries. The workspace wrappers fix the
environment locally:

```text
PATH            -> puts /usr/bin before Homebrew binutils
OpenCV_DIR      -> /usr/local/lib/cmake/opencv4
PKG_CONFIG_PATH -> /usr/local/lib/pkgconfig plus Ubuntu pkgconfig dirs
LD_LIBRARY_PATH -> /usr/local/lib plus ROS and Ubuntu library dirs
PYTHONPATH      -> ros2_ws/.venv site-packages first
```

Check the active wrapper environment:

```bash
cd ./ros2_ws
source ./scripts/source_ros2_uv.sh --skip-install-setup --
which ld
pkg-config --modversion opencv4
echo "$OpenCV_DIR"
```

Expected:

```text
/usr/bin/ld
4.12.0
/usr/local/lib/cmake/opencv4
```

After building, verify the native AprilTag node links to OpenCV 4.12:

```bash
ldd install/perception/lib/perception/direct_apriltag_node | grep opencv
```

Expected paths should start with:

```text
/usr/local/lib/libopencv_...
```

Do not run bare `colcon build` from an arbitrary interactive shell on this
machine; it can inherit Homebrew binutils, STM32Cube CMake, old OpenCV paths, or
system Python packages. Use:

```bash
./scripts/colcon_build_uv.sh --cmake-clean-cache
```

注：受限于ROS2 Humble对于 `.venv`的支持，我们只能使用 `colcon_build_uv.sh`进行处理达到最好的效果，同时不支持 `--symlink-install`，因为这可能会影响后面修改 `python`到新的 `.venv`的shebang指定的问题。

## Run

Start the gRPC adapter:

```bash
cd ./ros2_ws
./scripts/launch_grpc_ros_adapter.sh
```

Run the motion controller:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run motion_control motion_controller --ros-args \
  --params-file ./ros2_ws/src/motion_control/config/FinsROV/ppo_control_for_pose.yaml
```

Run any other ROS2 command the same way:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /FinsROV/thrusters_out
```

Run joystick teleoperation for the real vehicle:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch teleop joystick_teleop.launch.py \
  params_file:=src/teleop/config/finsrov_gamepad_v4_pro1.yaml
```

`teleop` publishes `/finsrov/thrusters_out` as 8 canonical
per-thruster force values in N, so the hardware bridge should use
`command_mode=force_n`. Stop `motion_controller` and other test publishers
before enabling teleop. Details are in
`src/teleop/docs/joystick_teleop.zh-CN.md`.

## Real FinsROV Hardware Pipeline

The real-vehicle pipeline makes the controller/RL interface controller-native:

```text
camera -> perception -> /finsrov/vision/refracted_pose_6d
MCU telemetry -> hardware_bridge -> /finsrov/hardware/imu_raw, /finsrov/hardware/depth_raw
vision + IMU + depth -> state_estimation -> /finsrov/pose, /finsrov/imu_link, /finsrov/depth_link, /finsrov/dvl_link
pool_world state -> controller_state_adapter -> /finsrov/controller/pose, /finsrov/controller/imu, /finsrov/controller/depth, /finsrov/controller/dvl
motion_control reads /finsrov/controller/* -> /finsrov/thrusters_out -> hardware_bridge -> MCU DSHOT RPM control
```

For real hardware, `/finsrov/thrusters_out` is interpreted by
`hardware_bridge` according to `command_mode`:

```text
force_n              input is force in N; bridge maps force -> target RPM -> normalized MCU command
normalized_rpm       input is [-1, 1] target-RPM fraction; useful for motor-order/sign tests
normalized_throttle  legacy [-1, 1] throttle-style compatibility mode
```

The default V4 Pro configs use `command_mode: force_n` with
`firmware_max_rpm: 3000.0`, matching the FineSUB firmware
`kDirectThrusterMaxRpm`. Horizontal thruster curves come from
`ros2_ws/data/thruster_curve/M005..M008_rpm_sweep`; vertical curve entries are
explicit placeholders until vertical thruster sweeps are measured.

With the current FineSUB firmware, the framed MCU telemetry also appends eight
motor RPM values. The bridge publishes them as:

```text
/finsrov/hardware/motor_rpm_raw -> std_msgs/Float32MultiArray
```

The RPM topic uses the same Unity canonical order as `/finsrov/thrusters_out`:

```text
[V_LF, V_LB, V_RB, V_RF, H_LF, H_RF, H_RB, H_LB]
```

The bridge converts MCU physical-order RPM `[M1..M8]` back through
`motor_order` and `motor_signs`. `/finsrov/hardware/status` also reports
`last_motor_rpm_mcu` for raw MCU-order debugging. Older 52-byte telemetry
firmware is still accepted; in that case this topic publishes zeros.

Build the real-vehicle packages:

```bash
cd ./ros2_ws
./scripts/colcon_build_uv.sh --packages-select \
  msgs hardware_bridge perception state_estimation motion_control
```

Start the hardware bridge in dry-run mode first:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run hardware_bridge hardware_bridge --ros-args \
  -p transport:=dry_run
```

For the current Ethernet hardware link, set the PC wired interface to
`192.168.0.10/24`; the lower computer is `192.168.0.2`.

Temporary IP setup example:

```bash
sudo ip addr add 192.168.0.10/24 dev enp2s0
ping -c 3 192.168.0.2
```

Start the real UDP bridge after the lower computer is online:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch hardware_bridge hardware_bridge.launch.py
```

Check hardware telemetry:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/hardware/imu_raw
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/motor_rpm_raw
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status --once --full-length
```

The hardware bridge starts disarmed by default and sends one disabled zero
frame on startup. This prevents an existing `/finsrov/thrusters_out` publisher
from moving the vehicle as soon as the bridge is launched.

Arm or disarm the downlink explicitly:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled true
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled false
```

Default hardware bridge parameters are in:

```text
src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml
```

Thruster layout mapping is documented in:

```text
../docs/hardware_bridge_thruster_mapping_zh-CN.md
```

Current bridge defaults map Unity canonical thrusters to MCU physical outputs
with:

```text
motor_order: [4, 5, 0, 1, 2, 7, 6, 3]
motor_signs: [-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
```

The default real-vehicle transport is UDP:

```text
bind:   0.0.0.0:54321
remote: 192.168.0.2:58766
```

The bridge subscribes to `/finsrov/thrusters_out`, sends 37-byte
`0xAA ... 0xBB` thruster frames, and publishes:

```text
/finsrov/hardware/imu_raw
/finsrov/hardware/depth_raw
/finsrov/hardware/motor_rpm_raw
/finsrov/hardware/thruster_cmd_echo
/finsrov/hardware/status
```

Check the receive side:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/hardware/imu_raw
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/hardware/depth_raw
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/motor_rpm_raw
```

Enable MCU-side thruster command echo dynamically:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge debug_mask 1
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge debug_echo_decimation 1
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/thruster_cmd_echo
```

`debug_mask` is a bitmask. Currently:

```text
0x00000001: thruster command echo from MCU
```

`debug_echo_decimation=10` means the MCU echoes every 10th valid thruster
command.

When echo is enabled, `/finsrov/hardware/thruster_cmd_echo` is the primary
verification topic for the downlink. Useful fields:

```text
received_thrust  -> MCU parsed from the host command frame
applied_thrust   -> thrust values the MCU reports as actually applied
accepted         -> whether the command passed MCU-side checks
reject_flags     -> MCU-side rejection / placeholder reason bits
```

Current firmware-side protection for direct thruster execution:

```text
ROS2 force_n mode: host force_N -> bridge target RPM -> MCU normalized RPM command
MCU RPM mode: host +/-1.0 -> firmware target +/-3000 RPM
MCU command timeout stop: 500 ms without a fresh thruster command
```

Quick verification:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge enabled true
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/thruster_cmd_echo
./scripts/run_ros2_uv.sh ros2 topic pub --once /finsrov/thrusters_out std_msgs/msg/Float32MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]}"
```

In `force_n` mode the example above requests `+0.2 N` on canonical thruster
`H_LF`. For force-mode tests, use the unified sender with `--command-mode force_n`
or `--force`:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test \
  index --index 4 --force 0.2 --rate 20 --duration 3

./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test \
  forward --command-mode force_n --force 0.6 --rate 20 --duration 3
```

For motor mapping/sign tests that bypass force curves, switch the bridge to
`normalized_rpm` first:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge command_mode normalized_rpm
./scripts/run_ros2_uv.sh ros2 run hardware_bridge send_thruster_test forward --amplitude 0.2
```

Disable all debug echo again with:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 param set /hardware_bridge debug_mask 0
```

Send a safe zero-thruster test frame:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic pub --once /finsrov/thrusters_out std_msgs/msg/Float32MultiArray \
  "{data: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}"
```

Serial is still available for bench tests:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run hardware_bridge hardware_bridge --ros-args \
  --params-file src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml \
  -p transport:=serial \
  -p serial_port:=/dev/ttyUSB0 \
  -p baudrate:=115200
```

## Perception Module

The unified calibration entry is now:

```text
../docs/calibration/index.zh-CN.md
```

Use it for camera intrinsics, AprilTag family/id/size, homography debug
calibration, `T_world_camera`, `T_body_tag`, and coordinate-system validation.
This README only keeps launch and quick-check commands.

Start the high-performance C++ AprilTag 3 perception node:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag.launch.py
```

For the infrared camera profile:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch perception direct_apriltag_ir.launch.py
```

The native node does not publish high-frequency `/finsrov/camera/image_raw`.
It directly reads the camera in C++, runs AprilTag 3, publishes lightweight
pose/status topics, and optionally publishes a low-rate JPEG debug stream:

```text
/finsrov/vision/refracted_pose_6d
/finsrov/vision/pose_3d_camera
/finsrov/vision/status
/finsrov/camera/status
/finsrov/camera/debug/compressed
```

Start the PyQt6 monitor in another terminal. It subscribes to the compressed
debug image, so it does not sit in the real-time detection path:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run perception perception_monitor
```

The Python camera capture and AprilTag tracking chain remains available for
debugging and fallback:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch perception perception.launch.py
```

Important configs and calibration files:

```text
src/perception/config/camera.yaml
src/perception/config/apriltag_tracker.yaml
src/perception/calibration/pool_homography.yaml
src/perception/calibration/rgb_camera.yaml
src/perception/calibration/ir_apriltag_camera.yaml
src/state_estimation/config/pool_world.yaml
src/state_estimation/config/state_fusion_extrinsics.yaml
```

Detailed calibration and algorithm docs:

```text
../docs/calibration/index.zh-CN.md
src/perception/README.zh-CN.md
src/perception/docs/refractive_apriltag_pose_algorithm.zh-CN.md
src/perception/README.zh-CN.md
src/perception/docs/rgb_camera_apriltag_calibration.zh-CN.md
src/state_estimation/docs/world_camera_extrinsic_calibration.zh-CN.md
src/state_estimation/docs/t_body_tag_extrinsic.zh-CN.md
src/state_estimation/docs/fusion_pipeline.zh-CN.md
```

`pool_homography.yaml` is only for the legacy Python 2D debug path. The current
real-vehicle 6D path uses native AprilTag/PnP/refractive output plus
`T_world_camera`, `T_body_tag`, IMU, and pressure depth.

Check whether the native camera pipeline is alive:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/camera/status
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/vision/status
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/camera/debug/compressed
```

Check what AprilTag data is detected:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/status
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/refracted_pose_6d
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/pose_3d_camera
```

Start state fusion:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 launch state_estimation state_fusion.launch.py
```

The current real-vehicle state fusion path:

```text
/finsrov/vision/refracted_pose_6d    # pool_world visual pose
/finsrov/hardware/depth_raw          # positive-down pressure depth
/finsrov/hardware/imu_raw            # IMU quaternion, gyro, accel
        ↓
state_estimation EKF
        ↓
/finsrov/pose
/finsrov/dvl_link
/finsrov/imu_link
```

Core transform chain:

```text
T_world_body = T_world_camera * T_camera_tag * inverse(T_body_tag)
```

The fused topics match the existing `motion_controller` config:

```text
/finsrov/pose
/finsrov/imu_link
/finsrov/depth_link
/finsrov/dvl_link
```

Coordinate conventions and calibration validation are documented in
`../docs/calibration/coordinate-system.zh-CN.md` and
`../docs/calibration/validation-checklist.zh-CN.md`.

Run the motion controller against controller-native real-vehicle topics:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 run motion_control motion_controller --ros-args \
  --params-file ./ros2_ws/src/motion_control/config/FinsROV/traditional_pid_position_yaw.yaml
```

Useful checks:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/vision/tag_poses_3d_camera --once
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/pose
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/state/status
./scripts/run_ros2_uv.sh ros2 topic hz /finsrov/controller/pose
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/controller/state/status
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/hardware/status
./scripts/run_ros2_uv.sh ros2 topic echo /finsrov/thrusters_out
```

`/finsrov/state/status` includes sensor freshness, used tag IDs, EKF covariance,
and the last reject reason. If the status says `unknown_tag_*`, add that tag ID
to `state_fusion_extrinsics.yaml`. If it says `*_mahalanobis_gate`, the
measurement was rejected as a jump relative to the current EKF state.

The MCU telemetry protocol is:

```text
0xA5 0x5A + type + version + payload_len + seq + mcu_time_ms + payload + CRC16-MODBUS
```

Payload v1 is little-endian:

```text
quat_wxyz[4] float32
angular_velocity_xyz[3] float32
linear_acceleration_xyz[3] float32
depth_m float32
pressure_pa float32
status_flags uint32
motor_rpm[8] float32  # MCU physical order [M1..M8]; bridge publishes canonical order
```

The bridge also publishes the atomic calibration topic
`/finsrov/hardware/telemetry` (`msgs/msg/HardwareTelemetry`). It uses one shared
MCU-to-ROS clock mapping for the IMU, depth, and RPM fields in each packet. This topic is
for hydrodynamic/thruster identification and diagnostics; `/finsrov/hardware/imu_raw`
keeps its host-receive timestamp so the existing fusion freshness behavior is unchanged.

## What Each Script Does

* `scripts/bootstrap_uv_env.sh`
  creates or recreates a clean workspace `.venv`, then runs `uv sync`
* `scripts/source_ros2_uv.sh`
  source this into the current shell to load ROS, `.venv`, `install/setup.bash`,
  `PYTHONNOUSERSITE=1`, and `FINSSIM_REPO_ROOT`
* `scripts/run_ros2_uv.sh`
  executes one command inside that same prepared environment
* `scripts/colcon_build_uv.sh`
  builds the workspace and patches generated ROS2 Python entrypoints to `.venv`
* `scripts/launch_grpc_ros_adapter.sh`
  launches the MARUS ROS2 gRPC adapter in the same clean environment

## Manual Recovery

If you ever suspect the workspace was built with the wrong interpreter:

```bash
cd ./ros2_ws
rm -rf build install log
./scripts/bootstrap_uv_env.sh
./scripts/colcon_build_uv.sh --cmake-clean-cache
```

Unity `RosConnection` should point to:

```text
serverIP: localhost
serverPort: 30052
```

For a Unity vehicle named `BlueROV2`, publish PWM commands on:

```bash
ros2 topic pub -r 20 /bluerov2/pwm_out std_msgs/msg/Float32MultiArray \
  "{data: [0.15, 0.15, 0.15, 0.15]}"
```

The full guide is in:

```text
src/grpc_ros_adapter/docs/marus_unity_ros2_userguide.html
```
