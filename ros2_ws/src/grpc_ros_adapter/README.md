# grpc_ros_adapter

## Introduction

This repo includes functionality for interfacing with Marus over gRPC and publish sensor data over ROS such that it is available to other ROS nodes.

Unity package is maintained in [marus-core](https://github.com/MARUSimulator/marus-core) repository.

Proto messages are maintained in [marus-proto](https://github.com/MARUSimulator/marus-proto) repository.

## Getting started
Recommended ROS distribution is Humble for the ROS2 flow in this workspace.

Setup:
* Clone this repository into your ROS workspace and pull submodules:
`git submodule update --init`
* Clone [uuv_sensor_msgs](https://github.com/labust/uuv_sensor_msgs) into the same workspace.
* Install ROS2 system dependencies such as `cv_bridge`, `python3-opencv`, and `rclpy` through apt.

## Workspace uv environment

This package now relies on the workspace-wide `uv` environment defined at:

```text
/pyproject.toml
```

The shared venv lives at:

```text
/.venv
```

Use the workspace-level wrappers from the repository root:

```bash
cd "${MARUS_ROS2_ROOT}"
./scripts/bootstrap_uv_env.sh
./scripts/colcon_build_uv.sh
./scripts/launch_grpc_ros_adapter.sh
```

The local scripts in `src/grpc_ros_adapter/scripts/` are thin wrappers that
forward to those workspace-level commands.

## Usage and documentation

For usage information and examples visit our [marus-example](https://github.com/MARUSimulator/marus-example) project repository.

For other information and documentation visit our [documentation homepage](https://marusimulator.github.io).

## Unity simulation namespace and connection timing

When Unity simulation and the real FinsROV stack are used at the same time, run
the adapter with a simulation-only topic prefix:

```bash
cd ./ros2_ws
./scripts/launch_grpc_ros_adapter.sh topic_prefix:=/sim
```

Unity scene and prefab fields can stay on their internal addresses, for example
`/finsrov/controller/imu` and `/finsrov/thrusters_out`. The adapter resolves
those absolute ROS topics to `/sim/finsrov/controller/imu` and
`/sim/finsrov/thrusters_out` on the ROS2 side. The real vehicle stack should
remain on `/finsrov/...` without a prefix.

Connection is asynchronous. Unity `RosConnection` retries connection after
Play starts, and `VehicleRosBridge` registers sensor streams and thruster
subscriptions only after `RosConnection` reaches the connected state. It is
normal to see a few Unity-side `DeadlineExceeded` messages if Play starts
before the adapter or if the adapter is restarted. Wait for reconnection before
running tests.

Useful checks:

```bash
cd ./ros2_ws
./scripts/run_ros2_uv.sh ros2 topic list | sort | grep /sim/finsrov
./scripts/run_ros2_uv.sh ros2 topic info /sim/finsrov/thrusters_out -v
./scripts/run_ros2_uv.sh ros2 topic hz /sim/finsrov/controller/imu --window 20
```

Expected checks:

- `/sim/finsrov/thrusters_out` has a `grpc_ros_adapter` subscription.
- `/sim/finsrov/controller/imu`, `/sim/finsrov/controller/dvl`,
  `/sim/finsrov/controller/depth`, and `/sim/finsrov/controller/pose` have
  `grpc_ros_adapter` publishers.
- Unity Editor log contains `Connected to the ROS Server`.
- Unity Editor log contains
  `[VehicleRosBridge] Listening for thrusters on /finsrov/thrusters_out.`

For hydrodynamic identification from Unity controller topics, note that
`VehicleRosBridge` publishes `/controller/imu` angular velocity in the Unity
local body-frame order. Use identity basis mapping in the identifier:

```bash
-p ros_to_body_basis_indices:="[0,1,2]" -p ros_to_body_basis_signs:="[1.0,1.0,1.0]"
```

The real vehicle identification path can still use the real FinsSim mapping
from its YAML, commonly `[x,z,y]`. Mixing these mappings causes yaw response to
land in the wrong recorded column, for example `nu_pitch_z_radps` instead of
`nu_yaw_radps`.

## Credits & Acknowledgements


* [gRPC](https://github.com/grpc/grpc)
* [protobuf](https://github.com/protocolbuffers/protobuf)
* [Gemini Unity simulator](https://github.com/Gemini-team/Gemini)


## Contact
Please feel free to provide feedback or ask questions by creating a Github issue. For other inquiries, please email us at labust@fer.hr or visit our web pages below:
* [Laboratory for Underwater Systems and Technologies - LABUST](https://labust.fer.hr/)

* [University of Zagreb, Faculty of Electrical Engineering and Computing](https://www.fer.unizg.hr/en)

## License
This project is released under the Apache 2.0 License. Please review the [License](https://github.com/MARUSimulator/grpc_ros_adapter/blob/dev/LICENSE) file for more details.
