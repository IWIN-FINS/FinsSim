# ROS Third-Party Code

The ROS2 workspace is managed directly by the FinsSim repository. It currently includes code originally imported from these upstream projects:

- `ros2_ws/src/grpc_ros_adapter`
  - Upstream: `https://github.com/MARUSimulator/grpc_ros_adapter`
  - License: Apache-2.0
  - License file: `ros2_ws/src/grpc_ros_adapter/LICENSE`

- `ros2_ws/src/uuv_sensor_msgs`
  - Upstream: `https://github.com/labust/uuv_sensor_msgs`
  - License: BSD-3-Clause (the package metadata and LICENSE text agree)
  - License file: `ros2_ws/src/uuv_sensor_msgs/LICENSE`

- `ros2_ws/src/grpc_ros_adapter/grpc_ros_adapter/protobuf`
  - Upstream proto project: `https://github.com/MARUSimulator/marus-proto` (the former `labust/labustsim-proto` URL redirects here).
  - License: Apache-2.0. The upstream default branch at audit time is `69a3b9188c3ff4def9c590061d30eefc57ab1f38`; its generated Python branch is `ba41c776f845365f9f09edbb291eb658206bfba3`.
  - The absorbed checkout contains an older generated Python revision whose exact source commit was not recorded. `protobuf/NOTICE.md` records this provenance boundary; preserve it and regenerate from a pinned upstream `.proto` commit before materially changing the generated modules.

When modifying absorbed third-party code, preserve existing copyright and license notices.
