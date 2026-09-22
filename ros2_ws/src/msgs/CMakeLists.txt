cmake_minimum_required(VERSION 3.10)
project(msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(std_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AprilTagDetection2D.msg"
  "msg/AprilTagDetection2DArray.msg"
  "msg/AprilTagDetection3D.msg"
  "msg/AprilTagDetection3DArray.msg"
  "msg/ThrusterCommandEcho.msg"
  "msg/HardwareTelemetry.msg"
  "msg/TrajectoryEvent.msg"
  "msg/StampedJson.msg"
  DEPENDENCIES geometry_msgs std_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
