cmake_minimum_required(VERSION 3.10)
project(uuv_sensor_msgs)

find_package(ament_cmake REQUIRED)
find_package(rosidl_default_generators REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/AcousticModemPayload.msg"
  "msg/AcousticModemRange.msg"
  "msg/AcousticModemRequest.msg"
  "msg/AISPositionReport.msg"
  "msg/Depth.msg"
  "msg/DVL.msg"
  "msg/ImagingSonarInfo.msg"
  "msg/SonarFix.msg"
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
