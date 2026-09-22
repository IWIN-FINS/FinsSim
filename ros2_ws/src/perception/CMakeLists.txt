cmake_minimum_required(VERSION 3.16)
project(perception)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)
find_package(ament_index_cpp REQUIRED)
find_package(msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(rclcpp REQUIRED)
find_package(rcl_interfaces REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(std_msgs REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc videoio imgcodecs calib3d)
find_package(yaml-cpp REQUIRED)
find_package(PkgConfig REQUIRED)
pkg_check_modules(APRILTAG REQUIRED apriltag)
find_library(APRILTAG_UTILS_LIBRARY NAMES apriltag-utils)

if(TARGET yaml-cpp::yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp::yaml-cpp)
elseif(TARGET yaml-cpp)
  set(YAML_CPP_TARGET yaml-cpp)
else()
  set(YAML_CPP_TARGET ${YAML_CPP_LIBRARIES})
endif()

set(APRILTAG_LINK_LIBRARIES ${APRILTAG_LIBRARIES})
if(APRILTAG_UTILS_LIBRARY)
  list(APPEND APRILTAG_LINK_LIBRARIES ${APRILTAG_UTILS_LIBRARY})
endif()

add_executable(direct_apriltag_node src/direct_apriltag_node.cpp)
target_include_directories(direct_apriltag_node PRIVATE
  ${APRILTAG_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)
target_link_directories(direct_apriltag_node PRIVATE ${APRILTAG_LIBRARY_DIRS})
target_link_libraries(direct_apriltag_node
  ${APRILTAG_LINK_LIBRARIES}
  ${OpenCV_LIBRARIES}
  ${YAML_CPP_TARGET}
)
target_compile_options(direct_apriltag_node PRIVATE ${APRILTAG_CFLAGS_OTHER})
ament_target_dependencies(direct_apriltag_node
  ament_index_cpp msgs geometry_msgs rclcpp rcl_interfaces sensor_msgs std_msgs
)

add_executable(refractive_apriltag_pose_node src/refractive_apriltag_pose_node.cpp)
target_include_directories(refractive_apriltag_pose_node PRIVATE
  ${OpenCV_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)
target_link_libraries(refractive_apriltag_pose_node ${OpenCV_LIBRARIES} ${YAML_CPP_TARGET})
ament_target_dependencies(refractive_apriltag_pose_node
  ament_index_cpp msgs geometry_msgs rclcpp sensor_msgs std_msgs
)

ament_python_install_package(${PROJECT_NAME})

install(TARGETS direct_apriltag_node refractive_apriltag_pose_node
  RUNTIME DESTINATION lib/${PROJECT_NAME}
)
install(PROGRAMS
  scripts/perception_monitor
  scripts/preview_stereo_udp
  scripts/record_stereo_udp_video
  scripts/capture_chessboard_images
  scripts/capture_stereo_chessboard_images
  scripts/calibrate_camera_chessboard
  scripts/calibrate_stereo_chessboard
  scripts/collect_homography_points
  scripts/evaluate_homography_yaml
  scripts/fit_pool_homography_from_e2
  scripts/scan_fiducial_tags
  DESTINATION lib/${PROJECT_NAME}
)
install(DIRECTORY calibration config launch docs
  DESTINATION share/${PROJECT_NAME}
)
install(FILES package.xml README.zh-CN.md
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_cmake_pytest REQUIRED)
  ament_add_pytest_test(test_calibration tests/test_calibration.py)
endif()

ament_package()
