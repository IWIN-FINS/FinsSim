from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")
    pool_world_params_file = LaunchConfiguration("pool_world_params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("state_estimation"), "config", "state_fusion.yaml"]
                ),
            ),
            DeclareLaunchArgument(
                "pool_world_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("state_estimation"), "config", "pool_world.yaml"]
                ),
            ),
            Node(
                package="state_estimation",
                executable="state_fusion",
                name="state_fusion",
                output="screen",
                parameters=[params_file, pool_world_params_file],
            ),
        ]
    )
