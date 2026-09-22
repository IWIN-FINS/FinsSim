from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from launch.substitutions import PathJoinSubstitution


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("thruster_bridge"),
                        "config",
                        "thruster_streamer_bridge.yaml",
                    ]
                ),
                description="Optional YAML parameter file for thruster_streamer_bridge.",
            ),
            Node(
                package="thruster_bridge",
                executable="thruster_streamer_bridge",
                name="thruster_streamer_bridge",
                output="screen",
                parameters=[params_file],
            ),
        ]
    )
