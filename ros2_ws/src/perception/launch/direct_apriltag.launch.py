import os
from pathlib import Path

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def source_or_share_path(*parts):
    repo_root = os.environ.get("FINSSIM_REPO_ROOT")
    if repo_root:
        source_path = Path(repo_root) / "ros2_ws" / "src" / "perception" / Path(*parts)
        if source_path.exists():
            return str(source_path)
    return PathJoinSubstitution([FindPackageShare("perception"), *parts])


def generate_launch_description():
    params = LaunchConfiguration("params")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params",
                default_value=source_or_share_path("config", "direct_apriltag_rgb.yaml"),
            ),
            Node(
                package="perception",
                executable="direct_apriltag_node",
                name="direct_apriltag_node",
                output="screen",
                parameters=[params],
            ),
        ]
    )
