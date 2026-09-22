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
    detector_params = LaunchConfiguration("detector_params")
    refractive_params = LaunchConfiguration("refractive_params")
    pool_world_params_file = LaunchConfiguration("pool_world_params_file")
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "detector_params",
                default_value=source_or_share_path("config", "direct_apriltag_ir.yaml"),
            ),
            DeclareLaunchArgument(
                "refractive_params",
                default_value=source_or_share_path("config", "refractive_apriltag_ir.yaml"),
            ),
            DeclareLaunchArgument(
                "pool_world_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("state_estimation"), "config", "pool_world.yaml"]
                ),
            ),
            Node(
                package="perception",
                executable="direct_apriltag_node",
                name="direct_apriltag_node",
                output="screen",
                parameters=[detector_params],
            ),
            Node(
                package="perception",
                executable="refractive_apriltag_pose_node",
                name="refractive_apriltag_pose",
                output="screen",
                parameters=[refractive_params, pool_world_params_file],
            ),
        ]
    )
