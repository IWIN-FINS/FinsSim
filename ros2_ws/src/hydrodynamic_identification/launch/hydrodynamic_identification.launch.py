from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = "hydrodynamic_identification"


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution(
        [FindPackageShare(PACKAGE_NAME), "config", "hydrodynamic_identification.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="YAML parameter file for hydrodynamic_identifier.",
            ),
            DeclareLaunchArgument(
                "auto_start",
                default_value="false",
                description="Start the excitation sequence immediately.",
            ),
            DeclareLaunchArgument(
                "run_role",
                default_value="identification",
                description="`identification` fits parameters; `held_out_fossen_validation` records CSV only.",
            ),
            DeclareLaunchArgument(
                "axes",
                default_value="surge_x,heave_y,sway_z,roll_x,pitch_z,yaw_y",
                description="Comma-separated axis list, e.g. surge_x or surge_x,yaw_y.",
            ),
            DeclareLaunchArgument(
                "amplitude",
                default_value="0.0",
                description="Single-axis force/torque amplitude override. N for translation, Nm for rotation.",
            ),
            DeclareLaunchArgument(
                "include_negative",
                default_value="true",
                description="Run both positive and negative trials for each positive amplitude.",
            ),
            DeclareLaunchArgument(
                "repeat_per_trial",
                default_value="1",
                description="Repeat every axis/direction/amplitude trial this many times.",
            ),
            DeclareLaunchArgument(
                "pause_between_trials",
                default_value="false",
                description="Wait for Enter or the continue service after each signed trial.",
            ),
            DeclareLaunchArgument(
                "hold_sec",
                default_value="9.0",
                description="Step-axis excitation duration in seconds.",
            ),
            DeclareLaunchArgument(
                "rest_sec",
                default_value="4.0",
                description="Rest duration after each trial in seconds.",
            ),
            DeclareLaunchArgument(
                "heave_y_identification_mode",
                default_value="dive_and_coast",
                description="Heave mode: step or dive_and_coast.",
            ),
            DeclareLaunchArgument(
                "heave_y_coast_sec",
                default_value="8.0",
                description="Maximum zero-thrust wait for upward heave velocity after a dive.",
            ),
            DeclareLaunchArgument(
                "heave_y_ascent_velocity_threshold_mps",
                default_value="0.01",
                description="Positive controller-y velocity required to start heave ascent sampling.",
            ),
            DeclareLaunchArgument(
                "heave_y_ascent_sample_sec",
                default_value="2.0",
                description="Seconds recorded for fit after true upward heave is detected.",
            ),
            DeclareLaunchArgument(
                "step_fit_start_sec",
                default_value="0.0",
                description="Fit window start offset after step excitation begins.",
            ),
            DeclareLaunchArgument(
                "max_wrench_scale",
                default_value="1.0",
                description="Fraction of thrust curve force limits allowed for allocation.",
            ),
            DeclareLaunchArgument(
                "output",
                default_value="screen",
                description="ROS launch output mode.",
            ),
            Node(
                package=PACKAGE_NAME,
                executable="hydrodynamic_identifier",
                name="hydrodynamic_identifier",
                output=LaunchConfiguration("output"),
                parameters=[
                    LaunchConfiguration("config_file"),
                    {
                        "auto_start": ParameterValue(LaunchConfiguration("auto_start"), value_type=bool),
                        "run_role": LaunchConfiguration("run_role"),
                        "axes": [LaunchConfiguration("axes")],
                        "amplitude": ParameterValue(LaunchConfiguration("amplitude"), value_type=float),
                        "include_negative": ParameterValue(
                            LaunchConfiguration("include_negative"), value_type=bool
                        ),
                        "repeat_per_trial": ParameterValue(
                            LaunchConfiguration("repeat_per_trial"), value_type=int
                        ),
                        "pause_between_trials": ParameterValue(
                            LaunchConfiguration("pause_between_trials"), value_type=bool
                        ),
                        "hold_sec": ParameterValue(LaunchConfiguration("hold_sec"), value_type=float),
                        "rest_sec": ParameterValue(LaunchConfiguration("rest_sec"), value_type=float),
                        "heave_y_identification_mode": LaunchConfiguration("heave_y_identification_mode"),
                        "heave_y_coast_sec": ParameterValue(
                            LaunchConfiguration("heave_y_coast_sec"), value_type=float
                        ),
                        "heave_y_ascent_velocity_threshold_mps": ParameterValue(
                            LaunchConfiguration("heave_y_ascent_velocity_threshold_mps"), value_type=float
                        ),
                        "heave_y_ascent_sample_sec": ParameterValue(
                            LaunchConfiguration("heave_y_ascent_sample_sec"), value_type=float
                        ),
                        "step_fit_start_sec": ParameterValue(
                            LaunchConfiguration("step_fit_start_sec"), value_type=float
                        ),
                        "max_wrench_scale": ParameterValue(
                            LaunchConfiguration("max_wrench_scale"), value_type=float
                        ),
                    },
                ],
            ),
        ]
    )
