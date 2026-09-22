from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


PACKAGE_NAME = "thruster_curve_measurement"


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _add_optional_arg(arguments: list[str], flag: str, value: str) -> None:
    if value:
        arguments.extend([flag, value])


def _launch_measure_node(context, *args, **kwargs):
    arguments = [
        "--config",
        LaunchConfiguration("config_file").perform(context),
        "--index",
        LaunchConfiguration("index").perform(context),
        "--sensor-port",
        LaunchConfiguration("sensor_port").perform(context),
        "--output-dir",
        LaunchConfiguration("output_dir").perform(context),
        "--motor-name",
        LaunchConfiguration("motor_name").perform(context),
    ]

    _add_optional_arg(arguments, "--csv", LaunchConfiguration("csv").perform(context))
    _add_optional_arg(arguments, "--summary-csv", LaunchConfiguration("summary_csv").perform(context))
    _add_optional_arg(arguments, "--plot", LaunchConfiguration("plot").perform(context))
    _add_optional_arg(arguments, "--points-plot", LaunchConfiguration("points_plot").perform(context))
    _add_optional_arg(arguments, "--fit-plot", LaunchConfiguration("fit_plot").perform(context))
    _add_optional_arg(arguments, "--fit-csv", LaunchConfiguration("fit_csv").perform(context))
    _add_optional_arg(arguments, "--rpm-fit-plot", LaunchConfiguration("rpm_fit_plot").perform(context))
    _add_optional_arg(arguments, "--rpm-fit-csv", LaunchConfiguration("rpm_fit_csv").perform(context))
    _add_optional_arg(arguments, "--mode", LaunchConfiguration("mode").perform(context))
    _add_optional_arg(arguments, "--amplitude", LaunchConfiguration("amplitude").perform(context))
    _add_optional_arg(arguments, "--ramp", LaunchConfiguration("ramp").perform(context))
    _add_optional_arg(arguments, "--rate", LaunchConfiguration("rate").perform(context))
    _add_optional_arg(arguments, "--fit-min-abs-command", LaunchConfiguration("fit_min_abs_command").perform(context))
    _add_optional_arg(arguments, "--rpm-fit-min-abs-rpm", LaunchConfiguration("rpm_fit_min_abs_rpm").perform(context))
    _add_optional_arg(arguments, "--force-topic", LaunchConfiguration("force_topic").perform(context))
    _add_optional_arg(arguments, "--force-log-period", LaunchConfiguration("force_log_period").perform(context))
    _add_optional_arg(arguments, "--rpm-topic", LaunchConfiguration("rpm_topic").perform(context))
    _add_optional_arg(arguments, "--rpm-stale-sec", LaunchConfiguration("rpm_stale_sec").perform(context))

    if _as_bool(LaunchConfiguration("print_plan").perform(context)):
        arguments.append("--print-plan")
    if _as_bool(LaunchConfiguration("no_fit").perform(context)):
        arguments.append("--no-fit")
    if _as_bool(LaunchConfiguration("no_stop_on_exit").perform(context)):
        arguments.append("--no-stop-on-exit")
    if _as_bool(LaunchConfiguration("overwrite").perform(context)):
        arguments.append("--overwrite")

    return [
        Node(
            package=PACKAGE_NAME,
            executable="thruster_curve_measure",
            name="thruster_curve_measure",
            output=LaunchConfiguration("output").perform(context),
            arguments=arguments,
        )
    ]


def generate_launch_description() -> LaunchDescription:
    default_config = PathJoinSubstitution(
        [
            FindPackageShare(PACKAGE_NAME),
            "config",
            "thruster_curve_measure_step.yaml",
        ]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=default_config,
                description="YAML defaults for thruster_curve_measure.",
            ),
            DeclareLaunchArgument(
                "index",
                default_value="4",
                description="Canonical thruster index [0..7]. Default 4 is H_LF.",
            ),
            DeclareLaunchArgument(
                "sensor_port",
                default_value="/dev/ttyUSB0",
                description="Force sensor Modbus serial port.",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="data/thruster_curve",
                description="Root output directory. Files are written under <output_dir>/<motor_name>.",
            ),
            DeclareLaunchArgument(
                "motor_name",
                default_value="M001",
                description="Motor identifier used as the output subdirectory name.",
            ),
            DeclareLaunchArgument(
                "overwrite",
                default_value="false",
                description="Allow writing into an existing non-empty output directory.",
            ),
            DeclareLaunchArgument(
                "csv",
                default_value="",
                description="Legacy override for raw CSV path. Empty uses <output_dir>/<motor_name>/raw.csv.",
            ),
            DeclareLaunchArgument(
                "summary_csv",
                default_value="",
                description="Legacy override for step steady-state summary CSV. Empty uses <output_dir>/<motor_name>/summary.csv.",
            ),
            DeclareLaunchArgument(
                "plot",
                default_value="",
                description="Legacy override for curve plot path. Empty uses <output_dir>/<motor_name>/curve.png.",
            ),
            DeclareLaunchArgument(
                "points_plot",
                default_value="",
                description="Legacy override for steady command-force point plot. Empty uses <output_dir>/<motor_name>/points.png.",
            ),
            DeclareLaunchArgument(
                "fit_plot",
                default_value="",
                description="Legacy override for signed quadratic fit plot. Empty uses <output_dir>/<motor_name>/fit.png.",
            ),
            DeclareLaunchArgument(
                "fit_csv",
                default_value="",
                description="Legacy override for signed quadratic fit parameter CSV. Empty uses <output_dir>/<motor_name>/fit.csv.",
            ),
            DeclareLaunchArgument(
                "rpm_fit_plot",
                default_value="",
                description="Legacy override for RPM/omega thrust model plot. Empty uses <output_dir>/<motor_name>/rpm_fit.png.",
            ),
            DeclareLaunchArgument(
                "rpm_fit_csv",
                default_value="",
                description="Legacy override for RPM/omega thrust model c1 CSV. Empty uses <output_dir>/<motor_name>/rpm_fit.csv.",
            ),
            DeclareLaunchArgument(
                "mode",
                default_value="",
                description="Measurement mode override, e.g. step or rpm-sweep. Empty uses config.",
            ),
            DeclareLaunchArgument(
                "amplitude",
                default_value="",
                description="Command amplitude override. Empty uses config.",
            ),
            DeclareLaunchArgument(
                "ramp",
                default_value="",
                description="Ramp duration override in seconds. In rpm-sweep this is one half-sweep. Empty uses config.",
            ),
            DeclareLaunchArgument(
                "rate",
                default_value="",
                description="Publish/read/log rate override in Hz. Empty uses config.",
            ),
            DeclareLaunchArgument(
                "fit_min_abs_command",
                default_value="0.0",
                description="Ignore commands with abs(command) <= this value when fitting.",
            ),
            DeclareLaunchArgument(
                "rpm_fit_min_abs_rpm",
                default_value="0.0",
                description="Ignore RPM values with abs(rpm) <= this value when fitting c1.",
            ),
            DeclareLaunchArgument(
                "no_fit",
                default_value="false",
                description="Disable steady point/fitting outputs.",
            ),
            DeclareLaunchArgument(
                "force_topic",
                default_value="/finsrov/thruster_curve/force_zeroed_n",
                description="Real-time zeroed force topic. Set empty to disable topic publishing.",
            ),
            DeclareLaunchArgument(
                "force_log_period",
                default_value="1.0",
                description="Terminal force log period in seconds. Set 0 to disable terminal force logs.",
            ),
            DeclareLaunchArgument(
                "rpm_topic",
                default_value="/finsrov/hardware/motor_rpm_raw",
                description="Canonical motor RPM Float32MultiArray topic. Set empty to disable RPM logging/fitting.",
            ),
            DeclareLaunchArgument(
                "rpm_stale_sec",
                default_value="0.5",
                description="Max RPM sample age before treating it as unavailable. Set 0 to never expire.",
            ),
            DeclareLaunchArgument(
                "print_plan",
                default_value="false",
                description="Print the resolved measurement plan and exit.",
            ),
            DeclareLaunchArgument(
                "no_stop_on_exit",
                default_value="false",
                description="Do not publish a final all-zero thruster command on exit.",
            ),
            DeclareLaunchArgument(
                "output",
                default_value="screen",
                description="ROS launch output mode.",
            ),
            OpaqueFunction(function=_launch_measure_node),
        ]
    )
