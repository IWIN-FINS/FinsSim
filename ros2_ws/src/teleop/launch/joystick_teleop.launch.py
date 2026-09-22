from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _optional_override(context, name: str, value_type: type):
    value = LaunchConfiguration(name).perform(context).strip()
    if value == "":
        return None
    if value_type is bool:
        return name, value.lower() in ("1", "true", "yes", "on")
    if value_type is int:
        return name, int(value, 0)
    if value_type is float:
        return name, float(value)
    return name, value_type(value)


def _launch_nodes(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context)
    parameters = [params_file]
    overrides = {}
    for item in (
        _optional_override(context, "mode", str),
        _optional_override(context, "joystick_index", int),
        _optional_override(context, "rate_hz", float),
        _optional_override(context, "thruster_topic", str),
    ):
        if item is not None:
            key, value = item
            overrides[key] = value
    if overrides:
        parameters.append(overrides)

    return [
        Node(
            package="teleop",
            executable="joystick_teleop",
            name="finsrov_joystick_teleop",
            output="screen",
            parameters=parameters,
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("teleop"), "config", "finsrov_gamepad_v4_pro1.yaml"]
                ),
                description="YAML parameter file for joystick teleop.",
            ),
            DeclareLaunchArgument("mode", default_value="", description="Optional override: wrench or direct."),
            DeclareLaunchArgument("joystick_index", default_value="", description="Optional joystick index override."),
            DeclareLaunchArgument("rate_hz", default_value="", description="Optional publish rate override."),
            DeclareLaunchArgument(
                "thruster_topic",
                default_value="",
                description="Optional override for /finsrov/thrusters_out.",
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
