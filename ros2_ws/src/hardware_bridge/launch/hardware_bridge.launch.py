from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError(f"expected boolean value, got {value!r}")


def _optional_override(context, name: str, value_type: type):
    value = LaunchConfiguration(name).perform(context).strip()
    if value == "":
        return None
    if value_type is bool:
        return name, _parse_bool(value)
    if value_type is int:
        return name, int(value, 0)
    return name, value_type(value)


def _launch_nodes(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context)
    parameter_sources = [params_file]

    overrides = {}
    for item in (
        _optional_override(context, "transport", str),
        _optional_override(context, "enabled", bool),
        _optional_override(context, "debug_mask", int),
        _optional_override(context, "debug_echo_decimation", int),
        _optional_override(context, "command_mode", str),
        _optional_override(context, "input_thruster_topic", str),
    ):
        if item is not None:
            key, value = item
            overrides[key] = value

    if overrides:
        parameter_sources.append(overrides)

    return [
        Node(
            package="hardware_bridge",
            executable="hardware_bridge",
            name="hardware_bridge",
            output="screen",
            parameters=parameter_sources,
        )
    ]


def generate_launch_description():
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("hardware_bridge"), "config", "finsrov_hardware_bridge_v4_pro1.yaml"]
                ),
                description="YAML parameter file for hardware_bridge.",
            ),
            DeclareLaunchArgument(
                "enabled",
                default_value="",
                description="Optional startup override for the hardware_bridge enabled parameter.",
            ),
            DeclareLaunchArgument(
                "transport",
                default_value="",
                description="Optional startup override for the hardware_bridge transport parameter.",
            ),
            DeclareLaunchArgument(
                "debug_mask",
                default_value="",
                description="Optional startup override for the hardware_bridge debug_mask parameter.",
            ),
            DeclareLaunchArgument(
                "debug_echo_decimation",
                default_value="",
                description="Optional startup override for the hardware_bridge debug_echo_decimation parameter.",
            ),
            DeclareLaunchArgument(
                "command_mode",
                default_value="",
                description="Optional startup override for the hardware_bridge command_mode parameter.",
            ),
            DeclareLaunchArgument(
                "input_thruster_topic",
                default_value="",
                description="Optional startup override for the hardware_bridge input_thruster_topic parameter.",
            ),
            OpaqueFunction(function=_launch_nodes),
        ]
    )
