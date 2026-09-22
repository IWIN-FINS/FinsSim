from pathlib import Path

import yaml
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import OpaqueFunction
from launch.actions import SetEnvironmentVariable
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def _read_controller_param(params_file: str, name: str, default=None):
    if not params_file:
        return default
    path = Path(params_file).expanduser()
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    for node_name in ("motion_controller", "/**"):
        value = data.get(node_name, {}).get("ros__parameters", {}).get(name)
        if value is not None:
            return value
    return default


def _launch_nodes(context, *args, **kwargs):
    params_file = LaunchConfiguration("params_file").perform(context)
    vehicle_name = LaunchConfiguration("vehicle_name").perform(context).strip()
    state_input_mode = LaunchConfiguration("state_input_mode").perform(context).strip()
    if state_input_mode == "auto":
        state_input_mode = str(_read_controller_param(params_file, "state_input_mode", "raw_fusion")).strip()
    if state_input_mode not in {"raw_fusion", "sim_truth"}:
        raise RuntimeError(
            "state_input_mode must be one of: auto, raw_fusion, sim_truth; "
            f"got {state_input_mode!r} from launch argument or params_file"
        )
    pool_world_params_file = LaunchConfiguration("pool_world_params_file").perform(context)
    checkpoint_path = LaunchConfiguration("checkpoint_path").perform(context).strip()
    controller_state_adapter_node_name = LaunchConfiguration("controller_state_adapter_node_name").perform(context).strip()
    motion_controller_node_name = LaunchConfiguration("motion_controller_node_name").perform(context).strip()
    sim_truth_state_status_node_name = LaunchConfiguration("sim_truth_state_status_node_name").perform(context).strip()

    runtime_overrides = {}
    if vehicle_name:
        runtime_overrides["vehicle_name"] = vehicle_name
    if checkpoint_path:
        # The experiment runner can freeze a checkpoint in its own protocol
        # YAML without editing the controller profile.  An empty value keeps
        # the profile's checkpoint_path unchanged.
        runtime_overrides["checkpoint_path"] = checkpoint_path
    # YAML wildcard parameters are normally resolved by launch_ros, but a
    # missed ``use_sim_time`` override silently mixes Unity /clock state with
    # wall-clock controller timers. Resolve this one timing-critical setting
    # explicitly from the selected profile and append it after params_file.
    configured_use_sim_time = _read_controller_param(params_file, "use_sim_time", None)
    if configured_use_sim_time is not None:
        runtime_overrides["use_sim_time"] = bool(configured_use_sim_time)

    nodes = []
    if state_input_mode == "raw_fusion":
        nodes.append(
            Node(
                package="motion_control",
                executable="controller_state_adapter",
                name=controller_state_adapter_node_name or "controller_state_adapter",
                output="screen",
                parameters=[
                    # Keep the state adapter's axis permutation synchronized
                    # with the controller profile.  Without this file the
                    # adapter always used its built-in default even when a
                    # trajectory profile explicitly selected another basis.
                    params_file,
                    pool_world_params_file,
                    {
                        "vehicle_name": vehicle_name or "FinsROV",
                    },
                ],
            )
        )

    nodes.append(
        Node(
            package="motion_control",
            executable="motion_controller",
            name=motion_controller_node_name or "motion_controller",
            output="screen",
            parameters=[
                params_file,
                pool_world_params_file,
                runtime_overrides,
            ],
        )
    )

    if state_input_mode == "sim_truth":
        nodes.append(
            Node(
                package="motion_control",
                executable="sim_truth_state_status",
                name=sim_truth_state_status_node_name or "sim_truth_state_status",
                output="screen",
                parameters=[
                    params_file,
                    {
                        "vehicle_name": vehicle_name or "FinsROV",
                        **({"use_sim_time": bool(configured_use_sim_time)} if configured_use_sim_time is not None else {}),
                    },
                ],
            )
        )

    return nodes


def generate_launch_description() -> LaunchDescription:
    default_params_file = PathJoinSubstitution(
        [FindPackageShare("motion_control"), "config", "controller.yaml"]
    )

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value=default_params_file,
                description="YAML parameter file for the motion controller.",
            ),
            DeclareLaunchArgument(
                "vehicle_name",
                default_value="FinsROV",
                description="Vehicle name used to derive default topic names.",
            ),
            DeclareLaunchArgument(
                "state_input_mode",
                default_value="auto",
                choices=["auto", "raw_fusion", "sim_truth"],
                description=(
                    "Controller-state source. auto reads state_input_mode from params_file and falls "
                    "back to raw_fusion. raw_fusion starts pool_world->controller adapter; "
                    "sim_truth expects Unity to publish /finsrov/controller/* directly."
                ),
            ),
            DeclareLaunchArgument(
                "pool_world_params_file",
                default_value=PathJoinSubstitution(
                    [FindPackageShare("state_estimation"), "config", "pool_world.yaml"]
                ),
                description="Shared pool/world geometry parameters.",
            ),
            DeclareLaunchArgument(
                "checkpoint_path",
                default_value="",
                description="Optional controller checkpoint override from an experiment protocol.",
            ),
            DeclareLaunchArgument(
                "cuda_visible_devices",
                default_value="",
                description=(
                    "Optional CUDA_VISIBLE_DEVICES override for controller inference. "
                    "Use 1 on the current lab workstation to select the RTX 3090."
                ),
            ),
            DeclareLaunchArgument(
                "controller_state_adapter_node_name",
                default_value="controller_state_adapter",
                description="ROS node name for the raw-fusion controller-state adapter.",
            ),
            DeclareLaunchArgument(
                "motion_controller_node_name",
                default_value="motion_controller",
                description="ROS node name for the motion controller.",
            ),
            DeclareLaunchArgument(
                "sim_truth_state_status_node_name",
                default_value="sim_truth_state_status",
                description="ROS node name for the sim-truth state status helper.",
            ),
            OpaqueFunction(function=_set_cuda_visible_devices),
            OpaqueFunction(function=_launch_nodes),
        ]
    )


def _set_cuda_visible_devices(context, *args, **kwargs):
    value = LaunchConfiguration("cuda_visible_devices").perform(context).strip()
    if not value:
        return []
    return [SetEnvironmentVariable("CUDA_VISIBLE_DEVICES", value)]
