from pathlib import Path

import pytest
import yaml

from motion_control.backend_loader import BACKEND_SPECS
from motion_control.controller_node import _normalize_thruster_output_mode


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
PROFILE_FILES = sorted((PACKAGE_ROOT / "config" / "FinsROV").glob("*.yaml"))


def _controller_parameters(path: Path) -> dict:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    namespace = data.get("motion_controller") or data.get("/**")
    assert isinstance(namespace, dict), f"{path} has no controller parameter namespace"
    parameters = namespace.get("ros__parameters")
    assert isinstance(parameters, dict), f"{path} has no ros__parameters"
    return parameters


@pytest.mark.parametrize("profile_path", PROFILE_FILES, ids=lambda path: path.name)
def test_controller_profile_owns_backend_and_checkpoint(profile_path: Path) -> None:
    parameters = _controller_parameters(profile_path)

    backend_name = parameters.get("backend_name")
    assert isinstance(backend_name, str) and backend_name.strip(), profile_path
    assert backend_name in BACKEND_SPECS, profile_path
    output_mode = parameters.get("thruster_output_mode", "force_n")
    assert isinstance(output_mode, str), profile_path
    _normalize_thruster_output_mode(output_mode)

    checkpoint_path = parameters.get("checkpoint_path")
    assert isinstance(checkpoint_path, str), profile_path
    if BACKEND_SPECS[backend_name].requires_checkpoint:
        checkpoint_status = parameters.get("checkpoint_status", "ready")
        assert isinstance(checkpoint_status, str), profile_path
        if checkpoint_status == "retraining_required":
            assert not checkpoint_path.strip(), profile_path
        else:
            assert checkpoint_status == "ready", profile_path
            assert checkpoint_path.strip(), profile_path


def test_isaaclab_finsrov_wrench_profile_matches_the_sim7n_task_contract() -> None:
    parameters = _controller_parameters(
        PACKAGE_ROOT / "config/FinsROV/ppo_wrench_isaaclab_finsrov_hold_for_position.yaml"
    )

    assert parameters["backend_name"] == "isaaclab_finsrov_hold_for_position_wrench"
    assert parameters["action_schema"] == "finsrov_physical_wrench_sim7n_v2"
    assert parameters["thruster_output_mode"] == "wrench6d_force_n"
    assert parameters["thruster_force_limits_n"]["positive"] == [7.0] * 8
    assert parameters["thruster_force_limits_n"]["negative"] == [7.0] * 8
    assert parameters["wrench6d"] == {
        "allocation_mode": "physical_wrench_allocator",
        "policy_axis_order": "legacy_surge_sway_heave",
        "wrench_limits": [19.528527, 18.415027, 22.501886, 3.863995, 3.079985, 7.080833],
        "wrench_scale": [1.0, 1.0, 1.0, 0.0, 0.0, 1.0],
    }


def test_isaaclab_finsrov_direct_ppo_profile_preserves_the_trained_action_to_force_contract() -> None:
    parameters = _controller_parameters(
        PACKAGE_ROOT / "config/FinsROV/ppo_isaaclab_finsrov_hold_for_position_direct_ppo.yaml"
    )

    assert parameters["backend_name"] == "isaaclab_finsrov_hold_for_position"
    assert parameters["thruster_output_mode"] == "isaaclab_finsrov_calibrated_thruster8_force_n"
    assert "thruster8_wrench_projection" not in parameters
    assert "wrench6d" not in parameters
    assert parameters["thruster_output_scale"] == [1.0] * 8
    assert parameters["thruster_force_limits_n"] == {
        "positive": [8.474877, 8.797188, 8.797188, 8.474877, 8.797188, 8.474877, 8.797188, 8.474877],
        "negative": [7.974983, 8.272793, 8.272793, 7.974983, 8.272793, 7.974983, 8.272793, 7.974983],
    }


def test_isaaclab_finsrov_reprojected_profile_is_explicitly_a_wrench_path() -> None:
    parameters = _controller_parameters(
        PACKAGE_ROOT
        / "config/FinsROV/ppo_isaaclab_finsrov_hold_for_position_reprojected_physical_wrench.yaml"
    )

    assert parameters["backend_name"] == "isaaclab_finsrov_hold_for_position"
    assert parameters["thruster_output_mode"] == "thruster8_reprojected_physical_wrench_force_n"
    assert parameters["thruster8_wrench_projection"]["input_thruster_scale"] == [1.0] * 8
    assert parameters["wrench6d"]["allocation_mode"] == "physical_wrench_allocator"
    assert parameters["wrench6d"]["wrench_scale"] == [1.0, 1.0, 1.0, 0.0, 0.0, 0.1]


def test_t2_traditional_pid_profile_is_translation_only_and_never_interprets_transport_quaternion_as_yaw():
    parameters = _controller_parameters(
        PACKAGE_ROOT / "config/FinsROV/traditional_pid_trajectory_tracking.yaml"
    )

    assert parameters["backend_name"] == "traditional_pid_position"
    assert parameters["checkpoint_path"] == ""
    assert parameters["control_rate_hz"] == 60.0
    assert parameters["use_target_orientation"] is False
    assert parameters["require_target_orientation"] is False
    assert parameters["traditional_enable_yaw_control"] is False
    assert parameters["stop_on_goal_reached"] is False
    assert parameters["command_trajectory_topic"] == "/motion_controller/command/trajectory"
    assert parameters["debug_trajectory_reference_topic"] == "/motion_controller/debug/trajectory_reference"
    assert parameters["thruster_output_mode"] == "force_n"


@pytest.mark.parametrize(
    "profile_name",
    (
        "traditional_pid_trajectory_tracking_sim.yaml",
        "ppo_trajectory_tracking_wrench6_sim.yaml",
        "ppo_trajectory_tracking_thruster8_sim.yaml",
    ),
)
def test_t2_simulation_profiles_declare_one_auditable_ros2_lockstep_contract(profile_name: str) -> None:
    parameters = _controller_parameters(PACKAGE_ROOT / "config" / "FinsROV" / profile_name)

    assert parameters["use_sim_time"] is True
    assert parameters["lockstep_enabled"] is True
    assert parameters["lockstep_clock_topic"] == "/clock"
    assert parameters["lockstep_ack_topic"] == "/sim/motion_controller/debug/control_tick_complete"


@pytest.mark.parametrize(
    "profile_name",
    (
        "traditional_pid_position_yaw_sim.yaml",
        "ppo_wrench_for_pose_physical_wrench_allocator_sim_evaluation.yaml",
        "ppo_control_for_pose_thruster8_raw_action_sim.yaml",
    ),
)
def test_t1_simulation_profiles_declare_lockstep_and_stamped_command_audits(profile_name: str) -> None:
    parameters = _controller_parameters(PACKAGE_ROOT / "config" / "FinsROV" / profile_name)

    assert parameters["use_sim_time"] is True
    assert parameters["lockstep_enabled"] is True
    assert parameters["lockstep_clock_topic"] == "/clock"
    assert parameters["lockstep_ack_topic"] == "/sim/motion_controller/debug/control_tick_complete"
    assert parameters["debug_thruster_command_stamped_topic"] == (
        "/sim/motion_controller/debug/thruster_command_stamped"
    )
