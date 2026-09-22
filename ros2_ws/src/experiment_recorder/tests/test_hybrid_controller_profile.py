from pathlib import Path

import pytest
import yaml

from experiment_recorder.runner import (
    _controller_checkpoint,
    _launch_commands,
    _validate_controller_profile_runtime,
)


def test_hybrid_profile_checkpoint_is_discovered_from_its_node_key():
    repo_root = Path(__file__).resolve().parents[4]
    profile = (
        repo_root
        / "ros2_ws/src/motion_control/config/FinsROV/hybrid_horizontal_ppo_depth_pid.yaml"
    )
    checkpoint = _controller_checkpoint(profile, repo_root)
    assert checkpoint is not None
    assert checkpoint.name == "best_model.zip"


def test_t1_runner_launches_the_registered_hybrid_executable():
    repo_root = Path(__file__).resolve().parents[4]
    protocol_path = repo_root / "ros2_ws/src/experiment_recorder/config/t1_hardware_experiment.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    controller_command = _launch_commands(protocol, "PPO_HYBRID", False, repo_root)[3][1]

    assert "controller_executable:=hybrid_horizontal_ppo_depth_pid_controller" in controller_command
    assert "motion_controller_node_name:=hybrid_horizontal_ppo_depth_pid_controller" in controller_command


def test_hybrid_runtime_rejects_regular_motion_controller_yaml():
    repo_root = Path(__file__).resolve().parents[4]
    profile = repo_root / "ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator_ablation.yaml"
    with pytest.raises(SystemExit, match="hybrid_horizontal_ppo_depth_pid.yaml"):
        _validate_controller_profile_runtime(
            {"controller_node_name": "hybrid_horizontal_ppo_depth_pid_controller"},
            profile,
        )


def test_standalone_ppo_uses_the_standard_motion_controller_runtime():
    repo_root = Path(__file__).resolve().parents[4]
    protocol_path = repo_root / "ros2_ws/src/experiment_recorder/config/t1_hardware_experiment.yaml"
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))

    controller_command = _launch_commands(protocol, "PPO", False, repo_root)[3][1]

    assert "controller_executable:=motion_controller" in controller_command
    assert "motion_controller_node_name:=motion_controller" in controller_command
