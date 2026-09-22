from pathlib import Path

import pytest

from experiment_recorder.sim_runner import (
    _require_matching_editor_status,
    _simulation_clock_config,
    _t2_required_controller_topics,
    _unity_sim_thruster_command_mode,
    _validate_simulation_time_controller_profile,
    _validate_sim_truth_status_contract,
)


def test_t2_pid_readiness_does_not_require_the_ppo_observation_topic():
    config = {
        "runner": {
            "auto_launch": {
                "required_controller_topics": [
                    "/sim/finsrov/controller/state/status",
                    "/sim/motion_controller/debug/observation",
                ]
            }
        },
        "controllers": {
            "PID_POSITION": {
                "required_controller_topics": [
                    "/sim/finsrov/controller/state/status",
                    "/sim/motion_controller/debug/trajectory_reference",
                    "/sim/motion_controller/debug/thruster_command",
                ]
            }
        },
    }

    assert _t2_required_controller_topics(config, "PID_POSITION") == [
        "/sim/finsrov/controller/state/status",
        "/sim/motion_controller/debug/trajectory_reference",
        "/sim/motion_controller/debug/thruster_command",
    ]


def test_t2_ppo_readiness_keeps_the_common_observation_gate():
    config = {
        "runner": {"auto_launch": {"required_controller_topics": ["/sim/motion_controller/debug/observation"]}},
        "controllers": {"PPO_WRENCH6": {}},
    }

    assert _t2_required_controller_topics(config, "PPO_WRENCH6") == ["/sim/motion_controller/debug/observation"]


def test_sim_truth_status_contract_accepts_one_shared_topic():
    _validate_sim_truth_status_contract(
        {
            "state_input_mode": "sim_truth",
            "status_topic": "/sim/finsrov/controller/state/status",
            "state_status_topic": "/sim/finsrov/controller/state/status",
        },
        Path("profile.yaml"),
    )


def test_sim_truth_status_contract_rejects_unprefixed_helper_topic():
    with pytest.raises(SystemExit, match="status_topic.*state_status_topic"):
        _validate_sim_truth_status_contract(
            {
                "state_input_mode": "sim_truth",
                "status_topic": "/finsrov/controller/state/status",
                "state_status_topic": "/sim/finsrov/controller/state/status",
            },
            Path("profile.yaml"),
        )


def test_non_sim_truth_profile_does_not_require_sim_status_topics():
    _validate_sim_truth_status_contract({"state_input_mode": "raw_fusion"}, Path("profile.yaml"))


def test_simulation_controller_selects_its_registered_unity_action_abi():
    assert _unity_sim_thruster_command_mode({"unity_sim_thruster_command_mode": "normalized_direct"}) == "normalized_direct"
    assert _unity_sim_thruster_command_mode({}) == "force_n"
    with pytest.raises(SystemExit, match="invalid unity_sim_thruster_command_mode"):
        _unity_sim_thruster_command_mode({"unity_sim_thruster_command_mode": "hardware_rpm"})


def test_simulation_clock_contract_uses_ros_clock_without_touching_training_defaults():
    clock = _simulation_clock_config(
        {
            "runner": {
                "simulation_clock": {
                    "use_sim_time": True,
                    "clock_topic": "/clock",
                    "time_scale": 10.0,
                    "clock_start_timeout_wall_sec": 30.0,
                    "clock_stall_timeout_wall_sec": 15.0,
                }
            }
        }
    )
    assert clock["time_scale"] == 10.0
    assert clock["protocol_time_basis"] == "ros_simulation_time"
    _validate_simulation_time_controller_profile({"use_sim_time": True}, Path("profile.yaml"), clock)


def test_simulation_clock_rejects_wall_clock_controller_profile():
    clock = _simulation_clock_config(
        {"runner": {"simulation_clock": {"use_sim_time": True, "time_scale": 10.0}}}
    )
    with pytest.raises(SystemExit, match="use_sim_time: true"):
        _validate_simulation_time_controller_profile({}, Path("profile.yaml"), clock)


def test_ros2_control_lockstep_requires_a_matching_controller_ack_contract():
    clock = _simulation_clock_config(
        {
            "runner": {
                "simulation_clock": {
                    "mode": "ros2_control_lockstep",
                    "use_sim_time": True,
                    "time_scale": 10.0,
                    "lockstep_ack_topic": "/sim/motion_controller/debug/control_tick_complete",
                    "lockstep_ack_timeout_wall_sec": 15.0,
                }
            }
        }
    )

    assert clock["mode"] == "ros2_control_lockstep"
    _validate_simulation_time_controller_profile(
        {
            "use_sim_time": True,
            "lockstep_enabled": True,
            "lockstep_clock_topic": "/clock",
            "lockstep_ack_topic": "/sim/motion_controller/debug/control_tick_complete",
        },
        Path("profile.yaml"),
        clock,
    )
    with pytest.raises(SystemExit, match="lockstep_enabled: true"):
        _validate_simulation_time_controller_profile({"use_sim_time": True}, Path("profile.yaml"), clock)


def test_free_running_clock_rejects_a_lockstep_only_controller_profile():
    clock = _simulation_clock_config({"runner": {"simulation_clock": {"use_sim_time": True}}})

    with pytest.raises(SystemExit, match="enables lockstep"):
        _validate_simulation_time_controller_profile(
            {"use_sim_time": True, "lockstep_enabled": True},
            Path("profile.yaml"),
            clock,
        )


def test_open_editor_status_must_match_the_requested_project_and_scene(tmp_path: Path):
    project = tmp_path / "marus-example"
    scene = project / "Assets" / "Scenes" / "ControlForPosition_Fossen.unity"
    scene.parent.mkdir(parents=True)
    scene.touch()
    payload = {
        "projectPath": str(project),
        "activeScenePath": str(scene),
        "unityVersion": "6000.3.20f1",
    }

    _require_matching_editor_status(
        payload,
        project=project,
        scene=scene,
        expected_version="6000.3.20f1",
    )


def test_open_editor_status_accepts_unity_project_relative_scene_path(tmp_path: Path):
    project = tmp_path / "marus-example"
    scene = project / "Assets" / "Scenes" / "ControlForPosition_Fossen.unity"
    scene.parent.mkdir(parents=True)
    scene.touch()
    payload = {
        "projectPath": str(project),
        "activeScenePath": "Assets/Scenes/ControlForPosition_Fossen.unity",
        "unityVersion": "6000.3.20f1",
    }

    _require_matching_editor_status(
        payload,
        project=project,
        scene=scene,
        expected_version="6000.3.20f1",
    )


def test_open_editor_with_wrong_scene_is_not_reused(tmp_path: Path):
    project = tmp_path / "marus-example"
    requested = project / "Assets" / "Scenes" / "ControlForPosition_Fossen.unity"
    active = project / "Assets" / "Scenes" / "Other.unity"
    requested.parent.mkdir(parents=True)
    requested.touch()
    active.touch()
    payload = {
        "projectPath": str(project),
        "activeScenePath": str(active),
        "unityVersion": "6000.3.20f1",
    }

    with pytest.raises(RuntimeError, match="different active scene"):
        _require_matching_editor_status(
            payload,
            project=project,
            scene=requested,
            expected_version="6000.3.20f1",
        )
