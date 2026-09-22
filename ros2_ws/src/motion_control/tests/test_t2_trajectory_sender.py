import json
from pathlib import Path

import pytest

from motion_control.trajectory_sender import _build_message, _load_t2_trajectory, t2_main


REPO_ROOT = Path(__file__).resolve().parents[4]
T2_CONFIG = REPO_ROOT / "ros2_ws/src/experiment_recorder/config/t2_hardware_experiment.yaml"


def test_t2_sender_uses_the_registered_circle_geometry_and_transport_contract():
    rows, summary = _load_t2_trajectory(T2_CONFIG, "circle")
    message = _build_message(rows, label="circle")

    assert summary["trajectory"] == "circle"
    assert summary["frame_id"] == "controller_world"
    assert summary["start_controller_world"] == pytest.approx([0.5, -0.5, 0.0])
    assert summary["duration_sec"] == pytest.approx(137.142857)
    assert summary["orientation_control"] == "disabled_identity_quaternion_transport"
    assert len(message.points) == summary["point_count"]
    assert message.header.frame_id == "controller_world"
    assert message.points[0].transforms[0].rotation.w == pytest.approx(1.0)
    assert message.points[0].velocities[0].linear.x == pytest.approx(0.0)


def test_t2_sender_dry_run_resolves_a_named_path_without_ros_initialization(capsys):
    t2_main(["--config", str(T2_CONFIG), "--trajectory", "straight", "--dry-run"])

    summary = json.loads(capsys.readouterr().out)
    assert summary["trajectory"] == "straight"
    assert summary["start_controller_world"] == pytest.approx([-1.0, -0.5, -0.4])


def test_t2_sender_rejects_a_disabled_or_unknown_path():
    with pytest.raises(ValueError, match="disabled or unknown"):
        _load_t2_trajectory(T2_CONFIG, "not_a_t2_path")
