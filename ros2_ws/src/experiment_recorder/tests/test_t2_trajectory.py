import math
from pathlib import Path

import pytest
import yaml

from experiment_recorder.t2_trajectory import generate_trajectory, validate_reference
from experiment_recorder.t2_runner import (
    _controller,
    _resolve_top_camera_video_config,
    _select_trajectory_ids,
    _snapshot_controller_provenance,
    _top_camera_video_command,
    _validate_arm_gates,
)


WORKSPACE = {
    "x_m": [-2.0, 2.0],
    "y_m": [-1.0, -0.1],
    "z_m": [-2.0, 2.0],
    "clearance_margin_m": 0.1,
    "maximum_horizontal_speed_mps": 1.0,
    "maximum_horizontal_acceleration_mps2": 2.0,
}


def test_line_has_explicit_zero_endpoint_velocity_and_fixed_depth():
    points = generate_trajectory(
        {"generator": "line", "duration_sec": 10.0, "start_xz_m": [-0.5, 0.0], "end_xz_m": [0.5, 0.0]},
        fixed_depth_m=-0.5,
        sample_rate_hz=10.0,
    )
    assert points[0].vx_mps == pytest.approx(0.0)
    assert points[-1].vx_mps == pytest.approx(0.0)
    assert all(point.y_m == pytest.approx(-0.5) and point.vy_mps == 0.0 for point in points)
    assert points[-1].x_m == pytest.approx(0.5)


def test_circle_returns_to_start_and_declared_turns_scale_path_speed():
    one_turn = generate_trajectory(
        {"generator": "circle_xz", "duration_sec": 60.0, "center_xz_m": [0.0, 0.0], "radius_m": 0.3, "turns": 1, "direction": "counterclockwise"},
        fixed_depth_m=-0.5,
        sample_rate_hz=10.0,
    )
    five_turns = generate_trajectory(
        {"generator": "circle_xz", "duration_sec": 60.0, "center_xz_m": [0.0, 0.0], "radius_m": 0.3, "turns": 5, "direction": "counterclockwise"},
        fixed_depth_m=0.0,
        sample_rate_hz=10.0,
    )
    assert five_turns[0].x_m == pytest.approx(five_turns[-1].x_m)
    assert five_turns[0].z_m == pytest.approx(five_turns[-1].z_m)
    assert max(point.vx_mps**2 + point.vz_mps**2 for point in five_turns) == pytest.approx(
        25.0 * max(point.vx_mps**2 + point.vz_mps**2 for point in one_turn)
    )
    assert validate_reference(one_turn, WORKSPACE)["peak_speed_mps"] > 0.0


def test_vertex_slowed_ellipse_preserves_geometry_depth_and_circle_relative_speed_contract():
    points = generate_trajectory(
        {
            "generator": "ellipse_xz_vertex_slow",
            "duration_sec": 92.70,
            "center_xz_m": [0.0, 0.0],
            "semi_major_x_m": 1.0,
            "semi_minor_z_m": 0.4,
            "turns": 1,
            "initial_phase_deg": 0.0,
            "direction": "counterclockwise",
            "vertex_speed_scale": 0.5,
            "target_peak_speed_mps": 0.103083509,
            "target_peak_speed_tolerance_mps": 0.002,
        },
        fixed_depth_m=-0.5,
        sample_rate_hz=10.0,
    )
    speeds = [math.hypot(point.vx_mps, point.vz_mps) for point in points]
    assert (points[0].x_m, points[0].z_m) == pytest.approx((1.0, 0.0))
    assert (points[-1].x_m, points[-1].z_m) == pytest.approx((1.0, 0.0))
    assert all(point.y_m == pytest.approx(-0.5) and point.vy_mps == 0.0 for point in points)
    assert max(point.x_m for point in points) == pytest.approx(1.0)
    assert min(point.x_m for point in points) == pytest.approx(-1.0, abs=2e-4)
    assert max(point.z_m for point in points) == pytest.approx(0.4, abs=2e-4)
    assert min(point.z_m for point in points) == pytest.approx(-0.4, abs=2e-4)
    peak_speed = max(speeds)
    assert peak_speed == pytest.approx(0.103083509, abs=0.002)
    # Excluding the trial start/end, the reference slows at all four cardinal
    # vertices relative to the between-vertex cruise portions.
    vertex_speeds = [
        (point.vx_mps**2 + point.vz_mps**2) ** 0.5
        for point in points[1:-1]
        if abs(abs(point.x_m) - 1.0) < 0.004 or abs(abs(point.z_m) - 0.4) < 0.004
    ]
    assert vertex_speeds
    assert max(vertex_speeds) < 0.65 * peak_speed
    assert validate_reference(points, WORKSPACE)["peak_speed_mps"] == pytest.approx(peak_speed)


def test_large_figure_eight_reaches_declared_extrema_and_matches_ellipse_peak_speed():
    points = generate_trajectory(
        {
            "generator": "gerono_xz",
            "duration_sec": 117.085700,
            "center_xz_m": [0.0, 0.0],
            "amplitude_x_m": 1.0,
            "amplitude_z_m": 0.8,
            "initial_phase_deg": 0.0,
            "direction": "forward",
            "target_peak_speed_mps": 0.103083509,
            "target_peak_speed_tolerance_mps": 0.0005,
        },
        fixed_depth_m=-0.5,
        sample_rate_hz=10.0,
    )
    speeds = [math.hypot(point.vx_mps, point.vz_mps) for point in points]
    assert (points[0].x_m, points[0].z_m) == pytest.approx((0.0, 0.0))
    assert (points[-1].x_m, points[-1].z_m) == pytest.approx((0.0, 0.0), abs=1e-9)
    assert all(point.y_m == pytest.approx(-0.5) and point.vy_mps == 0.0 for point in points)
    assert max(point.x_m for point in points) == pytest.approx(1.0, abs=2e-4)
    assert min(point.x_m for point in points) == pytest.approx(-1.0, abs=2e-4)
    assert max(point.z_m for point in points) == pytest.approx(0.4, abs=2e-4)
    assert min(point.z_m for point in points) == pytest.approx(-0.4, abs=2e-4)
    assert max(speeds) == pytest.approx(0.103083509, abs=0.0005)
    assert validate_reference(points, WORKSPACE)["peak_speed_mps"] == pytest.approx(max(speeds))


def test_polyline_hits_each_declared_waypoint_with_zero_corner_velocity():
    points = generate_trajectory(
        {
            "generator": "polyline_xz",
            "duration_sec": 9.0,
            "waypoints_xz_m": [[-1.0, -0.4], [1.0, 0.4], [1.0, -0.4], [-1.0, 0.4], [-1.0, -0.4]],
            "segment_durations_sec": [3.0, 1.0, 3.0, 2.0],
        },
        fixed_depth_m=-0.5,
        sample_rate_hz=10.0,
    )
    samples = {point.time_sec: point for point in points}
    for time_sec, expected_xz in ((0.0, (-1.0, -0.4)), (3.0, (1.0, 0.4)), (4.0, (1.0, -0.4)), (7.0, (-1.0, 0.4)), (9.0, (-1.0, -0.4))):
        point = samples[time_sec]
        assert (point.x_m, point.z_m) == pytest.approx(expected_xz)
        assert (point.vx_mps, point.vz_mps) == pytest.approx((0.0, 0.0))


def test_workspace_validation_fails_closed_on_out_of_bounds_path():
    points = generate_trajectory(
        {"generator": "line", "duration_sec": 5.0, "start_xz_m": [0.0, 0.0], "end_xz_m": [3.0, 0.0]},
        fixed_depth_m=-0.5,
        sample_rate_hz=10.0,
    )
    with pytest.raises(ValueError, match="outside"):
        validate_reference(points, WORKSPACE)


def test_reference_has_no_yaw_setpoint():
    points = generate_trajectory(
        {"generator": "line", "duration_sec": 10.0, "start_xz_m": [-0.5, 0.0], "end_xz_m": [0.5, 0.0]},
        fixed_depth_m=0.0,
        sample_rate_hz=10.0,
    )
    assert not hasattr(points[0], "tangent_yaw_deg")


def test_cli_trajectory_selection_requires_exactly_one_yaml_path():
    catalog = {"straight": ([], {}), "circle": ([], {}), "lemniscate": ([], {})}
    selected, source = _select_trajectory_ids(["circle"], {"trajectories": ["lemniscate"]}, catalog)
    assert selected == ["circle"]
    assert source == "cli_exactly_one"


def test_cli_trajectory_selection_rejects_unknown_or_multiple_ids():
    catalog = {"straight": ([], {}), "circle": ([], {})}
    with pytest.raises(SystemExit, match="unknown"):
        _select_trajectory_ids(["missing"], {}, catalog)
    with pytest.raises(SystemExit, match="exactly one"):
        _select_trajectory_ids(["straight", "straight"], {}, catalog)


def test_t2_arm_gate_is_explicit_but_survey_metadata_is_not_a_runtime_blocker():
    config = {
        "runner": {"require_explicit_arm_flag": True},
        "survey_signoff": {"required": True, "approved": False},
    }
    _validate_arm_gates(config, arm=True, dry_run=False)
    with pytest.raises(SystemExit, match="requires --arm"):
        _validate_arm_gates(config, arm=False, dry_run=False)


def test_t2_translation_only_pid_profile_is_checkpoint_free_and_explicitly_disables_yaw():
    repo_root = Path(__file__).resolve().parents[4]
    experiment_config = repo_root / "ros2_ws/src/experiment_recorder/config/t2_hardware_experiment.yaml"
    config = yaml.safe_load(experiment_config.read_text(encoding="utf-8"))

    selected, profile, checkpoint, checkpoint_manifest = _controller(
        config,
        "PID_POSITION",
        repo_root,
        dry_run=True,
    )

    assert selected["expected_backend_name"] == "traditional_pid_position"
    assert profile.name == "traditional_pid_trajectory_tracking.yaml"
    assert checkpoint is None
    assert checkpoint_manifest is None
    params = yaml.safe_load(profile.read_text(encoding="utf-8"))["motion_controller"]["ros__parameters"]
    assert params["use_target_orientation"] is False
    assert params["traditional_enable_yaw_control"] is False
    assert params["stop_on_goal_reached"] is False


def test_t2_ppo_checkpoint_and_training_metadata_follow_controller_profile():
    repo_root = Path(__file__).resolve().parents[4]
    experiment_config = repo_root / "ros2_ws/src/experiment_recorder/config/t2_hardware_experiment.yaml"
    config = yaml.safe_load(experiment_config.read_text(encoding="utf-8"))

    selected, profile, checkpoint, checkpoint_manifest = _controller(
        config,
        "PPO_WRENCH6",
        repo_root,
        dry_run=True,
    )

    params = yaml.safe_load(profile.read_text(encoding="utf-8"))["motion_controller"]["ros__parameters"]
    assert selected["config_file"].endswith("ppo_trajectory_tracking_wrench6.yaml")
    assert "checkpoint_path" not in selected
    assert "checkpoint_manifest_path" not in selected
    assert checkpoint == Path(params["checkpoint_path"])
    assert checkpoint_manifest == checkpoint.parent.parent / "metadata.json"
    assert checkpoint_manifest.is_file()


def test_t2_provenance_snapshots_preserve_controller_profile_and_training_metadata(tmp_path):
    profile = tmp_path / "controller.yaml"
    metadata = tmp_path / "metadata.json"
    profile.write_text("motion_controller: {}\n", encoding="utf-8")
    metadata.write_text('{"run_name": "test"}\n', encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir()

    snapshots = _snapshot_controller_provenance(
        run_dir,
        controller_config=profile,
        checkpoint_manifest=metadata,
    )

    for record in snapshots.values():
        snapshot = run_dir / record["snapshot"]
        assert snapshot.is_file()
        assert record["source_sha256"] == record["snapshot_sha256"]


def test_t2_top_camera_video_is_opt_in_and_can_be_overridden_from_cli():
    class Args:
        record_video = None

    config = {
        "recording": {
            "top_camera_video": {
                "enabled": False,
                "image_topic": "/finsrov/camera/raw/compressed",
                "filename": "top_camera_raw.mp4",
                "fps": 10.0,
                "codec": "mp4v",
            },
        },
    }
    disabled = _resolve_top_camera_video_config(config, Args())
    assert disabled["enabled"] is False

    Args.record_video = True
    enabled = _resolve_top_camera_video_config(config, Args())
    assert enabled["enabled"] is True
    command = _top_camera_video_command(enabled, Path("/tmp/trial/video/top_camera_raw.mp4"))
    assert command[:4] == ["ros2", "run", "experiment_recorder", "record_top_camera_video"]
    assert "--topic" in command and "/finsrov/camera/raw/compressed" in command
