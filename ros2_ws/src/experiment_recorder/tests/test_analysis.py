import json
import math
from pathlib import Path

from experiment_recorder.analysis import (
    _angle_error,
    _first_t1_protocol_success_time,
    _pose_row,
    _resample_pose_segment,
    _select_control_pose_topic,
    _t1_lockstep_alignment_audit,
    _trajectory_xy_poses,
    _trial_windows,
    _window_settings,
    analyze_session,
)


def test_yaw_error_uses_shortest_circular_distance():
    """The analysis must not turn the +/-180 degree display seam into 360 degrees."""

    measured = math.radians(-179.0)
    target = math.radians(179.0)
    assert math.isclose(_angle_error(measured, target) or 0.0, math.radians(2.0), abs_tol=1e-8)


def test_yaw_error_is_antisymmetric_across_display_seam():
    positive = _angle_error(math.radians(-179.0), math.radians(179.0))
    negative = _angle_error(math.radians(179.0), math.radians(-179.0))
    assert positive is not None
    assert negative is not None
    assert math.isclose(positive, -negative, abs_tol=1e-8)


def test_controller_world_pose_uses_unity_y_axis_yaw():
    class _Stamp:
        sec = 1
        nanosec = 0

    class _Header:
        frame_id = "controller_world"
        stamp = _Stamp()

    class _Position:
        x = 0.0
        y = 0.0
        z = 0.0

    class _Orientation:
        x = 0.0
        y = math.sin(math.pi / 4.0)
        z = 0.0
        w = math.cos(math.pi / 4.0)

    class _Pose:
        position = _Position()
        orientation = _Orientation()

    class _Message:
        header = _Header()
        pose = _Pose()

    row = _pose_row(_Message(), timestamp_ns=0, topic="/sim/finsrov/controller/pose")
    assert row is not None
    assert math.isclose(float(row["yaw_rad"]), math.pi / 2.0, abs_tol=1e-8)


def test_generic_xy_trajectory_excludes_controller_pose_topics():
    poses = [
        {"topic": "/finsrov/pose", "frame_id": "pool_world", "x_m": 1.0, "y_m": 2.0},
        {"topic": "/finsrov/controller/pose", "frame_id": "controller_world", "x_m": 1.0, "y_m": -0.5},
        {"topic": "/sim/finsrov/controller/pose", "frame_id": "controller_world", "x_m": 1.0, "y_m": -0.5},
        {"topic": "/finsrov/vision/refracted_pose_6d", "frame_id": "pool_world", "x_m": 1.1, "y_m": 2.1},
    ]

    selected = _trajectory_xy_poses(poses)

    assert [row["topic"] for row in selected] == [
        "/finsrov/pose",
        "/finsrov/vision/refracted_pose_6d",
    ]


def test_localization_noise_metrics_require_clean_evaluation_pose():
    poses = [
        {"topic": "/finsrov/controller/pose"},
        {"topic": "/finsrov/evaluation/pose"},
    ]
    assert _select_control_pose_topic(
        poses, required_topic="/finsrov/evaluation/pose"
    ) == "/finsrov/evaluation/pose"
    assert _select_control_pose_topic(poses, required_topic="/missing/evaluation/pose") is None


def test_t1_protocol_success_uses_pose_and_yaw_without_velocity_gate():
    contract = {
        "x_m": 0.10,
        "y_m": 0.10,
        "z_m": 0.10,
        "yaw_rad": math.radians(10.0),
        "continuous_hold_sec": 10.0,
        "maximum_sample_gap_sec": 0.25,
    }
    rows = [
        {
            "timestamp_sec": float(index),
            "x_m": 0.05,
            "y_m": -0.05,
            "z_m": 0.05,
            "yaw_rad": math.radians(5.0),
            # A deliberately large velocity field must not affect the
            # pose-only success criterion.
            "linear_velocity_mps": 9.0,
        }
        for index in range(11)
    ]
    assert _first_t1_protocol_success_time(rows, (0.0, 0.0, 0.0, 0.0), contract) is None

    dense_rows = [
        {**row, "timestamp_sec": index * 0.1}
        for index, row in enumerate(rows * 11)
    ]
    assert _first_t1_protocol_success_time(
        dense_rows, (0.0, 0.0, 0.0, 0.0), contract
    ) == 10.0


def test_t1_primary_window_is_fixed_to_late_hold_interval():
    settings = _window_settings(
        {
            "primary_evaluation_window": {
                "start_offset_sec": 40.0,
                "duration_sec": 20.0,
                "resample_hz": 10.0,
                "minimum_valid_fraction": 0.90,
            },
            "response_window": {"duration_sec": 60.0},
        },
        start_sec=100.0,
        end_sec=160.0,
    )
    assert settings["response_start_sec"] == 100.0
    assert settings["response_end_sec"] == 160.0
    assert settings["evaluation_start_sec"] == 140.0
    assert settings["evaluation_end_sec"] == 160.0
    assert settings["protocol_version"] == "primary_late_hold"


def test_primary_window_resampling_uses_shortest_yaw_path():
    rows, expected = _resample_pose_segment(
        [
            {"timestamp_sec": 40.0, "topic": "/finsrov/controller/pose", "x_m": 1.0, "y_m": 2.0, "z_m": 3.0, "yaw_rad": math.radians(179.0)},
            {"timestamp_sec": 40.2, "topic": "/finsrov/controller/pose", "x_m": 3.0, "y_m": 4.0, "z_m": 5.0, "yaw_rad": math.radians(-179.0)},
        ],
        start_sec=40.0,
        end_sec=40.2,
        rate_hz=10.0,
        max_interpolation_gap_sec=0.25,
    )
    assert expected == 3
    assert len(rows) == 3
    assert math.isclose(rows[1]["x_m"], 2.0)
    assert math.isclose(abs(rows[1]["yaw_rad"]), math.pi, abs_tol=1e-8)


def test_trial_windows_deduplicates_repeated_hold_start_publications():
    metadata = {
        "setpoint_id": "P01",
        "target_controller_world": [1.0, -0.5, 0.3, 0.0],
        "primary_evaluation_window": {"start_offset_sec": 0.0, "duration_sec": 1.0},
        "response_window": {"duration_sec": 1.0},
    }
    events = [
        {"event": "hold_start", "label": "P01", "timestamp_sec": 10.00, "metadata": metadata},
        {"event": "hold_start", "label": "P01", "timestamp_sec": 10.05, "metadata": metadata},
        {"event": "hold_start", "label": "P01", "timestamp_sec": 10.10, "metadata": metadata},
        {"event": "phase_end", "label": "hold", "timestamp_sec": 11.00, "metadata": metadata},
    ]

    windows = _trial_windows(events)

    assert len(windows) == 1
    assert windows[0]["label"] == "P01"
    assert windows[0]["start_sec"] == 10.0


def test_t1_lockstep_window_uses_the_first_controller_active_goal_tick_not_event_arrival():
    metadata = {
        "setpoint_id": "P01",
        "target_controller_world": [1.0, -0.5, 0.3, 0.0],
        "primary_evaluation_window": {"start_offset_sec": 0.0, "duration_sec": 1.0},
        "response_window": {"duration_sec": 1.0},
        "timing_contract": {"hold_start_mode": "first_controller_active_goal_tick_v1"},
    }
    events = [
        {"event": "hold_start", "label": "P01", "timestamp_sec": 10.2, "metadata": metadata},
        {"event": "phase_end", "label": "hold", "timestamp_sec": 11.2, "metadata": metadata},
    ]
    active_goals = [
        {
            "timestamp_sec": 10.0,
            "topic": "/sim/motion_controller/status/active_pose",
            "frame_id": "controller_world",
            "x_m": 1.0,
            "y_m": -0.5,
            "z_m": 0.3,
            "yaw_rad": 0.0,
        }
    ]

    windows = _trial_windows(events, active_position_goals=active_goals)

    assert len(windows) == 1
    assert windows[0]["start_sec"] == 10.0
    assert windows[0]["timing_provenance"] == "first_controller_active_goal_tick"


def test_t1_lockstep_audit_requires_state_command_and_ack_coverage_on_one_timebase():
    metadata = {
        "target_controller_world": [1.0, -0.5, 0.3, 0.0],
        "response_window": {"duration_sec": 1.0},
        "simulation_clock": {"mode": "ros2_control_lockstep"},
        "timing_contract": {"hold_start_mode": "first_controller_active_goal_tick_v1"},
    }
    active_goals = [
        {
            "timestamp_sec": 10.0,
            "topic": "/sim/motion_controller/status/active_pose",
            "frame_id": "controller_world",
            "x_m": 1.0,
            "y_m": -0.5,
            "z_m": 0.3,
            "yaw_rad": 0.0,
        }
    ]
    poses = [
        {
            "timestamp_sec": 10.0 + 0.02 * index,
            "topic": "/sim/finsrov/controller/pose",
            "frame_id": "controller_world",
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_rad": 0.0,
        }
        for index in range(51)
    ]
    stamped_commands = [
        {"timestamp_sec": 10.0 + 0.1 * index, "topic": "/sim/motion_controller/debug/thruster_command_stamped", "values": [0.0] * 8}
        for index in range(11)
    ]
    acks = [
        {"timestamp_sec": 10.0 + 0.02 * index, "topic": "/sim/motion_controller/debug/control_tick_complete", "tick_ns": 0}
        for index in range(51)
    ]

    audit = _t1_lockstep_alignment_audit(
        metadata=metadata,
        poses=poses,
        active_position_goals=active_goals,
        stamped_commands=stamped_commands,
        lockstep_acks=acks,
    )

    assert audit is not None
    assert audit["passes"] is True


def test_trial_windows_recovers_missing_hold_start_from_matching_position_goal():
    metadata = {
        "setpoint_id": "P03",
        "target_controller_world": [-1.0, -0.5, -0.3, -45.0],
        "primary_evaluation_window": {
            "start_offset_sec": 40.0,
            "duration_sec": 20.0,
            "resample_hz": 10.0,
            "minimum_valid_fraction": 0.90,
        },
        "response_window": {"duration_sec": 60.0},
    }
    events = [
        {"event": "phase_end", "label": "hold", "timestamp_sec": 70.0, "metadata": metadata},
        {"event": "trial_end", "label": "P03", "timestamp_sec": 75.0, "metadata": metadata},
    ]
    position_goals = [
        {
            "timestamp_sec": 10.0,
            "topic": "/sim/motion_controller/command/position_controller_world",
            "frame_id": "controller_world",
            "x_m": -1.0,
            "y_m": -0.5,
            "z_m": -0.3,
            "yaw_rad": math.radians(-45.0),
        },
        {
            "timestamp_sec": 11.0,
            "topic": "/sim/motion_controller/command/position_controller_world",
            "frame_id": "controller_world",
            "x_m": 0.0,
            "y_m": 0.0,
            "z_m": 0.0,
            "yaw_rad": 0.0,
        },
    ]

    windows = _trial_windows(
        events,
        manifest_metadata=metadata,
        position_goals=position_goals,
    )

    assert len(windows) == 1
    assert windows[0]["label"] == "P03"
    assert windows[0]["start_sec"] == 10.0
    assert windows[0]["evaluation_start_sec"] == 50.0
    assert windows[0]["timing_provenance"] == "position_goal_inferred_missing_hold_start"


def test_trial_windows_recovers_missing_hold_start_from_calibrated_phase_end():
    metadata = {
        "setpoint_id": "P04",
        "target_controller_world": [-1.0, -0.5, 0.3, 45.0],
        "primary_evaluation_window": {
            "start_offset_sec": 40.0,
            "duration_sec": 20.0,
            "resample_hz": 10.0,
            "minimum_valid_fraction": 0.90,
        },
        "response_window": {"duration_sec": 60.0},
    }
    events = [
        {"event": "phase_end", "label": "hold", "timestamp_sec": 70.6, "metadata": metadata},
        {"event": "trial_end", "label": "P04", "timestamp_sec": 75.0, "metadata": metadata},
    ]

    windows = _trial_windows(
        events,
        manifest_metadata=metadata,
        position_goals=[],
        phase_end_delay_sec=0.6,
    )

    assert len(windows) == 1
    assert math.isclose(windows[0]["start_sec"], 10.0)
    assert math.isclose(windows[0]["evaluation_start_sec"], 50.0)
    assert windows[0]["timing_provenance"] == "phase_end_calibrated_missing_hold_start"


def test_analysis_always_writes_report_when_bag_is_missing(tmp_path: Path):
    session = tmp_path / "E1" / "session_001"
    session.mkdir(parents=True)
    (session / "manifest.json").write_text(
        json.dumps({"session_id": "session_001", "experiment_id": "E1", "profile": "apriltag"}),
        encoding="utf-8",
    )
    report = analyze_session(session)
    assert report["status"] == "needs_review"
    assert (session / "derived" / "analysis_report.json").is_file()


def test_analysis_reads_horizontal_truth(tmp_path: Path):
    session = tmp_path / "E1" / "session_002"
    (session / "external_truth").mkdir(parents=True)
    (session / "manifest.json").write_text("{}", encoding="utf-8")
    (session / "external_truth" / "truth_xy.csv").write_text(
        "truth_x_m,truth_y_m\n0.25,-0.40\n", encoding="utf-8"
    )
    report = analyze_session(session)
    assert report["status"] == "needs_review"
    assert "error" in report or "warnings" in report
