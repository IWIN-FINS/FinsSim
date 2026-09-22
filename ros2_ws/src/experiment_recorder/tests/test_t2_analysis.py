import math

import pytest

from experiment_recorder.t2_analysis import (
    _controller_to_pool,
    _controller_state_is_fresh,
    _cross_track_error,
    _error_summary,
    _event_window,
    _interpolate_pose,
    _lockstep_controller_window,
    _latest_status_before,
    _pool_to_controller,
    _resample_runtime_reference,
    _resolve_reference_timing,
    _runtime_reference_inferred_window,
    _select_t2_pose_source,
    _split_circle_loops,
)


def test_interpolate_pose_respects_gap_without_creating_a_yaw_metric():
    samples = [
        {"timestamp_sec": 0.0, "x_m": 0.0, "y_m": -0.5, "z_m": 0.0, "yaw_rad": 3.10},
        {"timestamp_sec": 0.1, "x_m": 1.0, "y_m": -0.5, "z_m": 0.0, "yaw_rad": -3.10},
    ]
    result = _interpolate_pose(samples, 0.05, 0.25)
    assert result is not None
    assert result["x_m"] == pytest.approx(0.5)
    assert "yaw_rad" not in result
    assert _interpolate_pose(samples, 0.05, 0.05) is None


def test_cross_track_uses_local_reference_segments():
    rows = [
        {"actual_x_m": 0.0, "actual_z_m": 0.1, "ref_x_m": 0.0, "ref_z_m": 0.0},
        {"actual_x_m": 1.0, "actual_z_m": 0.1, "ref_x_m": 1.0, "ref_z_m": 0.0},
        {"actual_x_m": 2.0, "actual_z_m": 0.1, "ref_x_m": 2.0, "ref_z_m": 0.0},
    ]
    assert _cross_track_error(rows, 1) == pytest.approx(0.1)


def test_state_freshness_accepts_controller_coast_but_rejects_stale_or_missing_sensors():
    status = {
        "timestamp_sec": 1.0,
        "ready": True,
        "initialized": True,
        "imu_fresh": True,
        "depth_fresh": True,
        "vision_mode": "coast",
    }
    selected, age = _latest_status_before([status], 1.1, 0.25)
    assert age == pytest.approx(0.1)
    assert _controller_state_is_fresh(selected, {"allowed_vision_modes": ["fresh", "coast"]}) == (True, "coast")
    assert _latest_status_before([status], 1.3, 0.25)[0] is None
    stale_imu = {**status, "imu_fresh": False}
    assert _controller_state_is_fresh(stale_imu, {}) == (False, "imu_fresh")


def test_t2_window_starts_at_trajectory_start_and_uses_the_declared_duration():
    events = [
        {"event": "trial_start", "timestamp_sec": 2.0},
        {"event": "trajectory_start", "timestamp_sec": 5.0},
        {"event": "trajectory_end", "timestamp_sec": 40.0},
    ]
    assert _event_window(events, duration_sec=30.0) == pytest.approx((5.0, 35.0))
    early_abort = [*events[:-1], {"event": "safety_abort", "timestamp_sec": 17.0}]
    assert _event_window(early_abort, duration_sec=30.0) == pytest.approx((5.0, 17.0))
    operator_abort = [*events[:-1], {"event": "operator_abort", "timestamp_sec": 18.0}]
    assert _event_window(operator_abort, duration_sec=30.0) == pytest.approx((5.0, 18.0))


def test_missing_t2_start_event_can_only_recover_a_runtime_reference_plot_window():
    samples = [
        {"timestamp_sec": 10.2, "elapsed_sec": 0.2},
        {"timestamp_sec": 16.0, "elapsed_sec": 6.0},
        {"timestamp_sec": 39.9, "elapsed_sec": 29.9},
    ]
    events = [{"event": "trajectory_end", "timestamp_sec": 40.0}]
    assert _runtime_reference_inferred_window(samples, events, duration_sec=30.0) == pytest.approx((10.0, 40.0))


def test_runtime_reference_is_resampled_on_the_frozen_grid_without_large_gap_fill():
    samples = [
        {
            "timestamp_sec": time,
            "elapsed_sec": time,
            "ref_x_m": time,
            "ref_y_m": 0.0,
            "ref_z_m": 0.0,
            "ref_vx_mps": 1.0,
            "ref_vy_mps": 0.0,
            "ref_vz_mps": 0.0,
            "duration_sec": 1.0,
        }
        for time in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.0)
    ]
    result, expected = _resample_runtime_reference(
        samples, start_sec=0.0, end_sec=1.0, rate_hz=10.0, max_gap_sec=0.15,
    )
    assert expected == 11
    assert [row["timestamp_sec"] for row in result] == pytest.approx([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.9, 1.0])
    assert _error_summary([3.0, -4.0]) == {
        "rmse": pytest.approx(math.sqrt(12.5)),
        "mae": pytest.approx(3.5),
        "p95_absolute": pytest.approx(4.0),
    }


def test_header_stamped_sim_reference_uses_the_trajectory_start_clock():
    samples = [
        {
            "timestamp_sec": 100.2,
            "elapsed_sec": math.nan,
            "duration_sec": math.nan,
            "ref_x_m": 0.0,
            "ref_y_m": -0.5,
            "ref_z_m": 0.0,
            "timing_source": "message_header_stamped",
        },
        {
            "timestamp_sec": 102.7,
            "elapsed_sec": math.nan,
            "duration_sec": math.nan,
            "ref_x_m": 1.0,
            "ref_y_m": -0.5,
            "ref_z_m": 0.0,
            "timing_source": "message_header_stamped",
        },
    ]
    resolved = _resolve_reference_timing(samples, start_sec=100.0, duration_sec=30.0)
    assert [row["elapsed_sec"] for row in resolved] == pytest.approx([0.2, 2.7])
    assert [row["duration_sec"] for row in resolved] == pytest.approx([30.0, 30.0])


def test_lockstep_window_uses_the_first_exact_controller_tick():
    samples = [
        {
            "timestamp_sec": 101.0,
            "timing_source": "message_header_stamped",
        },
        {
            "timestamp_sec": 101.1,
            "timing_source": "message_header_stamped",
        },
        # A legacy bag receive timestamp must not be allowed to move the
        # lockstep T2 start boundary.
        {
            "timestamp_sec": 12.0,
            "timing_source": "rosbag_receive_time_legacy_array",
        },
    ]
    assert _lockstep_controller_window(samples, duration_sec=30.0) == pytest.approx((101.0, 131.0))


def test_circle_plot_windows_follow_smoothstep_loop_phase_not_equal_time_thirds():
    rows = [
        {"elapsed_sec": time_sec, "duration_sec": 144.0}
        for time_sec in (0.0, 48.0, 72.0, 96.0, 144.0)
    ]
    loops = _split_circle_loops(rows, 3)
    # At 48 s, smoothstep progress is below one third, while 72 s is halfway
    # through the trial and therefore halfway through the second reference loop.
    assert [[row["elapsed_sec"] for row in loop] for loop in loops] == [
        [0.0, 48.0],
        [72.0],
        [96.0, 144.0],
    ]


def test_pool_world_y_z_are_remapped_before_t2_reference_comparison():
    transform = {
        "basis_indices": [0, 2, 1],
        "basis_signs": [1.0, 1.0, 1.0],
        "post_basis_offset_m": [0.0, -0.96, 0.0],
    }
    pool = {"x_m": 0.40, "y_m": -0.25, "z_m": 0.46}
    controller = _pool_to_controller(pool, transform)
    # pool z is controller y (vertical); pool y is controller z (horizontal).
    assert controller == pytest.approx({"x_m": 0.40, "y_m": -0.50, "z_m": -0.25})
    assert _controller_to_pool(controller, transform) == pytest.approx(pool)


def test_t2_selector_refuses_unlabelled_non_pool_pose_but_accepts_pool_fusion_output():
    assert _select_t2_pose_source([
        {"topic": "/finsrov/pose", "frame_id": "pool_world"},
    ]) == ("/finsrov/pose", "pool_world_transformed")
    assert _select_t2_pose_source([
        {"topic": "/finsrov/pose", "frame_id": "camera_optical_frame"},
    ]) is None
