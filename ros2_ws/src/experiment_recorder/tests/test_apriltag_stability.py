from experiment_recorder.apriltag_stability import (
    _capture_observation,
    _false_intervals,
    _phase_for_hold_age,
    _rate_summary,
)


def _row(timestamp, value):
    return {"elapsed_sec": timestamp, "payload": {"valid": value}}


def test_phase_windows_are_event_relative_not_arrival_relative():
    windows = {"transient_end_sec": 40.0, "late_end_sec": 60.0}
    assert _phase_for_hold_age(0.0, windows) == "hold_transient"
    assert _phase_for_hold_age(39.9, windows) == "hold_transient"
    assert _phase_for_hold_age(40.0, windows) == "late_hold"
    assert _phase_for_hold_age(60.0, windows) == "hold_after_registered_window"


def test_t1_keeps_only_the_event_defined_control_window():
    assert not _capture_observation("t1", "waiting_for_t1")
    assert not _capture_observation("t1", "acquisition")
    assert _capture_observation("t1", "hold")
    assert not _capture_observation("t1", "post_trial")
    assert not _capture_observation("t2", "acquisition")
    assert _capture_observation("t2", "trajectory")
    assert not _capture_observation("t2", "post_trial")
    assert _capture_observation("free_drift", "free_drift")


def test_false_intervals_and_availability():
    samples = [_row(0.0, True), _row(1.0, False), _row(3.0, False), _row(4.0, True), _row(5.0, False)]
    assert _false_intervals(samples, value_key="valid", end_sec=6.0) == [
        {"start_sec": 1.0, "end_sec": 4.0, "duration_sec": 3.0},
        {"start_sec": 5.0, "end_sec": 6.0, "duration_sec": 1.0},
    ]
    assert _rate_summary(samples, value_key="valid") == {
        "sample_count": 5,
        "true_count": 2,
        "availability": 0.4,
    }
