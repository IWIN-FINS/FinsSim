from experiment_recorder.t1_disturbance import (
    DisturbanceDetector,
    make_state,
    state_within_thresholds,
    wrap_degrees,
)


DETECTION = {
    "x_error_m": 0.20,
    "z_error_m": 0.20,
    "persistence_sec": 0.20,
}

READY = {
    "x_m": 0.10,
    "depth_m": 0.10,
    "z_m": 0.10,
    "yaw_deg": 10.0,
}


def _state(time_sec: float, x: float, z: float = 0.0, *, healthy: bool = True):
    return make_state(
        timestamp_sec=time_sec,
        pose=(x, -0.5, z, 0.0),
        target=(0.0, -0.5, 0.0, 0.0),
        velocity=(None, None, None, None),
        health_ok=healthy,
    )


def test_detector_requires_horizontal_error_and_persistence() -> None:
    detector = DisturbanceDetector(DETECTION)
    assert detector.observe(_state(0.0, 0.19)) is None
    assert detector.observe(_state(0.1, 0.20)) is None
    assert detector.observe(_state(0.2, 0.20)) is None
    result = detector.observe(_state(0.31, 0.20))
    assert result is not None
    assert result["detected_axis"] == "x"
    assert result["candidate_persistence_sec"] >= 0.20
    assert result["detection_source"] == "controller_world_horizontal_position_error_debounced"


def test_detector_uses_z_but_not_depth_or_yaw() -> None:
    detector = DisturbanceDetector(DETECTION)
    depth_only = make_state(
        timestamp_sec=0.0,
        pose=(0.0, -0.1, 0.0, 30.0),
        target=(0.0, -0.5, 0.0, 0.0),
        velocity=(None, None, None, None),
        health_ok=True,
    )
    assert detector.observe(depth_only) is None
    assert detector.observe(_state(0.1, 0.0, 0.20)) is None
    assert detector.observe(_state(0.31, 0.0, 0.20)) is not None


def test_detector_rejects_unhealthy_state_and_resets_candidate() -> None:
    detector = DisturbanceDetector(DETECTION)
    assert detector.observe(_state(0.0, 0.20)) is None
    assert detector.observe(_state(0.1, 0.20, healthy=False)) is None
    assert detector.observe(_state(0.2, 0.20)) is None
    assert detector.observe(_state(0.41, 0.20)) is not None


def test_yaw_error_wraps_at_plus_minus_180() -> None:
    assert abs(wrap_degrees(-358.0) - 2.0) < 1e-9
    state = make_state(
        timestamp_sec=0.0,
        pose=(0.0, -0.5, 0.0, -179.0),
        target=(0.0, -0.5, 0.0, 179.0),
        velocity=(0.0, 0.0, 0.0, 0.0),
        health_ok=True,
    )
    assert abs(state.yaw_error_deg - 2.0) < 1e-9
    assert state_within_thresholds(state, READY)
