from motion_control.state_estimator import VehicleStateEstimator


def test_fresh_returns_false_before_first_stamp():
    assert VehicleStateEstimator._fresh(None, now_sec=10.0, timeout_sec=0.5) is False


def test_fresh_accepts_recent_stamp():
    assert VehicleStateEstimator._fresh(9.7, now_sec=10.0, timeout_sec=0.5) is True


def test_fresh_rejects_stale_stamp():
    assert VehicleStateEstimator._fresh(9.0, now_sec=10.0, timeout_sec=0.5) is False


def test_fresh_timeout_zero_disables_age_limit_after_first_stamp():
    assert VehicleStateEstimator._fresh(1.0, now_sec=10.0, timeout_sec=0.0) is True
