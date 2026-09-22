import numpy as np
import pytest

from hydrodynamic_identification.refit_hydrodynamic_identification import (
    _logged_linear_acceleration,
    _replay_linear_acceleration_filter,
)


def _row(*, logged_x: float, raw_x: float, gravity_x: float = 0.0) -> dict[str, str]:
    return {
        "nudot_x_mps2": str(logged_x),
        "nudot_y_mps2": "0.0",
        "nudot_z_mps2": "0.0",
        "imu_controller_linear_accel_x": str(raw_x),
        "imu_controller_linear_accel_y": "0.0",
        "imu_controller_linear_accel_z": "0.0",
        "imu_controller_gravity_x": str(gravity_x),
        "imu_controller_gravity_y": "0.0",
        "imu_controller_gravity_z": "0.0",
    }


def test_logged_acceleration_reuses_the_online_filtered_csv_value():
    assert _logged_linear_acceleration(_row(logged_x=1.25, raw_x=3.0)) == pytest.approx([1.25, 0.0, 0.0])


def test_recomputed_acceleration_replays_filter_over_all_rows():
    rows = [_row(logged_x=0.0, raw_x=2.0), _row(logged_x=0.0, raw_x=4.0)]

    accelerations, diagnostics = _replay_linear_acceleration_filter(
        rows,
        gravity_mps2=9.80665,
        gravity_compensation_enabled=False,
        acceleration_filter_alpha=0.5,
    )

    assert accelerations[0] == pytest.approx([1.0, 0.0, 0.0])
    assert accelerations[1] == pytest.approx([2.5, 0.0, 0.0])
    assert diagnostics[0][0] == pytest.approx([0.0, 0.0, 0.0])
    assert diagnostics[1][1] == pytest.approx([4.0, 0.0, 0.0])


def test_recomputed_acceleration_uses_recorded_gravity_before_filtering():
    rows = [_row(logged_x=0.0, raw_x=4.0, gravity_x=1.0)]

    accelerations, _ = _replay_linear_acceleration_filter(
        rows,
        gravity_mps2=9.80665,
        gravity_compensation_enabled=True,
        acceleration_filter_alpha=1.0,
    )

    assert np.asarray(accelerations[0]) == pytest.approx([3.0, 0.0, 0.0])
