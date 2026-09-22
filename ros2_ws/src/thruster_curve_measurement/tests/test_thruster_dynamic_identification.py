import math
from pathlib import Path

from thruster_curve_measurement.thruster_dynamic_identification import (
    Sample,
    fit_transition,
    load_c1_values,
)


def test_fopdt_fit_recovers_synthetic_response():
    samples = []
    for index in range(500):
        elapsed = index * 0.01
        thrust = 0.0 if elapsed < 0.3 else 1.0 * (1.0 - math.exp(-(elapsed - 0.3) / 0.2))
        samples.append(Sample(elapsed, "pulse", 1.0, None, None, None, None, thrust))

    fit = fit_transition(
        samples,
        label="synthetic",
        direction="rise",
        command_n=1.0,
        start_sec=0.0,
        end_sec=4.0,
    )

    assert fit is not None
    assert abs(fit.delay_sec - 0.3) < 0.03
    assert abs(fit.time_constant_sec - 0.2) < 0.03
    assert fit.first_order_adequate


def test_c1_config_contains_all_canonical_thrusters():
    values = load_c1_values(
        Path(
            "./ros2_ws/src/"
            "thruster_curve_measurement/config/thruster_rpm_force_unity.json"
        )
    )
    assert len(values) == 8
    assert all(positive > 0.0 and negative > 0.0 for positive, negative in values)
