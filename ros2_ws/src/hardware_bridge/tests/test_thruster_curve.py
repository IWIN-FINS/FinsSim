import math

import pytest

from hardware_bridge.thruster_curve import SignedQuadraticThrusterCurve


def test_force_to_normalized_rpm_uses_signed_c1_and_limits():
    curve = SignedQuadraticThrusterCurve(
        c1_positive=(1.0e-4,) * 8,
        c1_negative=(2.0e-4,) * 8,
        rpm_min=(-3000.0,) * 8,
        rpm_max=(3000.0,) * 8,
        force_deadband_n=(0.0,) * 8,
        min_effective_rpm_positive=(0.0,) * 8,
        min_effective_rpm_negative=(0.0,) * 8,
        firmware_max_rpm=3000.0,
    )

    normalized, rpm = curve.forces_to_normalized_rpm([0.0, 1.0, -1.0, 1000.0, 0.0, 0.0, 0.0, 0.0])

    assert rpm[0] == pytest.approx(0.0)
    assert rpm[1] == pytest.approx(math.sqrt(1.0 / 1.0e-4) * 60.0 / (2.0 * math.pi))
    assert rpm[2] == pytest.approx(-math.sqrt(1.0 / 2.0e-4) * 60.0 / (2.0 * math.pi))
    assert rpm[3] == pytest.approx(3000.0)
    assert normalized[3] == pytest.approx(1.0)


def test_negative_c1_fit_is_handled_with_absolute_value():
    curve = SignedQuadraticThrusterCurve(
        c1_positive=(-1.0e-4,) * 8,
        c1_negative=(-2.0e-4,) * 8,
        rpm_min=(-3000.0,) * 8,
        rpm_max=(3000.0,) * 8,
        force_deadband_n=(0.0,) * 8,
        min_effective_rpm_positive=(0.0,) * 8,
        min_effective_rpm_negative=(0.0,) * 8,
        firmware_max_rpm=3000.0,
    )

    _, rpm = curve.forces_to_normalized_rpm([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    assert rpm[0] > 0.0
    assert rpm[1] < 0.0


def test_rejects_missing_curve_values():
    with pytest.raises(ValueError):
        SignedQuadraticThrusterCurve(
            c1_positive=(1.0e-4,) * 7,
            c1_negative=(1.0e-4,) * 8,
            rpm_min=(-3000.0,) * 8,
            rpm_max=(3000.0,) * 8,
            force_deadband_n=(0.0,) * 8,
            min_effective_rpm_positive=(0.0,) * 8,
            min_effective_rpm_negative=(0.0,) * 8,
            firmware_max_rpm=3000.0,
        )


def test_force_deadband_and_min_effective_rpm_are_applied():
    curve = SignedQuadraticThrusterCurve(
        c1_positive=(1.0e-4,) * 8,
        c1_negative=(1.0e-4,) * 8,
        rpm_min=(-3000.0,) * 8,
        rpm_max=(3000.0,) * 8,
        force_deadband_n=(0.1,) * 8,
        min_effective_rpm_positive=(500.0,) * 8,
        min_effective_rpm_negative=(400.0,) * 8,
        firmware_max_rpm=3000.0,
    )

    assert curve.force_to_rpm(0.05, 0) == pytest.approx(0.0)
    assert curve.force_to_rpm(0.2, 0) == pytest.approx(500.0)
    assert curve.force_to_rpm(-0.15, 0) == pytest.approx(-400.0)
