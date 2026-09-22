import pytest

from hardware_bridge.hardware_bridge_node import (
    map_canonical_thrusters_to_mcu_normalized,
    map_motor_rpm_to_canonical,
)
from hardware_bridge.thruster_curve import SignedQuadraticThrusterCurve


def test_motor_rpm_uses_v4_pro1_inverse_mapping():
    motor_order = [4, 5, 0, 1, 2, 7, 6, 3]
    motor_signs = [-1.0, -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0]
    mcu_rpm = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0]

    canonical_rpm = map_motor_rpm_to_canonical(mcu_rpm, motor_order, motor_signs)

    assert canonical_rpm == pytest.approx(
        [
            -30.0,  # input 0 V_LF <- MCU output 2
            40.0,   # input 1 V_LB <- MCU output 3
            -50.0,  # input 2 V_RB <- MCU output 4
            -80.0,  # input 3 V_RF <- MCU output 7
            -10.0,  # input 4 H_LF <- MCU output 0
            -20.0,  # input 5 H_LB <- MCU output 1
            70.0,   # input 6 H_RB <- MCU output 6
            60.0,   # input 7 H_RF <- MCU output 5
        ]
    )


def test_motor_rpm_nonfinite_values_are_zeroed():
    motor_order = list(range(8))
    motor_signs = [1.0] * 8

    canonical_rpm = map_motor_rpm_to_canonical([1.0, float("nan"), 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], motor_order, motor_signs)

    assert canonical_rpm == pytest.approx([1.0, 0.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])


def test_force_n_mapping_uses_curve_then_motor_order_and_signs():
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
    forces_n = [1.0, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0]
    motor_order = [4, 0, 1, 7, 6, 2, 3, 5]
    motor_signs = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0]

    mapped, normalized_canonical, target_rpm = map_canonical_thrusters_to_mcu_normalized(
        forces_n,
        command_mode="force_n",
        output_scale=1.0,
        clamp=True,
        motor_order=motor_order,
        motor_signs=motor_signs,
        firmware_max_rpm=3000.0,
        thruster_curve=curve,
    )

    expected_normalized, expected_rpm = curve.forces_to_normalized_rpm(forces_n)
    expected_mapped = [
        expected_normalized[input_index] * motor_signs[output_index]
        for output_index, input_index in enumerate(motor_order)
    ]

    assert normalized_canonical == pytest.approx(expected_normalized)
    assert target_rpm == pytest.approx(expected_rpm)
    assert mapped == pytest.approx(expected_mapped)
