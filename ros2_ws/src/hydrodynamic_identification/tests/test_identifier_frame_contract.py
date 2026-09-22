import math

import numpy as np
import pytest

from hydrodynamic_identification.hydrodynamic_identifier_node import (
    HydrodynamicIdentifierNode,
    Sample,
    Trial,
    _axial_vector_to_controller_body,
    _build_basis_matrix,
    _gravity_controller_body_from_quaternion,
    _fit_axis,
    _quat_to_controller,
    _quat_to_controller_rpy_rad,
    _vector_to_controller_body,
)


def test_raw_body_vector_maps_to_controller_body():
    basis = _build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _vector_to_controller_body([1.0, -0.4, 0.2], basis, "finsrov_base_link")

    assert converted == pytest.approx([1.0, 0.2, -0.4])


def test_controller_body_vector_is_not_remapped_again():
    basis = _build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _vector_to_controller_body([1.0, 0.2, -0.4], basis, "controller_body")

    assert converted == pytest.approx([1.0, 0.2, -0.4])


def test_raw_ros_angular_velocity_uses_axial_handedness_transform():
    basis = _build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _axial_vector_to_controller_body([0.1, 0.2, 0.3], basis, "finsrov_base_link")

    assert converted == pytest.approx([-0.1, -0.3, -0.2])


def test_controller_body_angular_velocity_is_not_remapped_again():
    basis = _build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _axial_vector_to_controller_body([-0.1, -0.3, -0.2], basis, "controller_body")

    assert converted == pytest.approx([-0.1, -0.3, -0.2])


def test_raw_ros_yaw_quaternion_matches_controller_adapter_basis_transform():
    basis = _build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])
    half_yaw = math.radians(45.0)
    raw_ros_yaw_90 = [0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)]

    quat_controller, valid = _quat_to_controller(raw_ros_yaw_90, basis, "pool_world")
    roll_x, pitch_z, yaw_y, rpy_valid = _quat_to_controller_rpy_rad(quat_controller)

    assert valid
    assert rpy_valid
    assert roll_x == pytest.approx(0.0, abs=1e-6)
    assert pitch_z == pytest.approx(0.0, abs=1e-6)
    assert yaw_y == pytest.approx(math.radians(-90.0), abs=1e-6)


def test_controller_quaternion_is_not_remapped_again():
    basis = _build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])
    half_yaw = math.radians(45.0)
    controller_yaw_90 = [0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)]

    quat_controller, valid = _quat_to_controller(controller_yaw_90, basis, "controller_body")
    roll_x, pitch_z, yaw_y, rpy_valid = _quat_to_controller_rpy_rad(quat_controller)

    assert valid
    assert rpy_valid
    assert roll_x == pytest.approx(0.0, abs=1e-6)
    assert pitch_z == pytest.approx(0.0, abs=1e-6)
    assert yaw_y == pytest.approx(math.radians(90.0), abs=1e-6)


def test_controller_heading_does_not_appear_as_pitch_near_pi():
    half_yaw = math.radians(85.0)
    controller_yaw_minus_170 = [0.0, -math.sin(half_yaw), 0.0, math.cos(half_yaw)]

    roll_x, pitch_z, yaw_y, valid = _quat_to_controller_rpy_rad(controller_yaw_minus_170)

    assert valid
    assert roll_x == pytest.approx(0.0, abs=1e-6)
    assert pitch_z == pytest.approx(0.0, abs=1e-6)
    assert yaw_y == pytest.approx(math.radians(-170.0), abs=1e-6)


def test_attitude_safety_can_be_ignored_for_supervised_diagnostics():
    node = HydrodynamicIdentifierNode.__new__(HydrodynamicIdentifierNode)
    node._latest_orientation_valid = True
    node._latest_angle_body = [0.0, 0.0, 0.0, 0.0, 0.0, math.radians(5.1)]
    node._latest_angular_body = [0.0, 0.0, 0.0]
    node._attitude_max_abs_angle_rad = {"roll_x": math.radians(5.0), "pitch_z": math.radians(5.0)}
    node._attitude_max_abs_rate_radps = {"roll_x": 0.4, "pitch_z": 0.4}
    node._ignore_attitude_safety_checks = False
    trial = Trial("pitch_z", 0.5)

    exceeded, reason = node._attitude_limit_exceeded(trial)
    assert exceeded
    assert "angle" in reason

    node._ignore_attitude_safety_checks = True
    exceeded, _ = node._attitude_limit_exceeded(trial)
    assert not exceeded


def test_stationary_gravity_is_in_controller_up_axis():
    gravity = _gravity_controller_body_from_quaternion([0.0, 0.0, 0.0, 1.0], 9.80665)

    assert gravity == pytest.approx([0.0, 9.80665, 0.0])


def test_heave_dive_and_coast_trials_are_downward_only():
    node = HydrodynamicIdentifierNode.__new__(HydrodynamicIdentifierNode)
    node._axes = ("heave_y",)
    node._levels = {"heave_y": [4.0, -6.0]}
    node._include_negative = True
    node._heave_y_identification_mode = "dive_and_coast"

    assert node._build_trials() == [Trial("heave_y", -4.0), Trial("heave_y", -6.0)]


def test_held_out_repeats_preserve_axis_and_signed_trial_contract():
    node = HydrodynamicIdentifierNode.__new__(HydrodynamicIdentifierNode)
    node._axes = ("surge_x",)
    node._levels = {"surge_x": [4.0]}
    node._include_negative = True
    node._heave_y_identification_mode = "step"
    node._repeat_per_trial = 2

    assert node._build_trials() == [
        Trial("surge_x", 4.0, 1),
        Trial("surge_x", -4.0, 1),
        Trial("surge_x", 4.0, 2),
        Trial("surge_x", -4.0, 2),
    ]


def test_dive_and_coast_fit_uses_zero_thrust_coast_samples():
    mass, linear, quadratic, bias = 12.0, 4.0, 3.0, -2.0
    samples = []
    for phase, velocity in (("excitation", -0.40), ("excitation", -0.25), ("coast", 0.15), ("coast", 0.30)):
        thrust = -5.0 if phase == "excitation" else 0.0
        acceleration = (thrust - linear * velocity - quadratic * abs(velocity) * velocity - bias) / mass
        samples.append(
            Sample(
                time_sec=float(len(samples)),
                axis="heave_y",
                phase=phase,
                phase_elapsed_sec=0.2,
                sample_window=True,
                tau=[0.0, thrust, 0.0, 0.0, 0.0, 0.0],
                nu=[0.0, velocity, 0.0, 0.0, 0.0, 0.0],
                nu_dot=[0.0, acceleration, 0.0, 0.0, 0.0, 0.0],
                angle=[0.0] * 6,
                orientation_valid=True,
            )
        )

    # Repeat the deterministic pattern so the fitter has adequate samples.
    samples *= 3
    fit = _fit_axis(
        samples,
        "heave_y",
        min_abs_tau=0.05,
        velocity_deadband=0.005,
        max_abs_acceleration=5.0,
        accel_outlier_mad_threshold=0.0,
        accel_spike_local_mad_threshold=0.0,
        accel_spike_local_window=0,
        constrain_physical_coefficients=True,
        max_effective_mass=100.0,
        max_effective_inertia=20.0,
        max_linear_damping=500.0,
        max_quadratic_damping=1000.0,
        max_abs_bias=100.0,
        fit_phases=("excitation", "coast"),
        allow_zero_tau_samples=True,
        fit_type="dive_and_coast",
    )

    assert fit["fit_type"] == "dive_and_coast"
    assert fit["coast_sample_count"] == pytest.approx(6.0)
    assert fit["m_eff"] == pytest.approx(mass, abs=1e-6)
    assert fit["d_linear"] == pytest.approx(linear, abs=1e-6)
    assert fit["d_quadratic"] == pytest.approx(quadratic, abs=1e-6)
    assert fit["bias"] == pytest.approx(bias, abs=1e-6)


def test_heave_coast_waits_for_ascent_after_observed_descent():
    class Logger:
        def warn(self, _message):
            pass

    node = HydrodynamicIdentifierNode.__new__(HydrodynamicIdentifierNode)
    node._latest_linear_body = np.zeros(3)
    node._heave_y_ascent_velocity_threshold = 0.01
    node._heave_y_ascent_sample_sec = 2.0
    node._heave_y_coast_sec = 8.0
    node._heave_y_coast_seen_downward = False
    node._heave_y_ascent_sample_start_sec = None
    node._heave_y_ascent_events = []
    node._trial_index = 3
    node.get_logger = lambda: Logger()
    trial = Trial("heave_y", -6.0)

    # A positive initial drift cannot be mistaken for natural ascent.
    node._latest_linear_body[1] = 0.03
    assert not node._update_heave_y_coast_sampling(0.1, trial, 0.1)
    assert node._heave_y_ascent_sample_start_sec is None

    node._latest_linear_body[1] = -0.04
    assert not node._update_heave_y_coast_sampling(0.2, trial, 0.2)
    assert node._heave_y_coast_seen_downward

    node._latest_linear_body[1] = 0.02
    assert not node._update_heave_y_coast_sampling(0.5, trial, 0.5)
    assert node._heave_y_ascent_sample_start_sec == pytest.approx(0.5)
    assert node._heave_y_ascent_events[0]["detected"] is True

    assert not node._update_heave_y_coast_sampling(2.4, trial, 2.4)
    assert node._update_heave_y_coast_sampling(2.6, trial, 2.6)
