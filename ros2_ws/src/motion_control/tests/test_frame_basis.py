import math

import numpy as np
import pytest

from motion_control.controller_node import (
    _position_to_controller_frame,
    _quat_to_controller_frame,
    _vector_to_controller_frame,
)
from motion_control.math_utils import build_basis_matrix, quat_to_matrix, transform_quat_basis


def test_ros_vector_frame_maps_z_up_to_controller_y_up():
    basis = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _vector_to_controller_frame([1.0, 2.0, -3.0], basis, "pool_world")

    assert converted == pytest.approx([1.0, -3.0, 2.0])


def test_controller_vector_frame_is_not_remapped():
    basis = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _vector_to_controller_frame([1.0, 2.0, -3.0], basis, "controller_world")

    assert converted == pytest.approx([1.0, 2.0, -3.0])


def test_ros_position_frame_applies_controller_offset_after_basis_mapping():
    basis = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _position_to_controller_frame(
        [1.0, -0.4, 0.98],
        basis,
        [0.0, -0.98, 0.0],
        "pool_world",
    )

    assert converted == pytest.approx([1.0, 0.0, -0.4])


def test_unity_sim_pose_keeps_controller_y_without_water_surface_offset():
    basis = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])
    unity_position = np.array([0.2, -1.0, -0.4], dtype=np.float32)
    ros_position_from_unity2map = [unity_position[0], unity_position[2], unity_position[1]]

    converted = _position_to_controller_frame(
        ros_position_from_unity2map,
        basis,
        [0.0, 0.0, 0.0],
        "pool_world",
    )

    assert converted == pytest.approx(unity_position)


def test_controller_position_frame_does_not_apply_ros_offset_twice():
    basis = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])

    converted = _position_to_controller_frame(
        [1.0, 0.0, -0.4],
        basis,
        [0.0, -0.98, 0.0],
        "controller_world",
    )

    assert converted == pytest.approx([1.0, 0.0, -0.4])


def test_ros_quaternion_uses_matrix_basis_transform():
    basis = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])
    half_yaw = math.radians(45.0)
    ros_yaw_90 = np.array([0.0, 0.0, math.sin(half_yaw), math.cos(half_yaw)], dtype=np.float64)

    converted = _quat_to_controller_frame(ros_yaw_90, basis, "pool_world")

    assert converted == pytest.approx(transform_quat_basis(ros_yaw_90, basis))
    assert np.linalg.det(quat_to_matrix(converted)) == pytest.approx(1.0)


def test_controller_quaternion_frame_is_not_remapped():
    basis = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])
    quat = np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float64)
    quat /= np.linalg.norm(quat)

    converted = _quat_to_controller_frame(quat, basis, "controller_body")

    assert converted == pytest.approx(quat)
