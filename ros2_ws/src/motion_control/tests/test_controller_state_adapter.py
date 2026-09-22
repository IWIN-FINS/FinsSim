import numpy as np
import pytest

from motion_control.controller_state_adapter import ControllerStateAdapter
from motion_control.math_utils import (
    build_basis_matrix,
    transform_axial_vector,
    transform_quat_basis,
)


def make_adapter(water_surface_z_m=0.98):
    adapter = ControllerStateAdapter.__new__(ControllerStateAdapter)
    adapter._basis_matrix = build_basis_matrix([0, 2, 1], [1.0, 1.0, 1.0])
    adapter._water_surface_z_m = float(water_surface_z_m)
    return adapter


def test_pool_surface_maps_to_controller_y_zero():
    adapter = make_adapter(water_surface_z_m=0.98)

    converted = adapter.pool_position_to_controller([1.2, -0.4, 0.98])

    assert converted == pytest.approx([1.2, 0.0, -0.4])


def test_pool_underwater_z_maps_to_negative_controller_y():
    adapter = make_adapter(water_surface_z_m=0.98)

    converted = adapter.pool_position_to_controller(np.array([0.0, 0.2, 0.68], dtype=np.float32))

    assert converted == pytest.approx([0.0, -0.3, 0.2])


def test_raw_flu_body_vector_maps_to_controller_body():
    adapter = make_adapter()

    converted = adapter._vector_to_controller_body([1.0, -0.4, 0.2], "finsrov_base_link")

    assert converted == pytest.approx([1.0, 0.2, -0.4])


def test_controller_body_vector_is_passed_through():
    adapter = make_adapter()

    converted = adapter._vector_to_controller_body([1.0, 0.2, -0.4], "controller_body")

    assert converted == pytest.approx([1.0, 0.2, -0.4])


def test_axial_vector_uses_reflection_sign_for_ros_to_controller_rotation_rate():
    adapter = make_adapter()

    converted = transform_axial_vector([1.0, -0.4, 0.2], adapter._basis_matrix)

    # The default [x, z, y] map has determinant -1. Rotation rates are
    # axial vectors, so they need det(B) * B rather than the linear-vector B.
    assert converted == pytest.approx([-1.0, -0.2, 0.4])


def test_controller_yaw_rate_has_same_sign_as_transformed_pose_yaw():
    adapter = make_adapter()
    ros_yaw_rate = np.array([0.0, 0.0, 1.0], dtype=np.float64)

    controller_yaw_rate = transform_axial_vector(ros_yaw_rate, adapter._basis_matrix)

    assert controller_yaw_rate[1] == pytest.approx(-1.0)


def test_raw_pose_quaternion_is_basis_transformed():
    adapter = make_adapter()
    quat_ros = np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float64)
    quat_ros /= np.linalg.norm(quat_ros)

    converted = adapter._quat_to_controller(quat_ros, "pool_world")

    assert converted == pytest.approx(transform_quat_basis(quat_ros, adapter._basis_matrix))


def test_controller_quaternion_is_passed_through():
    adapter = make_adapter()
    quat = np.array([0.1, -0.2, 0.3, 0.9], dtype=np.float64)
    quat /= np.linalg.norm(quat)

    converted = adapter._quat_to_controller(quat, "controller_body")

    assert converted == pytest.approx(quat)
