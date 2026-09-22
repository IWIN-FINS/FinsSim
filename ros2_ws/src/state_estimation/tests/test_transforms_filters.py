import math

import numpy as np
import pytest

from state_estimation.filters import PositionVelocityEKF, VectorLowPass, YawEKF
from state_estimation.state_fusion_node import StateFusionNode
from state_estimation.transforms import (
    inverse_rotate_vector,
    matrix_to_pose,
    normalize_quat_xyzw,
    pose_matrix,
    quat_to_rpy_xyzw,
    rpy_to_quat_xyzw,
    wrap_angle,
)


def test_normalize_quat_falls_back_for_zero():
    assert normalize_quat_xyzw([0.0, 0.0, 0.0, 0.0]).tolist() == [0.0, 0.0, 0.0, 1.0]


def test_inverse_rotate_identity():
    result = inverse_rotate_vector([0.0, 0.0, 0.0, 1.0], [1.0, 2.0, 3.0])
    assert result == pytest.approx([1.0, 2.0, 3.0])


def test_low_pass_filter():
    filt = VectorLowPass(alpha=0.5, size=3)
    assert filt.update([2.0, 0.0, 0.0]) == pytest.approx([2.0, 0.0, 0.0])
    assert filt.update([0.0, 2.0, 0.0]) == pytest.approx([1.0, 1.0, 0.0])
    assert isinstance(filt.value, np.ndarray)


def test_transform_chain_translation_and_yaw():
    t_world_camera = pose_matrix([1.0, 2.0, 0.5], rpy_to_quat_xyzw(0.0, 0.0, math.pi / 2.0))
    t_camera_tag = pose_matrix([1.0, 0.0, 0.0], rpy_to_quat_xyzw(0.0, 0.0, 0.0))
    t_tag_body = pose_matrix([0.2, 0.0, 0.0], rpy_to_quat_xyzw(0.0, 0.0, 0.0))
    position, quat = matrix_to_pose(t_world_camera @ t_camera_tag @ t_tag_body)
    assert position == pytest.approx([1.0, 3.2, 0.5])
    assert quat_to_rpy_xyzw(quat)[2] == pytest.approx(math.pi / 2.0)


def test_transform_loader_accepts_rotation_rpy_deg():
    transform = StateFusionNode._transform_from_node(
        {
            "translation_xyz": [1.0, 2.0, 3.0],
            "rotation_rpy_deg": [0.0, 0.0, 90.0],
        },
        "test",
    )
    position, quat = matrix_to_pose(transform)
    assert position == pytest.approx([1.0, 2.0, 3.0])
    assert quat_to_rpy_xyzw(quat)[2] == pytest.approx(math.pi / 2.0)


def test_extrinsics_loader_accepts_t_body_tag_and_inverts():
    t_body_tag = pose_matrix([0.10, 0.03, 0.18], rpy_to_quat_xyzw(0.0, 0.0, math.pi / 2.0))
    tag_map = StateFusionNode._load_tag_body_map(
        {
            "T_body_tag": {
                15: {
                    "translation_xyz": [0.10, 0.03, 0.18],
                    "rotation_rpy_deg": [0.0, 0.0, 90.0],
                }
            }
        }
    )
    assert 15 in tag_map
    assert tag_map[15] == pytest.approx(np.linalg.inv(t_body_tag))


def test_extrinsics_loader_rejects_duplicate_tag_direction_entries():
    with pytest.raises(ValueError, match="defined in both T_tag_body and T_body_tag"):
        StateFusionNode._load_tag_body_map(
            {
                "T_tag_body": {
                    15: {
                        "translation_xyz": [0.0, 0.0, 0.0],
                        "rotation_rpy_deg": [0.0, 0.0, 0.0],
                    }
                },
                "T_body_tag": {
                    15: {
                        "translation_xyz": [0.0, 0.0, 0.0],
                        "rotation_rpy_deg": [0.0, 0.0, 0.0],
                    }
                },
            }
        )


def test_yaw_ekf_wraps_across_pi():
    ekf = YawEKF(process_noise=0.01)
    ekf.predict(0.0)
    ekf.update(math.radians(179.0), 0.01)
    ekf.predict(0.1, math.radians(40.0))
    ekf.update(math.radians(-179.0), 0.01)
    assert abs(wrap_angle(ekf.yaw - math.pi)) < math.radians(5.0)


def test_position_velocity_ekf_updates_and_gates_outlier():
    ekf = PositionVelocityEKF(process_noise_position=0.01, process_noise_velocity=0.1)
    ekf.predict(0.0)
    accepted, _ = ekf.update_indices([0, 1, 2], [1.0, 2.0, -0.5], np.eye(3) * 0.01)
    assert accepted
    ekf.predict(1.0)
    accepted, _ = ekf.update_indices([0, 1], [100.0, 100.0], np.eye(2) * 0.01, gate_mahalanobis=9.0)
    assert not accepted
    assert np.linalg.norm(ekf.x[:2] - [1.0, 2.0]) < 1.0


def test_position_velocity_ekf_caps_covariance_and_decays_velocity():
    ekf = PositionVelocityEKF(process_noise_position=0.01, process_noise_velocity=0.1)
    ekf.p[0, 0] = 100.0
    ekf.p[1, 1] = 50.0
    ekf.x[3] = 0.2
    ekf.x[4] = -0.00001

    ekf.cap_covariance([0, 1], 4.0)
    ekf.decay_velocity([3, 4], 0.5)

    assert ekf.p[0, 0] == pytest.approx(4.0)
    assert ekf.p[1, 1] == pytest.approx(4.0)
    assert ekf.x[3] == pytest.approx(0.1)
    assert ekf.x[4] == pytest.approx(0.0)
