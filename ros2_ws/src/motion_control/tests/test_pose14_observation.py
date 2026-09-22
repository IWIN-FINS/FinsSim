import numpy as np

from motion_control.math_utils import quat_from_controller_ypr, quat_to_matrix
from motion_control.observations import (
    build_pose13_observation,
    build_pose14_observation,
    build_pose16_rot6d_observation,
)
from motion_control.state_estimator import VehicleState


def test_pose13_observation_matches_moving_target_no_yaw_layout():
    state = VehicleState(
        position_world=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        position_ros=np.array([1.0, 3.0, 2.0], dtype=np.float32),
        orientation_world_body=np.array([0.0, 0.0, 0.70710677, 0.70710677], dtype=np.float32),
        linear_velocity_body=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        linear_velocity_world=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        linear_acceleration_body=np.zeros(3, dtype=np.float32),
        linear_acceleration_world=np.zeros(3, dtype=np.float32),
        angular_velocity_body_xyz=np.array([0.7, 0.8, 0.9], dtype=np.float32),
        angular_velocity_body_ypr=np.array([0.8, 0.9, 0.7], dtype=np.float32),
        angular_velocity_world=np.array([0.7, 0.8, 0.9], dtype=np.float32),
        stamp_sec=1.0,
    )
    target_position = np.array([4.0, 5.0, 6.0], dtype=np.float32)

    observation = build_pose13_observation(state, target_position)

    assert observation.shape == (13,)
    np.testing.assert_allclose(observation[0:3], [4.0, 5.0, 6.0], atol=1e-6)
    np.testing.assert_allclose(observation[3:6], [1.0, 2.0, 3.0], atol=1e-6)
    np.testing.assert_allclose(observation[6:10], [0.0, 0.0, 0.70710677, 0.70710677], atol=1e-6)
    np.testing.assert_allclose(observation[10:13], [0.1, 0.2, 0.3], atol=1e-6)


def test_pose14_observation_matches_control_for_position_layout():
    state = VehicleState(
        position_world=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        position_ros=np.array([1.0, 3.0, 2.0], dtype=np.float32),
        orientation_world_body=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        linear_velocity_body=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        linear_velocity_world=np.array([0.1, 0.2, 0.3], dtype=np.float32),
        linear_acceleration_body=np.zeros(3, dtype=np.float32),
        linear_acceleration_world=np.zeros(3, dtype=np.float32),
        angular_velocity_body_xyz=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        angular_velocity_body_ypr=np.array([0.5, 0.6, 0.4], dtype=np.float32),
        angular_velocity_world=np.array([0.4, 0.5, 0.6], dtype=np.float32),
        stamp_sec=1.0,
    )
    target_position = np.array([1.0, -1.0, 3.0], dtype=np.float32)
    target_orientation = quat_from_controller_ypr(np.deg2rad(90.0), 0.0, 0.0)

    observation = build_pose14_observation(state, target_position, target_orientation)

    assert observation.shape == (14,)
    np.testing.assert_allclose(observation[0:3], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(observation[3:7], target_orientation, atol=1e-6)
    np.testing.assert_allclose(observation[7:10], [0.1, 0.2, 0.3], atol=1e-6)
    np.testing.assert_allclose(observation[10:13], [0.4, 0.5, 0.6], atol=1e-6)
    np.testing.assert_allclose(observation[13], 1.0, atol=1e-6)


def test_pose16_rot6d_observation_matches_incremental_reward_layout():
    state = VehicleState(
        position_world=np.array([1.0, 2.0, 3.0], dtype=np.float32),
        position_ros=np.array([1.0, 3.0, 2.0], dtype=np.float32),
        orientation_world_body=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        linear_velocity_body=np.array([3.0, -3.0, 0.5], dtype=np.float32),
        linear_velocity_world=np.array([3.0, -3.0, 0.5], dtype=np.float32),
        linear_acceleration_body=np.zeros(3, dtype=np.float32),
        linear_acceleration_world=np.zeros(3, dtype=np.float32),
        angular_velocity_body_xyz=np.array([0.4, 5.0, -5.0], dtype=np.float32),
        angular_velocity_body_ypr=np.array([5.0, -5.0, 0.4], dtype=np.float32),
        angular_velocity_world=np.array([0.4, 5.0, -5.0], dtype=np.float32),
        stamp_sec=1.0,
    )
    target_position = np.array([1.0, -1.0, 3.0], dtype=np.float32)
    target_orientation = quat_from_controller_ypr(np.deg2rad(90.0), 0.0, 0.0)
    target_matrix = quat_to_matrix(target_orientation)

    observation = build_pose16_rot6d_observation(
        state,
        target_position,
        target_orientation,
        position_scale=3.0,
        linear_velocity_scale=(1.0, 2.0, 0.5),
        angular_velocity_scale=(0.2, 2.0, 1.0),
        velocity_clip=2.0,
    )

    assert observation.shape == (16,)
    np.testing.assert_allclose(observation[0:3], [0.0, -1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(
        observation[3:9],
        [
            target_matrix[0, 0],
            target_matrix[1, 0],
            target_matrix[2, 0],
            target_matrix[0, 1],
            target_matrix[1, 1],
            target_matrix[2, 1],
        ],
        atol=1e-6,
    )
    np.testing.assert_allclose(observation[9:12], [2.0, -1.5, 1.0], atol=1e-6)
    np.testing.assert_allclose(observation[12:15], [2.0, 2.0, -2.0], atol=1e-6)
    np.testing.assert_allclose(observation[15], 1.0, atol=1e-6)
