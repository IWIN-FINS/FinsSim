from __future__ import annotations

from typing import Sequence

import numpy as np

from .math_utils import inverse_rotate_vector, quat_multiply, quat_normalize, quat_to_matrix
from .state_estimator import VehicleState


def _vec3(values: Sequence[float], *, default: float = 0.0) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] < 3:
        array = np.pad(array, (0, 3 - array.shape[0]), constant_values=default)
    elif array.shape[0] > 3:
        array = array[:3]
    return array.astype(np.float32, copy=False)


def build_pose20_observation(
    state: VehicleState,
    target_position_world: Sequence[float],
    target_quaternion_world: Sequence[float] | None = None,
) -> np.ndarray:
    target_position = _vec3(target_position_world)
    if target_quaternion_world is None:
        target_quat = state.orientation_world_body.astype(np.float32, copy=True)
    else:
        target_quat = np.asarray(target_quaternion_world, dtype=np.float32).reshape(-1)
        if target_quat.shape[0] < 4:
            target_quat = np.pad(target_quat, (0, 4 - target_quat.shape[0]), constant_values=0.0)
        elif target_quat.shape[0] > 4:
            target_quat = target_quat[:4]
        quat_norm = float(np.linalg.norm(target_quat))
        if quat_norm <= 1e-8:
            target_quat = state.orientation_world_body.astype(np.float32, copy=True)
        else:
            target_quat = (target_quat / quat_norm).astype(np.float32, copy=False)
    observation = np.concatenate(
        [
            target_position,
            target_quat,
            state.position_world,
            state.orientation_world_body,
            state.linear_velocity_world,
            state.angular_velocity_world,
        ],
        axis=0,
    )
    return observation.astype(np.float32, copy=False)


def build_pose13_observation(
    state: VehicleState,
    target_position_world: Sequence[float],
) -> np.ndarray:
    target_position = _vec3(target_position_world)
    observation = np.concatenate(
        [
            target_position,
            state.position_world,
            state.orientation_world_body,
            state.linear_velocity_world,
        ],
        axis=0,
    )
    return observation.astype(np.float32, copy=False)


def build_pose14_observation(
    state: VehicleState,
    target_position_world: Sequence[float],
    target_quaternion_world: Sequence[float] | None = None,
) -> np.ndarray:
    target_position = _vec3(target_position_world)
    local_target_offset = inverse_rotate_vector(
        state.orientation_world_body,
        target_position - state.position_world,
    )
    if target_quaternion_world is None:
        relative_target_rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        current = quat_normalize(state.orientation_world_body)
        target = quat_normalize(target_quaternion_world)
        inverse_current = np.array([-current[0], -current[1], -current[2], current[3]], dtype=np.float32)
        relative_target_rotation = quat_multiply(inverse_current, target)

    observation = np.concatenate(
        [
            local_target_offset / np.float32(3.0),
            relative_target_rotation,
            state.linear_velocity_body,
            state.angular_velocity_body_xyz,
            np.array([np.clip(np.linalg.norm(local_target_offset) / 3.0, 0.0, 1.0)], dtype=np.float32),
        ],
        axis=0,
    )
    return observation.astype(np.float32, copy=False)


def build_pose16_rot6d_observation(
    state: VehicleState,
    target_position_world: Sequence[float],
    target_quaternion_world: Sequence[float] | None = None,
    *,
    position_scale: float = 3.0,
    linear_velocity_scale: Sequence[float] = (1.0, 1.0, 1.0),
    angular_velocity_scale: Sequence[float] = (1.0, 1.0, 1.0),
    velocity_clip: float = 2.0,
) -> np.ndarray:
    target_position = _vec3(target_position_world)
    local_target_offset = inverse_rotate_vector(
        state.orientation_world_body,
        target_position - state.position_world,
    )
    if target_quaternion_world is None:
        relative_target_rotation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    else:
        current = quat_normalize(state.orientation_world_body)
        target = quat_normalize(target_quaternion_world)
        inverse_current = np.array([-current[0], -current[1], -current[2], current[3]], dtype=np.float32)
        relative_target_rotation = quat_multiply(inverse_current, target)

    rotation_matrix = quat_to_matrix(relative_target_rotation)
    rotation_6d = np.array(
        [
            rotation_matrix[0, 0],
            rotation_matrix[1, 0],
            rotation_matrix[2, 0],
            rotation_matrix[0, 1],
            rotation_matrix[1, 1],
            rotation_matrix[2, 1],
        ],
        dtype=np.float32,
    )
    safe_position_scale = np.float32(max(abs(float(position_scale)), 1e-6))
    safe_linear_scale = np.maximum(_vec3(linear_velocity_scale, default=1.0), 1e-6)
    safe_angular_scale = np.maximum(_vec3(angular_velocity_scale, default=1.0), 1e-6)
    clip = np.float32(max(abs(float(velocity_clip)), 1e-6))
    local_linear_velocity = np.clip(state.linear_velocity_body / safe_linear_scale, -clip, clip)
    local_angular_velocity = np.clip(state.angular_velocity_body_xyz / safe_angular_scale, -clip, clip)

    observation = np.concatenate(
        [
            local_target_offset / safe_position_scale,
            rotation_6d,
            local_linear_velocity.astype(np.float32, copy=False),
            local_angular_velocity.astype(np.float32, copy=False),
            np.array([np.clip(np.linalg.norm(local_target_offset) / safe_position_scale, 0.0, 1.0)], dtype=np.float32),
        ],
        axis=0,
    )
    return observation.astype(np.float32, copy=False)


def build_velocity_normalized_observation(
    state: VehicleState,
    target_position_world: Sequence[float] | None = None,
    *,
    linear_velocity_scale: Sequence[float],
    angular_velocity_scale: Sequence[float],
    outer_loop_kp: Sequence[float] | None = None,
    max_body_velocity: Sequence[float] | None = None,
    desired_linear_velocity_body: Sequence[float] | None = None,
    desired_angular_velocity_ypr: Sequence[float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    linear_scale = np.maximum(_vec3(linear_velocity_scale, default=1.0), 1e-6)
    angular_scale = np.maximum(_vec3(angular_velocity_scale, default=1.0), 1e-6)
    max_velocity_values = (np.inf, np.inf, np.inf) if max_body_velocity is None else max_body_velocity
    max_velocity = np.maximum(_vec3(max_velocity_values), 0.0)

    if desired_linear_velocity_body is not None:
        desired_body_velocity = np.asarray(desired_linear_velocity_body, dtype=np.float32).reshape(-1)
        if desired_body_velocity.shape[0] < 3:
            desired_body_velocity = np.pad(desired_body_velocity, (0, 3 - desired_body_velocity.shape[0]), constant_values=0.0)
        elif desired_body_velocity.shape[0] > 3:
            desired_body_velocity = desired_body_velocity[:3]
        desired_body_velocity = np.clip(desired_body_velocity, -max_velocity, max_velocity).astype(np.float32, copy=False)
        error_body = np.zeros(3, dtype=np.float32)
    else:
        if target_position_world is None or outer_loop_kp is None:
            raise ValueError("target_position_world and outer_loop_kp are required for position-driven velocity observations")
        target_position = _vec3(target_position_world)
        kp = _vec3(outer_loop_kp)
        error_world = target_position - state.position_world
        error_body = inverse_rotate_vector(state.orientation_world_body, error_world)
        desired_body_velocity = np.clip(error_body * kp, -max_velocity, max_velocity).astype(np.float32)

    if desired_angular_velocity_ypr is not None:
        desired_angular_ypr = np.asarray(desired_angular_velocity_ypr, dtype=np.float32).reshape(-1)
        if desired_angular_ypr.shape[0] < 3:
            desired_angular_ypr = np.pad(desired_angular_ypr, (0, 3 - desired_angular_ypr.shape[0]), constant_values=0.0)
        elif desired_angular_ypr.shape[0] > 3:
            desired_angular_ypr = desired_angular_ypr[:3]
        desired_angular_ypr = np.clip(desired_angular_ypr, -angular_scale, angular_scale).astype(np.float32, copy=False)
    else:
        desired_angular_ypr = np.zeros(3, dtype=np.float32)

    observation = np.concatenate(
        [
            desired_body_velocity / linear_scale,
            desired_angular_ypr / angular_scale,
            state.linear_velocity_body / linear_scale,
            state.angular_velocity_body_ypr / angular_scale,
        ],
        axis=0,
    )
    return (
        observation.astype(np.float32, copy=False),
        error_body.astype(np.float32, copy=False),
        desired_body_velocity.astype(np.float32, copy=False),
    )
