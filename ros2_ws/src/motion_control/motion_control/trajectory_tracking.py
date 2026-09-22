"""Shared ROS-side contract for the Unity T2 trajectory tracking task.

Observation order is fixed at 30 values and mirrors
Assets/Scripts/RL/TrajectoryTrackingAgent.cs. Controller body axes are
X forward, Y up, Z left.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .math_utils import inverse_rotate_vector, quat_normalize
from .state_estimator import VehicleState

PREVIEW_COUNT = 4
OBSERVATION_SIZE = 30


@dataclass(frozen=True)
class TrajectoryPoint:
    time_sec: float
    position_world: np.ndarray
    orientation_world: np.ndarray
    linear_velocity_world: np.ndarray


def sample_trajectory(points: Sequence[TrajectoryPoint], elapsed_sec: float) -> TrajectoryPoint:
    if not points:
        raise ValueError("trajectory requires at least one point")
    ordered = tuple(sorted(points, key=lambda point: point.time_sec))
    if elapsed_sec <= ordered[0].time_sec:
        return ordered[0]
    if elapsed_sec >= ordered[-1].time_sec:
        return ordered[-1]
    for left, right in zip(ordered, ordered[1:]):
        if elapsed_sec <= right.time_sec:
            duration = max(right.time_sec - left.time_sec, 1e-6)
            alpha = float(np.clip((elapsed_sec - left.time_sec) / duration, 0.0, 1.0))
            position = (1.0 - alpha) * left.position_world + alpha * right.position_world
            velocity = (1.0 - alpha) * left.linear_velocity_world + alpha * right.linear_velocity_world
            quaternion = quat_normalize((1.0 - alpha) * left.orientation_world + alpha * right.orientation_world)
            return TrajectoryPoint(float(elapsed_sec), position.astype(np.float32), quaternion, velocity.astype(np.float32))
    return ordered[-1]


def build_trajectory30_observation(
    state: VehicleState,
    points: Sequence[TrajectoryPoint],
    elapsed_sec: float,
    *,
    preview_step_sec: float = 0.1,
    preview_offset_scale: float = 3.0,
    linear_velocity_scale: float = 1.0,
    angular_velocity_scale: float = 1.0,
    observation_clip: float = 2.0,
    trajectory_duration_sec: float | None = None,
) -> tuple[np.ndarray, TrajectoryPoint]:
    """Build the exact Unity T2 30D layout and return the current reference."""
    scale_position = max(abs(float(preview_offset_scale)), 1e-6)
    scale_linear = max(abs(float(linear_velocity_scale)), 1e-6)
    scale_angular = max(abs(float(angular_velocity_scale)), 1e-6)
    clip = max(abs(float(observation_clip)), 1e-6)
    preview = []
    for index in range(PREVIEW_COUNT):
        point = sample_trajectory(points, elapsed_sec + index * preview_step_sec)
        local_offset = inverse_rotate_vector(state.orientation_world_body, point.position_world - state.position_world)
        preview.append(np.clip(local_offset / scale_position, -clip, clip))
    current = sample_trajectory(points, elapsed_sec)
    desired_velocity_body = inverse_rotate_vector(state.orientation_world_body, current.linear_velocity_world)
    linear_velocity_body = np.clip(state.linear_velocity_body / scale_linear, -clip, clip)
    angular_velocity_body = np.clip(state.angular_velocity_body_xyz / scale_angular, -clip, clip)
    up_body = inverse_rotate_vector(state.orientation_world_body, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    tangent_yaw = float(np.arctan2(desired_velocity_body[2], desired_velocity_body[0]))
    duration = max(float(trajectory_duration_sec if trajectory_duration_sec is not None else points[-1].time_sec), 1e-6)
    progress = float(np.clip(elapsed_sec / duration, 0.0, 1.0))
    observation = np.concatenate(
        [
            *preview,
            np.clip(desired_velocity_body / scale_linear, -clip, clip),
            linear_velocity_body,
            angular_velocity_body,
            up_body,
            np.array([np.sin(tangent_yaw), np.cos(tangent_yaw)], dtype=np.float32),
            np.array([np.sin(2.0 * np.pi * progress), np.cos(2.0 * np.pi * progress), np.sin(4.0 * np.pi * progress), np.cos(4.0 * np.pi * progress)], dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    if observation.shape != (OBSERVATION_SIZE,):
        raise RuntimeError(f"trajectory30 observation shape is {observation.shape}, expected {(OBSERVATION_SIZE,)}")
    return observation, current
