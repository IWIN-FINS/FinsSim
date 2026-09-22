"""Observation helpers shared by the HoldForPosition static baselines."""

from __future__ import annotations

from typing import Dict, Union

import numpy as np


HOLD_OBSERVATION_DIM = 16


def as_hold_observation_batch(observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> tuple[np.ndarray, bool]:
    if isinstance(observation, dict):
        observation = observation.get("state", observation.get("observation", observation.get("obs", next(iter(observation.values())))))
    obs = np.nan_to_num(np.asarray(observation, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    was_vector = obs.ndim == 1
    if was_vector:
        obs = obs.reshape(1, -1)
    elif obs.ndim > 2:
        obs = obs.reshape(-1, obs.shape[-1])
    if obs.ndim != 2 or obs.shape[1] != HOLD_OBSERVATION_DIM:
        raise ValueError(f"HoldForPosition baseline requires 16D observation, got {obs.shape}")
    return obs, was_vector


def relative_rotation_vector(rotation6d: np.ndarray) -> np.ndarray:
    """Convert Unity's first-two-column rotation6D to a body-frame rotvec."""
    values = np.asarray(rotation6d, dtype=np.float32).copy()
    first = values[:, 0:3]
    second = values[:, 3:6]
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-6)
    second -= np.sum(first * second, axis=1, keepdims=True) * first
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-6)
    third = np.cross(first, second)
    matrix = np.stack([first, second, third], axis=2)
    skew = np.stack(
        [matrix[:, 2, 1] - matrix[:, 1, 2], matrix[:, 0, 2] - matrix[:, 2, 0], matrix[:, 1, 0] - matrix[:, 0, 1]],
        axis=1,
    )
    sin_angle = 0.5 * np.linalg.norm(skew, axis=1)
    cos_angle = np.clip((np.trace(matrix, axis1=1, axis2=2) - 1.0) * 0.5, -1.0, 1.0)
    angle = np.arctan2(sin_angle, cos_angle)
    axis = skew / np.maximum(2.0 * sin_angle[:, None], 1e-6)
    return (axis * angle[:, None]).astype(np.float32, copy=False)
