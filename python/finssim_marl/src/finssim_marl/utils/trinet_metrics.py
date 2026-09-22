"""Metrics derived from TriNetCapture's 34D homogeneous actor observation contract."""

from __future__ import annotations

import numpy as np


TARGET_RELATIVE_POSITION = slice(6, 9)
GOAL_RELATIVE_POSITION = slice(26, 29)


def target_goal_distance_m(observations) -> np.ndarray:
    """Return Target-to-Goal separation using two vectors in one body frame."""
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 2:
        obs = obs[None, ...]
    if obs.ndim != 3 or obs.shape[-1] < GOAL_RELATIVE_POSITION.stop:
        raise ValueError(f"Expected [batch, agents, >=29] TriNet observations, got {obs.shape}.")
    distances = np.linalg.norm(
        obs[..., GOAL_RELATIVE_POSITION] - obs[..., TARGET_RELATIVE_POSITION], axis=-1
    )
    # Every ROV observes the same physical quantity in its own body frame.
    return distances.mean(axis=1).astype(np.float32, copy=False)


def summarize_trinet_episode_metrics(initial, minimum, final, *, prefix: str = "") -> dict[str, float]:
    initial = np.asarray(initial, dtype=np.float32)
    minimum = np.asarray(minimum, dtype=np.float32)
    final = np.asarray(final, dtype=np.float32)
    if initial.size == 0:
        return {}
    return {
        f"{prefix}initial_target_to_goal_distance_m": float(initial.mean()),
        f"{prefix}min_target_to_goal_distance_m": float(minimum.mean()),
        f"{prefix}final_target_to_goal_distance_m": float(final.mean()),
        f"{prefix}target_to_goal_distance_reduction_m": float((initial - final).mean()),
        # This is a geometric entry diagnostic, not task success: Unity alone
        # validates net containment and the one-second hold condition.
        f"{prefix}target_inside_goal_proxy_rate": float(np.mean(final <= 0.30)),
    }
