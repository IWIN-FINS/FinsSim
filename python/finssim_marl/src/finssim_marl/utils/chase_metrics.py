"""Episode metrics derived from the 3Chase1 Unity observation contract."""

from __future__ import annotations

import numpy as np


CHASER_PREY_RELATIVE_POSITION = slice(6, 9)
PREY_ROLE_ID = 2


def nearest_chaser_prey_distance_m(observations, role_ids) -> np.ndarray:
    """Return the nearest chaser-to-prey distance for every batch element.

    The 3Chase1 chaser actor observation exposes the Unity body-frame vector
    from that chaser to Prey at indices 6:9. Its norm is frame invariant, so
    it is the physical world-space separation in metres.
    """
    obs = np.asarray(observations, dtype=np.float32)
    if obs.ndim == 2:
        obs = obs[None, ...]
    if obs.ndim != 3 or obs.shape[-1] < CHASER_PREY_RELATIVE_POSITION.stop:
        raise ValueError(
            "Expected observations shaped [batch, agents, >=9], got "
            f"{obs.shape}."
        )

    roles = np.asarray(role_ids, dtype=np.int64)
    if roles.ndim == 1:
        if roles.shape[0] != obs.shape[1]:
            raise ValueError(f"role_ids shape {roles.shape} does not match agents={obs.shape[1]}.")
        roles = np.broadcast_to(roles, obs.shape[:2])
    if roles.shape != obs.shape[:2]:
        raise ValueError(
            f"role_ids shape {roles.shape} does not match observation batch/agents {obs.shape[:2]}."
        )

    # TriNetCapture has no Prey policy. Its 33D contract appends Goal-relative
    # position at 30:33, so target-to-goal is available in every ROV body frame.
    if obs.shape[-1] >= 33 and not np.any(roles == PREY_ROLE_ID):
        from finssim_marl.utils.trinet_metrics import target_goal_distance_m
        return target_goal_distance_m(obs)

    chaser_distances = np.linalg.norm(obs[..., CHASER_PREY_RELATIVE_POSITION], axis=-1)
    chaser_distances = np.where(roles != PREY_ROLE_ID, chaser_distances, np.inf)
    nearest = np.min(chaser_distances, axis=1)
    if not np.all(np.isfinite(nearest)):
        raise ValueError("3Chase1 metrics require at least one finite chaser observation per batch item.")
    return nearest.astype(np.float32, copy=False)


def summarize_episode_distances(
    initial_distance_m,
    min_distance_m,
    final_distance_m,
    *,
    prefix: str = "",
) -> dict[str, float]:
    """Aggregate episode distance traces into stable TensorBoard scalars."""
    initial = np.asarray(initial_distance_m, dtype=np.float32)
    minimum = np.asarray(min_distance_m, dtype=np.float32)
    final = np.asarray(final_distance_m, dtype=np.float32)
    if initial.size == 0 or minimum.size == 0 or final.size == 0:
        return {}
    if not (initial.shape == minimum.shape == final.shape):
        raise ValueError(
            "Initial, minimum, and final distance arrays must have identical shapes, got "
            f"{initial.shape}, {minimum.shape}, {final.shape}."
        )
    return {
        f"{prefix}initial_nearest_chaser_distance_to_prey_m": float(initial.mean()),
        f"{prefix}min_nearest_chaser_distance_to_prey_m": float(minimum.mean()),
        f"{prefix}final_nearest_chaser_distance_to_prey_m": float(final.mean()),
        f"{prefix}nearest_chaser_distance_reduction_m": float((initial - final).mean()),
    }
