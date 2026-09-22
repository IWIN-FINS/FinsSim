"""Episode distance metrics for the DirectLocal14 OneChaseOne observation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


ONE_CHASE_ONE_DISTANCE_OBSERVATION_INDEX = 12
ONE_CHASE_ONE_OBSERVATION_DIM = 14


@dataclass(frozen=True)
class OneChaseOneEpisodeDistance:
    """Distance summary measured over one completed OneChaseOne episode."""

    minimum_m: float
    final_m: float


def _vector_observation(observation: Any, slot: int | None) -> np.ndarray | None:
    """Select a DirectLocal14 vector from ndarray, dict, or multi-observation data."""
    candidates: Sequence[Any]
    if isinstance(observation, Mapping):
        candidates = tuple(observation.values())
    elif isinstance(observation, (tuple, list)):
        candidates = observation
    else:
        candidates = (observation,)

    for candidate in candidates:
        array = np.asarray(candidate)
        if array.ndim == 0 or array.shape[-1] < ONE_CHASE_ONE_OBSERVATION_DIM:
            continue
        if slot is not None and array.ndim >= 2:
            if slot >= array.shape[0]:
                continue
            array = array[slot]
        if array.ndim != 1:
            continue
        return array
    return None


def distance_to_prey_from_observation(observation: Any, slot: int | None = None) -> float | None:
    """Return DirectLocal14's distance-to-prey field, or ``None`` when absent."""
    vector = _vector_observation(observation, slot)
    if vector is None:
        return None
    distance = float(vector[ONE_CHASE_ONE_DISTANCE_OBSERVATION_INDEX])
    return distance if np.isfinite(distance) and distance >= 0.0 else None


class OneChaseOneDistanceTracker:
    """Track per-slot min/final prey distance across auto-reset VecEnv episodes."""

    def __init__(self, num_envs: int) -> None:
        self.num_envs = max(1, int(num_envs))
        self._minimum_by_slot = np.full(self.num_envs, np.nan, dtype=np.float64)
        self.completed: list[OneChaseOneEpisodeDistance] = []

    def reset(self, observations: Any) -> None:
        self._minimum_by_slot.fill(np.nan)
        for slot in range(self.num_envs):
            self._set_episode_start(slot, observations)

    def record_transition(
        self,
        slot: int,
        next_observations: Any,
        done: bool,
        info: Mapping[str, Any] | None = None,
        previous_observations: Any | None = None,
    ) -> OneChaseOneEpisodeDistance | None:
        """Record one slot transition and return its metrics when it terminates."""
        slot = int(slot)
        if slot < 0 or slot >= self.num_envs:
            raise IndexError(f"slot {slot} is outside [0, {self.num_envs})")

        if not np.isfinite(self._minimum_by_slot[slot]) and previous_observations is not None:
            self._set_episode_start(slot, previous_observations)

        terminal_observation = (info or {}).get("terminal_observation") if done else None
        final_observation = terminal_observation if terminal_observation is not None else next_observations
        final_distance = distance_to_prey_from_observation(
            final_observation,
            None if terminal_observation is not None else slot,
        )
        if final_distance is not None:
            current_minimum = self._minimum_by_slot[slot]
            self._minimum_by_slot[slot] = (
                final_distance if not np.isfinite(current_minimum) else min(current_minimum, final_distance)
            )

        if not done:
            return None

        minimum = self._minimum_by_slot[slot]
        completed = None
        if final_distance is not None and np.isfinite(minimum):
            completed = OneChaseOneEpisodeDistance(float(minimum), float(final_distance))
            self.completed.append(completed)

        # VecEnv returns the next episode's observation in this slot after a
        # terminal transition. Seed its new running minimum independently.
        self._set_episode_start(slot, next_observations)
        return completed

    def summary(self, prefix: str = "eval") -> dict[str, float]:
        """Return aggregate TensorBoard scalars for all completed episodes."""
        if not self.completed:
            return {}
        minimums = np.asarray([episode.minimum_m for episode in self.completed], dtype=np.float64)
        finals = np.asarray([episode.final_m for episode in self.completed], dtype=np.float64)
        return {
            f"{prefix}/mean_min_distance_to_prey_m": float(np.mean(minimums)),
            f"{prefix}/mean_final_distance_to_prey_m": float(np.mean(finals)),
            f"{prefix}/best_min_distance_to_prey_m": float(np.min(minimums)),
            f"{prefix}/worst_final_distance_to_prey_m": float(np.max(finals)),
        }

    def _set_episode_start(self, slot: int, observations: Any) -> None:
        distance = distance_to_prey_from_observation(observations, slot)
        self._minimum_by_slot[slot] = np.nan if distance is None else distance


def is_one_chase_one_config(config: Any) -> bool:
    """Limit DirectLocal14 indexing to the OneChaseOne config family."""
    name = str(getattr(config, "name", "")).lower()
    return "1chase1" in name or "hierarchy_chase" in name or "traditional_chase" in name
