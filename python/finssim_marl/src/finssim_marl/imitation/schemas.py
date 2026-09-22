"""Data schemas for multi-agent imitation learning datasets."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class MultiAgentDataset:
    """Fixed-shape episode batch for MARL imitation learning.

    Shapes:
        obs: [E, T + 1, N, obs_dim]
        states: [E, T + 1, state_dim]
        actions: [E, T, N, action_dim]
        rewards: [E, T] or [E, T, N]
        dones: [E, T]
        truncateds: [E, T]
        valid_steps: [E, T], true for real transitions and false for padding
        role_ids: [N]
    """

    obs: np.ndarray
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    dones: np.ndarray
    truncateds: np.ndarray
    role_ids: np.ndarray
    valid_steps: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def num_episodes(self) -> int:
        return int(self.obs.shape[0])

    @property
    def horizon(self) -> int:
        return int(self.actions.shape[1])

    @property
    def num_agents(self) -> int:
        return int(self.actions.shape[2])

    @property
    def obs_dim(self) -> int:
        return int(self.obs.shape[-1])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[-1])
