"""Role-wise behavior cloning data preparation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datasets import transition_mask
from .schemas import MultiAgentDataset


@dataclass(frozen=True)
class RoleTransitions:
    """Flattened observation-action pairs for one role."""

    role_id: int
    obs: np.ndarray
    actions: np.ndarray


def split_role_transitions(dataset: MultiAgentDataset) -> dict[int, RoleTransitions]:
    """Split MARL demonstrations into role-wise supervised datasets."""
    role_ids = np.asarray(dataset.role_ids, dtype=np.int64)
    step_mask = transition_mask(dataset)
    result: dict[int, RoleTransitions] = {}
    for role_id in sorted(int(r) for r in np.unique(role_ids)):
        agent_indices = np.where(role_ids == role_id)[0]
        role_obs = dataset.obs[:, :-1, agent_indices, :]
        role_actions = dataset.actions[:, :, agent_indices, :]
        role_mask = np.broadcast_to(step_mask[:, :, None], role_obs.shape[:-1])
        obs = role_obs[role_mask].reshape(-1, dataset.obs_dim)
        actions = role_actions[role_mask].reshape(-1, dataset.action_dim)
        result[role_id] = RoleTransitions(role_id=role_id, obs=obs, actions=actions)
    return result
