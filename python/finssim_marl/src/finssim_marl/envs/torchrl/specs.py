"""Tensor specs for FinsSim MARL TorchRL environments."""

from __future__ import annotations

import torch
from torchrl.data import Bounded, Categorical, Composite, Unbounded

from . import keys


def build_observation_spec(
    *,
    n_agents: int,
    obs_dim: int,
    state_dim: int,
    device: torch.device | str | None = None,
) -> Composite:
    """Build the observation spec used by FinsSim multi-agent TensorDicts."""
    return Composite(
        {
            keys.AGENTS: Composite(
                {
                    keys.OBSERVATION: Unbounded(
                        shape=(n_agents, obs_dim),
                        dtype=torch.float32,
                        device=device,
                    ),
                    keys.ROLE_ID: Unbounded(
                        shape=(n_agents, 1),
                        dtype=torch.int64,
                        device=device,
                    ),
                },
                shape=(n_agents,),
                device=device,
            ),
            keys.STATE: Unbounded(
                shape=(state_dim,),
                dtype=torch.float32,
                device=device,
            ),
        },
        shape=(),
        device=device,
    )


def build_action_spec(
    *,
    n_agents: int,
    action_dim: int,
    device: torch.device | str | None = None,
) -> Composite:
    """Build the continuous action spec used by FinsSim MARL environments."""
    return Composite(
        {
            keys.AGENTS: Composite(
                {
                    keys.ACTION: Bounded(
                        low=-1.0,
                        high=1.0,
                        shape=(n_agents, action_dim),
                        dtype=torch.float32,
                        device=device,
                    )
                },
                shape=(n_agents,),
                device=device,
            )
        },
        shape=(),
        device=device,
    )


def build_reward_spec(
    *,
    n_agents: int,
    device: torch.device | str | None = None,
) -> Composite:
    """Build a per-agent reward spec.

    Existing Unity wrappers may expose a scalar team reward. The adapter expands
    that scalar to this per-agent shape unless ordered per-agent rewards are
    present in the step info.
    """
    return Composite(
        {
            keys.AGENTS: Composite(
                {
                    keys.REWARD: Unbounded(
                        shape=(n_agents, 1),
                        dtype=torch.float32,
                        device=device,
                    )
                },
                shape=(n_agents,),
                device=device,
            )
        },
        shape=(),
        device=device,
    )


def build_done_spec(device: torch.device | str | None = None) -> Composite:
    """Build shared done specs for TorchRL check_env_specs compatibility."""
    done_leaf = lambda: Categorical(n=2, shape=(1,), dtype=torch.bool, device=device)
    return Composite(
        {
            keys.DONE: done_leaf(),
            keys.TERMINATED: done_leaf(),
            keys.TRUNCATED: done_leaf(),
        },
        shape=(),
        device=device,
    )
