"""Dataset collection helpers for FinsSim MARL imitation learning."""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol

import numpy as np

from finssim_marl.envs.base import BaseEnvSpec

from .schemas import MultiAgentDataset


class MultiAgentPolicy(Protocol):
    """Minimal callable policy protocol used by dataset collection."""

    def __call__(self, obs: np.ndarray, state: np.ndarray, role_ids: np.ndarray) -> np.ndarray:
        """Return actions with shape [N, action_dim]."""


def collect_episodes(
    env: BaseEnvSpec,
    policy: MultiAgentPolicy | Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray],
    *,
    num_episodes: int,
    horizon: int | None = None,
    seed: int | None = None,
) -> MultiAgentDataset:
    """Collect fixed-horizon episodes from a FinsSim MARL environment."""
    max_horizon = int(horizon or env.max_steps)
    if num_episodes <= 0:
        raise ValueError(f"num_episodes must be positive, got {num_episodes}")
    if max_horizon <= 0:
        raise ValueError(f"horizon must be positive, got {max_horizon}")
    obs_batch = []
    state_batch = []
    action_batch = []
    reward_batch = []
    done_batch = []
    truncated_batch = []
    valid_step_batch = []

    for episode_idx in range(num_episodes):
        obs = env.reset(seed=None if seed is None else seed + episode_idx)
        state = env.get_state()
        episode_obs = [obs]
        episode_states = [state]
        episode_actions = []
        episode_rewards = []
        episode_dones = []
        episode_truncateds = []

        for _step in range(max_horizon):
            action = np.asarray(policy(obs, state, env.role_ids), dtype=np.float32)
            next_obs, reward, done, truncated, _info = env.step(action)
            next_state = env.get_state()
            episode_actions.append(action)
            episode_rewards.append(float(reward))
            episode_dones.append(bool(done))
            episode_truncateds.append(bool(truncated))
            episode_obs.append(next_obs)
            episode_states.append(next_state)
            obs, state = next_obs, next_state
            if done or truncated:
                break

        obs_batch.append(np.asarray(episode_obs, dtype=np.float32))
        state_batch.append(np.asarray(episode_states, dtype=np.float32))
        action_batch.append(np.asarray(episode_actions, dtype=np.float32))
        reward_batch.append(np.asarray(episode_rewards, dtype=np.float32))
        done_batch.append(np.asarray(episode_dones, dtype=np.bool_))
        truncated_batch.append(np.asarray(episode_truncateds, dtype=np.bool_))
        valid_step_batch.append(np.ones((len(episode_actions),), dtype=np.bool_))

    return _pad_episode_batch(
        obs_batch=obs_batch,
        state_batch=state_batch,
        action_batch=action_batch,
        reward_batch=reward_batch,
        done_batch=done_batch,
        truncated_batch=truncated_batch,
        valid_step_batch=valid_step_batch,
        role_ids=np.asarray(env.role_ids, dtype=np.int64),
    )


def _pad_episode_batch(
    *,
    obs_batch: list[np.ndarray],
    state_batch: list[np.ndarray],
    action_batch: list[np.ndarray],
    reward_batch: list[np.ndarray],
    done_batch: list[np.ndarray],
    truncated_batch: list[np.ndarray],
    valid_step_batch: list[np.ndarray],
    role_ids: np.ndarray,
) -> MultiAgentDataset:
    horizon = max(actions.shape[0] for actions in action_batch)
    episodes = len(action_batch)
    n_agents = role_ids.shape[0]
    obs_dim = obs_batch[0].shape[-1]
    state_dim = state_batch[0].shape[-1]
    action_dim = action_batch[0].shape[-1]

    obs = np.zeros((episodes, horizon + 1, n_agents, obs_dim), dtype=np.float32)
    states = np.zeros((episodes, horizon + 1, state_dim), dtype=np.float32)
    actions = np.zeros((episodes, horizon, n_agents, action_dim), dtype=np.float32)
    rewards = np.zeros((episodes, horizon), dtype=np.float32)
    dones = np.zeros((episodes, horizon), dtype=np.bool_)
    truncateds = np.zeros((episodes, horizon), dtype=np.bool_)
    valid_steps = np.zeros((episodes, horizon), dtype=np.bool_)

    for idx in range(episodes):
        t = action_batch[idx].shape[0]
        obs[idx, : t + 1] = obs_batch[idx]
        states[idx, : t + 1] = state_batch[idx]
        actions[idx, :t] = action_batch[idx]
        rewards[idx, :t] = reward_batch[idx]
        dones[idx, :t] = done_batch[idx]
        truncateds[idx, :t] = truncated_batch[idx]
        valid_steps[idx, :t] = valid_step_batch[idx]
        if t < horizon:
            obs[idx, t + 1 :] = obs_batch[idx][-1]
            states[idx, t + 1 :] = state_batch[idx][-1]

    return MultiAgentDataset(
        obs=obs,
        states=states,
        actions=actions,
        rewards=rewards,
        dones=dones,
        truncateds=truncateds,
        role_ids=role_ids,
        valid_steps=valid_steps,
    )
