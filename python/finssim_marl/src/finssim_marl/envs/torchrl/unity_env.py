"""TorchRL EnvBase wrapper for FinsSim MARL environments."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict
from torchrl.envs import EnvBase

from finssim_marl.envs.base import BaseEnvSpec
from finssim_marl.envs.unity.worker import EnvConfig, get_unity_env

from . import keys
from .specs import (
    build_action_spec,
    build_done_spec,
    build_observation_spec,
    build_reward_spec,
)


class FinsSimTorchRLEnv(EnvBase):
    """Wrap a FinsSim MARL environment as a TorchRL ``EnvBase``.

    The wrapped environment must follow :class:`finssim_marl.envs.base.BaseEnvSpec`.
    This adapter keeps the existing FinsSim array interface intact and only
    translates data to and from nested TensorDicts.
    """

    def __init__(
        self,
        env: BaseEnvSpec,
        *,
        device: torch.device | str | None = None,
        run_type_checks: bool = False,
    ):
        self._env = env
        self._last_seed: int | None = None
        device = torch.device(device or "cpu")
        super().__init__(
            device=device,
            batch_size=[],
            run_type_checks=run_type_checks,
        )
        self.observation_spec = build_observation_spec(
            n_agents=env.n_agents,
            obs_dim=env.obs_dim,
            state_dim=env.state_dim,
            device=device,
        )
        self.action_spec = build_action_spec(
            n_agents=env.n_agents,
            action_dim=env.action_dim,
            device=device,
        )
        self.reward_spec = build_reward_spec(n_agents=env.n_agents, device=device)
        self.done_spec = build_done_spec(device=device)

    @property
    def n_agents(self) -> int:
        return self._env.n_agents

    @property
    def agent_names(self) -> list[str]:
        return getattr(
            self._env,
            "agent_names",
            [f"agent_{idx}" for idx in range(self.n_agents)],
        )

    @property
    def group_map(self) -> dict[str, list[str]]:
        return {keys.AGENTS: self.agent_names}

    def _role_ids_tensor(self) -> torch.Tensor:
        role_ids = np.asarray(self._env.role_ids, dtype=np.int64).reshape(self.n_agents, 1)
        return torch.as_tensor(role_ids, dtype=torch.int64, device=self.device)

    def _obs_state_tensordict(
        self,
        *,
        obs: np.ndarray,
        state: np.ndarray,
        done: bool = False,
        terminated: bool | None = None,
        truncated: bool = False,
        reward: np.ndarray | float | None = None,
    ) -> TensorDict:
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        state_tensor = torch.as_tensor(state, dtype=torch.float32, device=self.device)

        agents_payload: dict[str, torch.Tensor] = {
            keys.OBSERVATION: obs_tensor,
            keys.ROLE_ID: self._role_ids_tensor(),
        }
        if reward is not None:
            agents_payload[keys.REWARD] = self._reward_tensor(reward)

        done_value = bool(done)
        terminated_value = done_value if terminated is None else bool(terminated)
        return TensorDict(
            {
                keys.AGENTS: TensorDict(agents_payload, batch_size=[self.n_agents]),
                keys.STATE: state_tensor,
                keys.DONE: torch.tensor([done_value], dtype=torch.bool, device=self.device),
                keys.TERMINATED: torch.tensor(
                    [terminated_value],
                    dtype=torch.bool,
                    device=self.device,
                ),
                keys.TRUNCATED: torch.tensor(
                    [bool(truncated)],
                    dtype=torch.bool,
                    device=self.device,
                ),
            },
            batch_size=[],
            device=self.device,
        )

    def _reward_tensor(self, reward: np.ndarray | float) -> torch.Tensor:
        reward_array = np.asarray(reward, dtype=np.float32)
        if reward_array.ndim == 0:
            reward_array = np.full((self.n_agents,), float(reward_array), dtype=np.float32)
        reward_array = reward_array.reshape(-1)
        if reward_array.shape[0] != self.n_agents:
            reward_array = np.resize(reward_array, self.n_agents).astype(np.float32)
        return torch.as_tensor(
            reward_array.reshape(self.n_agents, 1),
            dtype=torch.float32,
            device=self.device,
        )

    def _reset(self, tensordict: TensorDict | None = None) -> TensorDict:
        seed = self._last_seed
        if tensordict is not None and "seed" in tensordict.keys():
            seed = int(tensordict["seed"].item())
        obs = self._env.reset(seed=seed)
        state = self._env.get_state()
        return self._obs_state_tensordict(obs=obs, state=state)

    def _step(self, tensordict: TensorDict) -> TensorDict:
        action = tensordict.get(keys.AGENT_ACTION_KEY)
        action_np = action.detach().cpu().numpy()
        obs, reward, done, truncated, info = self._env.step(action_np)
        state = self._env.get_state()
        ordered_rewards = self._extract_ordered_rewards(info, fallback=reward)
        return self._obs_state_tensordict(
            obs=obs,
            state=state,
            reward=ordered_rewards,
            done=bool(done or truncated),
            terminated=bool(done),
            truncated=bool(truncated),
        )

    def _extract_ordered_rewards(self, info: dict[str, Any], fallback: float) -> np.ndarray | float:
        ordered = info.get("reward_all_agents_ordered") if isinstance(info, dict) else None
        if ordered is None:
            return fallback
        return np.asarray(ordered, dtype=np.float32)

    def _set_seed(self, seed: int | None) -> int | None:
        self._last_seed = None if seed is None else int(seed)
        return self._last_seed

    def close(self, *, raise_if_closed: bool = True):
        try:
            self._env.close()
        finally:
            return super().close(raise_if_closed=raise_if_closed)


def make_unity_torchrl_env(
    env_config: EnvConfig,
    *,
    seed: int = 0,
    device: torch.device | str | None = None,
    env_factory: Callable[[EnvConfig, int], BaseEnvSpec] = get_unity_env,
) -> FinsSimTorchRLEnv:
    """Create a TorchRL wrapper around a Unity-backed FinsSim MARL environment."""
    env = env_factory(env_config, seed)
    return FinsSimTorchRLEnv(env, device=device)
