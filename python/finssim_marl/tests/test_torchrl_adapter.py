from __future__ import annotations

import numpy as np
import torch
from torchrl.envs import check_env_specs

from finssim_marl.envs.base import BaseEnvSpec
from finssim_marl.envs.torchrl import FinsSimTorchRLEnv


class FakeMARLEnv(BaseEnvSpec):
    def __init__(self):
        self._step_count = 0
        self._role_ids = np.array([0, 1, 1, 2], dtype=np.int32)
        self.closed = False

    @property
    def n_agents(self) -> int:
        return 4

    @property
    def obs_dim(self) -> int:
        return 3

    @property
    def action_dim(self) -> int:
        return 2

    @property
    def state_dim(self) -> int:
        return 12

    @property
    def role_ids(self) -> np.ndarray:
        return self._role_ids

    @property
    def agent_names(self) -> list[str]:
        return ["herder_0", "netter_1", "netter_2", "prey_3"]

    @property
    def max_steps(self) -> int:
        return 5

    def reset(self, seed=None):
        self._step_count = 0
        return np.zeros((self.n_agents, self.obs_dim), dtype=np.float32)

    def step(self, actions):
        self._step_count += 1
        obs = np.full((self.n_agents, self.obs_dim), self._step_count, dtype=np.float32)
        reward = float(np.asarray(actions).sum())
        done = self._step_count >= 2
        truncated = False
        info = {"reward_all_agents_ordered": np.arange(self.n_agents, dtype=np.float32)}
        return obs, reward, done, truncated, info

    def get_state(self):
        return np.full((self.state_dim,), self._step_count, dtype=np.float32)

    def close(self):
        self.closed = True


def test_torchrl_adapter_specs_and_step():
    env = FinsSimTorchRLEnv(FakeMARLEnv())

    check_env_specs(env)
    reset_td = env.reset()
    assert reset_td["agents", "observation"].shape == torch.Size([4, 3])
    assert reset_td["agents", "role_id"].shape == torch.Size([4, 1])
    assert reset_td["state"].shape == torch.Size([12])

    step_td = env.rand_step()
    assert step_td["next", "agents", "reward"].shape == torch.Size([4, 1])
    assert step_td["next", "agents", "reward"].flatten().tolist() == [0.0, 1.0, 2.0, 3.0]
    assert env.group_map == {"agents": ["herder_0", "netter_1", "netter_2", "prey_3"]}

    wrapped = env._env
    env.close()
    assert wrapped.closed
