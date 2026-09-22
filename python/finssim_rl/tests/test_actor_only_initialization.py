from types import SimpleNamespace

import numpy as np
from gymnasium import Env, spaces
from stable_baselines3.common.vec_env import DummyVecEnv

from finssim_rl.models.ppo_control_v2 import PPOV2Config


class _ToyWrenchEnv(Env):
    observation_space = spaces.Box(-np.inf, np.inf, shape=(16,), dtype=np.float32)
    action_space = spaces.Box(-1.0, 1.0, shape=(8,), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(16, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(16, dtype=np.float32), 0.0, False, False, {}


def _make_model(seed: int):
    env = DummyVecEnv([_ToyWrenchEnv])
    model = PPOV2Config(name="actor-only-test").create_model(
        env=env,
        args=SimpleNamespace(device="cpu", tensorboard_dir=None, seed=seed, resume=False),
    )
    return model, env


def test_actor_only_initialization_keeps_new_critic_and_optimizer(tmp_path) -> None:
    source, source_env = _make_model(seed=11)
    checkpoint_path = tmp_path / "source.zip"
    source.save(checkpoint_path)

    target, target_env = _make_model(seed=22)
    critic_before = {
        name: value.detach().clone()
        for name, value in target.policy.state_dict().items()
        if name.startswith("mlp_extractor.value_net.") or name.startswith("value_net.")
    }

    copied = target.initialize_actor_from_checkpoint(checkpoint_path)
    source_state = source.policy.state_dict()
    target_state = target.policy.state_dict()

    assert copied
    assert target.num_timesteps == 0
    assert len(target.policy.optimizer.state) == 0
    assert all(np.array_equal(target_state[name].cpu(), source_state[name].cpu()) for name in copied)
    assert all(np.array_equal(target_state[name].cpu(), value.cpu()) for name, value in critic_before.items())
    assert any(
        not np.array_equal(target_state[name].cpu(), source_state[name].cpu())
        for name in critic_before
    )

    source_env.close()
    target_env.close()
