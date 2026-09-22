from types import SimpleNamespace

import numpy as np

from mlagents_envs.base_env import ActionTuple

from finssim_marl.envs.unity.worker import FinsSimUnityParallelEnv


def test_continuous_action_row_keeps_fractional_commands():
    action = np.array([0.43, 0.43, -0.804, -0.804, 0.1, -0.2, 0.3, -0.4], dtype=np.float32)

    result = FinsSimUnityParallelEnv._continuous_action_row(action, expected_size=8)

    np.testing.assert_allclose(result, action)


def test_process_action_writes_the_complete_continuous_vector():
    agent = "Herder?team=0?agent_id=7"
    behavior = "Herder?team=0"
    submitted = np.array([0.43, 0.43, -0.804, -0.804, 0.1, -0.2, 0.3, -0.4], dtype=np.float32)
    env = FinsSimUnityParallelEnv.__new__(FinsSimUnityParallelEnv)
    env._env = SimpleNamespace(
        behavior_specs={behavior: SimpleNamespace(action_spec=SimpleNamespace(continuous_size=8, discrete_size=0))},
        close=lambda: None,
    )
    env._dones = {agent: False}
    env._agent_id_to_index = {agent: 0}
    env._current_action = {behavior: ActionTuple(np.zeros((1, 8), dtype=np.float32), None)}

    env._process_action(agent, submitted)

    np.testing.assert_allclose(env._current_action[behavior].continuous[0], submitted)
