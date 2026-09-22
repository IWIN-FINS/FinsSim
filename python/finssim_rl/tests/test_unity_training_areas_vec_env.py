from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from finssim_rl.training.unity_training_areas_vec_env import UnityTrainingAreasVecEnv


class _Steps:
    def __init__(self, agent_ids, obs, rewards, interrupted=None):
        self.agent_id = np.asarray(agent_ids, dtype=np.int64)
        self.obs = [np.asarray(obs, dtype=np.float32)]
        self.reward = np.asarray(rewards, dtype=np.float32)
        self.interrupted = np.asarray(interrupted or [False] * len(agent_ids), dtype=bool)

    def __len__(self):
        return len(self.agent_id)


class _FakeUnity:
    def __init__(self):
        action_spec = SimpleNamespace(is_continuous=lambda: True, continuous_size=2)
        self.behavior_specs = {
            "HoldForPosition": SimpleNamespace(
                action_spec=action_spec,
                observation_specs=[SimpleNamespace(shape=(3,))],
            )
        }
        self.phase = 0
        self.last_actions = None
        self.closed = False

    def reset(self):
        self.phase = 0

    def get_steps(self, _behavior):
        if self.phase == 0:
            return _Steps([20, 10], [[2, 0, 0], [1, 0, 0]], [0, 0]), _Steps([], [], [])
        return (
            _Steps([10, 20], [[11, 0, 0], [22, 0, 0]], [1.0, 2.0]),
            _Steps([20], [[99, 0, 0]], [7.0], [True]),
        )

    def set_actions(self, _behavior, action_tuple):
        self.last_actions = action_tuple.continuous.copy()

    def step(self):
        self.phase = 1

    def close(self):
        self.closed = True


class _DelayedReplicaFakeUnity(_FakeUnity):
    """Models a replicated scene whose clones request after the base agent."""

    def get_steps(self, _behavior):
        if self.phase == 0:
            return _Steps([10], [[1, 0, 0]], [0]), _Steps([], [], [])
        return _Steps([10, 20], [[1, 0, 0], [2, 0, 0]], [0, 0]), _Steps([], [], [])


def test_training_areas_vec_env_reorders_actions_and_preserves_terminal_obs():
    unity = _FakeUnity()
    env = UnityTrainingAreasVecEnv(unity, num_areas=2)

    np.testing.assert_allclose(env.reset(), [[2, 0, 0], [1, 0, 0]])
    obs, rewards, dones, infos = env.step(np.asarray([[2, 3], [4, 5]], dtype=np.float32))

    # The first action order is the reset decision order [20, 10].
    np.testing.assert_allclose(unity.last_actions, [[2, 3], [4, 5]])
    np.testing.assert_allclose(obs, [[22, 0, 0], [11, 0, 0]])
    np.testing.assert_allclose(rewards, [7, 1])
    np.testing.assert_array_equal(dones, [True, False])
    np.testing.assert_allclose(infos[0]["terminal_observation"], [99, 0, 0])
    assert infos[0]["TimeLimit.truncated"] is True

    env.step(np.asarray([[20, 30], [40, 50]], dtype=np.float32))
    # Unity reordered its decision ids to [10, 20], while SB3 slot order
    # remains [20, 10].
    np.testing.assert_allclose(unity.last_actions, [[40, 50], [20, 30]])

    env.close()
    assert unity.closed


def test_training_areas_vec_env_waits_for_delayed_replicas_during_reset():
    env = UnityTrainingAreasVecEnv(_DelayedReplicaFakeUnity(), num_areas=2)

    np.testing.assert_allclose(env.reset(), [[1, 0, 0], [2, 0, 0]])


def test_training_areas_vec_env_reports_area_count_mismatch_after_reset_warmup():
    with pytest.raises(RuntimeError, match="expected num_areas=3"):
        UnityTrainingAreasVecEnv(_FakeUnity(), num_areas=3, ready_timeout_seconds=0.001)
