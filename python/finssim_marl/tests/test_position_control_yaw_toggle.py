import copy

import numpy as np
import torch

from finssim_marl.algorithms.mappo_multihead_position_controller import (
    MAPPOMultiHeadPositionControlAlgorithm,
)
from finssim_marl.algorithms.networks.actors import ROLE_MAPPING
from finssim_marl.algorithms.position_control_backends import (
    BodyFramePIDWrenchControllerBackend,
    BodyFramePIDWrenchControllerConfig,
)
from finssim_marl.training.config import get_config


def _build_obs(batch_size: int, n_agents: int) -> np.ndarray:
    return np.zeros((batch_size, n_agents, 30), dtype=np.float32)


def _role_ids() -> np.ndarray:
    return np.array(
        [
            ROLE_MAPPING["Herder"],
            ROLE_MAPPING["Netter"],
            ROLE_MAPPING["Netter"],
            ROLE_MAPPING["Prey"],
        ],
        dtype=np.int64,
    )


def test_position_control_yaw_toggle_uses_only_the_30d_actor_observation():
    config = copy.deepcopy(get_config("chasing_3_chase_1_position_control_test"))
    config.enable_yaw_control = False
    config.controller_backend = "zero"
    config.device = "cpu"
    train_config = config.create_train_config()
    train_config.device = "cpu"
    mappo_config = config.create_mappo_config(train_config)

    assert mappo_config.enable_yaw_control is False
    assert mappo_config.policy_action_dim == 3
    assert mappo_config.rollout_action_dim == 3
    assert mappo_config.chaser_team_obs_dim == 30

    role_ids = _role_ids()
    algorithm = MAPPOMultiHeadPositionControlAlgorithm(
        obs_dim=30,
        action_dim=8,
        state_dim=0,
        n_agents=4,
        role_ids=role_ids,
        config=mappo_config,
    )

    obs = _build_obs(batch_size=2, n_agents=4)
    batched_role_ids = np.broadcast_to(role_ids.reshape(1, -1), (2, 4)).copy()
    env_actions, log_probs, chosen_heads, extra = algorithm.select_action(
        obs,
        batched_role_ids,
        deterministic=True,
    )

    assert env_actions.shape == (2, 4, 8)
    assert log_probs.shape == (2, 4)
    assert chosen_heads.shape == (2, 4)
    assert extra["buffer_actions"].shape == (2, 4, 3)
    assert algorithm._last_target_yaw_abs_deg == 0.0
    np.testing.assert_array_equal(env_actions, np.zeros_like(env_actions))


def test_body_pid_wrench_backend_preserves_body_wrench_axis_signs():
    backend = BodyFramePIDWrenchControllerBackend(
        BodyFramePIDWrenchControllerConfig(device="cpu")
    )

    surge_action = backend.act(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        np.array([0.0], dtype=np.float32),
    )
    surge_wrench = backend.model.controller.thrust_allocator.B @ torch.from_numpy(
        surge_action[0] * 7.0
    )
    assert surge_wrench[0] > 0.0
    torch.testing.assert_close(surge_wrench[1:], torch.zeros(5), atol=2e-4, rtol=0.0)

    yaw_backend = BodyFramePIDWrenchControllerBackend(
        BodyFramePIDWrenchControllerConfig(device="cpu")
    )
    yaw_action = yaw_backend.act(
        np.zeros((1, 3), dtype=np.float32),
        np.array([10.0], dtype=np.float32),
    )
    yaw_wrench = yaw_backend.model.controller.thrust_allocator.B @ torch.from_numpy(
        yaw_action[0] * 7.0
    )
    assert yaw_wrench[4] > 0.0
    torch.testing.assert_close(
        torch.cat([yaw_wrench[:4], yaw_wrench[5:]]),
        torch.zeros(5),
        atol=2e-4,
        rtol=0.0,
    )


def test_body_pid_wrench_backend_zero_error_outputs_zero_thrusters():
    backend = BodyFramePIDWrenchControllerBackend(
        BodyFramePIDWrenchControllerConfig(device="cpu")
    )
    action = backend.act(
        np.zeros((3, 3), dtype=np.float32),
        np.zeros(3, dtype=np.float32),
    )
    np.testing.assert_array_equal(action, np.zeros((3, 8), dtype=np.float32))


def test_body_pid_wrench_backend_resets_each_parallel_area_independently():
    backend = BodyFramePIDWrenchControllerBackend(
        BodyFramePIDWrenchControllerConfig(device="cpu")
    )
    backend.act(
        np.zeros((2, 3, 3), dtype=np.float32),
        np.full((2, 3), 10.0, dtype=np.float32),
    )
    assert np.any(backend.model._yaw_prev_error_np[:3] != 0.0)
    assert np.any(backend.model._yaw_prev_error_np[3:] != 0.0)

    backend.reset(np.array([True, False]))

    np.testing.assert_array_equal(backend.model._yaw_prev_error_np[:3], np.zeros(3))
    assert np.any(backend.model._yaw_prev_error_np[3:] != 0.0)


def test_position_control_4d_action_reaches_the_body_pid_wrench_backend():
    config = copy.deepcopy(get_config("chasing_3_chase_1_position_control_test"))
    config.device = "cpu"
    train_config = config.create_train_config()
    train_config.device = "cpu"
    mappo_config = config.create_mappo_config(train_config)
    role_ids = _role_ids()
    algorithm = MAPPOMultiHeadPositionControlAlgorithm(
        obs_dim=30,
        action_dim=8,
        state_dim=0,
        n_agents=4,
        role_ids=role_ids,
        config=mappo_config,
    )

    env_actions, _, _, extra = algorithm.select_action(
        _build_obs(batch_size=1, n_agents=4),
        role_ids.reshape(1, -1),
        deterministic=True,
    )

    assert mappo_config.action_interface == "position_controller"
    assert mappo_config.control_interface_version == "body_subgoal_pid_physical_wrench_v1"
    assert extra["buffer_actions"].shape == (1, 4, 4)
    assert env_actions.shape == (1, 4, 8)
    np.testing.assert_array_equal(env_actions[:, 3], np.zeros((1, 8), dtype=np.float32))
