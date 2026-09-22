from __future__ import annotations

import numpy as np

from finssim_marl.algorithms.networks.actors import ROLE_MAPPING
from finssim_marl.algorithms.traditional_formation_baseline import (
    TraditionalFormationBaselineConfig,
    TraditionalFormationPIDWrenchBaseline,
)


ROLE_IDS = np.array(
    [ROLE_MAPPING["Herder"], ROLE_MAPPING["Netter"], ROLE_MAPPING["Netter"], ROLE_MAPPING["Prey"]],
    dtype=np.int64,
)


def _empty_obs() -> np.ndarray:
    return np.zeros((1, 4, 30), dtype=np.float32)


def _add_teammates(obs: np.ndarray, agent_index: int, first_role, first_position, second_role, second_position):
    obs[0, agent_index, 14:16] = first_role
    obs[0, agent_index, 16:19] = first_position
    obs[0, agent_index, 22:24] = second_role
    obs[0, agent_index, 24:27] = second_position


def test_formation_guidance_is_local_and_role_specific():
    baseline = TraditionalFormationPIDWrenchBaseline(
        n_agents=4,
        role_ids=ROLE_IDS,
        config=TraditionalFormationBaselineConfig(controller_backend="zero"),
    )
    obs = _empty_obs()

    # Herder sees the prey ahead and both netters centered around its body z axis.
    obs[0, 0, 6:9] = [10.0, 0.0, 0.0]
    _add_teammates(obs, 0, [0.0, 1.0], [0.0, 0.0, -3.0], [0.0, 1.0], [0.0, 0.0, 3.0])

    # Each netter sees Herder first and its partner second, matching the 30D contract.
    obs[0, 1, 6:9] = [10.0, 0.0, 0.0]
    _add_teammates(obs, 1, [1.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 6.0])
    obs[0, 2, 6:9] = [10.0, 0.0, 0.0]
    _add_teammates(obs, 2, [1.0, 0.0], [-4.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, -6.0])

    body_errors, yaw_errors = baseline._build_subgoals(obs, ROLE_IDS.reshape(1, -1))

    # The fixed 1.5 m lookahead limits are part of the shared 4D interface.
    np.testing.assert_allclose(body_errors[0, 0], [1.5, 0.0, 0.0])
    assert abs(float(yaw_errors[0, 0])) < 1e-5
    # Netters retain opposite geometric bearings, while the execution command
    # keeps lateral Fz at zero until each one has turned toward its point.
    assert yaw_errors[0, 1] * yaw_errors[0, 2] < 0.0
    np.testing.assert_allclose(body_errors[0, 1:, 2], 0.0)


def test_formation_baseline_turns_before_advancing_to_a_side_target():
    baseline = TraditionalFormationPIDWrenchBaseline(
        n_agents=4,
        role_ids=ROLE_IDS,
        config=TraditionalFormationBaselineConfig(controller_backend="zero"),
    )
    obs = _empty_obs()

    # The Herder's desired formation point is to its body-left.  It must send
    # yaw only until the nose is inside the heading gate, not side-slip with Fz.
    obs[0, 0, 6:9] = [0.0, 0.0, 10.0]
    _add_teammates(obs, 0, [0.0, 1.0], [1.0, 0.0, 7.0], [0.0, 1.0], [-1.0, 0.0, 7.0])

    body_errors, yaw_errors = baseline._build_subgoals(obs, ROLE_IDS.reshape(1, -1))

    assert float(yaw_errors[0, 0]) < -baseline.config.heading_gate_deg
    np.testing.assert_allclose(body_errors[0, 0, [0, 2]], [0.0, 0.0])


def test_formation_baseline_uses_pid_wrench_and_keeps_prey_zero():
    baseline = TraditionalFormationPIDWrenchBaseline(
        n_agents=4,
        role_ids=ROLE_IDS,
        config=TraditionalFormationBaselineConfig(),
    )
    obs = _empty_obs()
    for agent_index in range(3):
        obs[0, agent_index, 6:9] = [6.0, 0.0, 0.0]
    _add_teammates(obs, 0, [0.0, 1.0], [0.0, 0.0, -3.0], [0.0, 1.0], [0.0, 0.0, 3.0])
    _add_teammates(obs, 1, [1.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, 6.0])
    _add_teammates(obs, 2, [1.0, 0.0], [-3.0, 0.0, 0.0], [0.0, 1.0], [0.0, 0.0, -6.0])

    actions, _, _, _ = baseline.select_action(obs, ROLE_IDS.reshape(1, -1))

    assert baseline.controller.name == "body_pid_wrench"
    assert np.max(np.abs(actions)) <= 1.0
    assert np.any(np.abs(actions[0, :3]) > 1e-6)
    np.testing.assert_allclose(actions[0, 3], 0.0)
