from __future__ import annotations

import numpy as np
import torch

from finssim_rl.models.thrust_allocator import ThrustAllocator
from finssim_rl.models.traditional_chase_baselines import TraditionalChaseBaselineModel, TraditionalChaseBaselineTrainingConfig
from finssim_rl.training.config import get_config
from finssim_rl.training.hierarchy_chase_config import (
    HIERARCHY_ACTOR_OBS_DIM,
    HierarchyChaseModel,
    _normalize_obs_array,
)


def _unity_world_to_controller_body(
    world_delta: np.ndarray,
    *,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            float(np.dot(world_delta, right)),
            float(np.dot(world_delta, up)),
            float(np.dot(world_delta, forward)),
        ],
        dtype=np.float32,
    )


def _controller_body_to_unity_world(
    body_delta: np.ndarray,
    *,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    return (
        right * float(body_delta[0])
        + up * float(body_delta[1])
        + forward * float(body_delta[2])
    ).astype(np.float32)


def test_unity_visualizer_body_to_world_is_inverse_of_controller_body_frame() -> None:
    unity_basis = {
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "forward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    body_axes = (
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        np.array([0.0, 1.0, 0.0], dtype=np.float32),
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        np.array([0.25, -0.5, 0.75], dtype=np.float32),
    )

    for body_delta in body_axes:
        world_delta = _controller_body_to_unity_world(body_delta, **unity_basis)
        roundtrip = _unity_world_to_controller_body(world_delta, **unity_basis)
        np.testing.assert_allclose(roundtrip, body_delta, atol=1e-6)


def test_unity_visualizer_left_positive_maps_to_reference_forward() -> None:
    unity_basis = {
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "forward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    world_delta = _controller_body_to_unity_world(
        np.array([0.0, 0.0, 1.0], dtype=np.float32),
        **unity_basis,
    )

    np.testing.assert_allclose(world_delta, [0.0, 0.0, 1.0], atol=1e-6)


def test_unity_local_x_is_controller_body_forward() -> None:
    unity_basis = {
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "forward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    body_delta = _unity_world_to_controller_body(
        np.array([1.0, 0.0, 0.0], dtype=np.float32),
        **unity_basis,
    )

    np.testing.assert_allclose(body_delta, [1.0, 0.0, 0.0], atol=1e-6)


def test_hierarchy_action_forward_is_direct_pid_body_error() -> None:
    model = object.__new__(HierarchyChaseModel)
    model.target_body_delta_limits = (1.5, 0.5, 1.5)

    obs = np.zeros((1, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)

    body_error = model._build_low_level_pid_inputs(
        np.array([[1.0, 0.0, 0.0]], dtype=np.float32),
        obs,
    )

    np.testing.assert_allclose(body_error, [[1.5, 0.0, 0.0]], atol=1e-6)


def test_hierarchy_action_left_keeps_controller_body_left_sign() -> None:
    model = object.__new__(HierarchyChaseModel)
    model.target_body_delta_limits = (1.5, 0.5, 1.5)

    obs = np.zeros((1, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)

    body_error = model._build_low_level_pid_inputs(
        np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
        obs,
    )

    np.testing.assert_allclose(body_error, [[0.0, 0.0, 1.5]], atol=1e-6)


def test_hierarchy_action_is_clipped_before_scaling() -> None:
    model = object.__new__(HierarchyChaseModel)
    model.target_body_delta_limits = (1.5, 0.5, 1.5)

    obs = np.zeros((1, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)

    body_error = model._build_low_level_pid_inputs(
        np.array([[2.0, -2.0, 0.5]], dtype=np.float32),
        obs,
    )

    np.testing.assert_allclose(body_error, [[1.5, -0.5, 0.75]], atol=1e-6)


def test_hierarchy_action_batch_repeats_single_policy_action_for_obs_batch() -> None:
    model = object.__new__(HierarchyChaseModel)
    model.target_body_delta_limits = (1.5, 0.5, 1.5)

    obs = np.zeros((3, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)

    body_error = model._build_low_level_pid_inputs(
        np.array([[0.25, 0.5, -0.25]], dtype=np.float32),
        obs,
    )

    expected = np.repeat([[0.375, 0.25, -0.375]], 3, axis=0)
    np.testing.assert_allclose(body_error, expected, atol=1e-6)


def test_hierarchy_obs_fallback_is_14d() -> None:
    obs = _normalize_obs_array(None, default_num_envs=2)

    assert obs.shape == (2, HIERARCHY_ACTOR_OBS_DIM)


def test_thrust_allocator_axis_signs_match_controller_body_force_axes() -> None:
    allocator = ThrustAllocator(
        device="cpu",
        control_axis_ranges=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        allocation_mode="empirical_thruster_mixer",
        debug_print_interval=0,
    )

    thrust = allocator(torch.eye(6, dtype=torch.float32)[:3]).detach().cpu().numpy()

    np.testing.assert_array_equal(
        np.sign(thrust[0]),
        [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0],
    )
    np.testing.assert_array_equal(
        np.sign(thrust[1]),
        [1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0],
    )
    np.testing.assert_array_equal(
        np.sign(thrust[2]),
        [0.0, 0.0, 0.0, 0.0, -1.0, 1.0, 1.0, -1.0],
    )


def test_chase_baseline_yaw_command_is_negative_for_a_prey_on_the_left() -> None:
    model = TraditionalChaseBaselineModel("wrench", TraditionalChaseBaselineTrainingConfig(), device="cpu")
    obs = np.zeros(14, dtype=np.float32)
    obs[6:9] = [0.0, 0.0, 2.0]

    model.predict(obs, deterministic=True)

    assert float(model.last_diagnostics["yaw_error_deg"]) == -90.0
    wrench = np.asarray(model.last_diagnostics["wrench"])
    assert wrench[0] == 0.0
    assert wrench[2] == 0.0
    assert wrench[4] < 0.0
    assert model.last_diagnostics["forward_enabled"] == 0.0


def test_chase_baseline_wrench_advances_only_after_heading_alignment() -> None:
    model = TraditionalChaseBaselineModel("wrench", TraditionalChaseBaselineTrainingConfig(), device="cpu")
    obs = np.zeros(14, dtype=np.float32)
    obs[6:9] = [2.0, 0.0, 0.1]

    model.predict(obs, deterministic=True)

    wrench = np.asarray(model.last_diagnostics["wrench"])
    assert wrench[0] > 0.0
    assert wrench[2] == 0.0
    assert model.last_diagnostics["forward_enabled"] == 1.0


def test_physical_wrench_allocator_chase_baseline_uses_the_simulation_force_envelope() -> None:
    config = get_config("traditional_chase_wrench_physical_wrench_allocator_1chase1")
    assert config.training_config.allocator_allocation_mode == "physical_wrench_allocator"
    np.testing.assert_allclose(
        config.training_config.wrench_force_limits,
        (19.528527, 22.501886, 18.415027),
        atol=1e-6,
    )
    np.testing.assert_allclose(config.training_config.wrench_yaw_limit, 7.080833, atol=1e-6)

    model = TraditionalChaseBaselineModel("wrench", config.training_config, device="cpu")
    target = np.array([[3.0, 0.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    action = model._allocate_wrench(target)
    force_n = action * 7.0

    np.testing.assert_allclose(
        (model.allocator.B.detach().cpu().numpy() @ force_n[0]),
        target[0],
        atol=2e-4,
    )


def test_chase_baseline_direct_thruster_forward_mixer_matches_canonical_order() -> None:
    model = TraditionalChaseBaselineModel("thruster", TraditionalChaseBaselineTrainingConfig(), device="cpu")
    obs = np.zeros(14, dtype=np.float32)
    obs[6:9] = [4.0, 0.0, 0.0]

    action, _ = model.predict(obs, deterministic=True)

    np.testing.assert_allclose(action, [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, -1.0, -1.0], atol=1e-6)


def test_chase_baseline_direct_thruster_turns_in_place_for_a_prey_on_the_left() -> None:
    model = TraditionalChaseBaselineModel("thruster", TraditionalChaseBaselineTrainingConfig(), device="cpu")
    obs = np.zeros(14, dtype=np.float32)
    obs[6:9] = [0.0, 0.0, 2.0]

    action, _ = model.predict(obs, deterministic=True)

    # +z is controller-body left.  A left-side prey requires a negative yaw
    # command, represented by equal horizontal thruster commands.  Equal
    # commands produce yaw only: no surge and no sway.
    np.testing.assert_allclose(action, [0.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0, -1.0], atol=1e-6)
    wrench = np.asarray(model.last_diagnostics["wrench"])
    assert wrench[0] == 0.0
    assert wrench[2] == 0.0
    assert wrench[4] < 0.0
    assert model.last_diagnostics["forward_enabled"] == 0.0


def test_chase_baseline_direct_thruster_uses_surge_not_sway_after_alignment() -> None:
    model = TraditionalChaseBaselineModel("thruster", TraditionalChaseBaselineTrainingConfig(), device="cpu")
    obs = np.zeros(14, dtype=np.float32)
    obs[6:9] = [4.0, 0.0, 0.1]

    action, _ = model.predict(obs, deterministic=True)

    # A residual left-side error only corrects yaw.  The diagonal pairs stay
    # equal, proving that the mixer has no Fz/sway component.
    assert model.last_diagnostics["forward_enabled"] == 1.0
    np.testing.assert_allclose(action[4], action[5], atol=1e-6)
    np.testing.assert_allclose(action[6], action[7], atol=1e-6)
    wrench = np.asarray(model.last_diagnostics["wrench"])
    assert wrench[0] > 0.0
    assert wrench[2] == 0.0
    assert wrench[4] < 0.0
    assert np.all(np.abs(action) <= 1.0)


def test_chase_baseline_paths_return_finite_eight_thruster_actions() -> None:
    obs = np.zeros((2, 14), dtype=np.float32)
    obs[:, 6:9] = [[2.0, 0.5, -1.0], [1.0, -0.5, 1.0]]
    for kind in ("pid", "wrench", "thruster"):
        model = TraditionalChaseBaselineModel(kind, TraditionalChaseBaselineTrainingConfig(), device="cpu")
        action, _ = model.predict(obs, deterministic=True)
        assert np.asarray(action).shape == (2, 8)
        assert np.all(np.isfinite(action))
        assert np.all(np.abs(action) <= 1.0)
