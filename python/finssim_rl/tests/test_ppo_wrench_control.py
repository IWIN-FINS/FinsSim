from __future__ import annotations

import numpy as np

from finssim_rl.models.ppo_wrench_control import (
    POLICY_ACTION_ORDER,
    SIM_PHYSICAL_WRENCH_LIMITS,
    SIM_REDUCED_WRENCH_LIMITS,
    WRENCH_ORDER,
    normalized_policy_actions_to_wrench,
)
from finssim_rl.models.ppo_control_v2 import (
    POLICY_AXIS_ORDER_ALLOCATOR_BODY,
    policy_actions_to_allocator_tau,
)
from finssim_rl.training.config import get_config, list_configs


def test_ppo_wrench_config_is_registered():
    config = get_config("ppo_wrench_for_pose_empirical_thruster_mixer")

    assert "ppo_wrench_for_pose_empirical_thruster_mixer" in list_configs()
    assert config.model_type == "PPO_WRENCH"
    assert config.training_config.virtual_control_limits == (20.0, 20.0, 20.0, 0.8, 0.8, 0.8)


def test_pose_v2_config_uses_real_vehicle_scale_limits():
    config = get_config("ppo_control_v2_for_pose")

    assert "ppo_control_v2_for_pose" in list_configs()
    assert config.model_type == "PPO_VIRTUAL_CONTROL"
    assert config.training_config.virtual_control_limits == (20.0, 20.0, 20.0, 0.8, 0.8, 0.8)
    assert config.training_config.allocator_control_axis_ranges == (20.0, 20.0, 20.0, 0.8, 0.8, 0.8)


def test_physical_wrench_allocator_config_uses_documented_simulation_limits():
    config = get_config("ppo_wrench_for_pose_physical_wrench_allocator")

    assert config.training_config.virtual_control_limits == SIM_PHYSICAL_WRENCH_LIMITS
    assert config.training_config.allocator_allocation_mode == "physical_wrench_allocator"
    assert config.training_config.allocator_thruster_force_limit_positive == (7.0,) * 8


def test_reduced_wrench_config_disables_roll_and_pitch_policy_outputs():
    config = get_config("ppo_wrench_for_pose_physical_wrench_allocator_reduced")

    assert config.training_config.virtual_control_limits == SIM_REDUCED_WRENCH_LIMITS
    assert config.training_config.virtual_control_limits[3:5] == (0.0, 0.0)
    assert config.training_config.allocator_allocation_mode == "physical_wrench_allocator"


def test_trajectory_wrench6_config_keeps_unity_at_eight_thruster_actions():
    config = get_config("ppo_trajectory_tracking_wrench6")

    assert config.model_type == "PPO_WRENCH"
    assert config.training_config.policy_axis_order == POLICY_AXIS_ORDER_ALLOCATOR_BODY
    assert config.training_config.virtual_control_limits == SIM_PHYSICAL_WRENCH_LIMITS
    assert config.training_config.allocator_allocation_mode == "physical_wrench_allocator"
    assert config.training_config.allocator_thruster_force_limit_positive == (7.0,) * 8


def test_allocator_body_policy_order_maps_directly_to_allocator_wrench():
    actions = np.array([[1.0, 0.5, -0.25, 0.1, -0.2, 0.3]], dtype=np.float32)
    limits = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

    wrench = policy_actions_to_allocator_tau(
        actions,
        limits,
        POLICY_AXIS_ORDER_ALLOCATOR_BODY,
    )

    # T2 policy action is already [Fx, Fy, Fz, Mx, My, Mz].
    np.testing.assert_allclose(
        wrench,
        np.array([[10.0, 10.0, -7.5, 4.0, -10.0, 18.0]], dtype=np.float32),
    )


def test_normalized_policy_actions_to_wrench_axis_order():
    actions = np.array([[1.0, 0.5, -0.25, 0.1, -0.2, 0.3]], dtype=np.float32)
    limits = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

    wrench = normalized_policy_actions_to_wrench(actions, limits)

    assert POLICY_ACTION_ORDER == ("surge", "sway", "heave", "roll", "pitch", "yaw")
    assert WRENCH_ORDER == ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
    np.testing.assert_allclose(
        wrench,
        np.array([[10.0, -7.5, 10.0, 4.0, 18.0, -10.0]], dtype=np.float32),
    )


def test_normalized_policy_actions_to_wrench_clips_actions():
    actions = np.array([[2.0, -2.0, 0.0, 0.0, 0.0, 0.0]], dtype=np.float32)
    limits = (10.0, 20.0, 30.0, 40.0, 50.0, 60.0)

    wrench = normalized_policy_actions_to_wrench(actions, limits)

    np.testing.assert_allclose(
        wrench,
        np.array([[10.0, 0.0, -20.0, 0.0, 0.0, 0.0]], dtype=np.float32),
    )
