"""PPO configs for 6-DOF wrench-policy training.

The Unity training scenes currently consume 8 direct thruster commands.  This
module keeps that simulator interface intact while making the learned policy
head explicitly 6D.  During rollout the normalized policy action is converted
to a body wrench and allocated to 8 thrusters only for the simulator step.

Deployment should use the policy output, not ``model.predict()``, when the real
vehicle will perform allocation downstream.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Tuple, Union

import numpy as np

from finssim_rl.models.ppo_control import PPOEnvironmentConfig
from finssim_rl.models.ppo_control_v2 import (
    DEFAULT_VIRTUAL_CONTROL_LIMITS,
    POLICY_AXIS_ORDER_ALLOCATOR_BODY,
    PPOV2Config,
    PPOV2TrainingConfig,
    PPOVirtualControlModel,
    VIRTUAL_CONTROL_DIM,
    VIRTUAL_CONTROL_NAMES,
    virtual_controls_to_allocator_tau,
)
from finssim_rl.training.utils import linear_schedule

WRENCH_ORDER: Tuple[str, ...] = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
"""Physical body-wrench order after scaling and axis reordering."""

POLICY_ACTION_ORDER: Tuple[str, ...] = VIRTUAL_CONTROL_NAMES
"""Normalized policy output order: surge, sway, heave, roll, pitch, yaw."""

# FinsROV_Fossen, with the current canonical geometry and +/-7 N per
# thruster.  These are pure-axis capabilities, in policy action order.
SIM_PHYSICAL_WRENCH_LIMITS: Tuple[float, float, float, float, float, float] = (
    19.528527,
    18.415027,
    22.501886,
    3.863995,
    3.079985,
    7.080833,
)

# Actuation-interface ablation: retain 10% of the calibrated translational
# pure-axis limits, retain 5% yaw authority, and explicitly disable roll/pitch
# policy outputs. The order is [surge, sway, heave, roll, pitch, yaw].
SIM_REDUCED_WRENCH_LIMITS: Tuple[float, float, float, float, float, float] = (
    1.9528527,
    1.8415027,
    2.2501886,
    0.0,
    0.0,
    0.35404165,
)


def normalized_policy_actions_to_wrench(
    policy_actions: np.ndarray,
    control_limits: Tuple[float, float, float, float, float, float] = DEFAULT_VIRTUAL_CONTROL_LIMITS,
) -> np.ndarray:
    """Convert normalized 6D policy actions to physical wrench values.

    Input order is ``[surge, sway, heave, roll, pitch, yaw]`` in ``[-1, 1]``.
    Output order is ``[Fx, Fy, Fz, Mx, My, Mz]`` for downstream allocation.
    """

    return virtual_controls_to_allocator_tau(policy_actions, control_limits)


@dataclass
class PPOWrenchTrainingConfig(PPOV2TrainingConfig):
    """PPO wrench training config.

    ``virtual_control_limits`` maps the normalized policy action to physical
    wrench units before the simulation-only thrust allocator:

    - policy order: [surge, sway, heave, roll, pitch, yaw]
    - physical wrench order: [Fx, Fy, Fz, Mx, My, Mz]
    """

    learning_rate: Union[float, Callable[[float], float]] = 3e-4
    batch_size: int = 1024
    n_steps: int = 512
    n_epochs: int = 10
    gamma: float = 0.99
    gae_lambda: float = 0.95
    total_timesteps: int = 5_000_000
    checkpoint_freq: int = 50_000
    eval_freq: int = 5_000
    virtual_control_limits: Tuple[float, float, float, float, float, float] = DEFAULT_VIRTUAL_CONTROL_LIMITS


@dataclass
class PPOWrenchConfig(PPOV2Config):
    """Explicit config type for 6D wrench policies."""

    model_type: str = "PPO_WRENCH"
    env_config: PPOEnvironmentConfig = field(default_factory=PPOEnvironmentConfig)
    training_config: PPOWrenchTrainingConfig = field(default_factory=PPOWrenchTrainingConfig)

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        return PPOVirtualControlModel.load(model_path, env=env, device=device)


def get_ppo_wrench_control_configs() -> Dict[str, PPOWrenchConfig]:
    """Return single-agent PPO configs that learn 6D body wrench commands."""

    return {
        "ppo_wrench_for_pose_empirical_thruster_mixer": PPOWrenchConfig(
            name="ppo_wrench_for_pose_empirical_thruster_mixer",
            model_type="PPO_WRENCH",
            description=(
                "PPO pose control with the historical empirical 6D wrench mixer; "
                "simulator rollout allocates to 8 thrusters as an environment adapter."
            ),
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/ControlForPosition_IncrementalReward_10Hz_Server/ControlForPosition.x86_64",
                num_envs=32,
                env_base_port=5005,
                timeout_wait=180,
                no_graphics=True,
            ),
            training_config=PPOWrenchTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
                virtual_control_limits=DEFAULT_VIRTUAL_CONTROL_LIMITS,
            ),
        ),
        "ppo_wrench_for_pose_physical_wrench_allocator": PPOWrenchConfig(
            name="ppo_wrench_for_pose_physical_wrench_allocator",
            model_type="PPO_WRENCH",
            description=(
                "PPO pose control with physical FinsROV B allocation: normalized "
                "6D action -> physical pure-axis wrench limit -> bounded 8-thruster force."
            ),
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/ControlForPosition_Fossen_10Hz_Server/ControlForPosition.x86_64",
                num_envs=32,
                env_base_port=5005,
                timeout_wait=180,
                no_graphics=True,
            ),
            training_config=PPOWrenchTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
                virtual_control_limits=SIM_PHYSICAL_WRENCH_LIMITS,
                allocator_allocation_mode="physical_wrench_allocator",
                allocator_thruster_force_limit_positive=(7.0,) * 8,
                allocator_thruster_force_limit_negative=(7.0,) * 8,
            ),
        ),
        "ppo_wrench_for_pose_physical_wrench_allocator_reduced": PPOWrenchConfig(
            name="ppo_wrench_for_pose_physical_wrench_allocator_reduced",
            model_type="PPO_WRENCH",
            description=(
                "PPO pose control with the physical FinsROV allocator and a "
                "restricted wrench envelope: 0.1 translational, 0 roll/pitch, "
                "and 0.05 yaw authority relative to the simulation baseline."
            ),
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/HoldForPosition_Fossen_Parallel_30s_NormalizedMaxForce_NoDR_WrenchScaled_Server/HoldForPosition_Fossen_WrenchScaled.x86_64",
                num_envs=32,
                env_base_port=5005,
                timeout_wait=180,
                no_graphics=True,
            ),
            training_config=PPOWrenchTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
                virtual_control_limits=SIM_REDUCED_WRENCH_LIMITS,
                allocator_allocation_mode="physical_wrench_allocator",
                allocator_thruster_force_limit_positive=(7.0,) * 8,
                allocator_thruster_force_limit_negative=(7.0,) * 8,
            ),
        ),
        "ppo_trajectory_tracking_wrench6": PPOWrenchConfig(
            name="ppo_trajectory_tracking_wrench6",
            model_type="PPO_WRENCH",
            description=(
                "T2 trajectory tracking: 6D allocator-body policy action -> "
                "Python physical allocator -> Unity canonical 8-thruster action."
            ),
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/TrajectoryTracking_Fossen_Parallel_30s_Server/TrajectoryTracking.x86_64",
                num_envs=32,
                env_base_port=5005,
                timeout_wait=300,
                no_graphics=True,
            ),
            training_config=PPOWrenchTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
                # T2 order is [Fx,Fy,Fz,Mx,My,Mz], unlike the legacy wrench policy.
                virtual_control_limits=SIM_PHYSICAL_WRENCH_LIMITS,
                policy_axis_order=POLICY_AXIS_ORDER_ALLOCATOR_BODY,
                allocator_allocation_mode="physical_wrench_allocator",
                allocator_thruster_force_limit_positive=(7.0,) * 8,
                allocator_thruster_force_limit_negative=(7.0,) * 8,
            ),
        ),
    }


__all__ = [
    "POLICY_ACTION_ORDER",
    "PPOWrenchConfig",
    "PPOWrenchTrainingConfig",
    "SIM_PHYSICAL_WRENCH_LIMITS",
    "SIM_REDUCED_WRENCH_LIMITS",
    "WRENCH_ORDER",
    "get_ppo_wrench_control_configs",
    "normalized_policy_actions_to_wrench",
]
