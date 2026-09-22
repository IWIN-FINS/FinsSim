"""Traditional fixed wrench-PD baseline for the 16D HoldForPosition task."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple

import numpy as np
import torch

from finssim_rl.models.hold_position_baseline_common import as_hold_observation_batch, relative_rotation_vector
from finssim_rl.models.thrust_allocator import ThrustAllocator
from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig


SIM_BODY_WRENCH_LIMITS: Tuple[float, float, float, float, float, float] = (
    19.528527,
    22.501886,
    18.415027,
    3.863995,
    7.080833,
    3.079985,
)
"""FinsROV_Fossen pure-axis simulation capability in [Fx, Fy, Fz, Mx, My, Mz]."""


@dataclass
class TraditionalHoldPositionWrenchTrainingConfig(BaseTrainingConfig):
    dt: float = 0.1
    position_observation_scale_m: float = 3.0
    linear_velocity_observation_scale_mps: float = 1.0
    angular_velocity_observation_scale_radps: float = 1.0
    position_kp: Tuple[float, float, float] = (8.0, 8.0, 8.0)
    position_kd: Tuple[float, float, float] = (4.0, 4.0, 4.0)
    attitude_kp: Tuple[float, float, float] = (0.35, 0.35, 0.35)
    attitude_kd: Tuple[float, float, float] = (0.12, 0.12, 0.12)
    wrench_limits: Tuple[float, float, float, float, float, float] = (20.0, 20.0, 20.0, 0.8, 0.8, 0.8)
    allocator_control_axis_ranges: Tuple[float, ...] = (20.0, 20.0, 20.0, 0.8, 0.8, 0.8)
    allocator_allocation_mode: str = "empirical_thruster_mixer"
    allocator_physical_wrench_limits: Tuple[float, ...] = ()
    allocator_thruster_force_limit_positive: Tuple[float, ...] = (7.0,) * 8
    allocator_thruster_force_limit_negative: Tuple[float, ...] = (7.0,) * 8
    output_slew_rate_limit: float = 100.0


class TraditionalHoldPositionWrenchModel:
    """Generate physical 6D wrench directly, then allocate to eight thrusters."""

    def __init__(self, config: TraditionalHoldPositionWrenchTrainingConfig, device: str = "auto") -> None:
        self.config = config
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.allocator = ThrustAllocator(
            device=self.device,
            control_axis_ranges=config.allocator_control_axis_ranges,
            allocation_mode=config.allocator_allocation_mode,
            physical_wrench_limits=config.allocator_physical_wrench_limits or config.wrench_limits,
            thruster_force_limit_positive=config.allocator_thruster_force_limit_positive,
            thruster_force_limit_negative=config.allocator_thruster_force_limit_negative,
            debug_print_interval=0,
        )
        self.previous_wrench = np.zeros((0, 6), dtype=np.float32)
        self.last_diagnostics: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.previous_wrench = np.zeros((0, 6), dtype=np.float32)
        self.last_diagnostics = {}

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True):
        del deterministic
        if episode_start is not None and np.any(episode_start):
            self.reset()
        obs, was_vector = as_hold_observation_batch(observation)
        if self.previous_wrench.shape[0] != obs.shape[0]:
            self.previous_wrench = np.zeros((obs.shape[0], 6), dtype=np.float32)
        config = self.config
        position_error = obs[:, 0:3] * float(config.position_observation_scale_m)
        attitude_error = relative_rotation_vector(obs[:, 3:9])
        linear_velocity = obs[:, 9:12] * float(config.linear_velocity_observation_scale_mps)
        angular_velocity = obs[:, 12:15] * float(config.angular_velocity_observation_scale_radps)
        force = position_error * np.asarray(config.position_kp) - linear_velocity * np.asarray(config.position_kd)
        moment = attitude_error * np.asarray(config.attitude_kp) - angular_velocity * np.asarray(config.attitude_kd)
        wrench = np.column_stack([force, moment]).astype(np.float32)
        wrench = np.clip(wrench, -np.asarray(config.wrench_limits), np.asarray(config.wrench_limits))
        max_delta = max(float(config.output_slew_rate_limit), 0.0) * max(float(config.dt), 1e-6)
        if max_delta > 0.0:
            wrench = np.clip(wrench, self.previous_wrench - max_delta, self.previous_wrench + max_delta)
        self.previous_wrench = wrench.copy()
        with torch.no_grad():
            action = self.allocator(torch.as_tensor(wrench, dtype=torch.float32, device=self.device)).cpu().numpy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
        self.last_diagnostics = {"wrench": wrench[0].copy(), "action": action[0].copy()}
        return (action[0] if was_vector else action), state


@dataclass
class TraditionalHoldPositionWrenchConfig(BaseConfig):
    model_type: str = "TRADITIONAL_HOLD_POSITION_WRENCH"
    requires_checkpoint: bool = False
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: TraditionalHoldPositionWrenchTrainingConfig = field(default_factory=TraditionalHoldPositionWrenchTrainingConfig)

    def create_model(self, env, args, load_checkpoint: Optional[str] = None):
        del env, load_checkpoint
        return TraditionalHoldPositionWrenchModel(self.training_config, device=args.device)

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        del model_path, env
        return TraditionalHoldPositionWrenchModel(self.training_config, device=device)


def get_traditional_hold_position_wrench_configs() -> Dict[str, TraditionalHoldPositionWrenchConfig]:
    env = BaseEnvironmentConfig(
        env_path="../../artifacts/unity_builds/rl/linux/HoldForPosition_Fossen_Parallel_15s_10Hz_Server/HoldForPosition_Parallel_15s.x86_64",
        num_envs=1, eval_num_envs=1, env_base_port=47125, timeout_wait=600,
        no_graphics=False, time_scale=1.0, parallel_mode="multi_area",
    )
    empirical = TraditionalHoldPositionWrenchTrainingConfig()
    physical = replace(
        empirical,
        wrench_limits=SIM_BODY_WRENCH_LIMITS,
        allocator_allocation_mode="physical_wrench_allocator",
        allocator_physical_wrench_limits=SIM_BODY_WRENCH_LIMITS,
    )
    return {
        "traditional_hold_position_wrench_empirical_thruster_mixer": TraditionalHoldPositionWrenchConfig(
            name="traditional_hold_position_wrench_empirical_thruster_mixer",
            description="Traditional fixed wrench-PD -> historical empirical FinsROV mixer for HoldForPosition",
            env_config=replace(env),
            training_config=empirical,
        ),
        "traditional_hold_position_wrench_physical_wrench_allocator": TraditionalHoldPositionWrenchConfig(
            name="traditional_hold_position_wrench_physical_wrench_allocator",
            description="Traditional fixed wrench-PD -> bounded physical FinsROV wrench allocator for HoldForPosition",
            env_config=replace(env, env_base_port=47127),
            training_config=physical,
        ),
    }
