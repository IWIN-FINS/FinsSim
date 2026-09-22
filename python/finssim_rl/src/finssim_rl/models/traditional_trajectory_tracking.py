"""Static wrench-PD baseline for the T2 30D trajectory-tracking task.

The Unity task always consumes the canonical eight FinsROV thruster actions.
This controller works in the same Python-side physical wrench space as the T2
wrench6 policy, then uses the shared allocator for the final eight commands.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from finssim_rl.models.thrust_allocator import ThrustAllocator
from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig


TRAJECTORY_OBSERVATION_DIM = 30
THRUSTER_ACTION_DIM = 8

# Must match ppo_trajectory_tracking_wrench6.  T2 uses
# [Fx, Fy, Fz, Mx, My, Mz] = [forward, up, left, roll, yaw, pitch].
T2_ALLOCATOR_BODY_WRENCH_LIMITS: Tuple[float, float, float, float, float, float] = (
    19.528527,
    18.415027,
    22.501886,
    3.863995,
    3.079985,
    7.080833,
)


def as_trajectory_observation_batch(
    observation: Union[np.ndarray, Dict[str, np.ndarray]],
) -> tuple[np.ndarray, bool]:
    """Normalize the fixed Unity T2 30D observation into a batch."""
    if isinstance(observation, dict):
        observation = observation.get(
            "state",
            observation.get("observation", observation.get("obs", next(iter(observation.values())))),
        )
    obs = np.nan_to_num(np.asarray(observation, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    was_vector = obs.ndim == 1
    if was_vector:
        obs = obs.reshape(1, -1)
    elif obs.ndim > 2:
        obs = obs.reshape(-1, obs.shape[-1])
    if obs.ndim != 2 or obs.shape[1] != TRAJECTORY_OBSERVATION_DIM:
        raise ValueError(f"TrajectoryTracking baseline requires 30D observation, got {obs.shape}")
    return obs, was_vector


@dataclass
class TraditionalTrajectoryTrackingTrainingConfig(BaseTrainingConfig):
    """Fixed gains for body-frame preview tracking."""

    dt: float = 0.1
    preview_offset_scale_m: float = 3.0
    linear_velocity_scale_mps: float = 1.0
    angular_velocity_scale_radps: float = 1.0
    position_kp: Tuple[float, float, float] = (6.0, 6.0, 6.0)
    velocity_kd: Tuple[float, float, float] = (5.0, 5.0, 5.0)
    tilt_kp: Tuple[float, float] = (1.2, 1.2)
    tilt_kd: Tuple[float, float] = (0.45, 0.45)
    yaw_kp: float = 1.2
    yaw_rate_kd: float = 0.6
    wrench_limits: Tuple[float, float, float, float, float, float] = T2_ALLOCATOR_BODY_WRENCH_LIMITS
    allocator_allocation_mode: str = "physical_wrench_allocator"
    allocator_thruster_force_limit_positive: Tuple[float, ...] = (7.0,) * THRUSTER_ACTION_DIM
    allocator_thruster_force_limit_negative: Tuple[float, ...] = (7.0,) * THRUSTER_ACTION_DIM
    output_slew_rate_limit: float = 100.0


class TraditionalTrajectoryTrackingWrenchModel:
    """Body-frame preview PD controller allocated to Unity's 8D action."""

    def __init__(self, config: TraditionalTrajectoryTrackingTrainingConfig, device: str = "auto") -> None:
        self.config = config
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
        self.allocator = ThrustAllocator(
            device=self.device,
            allocation_mode=config.allocator_allocation_mode,
            physical_wrench_limits=config.wrench_limits,
            thruster_force_limit_positive=config.allocator_thruster_force_limit_positive,
            thruster_force_limit_negative=config.allocator_thruster_force_limit_negative,
            debug_print_interval=0,
        )
        self.previous_wrench = np.zeros((0, 6), dtype=np.float32)
        self.last_diagnostics: dict[str, np.ndarray | float] = {}

    def reset(self) -> None:
        self.previous_wrench = np.zeros((0, 6), dtype=np.float32)
        self.last_diagnostics = {}

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True):
        del deterministic
        if episode_start is not None and np.any(episode_start):
            self.reset()
        obs, was_vector = as_trajectory_observation_batch(observation)
        if self.previous_wrench.shape[0] != obs.shape[0]:
            self.previous_wrench = np.zeros((obs.shape[0], 6), dtype=np.float32)

        config = self.config
        position_error = obs[:, 0:3] * float(config.preview_offset_scale_m)
        desired_velocity = obs[:, 12:15] * float(config.linear_velocity_scale_mps)
        linear_velocity = obs[:, 15:18] * float(config.linear_velocity_scale_mps)
        angular_velocity = obs[:, 18:21] * float(config.angular_velocity_scale_radps)
        up_body = obs[:, 21:24]

        force = (
            position_error * np.asarray(config.position_kp, dtype=np.float32)
            + (desired_velocity - linear_velocity) * np.asarray(config.velocity_kd, dtype=np.float32)
        )

        # Desired world-up is [0, 1, 0] in controller_body. This produces
        # restoring roll/pitch moments for the observed local up direction.
        upright_error = np.cross(
            np.broadcast_to(np.array([0.0, 1.0, 0.0], dtype=np.float32), up_body.shape),
            up_body,
        )
        tangent_yaw_error = -np.arctan2(obs[:, 24], obs[:, 25])
        roll = float(config.tilt_kp[0]) * upright_error[:, 0] - float(config.tilt_kd[0]) * angular_velocity[:, 0]
        yaw = float(config.yaw_kp) * tangent_yaw_error - float(config.yaw_rate_kd) * angular_velocity[:, 1]
        pitch = float(config.tilt_kp[1]) * upright_error[:, 2] - float(config.tilt_kd[1]) * angular_velocity[:, 2]
        wrench = np.column_stack([force, roll, yaw, pitch]).astype(np.float32)
        limits = np.asarray(config.wrench_limits, dtype=np.float32).reshape(1, 6)
        wrench = np.clip(wrench, -limits, limits)

        max_delta = max(float(config.output_slew_rate_limit), 0.0) * max(float(config.dt), 1e-6)
        if max_delta > 0.0:
            wrench = np.clip(wrench, self.previous_wrench - max_delta, self.previous_wrench + max_delta)
        self.previous_wrench = wrench.copy()

        with torch.no_grad():
            action = self.allocator(torch.as_tensor(wrench, dtype=torch.float32, device=self.device)).detach().cpu().numpy()
        action = np.clip(action, -1.0, 1.0).astype(np.float32, copy=False)
        self.last_diagnostics = {
            "preview_offset_body_m": position_error[0].copy(),
            "desired_velocity_body_mps": desired_velocity[0].copy(),
            "linear_velocity_body_mps": linear_velocity[0].copy(),
            "tangent_yaw_error_deg": float(np.degrees(tangent_yaw_error[0])),
            "wrench": wrench[0].copy(),
            "action": action[0].copy(),
        }
        return (action[0] if was_vector else action), state

    def format_diagnostics(self) -> str:
        if not self.last_diagnostics:
            return "[TraditionalTrajectoryTracking] no action has been generated yet"
        return (
            "[TraditionalTrajectoryTracking] "
            f"preview={np.array2string(np.asarray(self.last_diagnostics['preview_offset_body_m']), precision=3)} "
            f"yaw={float(self.last_diagnostics['tangent_yaw_error_deg']):+.1f}deg "
            f"wrench={np.array2string(np.asarray(self.last_diagnostics['wrench']), precision=3)} "
            f"action_abs_mean={float(np.mean(np.abs(self.last_diagnostics['action']))):.3f}"
        )


@dataclass
class TraditionalTrajectoryTrackingConfig(BaseConfig):
    model_type: str = "TRADITIONAL_TRAJECTORY_TRACKING_WRENCH_PD"
    requires_checkpoint: bool = False
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: TraditionalTrajectoryTrackingTrainingConfig = field(default_factory=TraditionalTrajectoryTrackingTrainingConfig)

    def create_model(self, env, args, load_checkpoint: Optional[str] = None):
        del env, load_checkpoint
        return TraditionalTrajectoryTrackingWrenchModel(self.training_config, device=args.device)

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        del model_path, env
        return TraditionalTrajectoryTrackingWrenchModel(self.training_config, device=device)


def get_traditional_trajectory_tracking_configs() -> Dict[str, TraditionalTrajectoryTrackingConfig]:
    return {
        "traditional_trajectory_tracking_wrench_pd": TraditionalTrajectoryTrackingConfig(
            name="traditional_trajectory_tracking_wrench_pd",
            description="T2 trajectory preview PD -> physical FinsROV wrench allocator -> Unity 8 thrusters",
            env_config=BaseEnvironmentConfig(
                env_path="../../artifacts/unity_builds/rl/linux/TrajectoryTracking_Fossen_Parallel_30s_Server/TrajectoryTracking.x86_64",
                num_envs=1,
                eval_num_envs=1,
                env_base_port=36105,
                timeout_wait=600,
                no_graphics=False,
                time_scale=1.0,
                parallel_mode="multi_area",
            ),
            training_config=TraditionalTrajectoryTrackingTrainingConfig(),
        )
    }


__all__ = [
    "T2_ALLOCATOR_BODY_WRENCH_LIMITS",
    "THRUSTER_ACTION_DIM",
    "TRAJECTORY_OBSERVATION_DIM",
    "TraditionalTrajectoryTrackingConfig",
    "TraditionalTrajectoryTrackingTrainingConfig",
    "TraditionalTrajectoryTrackingWrenchModel",
    "as_trajectory_observation_batch",
    "get_traditional_trajectory_tracking_configs",
]
