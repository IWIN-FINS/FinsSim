"""Original traditional PID -> 8-thruster baseline adapted to HoldForPosition."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import numpy as np

from finssim_rl.models.hold_position_baseline_common import as_hold_observation_batch, relative_rotation_vector
from finssim_rl.models.pid_controller import PIDController
from finssim_rl.models.traditional_position_pid import TraditionalPositionPIDModel
from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig


@dataclass
class TraditionalHoldPositionPidTrainingConfig(BaseTrainingConfig):
    dt: float = 0.1
    pid_params: Tuple[float, ...] = (5.0, 0.1, 0.5, 5.0, 0.3, 0.5, 5.0, 0.3, 0.5)
    yaw_pid_params: Tuple[float, float, float] = (1e-3, 0.0, 0.0)
    enable_yaw_control: bool = True
    output_range: Tuple[float, float] = (-250.0, 250.0)
    surge_output_limit: float = 120.0
    sway_output_limit: float = 120.0
    yaw_deadband_deg: float = 0.5
    yaw_output_limit: float = 40.0
    integral_state_limit: float = 100.0
    yaw_integral_state_limit: float = 60.0
    derivative_filter_alpha: float = 0.0183156
    output_slew_rate_limit: float = 500.0
    allocator_control_linear_range: float = 10.0
    allocator_control_axis_ranges: Tuple[float, ...] = ()
    allocator_allocation_mode: str = "matrix"
    integral_decay: float = 0.95


class TraditionalHoldPositionPidModel:
    """Decode 16D state, then use the original traditional PID -> 8D path."""

    def __init__(self, config: TraditionalHoldPositionPidTrainingConfig, device: str = "auto") -> None:
        self.config = config
        controller = PIDController(
            action_dim=8, device=device, dt=config.dt, integral_decay=config.integral_decay,
            output_range=config.output_range, yaw_deadband_deg=config.yaw_deadband_deg,
            yaw_output_limit=config.yaw_output_limit, yaw_integral_state_limit=config.yaw_integral_state_limit,
            surge_output_limit=config.surge_output_limit, sway_output_limit=config.sway_output_limit,
            integral_state_limit=config.integral_state_limit, derivative_filter_alpha=config.derivative_filter_alpha,
            output_slew_rate_limit=config.output_slew_rate_limit,
            allocator_control_linear_range=config.allocator_control_linear_range,
            allocator_control_axis_ranges=config.allocator_control_axis_ranges or None,
            allocator_allocation_mode=config.allocator_allocation_mode, debug_print_interval=0,
        )
        self.controller = TraditionalPositionPIDModel(
            controller=controller, pid_params=config.pid_params, yaw_pid_params=config.yaw_pid_params,
            enable_yaw_control=config.enable_yaw_control, action_dim=8, enable_logging=False,
        )
        self.last_diagnostics: dict[str, np.ndarray] = {}

    def reset(self) -> None:
        self.controller.reset()
        self.last_diagnostics = {}

    def predict(self, observation, state=None, episode_start=None, deterministic: bool = True):
        del deterministic
        if episode_start is not None and np.any(episode_start):
            self.reset()
        obs, was_vector = as_hold_observation_batch(observation)
        position_error = obs[:, 0:3] * 3.0
        rotvec = relative_rotation_vector(obs[:, 3:9])
        yaw_error_deg = np.degrees(rotvec[:, 1]).astype(np.float32)
        action, _ = self.controller.predict_from_body_position_error(
            position_error, yaw_error_deg=yaw_error_deg, deterministic=True,
        )
        action = np.asarray(action, dtype=np.float32).reshape(-1, 8)
        self.last_diagnostics = {"position_error_body_m": position_error[0].copy(), "yaw_error_deg": np.array([yaw_error_deg[0]], dtype=np.float32), "action": action[0].copy()}
        return (action[0] if was_vector else action), state


@dataclass
class TraditionalHoldPositionPidConfig(BaseConfig):
    model_type: str = "TRADITIONAL_HOLD_POSITION_PID"
    requires_checkpoint: bool = False
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: TraditionalHoldPositionPidTrainingConfig = field(default_factory=TraditionalHoldPositionPidTrainingConfig)

    def create_model(self, env, args, load_checkpoint: Optional[str] = None):
        del env, load_checkpoint
        return TraditionalHoldPositionPidModel(self.training_config, device=args.device)

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        del model_path, env
        return TraditionalHoldPositionPidModel(self.training_config, device=device)


def get_traditional_hold_position_pid_configs() -> Dict[str, TraditionalHoldPositionPidConfig]:
    return {"traditional_hold_position_pid": TraditionalHoldPositionPidConfig(
        name="traditional_hold_position_pid",
        description="Original traditional position PID -> FinsROV 8-thruster output for HoldForPosition",
        env_config=BaseEnvironmentConfig(
            env_path="../../artifacts/unity_builds/rl/linux/HoldForPosition_Fossen_Parallel_15s_10Hz_Server/HoldForPosition_Parallel_15s.x86_64",
            num_envs=1, eval_num_envs=1, env_base_port=47126, timeout_wait=600,
            no_graphics=False, time_scale=1.0, parallel_mode="multi_area",
        ),
        training_config=TraditionalHoldPositionPidTrainingConfig(),
    )}
