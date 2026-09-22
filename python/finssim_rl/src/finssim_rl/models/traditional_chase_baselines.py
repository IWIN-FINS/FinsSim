"""Static 1Chase1 chase baselines sharing the 14D controller-body observation.

All controllers consume the same observation contract:
  [0:3] self linear velocity body, [3:6] self angular velocity body,
  [6:9] prey relative position body, [9:12] prey relative velocity body,
  [12] distance, [13] bearing.

The controller body convention is x=forward, y=up, z=left. The measured
FinsROV actuator convention is positive yaw turns right, so a target with
positive z (left) requires a negative yaw command.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from finssim_rl.models.pid_controller import PIDController
from finssim_rl.models.thrust_allocator import ThrustAllocator
from finssim_rl.models.traditional_position_pid import TraditionalPositionPIDModel
from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig


CHASE_OBSERVATION_DIM = 14
THRUSTER_ACTION_DIM = 8


def _resolve_device(device: str) -> str:
    return "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)


def _normalize_obs(observation: Union[np.ndarray, Dict[str, np.ndarray]]) -> tuple[np.ndarray, bool]:
    if isinstance(observation, dict):
        observation = observation.get("state", observation.get("observation", observation.get("obs", next(iter(observation.values())))))
    obs = np.asarray(observation, dtype=np.float32)
    is_vector = obs.ndim == 1
    if is_vector:
        obs = obs.reshape(1, -1)
    elif obs.ndim > 2:
        obs = obs.reshape(-1, obs.shape[-1])
    if obs.ndim != 2:
        raise ValueError(f"Expected vector observation with shape (batch, >=14), got {obs.shape}")
    if obs.shape[1] < CHASE_OBSERVATION_DIM:
        raise ValueError(f"1Chase1 baseline requires {CHASE_OBSERVATION_DIM}D observation, got {obs.shape[1]}D")
    return np.nan_to_num(obs[:, :CHASE_OBSERVATION_DIM], nan=0.0, posinf=0.0, neginf=0.0), is_vector


@dataclass
class TraditionalChaseBaselineTrainingConfig(BaseTrainingConfig):
    """Shared tuning for static PID, wrench-PD, and direct-thruster baselines."""

    dt: float = 0.02
    guidance_lookahead_seconds: float = 0.30
    guidance_error_limits: Tuple[float, float, float] = (6.0, 2.0, 6.0)
    yaw_deadband_distance: float = 0.10

    # PID variant: [depth Kp, Ki, Kd, surge Kp, Ki, Kd, sway Kp, Ki, Kd].
    pid_params: Tuple[float, ...] = (8.0, 0.05, 0.20, 8.0, 0.05, 0.20, 8.0, 0.05, 0.20)
    yaw_pid_params: Tuple[float, float, float] = (0.12, 0.0, 0.01)

    # Wrench-PD variant. The velocity is prey velocity minus ROV velocity.
    wrench_position_gains: Tuple[float, float, float] = (8.0, 8.0, 8.0)
    wrench_relative_velocity_gains: Tuple[float, float, float] = (2.0, 2.0, 2.0)
    wrench_force_limits: Tuple[float, float, float] = (120.0, 120.0, 120.0)
    wrench_yaw_kp: float = 0.12
    wrench_yaw_rate_kd: float = 0.40
    wrench_yaw_limit: float = 30.0
    # Pursuit is deliberately heading-first: hold horizontal translation until
    # the vehicle is sufficiently aligned with the prey, then use surge only.
    wrench_forward_enable_heading_error_deg: float = 15.0

    # Direct 8-thruster variant, normalized before the explicit mixer.
    direct_position_full_scale: Tuple[float, float, float] = (4.0, 1.5, 4.0)
    direct_relative_velocity_gains: Tuple[float, float, float] = (0.12, 0.12, 0.12)
    direct_yaw_full_error_deg: float = 60.0
    direct_yaw_rate_gain: float = 0.10
    # The direct mixer follows the same heading-first pursuit contract as the
    # wrench controller.  It must not combine sway with yaw for a side-on prey.
    direct_forward_enable_heading_error_deg: float = 5.0

    # The allocator is intentionally common to PID and wrench-PD variants.
    allocator_control_axis_ranges: Tuple[float, float, float, float, float, float] = (25.0, 25.0, 25.0, 1.0, 25.0, 1.0)
    allocator_allocation_mode: str = "empirical_thruster_mixer"
    # [Fx, Fy, Fz, Mx, My, Mz], derived for FinsROV_Fossen at COM=[0,-0.08,0].
    # Only physical_wrench_allocator consumes these values as residual weights.
    allocator_physical_wrench_limits: Tuple[float, float, float, float, float, float] = (
        19.528527, 22.501886, 18.415027, 3.863995, 7.080833, 3.079985,
    )
    allocator_thruster_force_limit_positive: Tuple[float, float, float, float, float, float, float, float] = (7.0,) * 8
    allocator_thruster_force_limit_negative: Tuple[float, float, float, float, float, float, float, float] = (7.0,) * 8
    output_slew_rate_limit: float = 500.0


class TraditionalChaseBaselineModel:
    """Predict-compatible static chase controller for one chosen low-level path."""

    VALID_KINDS = {"pid", "wrench", "thruster"}

    def __init__(self, kind: str, config: TraditionalChaseBaselineTrainingConfig, device: str = "auto") -> None:
        if kind not in self.VALID_KINDS:
            raise ValueError(f"Unknown chase baseline kind {kind!r}; expected one of {sorted(self.VALID_KINDS)}")
        self.kind = kind
        self.config = config
        self.device = _resolve_device(device)
        self.last_diagnostics: dict[str, np.ndarray | float] = {}

        self.allocator = ThrustAllocator(
            device=self.device,
            control_axis_ranges=config.allocator_control_axis_ranges,
            allocation_mode=config.allocator_allocation_mode,
            physical_wrench_limits=config.allocator_physical_wrench_limits,
            thruster_force_limit_positive=config.allocator_thruster_force_limit_positive,
            thruster_force_limit_negative=config.allocator_thruster_force_limit_negative,
            debug_print_interval=0,
        )
        self.pid_model: Optional[TraditionalPositionPIDModel] = None
        if kind == "pid":
            controller = PIDController(
                action_dim=THRUSTER_ACTION_DIM,
                device=self.device,
                dt=config.dt,
                output_range=(-250.0, 250.0),
                surge_output_limit=config.wrench_force_limits[0],
                sway_output_limit=config.wrench_force_limits[2],
                yaw_deadband_deg=0.0,
                yaw_output_limit=config.wrench_yaw_limit,
                output_slew_rate_limit=config.output_slew_rate_limit,
                allocator_control_axis_ranges=config.allocator_control_axis_ranges,
                allocator_allocation_mode="empirical_thruster_mixer",
                debug_print_interval=0,
            )
            self.pid_model = TraditionalPositionPIDModel(
                controller=controller,
                pid_params=config.pid_params,
                yaw_pid_params=config.yaw_pid_params,
                enable_yaw_control=True,
                action_dim=THRUSTER_ACTION_DIM,
                enable_logging=False,
            )

    def reset(self) -> None:
        if self.pid_model is not None:
            self.pid_model.reset()
        self.last_diagnostics = {}

    def _guidance(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        position_error = obs[:, 6:9].copy()
        relative_velocity = obs[:, 9:12].copy()
        angular_velocity = obs[:, 3:6].copy()
        limits = np.asarray(self.config.guidance_error_limits, dtype=np.float32).reshape(1, 3)
        guided_error = position_error + relative_velocity * float(self.config.guidance_lookahead_seconds)
        guided_error = np.clip(guided_error, -limits, limits)

        planar_distance = np.linalg.norm(position_error[:, (0, 2)], axis=1)
        # ControllerBodyFrame uses positive z for left, while the measured
        # positive yaw command turns the FinsROV right.
        yaw_error_deg = -np.degrees(np.arctan2(position_error[:, 2], position_error[:, 0])).astype(np.float32)
        yaw_error_deg[planar_distance <= float(self.config.yaw_deadband_distance)] = 0.0
        return position_error, relative_velocity, angular_velocity, np.concatenate([guided_error, yaw_error_deg[:, None]], axis=1)

    def _allocate_wrench(self, wrench: np.ndarray) -> np.ndarray:
        wrench_tensor = torch.as_tensor(wrench, dtype=torch.float32, device=self.allocator.A.device)
        with torch.no_grad():
            return self.allocator(wrench_tensor).detach().cpu().numpy().astype(np.float32, copy=False)

    def _wrench_pd(self, guided_error4: np.ndarray, relative_velocity: np.ndarray, angular_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        gains_p = np.asarray(config.wrench_position_gains, dtype=np.float32).reshape(1, 3)
        gains_v = np.asarray(config.wrench_relative_velocity_gains, dtype=np.float32).reshape(1, 3)
        force_limits = np.asarray(config.wrench_force_limits, dtype=np.float32).reshape(1, 3)
        position_force = gains_p * guided_error4[:, :3] + gains_v * relative_velocity

        # This baseline models a forward-facing pursuit vehicle rather than a
        # holonomic point tracker.  When the prey is outside the heading gate,
        # yaw is the only horizontal command.  Once aligned, advance in surge;
        # do not strafe toward the prey with Fz.
        heading_error_abs_deg = np.abs(guided_error4[:, 3])
        forward_enabled = heading_error_abs_deg <= float(config.wrench_forward_enable_heading_error_deg)
        force = np.zeros_like(position_force)
        force[:, 1] = position_force[:, 1]  # Keep depth control active while turning.
        force[:, 0] = np.where(forward_enabled, np.maximum(position_force[:, 0], 0.0), 0.0)
        force = np.clip(force, -force_limits, force_limits)
        yaw = (
            float(config.wrench_yaw_kp) * guided_error4[:, 3]
            - float(config.wrench_yaw_rate_kd) * angular_velocity[:, 1]
        )
        yaw = np.clip(yaw, -float(config.wrench_yaw_limit), float(config.wrench_yaw_limit))
        wrench = np.column_stack(
            [force[:, 0], force[:, 1], force[:, 2], np.zeros_like(yaw), yaw, np.zeros_like(yaw)]
        ).astype(np.float32)
        return wrench, self._allocate_wrench(wrench)

    def _direct_thruster(self, guided_error4: np.ndarray, relative_velocity: np.ndarray, angular_velocity: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        config = self.config
        full_scale = np.asarray(config.direct_position_full_scale, dtype=np.float32).reshape(1, 3)
        velocity_gain = np.asarray(config.direct_relative_velocity_gains, dtype=np.float32).reshape(1, 3)
        linear = np.clip(guided_error4[:, :3] / full_scale + velocity_gain * relative_velocity, -1.0, 1.0)
        yaw = np.clip(
            guided_error4[:, 3] / max(float(config.direct_yaw_full_error_deg), 1e-3)
            - float(config.direct_yaw_rate_gain) * angular_velocity[:, 1],
            -1.0,
            1.0,
        )

        # A direct thruster command is still a forward-facing pursuit
        # controller, not a holonomic point tracker.  For a prey at body +z
        # (left), combining sway-left with yaw-left made the ROV side-slip
        # toward the prey instead of first pointing its nose at it.  Hold all
        # horizontal translation outside the heading gate, then use surge only.
        heading_error_abs_deg = np.abs(guided_error4[:, 3])
        forward_enabled = heading_error_abs_deg <= float(config.direct_forward_enable_heading_error_deg)
        forward = np.where(forward_enabled, np.maximum(linear[:, 0], 0.0), 0.0)
        up = linear[:, 1]

        # Canonical order: V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF.
        # Canonical FinsROV surge (+Unity local X) is [+,+,-,-]. This is the
        # same controller body frame used by the 14D Unity observation.
        horizontal = np.column_stack([forward + yaw, forward + yaw, -forward + yaw, -forward + yaw])
        horizontal_peak = np.maximum(1.0, np.max(np.abs(horizontal), axis=1, keepdims=True))
        horizontal /= horizontal_peak
        action = np.column_stack([up, up, up, up, horizontal]).astype(np.float32)
        wrench = np.column_stack(
            [forward, up, np.zeros_like(forward), np.zeros_like(yaw), yaw, np.zeros_like(yaw)]
        ).astype(np.float32)
        return wrench, action

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state=None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ):
        del deterministic
        if episode_start is not None and np.any(episode_start):
            self.reset()
        obs, is_vector = _normalize_obs(observation)
        position_error, relative_velocity, angular_velocity, guided_error4 = self._guidance(obs)

        if self.kind == "pid":
            assert self.pid_model is not None
            action, _ = self.pid_model.predict_from_body_position_error(
                guided_error4[:, :3], yaw_error_deg=guided_error4[:, 3], deterministic=True
            )
            action = np.asarray(action, dtype=np.float32).reshape(-1, THRUSTER_ACTION_DIM)
            wrench = np.full((action.shape[0], 6), np.nan, dtype=np.float32)
        elif self.kind == "wrench":
            wrench, action = self._wrench_pd(guided_error4, relative_velocity, angular_velocity)
        else:
            wrench, action = self._direct_thruster(guided_error4, relative_velocity, angular_velocity)

        heading_gate_deg = (
            self.config.direct_forward_enable_heading_error_deg
            if self.kind == "thruster"
            else self.config.wrench_forward_enable_heading_error_deg
        )
        self.last_diagnostics = {
            "position_error_body": position_error[0].copy(),
            "guided_error_body": guided_error4[0, :3].copy(),
            "relative_velocity_body": relative_velocity[0].copy(),
            "yaw_error_deg": float(guided_error4[0, 3]),
            "forward_enabled": float(abs(guided_error4[0, 3]) <= heading_gate_deg),
            "wrench": wrench[0].copy(),
            "action": action[0].copy(),
        }
        return (action[0] if is_vector else action), state

    def format_diagnostics(self) -> str:
        if not self.last_diagnostics:
            return f"[TraditionalChase/{self.kind}] no action has been generated yet"
        error = np.asarray(self.last_diagnostics["position_error_body"])
        guided = np.asarray(self.last_diagnostics["guided_error_body"])
        action = np.asarray(self.last_diagnostics["action"])
        wrench = np.asarray(self.last_diagnostics["wrench"])
        wrench_text = "PID internal" if np.isnan(wrench).all() else np.array2string(wrench, precision=3, suppress_small=True)
        return (
            f"[TraditionalChase/{self.kind}] e_body={np.array2string(error, precision=3, suppress_small=True)} "
            f"guided={np.array2string(guided, precision=3, suppress_small=True)} "
            f"yaw={float(self.last_diagnostics['yaw_error_deg']):+.1f}deg wrench={wrench_text} "
            f"forward_enabled={bool(self.last_diagnostics['forward_enabled'])} "
            f"action_abs_mean={float(np.mean(np.abs(action))):.3f}"
        )


@dataclass
class TraditionalChaseBaselineConfig(BaseConfig):
    """Config entry for one no-checkpoint 1Chase1 baseline."""

    model_type: str = "TRADITIONAL_CHASE_BASELINE"
    requires_checkpoint: bool = False
    controller_kind: str = "pid"
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: TraditionalChaseBaselineTrainingConfig = field(default_factory=TraditionalChaseBaselineTrainingConfig)

    def create_model(self, env, args, load_checkpoint: Optional[str] = None):
        del env, load_checkpoint
        return TraditionalChaseBaselineModel(self.controller_kind, self.training_config, device=args.device)

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        del model_path, env
        return TraditionalChaseBaselineModel(self.controller_kind, self.training_config, device=device)


def get_traditional_chase_baseline_configs() -> Dict[str, TraditionalChaseBaselineConfig]:
    env = BaseEnvironmentConfig(
        time_scale=1.0,
        env_path="../../artifacts/unity_builds/rl/linux/FinsROV/1Chase1_visual/1Chase1.x86_64",
        num_envs=1,
        eval_num_envs=1,
        env_base_port=16200,
        timeout_wait=600,
        no_graphics=False,
        environment_parameters={"finsim_1chase1_observation_mode": float(CHASE_OBSERVATION_DIM)},
        unity_additional_args=["-fins-1chase1-mode", "direct14"],
    )
    common = TraditionalChaseBaselineTrainingConfig()
    physical_wrench_allocator = replace(
        common,
        # Keep the PD target within the independently achievable pure-axis
        # envelope before the bounded physical allocator handles combinations.
        wrench_force_limits=(19.528527, 22.501886, 18.415027),
        wrench_yaw_limit=7.080833,
        allocator_allocation_mode="physical_wrench_allocator",
    )
    return {
        "traditional_chase_pid_1chase1": TraditionalChaseBaselineConfig(
            name="traditional_chase_pid_1chase1",
            description="1Chase1 prey pursuit: position PID + yaw PID + thrust allocator",
            controller_kind="pid", env_config=replace(env), training_config=replace(common),
        ),
        "traditional_chase_wrench_1chase1": TraditionalChaseBaselineConfig(
            name="traditional_chase_wrench_1chase1",
            description="1Chase1 prey pursuit: direct wrench PD + thrust allocator",
            controller_kind="wrench", env_config=replace(env, env_base_port=16210), training_config=replace(common),
        ),
        "traditional_chase_wrench_physical_wrench_allocator_1chase1": TraditionalChaseBaselineConfig(
            name="traditional_chase_wrench_physical_wrench_allocator_1chase1",
            description="1Chase1 prey pursuit: wrench PD + bounded physical wrench allocator",
            controller_kind="wrench", env_config=replace(env, env_base_port=16230),
            training_config=replace(physical_wrench_allocator),
        ),
        "traditional_chase_thruster_1chase1": TraditionalChaseBaselineConfig(
            name="traditional_chase_thruster_1chase1",
            description="1Chase1 prey pursuit: explicit canonical 8-thruster mixer",
            controller_kind="thruster", env_config=replace(env, env_base_port=16220), training_config=replace(common),
        ),
    }


__all__ = [
    "CHASE_OBSERVATION_DIM", "TraditionalChaseBaselineConfig", "TraditionalChaseBaselineModel",
    "TraditionalChaseBaselineTrainingConfig", "get_traditional_chase_baseline_configs",
]
