"""
Traditional velocity controller: PID + feedforward + second-order ESO.

The controller works in a 3-axis velocity space and maps the resulting
virtual wrench through the existing ThrustAllocator.  The default axis order
is body-frame linear velocity [surge, heave, sway] -> [Fx, Fy, Fz].
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from finssim_rl.models.thrust_allocator import ThrustAllocator
from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig


AXIS_MODE_ANGULAR_YPR = "angular_yaw_pitch_roll"
AXIS_MODE_LINEAR_XYZ = "linear_body_xyz"
OBS_FORMAT_POSE20 = "pose20"
OBS_FORMAT_VELOCITY_NORMALIZED_BODY = "velocity_normalized_body"

DEFAULT_J_LINEAR_XYZ: Tuple[float, float, float] = (37.7, 65.3, 34.0)
DEFAULT_B_LINEAR_XYZ: Tuple[float, float, float] = (96.59, 276.35, 154.37)
DEFAULT_D_LINEAR_XYZ: Tuple[float, float, float] = (12.35, 47.70, 11.69)
DEFAULT_LINEAR_VELOCITY_SCALE: Tuple[float, float, float] = (0.5641, 0.4965, 0.8870)
DEFAULT_ANGULAR_VELOCITY_SCALE: Tuple[float, float, float] = (0.6163, 0.6000, 0.6621)
DEFAULT_REFERENCE_VELOCITY_LIMIT: Tuple[float, float, float] = (0.30, 0.30, 0.30)
DEFAULT_REFERENCE_VELOCITY_RATE_LIMIT: Tuple[float, float, float] = (0.12, 0.10, 0.12)
DEFAULT_ANGULAR_DAMPING_GAINS_XYZ: Tuple[float, float, float] = (0.0, 0.05, 0.0)
DEFAULT_OUTPUT_LIMIT_XYZ: Tuple[float, float, float] = (3.5, 3.5, 3.5)
DEFAULT_INTEGRAL_LIMIT_XYZ: Tuple[float, float, float] = (6.0, 3.0, 6.0)
DEFAULT_PID_OUTPUT_GAIN_XYZ: Tuple[float, float, float] = (0.25, 0.25, 0.25)
DEFAULT_FEEDFORWARD_GAIN_XYZ: Tuple[float, float, float] = (0.02, 0.05, 0.04)
DEFAULT_FEEDFORWARD_ACCEL_GAIN_XYZ: Tuple[float, float, float] = (0.0, 0.0, 0.0)
DEFAULT_FEEDFORWARD_OUTPUT_LIMIT_XYZ: Tuple[float, float, float] = (0.7, 1.5 , 0.9)
DEFAULT_FEEDFORWARD_OVERSPEED_DEADBAND: float = 0.03
DEFAULT_ESO_GAIN_XYZ: Tuple[float, float, float] = (0.01, 0.0, 0.01)
DEFAULT_ESO_OUTPUT_LIMIT_XYZ: Tuple[float, float, float] = (0.6, 0.0, 0.6)

# Conservative force-domain PID gains for [surge, heave, sway].
DEFAULT_PID_GAINS_LINEAR_XYZ: Tuple[Tuple[float, float, float], ...] = (
    (2.40, 0.220, 0.045),
    (4.50, 0.950, 0.080),
    (2.70, 0.220, 0.045),
)
PID_ASSIST_GAINS_LINEAR_XYZ: Tuple[Tuple[float, float, float], ...] = (
    (0.00, 0.010, 0.00),
    (0.00, 0.010, 0.00),
    (0.00, 0.010, 0.00),
)
ZERO_PID_GAINS_LINEAR_XYZ: Tuple[Tuple[float, float, float], ...] = (
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0),
)


def _as_batch3(values: np.ndarray) -> Tuple[np.ndarray, bool]:
    array = np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    was_1d = array.ndim == 1
    if was_1d:
        array = array.reshape(1, -1)
    elif array.ndim > 2:
        array = array.reshape(array.shape[0], -1)

    if array.shape[1] < 3:
        array = np.pad(array, ((0, 0), (0, 3 - array.shape[1])), constant_values=0.0)
    elif array.shape[1] > 3:
        array = array[:, :3]
    return array.astype(np.float32, copy=False), was_1d


def _fit_action_dim(actions: np.ndarray, expected_dim: int) -> np.ndarray:
    actions = np.nan_to_num(np.asarray(actions, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if actions.ndim == 1:
        if actions.shape[0] < expected_dim:
            actions = np.pad(actions, (0, expected_dim - actions.shape[0]), constant_values=0.0)
        elif actions.shape[0] > expected_dim:
            actions = actions[:expected_dim]
        return np.clip(actions, -1.0, 1.0).astype(np.float32, copy=False)

    if actions.shape[1] < expected_dim:
        actions = np.pad(actions, ((0, 0), (0, expected_dim - actions.shape[1])), constant_values=0.0)
    elif actions.shape[1] > expected_dim:
        actions = actions[:, :expected_dim]
    return np.clip(actions, -1.0, 1.0).astype(np.float32, copy=False)


def _axis3_array(
    values: Union[float, Sequence[float]],
    *,
    scalar_repeats: bool = True,
    pad_value: float = 0.0,
) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=np.float32).reshape(-1), nan=0.0, posinf=0.0, neginf=0.0)
    if array.shape[0] == 0:
        array = np.full(3, pad_value, dtype=np.float32)
    elif array.shape[0] == 1 and scalar_repeats:
        array = np.repeat(array, 3)
    elif array.shape[0] < 3:
        array = np.pad(array, (0, 3 - array.shape[0]), constant_values=pad_value)
    elif array.shape[0] > 3:
        array = array[:3]
    return array.astype(np.float32, copy=False)


def _normalize_obs_array(obs: Union[np.ndarray, Dict[str, np.ndarray]], default_num_envs: int = 1) -> np.ndarray:
    """Normalize Unity/Gym vector observations into shape (batch, obs_dim)."""
    if isinstance(obs, dict):
        if "state" in obs:
            obs = obs["state"]
        elif "observation" in obs:
            obs = obs["observation"]
        elif "obs" in obs:
            obs = obs["obs"]
        else:
            obs = next(iter(obs.values()))

    if isinstance(obs, (list, tuple)):
        vector_candidates = []
        for item in obs:
            item_array = np.asarray(item, dtype=np.float32)
            if item_array.ndim >= 1 and item_array.shape[-1] >= 14:
                vector_candidates.append(item_array)
        obs = vector_candidates[0] if vector_candidates else obs[0]

    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.ndim == 1:
        obs_array = obs_array.reshape(1, -1)
    elif obs_array.ndim > 2:
        obs_array = np.squeeze(obs_array)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
        elif obs_array.ndim > 2 and obs_array.shape[-1] >= 14:
            obs_array = obs_array.reshape(-1, obs_array.shape[-1])

    if obs_array.ndim != 2:
        if not getattr(_normalize_obs_array, "_warned_bad_shape", False):
            print(f"[TraditionalVelocityPID ObsParse WARNING] unsupported obs shape={obs_array.shape}, fallback to zeros")
            setattr(_normalize_obs_array, "_warned_bad_shape", True)
        obs_array = np.zeros((default_num_envs, 20), dtype=np.float32)
    return obs_array


def _select_obs_columns(obs_array: np.ndarray, indices: Sequence[int]) -> np.ndarray:
    selected = np.zeros((obs_array.shape[0], 3), dtype=np.float32)
    for out_idx, obs_idx in enumerate(tuple(indices)[:3]):
        if 0 <= int(obs_idx) < obs_array.shape[1]:
            selected[:, out_idx] = obs_array[:, int(obs_idx)]
    return selected


def _velocity_scale_for_axis( 
    axis_mode: str,
    linear_scale: Sequence[float],
    angular_scale: Sequence[float],
) -> np.ndarray:
    scale = angular_scale if axis_mode == AXIS_MODE_ANGULAR_YPR else linear_scale
    scale_array = np.asarray(scale, dtype=np.float32).reshape(-1)
    if scale_array.shape[0] < 3:
        scale_array = np.pad(scale_array, (0, 3 - scale_array.shape[0]), constant_values=1.0)
    elif scale_array.shape[0] > 3:
        scale_array = scale_array[:3]
    return scale_array.reshape(1, 3)


def _pos_unity_to_submarine(pos_xyz: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_xyz, dtype=np.float32)
    return pos.astype(np.float32, copy=False)


def _quat_rotate_inverse(quat_xyzw: np.ndarray, vec_world: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_xyzw, dtype=np.float32)
    v = np.asarray(vec_world, dtype=np.float32)
    if q.ndim == 1:
        q = q.reshape(1, 4)
    if v.ndim == 1:
        v = v.reshape(1, 3)

    q_inv = np.stack([-q[:, 0], -q[:, 1], -q[:, 2], q[:, 3]], axis=1)
    qv = np.stack(
        [
            q_inv[:, 3] * v[:, 0] + q_inv[:, 1] * v[:, 2] - q_inv[:, 2] * v[:, 1],
            q_inv[:, 3] * v[:, 1] + q_inv[:, 2] * v[:, 0] - q_inv[:, 0] * v[:, 2],
            q_inv[:, 3] * v[:, 2] + q_inv[:, 0] * v[:, 1] - q_inv[:, 1] * v[:, 0],
            -q_inv[:, 0] * v[:, 0] - q_inv[:, 1] * v[:, 1] - q_inv[:, 2] * v[:, 2],
        ],
        axis=1,
    )
    result = np.stack(
        [
            qv[:, 3] * q[:, 0] + qv[:, 0] * q[:, 3] + qv[:, 1] * q[:, 2] - qv[:, 2] * q[:, 1],
            qv[:, 3] * q[:, 1] + qv[:, 1] * q[:, 3] + qv[:, 2] * q[:, 0] - qv[:, 0] * q[:, 2],
            qv[:, 3] * q[:, 2] + qv[:, 2] * q[:, 3] + qv[:, 0] * q[:, 1] - qv[:, 1] * q[:, 0],
        ],
        axis=1,
    )
    return result.astype(np.float32)


def _angular_world_xyz_to_ypr(obs_array: np.ndarray, angular_world_xyz: np.ndarray) -> np.ndarray:
    if obs_array.shape[1] >= 14:
        local_xyz = _quat_rotate_inverse(obs_array[:, 10:14], angular_world_xyz)
    else:
        local_xyz = angular_world_xyz
    return np.stack([local_xyz[:, 1], local_xyz[:, 2], local_xyz[:, 0]], axis=1).astype(np.float32)


def extract_current_velocity(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    axis_mode: str = AXIS_MODE_LINEAR_XYZ,
    observation_format: str = OBS_FORMAT_POSE20,
    linear_velocity_scale: Sequence[float] = DEFAULT_LINEAR_VELOCITY_SCALE,
    angular_velocity_scale: Sequence[float] = DEFAULT_ANGULAR_VELOCITY_SCALE,
    default_num_envs: int = 1,
) -> np.ndarray:
    """Extract current body-frame velocity for the selected control axes."""
    obs_array = _normalize_obs_array(obs, default_num_envs=default_num_envs)
    batch_size = obs_array.shape[0]

    if observation_format == OBS_FORMAT_VELOCITY_NORMALIZED_BODY:
        # ControlForVelocity_IncrementalReward CollectObservations:
        # [0:3] target linear, [3:6] target angular,
        # [6:9] current linear, [9:12] current angular, all normalized.
        start = 9 if axis_mode == AXIS_MODE_ANGULAR_YPR else 6
        raw = _select_obs_columns(obs_array, (start, start + 1, start + 2))
        scale = _velocity_scale_for_axis(axis_mode, linear_velocity_scale, angular_velocity_scale)
        return (raw * scale).astype(np.float32)

    if axis_mode == AXIS_MODE_LINEAR_XYZ:
        if obs_array.shape[1] >= 17:
            linear_world = obs_array[:, 14:17]
            if obs_array.shape[1] >= 14:
                linear_local = _quat_rotate_inverse(obs_array[:, 10:14], linear_world)
                return _pos_unity_to_submarine(linear_local).astype(np.float32)
            return linear_world.astype(np.float32)
        return np.zeros((batch_size, 3), dtype=np.float32)

    if obs_array.shape[1] >= 20:
        return _angular_world_xyz_to_ypr(obs_array, obs_array[:, 17:20])
    return np.zeros((batch_size, 3), dtype=np.float32)


def extract_reference_velocity(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    axis_mode: str = AXIS_MODE_LINEAR_XYZ,
    reference_indices: Sequence[int] = (0, 1, 2),
    reference_frame: str = "body",
    observation_format: str = OBS_FORMAT_POSE20,
    linear_velocity_scale: Sequence[float] = DEFAULT_LINEAR_VELOCITY_SCALE,
    angular_velocity_scale: Sequence[float] = DEFAULT_ANGULAR_VELOCITY_SCALE,
    default_num_envs: int = 1,
) -> np.ndarray:
    """
    Extract reference velocity from observation columns.

    reference_frame:
      - "body": selected columns are already in controller axis order.
      - "unity_world": selected columns are a Unity world XYZ vector.
    """
    obs_array = _normalize_obs_array(obs, default_num_envs=default_num_envs)
    raw = _select_obs_columns(obs_array, reference_indices)

    if observation_format == OBS_FORMAT_VELOCITY_NORMALIZED_BODY or reference_frame == "normalized_body":
        scale = _velocity_scale_for_axis(axis_mode, linear_velocity_scale, angular_velocity_scale)
        return (raw * scale).astype(np.float32)

    if reference_frame == "unity_world":
        if axis_mode == AXIS_MODE_LINEAR_XYZ:
            if obs_array.shape[1] >= 14:
                return _pos_unity_to_submarine(_quat_rotate_inverse(obs_array[:, 10:14], raw)).astype(np.float32)
            return raw.astype(np.float32)
        return _angular_world_xyz_to_ypr(obs_array, raw)

    return raw.astype(np.float32)


class VelocityPIDFeedforwardESO(nn.Module):
    """3-axis PID + feedforward + second-order ESO controller."""

    def __init__(
        self,
        dt: float = 0.02,
        pid_gains: Sequence[Sequence[float]] = DEFAULT_PID_GAINS_LINEAR_XYZ,
        inertia: Sequence[float] = DEFAULT_J_LINEAR_XYZ,
        linear_damping: Sequence[float] = DEFAULT_B_LINEAR_XYZ,
        quadratic_damping: Sequence[float] = DEFAULT_D_LINEAR_XYZ,
        output_limit: float = 80.0,
        output_limit_xyz: Sequence[float] = DEFAULT_OUTPUT_LIMIT_XYZ,
        integral_limit: float = 20.0,
        integral_limit_xyz: Optional[Sequence[float]] = None,
        pid_output_gain: Union[float, Sequence[float]] = DEFAULT_PID_OUTPUT_GAIN_XYZ,
        derivative_filter_alpha: float = 0.2,
        use_feedforward: bool = True,
        feedforward_gain: Union[float, Sequence[float]] = 1.0,
        feedforward_accel_gain: Union[float, Sequence[float]] = 0.0,
        feedforward_output_limit_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        use_feedforward_overspeed_gate: bool = True,
        feedforward_overspeed_deadband: float = DEFAULT_FEEDFORWARD_OVERSPEED_DEADBAND,
        use_eso: bool = True,
        eso_gain: Union[float, Sequence[float]] = 1.0,
        eso_bandwidth: float = 8.0,
        eso_disturbance_accel_limit: float = 50.0,
        eso_output_limit_xyz: Sequence[float] = (0.0, 0.0, 0.0),
        angular_damping_gains: Sequence[float] = DEFAULT_ANGULAR_DAMPING_GAINS_XYZ,
        angular_moment_limit: float = 8.0,
        device: str = "cpu",
        allocator_control_linear_range: float = 1.0,
        allocator_deadzone_comp: float = 0.0,
        allocator_allocation_mode: str = "empirical_thruster_mixer",
        allocator_preserve_direction: bool = True,
        allocator_saturation_margin: float = 0.95,
        allocator_debug_print_interval: int = 0,
        debug_print_interval: int = 0,
    ) -> None:
        super().__init__()
        resolved_device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu" if device == "auto" else device
        self.device = resolved_device
        self.dt = max(float(dt), 1e-6)
        self.output_limit = max(float(output_limit), 0.0)
        output_limit_xyz_array = np.asarray(output_limit_xyz, dtype=np.float32).reshape(-1)
        if output_limit_xyz_array.shape[0] < 3:
            output_limit_xyz_array = np.pad(output_limit_xyz_array, (0, 3 - output_limit_xyz_array.shape[0]), constant_values=0.0)
        elif output_limit_xyz_array.shape[0] > 3:
            output_limit_xyz_array = output_limit_xyz_array[:3]
        self.integral_limit = max(float(integral_limit), 0.0)
        self.derivative_filter_alpha = float(np.clip(derivative_filter_alpha, 0.0, 1.0))
        self.use_feedforward = bool(use_feedforward)
        self.use_feedforward_overspeed_gate = bool(use_feedforward_overspeed_gate)
        self.feedforward_overspeed_deadband = max(float(feedforward_overspeed_deadband), 1e-6)
        self.use_eso = bool(use_eso)
        self.eso_bandwidth = max(float(eso_bandwidth), 1e-6)
        self.eso_disturbance_accel_limit = max(float(eso_disturbance_accel_limit), 0.0)
        self.angular_moment_limit = max(float(angular_moment_limit), 0.0)
        self.allocator_preserve_direction = bool(allocator_preserve_direction)
        self.allocator_saturation_margin = float(np.clip(allocator_saturation_margin, 0.0, 1.0))
        self.debug_print_interval = max(int(debug_print_interval), 0)

        gains = torch.as_tensor(pid_gains, dtype=torch.float32, device=resolved_device)
        if gains.shape != (3, 3):
            raise ValueError(f"pid_gains must have shape (3, 3), got {tuple(gains.shape)}")
        self.register_buffer("pid_gains", gains)
        self.register_buffer("J", torch.as_tensor(inertia, dtype=torch.float32, device=resolved_device).view(1, 3))
        self.register_buffer("B", torch.as_tensor(linear_damping, dtype=torch.float32, device=resolved_device).view(1, 3))
        self.register_buffer("D", torch.as_tensor(quadratic_damping, dtype=torch.float32, device=resolved_device).view(1, 3))
        self.register_buffer(
            "output_limit_xyz",
            torch.as_tensor(output_limit_xyz_array, dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        if integral_limit_xyz is None:
            integral_limit_xyz_array = np.full(3, self.integral_limit, dtype=np.float32)
        else:
            integral_limit_xyz_array = np.maximum(_axis3_array(integral_limit_xyz), 0.0)
        self.register_buffer(
            "integral_limit_xyz",
            torch.as_tensor(integral_limit_xyz_array, dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        self.register_buffer(
            "pid_output_gain",
            torch.as_tensor(np.maximum(_axis3_array(pid_output_gain), 0.0), dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        self.register_buffer(
            "feedforward_gain",
            torch.as_tensor(np.maximum(_axis3_array(feedforward_gain), 0.0), dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        self.register_buffer(
            "feedforward_accel_gain",
            torch.as_tensor(np.maximum(_axis3_array(feedforward_accel_gain), 0.0), dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        self.register_buffer(
            "feedforward_output_limit_xyz",
            torch.as_tensor(np.maximum(_axis3_array(feedforward_output_limit_xyz, scalar_repeats=False), 0.0), dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        self.register_buffer(
            "eso_gain",
            torch.as_tensor(np.maximum(_axis3_array(eso_gain), 0.0), dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        self.register_buffer(
            "eso_output_limit_xyz",
            torch.as_tensor(np.maximum(_axis3_array(eso_output_limit_xyz, scalar_repeats=False), 0.0), dtype=torch.float32, device=resolved_device).view(1, 3),
        )
        self.register_buffer(
            "angular_damping_gains",
            torch.as_tensor(angular_damping_gains, dtype=torch.float32, device=resolved_device).view(1, 3),
        )

        self.register_buffer("integral_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("prev_error_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("derivative_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("prev_ref_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("prev_velocity_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("prev_tau_cmd_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("pid_initialized_state", torch.zeros(0, dtype=torch.bool, device=resolved_device))
        self.register_buffer("eso_z1_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("eso_z2_state", torch.zeros(0, 3, dtype=torch.float32, device=resolved_device))
        self.register_buffer("initialized_state", torch.zeros(0, dtype=torch.bool, device=resolved_device))

        self.thrust_allocator = ThrustAllocator(
            device=resolved_device,
            deadzone_comp=allocator_deadzone_comp,
            control_linear_range=allocator_control_linear_range,
            allocation_mode=allocator_allocation_mode,
            debug_print_interval=allocator_debug_print_interval,
        )
        self._debug_step_count = 0

    def _prescale_tau_for_allocator(self, tau6: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Scale vertical/horizontal force groups before allocator clipping.

        The allocator clips each thruster independently.  If a large Fx command
        saturates all horizontal thrusters, the smaller Fz command disappears.
        Group-wise pre-scaling preserves the Fx/Fz/My ratio before clipping.
        """
        if not self.allocator_preserve_direction or self.allocator_saturation_margin <= 0.0:
            scales = torch.ones(tau6.shape[0], 2, dtype=tau6.dtype, device=tau6.device)
            return tau6, scales

        if self.thrust_allocator.allocation_mode == "axis_map":
            allocation_matrix = self.thrust_allocator.linear_axis_map
        else:
            allocation_matrix = self.thrust_allocator.A
        allocation_matrix = allocation_matrix.to(device=tau6.device, dtype=tau6.dtype)

        limit = torch.as_tensor(
            self.thrust_allocator.control_linear_range * self.allocator_saturation_margin,
            dtype=tau6.dtype,
            device=tau6.device,
        )
        eps = torch.as_tensor(1e-6, dtype=tau6.dtype, device=tau6.device)

        tau_scaled = tau6.clone()

        vertical_cols = [1, 3, 5]  # Fy, Mx, Mz -> T1..T4
        raw_vertical = torch.matmul(tau6[:, vertical_cols], allocation_matrix[:, vertical_cols].T)
        max_vertical = raw_vertical[:, :4].abs().amax(dim=1, keepdim=True)
        vertical_scale = torch.minimum(torch.ones_like(max_vertical), limit / torch.clamp(max_vertical, min=eps))
        tau_scaled[:, vertical_cols] = tau_scaled[:, vertical_cols] * vertical_scale

        horizontal_cols = [0, 2, 4]  # Fx, Fz, My -> T5..T8
        raw_horizontal = torch.matmul(tau6[:, horizontal_cols], allocation_matrix[:, horizontal_cols].T)
        max_horizontal = raw_horizontal[:, 4:8].abs().amax(dim=1, keepdim=True)
        horizontal_scale = torch.minimum(torch.ones_like(max_horizontal), limit / torch.clamp(max_horizontal, min=eps))
        tau_scaled[:, horizontal_cols] = tau_scaled[:, horizontal_cols] * horizontal_scale

        return tau_scaled, torch.cat([horizontal_scale, vertical_scale], dim=1)

    def _ensure_state(self, batch_size: int, dtype: torch.dtype, device: torch.device) -> None:
        needs_resize = (
            self.integral_state.shape[0] != batch_size
            or self.integral_state.device != device
            or self.integral_state.dtype != dtype
        )
        if not needs_resize:
            return

        self.integral_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.prev_error_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.derivative_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.prev_ref_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.prev_velocity_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.prev_tau_cmd_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.pid_initialized_state = torch.zeros(batch_size, dtype=torch.bool, device=device)
        self.eso_z1_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.eso_z2_state = torch.zeros(batch_size, 3, dtype=dtype, device=device)
        self.initialized_state = torch.zeros(batch_size, dtype=torch.bool, device=device)

    def _update_eso(self, current_velocity: torch.Tensor) -> torch.Tensor:
        if not self.use_eso:
            return torch.zeros_like(current_velocity)

        not_initialized = ~self.initialized_state
        if torch.any(not_initialized):
            self.eso_z1_state[not_initialized] = current_velocity[not_initialized]
            self.eso_z2_state[not_initialized] = 0.0
            self.prev_ref_state[not_initialized] = current_velocity[not_initialized]
            self.prev_tau_cmd_state[not_initialized] = 0.0
            self.initialized_state[not_initialized] = True

        beta1 = 2.0 * self.eso_bandwidth
        beta2 = self.eso_bandwidth * self.eso_bandwidth
        e = current_velocity - self.eso_z1_state
        # tau_prev is the previous control command:
        #   tau_prev = tau_cmd[k-1]
        u_eso = (
            self.prev_tau_cmd_state
            - self.B * current_velocity
            - self.D * current_velocity * torch.abs(current_velocity)
        )

        # Discrete second-order ESO:
        #   z1 = z1 + Ts * (z2 + u_eso / J + beta1 * e)
        #   z2 = z2 + Ts * beta2 * e
        #   tau_eso = J * z2
        self.eso_z1_state = self.eso_z1_state + self.dt * (
            self.eso_z2_state + u_eso / torch.clamp(self.J, min=1e-6) + beta1 * e
        )
        self.eso_z2_state = self.eso_z2_state + self.dt * beta2 * e

        if self.eso_disturbance_accel_limit > 0.0:
            self.eso_z2_state = torch.clamp(
                self.eso_z2_state,
                -self.eso_disturbance_accel_limit,
                self.eso_disturbance_accel_limit,
            )
        tau_eso = self.J * self.eso_z2_state
        return tau_eso

    def forward(
        self,
        reference_velocity: torch.Tensor,
        current_velocity: torch.Tensor,
        current_angular_velocity: Optional[torch.Tensor] = None,
        return_thrust: bool = True,
    ) -> torch.Tensor:
        ref = reference_velocity.to(device=self.J.device, dtype=self.J.dtype)
        cur = current_velocity.to(device=self.J.device, dtype=self.J.dtype)
        if ref.dim() == 1:
            ref = ref.unsqueeze(0)
        if cur.dim() == 1:
            cur = cur.unsqueeze(0)
        if ref.shape[-1] != 3 or cur.shape[-1] != 3:
            raise ValueError(f"reference/current velocity must end with dim 3, got {ref.shape} and {cur.shape}")
        if current_angular_velocity is None:
            angular = torch.zeros_like(cur)
        else:
            angular = current_angular_velocity.to(device=self.J.device, dtype=self.J.dtype)
            if angular.dim() == 1:
                angular = angular.unsqueeze(0)
            if angular.shape[-1] != 3:
                raise ValueError(f"current angular velocity must end with dim 3, got {angular.shape}")

        self._ensure_state(ref.shape[0], ref.dtype, ref.device)
        self._debug_step_count += 1

        tau_eso = self.eso_gain * self._update_eso(cur)
        if torch.any(self.eso_output_limit_xyz > 0.0):
            eso_axis_limit = torch.where(
                self.eso_output_limit_xyz > 0.0,
                self.eso_output_limit_xyz,
                torch.full_like(self.eso_output_limit_xyz, 1.0e12),
            )
            tau_eso = torch.clamp(tau_eso, -eso_axis_limit, eso_axis_limit)

        error = ref - cur
        self.integral_state = self.integral_state + error * self.dt
        if torch.any(self.integral_limit_xyz > 0.0):
            integral_axis_limit = torch.where(
                self.integral_limit_xyz > 0.0,
                self.integral_limit_xyz,
                torch.full_like(self.integral_limit_xyz, 1.0e12),
            )
            self.integral_state = torch.clamp(self.integral_state, -integral_axis_limit, integral_axis_limit)

        pid_not_initialized = ~self.pid_initialized_state
        if torch.any(pid_not_initialized):
            self.prev_error_state[pid_not_initialized] = error[pid_not_initialized]
            self.prev_velocity_state[pid_not_initialized] = cur[pid_not_initialized]
            self.derivative_state[pid_not_initialized] = 0.0
            self.pid_initialized_state[pid_not_initialized] = True

        raw_derivative = -(cur - self.prev_velocity_state) / self.dt
        alpha = self.derivative_filter_alpha
        self.derivative_state = (1.0 - alpha) * self.derivative_state + alpha * raw_derivative
        self.prev_error_state = error
        self.prev_velocity_state = cur

        kp = self.pid_gains[:, 0].view(1, 3)
        ki = self.pid_gains[:, 1].view(1, 3)
        kd = self.pid_gains[:, 2].view(1, 3)
        tau_pid = kp * error + ki * self.integral_state + kd * self.derivative_state

        ref_dot = (ref - self.prev_ref_state) / self.dt
        self.prev_ref_state = ref
        if self.use_feedforward:
            # Desired steady velocity requires damping compensation at the
            # reference velocity.  Fade it out when the vehicle is already
            # faster than the reference in the same direction, so feedback can
            # brake without fighting an open-loop push.
            if self.use_feedforward_overspeed_gate:
                same_direction_speed = torch.sign(ref) * cur
                overspeed = same_direction_speed - torch.abs(ref)
                tau_ff_gate = torch.clamp(1.0 - overspeed / self.feedforward_overspeed_deadband, 0.0, 1.0)
            else:
                tau_ff_gate = torch.ones_like(tau_pid)
            tau_ff = (
                self.feedforward_accel_gain * self.J * ref_dot
                + self.feedforward_gain * (self.B * ref + self.D * ref * torch.abs(ref))
            ) * tau_ff_gate
            if torch.any(self.feedforward_output_limit_xyz > 0.0):
                ff_axis_limit = torch.where(
                    self.feedforward_output_limit_xyz > 0.0,
                    self.feedforward_output_limit_xyz,
                    torch.full_like(self.feedforward_output_limit_xyz, 1.0e12),
                )
                tau_ff = torch.clamp(tau_ff, -ff_axis_limit, ff_axis_limit)
        else:
            tau_ff = torch.zeros_like(tau_pid)
            tau_ff_gate = torch.zeros_like(tau_pid)

        tau_pid_scaled = self.pid_output_gain * tau_pid
        tau_cmd = tau_pid_scaled + tau_ff - tau_eso
        if self.output_limit > 0.0:
            tau_cmd = torch.clamp(tau_cmd, -self.output_limit, self.output_limit)
        if torch.any(self.output_limit_xyz > 0.0):
            axis_limit = torch.where(
                self.output_limit_xyz > 0.0,
                self.output_limit_xyz,
                torch.full_like(self.output_limit_xyz, self.output_limit),
            )
            tau_cmd = torch.clamp(tau_cmd, -axis_limit, axis_limit)

        tau6 = torch.zeros(ref.shape[0], 6, dtype=ref.dtype, device=ref.device)
        tau6[:, 0] = tau_cmd[:, 0]  # Fx: surge
        tau6[:, 1] = tau_cmd[:, 1]  # Fy: heave
        tau6[:, 2] = tau_cmd[:, 2]  # Fz: sway
        angular_moment = -self.angular_damping_gains * angular
        if self.angular_moment_limit > 0.0:
            angular_moment = torch.clamp(angular_moment, -self.angular_moment_limit, self.angular_moment_limit)
        tau6[:, 3] = angular_moment[:, 0]  # Mx: roll damping
        tau6[:, 4] = angular_moment[:, 1]  # My: yaw damping
        tau6[:, 5] = angular_moment[:, 2]  # Mz: pitch damping
        tau6_for_alloc, allocation_scales = self._prescale_tau_for_allocator(tau6)
        self.prev_tau_cmd_state = tau6_for_alloc[:, 0:3].detach()

        if self.debug_print_interval > 0 and self._debug_step_count % self.debug_print_interval == 0:
            sample = 0
            print(
                f"[VelocityPIDFFESO Step {self._debug_step_count}] "
                f"ref_linear_xyz={ref[sample].detach().cpu().numpy()} "
                f"cur_linear_xyz={cur[sample].detach().cpu().numpy()} "
                f"tau_pid={tau_pid[sample].detach().cpu().numpy()} "
                f"tau_pid_scaled={tau_pid_scaled[sample].detach().cpu().numpy()} "
                f"tau_ff={tau_ff[sample].detach().cpu().numpy()} "
                f"tau_ff_gate={tau_ff_gate[sample].detach().cpu().numpy()} "
                f"tau_eso={tau_eso[sample].detach().cpu().numpy()} "
                f"tau_cmd={tau_cmd[sample].detach().cpu().numpy()} "
                f"angular_xyz={angular[sample].detach().cpu().numpy()} "
                f"angular_moment={angular_moment[sample].detach().cpu().numpy()} "
                f"tau_alloc_linear_xyz={tau6_for_alloc[sample, 0:3].detach().cpu().numpy()} "
                f"tau_alloc_moment_xyz={tau6_for_alloc[sample, 3:6].detach().cpu().numpy()} "
                f"alloc_scale_hv={allocation_scales[sample].detach().cpu().numpy()}",
                flush=True,
            )

        if return_thrust:
            return self.thrust_allocator(tau6_for_alloc)
        return tau6_for_alloc

    def reset(self) -> None:
        self.integral_state.zero_()
        self.prev_error_state.zero_()
        self.derivative_state.zero_()
        self.prev_ref_state.zero_()
        self.prev_velocity_state.zero_()
        self.prev_tau_cmd_state.zero_()
        self.pid_initialized_state.zero_()
        self.eso_z1_state.zero_()
        self.eso_z2_state.zero_()
        self.initialized_state.zero_()


class TraditionalVelocityPIDModel:
    """Small predict-compatible wrapper for evaluation scripts."""

    def __init__(
        self,
        controller: VelocityPIDFeedforwardESO,
        axis_mode: str = AXIS_MODE_LINEAR_XYZ,
        reference_indices: Sequence[int] = (0, 1, 2),
        reference_frame: str = "body",
        observation_format: str = OBS_FORMAT_POSE20,
        linear_velocity_scale: Sequence[float] = DEFAULT_LINEAR_VELOCITY_SCALE,
        angular_velocity_scale: Sequence[float] = DEFAULT_ANGULAR_VELOCITY_SCALE,
        reference_velocity_limit: Sequence[float] = DEFAULT_REFERENCE_VELOCITY_LIMIT,
        reference_velocity_rate_limit: Sequence[float] = DEFAULT_REFERENCE_VELOCITY_RATE_LIMIT,
        action_dim: int = 8,
    ) -> None:
        self.controller = controller
        self.axis_mode = axis_mode
        self.reference_indices = tuple(int(v) for v in reference_indices)
        self.reference_frame = reference_frame
        self.observation_format = observation_format
        self.linear_velocity_scale = tuple(float(v) for v in linear_velocity_scale)
        self.angular_velocity_scale = tuple(float(v) for v in angular_velocity_scale)
        self.reference_velocity_limit = tuple(float(v) for v in reference_velocity_limit)
        self.reference_velocity_rate_limit = tuple(float(v) for v in reference_velocity_rate_limit)
        self.action_dim = int(action_dim)
        self.device = controller.device
        self._filtered_ref_np: Optional[np.ndarray] = None

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        del deterministic
        if episode_start is not None and np.any(episode_start):
            self.reset()

        obs_array = _normalize_obs_array(observation)
        ref_np = extract_reference_velocity(
            obs_array,
            axis_mode=self.axis_mode,
            reference_indices=self.reference_indices,
            reference_frame=self.reference_frame,
            observation_format=self.observation_format,
            linear_velocity_scale=self.linear_velocity_scale,
            angular_velocity_scale=self.angular_velocity_scale,
            default_num_envs=obs_array.shape[0],
        )
        cur_np = extract_current_velocity(
            obs_array,
            axis_mode=self.axis_mode,
            observation_format=self.observation_format,
            linear_velocity_scale=self.linear_velocity_scale,
            angular_velocity_scale=self.angular_velocity_scale,
            default_num_envs=obs_array.shape[0],
        )
        angular_np = extract_current_velocity(
            obs_array,
            axis_mode=AXIS_MODE_ANGULAR_YPR,
            observation_format=self.observation_format,
            linear_velocity_scale=self.linear_velocity_scale,
            angular_velocity_scale=self.angular_velocity_scale,
            default_num_envs=obs_array.shape[0],
        )
        ref_limit = np.asarray(self.reference_velocity_limit, dtype=np.float32).reshape(1, 3)
        ref_np = np.clip(ref_np, -ref_limit, ref_limit)
        rate_limit = np.asarray(self.reference_velocity_rate_limit, dtype=np.float32).reshape(1, 3)
        if np.any(rate_limit > 0.0):
            max_delta = rate_limit * self.controller.dt
            if self._filtered_ref_np is None or self._filtered_ref_np.shape != ref_np.shape:
                self._filtered_ref_np = np.zeros_like(ref_np, dtype=np.float32)
            ref_np = self._filtered_ref_np + np.clip(ref_np - self._filtered_ref_np, -max_delta, max_delta)
            self._filtered_ref_np = ref_np.astype(np.float32, copy=True)

        ref = torch.as_tensor(ref_np, dtype=torch.float32, device=self.controller.J.device)
        cur = torch.as_tensor(cur_np, dtype=torch.float32, device=self.controller.J.device)
        angular = torch.as_tensor(angular_np, dtype=torch.float32, device=self.controller.J.device)
        with torch.no_grad():
            action = self.controller(ref, cur, current_angular_velocity=angular, return_thrust=True).detach().cpu().numpy()

        action = _fit_action_dim(action, self.action_dim)
        if np.asarray(observation).ndim == 1 and action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        return action, state

    def reset(self) -> None:
        self.controller.reset()
        self._filtered_ref_np = None


@dataclass
class TraditionalVelocityPIDTrainingConfig(BaseTrainingConfig):
    """Static parameters for the traditional controller."""

    dt: float = 0.02
    axis_mode: str = AXIS_MODE_LINEAR_XYZ
    observation_format: str = OBS_FORMAT_VELOCITY_NORMALIZED_BODY
    reference_indices: Tuple[int, int, int] = (0, 1, 2)
    reference_frame: str = "normalized_body"
    linear_velocity_scale: Tuple[float, float, float] = DEFAULT_LINEAR_VELOCITY_SCALE
    angular_velocity_scale: Tuple[float, float, float] = DEFAULT_ANGULAR_VELOCITY_SCALE
    reference_velocity_limit: Tuple[float, float, float] = DEFAULT_REFERENCE_VELOCITY_LIMIT
    reference_velocity_rate_limit: Tuple[float, float, float] = DEFAULT_REFERENCE_VELOCITY_RATE_LIMIT
    pid_gains: Tuple[Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]] = (
        DEFAULT_PID_GAINS_LINEAR_XYZ[0],
        DEFAULT_PID_GAINS_LINEAR_XYZ[1],
        DEFAULT_PID_GAINS_LINEAR_XYZ[2],
    )
    inertia: Tuple[float, float, float] = DEFAULT_J_LINEAR_XYZ
    linear_damping: Tuple[float, float, float] = DEFAULT_B_LINEAR_XYZ
    quadratic_damping: Tuple[float, float, float] = DEFAULT_D_LINEAR_XYZ
    output_limit: float = 5.0
    output_limit_xyz: Tuple[float, float, float] = DEFAULT_OUTPUT_LIMIT_XYZ
    integral_limit: float = 20.0
    integral_limit_xyz: Tuple[float, float, float] = DEFAULT_INTEGRAL_LIMIT_XYZ
    pid_output_gain: Tuple[float, float, float] = DEFAULT_PID_OUTPUT_GAIN_XYZ
    derivative_filter_alpha: float = 0.2
    use_feedforward: bool = False
    feedforward_gain: Tuple[float, float, float] = DEFAULT_FEEDFORWARD_GAIN_XYZ
    feedforward_accel_gain: Tuple[float, float, float] = DEFAULT_FEEDFORWARD_ACCEL_GAIN_XYZ
    feedforward_output_limit_xyz: Tuple[float, float, float] = DEFAULT_FEEDFORWARD_OUTPUT_LIMIT_XYZ
    use_feedforward_overspeed_gate: bool = True
    feedforward_overspeed_deadband: float = DEFAULT_FEEDFORWARD_OVERSPEED_DEADBAND
    use_eso: bool = False
    eso_gain: Tuple[float, float, float] = DEFAULT_ESO_GAIN_XYZ
    eso_bandwidth: float = 2.0
    eso_disturbance_accel_limit: float = 50.0
    eso_output_limit_xyz: Tuple[float, float, float] = DEFAULT_ESO_OUTPUT_LIMIT_XYZ
    angular_damping_gains: Tuple[float, float, float] = DEFAULT_ANGULAR_DAMPING_GAINS_XYZ
    angular_moment_limit: float = 0.25
    allocator_control_linear_range: float = 0.5
    allocator_deadzone_comp: float = 0.0
    allocator_allocation_mode: str = "empirical_thruster_mixer"
    allocator_preserve_direction: bool = True
    allocator_saturation_margin: float = 0.95
    allocator_debug_print_interval: int = 200
    debug_print_interval: int = 200


@dataclass
class TraditionalVelocityPIDConfig(BaseConfig):
    """Configuration entry for the non-learning traditional controller."""

    model_type: str = "TRADITIONAL_VELOCITY_PID_FF_ESO"
    requires_checkpoint: bool = False
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: TraditionalVelocityPIDTrainingConfig = field(default_factory=TraditionalVelocityPIDTrainingConfig)

    def create_model(self, env, args, load_checkpoint: Optional[str] = None):
        del env, load_checkpoint
        return self._build_model(device=args.device)

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        del model_path
        del env
        return self._build_model(device=device)

    def _build_model(self, device: str = "auto") -> TraditionalVelocityPIDModel:
        cfg: TraditionalVelocityPIDTrainingConfig = self.training_config  # type: ignore[assignment]
        controller = VelocityPIDFeedforwardESO(
            dt=cfg.dt,
            pid_gains=cfg.pid_gains,
            inertia=cfg.inertia,
            linear_damping=cfg.linear_damping,
            quadratic_damping=cfg.quadratic_damping,
            output_limit=cfg.output_limit,
            output_limit_xyz=cfg.output_limit_xyz,
            integral_limit=cfg.integral_limit,
            integral_limit_xyz=cfg.integral_limit_xyz,
            pid_output_gain=cfg.pid_output_gain,
            derivative_filter_alpha=cfg.derivative_filter_alpha,
            use_feedforward=cfg.use_feedforward,
            feedforward_gain=cfg.feedforward_gain,
            feedforward_accel_gain=cfg.feedforward_accel_gain,
            feedforward_output_limit_xyz=cfg.feedforward_output_limit_xyz,
            use_feedforward_overspeed_gate=cfg.use_feedforward_overspeed_gate,
            feedforward_overspeed_deadband=cfg.feedforward_overspeed_deadband,
            use_eso=cfg.use_eso,
            eso_gain=cfg.eso_gain,
            eso_bandwidth=cfg.eso_bandwidth,
            eso_disturbance_accel_limit=cfg.eso_disturbance_accel_limit,
            eso_output_limit_xyz=cfg.eso_output_limit_xyz,
            angular_damping_gains=cfg.angular_damping_gains,
            angular_moment_limit=cfg.angular_moment_limit,
            device=device,
            allocator_control_linear_range=cfg.allocator_control_linear_range,
            allocator_deadzone_comp=cfg.allocator_deadzone_comp,
            allocator_allocation_mode=cfg.allocator_allocation_mode,
            allocator_preserve_direction=cfg.allocator_preserve_direction,
            allocator_saturation_margin=cfg.allocator_saturation_margin,
            allocator_debug_print_interval=cfg.allocator_debug_print_interval,
            debug_print_interval=cfg.debug_print_interval,
        )
        return TraditionalVelocityPIDModel(
            controller=controller,
            axis_mode=cfg.axis_mode,
            reference_indices=cfg.reference_indices,
            reference_frame=cfg.reference_frame,
            observation_format=cfg.observation_format,
            linear_velocity_scale=cfg.linear_velocity_scale,
            angular_velocity_scale=cfg.angular_velocity_scale,
            reference_velocity_limit=cfg.reference_velocity_limit,
            reference_velocity_rate_limit=cfg.reference_velocity_rate_limit,
            action_dim=8,
        )


def get_traditional_velocity_pid_configs() -> Dict[str, TraditionalVelocityPIDConfig]:
    env_config = BaseEnvironmentConfig(
        time_scale=1.0,
        env_path="/RLControl/build/ControlForVelocity_Decrease.x86_64",
        num_envs=1,
    )
    base_training = TraditionalVelocityPIDTrainingConfig()
    pid_only = replace(
        base_training,
        pid_output_gain=(1.0, 1.0, 1.0),
        derivative_filter_alpha=0.08,
        allocator_control_linear_range=0.35,
    )
    pid_ff = replace(base_training, use_feedforward=True, pid_gains=PID_ASSIST_GAINS_LINEAR_XYZ)
    ff_only = replace(pid_ff, pid_gains=ZERO_PID_GAINS_LINEAR_XYZ, use_feedforward_overspeed_gate=False)
    pid_ff_eso = replace(pid_ff, use_eso=True)

    return {
        "traditional_ff_velocity": TraditionalVelocityPIDConfig(
            name="traditional_ff_velocity",
            model_type="TRADITIONAL_VELOCITY_FF",
            description="Traditional velocity controller: feedforward only + thrust allocator",
            env_config=replace(env_config),
            training_config=ff_only,
        ),
        "traditional_pid_velocity": TraditionalVelocityPIDConfig(
            name="traditional_pid_velocity",
            model_type="TRADITIONAL_VELOCITY_PID",
            description="Traditional velocity controller: PID only + thrust allocator",
            env_config=replace(env_config),
            training_config=pid_only,
        ),
        "traditional_pid_ff_velocity": TraditionalVelocityPIDConfig(
            name="traditional_pid_ff_velocity",
            model_type="TRADITIONAL_VELOCITY_PID_FF",
            description="Traditional velocity controller: PID + scaled feedforward + thrust allocator",
            env_config=replace(env_config),
            training_config=pid_ff,
        ),
        "traditional_pid_ff_eso_velocity": TraditionalVelocityPIDConfig(
            name="traditional_pid_ff_eso_velocity",
            model_type="TRADITIONAL_VELOCITY_PID_FF_ESO",
            description="Traditional velocity controller: PID + scaled feedforward + scaled second-order ESO + thrust allocator",
            env_config=replace(env_config),
            training_config=pid_ff_eso,
        ),
    }


__all__ = [
    "VelocityPIDFeedforwardESO",
    "TraditionalVelocityPIDModel",
    "TraditionalVelocityPIDConfig",
    "get_traditional_velocity_pid_configs",
    "extract_current_velocity",
    "extract_reference_velocity",
    "OBS_FORMAT_POSE20",
    "OBS_FORMAT_VELOCITY_NORMALIZED_BODY",
]
