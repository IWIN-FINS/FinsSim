"""
Traditional position controller: fixed PID gains + thrust allocator.

This controller reuses the same 20D pose observation convention as
`hybrid_ppo_pid_v3.py`, but removes PPO entirely. It directly maps body-frame
position error [surge, heave, sway] through a fixed three-loop PID controller,
adds an independent yaw PID moment from the target/current quaternions, and
outputs 8 thruster commands.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Dict, Optional, Sequence, Tuple, Union

import numpy as np
import torch

from finssim_rl.models.pid_controller import PIDController
from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig


DEFAULT_POSITION_PID_PARAMS: Tuple[float, ...] = (
   5.0, 0.1, 0.5,     # depth
   5.0, 0.3, 0.5,     # surge
   5.0, 0.3, 0.5,     # sway
)
DEFAULT_YAW_PID_PARAMS: Tuple[float, float, float] = (
   1e-3, 0, 0,    # yaw [Kp, Ki, Kd], input unit: deg, output unit: My
)

POSE_ERROR_MASK_6D = np.array([1.0, 1.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
DEFAULT_DERIVATIVE_FILTER_ALPHA = float(np.exp(-4.0))


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


def _normalize_obs_array(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    default_num_envs: int = 1,
    enable_logging: bool = True,
) -> np.ndarray:
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
        if enable_logging and not getattr(_normalize_obs_array, "_warned_bad_shape", False):
            print(f"[TraditionalPositionPID ObsParse WARNING] unsupported obs shape={obs_array.shape}, fallback to zeros")
            setattr(_normalize_obs_array, "_warned_bad_shape", True)
        obs_array = np.zeros((default_num_envs, 20), dtype=np.float32)
    return obs_array


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


def _pos_unity_to_submarine(pos_xyz: np.ndarray) -> np.ndarray:
    pos = np.asarray(pos_xyz, dtype=np.float32)
    # The ROS2 motion_controller already converts physical ROS FLU/z-up state
    # into the Unity-trained controller convention: x forward, y up, z left.
    # Do not flip z here again, otherwise lateral position goals are driven in
    # the opposite direction on the real vehicle.
    return pos.astype(np.float32, copy=False)


def _wrap_to_180(angle_deg: np.ndarray) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=np.float32)
    return ((angle + 180.0) % 360.0 - 180.0).astype(np.float32)


def _quat_to_euler_deg_controller(quat_xyzw: np.ndarray) -> np.ndarray:
    """Convert controller-frame quaternion [x, y, z, w] to [roll, pitch, yaw] in degrees."""
    quat = np.asarray(quat_xyzw, dtype=np.float32)
    if quat.ndim == 1:
        quat = quat.reshape(1, 4)

    x = quat[:, 0]
    y = quat[:, 1]
    z = quat[:, 2]
    w = quat[:, 3]

    sinr = 2.0 * (w * x - y * z)
    sinr = np.clip(sinr, -1.0, 1.0)
    roll = np.arcsin(sinr)

    pitch = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))

    euler_rad = np.stack([roll, pitch, yaw], axis=1)
    return np.degrees(euler_rad).astype(np.float32)


def _target_bearing_body_deg(error_body_xyz: np.ndarray) -> np.ndarray:
    """Signed horizontal target bearing in body frame: 0 front, +right, -left."""
    error = np.asarray(error_body_xyz, dtype=np.float32)
    if error.ndim == 1:
        error = error.reshape(1, 3)
    return np.degrees(np.arctan2(-error[:, 2], error[:, 0])).astype(np.float32)


def extract_pose_error(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    default_num_envs: int = 1,
    enable_logging: bool = True,
) -> np.ndarray:
    """
    Extract 6D pose error.

    For the current position-hold task only x/y/z translation is controlled,
    while attitude error stays zeroed out.
    """
    obs_array = _normalize_obs_array(obs, default_num_envs, enable_logging=enable_logging)
    batch_size, obs_dim = obs_array.shape

    if obs_dim >= 14:
        target_xyz_unity = obs_array[:, 0:3]
        current_xyz_unity = obs_array[:, 7:10]
        current_quat = obs_array[:, 10:14]
        pos_diff_unity = target_xyz_unity - current_xyz_unity
        pos_error_unity_local = _quat_rotate_inverse(current_quat, pos_diff_unity)
        pos_error_local = _pos_unity_to_submarine(pos_error_unity_local)
        zero_attitude_error = np.zeros((batch_size, 3), dtype=np.float32)
        error6 = np.concatenate([pos_error_local, zero_attitude_error], axis=1).astype(np.float32)
        return (error6 * POSE_ERROR_MASK_6D.reshape(1, -1)).astype(np.float32)

    target_xyz = np.tile(np.array([0.0, -3.0, 0.0], dtype=np.float32), (batch_size, 1))
    current_xyz = np.zeros((batch_size, 3), dtype=np.float32)
    use_dim = min(obs_dim, 3)
    if use_dim > 0:
        current_xyz[:, :use_dim] = obs_array[:, :use_dim]
    pos_error = target_xyz - current_xyz
    zero_attitude_error = np.zeros((batch_size, 3), dtype=np.float32)
    error6 = np.concatenate([pos_error, zero_attitude_error], axis=1).astype(np.float32)
    return (error6 * POSE_ERROR_MASK_6D.reshape(1, -1)).astype(np.float32)


def extract_target_yaw_error_deg(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    default_num_envs: int = 1,
    enable_logging: bool = True,
) -> np.ndarray:
    """
    Extract world-frame yaw error in controller coordinates.

    pose20 convention:
      [0:3]   target position
      [3:7]   target quaternion xyzw
      [7:10]  current position
      [10:14] current quaternion xyzw
    """
    obs_array = _normalize_obs_array(obs, default_num_envs, enable_logging=enable_logging)
    batch_size, obs_dim = obs_array.shape

    if obs_dim >= 14:
        target_quat = obs_array[:, 3:7]
        current_quat = obs_array[:, 10:14]
        target_yaw_deg = _quat_to_euler_deg_controller(target_quat)[:, 2]
        current_yaw_deg = _quat_to_euler_deg_controller(current_quat)[:, 2]
        return _wrap_to_180(target_yaw_deg - current_yaw_deg).astype(np.float32)

    return np.zeros((batch_size,), dtype=np.float32)


def build_error4_from_obs(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    default_num_envs: int = 1,
    enable_logging: bool = True,
) -> np.ndarray:
    """Build [x_error, y_error, z_error, yaw_error_deg] for PIDController."""
    error6 = extract_pose_error(obs, default_num_envs=default_num_envs, enable_logging=enable_logging)
    yaw_error_deg = extract_target_yaw_error_deg(
        obs,
        default_num_envs=default_num_envs,
        enable_logging=enable_logging,
    ).reshape(-1, 1)
    return np.concatenate([error6[:, 0:3], yaw_error_deg], axis=1).astype(np.float32)


def build_error4_from_body_position_error(
    body_position_error: np.ndarray,
    yaw_error_deg: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Build PID input from direct controller_body [forward, up, left] position error."""
    error = np.nan_to_num(np.asarray(body_position_error, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if error.ndim == 1:
        error = error.reshape(1, -1)
    if error.ndim != 2 or error.shape[1] < 3:
        raise ValueError(f"body_position_error must have shape (*, >=3), got {error.shape}")

    batch_size = error.shape[0]
    if yaw_error_deg is None:
        yaw = np.zeros((batch_size, 1), dtype=np.float32)
    else:
        yaw = np.nan_to_num(np.asarray(yaw_error_deg, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
        yaw = yaw.reshape(-1, 1)
        if yaw.shape[0] == 1 and batch_size > 1:
            yaw = np.repeat(yaw, batch_size, axis=0)
        if yaw.shape[0] != batch_size:
            raise ValueError(f"yaw_error_deg batch {yaw.shape[0]} does not match body error batch {batch_size}")

    return np.concatenate([error[:, 0:3], yaw], axis=1).astype(np.float32)


def extract_raw_position_error(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    default_num_envs: int = 1,
    enable_logging: bool = True,
) -> np.ndarray:
    """Extract world-frame position error for debugging prints in PIDController."""
    obs_array = _normalize_obs_array(obs, default_num_envs, enable_logging=enable_logging)
    batch_size, obs_dim = obs_array.shape

    if obs_dim >= 14:
        return (obs_array[:, 0:3] - obs_array[:, 7:10]).astype(np.float32)

    target_xyz = np.tile(np.array([0.0, -3.0, 0.0], dtype=np.float32), (batch_size, 1))
    current_xyz = np.zeros((batch_size, 3), dtype=np.float32)
    use_dim = min(obs_dim, 3)
    if use_dim > 0:
        current_xyz[:, :use_dim] = obs_array[:, :use_dim]
    return (target_xyz - current_xyz).astype(np.float32)


def extract_target_and_current_pose_debug(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
    default_num_envs: int = 1,
    enable_logging: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray, np.ndarray]:
    """Return target/current position debug values and yaw in degrees."""
    obs_array = _normalize_obs_array(obs, default_num_envs, enable_logging=enable_logging)
    batch_size, obs_dim = obs_array.shape

    if obs_dim >= 14:
        target_xyz = obs_array[:, 0:3].astype(np.float32)
        current_xyz_controller = obs_array[:, 7:10].astype(np.float32)
        current_xyz_ros = obs_array[:, 20:23].astype(np.float32) if obs_dim >= 23 else None
        target_yaw_deg = _quat_to_euler_deg_controller(obs_array[:, 3:7])[:, 2].astype(np.float32)
        current_yaw_deg = _quat_to_euler_deg_controller(obs_array[:, 10:14])[:, 2].astype(np.float32)
        return target_xyz, current_xyz_controller, current_xyz_ros, target_yaw_deg, current_yaw_deg

    target_xyz = np.tile(np.array([0.0, -3.0, 0.0], dtype=np.float32), (batch_size, 1))
    current_xyz_controller = np.zeros((batch_size, 3), dtype=np.float32)
    use_dim = min(obs_dim, 3)
    if use_dim > 0:
        current_xyz_controller[:, :use_dim] = obs_array[:, :use_dim]
    target_yaw_deg = np.zeros((batch_size,), dtype=np.float32)
    current_yaw_deg = np.zeros((batch_size,), dtype=np.float32)
    return target_xyz, current_xyz_controller, None, target_yaw_deg, current_yaw_deg


class TraditionalPositionPIDModel:
    """Predict-compatible wrapper around the fixed-gain position PID controller."""

    def __init__(
        self,
        controller: PIDController,
        pid_params: Sequence[float] = DEFAULT_POSITION_PID_PARAMS,
        yaw_pid_params: Sequence[float] = DEFAULT_YAW_PID_PARAMS,
        enable_yaw_control: bool = True,
        action_dim: int = 8,
        enable_logging: bool = True,
    ) -> None:
        self.controller = controller
        self.action_dim = int(action_dim)
        self.device = controller.device
        self.enable_yaw_control = bool(enable_yaw_control)
        self.enable_logging = bool(enable_logging)
        if not self.enable_logging:
            self.controller.debug_print_interval = 0
        pid_params_np = np.asarray(tuple(float(v) for v in pid_params), dtype=np.float32).reshape(-1)
        if pid_params_np.shape[0] != PIDController.PARAM_DIM:
            raise ValueError(
                f"Expected {PIDController.PARAM_DIM} PID parameters, got {pid_params_np.shape[0]}"
            )
        self.pid_params_np = pid_params_np.astype(np.float32, copy=False)
        yaw_pid_params_np = np.asarray(tuple(float(v) for v in yaw_pid_params), dtype=np.float32).reshape(-1)
        if yaw_pid_params_np.shape[0] != 3:
            raise ValueError(f"Expected 3 yaw PID parameters [Kp, Ki, Kd], got {yaw_pid_params_np.shape[0]}")
        self.yaw_pid_params_np = yaw_pid_params_np.astype(np.float32, copy=False)
        self._yaw_integral_np = np.zeros((0,), dtype=np.float32)
        self._yaw_prev_error_np = np.zeros((0,), dtype=np.float32)
        self._yaw_derivative_filter_np = np.zeros((0,), dtype=np.float32)
        self._latest_yaw_debug: dict[str, np.ndarray] = {}

    def _ensure_yaw_state(self, batch_size: int) -> None:
        if self._yaw_integral_np.shape[0] == batch_size:
            return
        self._yaw_integral_np = np.zeros((batch_size,), dtype=np.float32)
        self._yaw_prev_error_np = np.zeros((batch_size,), dtype=np.float32)
        self._yaw_derivative_filter_np = np.zeros((batch_size,), dtype=np.float32)

    def _compute_yaw_moment(self, yaw_error_deg: np.ndarray) -> np.ndarray:
        batch_size = int(yaw_error_deg.shape[0])
        self._ensure_yaw_state(batch_size)

        if not self.enable_yaw_control:
            zeros = np.zeros((batch_size,), dtype=np.float32)
            self._yaw_integral_np.fill(0.0)
            self._yaw_prev_error_np.fill(0.0)
            self._yaw_derivative_filter_np.fill(0.0)
            self._latest_yaw_debug = {
                "error_deg": zeros,
                "integral": zeros,
                "derivative": zeros,
                "p_term": zeros,
                "i_term": zeros,
                "d_term": zeros,
                "moment": zeros,
            }
            return zeros

        error = np.asarray(yaw_error_deg, dtype=np.float32).reshape(batch_size)
        deadband = float(max(self.controller.yaw_deadband_deg, 0.0))
        if deadband > 0.0:
            error = np.where(np.abs(error) <= deadband, 0.0, error).astype(np.float32)

        kp, ki, kd = self.yaw_pid_params_np.tolist()
        dt = max(float(self.controller.dt), 1e-6)

        decayed_integral = self._yaw_integral_np * float(self.controller.integral_decay)
        yaw_integral_limit = float(max(self.controller.yaw_integral_state_limit, 0.0))
        if yaw_integral_limit > 0.0:
            decayed_integral = np.clip(decayed_integral, -yaw_integral_limit, yaw_integral_limit).astype(np.float32)
        candidate_integral = decayed_integral + error * dt
        if yaw_integral_limit > 0.0:
            candidate_integral = np.clip(candidate_integral, -yaw_integral_limit, yaw_integral_limit).astype(np.float32)

        raw_derivative = (error - self._yaw_prev_error_np) / dt
        alpha = float(np.clip(self.controller.derivative_filter_alpha, 0.0, 1.0))
        filtered_derivative = (
            (1.0 - alpha) * self._yaw_derivative_filter_np
            + alpha * raw_derivative
        ).astype(np.float32)

        self._yaw_derivative_filter_np = filtered_derivative
        self._yaw_prev_error_np = error.astype(np.float32, copy=False)

        p_term = (kp * error).astype(np.float32)
        candidate_i_term = (ki * candidate_integral).astype(np.float32)
        d_term = (kd * filtered_derivative).astype(np.float32)
        candidate_yaw_moment = (p_term + candidate_i_term + d_term).astype(np.float32)

        yaw_output_limit = float(max(self.controller.yaw_output_limit, 0.0))
        if yaw_output_limit > 0.0:
            integral_push = ki * error
            blocked = (
                ((candidate_yaw_moment > yaw_output_limit) & (integral_push > 0.0))
                | ((candidate_yaw_moment < -yaw_output_limit) & (integral_push < 0.0))
            )
            self._yaw_integral_np = np.where(blocked, decayed_integral, candidate_integral).astype(np.float32)
        else:
            self._yaw_integral_np = candidate_integral.astype(np.float32)

        i_term = (ki * self._yaw_integral_np).astype(np.float32)
        yaw_moment = (
            p_term
            + i_term
            + d_term
        ).astype(np.float32)

        if yaw_output_limit > 0.0:
            yaw_moment = np.clip(yaw_moment, -yaw_output_limit, yaw_output_limit).astype(np.float32)

        self._latest_yaw_debug = {
            "error_deg": error.astype(np.float32, copy=False),
            "integral": self._yaw_integral_np.astype(np.float32, copy=False),
            "derivative": filtered_derivative.astype(np.float32, copy=False),
            "p_term": p_term.astype(np.float32, copy=False),
            "i_term": i_term.astype(np.float32, copy=False),
            "d_term": d_term.astype(np.float32, copy=False),
            "moment": yaw_moment.astype(np.float32, copy=False),
        }
        return yaw_moment

    def predict_from_body_position_error(
        self,
        body_position_error: np.ndarray,
        yaw_error_deg: Optional[np.ndarray] = None,
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = True,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Run PID directly from controller_body position error [forward, up, left]."""
        del deterministic
        if episode_start is not None and np.any(episode_start):
            self.reset()

        input_is_vector = isinstance(body_position_error, np.ndarray) and body_position_error.ndim == 1
        error4_np = build_error4_from_body_position_error(
            body_position_error,
            yaw_error_deg=yaw_error_deg,
        )
        batch_size = error4_np.shape[0]
        raw_error_np = error4_np[:, 0:3].astype(np.float32, copy=False)
        pid_params_np = np.repeat(self.pid_params_np.reshape(1, -1), batch_size, axis=0).astype(np.float32, copy=False)
        yaw_moment_np = self._compute_yaw_moment(error4_np[:, 3])

        device = self.controller.attitude_coupling.device
        error4 = torch.as_tensor(error4_np, dtype=torch.float32, device=device)
        raw_error = torch.as_tensor(raw_error_np, dtype=torch.float32, device=device)
        pid_params = torch.as_tensor(pid_params_np, dtype=torch.float32, device=device)
        yaw_moment = torch.as_tensor(yaw_moment_np, dtype=torch.float32, device=device)

        with torch.no_grad():
            control_6d = self.controller(
                error4,
                pid_params,
                return_thrust=False,
                raw_error=raw_error,
                apply_slew_rate_limit=False,
            )
            control_6d[:, 4] = control_6d[:, 4] + yaw_moment
            control_6d = torch.clamp(
                control_6d,
                float(self.controller.output_range[0]),
                float(self.controller.output_range[1]),
            )
            control_6d = self.controller._apply_control_slew_rate_limit(control_6d)
            if self.controller.use_thrust_allocator:
                action = self.controller.thrust_allocator(control_6d).detach().cpu().numpy()
            else:
                action = control_6d.detach().cpu().numpy()

        action = _fit_action_dim(action, self.action_dim)
        if input_is_vector and action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        return action, state

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

        observation_is_vector = isinstance(observation, np.ndarray) and observation.ndim == 1
        obs_array = _normalize_obs_array(observation, enable_logging=self.enable_logging)
        batch_size = obs_array.shape[0]
        error4_np = build_error4_from_obs(
            obs_array,
            default_num_envs=batch_size,
            enable_logging=self.enable_logging,
        )
        raw_error_np = extract_raw_position_error(
            obs_array,
            default_num_envs=batch_size,
            enable_logging=self.enable_logging,
        )
        (
            target_xyz_np,
            current_xyz_controller_np,
            current_xyz_ros_np,
            target_yaw_deg_np,
            current_yaw_deg_np,
        ) = extract_target_and_current_pose_debug(
            obs_array,
            default_num_envs=batch_size,
            enable_logging=self.enable_logging,
        )
        pid_params_np = np.repeat(self.pid_params_np.reshape(1, -1), batch_size, axis=0).astype(np.float32, copy=False)
        yaw_moment_np = self._compute_yaw_moment(error4_np[:, 3])

        error4 = torch.as_tensor(error4_np, dtype=torch.float32, device=self.controller.attitude_coupling.device)
        raw_error = torch.as_tensor(raw_error_np, dtype=torch.float32, device=self.controller.attitude_coupling.device)
        pid_params = torch.as_tensor(pid_params_np, dtype=torch.float32, device=self.controller.attitude_coupling.device)
        yaw_moment = torch.as_tensor(yaw_moment_np, dtype=torch.float32, device=self.controller.attitude_coupling.device)

        with torch.no_grad():
            control_6d = self.controller(
                error4,
                pid_params,
                return_thrust=False,
                raw_error=raw_error,
                apply_slew_rate_limit=False,
            )
            control_6d[:, 4] = control_6d[:, 4] + yaw_moment
            control_6d = torch.clamp(
                control_6d,
                float(self.controller.output_range[0]),
                float(self.controller.output_range[1]),
            )
            control_6d = self.controller._apply_control_slew_rate_limit(control_6d)
            if self.controller.use_thrust_allocator:
                action = self.controller.thrust_allocator(control_6d).detach().cpu().numpy()
            else:
                action = control_6d.detach().cpu().numpy()

        if (
            self.enable_logging
            and self.controller.debug_print_interval > 0
            and self.controller._debug_step_count > 0
            and self.controller._debug_step_count % self.controller.debug_print_interval == 0
        ):
            batch_idx = 0
            yaw_debug = self._latest_yaw_debug
            yaw_kp, yaw_ki, yaw_kd = self.yaw_pid_params_np.tolist()
            target_bearing_body_deg = _target_bearing_body_deg(error4_np[:, 0:3])
            current_xyz_ros_text = (
                f"\n current_xyz_ros={current_xyz_ros_np[batch_idx]}"
                if current_xyz_ros_np is not None
                else ""
            )
            print(
                f"[TraditionalPositionPID INFO] \n"
                f"target_xyz={target_xyz_np[batch_idx]} \n"
                f"current_xyz_controller={current_xyz_controller_np[batch_idx]}"
                f"{current_xyz_ros_text} \n"
                f"error_world_xyz={raw_error_np[batch_idx]} \n error_body_xyz={error4_np[batch_idx, 0:3]} \n"
                f"target_bearing_body_deg={float(target_bearing_body_deg[batch_idx]):+.2f} \n"
                f"target_yaw_deg={float(target_yaw_deg_np[batch_idx]):+.2f} \n"
                f"current_yaw_deg={float(current_yaw_deg_np[batch_idx]):+.2f} \n"
                f"yaw_control={'on' if self.enable_yaw_control else 'off'} \n"
                f"yaw_error_deg={float(yaw_debug.get('error_deg', np.zeros(1, dtype=np.float32))[batch_idx]):+.2f} \n"
                f"yaw_pid[Kp={yaw_kp:.6g}, Ki={yaw_ki:.6g}, Kd={yaw_kd:.6g}] \n"
                f"yaw_terms[P={float(yaw_debug.get('p_term', np.zeros(1, dtype=np.float32))[batch_idx]):+.4f}, \n"
                f"I={float(yaw_debug.get('i_term', np.zeros(1, dtype=np.float32))[batch_idx]):+.4f}, \n"
                f"D={float(yaw_debug.get('d_term', np.zeros(1, dtype=np.float32))[batch_idx]):+.4f}] \n"
                f"yaw_integral={float(yaw_debug.get('integral', np.zeros(1, dtype=np.float32))[batch_idx]):+.4f} \n"
                f"yaw_derivative={float(yaw_debug.get('derivative', np.zeros(1, dtype=np.float32))[batch_idx]):+.4f} \n"
                f"yaw_moment={float(yaw_debug.get('moment', np.zeros(1, dtype=np.float32))[batch_idx]):+.4f}",
                flush=True,
            )

        action = _fit_action_dim(action, self.action_dim)
        if observation_is_vector and action.ndim == 2 and action.shape[0] == 1:
            action = action[0]
        return action, state

    def reset(self) -> None:
        self.controller.reset()
        self._yaw_integral_np = np.zeros((0,), dtype=np.float32)
        self._yaw_prev_error_np = np.zeros((0,), dtype=np.float32)
        self._yaw_derivative_filter_np = np.zeros((0,), dtype=np.float32)
        self._latest_yaw_debug = {}


@dataclass
class TraditionalPositionPIDTrainingConfig(BaseTrainingConfig):
    """Static parameters for the fixed-gain position PID controller."""

    dt: float = 0.02
    pid_params: Tuple[float, float, float, float, float, float, float, float, float] = DEFAULT_POSITION_PID_PARAMS
    yaw_pid_params: Tuple[float, float, float] = DEFAULT_YAW_PID_PARAMS
    enable_yaw_control: bool = True
    output_range: Tuple[float, float] = (-250.0, 250.0)
    surge_output_limit: float = 120.0
    sway_output_limit: float = 120.0
    yaw_deadband_deg: float = 0.5
    yaw_output_limit: float = 40.0
    integral_state_limit: float = 100.0
    yaw_integral_state_limit: float = 60.0
    derivative_filter_alpha: float = DEFAULT_DERIVATIVE_FILTER_ALPHA
    output_slew_rate_limit: float = 500.0
    allocator_control_linear_range: float = 10.0
    allocator_control_axis_ranges: Tuple[float, float, float, float, float, float] = ()
    allocator_allocation_mode: str = "empirical_thruster_mixer"
    # These are used only by physical_wrench_allocator.  Keep the empty/default
    # values for legacy empirical profiles, whose allocator does not consume
    # physical force limits.
    allocator_physical_wrench_limits: Tuple[float, float, float, float, float, float] = ()
    allocator_thruster_force_limit_positive: Tuple[float, float, float, float, float, float, float, float] = ()
    allocator_thruster_force_limit_negative: Tuple[float, float, float, float, float, float, float, float] = ()
    integral_decay: float = 0.95
    debug_print_interval: int = 200
    enable_logging: bool = True


@dataclass
class TraditionalPositionPIDConfig(BaseConfig):
    """Configuration entry for the non-learning traditional position PID controller."""

    model_type: str = "TRADITIONAL_POSITION_PID"
    requires_checkpoint: bool = False
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: TraditionalPositionPIDTrainingConfig = field(default_factory=TraditionalPositionPIDTrainingConfig)

    def create_model(self, env, args, load_checkpoint: Optional[str] = None):
        del env, load_checkpoint
        return self._build_model(device=args.device)

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        del model_path
        del env
        return self._build_model(device=device)

    def _build_model(self, device: str = "auto") -> TraditionalPositionPIDModel:
        cfg: TraditionalPositionPIDTrainingConfig = self.training_config  # type: ignore[assignment]
        controller = PIDController(
            action_dim=8,
            device=device,
            dt=cfg.dt,
            integral_decay=cfg.integral_decay,
            output_range=cfg.output_range,
            yaw_deadband_deg=cfg.yaw_deadband_deg,
            yaw_output_limit=cfg.yaw_output_limit,
            yaw_integral_state_limit=cfg.yaw_integral_state_limit,
            surge_output_limit=cfg.surge_output_limit,
            sway_output_limit=cfg.sway_output_limit,
            integral_state_limit=cfg.integral_state_limit,
            derivative_filter_alpha=cfg.derivative_filter_alpha,
            output_slew_rate_limit=cfg.output_slew_rate_limit,
            allocator_control_linear_range=cfg.allocator_control_linear_range,
            allocator_control_axis_ranges=cfg.allocator_control_axis_ranges or None,
            allocator_allocation_mode=cfg.allocator_allocation_mode,
            allocator_physical_wrench_limits=cfg.allocator_physical_wrench_limits or None,
            allocator_thruster_force_limit_positive=cfg.allocator_thruster_force_limit_positive or None,
            allocator_thruster_force_limit_negative=cfg.allocator_thruster_force_limit_negative or None,
            debug_print_interval=cfg.debug_print_interval if cfg.enable_logging else 0,
        )
        return TraditionalPositionPIDModel(
            controller=controller,
            pid_params=cfg.pid_params,
            yaw_pid_params=cfg.yaw_pid_params,
            enable_yaw_control=cfg.enable_yaw_control,
            action_dim=8,
            enable_logging=cfg.enable_logging,
        )


def get_traditional_position_pid_configs() -> Dict[str, TraditionalPositionPIDConfig]:
    env_config = BaseEnvironmentConfig(
        time_scale=1.0,
        env_path="/RLControl/build/ControlForPosition.x86_64",
        num_envs=1,
    )
    base_training = TraditionalPositionPIDTrainingConfig()
    conservative = replace(
        base_training,
        debug_print_interval=200,
    )

    return {
        "traditional_pid_position": TraditionalPositionPIDConfig(
            name="traditional_pid_position",
            model_type="TRADITIONAL_POSITION_PID",
            description="Traditional position controller: fixed PID gains + thrust allocator",
            env_config=replace(env_config),
            training_config=conservative,
        ),
    }


__all__ = [
    "TraditionalPositionPIDModel",
    "TraditionalPositionPIDConfig",
    "TraditionalPositionPIDTrainingConfig",
    "get_traditional_position_pid_configs",
    "build_error4_from_obs",
    "build_error4_from_body_position_error",
    "extract_pose_error",
    "extract_raw_position_error",
]
