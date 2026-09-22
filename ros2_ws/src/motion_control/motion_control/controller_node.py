from __future__ import annotations

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
from typing import Sequence
import traceback

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Time as TimeMessage
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from rosgraph_msgs.msg import Clock
from std_msgs.msg import Bool, Empty, Float32MultiArray, String, UInt64
from trajectory_msgs.msg import MultiDOFJointTrajectory
from msgs.msg import StampedJson

from .backend_loader import load_backend
from .math_utils import (
    inverse_rotate_vector,
    quat_angle_error_rad,
    quat_from_controller_ypr,
    quat_multiply,
    quat_normalize,
    reorder_vector,
    rotate_vector,
    transform_quat_basis,
)
from .observations import (
    build_pose13_observation,
    build_pose14_observation,
    build_pose16_rot6d_observation,
    build_pose20_observation,
    build_velocity_normalized_observation,
)
from .state_estimator import VehicleStateEstimator
from .trajectory_tracking import TrajectoryPoint, build_trajectory30_observation


@dataclass
class MotionCommand:
    mode: str
    source_frame: str
    issued_at_sec: float
    target_position_world: np.ndarray | None = None
    target_orientation_world: np.ndarray | None = None
    desired_linear_velocity_body: np.ndarray | None = None
    desired_angular_velocity_ypr: np.ndarray | None = None
    trajectory_points: tuple[TrajectoryPoint, ...] | None = None
    trajectory_duration_sec: float | None = None


CONTROL_MODE_ROS_MANUAL = "ros_manual"
CONTROL_MODE_UNITY_RANDOM = "unity_random"
CONTROL_MODES = {CONTROL_MODE_ROS_MANUAL, CONTROL_MODE_UNITY_RANDOM}
THRUSTER_OUTPUT_FORCE_N = "force_n"
THRUSTER_OUTPUT_NORMALIZED_DIRECT = "normalized_direct"
THRUSTER_OUTPUT_WRENCH6D_FORCE_N = "wrench6d_force_n"
THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N = "isaaclab_virtual_wrench_force_n"
THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N = "isaaclab_finsrov_calibrated_thruster8_force_n"
THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N = (
    "thruster8_reprojected_physical_wrench_force_n"
)
THRUSTER_OUTPUT_MODES = {
    THRUSTER_OUTPUT_FORCE_N,
    THRUSTER_OUTPUT_NORMALIZED_DIRECT,
    THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
    THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N,
    THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N,
    THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N,
}

WRENCH6D_POLICY_DIM = 6
THRUSTER_COUNT = 8
DEFAULT_WRENCH6D_WRENCH_LIMITS = (1.0,) * WRENCH6D_POLICY_DIM
DEFAULT_WRENCH6D_SCALE = (1.0,) * WRENCH6D_POLICY_DIM
DEFAULT_WRENCH6D_ALLOCATOR_CONTROL_LINEAR_RANGE = 10.0
DEFAULT_WRENCH6D_ALLOCATOR_ALLOCATION_MODE = "empirical_thruster_mixer"


def _normalize_control_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "ros": CONTROL_MODE_ROS_MANUAL,
        "manual": CONTROL_MODE_ROS_MANUAL,
        "ros_manual": CONTROL_MODE_ROS_MANUAL,
        "unity": CONTROL_MODE_UNITY_RANDOM,
        "random": CONTROL_MODE_UNITY_RANDOM,
        "unity_random": CONTROL_MODE_UNITY_RANDOM,
    }
    if mode not in aliases:
        raise ValueError(f"unknown control_mode `{value}`; expected one of {sorted(CONTROL_MODES)}")
    return aliases[mode]


def _normalize_thruster_output_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "force": THRUSTER_OUTPUT_FORCE_N,
        "force_n": THRUSTER_OUTPUT_FORCE_N,
        "normalized": THRUSTER_OUTPUT_NORMALIZED_DIRECT,
        "normalized_direct": THRUSTER_OUTPUT_NORMALIZED_DIRECT,
        "direct": THRUSTER_OUTPUT_NORMALIZED_DIRECT,
        "pwm": THRUSTER_OUTPUT_NORMALIZED_DIRECT,
        "wrench6d": THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
        "wrench_6d": THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
        "wrench6d_force": THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
        "wrench6d_force_n": THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
        "policy_wrench6d_force_n": THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
        "isaaclab_virtual_wrench": THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N,
        "isaaclab_virtual_wrench_force_n": THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N,
        "isaaclab_finsrov_calibrated_thruster8": THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N,
        "isaaclab_finsrov_calibrated_thruster8_force_n": THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N,
        "thruster8_reprojected_physical_wrench": THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N,
        "thruster8_reprojected_physical_wrench_force_n": THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N,
    }
    if mode not in aliases:
        raise ValueError(f"unknown thruster_output_mode `{value}`; expected one of {sorted(THRUSTER_OUTPUT_MODES)}")
    return aliases[mode]


def action_to_thruster_command(
    action: Sequence[float],
    *,
    mode: str,
    force_limit_positive: Sequence[float],
    force_limit_negative: Sequence[float],
) -> np.ndarray:
    normalized_mode = _normalize_thruster_output_mode(mode)
    if normalized_mode in {
        THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
        THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N,
        THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N,
    }:
        raise ValueError(
            "output modes with custom force conversion must be mapped by MotionControllerNode; "
            "call MotionControllerNode._action_to_thruster_command() instead."
        )

    action_array = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if action_array.shape[0] < THRUSTER_COUNT:
        action_array = np.pad(action_array, (0, THRUSTER_COUNT - action_array.shape[0]), constant_values=0.0)
    elif action_array.shape[0] > THRUSTER_COUNT:
        action_array = action_array[:THRUSTER_COUNT]
    if normalized_mode == THRUSTER_OUTPUT_NORMALIZED_DIRECT:
        return action_array.astype(np.float32, copy=False)
    positive_limit = np.abs(np.asarray(force_limit_positive, dtype=np.float32).reshape(8))
    negative_limit = np.abs(np.asarray(force_limit_negative, dtype=np.float32).reshape(8))
    positive = np.maximum(action_array, 0.0) * positive_limit
    negative = np.minimum(action_array, 0.0) * negative_limit
    return (positive + negative).astype(np.float32, copy=False)


def isaaclab_finsrov_calibrated_action_to_force_n(
    action: Sequence[float],
    *,
    force_limit_positive: Sequence[float],
    force_limit_negative: Sequence[float],
) -> np.ndarray:
    """Apply FinsROV's calibrated normalized-action contract exactly.

    Each policy action is clipped to ``[-1, 1]`` and scaled by its matching
    calibrated positive or negative force magnitude. This matches the IsaacLab
    task's ``F_i = a_i * Fmax_i`` actuator contract.
    """

    action_array = np.asarray(action, dtype=np.float32).reshape(-1)
    if action_array.shape[0] != THRUSTER_COUNT:
        raise ValueError(
            f"isaaclab_finsrov_calibrated_thruster8_force_n requires {THRUSTER_COUNT} actions, "
            f"got {action_array.shape[0]}"
        )
    positive_limit = np.abs(np.asarray(force_limit_positive, dtype=np.float32).reshape(-1))
    negative_limit = np.abs(np.asarray(force_limit_negative, dtype=np.float32).reshape(-1))
    if positive_limit.shape[0] != THRUSTER_COUNT or negative_limit.shape[0] != THRUSTER_COUNT:
        raise ValueError(f"calibrated FinsROV force limits must each contain {THRUSTER_COUNT} values")
    if not np.all(np.isfinite(positive_limit)) or not np.all(np.isfinite(negative_limit)):
        raise ValueError("calibrated FinsROV force limits must be finite")
    if np.any(positive_limit <= 0.0) or np.any(negative_limit <= 0.0):
        raise ValueError("calibrated FinsROV force limits must be positive magnitudes")
    normalized_action = np.clip(action_array, -1.0, 1.0)
    return np.where(
        normalized_action >= 0.0,
        normalized_action * positive_limit,
        normalized_action * negative_limit,
    ).astype(np.float32, copy=False)


def policy_action6d_to_allocator_input(
    action: Sequence[float],
    *,
    wrench_limits: Sequence[float] = DEFAULT_WRENCH6D_WRENCH_LIMITS,
    wrench_scale: Sequence[float] = DEFAULT_WRENCH6D_SCALE,
    policy_axis_order: str = "legacy_surge_sway_heave",
) -> np.ndarray:
    """Map normalized policy output into a physical body-wrench target.

    ``legacy_surge_sway_heave`` policy order is
    [surge, sway, heave, roll, pitch, yaw]. ``allocator_body`` is the T2
    contract and already has [Fx, Fy, Fz, Mx, My, Mz] ordering.
    Allocator-body order is [Fx, Fy, Fz, Mx, My, Mz] where the axes are
    Unity-compatible FinsROV body axes [forward, up, left] and rotations
    [roll-about-x, yaw-about-y, pitch-about-z]. This is not controller_world
    and is not the standard Fossen [X, Y, Z, K, M, N] ordering.  ``wrench_limits``
    is in the selected policy order and has units [N, N, N, N*m, N*m, N*m].  ``wrench_scale``
    applies a per-axis multiplier directly to the normalized policy action
    before applying ``wrench_limits``. The legacy ``matrix`` allocator still
    accepts this result as its old dimensionless input for backwards
    compatibility only.
    """

    action_array = np.clip(np.asarray(action, dtype=np.float32).reshape(-1), -1.0, 1.0)
    if action_array.shape[0] != WRENCH6D_POLICY_DIM:
        raise ValueError(f"wrench6d policy action must contain 6 values, got {action_array.shape[0]}")
    scale = np.asarray(wrench_scale, dtype=np.float32).reshape(WRENCH6D_POLICY_DIM)
    if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
        raise ValueError("wrench_scale must contain 6 finite non-negative values")
    limits = np.abs(np.asarray(wrench_limits, dtype=np.float32).reshape(WRENCH6D_POLICY_DIM))
    scaled_action = (action_array * scale) * limits
    normalized_axis_order = str(policy_axis_order).strip().lower().replace("-", "_")
    if normalized_axis_order in {"allocator_body", "physical_allocator_body"}:
        return scaled_action.astype(np.float32, copy=False)
    if normalized_axis_order not in {"legacy_surge_sway_heave", "legacy", "surge_sway_heave"}:
        raise ValueError(
            "wrench6d.policy_axis_order must be `legacy_surge_sway_heave` or `allocator_body`"
        )
    return np.asarray(
        [
            scaled_action[0],  # Fx = surge
            scaled_action[2],  # Fy = heave
            scaled_action[1],  # Fz = sway
            scaled_action[3],  # Mx = roll
            scaled_action[5],  # My = yaw
            scaled_action[4],  # Mz = pitch
        ],
        dtype=np.float32,
    )


def scale_allocator_body_wrench(
    wrench: Sequence[float],
    *,
    wrench_scale: Sequence[float] = DEFAULT_WRENCH6D_SCALE,
    policy_axis_order: str = "legacy_surge_sway_heave",
) -> np.ndarray:
    """Scale a physical body wrench using the configured policy-axis contract."""

    wrench_array = np.asarray(wrench, dtype=np.float32).reshape(WRENCH6D_POLICY_DIM)
    scale = np.asarray(wrench_scale, dtype=np.float32).reshape(WRENCH6D_POLICY_DIM)
    if not np.all(np.isfinite(wrench_array)):
        raise ValueError("allocator body wrench must contain 6 finite values")
    if not np.all(np.isfinite(scale)) or np.any(scale < 0.0):
        raise ValueError("wrench_scale must contain 6 finite non-negative values")

    normalized_axis_order = str(policy_axis_order).strip().lower().replace("-", "_")
    if normalized_axis_order in {"allocator_body", "physical_allocator_body"}:
        return (wrench_array * scale).astype(np.float32, copy=False)
    if normalized_axis_order not in {"legacy_surge_sway_heave", "legacy", "surge_sway_heave"}:
        raise ValueError(
            "wrench6d.policy_axis_order must be `legacy_surge_sway_heave` or `allocator_body`"
        )

    # policy [surge, sway, heave, roll, pitch, yaw]
    # body   [Fx,    Fy,   Fz,    Mx,   My,    Mz]
    return np.asarray(
        [
            wrench_array[0] * scale[0],
            wrench_array[1] * scale[2],
            wrench_array[2] * scale[1],
            wrench_array[3] * scale[3],
            wrench_array[4] * scale[5],
            wrench_array[5] * scale[4],
        ],
        dtype=np.float32,
    )


def rate_limit_policy_action(
    action: Sequence[float],
    previous_action: Sequence[float] | None,
    delta_time_sec: float,
    max_rate_per_sec: float,
) -> np.ndarray:
    """Limit normalized policy-action changes for real-vehicle deployment."""
    current = np.asarray(action, dtype=np.float32).reshape(-1)
    if previous_action is None or max_rate_per_sec <= 0.0:
        return current.astype(np.float32, copy=True)

    previous = np.asarray(previous_action, dtype=np.float32).reshape(-1)
    if previous.shape != current.shape:
        return current.astype(np.float32, copy=True)

    max_delta = max(float(max_rate_per_sec), 0.0) * max(float(delta_time_sec), 0.0)
    if max_delta <= 0.0:
        return previous.astype(np.float32, copy=True)
    return np.clip(current, previous - max_delta, previous + max_delta).astype(np.float32, copy=False)

def _vec3_param(node: Node, name: str, default: Sequence[float]) -> tuple[float, float, float]:
    values = node.declare_parameter(name, _float_param_default(default)).value
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] < 3:
        array = np.pad(array, (0, 3 - array.shape[0]), constant_values=0.0)
    elif array.shape[0] > 3:
        array = array[:3]
    return (float(array[0]), float(array[1]), float(array[2]))


def _float_param_default(default: Sequence[float]) -> list[float]:
    return [float(value) for value in np.asarray(default, dtype=np.float64).reshape(-1)]


def _vec6_param(node: Node, name: str, default: Sequence[float]) -> np.ndarray:
    values = node.declare_parameter(name, _float_param_default(default)).value
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] < WRENCH6D_POLICY_DIM:
        array = np.pad(array, (0, WRENCH6D_POLICY_DIM - array.shape[0]), constant_values=0.0)
    elif array.shape[0] > WRENCH6D_POLICY_DIM:
        array = array[:WRENCH6D_POLICY_DIM]
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.astype(np.float32, copy=False)


def _vec8_param(node: Node, name: str, default: Sequence[float]) -> np.ndarray:
    values = node.declare_parameter(name, _float_param_default(default)).value
    return _coerce_vec8(name, values, pad_or_truncate=True)


def _coerce_vec8(name: str, values: Sequence[float], *, pad_or_truncate: bool) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if pad_or_truncate and array.shape[0] < 8:
        array = np.pad(array, (0, 8 - array.shape[0]), constant_values=0.0)
    elif pad_or_truncate and array.shape[0] > 8:
        array = array[:8]
    elif array.shape[0] != 8:
        raise ValueError(f"{name} must contain exactly 8 values, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array.astype(np.float32, copy=False)


def _optional_float_tuple_param(
    node: Node,
    name: str,
    default: Sequence[float],
    expected_len: int,
) -> tuple[float, ...] | None:
    param_default = _float_param_default(default) if len(default) > 0 else [float("nan")] * expected_len
    values = node.declare_parameter(name, param_default).value
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] == expected_len and not np.any(np.isfinite(array)):
        return None
    if array.shape[0] == 0:
        return None
    if array.shape[0] != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return tuple(float(v) for v in array)


def _string_param(node: Node, name: str, default: str) -> str:
    return os.path.expandvars(str(node.declare_parameter(name, default).value).strip())


def _string_list_param(node: Node, name: str, default: Sequence[str]) -> tuple[str, ...]:
    values = node.declare_parameter(name, list(default)).value
    if isinstance(values, str):
        return (values.strip(),) if values.strip() else ()
    return tuple(str(value).strip() for value in values if str(value).strip())


def _optional_quat(quat_xyzw: Sequence[float]) -> np.ndarray | None:
    quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(-1)
    if quat.shape[0] < 4:
        quat = np.pad(quat, (0, 4 - quat.shape[0]), constant_values=0.0)
    elif quat.shape[0] > 4:
        quat = quat[:4]
    if float(np.linalg.norm(quat)) <= 1e-8:
        return None
    return quat_normalize(quat)


def _quat_to_controller_rpy_deg(quat_xyzw: Sequence[float]) -> np.ndarray:
    quat = quat_normalize(quat_xyzw).astype(np.float64)
    x, y, z, w = quat

    sinr = 2.0 * (w * x - y * z)
    sinr = float(np.clip(sinr, -1.0, 1.0))
    roll = np.arcsin(sinr)
    pitch = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    yaw = np.arctan2(2.0 * (w * y + x * z), 1.0 - 2.0 * (x * x + y * y))
    return np.degrees(np.array([roll, pitch, yaw], dtype=np.float64)).astype(np.float32)


def _wrap_to_180(angle_deg: float | np.ndarray) -> np.ndarray:
    angle = np.asarray(angle_deg, dtype=np.float32)
    return ((angle + 180.0) % 360.0 - 180.0).astype(np.float32)


def _update_goal_hold_state(
    *,
    within_tolerance: bool,
    now_sec: float,
    entered_sec: float | None,
    hold_time_sec: float,
) -> tuple[bool, float | None, float]:
    if not within_tolerance:
        return False, None, 0.0

    effective_hold_time_sec = max(float(hold_time_sec), 0.0)
    if entered_sec is None:
        entered_sec = now_sec
    held_duration_sec = max(0.0, now_sec - entered_sec)
    return held_duration_sec >= effective_hold_time_sec, entered_sec, held_duration_sec


def _update_stuck_hold_state(
    *,
    candidate: bool,
    now_sec: float,
    position_error: float,
    entered_sec: float | None,
    best_error: float | None,
    progress_epsilon_m: float,
    trigger_duration_sec: float,
) -> tuple[bool, float | None, float | None, float]:
    if not candidate:
        return False, None, None, 0.0

    if entered_sec is None or best_error is None:
        return False, now_sec, position_error, 0.0

    if position_error < best_error - max(float(progress_epsilon_m), 0.0):
        return False, now_sec, position_error, 0.0

    best_error = min(best_error, position_error)
    held_duration_sec = max(0.0, now_sec - entered_sec)
    return held_duration_sec >= max(float(trigger_duration_sec), 0.0), entered_sec, best_error, held_duration_sec


def _yaw_only_target_quat(quat_xyzw: Sequence[float]) -> np.ndarray:
    yaw_deg = float(_quat_to_controller_rpy_deg(quat_xyzw)[2])
    return quat_from_controller_ypr(np.deg2rad(yaw_deg), 0.0, 0.0)


def _controller_yaw_deg(quat_xyzw: Sequence[float]) -> float:
    return float(_quat_to_controller_rpy_deg(quat_xyzw)[2])


def _yaw_error_deg(current_quat_xyzw: Sequence[float], target_quat_xyzw: Sequence[float]) -> float:
    current_yaw_deg = _controller_yaw_deg(current_quat_xyzw)
    target_yaw_deg = _controller_yaw_deg(target_quat_xyzw)
    return float(abs(_wrap_to_180(target_yaw_deg - current_yaw_deg)))


def _is_controller_frame(frame_id: str | None) -> bool:
    frame = (frame_id or "").strip().lower()
    return frame in {"controller_world", "controller_body"}


def _is_exact_frame(frame_id: str | None, expected: str) -> bool:
    return (frame_id or "").strip().lower() == expected


def _vector_to_controller_frame(vec_xyz: Sequence[float], basis_matrix: np.ndarray, frame_id: str | None) -> np.ndarray:
    vector = np.asarray(vec_xyz, dtype=np.float32).reshape(3)
    if _is_controller_frame(frame_id):
        return vector.astype(np.float32, copy=True)
    return reorder_vector(vector, basis_matrix)


def _position_to_controller_frame(
    position_xyz: Sequence[float],
    basis_matrix: np.ndarray,
    position_offset_controller: Sequence[float],
    frame_id: str | None,
) -> np.ndarray:
    position = np.asarray(position_xyz, dtype=np.float32).reshape(3)
    if _is_controller_frame(frame_id):
        return position.astype(np.float32, copy=True)
    offset = np.asarray(position_offset_controller, dtype=np.float32).reshape(3)
    return reorder_vector(position, basis_matrix) + offset


def _quat_to_controller_frame(quat_xyzw: Sequence[float], basis_matrix: np.ndarray, frame_id: str | None) -> np.ndarray:
    if _is_controller_frame(frame_id):
        return quat_normalize(quat_xyzw)
    return transform_quat_basis(quat_xyzw, basis_matrix)


def _load_installed_default_params() -> dict:
    try:
        config_path = (
            Path(get_package_share_directory("motion_control"))
            / "config"
            / "controller.yaml"
        )
    except Exception:
        return {}

    if not config_path.is_file():
        return {}

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    params = data.get("motion_controller", {}).get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


class MotionControllerNode(Node):
    def __init__(self) -> None:
        super().__init__("motion_controller")
        config_defaults = _load_installed_default_params()

        vehicle_name = _string_param(self, "vehicle_name", str(config_defaults.get("vehicle_name", "FinsROV")))
        control_rate_hz = float(
            self.declare_parameter("control_rate_hz", float(config_defaults.get("control_rate_hz", 20.0))).value
        )
        self._control_rate_hz = control_rate_hz
        self._lockstep_enabled = bool(
            self.declare_parameter("lockstep_enabled", bool(config_defaults.get("lockstep_enabled", False))).value
        )
        self._lockstep_clock_topic = _string_param(
            self,
            "lockstep_clock_topic",
            str(config_defaults.get("lockstep_clock_topic", "/clock")),
        )
        self._lockstep_ack_topic = _string_param(
            self,
            "lockstep_ack_topic",
            str(config_defaults.get("lockstep_ack_topic", "/motion_controller/debug/control_tick_complete")),
        )
        if self._lockstep_enabled and (
            not self._lockstep_clock_topic.startswith("/") or not self._lockstep_ack_topic.startswith("/")
        ):
            raise ValueError("lockstep_clock_topic and lockstep_ack_topic must be absolute ROS topic names")
        backend_name = _string_param(
            self,
            "backend_name",
            str(config_defaults.get("backend_name", "traditional_pid_position")),
        )
        checkpoint_path = _string_param(self, "checkpoint_path", str(config_defaults.get("checkpoint_path", "")))
        device = _string_param(self, "device", str(config_defaults.get("device", "auto")))
        self._device = device
        self._position_tolerance = float(
            self.declare_parameter("position_tolerance", float(config_defaults.get("position_tolerance", 0.1))).value
        )
        self._orientation_tolerance_deg = float(
            self.declare_parameter(
                "orientation_tolerance_deg",
                float(config_defaults.get("orientation_tolerance_deg", 10.0)),
            ).value
        )
        self._velocity_tolerance = float(
            self.declare_parameter("velocity_tolerance", float(config_defaults.get("velocity_tolerance", 0.05))).value
        )
        self._use_target_orientation = bool(
            self.declare_parameter(
                "use_target_orientation",
                bool(config_defaults.get("use_target_orientation", False)),
            ).value
        )
        self._require_target_orientation = bool(
            self.declare_parameter(
                "require_target_orientation",
                bool(config_defaults.get("require_target_orientation", False)),
            ).value
        )
        self._stop_on_goal_reached = bool(
            self.declare_parameter("stop_on_goal_reached", bool(config_defaults.get("stop_on_goal_reached", True))).value
        )
        self._goal_reach_hold_time_sec = float(
            self.declare_parameter(
                "goal_reach_hold_time_sec",
                float(config_defaults.get("goal_reach_hold_time_sec", 10.0)),
            ).value
        )
        self._frame_id = _string_param(
            self,
            "controller_frame_id",
            str(config_defaults.get("controller_frame_id", "controller_world")),
        )
        self._control_mode = _normalize_control_mode(
            _string_param(
                self,
                "control_mode",
                str(config_defaults.get("control_mode", CONTROL_MODE_ROS_MANUAL)),
            )
        )
        self._control_mode_topic = _string_param(
            self,
            "control_mode_topic",
            str(config_defaults.get("control_mode_topic", "/motion_controller/control_mode")),
        )
        water_surface_z_m = float(
            self.declare_parameter(
                "water_surface_z_m",
                float(config_defaults.get("water_surface_z_m", 0.98)),
            ).value
        )

        basis_indices = tuple(
            int(v)
            for v in self.declare_parameter(
                "ros_to_controller_basis_indices",
                list(config_defaults.get("ros_to_controller_basis_indices", [0, 2, 1])),
            ).value
        )
        basis_signs = tuple(
            float(v)
            for v in self.declare_parameter(
                "ros_to_controller_basis_signs",
                list(config_defaults.get("ros_to_controller_basis_signs", [1.0, 1.0, 1.0])),
            ).value
        )
        position_offset_default = config_defaults.get(
            "ros_to_controller_position_offset",
            (0.0, 0.0, 0.0),
        )
        position_offset_controller = np.asarray(
            _vec3_param(
                self,
                "ros_to_controller_position_offset",
                tuple(position_offset_default),
            ),
            dtype=np.float32,
        )
        self._outer_loop_kp = np.asarray(
            _vec3_param(
                self,
                "position_outer_loop_kp",
                tuple(config_defaults.get("position_outer_loop_kp", (0.8, 0.6, 0.8))),
            ),
            dtype=np.float32,
        )
        self._max_body_velocity = np.asarray(
            _vec3_param(
                self,
                "max_body_velocity",
                tuple(config_defaults.get("max_body_velocity", (0.30, 0.20, 0.30))),
            ),
            dtype=np.float32,
        )
        pose16_defaults = config_defaults.get("pose16_rot6d", {})
        if not isinstance(pose16_defaults, dict):
            pose16_defaults = {}
        self._pose16_position_scale = float(
            self.declare_parameter(
                "pose16_rot6d.position_scale",
                float(pose16_defaults.get("position_scale", 3.0)),
            ).value
        )
        self._pose16_linear_velocity_scale = np.asarray(
            _vec3_param(
                self,
                "pose16_rot6d.linear_velocity_scale",
                tuple(pose16_defaults.get("linear_velocity_scale", (1.0, 1.0, 1.0))),
            ),
            dtype=np.float32,
        )
        self._pose16_angular_velocity_scale = np.asarray(
            _vec3_param(
                self,
                "pose16_rot6d.angular_velocity_scale",
                tuple(pose16_defaults.get("angular_velocity_scale", (1.0, 1.0, 1.0))),
            ),
            dtype=np.float32,
        )
        self._pose16_velocity_clip = float(
            self.declare_parameter(
                "pose16_rot6d.velocity_clip",
                float(pose16_defaults.get("velocity_clip", 2.0)),
            ).value
        )

        default_thruster_topic = str(
            config_defaults.get("thruster_topic", config_defaults.get("pwm_topic", "/finsrov/thrusters_out"))
        )
        default_pose_topic = f"/{vehicle_name}/controller/pose"
        default_imu_topic = f"/{vehicle_name}/controller/imu"
        default_depth_topic = f"/{vehicle_name}/controller/depth"
        default_dvl_topic = f"/{vehicle_name}/controller/dvl"

        legacy_pwm_topic = _string_param(self, "pwm_topic", default_thruster_topic)
        thruster_topic = _string_param(self, "thruster_topic", "") or legacy_pwm_topic or default_thruster_topic
        self._thruster_output_mode = _normalize_thruster_output_mode(
            _string_param(
                self,
                "thruster_output_mode",
                str(config_defaults.get("thruster_output_mode", THRUSTER_OUTPUT_FORCE_N)),
            )
        )
        force_limits_default = config_defaults.get("thruster_force_limits_n", {})
        if not isinstance(force_limits_default, dict):
            force_limits_default = {}
        self._thruster_force_limit_positive = np.abs(
            _vec8_param(
                self,
                "thruster_force_limits_n.positive",
                tuple(force_limits_default.get("positive", [1.0] * 8)),
            )
        )
        self._thruster_force_limit_negative = np.abs(
            _vec8_param(
                self,
                "thruster_force_limits_n.negative",
                tuple(force_limits_default.get("negative", [1.0] * 8)),
            )
        )
        self._thruster_output_scale = _vec8_param(
            self,
            "thruster_output_scale",
            tuple(config_defaults.get("thruster_output_scale", [1.0] * 8)),
        )
        projection_defaults = config_defaults.get("thruster8_wrench_projection", {})
        if not isinstance(projection_defaults, dict):
            projection_defaults = {}
        self._thruster8_projection_input_scale = _vec8_param(
            self,
            "thruster8_wrench_projection.input_thruster_scale",
            tuple(projection_defaults.get("input_thruster_scale", [1.0] * 8)),
        )
        wrench6d_defaults = config_defaults.get("wrench6d", {})
        if not isinstance(wrench6d_defaults, dict):
            wrench6d_defaults = {}
        # wrench_limits is the physical contract for physical_wrench_allocator policies.
        # action_gains remains a read-only YAML fallback for old checkpoints.
        legacy_action_gains = wrench6d_defaults.get("action_gains", DEFAULT_WRENCH6D_WRENCH_LIMITS)
        self._wrench6d_wrench_limits = _vec6_param(
            self,
            "wrench6d.wrench_limits",
            tuple(wrench6d_defaults.get("wrench_limits", legacy_action_gains)),
        )
        self._wrench6d_wrench_scale = _vec6_param(
            self,
            "wrench6d.wrench_scale",
            tuple(wrench6d_defaults.get("wrench_scale", DEFAULT_WRENCH6D_SCALE)),
        )
        if np.any(self._wrench6d_wrench_scale < 0.0):
            raise ValueError("wrench6d.wrench_scale must contain non-negative values")
        self._wrench6d_policy_axis_order = _string_param(
            self,
            "wrench6d.policy_axis_order",
            str(wrench6d_defaults.get("policy_axis_order", "legacy_surge_sway_heave")),
        )
        self._wrench6d_allocation_mode = _string_param(
            self,
            "wrench6d.allocation_mode",
            str(wrench6d_defaults.get("allocation_mode", "empirical_thruster_mixer")),
        )
        # IsaacLab virtual-wrench is a separate physical-wrench adapter. Its
        # allocator ranges are intentionally not part of PPO-wrench control.
        if self._thruster_output_mode == THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N:
            self._wrench6d_allocator_control_linear_range = float(
                self.declare_parameter(
                    "wrench6d.allocator_control_linear_range",
                    float(
                        wrench6d_defaults.get(
                            "allocator_control_linear_range",
                            DEFAULT_WRENCH6D_ALLOCATOR_CONTROL_LINEAR_RANGE,
                        )
                    ),
                ).value
            )
            self._wrench6d_allocator_control_axis_ranges = _vec6_param(
                self,
                "wrench6d.allocator_control_axis_ranges",
                tuple(wrench6d_defaults.get("allocator_control_axis_ranges", DEFAULT_WRENCH6D_WRENCH_LIMITS)),
            )
            self._wrench6d_allocator_allocation_mode = _string_param(
                self,
                "wrench6d.allocator_allocation_mode",
                str(wrench6d_defaults.get("allocator_allocation_mode", DEFAULT_WRENCH6D_ALLOCATOR_ALLOCATION_MODE)),
            )
        else:
            self._wrench6d_allocator_control_linear_range = 1.0
            self._wrench6d_allocator_control_axis_ranges = np.ones(WRENCH6D_POLICY_DIM, dtype=np.float32)
            self._wrench6d_allocator_allocation_mode = "empirical_thruster_mixer"
        isaaclab_defaults = config_defaults.get("isaaclab", {})
        if not isinstance(isaaclab_defaults, dict):
            isaaclab_defaults = {}
        self._isaaclab_auto_scale_from_envelope = bool(
            self.declare_parameter(
                "isaaclab.auto_scale_from_envelope",
                bool(isaaclab_defaults.get("auto_scale_from_envelope", True)),
            ).value
        )
        self._isaaclab_fins_reference_wrench = _vec6_param(
            self,
            "isaaclab.fins_reference_wrench",
            tuple(isaaclab_defaults.get("fins_reference_wrench", (10.0, 10.0, 10.0, 0.1, 0.1, 0.1))),
        )
        self._isaaclab_virtual_to_fins_scale = _vec6_param(
            self,
            "isaaclab.virtual_to_fins_scale",
            tuple(isaaclab_defaults.get("virtual_to_fins_scale", (1.0,) * 6)),
        )
        self._isaaclab_force_axis_permutation = tuple(
            int(v)
            for v in self.declare_parameter(
                "isaaclab.force_axis_permutation",
                list(isaaclab_defaults.get("force_axis_permutation", (0, 2, 1))),
            ).value
        )
        self._isaaclab_force_axis_signs = _vec3_param(
            self,
            "isaaclab.force_axis_signs",
            tuple(isaaclab_defaults.get("force_axis_signs", (1.0, 1.0, 1.0))),
        )
        self._isaaclab_torque_axis_permutation = tuple(
            int(v)
            for v in self.declare_parameter(
                "isaaclab.torque_axis_permutation",
                list(isaaclab_defaults.get("torque_axis_permutation", (0, 2, 1))),
            ).value
        )
        self._isaaclab_torque_axis_signs = _vec3_param(
            self,
            "isaaclab.torque_axis_signs",
            tuple(isaaclab_defaults.get("torque_axis_signs", (1.0, 1.0, 1.0))),
        )
        self._policy_action_rate_limit_per_sec = float(
            self.declare_parameter(
                "policy_action_rate_limit_per_sec",
                float(config_defaults.get("policy_action_rate_limit_per_sec", 0.0)),
            ).value
        )
        odom_topic = _string_param(self, "odom_topic", str(config_defaults.get("odom_topic", "")))
        pose_topic = _string_param(self, "pose_topic", str(config_defaults.get("pose_topic", default_pose_topic)))
        imu_topic = _string_param(self, "imu_topic", str(config_defaults.get("imu_topic", default_imu_topic)))
        depth_topic = _string_param(self, "depth_topic", str(config_defaults.get("depth_topic", default_depth_topic)))
        dvl_topic = _string_param(self, "dvl_topic", str(config_defaults.get("dvl_topic", default_dvl_topic)))
        self._state_status_topic = _string_param(
            self,
            "state_status_topic",
            str(config_defaults.get("state_status_topic", "/finsrov/controller/state/status")),
        )
        self._require_fusion_ready = bool(
            self.declare_parameter(
                "require_fusion_ready",
                bool(config_defaults.get("require_fusion_ready", True)),
            ).value
        )
        self._state_timeout_sec = float(
            self.declare_parameter(
                "state_timeout_sec",
                float(config_defaults.get("state_timeout_sec", 0.5)),
            ).value
        )
        self._allowed_vision_modes = _string_list_param(
            self,
            "allowed_vision_modes",
            tuple(config_defaults.get("allowed_vision_modes", ("fresh", "coast", "hold"))),
        )
        self._vision_loss_hold_modes = _string_list_param(
            self,
            "vision_loss_hold_modes",
            tuple(config_defaults.get("vision_loss_hold_modes", ("hold",))),
        )
        self._vision_loss_hold_timeout_sec = float(
            self.declare_parameter(
                "vision_loss_hold_timeout_sec",
                float(config_defaults.get("vision_loss_hold_timeout_sec", 3.0)),
            ).value
        )
        stuck_defaults = config_defaults.get("stuck_recovery", {})
        if not isinstance(stuck_defaults, dict):
            stuck_defaults = {}
        self._stuck_recovery_enabled = bool(
            self.declare_parameter(
                "stuck_recovery.enabled",
                bool(stuck_defaults.get("enabled", False)),
            ).value
        )
        self._stuck_min_position_error = float(
            self.declare_parameter(
                "stuck_recovery.min_position_error_m",
                float(stuck_defaults.get("min_position_error_m", 0.12)),
            ).value
        )
        self._stuck_velocity_threshold = float(
            self.declare_parameter(
                "stuck_recovery.velocity_threshold_mps",
                float(stuck_defaults.get("velocity_threshold_mps", 0.03)),
            ).value
        )
        self._stuck_horizontal_effort_threshold = float(
            self.declare_parameter(
                "stuck_recovery.horizontal_effort_threshold",
                float(stuck_defaults.get("horizontal_effort_threshold", 0.18)),
            ).value
        )
        self._stuck_progress_epsilon = float(
            self.declare_parameter(
                "stuck_recovery.progress_epsilon_m",
                float(stuck_defaults.get("progress_epsilon_m", 0.02)),
            ).value
        )
        self._stuck_trigger_duration = float(
            self.declare_parameter(
                "stuck_recovery.trigger_duration_sec",
                float(stuck_defaults.get("trigger_duration_sec", 4.0)),
            ).value
        )
        self._stuck_escape_duration = float(
            self.declare_parameter(
                "stuck_recovery.escape_duration_sec",
                float(stuck_defaults.get("escape_duration_sec", 1.0)),
            ).value
        )
        self._stuck_backoff_scale = float(
            self.declare_parameter(
                "stuck_recovery.backoff_horizontal_scale",
                float(stuck_defaults.get("backoff_horizontal_scale", 0.35)),
            ).value
        )
        self._stuck_vertical_hold_scale = float(
            self.declare_parameter(
                "stuck_recovery.vertical_hold_scale",
                float(stuck_defaults.get("vertical_hold_scale", 1.0)),
            ).value
        )
        self._stuck_cooldown_sec = float(
            self.declare_parameter(
                "stuck_recovery.cooldown_sec",
                float(stuck_defaults.get("cooldown_sec", 1.0)),
            ).value
        )
        self._stuck_status_topic = _string_param(
            self,
            "stuck_recovery.status_topic",
            str(stuck_defaults.get("status_topic", "/motion_controller/status/stuck")),
        )

        self._command_position_world_topic = _string_param(
            self,
            "command_position_world_topic",
            str(
                config_defaults.get(
                    "command_position_world_topic",
                    "/motion_controller/command/position_controller_world",
                )
            ),
        )
        self._command_position_body_topic = _string_param(
            self,
            "command_position_body_topic",
            str(
                config_defaults.get(
                    "command_position_body_topic",
                    "/motion_controller/command/position_controller_body",
                )
            ),
        )
        self._command_pose_body_topic = _string_param(
            self,
            "command_pose_body_topic",
            str(config_defaults.get("command_pose_body_topic", "/motion_controller/command/pose_controller_body")),
        )
        self._command_velocity_topic = _string_param(
            self,
            "command_velocity_body_topic",
            str(config_defaults.get("command_velocity_body_topic", "/motion_controller/command/velocity_controller_body")),
        )
        self._command_trajectory_topic = _string_param(
            self,
            "command_trajectory_topic",
            str(config_defaults.get("command_trajectory_topic", "/motion_controller/command/trajectory")),
        )
        trajectory_defaults = config_defaults.get("trajectory30", {})
        if not isinstance(trajectory_defaults, dict):
            trajectory_defaults = {}
        self._trajectory_preview_step_sec = float(
            self.declare_parameter("trajectory30.preview_step_sec", float(trajectory_defaults.get("preview_step_sec", 0.1))).value
        )
        self._trajectory_preview_offset_scale = float(
            self.declare_parameter("trajectory30.preview_offset_scale", float(trajectory_defaults.get("preview_offset_scale", 3.0))).value
        )
        self._trajectory_linear_velocity_scale = float(
            self.declare_parameter("trajectory30.linear_velocity_scale", float(trajectory_defaults.get("linear_velocity_scale", 1.0))).value
        )
        self._trajectory_angular_velocity_scale = float(
            self.declare_parameter("trajectory30.angular_velocity_scale", float(trajectory_defaults.get("angular_velocity_scale", 1.0))).value
        )
        self._trajectory_observation_clip = float(
            self.declare_parameter("trajectory30.observation_clip", float(trajectory_defaults.get("observation_clip", 2.0))).value
        )
        self._command_cancel_topic = _string_param(
            self,
            "command_cancel_topic",
            str(config_defaults.get("command_cancel_topic", "/motion_controller/command/cancel")),
        )
        self._status_active_topic = _string_param(
            self,
            "status_active_pose_topic",
            str(config_defaults.get("status_active_pose_topic", "/motion_controller/status/active_pose")),
        )
        self._status_error_topic = _string_param(
            self,
            "status_error_body_topic",
            str(config_defaults.get("status_error_body_topic", "/motion_controller/status/error_body")),
        )
        self._status_reached_topic = _string_param(
            self,
            "status_reached_topic",
            str(config_defaults.get("status_reached_topic", "/motion_controller/status/reached")),
        )
        self._debug_observation_topic = _string_param(
            self,
            "debug_observation_topic",
            str(config_defaults.get("debug_observation_topic", "/motion_controller/debug/observation")),
        )
        self._debug_action_topic = _string_param(
            self,
            "debug_action_topic",
            str(config_defaults.get("debug_action_topic", "/motion_controller/debug/action")),
        )
        self._debug_thruster_command_topic = _string_param(
            self,
            "debug_thruster_command_topic",
            str(config_defaults.get("debug_thruster_command_topic", "/motion_controller/debug/thruster_command")),
        )
        self._debug_thruster_command_stamped_topic = _string_param(
            self,
            "debug_thruster_command_stamped_topic",
            str(config_defaults.get("debug_thruster_command_stamped_topic", "")),
        )
        self._debug_wrench6d_topic = _string_param(
            self,
            "debug_wrench6d_topic",
            str(config_defaults.get("debug_wrench6d_topic", "/motion_controller/debug/wrench6d")),
        )
        self._debug_policy_info_topic = _string_param(
            self,
            "debug_policy_info_topic",
            str(config_defaults.get("debug_policy_info_topic", "/motion_controller/debug/policy_info")),
        )
        self._debug_trajectory_reference_topic = _string_param(
            self,
            "debug_trajectory_reference_topic",
            str(config_defaults.get("debug_trajectory_reference_topic", "/motion_controller/debug/trajectory_reference")),
        )
        self._debug_trajectory_reference_stamped_topic = _string_param(
            self,
            "debug_trajectory_reference_stamped_topic",
            str(config_defaults.get("debug_trajectory_reference_stamped_topic", "")),
        )
        self._pwm_pub = self.create_publisher(Float32MultiArray, thruster_topic, 10)
        self._goal_active_pub = self.create_publisher(PoseStamped, self._status_active_topic, 10)
        self._goal_error_pub = self.create_publisher(Vector3Stamped, self._status_error_topic, 10)
        self._goal_reached_pub = self.create_publisher(Bool, self._status_reached_topic, 10)
        self._stuck_status_pub = (
            self.create_publisher(String, self._stuck_status_topic, 10) if self._stuck_status_topic else None
        )
        self._debug_observation_pub = (
            self.create_publisher(Float32MultiArray, self._debug_observation_topic, 10)
            if self._debug_observation_topic
            else None
        )
        self._debug_action_pub = (
            self.create_publisher(Float32MultiArray, self._debug_action_topic, 10)
            if self._debug_action_topic
            else None
        )
        self._debug_thruster_command_pub = (
            self.create_publisher(Float32MultiArray, self._debug_thruster_command_topic, 10)
            if self._debug_thruster_command_topic
            else None
        )
        # Float32MultiArray intentionally remains the established interactive
        # debug ABI.  The optional StampedJson mirror is enabled only by the
        # accelerated /sim experiment profiles so an offline analysis can use
        # the controller's exact /clock tick rather than rosbag arrival time.
        self._debug_thruster_command_stamped_pub = (
            self.create_publisher(StampedJson, self._debug_thruster_command_stamped_topic, 10)
            if self._debug_thruster_command_stamped_topic
            else None
        )
        self._debug_wrench6d_pub = (
            self.create_publisher(Float32MultiArray, self._debug_wrench6d_topic, 10)
            if self._debug_wrench6d_topic
            else None
        )
        self._debug_policy_info_pub = (
            self.create_publisher(String, self._debug_policy_info_topic, 10)
            if self._debug_policy_info_topic
            else None
        )
        self._debug_trajectory_reference_pub = (
            self.create_publisher(Float32MultiArray, self._debug_trajectory_reference_topic, 10)
            if self._debug_trajectory_reference_topic
            else None
        )
        # Float32MultiArray has no Header.  That is acceptable for interactive
        # debugging, but cannot be aligned with Unity /clock during accelerated
        # T2 runs.  The optional PoseStamped mirror is enabled by the T2 sim
        # profile and carries the exact controller-clock stamp and reference
        # position without changing the established array ABI.
        self._debug_trajectory_reference_stamped_pub = (
            self.create_publisher(PoseStamped, self._debug_trajectory_reference_stamped_topic, 10)
            if self._debug_trajectory_reference_stamped_topic
            else None
        )

        self.create_subscription(PoseStamped, self._command_position_world_topic, self._goal_global_callback, 10)
        self.create_subscription(Vector3Stamped, self._command_position_body_topic, self._goal_body_callback, 10)
        self.create_subscription(PoseStamped, self._command_pose_body_topic, self._goal_body_pose_callback, 10)
        self.create_subscription(TwistStamped, self._command_velocity_topic, self._velocity_command_callback, 10)
        self.create_subscription(MultiDOFJointTrajectory, self._command_trajectory_topic, self._trajectory_callback, 10)
        self.create_subscription(Empty, self._command_cancel_topic, self._goal_cancel_callback, 10)
        self.create_subscription(String, self._control_mode_topic, self._control_mode_callback, 10)
        if self._state_status_topic:
            self.create_subscription(String, self._state_status_topic, self._fusion_status_callback, 10)

        self._estimator = VehicleStateEstimator(
            self,
            odom_topic=odom_topic,
            pose_topic=pose_topic,
            imu_topic=imu_topic,
            depth_topic=depth_topic,
            dvl_topic=dvl_topic,
            basis_indices=basis_indices,
            basis_signs=basis_signs,
            position_offset_controller=position_offset_controller,
        )
        traditional_overrides: dict[str, object] = {
            "control_rate_hz": control_rate_hz,
            "enable_yaw_control": bool(
                self.declare_parameter("traditional_enable_yaw_control", True).value
            ),
            "allocator_control_linear_range": float(
                self.declare_parameter("traditional_allocator_control_linear_range", 10.0).value
            ),
            "allocator_allocation_mode": _string_param(
                self, "traditional_allocator_allocation_mode", "empirical_thruster_mixer"
            ),
            "surge_output_limit": float(self.declare_parameter("traditional_surge_output_limit", 120.0).value),
            "sway_output_limit": float(self.declare_parameter("traditional_sway_output_limit", 120.0).value),
            "yaw_deadband_deg": float(self.declare_parameter("traditional_yaw_deadband_deg", 0.5).value),
            "yaw_output_limit": float(self.declare_parameter("traditional_yaw_output_limit", 40.0).value),
            "integral_decay": float(self.declare_parameter("traditional_integral_decay", 0.95).value),
        }
        traditional_pid_params = _optional_float_tuple_param(self, "traditional_pid_params", (), 9)
        if traditional_pid_params is not None:
            traditional_overrides["pid_params"] = traditional_pid_params
        traditional_yaw_pid_params = _optional_float_tuple_param(self, "traditional_yaw_pid_params", (), 3)
        if traditional_yaw_pid_params is not None:
            traditional_overrides["yaw_pid_params"] = traditional_yaw_pid_params
        traditional_output_range = _optional_float_tuple_param(self, "traditional_output_range", (), 2)
        if traditional_output_range is not None:
            traditional_overrides["output_range"] = traditional_output_range
        traditional_allocator_axis_ranges = _optional_float_tuple_param(
            self,
            "traditional_allocator_control_axis_ranges",
            (),
            6,
        )
        if traditional_allocator_axis_ranges is not None:
            traditional_overrides["allocator_control_axis_ranges"] = traditional_allocator_axis_ranges
        traditional_reference_velocity_limit = _optional_float_tuple_param(
            self,
            "traditional_reference_velocity_limit",
            (),
            3,
        )
        if traditional_reference_velocity_limit is not None:
            traditional_overrides["reference_velocity_limit"] = traditional_reference_velocity_limit
        traditional_reference_velocity_rate_limit = _optional_float_tuple_param(
            self,
            "traditional_reference_velocity_rate_limit",
            (),
            3,
        )
        if traditional_reference_velocity_rate_limit is not None:
            traditional_overrides["reference_velocity_rate_limit"] = traditional_reference_velocity_rate_limit
        self._backend = load_backend(
            backend_name,
            checkpoint_path=checkpoint_path,
            device=device,
            training_overrides=traditional_overrides,
        )
        self._wrench6d_thrust_allocator = None
        if self._thruster_output_mode in {
            THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
            THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N,
            THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N,
        }:
            from finssim_rl.models.thrust_allocator import ThrustAllocator

            uses_policy_wrench = self._thruster_output_mode == THRUSTER_OUTPUT_WRENCH6D_FORCE_N
            uses_thruster8_reprojection = (
                self._thruster_output_mode == THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N
            )
            if uses_thruster8_reprojection and self._wrench6d_allocation_mode != "physical_wrench_allocator":
                raise ValueError(
                    "thruster8_reprojected_physical_wrench_force_n requires "
                    "wrench6d.allocation_mode=physical_wrench_allocator"
                )
            policy_wrench_limits_body = policy_action6d_to_allocator_input(
                np.ones(WRENCH6D_POLICY_DIM, dtype=np.float32),
                wrench_limits=self._wrench6d_wrench_limits,
                policy_axis_order=self._wrench6d_policy_axis_order,
            )
            self._wrench6d_thrust_allocator = ThrustAllocator(
                device=device,
                deadzone_comp=0.0,
                control_linear_range=1.0 if uses_policy_wrench else self._wrench6d_allocator_control_linear_range,
                control_axis_ranges=(1.0,) * WRENCH6D_POLICY_DIM
                if uses_policy_wrench
                else tuple(float(v) for v in self._wrench6d_allocator_control_axis_ranges),
                allocation_mode=(
                    self._wrench6d_allocation_mode
                    if uses_policy_wrench or uses_thruster8_reprojection
                    else self._wrench6d_allocator_allocation_mode
                ),
                physical_wrench_limits=policy_wrench_limits_body,
                thruster_force_limit_positive=self._thruster_force_limit_positive,
                thruster_force_limit_negative=self._thruster_force_limit_negative,
                debug_print_interval=0,
            )

        self._isaaclab_adapter = None
        if self._thruster_output_mode == THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N:
            from finssim_isaaclab.adapter import IsaacLabToFinsRovAdapter

            self._isaaclab_adapter = IsaacLabToFinsRovAdapter(
                auto_scale_from_envelope=self._isaaclab_auto_scale_from_envelope,
                fins_reference_wrench=self._isaaclab_fins_reference_wrench,
                virtual_to_fins_scale=self._isaaclab_virtual_to_fins_scale,
                force_axis_permutation=self._isaaclab_force_axis_permutation,
                force_axis_signs=self._isaaclab_force_axis_signs,
                torque_axis_permutation=self._isaaclab_torque_axis_permutation,
                torque_axis_signs=self._isaaclab_torque_axis_signs,
            )

        self._active_command: MotionCommand | None = None
        self._goal_reached_latched = False
        self._goal_within_tolerance_since_sec: float | None = None
        self._last_idle_zero_sec = 0.0
        self._last_control_exception: str | None = None
        self._fusion_status_received = False
        self._last_fusion_status_sec: float | None = None
        self._fusion_ready = False
        self._fusion_initialized = False
        self._fusion_vision_mode = "unknown"
        self._fusion_reject_reason = ""
        self._fusion_age_vision_sec: float | None = None
        self._last_navigation_gate_state: tuple[bool, str] | None = None
        self._stuck_candidate_since_sec: float | None = None
        self._stuck_best_error_m: float | None = None
        self._stuck_last_error_m: float | None = None
        self._stuck_escape_until_sec: float | None = None
        self._stuck_cooldown_until_sec: float | None = None
        self._stuck_escape_action = np.zeros(8, dtype=np.float32)
        self._stuck_phase = "idle"
        self._last_wrench6d_command: np.ndarray | None = None
        self._last_thruster_command: np.ndarray | None = None
        self._last_policy_action: np.ndarray | None = None
        self._last_policy_action_sec: float | None = None
        if self._thruster_output_mode in {
            THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
            THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N,
        } and self._stuck_recovery_enabled:
            self.get_logger().warning("stuck_recovery is disabled for wrench-based output modes")
            self._stuck_recovery_enabled = False
        self.add_on_set_parameters_callback(self._on_parameter_update)

        self._lockstep_period_ns = max(1, int(round(1_000_000_000.0 / max(control_rate_hz, 1e-3))))
        self._lockstep_last_control_tick_ns: int | None = None
        self._lockstep_clock_override_sec: float | None = None
        self._lockstep_clock_override_ns: int | None = None
        self._lockstep_ack_pub = None
        self._lockstep_clock_sub = None
        if self._lockstep_enabled:
            self._lockstep_ack_pub = self.create_publisher(UInt64, self._lockstep_ack_topic, 50)
            self._lockstep_clock_sub = self.create_subscription(
                Clock,
                self._lockstep_clock_topic,
                self._lockstep_clock_callback,
                50,
            )
        else:
            period = 1.0 / max(control_rate_hz, 1e-3)
            self.create_timer(period, self._safe_control_step)

        self.get_logger().info(
            f"motion controller started: backend={backend_name}, thruster_topic={thruster_topic}, "
            f"odom_topic={odom_topic or '<disabled>'}, pose_topic={pose_topic or '<disabled>'}, imu_topic={imu_topic}, "
            f"depth_topic={depth_topic}, dvl_topic={dvl_topic}, use_target_orientation={self._use_target_orientation}, "
            f"require_target_orientation={self._require_target_orientation}, "
            f"thruster_output_mode={self._thruster_output_mode}, "
            f"thruster_output_scale={self._thruster_output_scale.tolist()}, "
            f"control_mode={self._control_mode}, control_mode_topic={self._control_mode_topic}, "
            f"controller_native_state=true, water_surface_z_m_adapter_param={water_surface_z_m:.3f}, "
            f"ros_to_controller_position_offset={position_offset_controller.tolist()}, "
            f"state_status_topic={self._state_status_topic or '<disabled>'}, "
            f"require_fusion_ready={self._require_fusion_ready}, state_timeout_sec={self._state_timeout_sec:.3f}, "
            f"vision_loss_hold_modes={self._vision_loss_hold_modes}, "
            f"vision_loss_hold_timeout_sec={self._vision_loss_hold_timeout_sec:.3f}, "
            f"policy_action_rate_limit_per_sec={self._policy_action_rate_limit_per_sec:.3f}, "
            f"stuck_recovery_enabled={self._stuck_recovery_enabled}, "
            f"lockstep_enabled={self._lockstep_enabled}, "
            f"lockstep_clock_topic={self._lockstep_clock_topic if self._lockstep_enabled else '<timer>'}, "
            f"lockstep_ack_topic={self._lockstep_ack_topic if self._lockstep_enabled else '<disabled>'}"
        )

    def _now_sec(self) -> float:
        if self._lockstep_enabled and self._lockstep_clock_override_sec is not None:
            return self._lockstep_clock_override_sec
        return self.get_clock().now().nanoseconds * 1e-9

    def _now_time_message(self) -> TimeMessage:
        """Return the exact time associated with the current control tick.

        In lockstep, a node's regular ``use_sim_time`` callback can be
        dispatched after this node's explicit ``/clock`` subscription.  Using
        ``get_clock().now()`` from inside that callback would therefore stamp
        controller outputs with a previous tick.  The lockstep handoff carries
        an integer timestamp, so preserve that value for all messages emitted
        by this control step.
        """

        if self._lockstep_enabled and self._lockstep_clock_override_ns is not None:
            total_nanoseconds = self._lockstep_clock_override_ns
        else:
            total_nanoseconds = self.get_clock().now().nanoseconds
        message = TimeMessage()
        message.sec = int(total_nanoseconds // 1_000_000_000)
        message.nanosec = int(total_nanoseconds % 1_000_000_000)
        return message

    def _lockstep_clock_callback(self, message: Clock) -> None:
        """Complete one ROS2 side of the Unity--ROS2 lockstep handshake.

        A Unity physics step publishes a `/clock` value through the gRPC
        adapter and waits for this acknowledgement.  Policy/PID updates retain
        their configured control rate; intermediate physics ticks explicitly
        hold the previous command but are still acknowledged, so Unity cannot
        race ahead of the ROS2 executor.
        """

        tick_nanoseconds = int(message.clock.sec) * 1_000_000_000 + int(message.clock.nanosec)
        if tick_nanoseconds < 0:
            return
        self._lockstep_clock_override_sec = tick_nanoseconds * 1e-9
        self._lockstep_clock_override_ns = tick_nanoseconds
        try:
            due = (
                self._lockstep_last_control_tick_ns is None
                or tick_nanoseconds - self._lockstep_last_control_tick_ns >= self._lockstep_period_ns
            )
            if due:
                self._safe_control_step()
                self._lockstep_last_control_tick_ns = tick_nanoseconds
        finally:
            acknowledgement = UInt64()
            acknowledgement.data = tick_nanoseconds
            if self._lockstep_ack_pub is not None:
                self._lockstep_ack_pub.publish(acknowledgement)
            self._lockstep_clock_override_sec = None
            self._lockstep_clock_override_ns = None

    def _on_parameter_update(self, parameters: list[Parameter]) -> SetParametersResult:
        for parameter in parameters:
            if parameter.name != "thruster_output_scale":
                continue
            try:
                values = parameter.value
                scale = _coerce_vec8(parameter.name, values, pad_or_truncate=False)
            except Exception as exc:
                return SetParametersResult(successful=False, reason=str(exc))
            self._thruster_output_scale = scale
            self.get_logger().info(f"updated thruster_output_scale={scale.tolist()}")
        return SetParametersResult(successful=True)

    def _publish_zero_pwm(self) -> None:
        self._last_wrench6d_command = None
        msg = Float32MultiArray()
        msg.data = [0.0] * 8
        self._pwm_pub.publish(msg)
        self._last_thruster_command = np.zeros(8, dtype=np.float32)

    def _publish_thruster_hold(self) -> None:
        if self._last_thruster_command is None:
            self._publish_zero_pwm()
            return
        msg = Float32MultiArray()
        msg.data = [float(v) for v in self._last_thruster_command]
        self._pwm_pub.publish(msg)

    def _publish_stuck_status(
        self,
        *,
        active: bool,
        phase: str,
        position_error_m: float | None = None,
        speed_norm_mps: float | None = None,
        horizontal_effort: float | None = None,
        held_duration_sec: float | None = None,
    ) -> None:
        if self._stuck_status_pub is None:
            return
        payload = {
            "active": bool(active),
            "phase": str(phase),
            "position_error_m": position_error_m,
            "speed_norm_mps": speed_norm_mps,
            "horizontal_effort": horizontal_effort,
            "held_duration_sec": held_duration_sec,
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._stuck_status_pub.publish(msg)

    def _reset_stuck_recovery(self, *, phase: str = "idle") -> None:
        self._stuck_candidate_since_sec = None
        self._stuck_best_error_m = None
        self._stuck_last_error_m = None
        self._stuck_escape_until_sec = None
        self._stuck_cooldown_until_sec = None
        self._stuck_escape_action = np.zeros(8, dtype=np.float32)
        self._stuck_phase = phase

    def _make_stuck_escape_action(self, action: np.ndarray) -> np.ndarray:
        escape = np.asarray(action, dtype=np.float32).reshape(-1)
        if escape.shape[0] < 8:
            escape = np.pad(escape, (0, 8 - escape.shape[0]), constant_values=0.0)
        elif escape.shape[0] > 8:
            escape = escape[:8]
        escape = escape.astype(np.float32, copy=True)
        escape[:4] *= np.float32(np.clip(self._stuck_vertical_hold_scale, 0.0, 1.0))
        escape[4:8] *= np.float32(-np.clip(self._stuck_backoff_scale, 0.0, 1.0))
        return np.clip(escape, -1.0, 1.0).astype(np.float32, copy=False)

    @staticmethod
    def _float_list(values: np.ndarray | Sequence[float] | None) -> list[float] | None:
        if values is None:
            return None
        array = np.asarray(values, dtype=np.float32).reshape(-1)
        return [float(v) for v in array]

    def _publish_control_debug(
        self,
        *,
        observation: np.ndarray,
        action: np.ndarray,
        command: np.ndarray,
        state: object,
        error_body: np.ndarray | None,
        position_error_m: float | None,
        speed_norm_mps: float | None,
        desired_body_velocity: np.ndarray | None = None,
        wrench6d_command: np.ndarray | None = None,
    ) -> None:
        if self._debug_observation_pub is not None:
            msg = Float32MultiArray()
            msg.data = self._float_list(observation) or []
            self._debug_observation_pub.publish(msg)
        if self._debug_action_pub is not None:
            msg = Float32MultiArray()
            msg.data = self._float_list(action) or []
            self._debug_action_pub.publish(msg)
        if self._debug_thruster_command_pub is not None:
            msg = Float32MultiArray()
            msg.data = self._float_list(command) or []
            self._debug_thruster_command_pub.publish(msg)
        if self._debug_thruster_command_stamped_pub is not None:
            stamped = StampedJson()
            stamped.header.stamp = self._now_time_message()
            stamped.header.frame_id = self._frame_id
            stamped.data = json.dumps(
                {
                    "schema": "canonical_thruster_command_v1",
                    "values": self._float_list(command) or [],
                    "thruster_output_mode": self._thruster_output_mode,
                },
                separators=(",", ":"),
                ensure_ascii=True,
            )
            self._debug_thruster_command_stamped_pub.publish(stamped)
        if self._debug_wrench6d_pub is not None and wrench6d_command is not None:
            msg = Float32MultiArray()
            msg.data = self._float_list(wrench6d_command) or []
            self._debug_wrench6d_pub.publish(msg)
        if self._debug_policy_info_pub is not None:
            active_command = self._active_command
            payload = {
                "backend": self._backend.name,
                "observation_mode": self._backend.observation_mode,
                "command_mode": active_command.mode if active_command is not None else None,
                "command_source_frame": active_command.source_frame if active_command is not None else None,
                "control_mode": self._control_mode,
                "fusion_vision_mode": self._fusion_vision_mode,
                "fusion_age_vision_sec": self._fusion_age_vision_sec,
                "position_error_m": position_error_m,
                "speed_norm_mps": speed_norm_mps,
                "error_body": self._float_list(error_body),
                "desired_body_velocity": self._float_list(desired_body_velocity),
                "linear_velocity_body": self._float_list(getattr(state, "linear_velocity_body", None)),
                "angular_velocity_body_xyz": self._float_list(getattr(state, "angular_velocity_body_xyz", None)),
                "angular_velocity_body_ypr": self._float_list(getattr(state, "angular_velocity_body_ypr", None)),
                "thruster_output_mode": self._thruster_output_mode,
                "thruster_output_scale": self._float_list(self._thruster_output_scale),
                "wrench6d_wrench_limits": self._float_list(self._wrench6d_wrench_limits)
                if self._thruster_output_mode == THRUSTER_OUTPUT_WRENCH6D_FORCE_N
                else None,
                "wrench6d_allocation_mode": self._wrench6d_allocation_mode
                if self._thruster_output_mode == THRUSTER_OUTPUT_WRENCH6D_FORCE_N
                else None,
                "wrench6d_allocator_input": self._float_list(wrench6d_command),
                "isaaclab_adapter": (
                    self._isaaclab_adapter.diagnostics()
                    if getattr(self, "_isaaclab_adapter", None) is not None
                    else None
                ),
                "policy_action_rate_limit_per_sec": self._policy_action_rate_limit_per_sec,
                "stuck_phase": self._stuck_phase,
            }
            msg = String()
            msg.data = json.dumps(payload, separators=(",", ":"), ensure_ascii=True)
            self._debug_policy_info_pub.publish(msg)

    def _publish_thruster_action(self, action: np.ndarray) -> np.ndarray:
        command = self._action_to_thruster_command(action)
        pwm_msg = Float32MultiArray()
        pwm_msg.data = [float(v) for v in command]
        self._pwm_pub.publish(pwm_msg)
        self._last_thruster_command = command.astype(np.float32, copy=True)
        return command

    def _stuck_escape_active(self, now_sec: float) -> bool:
        return self._stuck_escape_until_sec is not None and now_sec < self._stuck_escape_until_sec

    def _update_stuck_recovery(
        self,
        *,
        now_sec: float,
        position_error_m: float,
        speed_norm_mps: float,
        action: np.ndarray,
    ) -> tuple[bool, float, float]:
        if not self._stuck_recovery_enabled:
            self._reset_stuck_recovery()
            return False, 0.0, 0.0
        if self._stuck_cooldown_until_sec is not None and now_sec < self._stuck_cooldown_until_sec:
            self._stuck_phase = "cooldown"
            return False, 0.0, 0.0

        action = np.asarray(action, dtype=np.float32).reshape(-1)
        horizontal_effort = float(np.max(np.abs(action[4:8]))) if action.shape[0] >= 8 else 0.0
        error_worsening = (
            self._stuck_last_error_m is not None
            and position_error_m >= self._stuck_last_error_m + max(self._stuck_progress_epsilon, 0.0)
        )
        candidate = (
            position_error_m >= self._stuck_min_position_error
            and (speed_norm_mps <= self._stuck_velocity_threshold or error_worsening)
            and horizontal_effort >= self._stuck_horizontal_effort_threshold
        )
        self._stuck_last_error_m = position_error_m
        triggered, self._stuck_candidate_since_sec, self._stuck_best_error_m, held_duration_sec = (
            _update_stuck_hold_state(
                candidate=candidate,
                now_sec=now_sec,
                position_error=position_error_m,
                entered_sec=self._stuck_candidate_since_sec,
                best_error=self._stuck_best_error_m,
                progress_epsilon_m=self._stuck_progress_epsilon,
                trigger_duration_sec=self._stuck_trigger_duration,
            )
        )
        self._stuck_phase = "candidate" if candidate else "idle"
        if triggered:
            self._stuck_escape_action = self._make_stuck_escape_action(action)
            self._stuck_escape_until_sec = now_sec + max(self._stuck_escape_duration, 0.0)
            self._stuck_cooldown_until_sec = self._stuck_escape_until_sec + max(self._stuck_cooldown_sec, 0.0)
            self._stuck_candidate_since_sec = None
            self._stuck_best_error_m = None
            self._stuck_phase = "escape"
            self._backend.reset()
            self.get_logger().warning(
                "stuck recovery triggered: "
                f"position_error_m={position_error_m:.3f}, speed_norm_mps={speed_norm_mps:.3f}, "
                f"horizontal_effort={horizontal_effort:.3f}, held_duration_sec={held_duration_sec:.2f}"
            )
        return triggered, horizontal_effort, held_duration_sec

    def _set_control_mode(self, mode: str) -> None:
        next_mode = _normalize_control_mode(mode)
        if next_mode == self._control_mode:
            self.get_logger().info(f"control mode already `{self._control_mode}`")
            return

        previous_mode = self._control_mode
        self._control_mode = next_mode
        self._active_command = None
        self._goal_reached_latched = False
        self._goal_within_tolerance_since_sec = None
        self._reset_stuck_recovery(phase="mode_change")
        self._backend.reset()
        if next_mode == CONTROL_MODE_UNITY_RANDOM:
            self._publish_zero_pwm()
            self._publish_goal_status(
                target_position_world=None,
                target_orientation_world=None,
                error_body=None,
                reached=False,
            )
        self.get_logger().info(f"control mode changed: {previous_mode} -> {next_mode}")

    def _control_mode_callback(self, msg: String) -> None:
        try:
            self._set_control_mode(msg.data)
        except ValueError as exc:
            self.get_logger().warning(str(exc))

    def _fusion_status_callback(self, msg: String) -> None:
        try:
            payload = json.loads(msg.data)
        except json.JSONDecodeError as exc:
            self.get_logger().warning(f"ignored malformed fusion status JSON: {exc}")
            return

        self._fusion_status_received = True
        self._last_fusion_status_sec = self._now_sec()
        self._fusion_ready = bool(payload.get("ready", False))
        self._fusion_initialized = bool(payload.get("initialized", False))
        self._fusion_vision_mode = str(payload.get("vision_mode", "unknown"))
        self._fusion_reject_reason = str(payload.get("reject_reason", ""))
        age_vision = payload.get("age_vision")
        self._fusion_age_vision_sec = float(age_vision) if age_vision is not None else None

    def _check_navigation_ready(self, now_sec: float) -> tuple[bool, str]:
        if not self._estimator.is_fresh(now_sec, self._state_timeout_sec):
            return False, "state_topics_stale_or_missing"

        if self._require_fusion_ready:
            if not self._state_status_topic:
                return False, "fusion_status_topic_disabled"
            if not self._fusion_status_received or self._last_fusion_status_sec is None:
                return False, "fusion_status_missing"
            if self._state_timeout_sec > 0.0 and now_sec - self._last_fusion_status_sec > self._state_timeout_sec:
                return False, "fusion_status_stale"
            if not self._fusion_initialized:
                return False, f"fusion_uninitialized(mode={self._fusion_vision_mode})"
            if not self._fusion_ready:
                suffix = f", reject={self._fusion_reject_reason}" if self._fusion_reject_reason else ""
                return False, f"fusion_not_ready(mode={self._fusion_vision_mode}{suffix})"
            if self._vision_loss_hold_modes and self._fusion_vision_mode in self._vision_loss_hold_modes:
                return False, f"fusion_vision_loss_hold(mode={self._fusion_vision_mode})"
            if self._allowed_vision_modes and self._fusion_vision_mode not in self._allowed_vision_modes:
                return False, f"fusion_vision_mode_not_allowed(mode={self._fusion_vision_mode})"

        return True, "ready"

    def _should_hold_thrusters_for_navigation_loss(self, now_sec: float, reason: str) -> bool:
        if not self._vision_loss_hold_modes:
            return False
        if self._last_thruster_command is None:
            return False
        if not self._fusion_initialized:
            return False
        if self._fusion_vision_mode not in self._vision_loss_hold_modes:
            return False
        if not (
            reason.startswith("fusion_vision_loss_hold")
            or reason.startswith("fusion_vision_mode_not_allowed")
        ):
            return False
        if (
            self._state_timeout_sec > 0.0
            and self._last_fusion_status_sec is not None
            and now_sec - self._last_fusion_status_sec > self._state_timeout_sec
        ):
            return False
        if self._vision_loss_hold_timeout_sec <= 0.0:
            return True
        if self._fusion_age_vision_sec is None:
            return False
        return self._fusion_age_vision_sec <= self._vision_loss_hold_timeout_sec

    def _log_navigation_gate_change(self, ready: bool, reason: str) -> None:
        state = (ready, reason)
        if state == self._last_navigation_gate_state:
            return
        self._last_navigation_gate_state = state
        if ready:
            self.get_logger().info(f"[LOG] controller navigation gate open: {reason}")
        else:
            self.get_logger().warning(f"[LOG] controller navigation gate closed: {reason}")

    def _publish_goal_status(
        self,
        *,
        target_position_world: np.ndarray | None,
        target_orientation_world: np.ndarray | None,
        error_body: np.ndarray | None,
        reached: bool,
    ) -> None:
        reached_msg = Bool()
        reached_msg.data = bool(reached)
        self._goal_reached_pub.publish(reached_msg)

        if target_position_world is not None:
            goal_msg = PoseStamped()
            goal_msg.header.stamp = self._now_time_message()
            goal_msg.header.frame_id = self._frame_id
            goal_msg.pose.position.x = float(target_position_world[0])
            goal_msg.pose.position.y = float(target_position_world[1])
            goal_msg.pose.position.z = float(target_position_world[2])
            if target_orientation_world is not None:
                goal_msg.pose.orientation.x = float(target_orientation_world[0])
                goal_msg.pose.orientation.y = float(target_orientation_world[1])
                goal_msg.pose.orientation.z = float(target_orientation_world[2])
                goal_msg.pose.orientation.w = float(target_orientation_world[3])
            else:
                goal_msg.pose.orientation.w = 1.0
            self._goal_active_pub.publish(goal_msg)

        if error_body is not None:
            error_msg = Vector3Stamped()
            error_msg.header.stamp = self._now_time_message()
            error_msg.header.frame_id = "controller_body"
            error_msg.vector.x = float(error_body[0])
            error_msg.vector.y = float(error_body[1])
            error_msg.vector.z = float(error_body[2])
            self._goal_error_pub.publish(error_msg)

    def _resolve_target_orientation(self, requested_orientation_world: np.ndarray | None) -> np.ndarray | None:
        if requested_orientation_world is None:
            return None
        if not self._use_target_orientation:
            self.get_logger().warning(
                "received target orientation, but `use_target_orientation` is false in the current YAML; "
                "orientation target will be ignored"
            )
            return None
        if self._backend.observation_mode not in {
            "pose13",
            "pose14",
            "pose16_rot6d",
            "pose20",
            "isaaclab_pose17",
        }:
            self.get_logger().warning(
                f"received target orientation, but backend `{self._backend.name}` uses observation "
                f"`{self._backend.observation_mode}` and cannot consume target orientation; ignoring it"
            )
            return None
        if not self._backend.supports_target_orientation:
            self.get_logger().warning(
                f"received target orientation, but backend `{self._backend.name}` is marked as not using target "
                "orientation in ROS2; orientation target will be ignored"
            )
            return None
        target_orientation_mode = self._backend.target_orientation_mode
        if target_orientation_mode == "yaw_only":
            requested_rpy_deg = _quat_to_controller_rpy_deg(requested_orientation_world)
            if abs(float(requested_rpy_deg[0])) > 1e-3 or abs(float(requested_rpy_deg[1])) > 1e-3:
                self.get_logger().warning(
                    f"backend `{self._backend.name}` only supports yaw alignment in ROS2; "
                    "requested roll/pitch will be ignored"
                )
            return _yaw_only_target_quat(requested_orientation_world)
        return quat_normalize(requested_orientation_world)

    def _set_position_command(
        self,
        target_position_world: np.ndarray,
        source_frame: str,
        target_orientation_world: np.ndarray | None = None,
    ) -> None:
        if self._control_mode != CONTROL_MODE_ROS_MANUAL:
            self.get_logger().warning(
                f"ignored position command while control_mode=`{self._control_mode}`; "
                f"switch to `{CONTROL_MODE_ROS_MANUAL}` first"
            )
            return
        if self._require_target_orientation and target_orientation_world is None:
            self.get_logger().warning(
                "ignored position command without target orientation because "
                "`require_target_orientation` is true in the current YAML"
            )
            return
        resolved_orientation = self._resolve_target_orientation(target_orientation_world)
        self._active_command = MotionCommand(
            mode="position",
            source_frame=source_frame,
            issued_at_sec=self._now_sec(),
            target_position_world=np.asarray(target_position_world, dtype=np.float32).reshape(3),
            target_orientation_world=resolved_orientation,
        )
        self._goal_reached_latched = False
        self._goal_within_tolerance_since_sec = None
        self._last_policy_action = None
        self._last_policy_action_sec = None
        self._reset_stuck_recovery(phase="new_position")
        self._backend.reset()
        orientation_suffix = (
            f", target_yaw_deg={_controller_yaw_deg(resolved_orientation):+.2f}"
            if resolved_orientation is not None
            else ", target_yaw_deg=<none>"
        )
        self.get_logger().info(
            "new position command accepted: "
            f"frame={source_frame}, target=[{target_position_world[0]:+.3f}, "
            f"{target_position_world[1]:+.3f}, {target_position_world[2]:+.3f}]{orientation_suffix}"
        )

    def _goal_global_callback(self, msg: PoseStamped) -> None:
        if not _is_exact_frame(msg.header.frame_id, self._frame_id):
            self.get_logger().warning(
                "ignored controller-world goal with invalid frame_id="
                f"{msg.header.frame_id or '<empty>'}; expected {self._frame_id}"
            )
            return
        target = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        target_orientation_world = _optional_quat(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        if target_orientation_world is not None:
            self.get_logger().info(
                "received controller-world position goal orientation: "
                f"frame_id={msg.header.frame_id or '<empty>'}, "
                f"target_yaw_deg={_controller_yaw_deg(target_orientation_world):+.2f}, "
                f"quat_xyzw=[{target_orientation_world[0]:+.4f}, {target_orientation_world[1]:+.4f}, "
                f"{target_orientation_world[2]:+.4f}, {target_orientation_world[3]:+.4f}]"
            )
        elif self._require_target_orientation:
            self.get_logger().warning(
                "received controller-world position goal without orientation quaternion; "
                "check that send_position_goal was rebuilt and called with --yaw"
            )
        self._set_position_command(target, "controller_world", target_orientation_world)

    def _goal_body_callback(self, msg: Vector3Stamped) -> None:
        now_sec = self._now_sec()
        self._estimator.advance(now_sec)
        state = self._estimator.snapshot(now_sec)
        ready, reason = self._check_navigation_ready(now_sec)
        self._log_navigation_gate_change(ready, reason)
        if not ready or state is None:
            self.get_logger().warning("ignored body-frame goal because vehicle state is not ready yet")
            return

        if not _is_exact_frame(msg.header.frame_id, "controller_body"):
            self.get_logger().warning(
                "ignored controller-body goal with invalid frame_id="
                f"{msg.header.frame_id or '<empty>'}; expected controller_body"
            )
            return

        offset_body = np.array([msg.vector.x, msg.vector.y, msg.vector.z], dtype=np.float32)
        target_world = state.position_world + rotate_vector(state.orientation_world_body, offset_body)
        self._set_position_command(target_world, "controller_body")

    def _goal_body_pose_callback(self, msg: PoseStamped) -> None:
        now_sec = self._now_sec()
        self._estimator.advance(now_sec)
        state = self._estimator.snapshot(now_sec)
        ready, reason = self._check_navigation_ready(now_sec)
        self._log_navigation_gate_change(ready, reason)
        if not ready or state is None:
            self.get_logger().warning("ignored body-frame pose goal because vehicle state is not ready yet")
            return

        if not _is_exact_frame(msg.header.frame_id, "controller_body"):
            self.get_logger().warning(
                "ignored controller-body pose goal with invalid frame_id="
                f"{msg.header.frame_id or '<empty>'}; expected controller_body"
            )
            return

        offset_body = np.array([msg.pose.position.x, msg.pose.position.y, msg.pose.position.z], dtype=np.float32)
        target_world = state.position_world + rotate_vector(state.orientation_world_body, offset_body)
        relative_orientation_body = _optional_quat(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        target_orientation_world = (
            quat_multiply(state.orientation_world_body, relative_orientation_body)
            if relative_orientation_body is not None
            else None
        )
        self._set_position_command(target_world, "controller_body", target_orientation_world)

    def _trajectory_callback(self, msg: MultiDOFJointTrajectory) -> None:
        if self._control_mode != CONTROL_MODE_ROS_MANUAL:
            self.get_logger().warning("ignored trajectory command while controller is not in ros_manual mode")
            return
        accepts_trajectory_reference = (
            self._backend.observation_mode == "trajectory30"
            or self._backend.name == "traditional_pid_position"
        )
        if not accepts_trajectory_reference:
            self.get_logger().warning(
                f"backend `{self._backend.name}` uses `{self._backend.observation_mode}` and does not accept "
                "a T2 MultiDOFJointTrajectory reference"
            )
            return
        if not _is_exact_frame(msg.header.frame_id, self._frame_id):
            self.get_logger().warning(
                f"ignored trajectory with frame_id={msg.header.frame_id or '<empty>'}; expected {self._frame_id}"
            )
            return
        points: list[TrajectoryPoint] = []
        for point in msg.points:
            if not point.transforms:
                continue
            transform = point.transforms[0]
            time_sec = float(point.time_from_start.sec) + float(point.time_from_start.nanosec) * 1e-9
            orientation = _optional_quat(
                [transform.rotation.x, transform.rotation.y, transform.rotation.z, transform.rotation.w]
            )
            if orientation is None:
                orientation = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
            velocity = np.zeros(3, dtype=np.float32)
            if point.velocities:
                velocity_message = point.velocities[0]
                velocity = np.array(
                    [velocity_message.linear.x, velocity_message.linear.y, velocity_message.linear.z], dtype=np.float32
                )
            points.append(
                TrajectoryPoint(
                    max(0.0, time_sec),
                    np.array([transform.translation.x, transform.translation.y, transform.translation.z], dtype=np.float32),
                    orientation,
                    velocity,
                )
            )
        points.sort(key=lambda value: value.time_sec)
        if len(points) < 2:
            self.get_logger().warning("ignored trajectory command: at least two points are required")
            return
        # A sender may omit velocity. Derive it once from the supplied spline
        # points; the control loop then only interpolates deterministic data.
        resolved: list[TrajectoryPoint] = []
        for index, point in enumerate(points):
            velocity = point.linear_velocity_world
            if float(np.linalg.norm(velocity)) <= 1e-8:
                left = points[max(index - 1, 0)]
                right = points[min(index + 1, len(points) - 1)]
                delta = max(right.time_sec - left.time_sec, 1e-6)
                velocity = ((right.position_world - left.position_world) / delta).astype(np.float32)
            resolved.append(TrajectoryPoint(point.time_sec, point.position_world, point.orientation_world, velocity))
        # A trajectory can arrive between explicit lockstep callbacks, while
        # this node's regular ``use_sim_time`` clock still reflects an older
        # delivery.  Mark it pending and establish t=0 at the next controller
        # tick, rather than introducing a fictitious elapsed segment before
        # the first policy action.
        now_sec = math.nan if getattr(self, "_lockstep_enabled", False) else self._now_sec()
        self._active_command = MotionCommand(
            # Keep the established position-command lifecycle while the
            # trajectory payload selects the non-resetting T2 branch in the
            # control tick.
            mode="position",
            source_frame="controller_world",
            issued_at_sec=now_sec,
            trajectory_points=tuple(resolved),
            trajectory_duration_sec=resolved[-1].time_sec,
        )
        self._goal_reached_latched = False
        self._goal_within_tolerance_since_sec = None
        self._last_policy_action = None
        self._last_policy_action_sec = None
        self._reset_stuck_recovery(phase="new_trajectory")
        self._backend.reset()
        self.get_logger().info(
            f"new trajectory command accepted: points={len(resolved)}, duration_sec={resolved[-1].time_sec:.3f}, "
            f"topic={self._command_trajectory_topic}"
        )

    def _velocity_command_callback(self, msg: TwistStamped) -> None:
        if self._control_mode != CONTROL_MODE_ROS_MANUAL:
            self.get_logger().warning(
                f"ignored velocity command while control_mode=`{self._control_mode}`; "
                f"switch to `{CONTROL_MODE_ROS_MANUAL}` first"
            )
            return
        if self._backend.observation_mode != "velocity_normalized_body":
            self.get_logger().warning(
                f"backend `{self._backend.name}` does not accept direct velocity commands; "
                "use a velocity backend or send a position command instead"
            )
            return

        frame_hint = (msg.header.frame_id or "").strip().lower()
        if frame_hint not in {"controller_world", "controller_body"}:
            self.get_logger().warning(
                "ignored velocity command with invalid frame_id="
                f"{msg.header.frame_id or '<empty>'}; expected controller_world or controller_body"
            )
            return
        linear_ros = np.array(
            [msg.twist.linear.x, msg.twist.linear.y, msg.twist.linear.z],
            dtype=np.float32,
        )
        angular_ros = np.array(
            [msg.twist.angular.x, msg.twist.angular.y, msg.twist.angular.z],
            dtype=np.float32,
        )

        if frame_hint == "controller_world":
            now_sec = self._now_sec()
            self._estimator.advance(now_sec)
            state = self._estimator.snapshot(now_sec)
            ready, reason = self._check_navigation_ready(now_sec)
            self._log_navigation_gate_change(ready, reason)
            if not ready or state is None:
                self.get_logger().warning("ignored world-frame velocity command because vehicle state is not ready yet")
                return
            linear_world = linear_ros
            angular_world = angular_ros
            linear_body = inverse_rotate_vector(state.orientation_world_body, linear_world)
            angular_body_xyz = inverse_rotate_vector(state.orientation_world_body, angular_world)
            source_frame = "controller_world"
        else:
            linear_body = linear_ros
            angular_body_xyz = angular_ros
            source_frame = "controller_body"

        angular_body_ypr = np.array(
            [
                angular_body_xyz[1],
                angular_body_xyz[2],
                angular_body_xyz[0],
            ],
            dtype=np.float32,
        )

        previous_command = self._active_command
        is_velocity_refresh = (
            previous_command is not None
            and previous_command.mode == "velocity"
            and previous_command.source_frame == source_frame
        )

        self._active_command = MotionCommand(
            mode="velocity",
            source_frame=source_frame,
            issued_at_sec=self._now_sec(),
            desired_linear_velocity_body=np.asarray(linear_body, dtype=np.float32).reshape(3),
            desired_angular_velocity_ypr=np.asarray(angular_body_ypr, dtype=np.float32).reshape(3),
        )
        if not is_velocity_refresh:
            self._goal_reached_latched = False
            self._goal_within_tolerance_since_sec = None
            self._reset_stuck_recovery(phase="new_velocity")
            self._backend.reset()
            self._last_policy_action = None
            self._last_policy_action_sec = None
            self.get_logger().info(
                "new velocity command accepted: "
                f"frame={source_frame}, linear_body=[{linear_body[0]:+.3f}, {linear_body[1]:+.3f}, {linear_body[2]:+.3f}], "
                f"angular_ypr=[{angular_body_ypr[0]:+.3f}, {angular_body_ypr[1]:+.3f}, {angular_body_ypr[2]:+.3f}]"
            )

    def _goal_cancel_callback(self, _msg: Empty) -> None:
        self._active_command = None
        self._last_policy_action = None
        self._last_policy_action_sec = None
        self._goal_reached_latched = False
        self._goal_within_tolerance_since_sec = None
        self._reset_stuck_recovery(phase="cancel")
        self._publish_zero_pwm()
        self._publish_goal_status(target_position_world=None, target_orientation_world=None, error_body=None, reached=False)
        self.get_logger().info("active command cancelled")

    def _safe_control_step(self) -> None:
        try:
            self._control_step()
            self._last_control_exception = None
        except Exception as exc:
            exception_name = type(exc).__name__
            exception_message = str(exc)
            exception_key = f"{exception_name}: {exception_message}"
            self._active_command = None
            self._goal_reached_latched = False
            self._goal_within_tolerance_since_sec = None
            self._reset_stuck_recovery(phase="exception")
            self._publish_zero_pwm()
            self._publish_goal_status(
                target_position_world=None,
                target_orientation_world=None,
                error_body=None,
                reached=False,
            )
            if exception_key != self._last_control_exception:
                self.get_logger().error(
                    "control step failed; active command cleared and zero PWM published. "
                    f"{exception_key}\n{traceback.format_exc()}"
                )
                self._last_control_exception = exception_key

    def _goal_is_reached(
        self,
        error_world: np.ndarray,
        linear_velocity_body: np.ndarray,
        current_orientation_world: np.ndarray,
        target_orientation_world: np.ndarray | None,
    ) -> tuple[bool, float | None]:
        position_error = float(np.linalg.norm(error_world))
        velocity_norm = float(np.linalg.norm(linear_velocity_body))
        orientation_error_deg = None
        if target_orientation_world is not None:
            if self._backend.target_orientation_mode == "yaw_only":
                orientation_error_deg = _yaw_error_deg(current_orientation_world, target_orientation_world)
            else:
                orientation_error_deg = float(
                    np.degrees(quat_angle_error_rad(current_orientation_world, target_orientation_world))
                )
        reached = position_error <= self._position_tolerance and velocity_norm <= self._velocity_tolerance
        if orientation_error_deg is not None:
            reached = reached and orientation_error_deg <= self._orientation_tolerance_deg
        return reached, orientation_error_deg

    def _control_step(self) -> None:
        if self._control_mode == CONTROL_MODE_UNITY_RANDOM:
            return

        now_sec = self._now_sec()
        self._estimator.advance(now_sec)
        state = self._estimator.snapshot(now_sec)

        if state is None or self._active_command is None:
            if now_sec - self._last_idle_zero_sec >= 0.5:
                self._publish_zero_pwm()
                self._last_idle_zero_sec = now_sec
            if self._active_command is None:
                self._publish_goal_status(
                    target_position_world=None,
                    target_orientation_world=None,
                    error_body=None,
                    reached=False,
                )
                self._publish_stuck_status(active=False, phase="idle")
            return

        ready, reason = self._check_navigation_ready(now_sec)
        self._log_navigation_gate_change(ready, reason)
        if not ready:
            self._goal_reached_latched = False
            self._goal_within_tolerance_since_sec = None
            self._reset_stuck_recovery(phase="navigation_not_ready")
            if now_sec - self._last_idle_zero_sec >= 0.2:
                if self._should_hold_thrusters_for_navigation_loss(now_sec, reason):
                    self._publish_thruster_hold()
                else:
                    self._publish_zero_pwm()
                self._last_idle_zero_sec = now_sec
            self._publish_goal_status(
                target_position_world=None,
                target_orientation_world=None,
                error_body=None,
                reached=False,
            )
            self._publish_stuck_status(active=False, phase="navigation_not_ready")
            return

        if self._active_command.mode == "position":
            trajectory_points = self._active_command.trajectory_points
            is_trajectory = trajectory_points is not None and len(trajectory_points) >= 2
            trajectory_reference = None
            if is_trajectory:
                from .trajectory_tracking import sample_trajectory

                if not math.isfinite(self._active_command.issued_at_sec):
                    # The first control tick after trajectory receipt is the
                    # auditable t=0 boundary of a lockstep T2 trial.
                    self._active_command.issued_at_sec = now_sec
                elapsed_sec = max(0.0, now_sec - self._active_command.issued_at_sec)
                trajectory_reference = sample_trajectory(trajectory_points, elapsed_sec)
                target_world = trajectory_reference.position_world
                # T2 transports an identity orientation alongside every
                # reference position, but translation-only profiles must not
                # interpret it as a yaw=0 command.  The pose observation then
                # uses the measured orientation as its target and the
                # traditional PID yaw loop is explicitly disabled by profile.
                target_orientation_world = (
                    trajectory_reference.orientation_world
                    if self._use_target_orientation
                    else None
                )
            elif self._active_command.target_position_world is None:
                self.get_logger().error("position command is missing target_position_world")
                self._publish_zero_pwm()
                return
            else:
                target_world = self._active_command.target_position_world
                target_orientation_world = self._active_command.target_orientation_world
            error_world = (target_world - state.position_world).astype(np.float32)
            error_body = self._estimator.world_error_to_body(error_world)
            position_error_m = float(np.linalg.norm(error_world))
            speed_norm_mps = float(np.linalg.norm(state.linear_velocity_body))
            if is_trajectory:
                # A completed trajectory stays on its final reference; it is
                # never treated as a static goal and therefore never resets or
                # stops the policy in the middle of a path.
                within_tolerance, orientation_error_deg = False, None
            else:
                within_tolerance, orientation_error_deg = self._goal_is_reached(
                    error_world,
                    state.linear_velocity_body,
                    state.orientation_world_body,
                    target_orientation_world,
                )
            reached, self._goal_within_tolerance_since_sec, held_duration_sec = _update_goal_hold_state(
                within_tolerance=within_tolerance,
                now_sec=now_sec,
                entered_sec=self._goal_within_tolerance_since_sec,
                hold_time_sec=self._goal_reach_hold_time_sec,
            )

            self._publish_goal_status(
                target_position_world=target_world,
                target_orientation_world=target_orientation_world,
                error_body=error_body,
                reached=reached,
            )
            if is_trajectory and self._debug_trajectory_reference_pub is not None:
                reference_msg = Float32MultiArray()
                reference_msg.data = [
                    float(max(0.0, now_sec - self._active_command.issued_at_sec)),
                    float(target_world[0]), float(target_world[1]), float(target_world[2]),
                    float(trajectory_reference.linear_velocity_world[0]),
                    float(trajectory_reference.linear_velocity_world[1]),
                    float(trajectory_reference.linear_velocity_world[2]),
                    float(self._active_command.trajectory_duration_sec or 0.0),
                ]
                self._debug_trajectory_reference_pub.publish(reference_msg)
            if is_trajectory and self._debug_trajectory_reference_stamped_pub is not None:
                stamped_reference = PoseStamped()
                stamped_reference.header.stamp = self._now_time_message()
                stamped_reference.header.frame_id = self._frame_id
                stamped_reference.pose.position.x = float(target_world[0])
                stamped_reference.pose.position.y = float(target_world[1])
                stamped_reference.pose.position.z = float(target_world[2])
                # T2 is translation-only.  Identity makes this a stamped
                # reference-position audit stream, not an attitude command.
                stamped_reference.pose.orientation.w = 1.0
                self._debug_trajectory_reference_stamped_pub.publish(stamped_reference)
            if reached:
                if not self._goal_reached_latched:
                    orientation_suffix = (
                        f", orientation_error_deg={orientation_error_deg:.2f}"
                        if orientation_error_deg is not None
                        else ""
                    )
                    self.get_logger().info(
                        "position command reached: "
                        f"error_world_norm={float(np.linalg.norm(error_world)):.4f}, "
                        f"speed_norm={float(np.linalg.norm(state.linear_velocity_body)):.4f}, "
                        f"held_duration_sec={held_duration_sec:.2f}{orientation_suffix}"
                    )
                self._goal_reached_latched = True
                self._reset_stuck_recovery(phase="reached")
                if self._stop_on_goal_reached:
                    self._publish_zero_pwm()
                    self._active_command = None
                    self._goal_within_tolerance_since_sec = None
                    return
            else:
                self._goal_reached_latched = False

            if not is_trajectory and self._stuck_escape_active(now_sec):
                self._publish_stuck_status(
                    active=True,
                    phase="escape",
                    position_error_m=position_error_m,
                    speed_norm_mps=speed_norm_mps,
                    horizontal_effort=float(np.max(np.abs(self._stuck_escape_action[4:8]))),
                )
                self._publish_thruster_action(self._stuck_escape_action)
                return
            if self._stuck_phase == "escape":
                self._stuck_phase = "cooldown"

            if self._backend.observation_mode == "trajectory30":
                if not is_trajectory:
                    self.get_logger().error("trajectory30 backend requires a MultiDOFJointTrajectory command")
                    self._publish_zero_pwm()
                    return
                observation, _ = build_trajectory30_observation(
                    state,
                    trajectory_points,
                    max(0.0, now_sec - self._active_command.issued_at_sec),
                    preview_step_sec=self._trajectory_preview_step_sec,
                    preview_offset_scale=self._trajectory_preview_offset_scale,
                    linear_velocity_scale=self._trajectory_linear_velocity_scale,
                    angular_velocity_scale=self._trajectory_angular_velocity_scale,
                    observation_clip=self._trajectory_observation_clip,
                    trajectory_duration_sec=self._active_command.trajectory_duration_sec,
                )
            elif self._backend.observation_mode == "isaaclab_pose17":
                from finssim_isaaclab.observation import build_warpauv_observation

                observation = build_warpauv_observation(
                    target_position_fins=target_world,
                    target_quaternion_fins_xyzw=(
                        target_orientation_world
                        if target_orientation_world is not None
                        else state.orientation_world_body
                    ),
                    position_fins=state.position_world,
                    orientation_fins_xyzw=state.orientation_world_body,
                    linear_velocity_body_fins=state.linear_velocity_body,
                    angular_velocity_body_fins=state.angular_velocity_body_xyz,
                )
            elif self._backend.observation_mode == "pose13":
                observation = build_pose13_observation(state, target_world)
            elif self._backend.observation_mode == "pose14":
                observation = build_pose14_observation(state, target_world, target_orientation_world)
            elif self._backend.observation_mode == "pose16_rot6d":
                observation = build_pose16_rot6d_observation(
                    state,
                    target_world,
                    target_orientation_world,
                    position_scale=self._pose16_position_scale,
                    linear_velocity_scale=self._pose16_linear_velocity_scale,
                    angular_velocity_scale=self._pose16_angular_velocity_scale,
                    velocity_clip=self._pose16_velocity_clip,
                )
            elif self._backend.observation_mode == "pose20":
                observation = build_pose20_observation(state, target_world, target_orientation_world)
                if self._backend.name == "traditional_pid_position":
                    observation = np.concatenate(
                        [
                            observation,
                            state.position_ros.astype(np.float32, copy=False),
                        ],
                        axis=0,
                    ).astype(np.float32, copy=False)
            elif self._backend.observation_mode == "velocity_normalized_body":
                observation, _, _ = build_velocity_normalized_observation(
                    state,
                    target_position_world=target_world,
                    linear_velocity_scale=self._backend.linear_velocity_scale,
                    angular_velocity_scale=self._backend.angular_velocity_scale,
                    outer_loop_kp=self._outer_loop_kp,
                    max_body_velocity=np.minimum(
                        self._max_body_velocity,
                        np.asarray(self._backend.reference_velocity_limit, dtype=np.float32),
                    ),
                )
            else:
                self.get_logger().error(f"unknown observation mode `{self._backend.observation_mode}`")
                self._publish_zero_pwm()
                return
        elif self._active_command.mode == "velocity":
            desired_linear = self._active_command.desired_linear_velocity_body
            desired_angular = self._active_command.desired_angular_velocity_ypr
            if desired_linear is None:
                self.get_logger().error("velocity command is missing desired_linear_velocity_body")
                self._publish_zero_pwm()
                return
            self._publish_goal_status(
                target_position_world=None,
                target_orientation_world=None,
                error_body=None,
                reached=False,
            )
            if self._backend.observation_mode != "velocity_normalized_body":
                self.get_logger().error(
                    f"backend `{self._backend.name}` cannot run direct velocity mode with observation "
                    f"`{self._backend.observation_mode}`"
                )
                self._publish_zero_pwm()
                return
            observation, _, _ = build_velocity_normalized_observation(
                state,
                target_position_world=None,
                linear_velocity_scale=self._backend.linear_velocity_scale,
                angular_velocity_scale=self._backend.angular_velocity_scale,
                max_body_velocity=np.minimum(
                    self._max_body_velocity,
                    np.asarray(self._backend.reference_velocity_limit, dtype=np.float32),
                ),
                desired_linear_velocity_body=desired_linear,
                desired_angular_velocity_ypr=desired_angular,
            )
        else:
            self.get_logger().error(f"unknown command mode `{self._active_command.mode}`")
            self._publish_zero_pwm()
            return

        output_mode = getattr(self, "_thruster_output_mode", THRUSTER_OUTPUT_FORCE_N)
        if output_mode == THRUSTER_OUTPUT_WRENCH6D_FORCE_N:
            action = np.asarray(self._backend.predict_policy_action(observation), dtype=np.float32).reshape(-1)
            if action.shape[0] != WRENCH6D_POLICY_DIM:
                raise RuntimeError(
                    f"wrench6d_force_n requires a 6D raw policy action, got {action.shape[0]} values. "
                    "Use a PPO v2/wrench checkpoint trained with a 6D policy head."
                )
        else:
            action = self._backend.predict_action(observation)
            expected_action_dim = 6 if output_mode == THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N else THRUSTER_COUNT
            if action.shape[0] < expected_action_dim:
                action = np.pad(action, (0, expected_action_dim - action.shape[0]), constant_values=0.0)
            elif action.shape[0] > expected_action_dim:
                action = action[:expected_action_dim]
        action = np.clip(action.astype(np.float32, copy=False), -1.0, 1.0)
        previous_action_time = self._last_policy_action_sec
        if self._last_policy_action is None and self._policy_action_rate_limit_per_sec > 0.0:
            self._last_policy_action = np.zeros_like(action, dtype=np.float32)
        delta_time_sec = (
            1.0 / max(self._control_rate_hz, 1e-6)
            if previous_action_time is None
            else max(now_sec - previous_action_time, 0.0)
        )
        action = rate_limit_policy_action(
            action,
            self._last_policy_action,
            delta_time_sec,
            self._policy_action_rate_limit_per_sec,
        )
        self._last_policy_action = action.astype(np.float32, copy=True)
        self._last_policy_action_sec = now_sec
        if self._active_command.mode == "position" and not (
            self._active_command.trajectory_points is not None and len(self._active_command.trajectory_points) >= 2
        ):
            triggered, horizontal_effort, stuck_held_duration_sec = self._update_stuck_recovery(
                now_sec=now_sec,
                position_error_m=position_error_m,
                speed_norm_mps=speed_norm_mps,
                action=action,
            )
            self._publish_stuck_status(
                active=triggered or self._stuck_phase in {"candidate", "escape", "cooldown"},
                phase=self._stuck_phase,
                position_error_m=position_error_m,
                speed_norm_mps=speed_norm_mps,
                horizontal_effort=horizontal_effort,
                held_duration_sec=stuck_held_duration_sec,
            )
            if triggered:
                action = self._stuck_escape_action
        else:
            self._publish_stuck_status(active=False, phase="velocity_mode")

        command = self._publish_thruster_action(action)
        if hasattr(self, "_publish_control_debug"):
            self._publish_control_debug(
                observation=observation,
                action=action,
                command=command,
                state=state,
                error_body=error_body if self._active_command.mode == "position" else None,
                position_error_m=position_error_m if self._active_command.mode == "position" else None,
                speed_norm_mps=speed_norm_mps if self._active_command.mode == "position" else None,
                desired_body_velocity=desired_body_velocity if self._active_command.mode == "velocity" else None,
                wrench6d_command=getattr(self, "_last_wrench6d_command", None),
            )

    def _action_to_thruster_command(self, action: np.ndarray) -> np.ndarray:
        if self._thruster_output_mode == THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N:
            command = isaaclab_finsrov_calibrated_action_to_force_n(
                action,
                force_limit_positive=self._thruster_force_limit_positive,
                force_limit_negative=self._thruster_force_limit_negative,
            )
            self._last_wrench6d_command = None
            return (command * self._thruster_output_scale).astype(np.float32, copy=False)
        if self._thruster_output_mode == THRUSTER_OUTPUT_ISAACLAB_VIRTUAL_WRENCH_FORCE_N:
            adapter = getattr(self, "_isaaclab_adapter", None)
            if adapter is None:
                raise RuntimeError("IsaacLab output mode is enabled but its wrench adapter is not initialized")
            self._last_wrench6d_command = adapter.action_to_fins_wrench(action)
            normalized_action = self._wrench6d_to_normalized_thruster_action(self._last_wrench6d_command)
            command = action_to_thruster_command(
                normalized_action,
                mode=THRUSTER_OUTPUT_FORCE_N,
                force_limit_positive=self._thruster_force_limit_positive,
                force_limit_negative=self._thruster_force_limit_negative,
            )
            return (command * self._thruster_output_scale).astype(np.float32, copy=False)
        if self._thruster_output_mode == THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N:
            # Preserve the source 8-thruster policy's force semantics before
            # projecting into body wrench coordinates. The projection gain is
            # separate from final output scaling to avoid applying it twice.
            source_force_n = action_to_thruster_command(
                action,
                mode=THRUSTER_OUTPUT_FORCE_N,
                force_limit_positive=self._thruster_force_limit_positive,
                force_limit_negative=self._thruster_force_limit_negative,
            )
            source_force_n = source_force_n * self._thruster8_projection_input_scale
            allocator = getattr(self, "_wrench6d_thrust_allocator", None)
            if allocator is None or not hasattr(allocator, "B"):
                raise RuntimeError("thruster8 reprojection requires an initialized physical ThrustAllocator")
            body_wrench = allocator.B.detach().cpu().numpy() @ source_force_n
            self._last_wrench6d_command = scale_allocator_body_wrench(
                body_wrench,
                wrench_scale=self._wrench6d_wrench_scale,
                policy_axis_order=self._wrench6d_policy_axis_order,
            )
            normalized_action = self._wrench6d_to_normalized_thruster_action(self._last_wrench6d_command)
            command = action_to_thruster_command(
                normalized_action,
                mode=THRUSTER_OUTPUT_FORCE_N,
                force_limit_positive=self._thruster_force_limit_positive,
                force_limit_negative=self._thruster_force_limit_negative,
            )
            return (command * self._thruster_output_scale).astype(np.float32, copy=False)
        if self._thruster_output_mode == THRUSTER_OUTPUT_WRENCH6D_FORCE_N:
            self._last_wrench6d_command = policy_action6d_to_allocator_input(
                action,
                wrench_limits=self._wrench6d_wrench_limits,
                wrench_scale=getattr(self, "_wrench6d_wrench_scale", DEFAULT_WRENCH6D_SCALE),
                policy_axis_order=getattr(self, "_wrench6d_policy_axis_order", "legacy_surge_sway_heave"),
            )
            normalized_action = self._wrench6d_to_normalized_thruster_action(self._last_wrench6d_command)
            command = action_to_thruster_command(
                normalized_action,
                mode=THRUSTER_OUTPUT_FORCE_N,
                force_limit_positive=self._thruster_force_limit_positive,
                force_limit_negative=self._thruster_force_limit_negative,
            )
            return (command * self._thruster_output_scale).astype(np.float32, copy=False)
        else:
            self._last_wrench6d_command = None
        command = action_to_thruster_command(
            action,
            mode=self._thruster_output_mode,
            force_limit_positive=self._thruster_force_limit_positive,
            force_limit_negative=self._thruster_force_limit_negative,
        )
        return (command * self._thruster_output_scale).astype(np.float32, copy=False)

    def _wrench6d_to_normalized_thruster_action(self, wrench: np.ndarray) -> np.ndarray:
        allocator = getattr(self, "_wrench6d_thrust_allocator", None)
        if allocator is None:
            from finssim_rl.models.thrust_allocator import ThrustAllocator

            allocator = ThrustAllocator(
                device=getattr(self, "_device", "cpu"),
                deadzone_comp=0.0,
                control_linear_range=1.0,
                control_axis_ranges=(1.0,) * WRENCH6D_POLICY_DIM,
                allocation_mode=getattr(self, "_wrench6d_allocation_mode", "empirical_thruster_mixer"),
                physical_wrench_limits=getattr(self, "_wrench6d_wrench_limits", (1.0,) * WRENCH6D_POLICY_DIM),
                thruster_force_limit_positive=getattr(self, "_thruster_force_limit_positive", (7.0,) * THRUSTER_COUNT),
                thruster_force_limit_negative=getattr(self, "_thruster_force_limit_negative", (7.0,) * THRUSTER_COUNT),
                debug_print_interval=0,
            )
            self._wrench6d_thrust_allocator = allocator

        import torch

        device = getattr(allocator, "device", "cpu")
        wrench_tensor = torch.as_tensor(np.asarray(wrench, dtype=np.float32), dtype=torch.float32, device=device)
        with torch.no_grad():
            thrust = allocator(wrench_tensor)
        if hasattr(thrust, "detach"):
            thrust_np = thrust.detach().cpu().numpy()
        else:
            thrust_np = np.asarray(thrust, dtype=np.float32)
        thrust_np = np.asarray(thrust_np, dtype=np.float32).reshape(-1)
        if thrust_np.shape[0] < THRUSTER_COUNT:
            thrust_np = np.pad(thrust_np, (0, THRUSTER_COUNT - thrust_np.shape[0]), constant_values=0.0)
        elif thrust_np.shape[0] > THRUSTER_COUNT:
            thrust_np = thrust_np[:THRUSTER_COUNT]
        return np.clip(thrust_np, -1.0, 1.0).astype(np.float32, copy=False)


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = MotionControllerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
