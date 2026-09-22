from __future__ import annotations

import csv
from dataclasses import dataclass
from collections import deque
from itertools import product
import json
import math
from pathlib import Path
import threading
from typing import Sequence

import numpy as np
import rclpy
import yaml
from msgs.msg import HardwareTelemetry
from geometry_msgs.msg import TwistWithCovarianceStamped
from rcl_interfaces.msg import ParameterDescriptor
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray
from std_srvs.srv import Trigger


THRUSTER_COUNT = 8
THRUSTER_NAMES = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")
STEP_AXIS_ORDER = ("surge_x", "heave_y", "sway_z", "yaw_y")
ATTITUDE_PULSE_AXIS_ORDER = ("roll_x", "pitch_z")
AXIS_ORDER = ("surge_x", "heave_y", "sway_z", "roll_x", "pitch_z", "yaw_y")
HEAVE_IDENTIFICATION_MODES = ("step", "dive_and_coast")

# Existing FinsSim control frame: x forward, y up, z left.
# Wrench order is [Fx, Fy, Fz, Mx, My, Mz]; yaw is My in this frame.
# Roll follows the right-hand rule Mx = r_y * F_z - r_z * F_y:
# left-side vertical thrusters contribute negative Mx and right-side
# vertical thrusters contribute positive Mx.
ALLOCATOR_A = np.asarray(
    [
        [0.000000, 0.249400, 0.000000, -1.601400, 0.000000, 1.883600],
        [0.000000, 0.249400, 0.000000, -1.601400, 0.000000, -1.883600],
        [0.000000, 0.249400, 0.000000, 1.601400, 0.000000, -1.883600],
        [0.000000, 0.249400, 0.000000, 1.601400, 0.000000, 1.883600],
        [0.353600, 0.000000, -0.353600, 0.000000, 0.951000, 0.000000],
        [0.353600, 0.000000, 0.353600, 0.000000, 0.951000, 0.000000],
        [-0.353600, 0.000000, 0.353600, 0.000000, 0.951000, 0.000000],
        [-0.353600, 0.000000, -0.353600, 0.000000, 0.951000, 0.000000],
    ],
    dtype=np.float64,
)

# Same effective wrench reconstruction used by the Unity identifier.
WRENCH_FROM_THRUST = np.asarray(
    [
        [0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.707013575, 0.707013575, -0.707013575, -0.707013575],
        [1.002405774, 1.002405774, 1.002405774, 1.002405774, 0.000000000, 0.000000000, 0.000000000, 0.000000000],
        [0.000000000, 0.000000000, 0.000000000, 0.000000000, -0.707013575, 0.707013575, 0.707013575, -0.707013575],
        [-0.156113401, -0.156113401, 0.156113401, 0.156113401, 0.000000000, 0.000000000, 0.000000000, 0.000000000],
        [0.000000000, 0.000000000, 0.000000000, 0.000000000, 0.262881178, 0.262881178, 0.262881178, 0.262881178],
        [0.132724570, -0.132724570, -0.132724570, 0.132724570, 0.000000000, 0.000000000, 0.000000000, 0.000000000],
    ],
    dtype=np.float64,
)

AXIS_TO_TAU_INDEX = {"surge_x": 0, "heave_y": 1, "sway_z": 2, "roll_x": 3, "yaw_y": 4, "pitch_z": 5}
AXIS_TO_NU_INDEX = {"surge_x": 0, "heave_y": 1, "sway_z": 2, "roll_x": 3, "yaw_y": 4, "pitch_z": 5}


@dataclass(frozen=True)
class Trial:
    axis: str
    level: float
    repeat_index: int = 1


@dataclass(frozen=True)
class Phase:
    name: str
    duration: float


@dataclass
class Sample:
    time_sec: float
    axis: str
    phase: str
    phase_elapsed_sec: float
    sample_window: bool
    tau: np.ndarray
    nu: np.ndarray
    nu_dot: np.ndarray
    angle: np.ndarray
    orientation_valid: bool
    synchronization_valid: bool = True


@dataclass(frozen=True)
class TelemetryBufferSample:
    time_sec: float
    frame_id: str
    mcu_time_ms: int
    telemetry_sequence: int
    host_receive_time_ns: int
    orientation_xyzw: np.ndarray
    angular_velocity_xyz: np.ndarray
    linear_acceleration_xyz: np.ndarray
    rpm: np.ndarray


@dataclass(frozen=True)
class DvlBufferSample:
    time_sec: float
    raw_linear: np.ndarray
    controller_linear: np.ndarray
    frame_id: str


@dataclass(frozen=True)
class ThrusterCurve:
    c1_positive: np.ndarray
    c1_negative: np.ndarray
    rpm_min: np.ndarray
    rpm_max: np.ndarray
    force_deadband_n: np.ndarray
    min_effective_rpm_positive: np.ndarray
    min_effective_rpm_negative: np.ndarray
    firmware_max_rpm: float

    @classmethod
    def defaults(cls) -> "ThrusterCurve":
        return cls(
            c1_positive=np.full(THRUSTER_COUNT, 1.0e-4, dtype=np.float64),
            c1_negative=np.full(THRUSTER_COUNT, 1.0e-4, dtype=np.float64),
            rpm_min=np.full(THRUSTER_COUNT, -3000.0, dtype=np.float64),
            rpm_max=np.full(THRUSTER_COUNT, 3000.0, dtype=np.float64),
            force_deadband_n=np.zeros(THRUSTER_COUNT, dtype=np.float64),
            min_effective_rpm_positive=np.zeros(THRUSTER_COUNT, dtype=np.float64),
            min_effective_rpm_negative=np.zeros(THRUSTER_COUNT, dtype=np.float64),
            firmware_max_rpm=3000.0,
        )

    def force_to_rpm(self, forces_n: Sequence[float]) -> np.ndarray:
        forces = _as_vec(forces_n, THRUSTER_COUNT)
        rpm = np.zeros(THRUSTER_COUNT, dtype=np.float64)
        for i, force in enumerate(forces):
            if abs(force) <= self.force_deadband_n[i]:
                continue
            c1 = abs(self.c1_positive[i] if force > 0.0 else self.c1_negative[i])
            value = math.copysign(math.sqrt(abs(force) / max(c1, 1e-12)), force) * 60.0 / (2.0 * math.pi)
            if force > 0.0:
                value = max(value, self.min_effective_rpm_positive[i])
            else:
                value = min(value, -self.min_effective_rpm_negative[i])
            rpm[i] = float(np.clip(value, self.rpm_min[i], self.rpm_max[i]))
        return rpm

    def rpm_to_force(self, rpm_values: Sequence[float]) -> np.ndarray:
        rpm = _as_vec(rpm_values, THRUSTER_COUNT)
        omega = rpm * (2.0 * math.pi / 60.0)
        force = np.zeros(THRUSTER_COUNT, dtype=np.float64)
        positive = omega >= 0.0
        force[positive] = self.c1_positive[positive] * omega[positive] * omega[positive]
        force[~positive] = -self.c1_negative[~positive] * omega[~positive] * omega[~positive]
        return force

    def force_limits(self, scale: float) -> tuple[np.ndarray, np.ndarray]:
        pos = self.rpm_to_force(np.maximum(self.rpm_max, 0.0))
        neg = -self.rpm_to_force(np.minimum(self.rpm_min, 0.0))
        scale = float(np.clip(scale, 0.0, 1.0))
        return np.maximum(pos, 0.0) * scale, np.maximum(neg, 0.0) * scale


def _as_vec(values: Sequence[float], size: int) -> np.ndarray:
    if isinstance(values, str):
        text = values.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        values = [item.strip() for item in text.split(",") if item.strip()]
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < size:
        array = np.pad(array, (0, size - array.size), constant_values=0.0)
    elif array.size > size:
        array = array[:size]
    array[~np.isfinite(array)] = 0.0
    return array.astype(np.float64, copy=False)


def _string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("[") and text.endswith("]"):
            text = text[1:-1]
        return tuple(item.strip().strip("'\"") for item in text.split(",") if item.strip())
    if isinstance(value, Sequence):
        items: list[str] = []
        for item in value:
            items.extend(_string_list(str(item)))
        return tuple(items)
    return ()


def _bool_value(value: object) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _normalize_heave_identification_mode(value: object) -> str:
    mode = str(value).strip().lower()
    if mode not in HEAVE_IDENTIFICATION_MODES:
        raise ValueError(
            f"unknown heave_y_identification_mode `{value}`; "
            f"expected one of {list(HEAVE_IDENTIFICATION_MODES)}"
        )
    return mode


def _normalized_frame(frame_id: str | None) -> str:
    return (frame_id or "").strip().lower()


def _is_controller_frame(frame_id: str | None) -> bool:
    return _normalized_frame(frame_id) in {"controller_world", "controller_body"}


def _build_basis_matrix(indices: Sequence[int], signs: Sequence[float]) -> np.ndarray:
    rows = []
    resolved_indices = tuple(int(v) for v in indices)
    resolved_signs = tuple(float(v) for v in signs)
    if len(resolved_indices) != 3 or len(resolved_signs) != 3:
        raise ValueError("basis indices/signs must both have length 3")
    for source_index, sign in zip(resolved_indices, resolved_signs):
        if source_index < 0 or source_index > 2:
            raise ValueError(f"basis axis index must be in [0, 2], got {source_index}")
        row = np.zeros(3, dtype=np.float64)
        row[source_index] = sign
        rows.append(row)
    return np.stack(rows, axis=0)


def _reorder_vector(vector: Sequence[float], basis_matrix: np.ndarray) -> np.ndarray:
    source = _as_vec(vector, 3)
    return (basis_matrix @ source).astype(np.float64)


def _quat_normalize(quat_xyzw: Sequence[float]) -> tuple[np.ndarray, bool]:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm <= 1e-9 or not math.isfinite(norm):
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64), False
    return quat / norm, True


def _quat_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    quat, _ = _quat_normalize(quat_xyzw)
    x, y, z, w = quat
    xx = x * x
    yy = y * y
    zz = z * z
    xy = x * y
    xz = x * z
    yz = y * z
    wx = w * x
    wy = w * y
    wz = w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat(rotation_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale
    return _quat_normalize([x, y, z, w])[0]


def _transform_quat_basis(quat_xyzw: Sequence[float], basis_matrix: np.ndarray) -> np.ndarray:
    rotation_source = _quat_to_matrix(quat_xyzw)
    rotation_controller = basis_matrix @ rotation_source @ basis_matrix.T
    return _matrix_to_quat(rotation_controller)


def _quat_to_controller(quat_xyzw: Sequence[float], basis_matrix: np.ndarray, frame_id: str | None) -> tuple[np.ndarray, bool]:
    quat, valid = _quat_normalize(quat_xyzw)
    if not valid:
        return quat, False
    if _is_controller_frame(frame_id):
        return quat, True
    return _transform_quat_basis(quat, basis_matrix), True


def _gravity_controller_body_from_quaternion(quat_controller: Sequence[float], gravity_mps2: float) -> np.ndarray:
    """Return stationary accelerometer gravity in controller body axes."""
    gravity_world_controller = np.asarray([0.0, float(gravity_mps2), 0.0], dtype=np.float64)
    return _quat_to_matrix(quat_controller).T @ gravity_world_controller


def _quat_to_controller_rpy_rad(quat_xyzw: Sequence[float]) -> tuple[float, float, float, bool]:
    """Extract ``[roll_x, pitch_z, yaw_y]`` in the controller convention.

    The controller frame is ``x`` forward, ``y`` up, ``z`` left, with
    ``R = Ry(yaw_y) @ Rz(pitch_z) @ Rx(roll_x)``.  This is not ROS's usual
    ZYX Euler order: applying the latter makes a heading near +/-180 degrees
    appear as a pitch near +/-180 degrees and aborts pitch pulse trials.
    """
    quat, valid = _quat_normalize(quat_xyzw)
    if not valid:
        return 0.0, 0.0, 0.0, False
    x, y, z, w = quat

    # Matrix entries for R = Ry(yaw_y) @ Rz(pitch_z) @ Rx(roll_x).
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)

    pitch_z = math.asin(float(np.clip(r10, -1.0, 1.0)))
    yaw_y = math.atan2(-r20, r00)
    roll_x = math.atan2(-r12, r11)
    return roll_x, pitch_z, yaw_y, True


def _vector_to_controller_body(vector: Sequence[float], basis_matrix: np.ndarray, frame_id: str | None) -> np.ndarray:
    source = _as_vec(vector, 3)
    if _is_controller_frame(frame_id):
        return source.astype(np.float64, copy=True)
    return _reorder_vector(source, basis_matrix)


def _axial_vector_to_controller_body(
    vector: Sequence[float],
    basis_matrix: np.ndarray,
    frame_id: str | None,
) -> np.ndarray:
    source = _as_vec(vector, 3)
    if _is_controller_frame(frame_id):
        return source.astype(np.float64, copy=True)
    determinant = float(np.linalg.det(basis_matrix))
    if not np.isclose(abs(determinant), 1.0, atol=1e-6):
        raise ValueError(
            "axial-vector conversion requires an orthogonal basis with determinant +/-1, "
            f"got determinant={determinant}"
        )
    return (determinant * basis_matrix @ source).astype(np.float64)


def _json_safe(value):
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def _load_hardware_bridge_config(path: str) -> tuple[ThrusterCurve, list[int], list[float], str]:
    if not path:
        return ThrusterCurve.defaults(), list(range(THRUSTER_COUNT)), [1.0] * THRUSTER_COUNT, ""

    config_path = Path(path).expanduser()
    if not config_path.is_file():
        raise FileNotFoundError(f"hardware_bridge_config does not exist: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}

    params = data.get("hardware_bridge", {}).get("ros__parameters", data)
    curve_data = params.get("thruster_curve", {})
    defaults = ThrusterCurve.defaults()

    def arr(name: str, default: np.ndarray) -> np.ndarray:
        return _as_vec(curve_data.get(name, default.tolist()), THRUSTER_COUNT)

    curve = ThrusterCurve(
        c1_positive=arr("c1_positive", defaults.c1_positive),
        c1_negative=arr("c1_negative", defaults.c1_negative),
        rpm_min=arr("rpm_min", defaults.rpm_min),
        rpm_max=arr("rpm_max", defaults.rpm_max),
        force_deadband_n=arr("force_deadband_n", defaults.force_deadband_n),
        min_effective_rpm_positive=arr("min_effective_rpm_positive", defaults.min_effective_rpm_positive),
        min_effective_rpm_negative=arr("min_effective_rpm_negative", defaults.min_effective_rpm_negative),
        firmware_max_rpm=float(params.get("firmware_max_rpm", defaults.firmware_max_rpm)),
    )
    motor_order = [int(v) for v in params.get("motor_order", list(range(THRUSTER_COUNT)))]
    motor_signs = [float(v) for v in params.get("motor_signs", [1.0] * THRUSTER_COUNT)]
    if len(motor_order) != THRUSTER_COUNT or sorted(motor_order) != list(range(THRUSTER_COUNT)):
        raise ValueError("hardware bridge motor_order must be a permutation of [0..7]")
    if len(motor_signs) != THRUSTER_COUNT or any(not math.isfinite(v) for v in motor_signs):
        raise ValueError("hardware bridge motor_signs must contain 8 finite values")
    return curve, motor_order, motor_signs, str(config_path)


def _robust_acceleration_mask(
    values: Sequence[float],
    max_abs_value: float,
    mad_threshold: float,
    local_mad_threshold: float,
    local_window: int,
) -> tuple[np.ndarray, int, int, int]:
    array = np.asarray(values, dtype=np.float64)
    mask = np.isfinite(array)
    absolute_rejected = 0
    if max_abs_value > 0.0:
        absolute_mask = np.abs(array) <= max_abs_value
        absolute_rejected = int(np.count_nonzero(mask & ~absolute_mask))
        mask &= absolute_mask

    mad_rejected = 0
    if mad_threshold > 0.0 and np.count_nonzero(mask) >= 8:
        kept = array[mask]
        median = float(np.median(kept))
        mad = float(np.median(np.abs(kept - median)))
        robust_sigma = 1.4826 * mad
        if robust_sigma > 1e-9:
            robust_mask = np.abs(array - median) <= mad_threshold * robust_sigma
            mad_rejected = int(np.count_nonzero(mask & ~robust_mask))
            mask &= robust_mask

    local_rejected = 0
    if local_mad_threshold > 0.0 and local_window >= 1 and np.count_nonzero(mask) >= 8:
        local_mask = np.ones(array.size, dtype=bool)
        window = int(local_window)
        for i, value in enumerate(array):
            if not mask[i] or not math.isfinite(float(value)):
                continue
            start = max(0, i - window)
            stop = min(array.size, i + window + 1)
            neighborhood_mask = mask[start:stop].copy()
            if i - start < neighborhood_mask.size:
                neighborhood_mask[i - start] = False
            neighborhood = array[start:stop][neighborhood_mask]
            if neighborhood.size < 5:
                continue
            median = float(np.median(neighborhood))
            mad = float(np.median(np.abs(neighborhood - median)))
            robust_sigma = 1.4826 * mad
            if robust_sigma <= 1e-9:
                continue
            if abs(float(value) - median) > local_mad_threshold * robust_sigma:
                local_mask[i] = False
        local_rejected = int(np.count_nonzero(mask & ~local_mask))
        mask &= local_mask
    return mask, absolute_rejected, mad_rejected, local_rejected


def _bounded_lstsq(
    x: np.ndarray,
    y: np.ndarray,
    *,
    lower: Sequence[float],
    upper: Sequence[float],
    param_names: Sequence[str],
) -> tuple[np.ndarray, list[str]]:
    lower_array = np.asarray(lower, dtype=np.float64)
    upper_array = np.asarray(upper, dtype=np.float64)
    param_count = x.shape[1]
    if lower_array.size != param_count or upper_array.size != param_count:
        raise ValueError("bounds must match parameter count")

    best_coeff: np.ndarray | None = None
    best_error = float("inf")
    best_active: list[str] = []

    choices: list[tuple[int, ...]] = []
    for i in range(param_count):
        param_choices = [0]
        if math.isfinite(float(lower_array[i])):
            param_choices.append(-1)
        if math.isfinite(float(upper_array[i])):
            param_choices.append(1)
        choices.append(tuple(param_choices))

    for state in product(*choices):
        coeff = np.zeros(param_count, dtype=np.float64)
        fixed = np.zeros(param_count, dtype=bool)
        active: list[str] = []
        feasible = True
        for i, marker in enumerate(state):
            name = param_names[i] if i < len(param_names) else f"p{i}"
            if marker < 0:
                coeff[i] = lower_array[i]
                fixed[i] = True
                active.append(f"{name}=lower")
            elif marker > 0:
                coeff[i] = upper_array[i]
                fixed[i] = True
                active.append(f"{name}=upper")

        free_indices = np.flatnonzero(~fixed)
        residual_target = y - x[:, fixed] @ coeff[fixed] if np.any(fixed) else y
        if free_indices.size:
            try:
                free_coeff, *_ = np.linalg.lstsq(x[:, free_indices], residual_target, rcond=None)
            except np.linalg.LinAlgError:
                continue
            coeff[free_indices] = free_coeff

        for i in range(param_count):
            if coeff[i] < lower_array[i] - 1e-9 or coeff[i] > upper_array[i] + 1e-9:
                feasible = False
                break
        if not feasible:
            continue

        residual = y - x @ coeff
        error = float(residual @ residual)
        if error < best_error:
            best_error = error
            best_coeff = coeff.copy()
            best_active = active

    if best_coeff is None:
        coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
        return np.clip(coeff, lower_array, upper_array), ["fallback_clipped_lstsq"]
    return best_coeff, best_active


def _fit_axis(
    samples: list[Sample],
    axis: str,
    *,
    min_abs_tau: float,
    velocity_deadband: float,
    max_abs_acceleration: float,
    accel_outlier_mad_threshold: float,
    accel_spike_local_mad_threshold: float,
    accel_spike_local_window: int,
    constrain_physical_coefficients: bool,
    max_effective_mass: float,
    max_effective_inertia: float,
    max_linear_damping: float,
    max_quadratic_damping: float,
    max_abs_bias: float,
    fit_phases: Sequence[str] = ("excitation",),
    allow_zero_tau_samples: bool = False,
    fit_type: str = "forced_axis",
) -> dict[str, float | str]:
    tau_index = AXIS_TO_TAU_INDEX[axis]
    nu_index = AXIS_TO_NU_INDEX[axis]
    allowed_phases = frozenset(str(phase) for phase in fit_phases)
    rows: list[list[float]] = []
    targets: list[float] = []
    accelerations: list[float] = []
    phases: list[str] = []
    for sample in samples:
        if (
            sample.axis != axis
            or sample.phase not in allowed_phases
            or not sample.sample_window
            or not sample.synchronization_valid
        ):
            continue
        tau = float(sample.tau[tau_index])
        nu = float(sample.nu[nu_index])
        nu_dot = float(sample.nu_dot[nu_index])
        if abs(tau) < min_abs_tau and not allow_zero_tau_samples:
            continue
        if abs(nu) < velocity_deadband and abs(nu_dot) < velocity_deadband:
            continue
        rows.append([nu_dot, nu, abs(nu) * nu, 1.0])
        targets.append(tau)
        accelerations.append(nu_dot)
        phases.append(sample.phase)

    raw_count = len(rows)
    if rows:
        mask, absolute_rejected, mad_rejected, local_rejected = _robust_acceleration_mask(
            accelerations,
            max_abs_value=max_abs_acceleration,
            mad_threshold=accel_outlier_mad_threshold,
            local_mad_threshold=accel_spike_local_mad_threshold,
            local_window=accel_spike_local_window,
        )
        rows = [row for row, keep in zip(rows, mask) if bool(keep)]
        targets = [target for target, keep in zip(targets, mask) if bool(keep)]
        phases = [phase for phase, keep in zip(phases, mask) if bool(keep)]
    else:
        absolute_rejected = 0
        mad_rejected = 0
        local_rejected = 0

    if len(rows) < 6:
        return {
            "fit_type": fit_type,
            "fit_phases": ",".join(sorted(allowed_phases)),
            "allow_zero_tau_samples": allow_zero_tau_samples,
            "excitation_sample_count": 0.0,
            "coast_sample_count": 0.0,
            "raw_sample_count": float(raw_count),
            "sample_count": float(len(rows)),
            "rejected_outlier_count": float(raw_count - len(rows)),
            "rejected_abs_accel_count": float(absolute_rejected),
            "rejected_mad_accel_count": float(mad_rejected),
            "rejected_local_spike_count": float(local_rejected),
            "m_eff": float("nan"),
            "d_linear": float("nan"),
            "d_quadratic": float("nan"),
            "bias": float("nan"),
            "rmse": float("nan"),
            "rmse_unit": "N_or_Nm",
            "bounded_fit": constrain_physical_coefficients,
        }

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if constrain_physical_coefficients:
        max_mass_or_inertia = max_effective_inertia if axis == "yaw_y" else max_effective_mass
        coeff, active_bounds = _bounded_lstsq(
            x,
            y,
            lower=[0.0, 0.0, 0.0, -max_abs_bias],
            upper=[max_mass_or_inertia, max_linear_damping, max_quadratic_damping, max_abs_bias],
            param_names=["m_eff", "d_linear", "d_quadratic", "bias"],
        )
    else:
        coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
        active_bounds = []
    residual = y - x @ coeff
    return {
        "fit_type": fit_type,
        "fit_phases": ",".join(sorted(allowed_phases)),
        "allow_zero_tau_samples": allow_zero_tau_samples,
        "excitation_sample_count": float(sum(phase == "excitation" for phase in phases)),
        "coast_sample_count": float(sum(phase == "coast" for phase in phases)),
        "bounded_fit": constrain_physical_coefficients,
        "active_bounds": ",".join(active_bounds),
        "raw_sample_count": float(raw_count),
        "sample_count": float(len(rows)),
        "rejected_outlier_count": float(raw_count - len(rows)),
        "rejected_abs_accel_count": float(absolute_rejected),
        "rejected_mad_accel_count": float(mad_rejected),
        "rejected_local_spike_count": float(local_rejected),
        "m_eff": float(coeff[0]),
        "d_linear": float(coeff[1]),
        "d_quadratic": float(coeff[2]),
        "bias": float(coeff[3]),
        "rmse": float(math.sqrt(float(np.mean(residual * residual)))),
        "rmse_unit": "N_or_Nm",
    }


def _fit_attitude_pulse_axis(
    samples: list[Sample],
    axis: str,
    *,
    min_abs_angle: float,
    min_abs_rate: float,
    min_abs_tau: float,
    fit_start_sec: float,
    max_abs_acceleration: float,
    accel_outlier_mad_threshold: float,
    accel_spike_local_mad_threshold: float,
    accel_spike_local_window: int,
    constrain_physical_coefficients: bool,
    max_effective_inertia: float,
    max_linear_damping: float,
    max_quadratic_damping: float,
    max_restoring_stiffness: float,
    max_abs_bias: float,
) -> dict[str, float | str]:
    tau_index = AXIS_TO_TAU_INDEX[axis]
    nu_index = AXIS_TO_NU_INDEX[axis]
    rows: list[list[float]] = []
    targets: list[float] = []
    angles: list[float] = []
    rates: list[float] = []
    torques: list[float] = []
    accelerations: list[float] = []
    for sample in samples:
        if (
            sample.axis != axis
            or sample.phase != "excitation"
            or not sample.sample_window
            or not sample.synchronization_valid
        ):
            continue
        if sample.phase_elapsed_sec < fit_start_sec or not sample.orientation_valid:
            continue
        tau = float(sample.tau[tau_index])
        theta = float(sample.angle[nu_index])
        omega = float(sample.nu[nu_index])
        alpha = float(sample.nu_dot[nu_index])
        if abs(tau) < min_abs_tau and abs(theta) < min_abs_angle and abs(omega) < min_abs_rate:
            continue
        rows.append([alpha, omega, abs(omega) * omega, math.sin(theta), 1.0])
        targets.append(tau)
        angles.append(theta)
        rates.append(omega)
        torques.append(tau)
        accelerations.append(alpha)

    raw_count = len(rows)
    if rows:
        mask, absolute_rejected, mad_rejected, local_rejected = _robust_acceleration_mask(
            accelerations,
            max_abs_value=max_abs_acceleration,
            mad_threshold=accel_outlier_mad_threshold,
            local_mad_threshold=accel_spike_local_mad_threshold,
            local_window=accel_spike_local_window,
        )
        rows = [row for row, keep in zip(rows, mask) if bool(keep)]
        targets = [target for target, keep in zip(targets, mask) if bool(keep)]
    else:
        absolute_rejected = 0
        mad_rejected = 0
        local_rejected = 0

    if len(rows) < 10:
        return {
            "fit_type": "attitude_pulse",
            "raw_sample_count": float(raw_count),
            "sample_count": float(len(rows)),
            "rejected_outlier_count": float(raw_count - len(rows)),
            "rejected_abs_accel_count": float(absolute_rejected),
            "rejected_mad_accel_count": float(mad_rejected),
            "rejected_local_spike_count": float(local_rejected),
            "m_eff": float("nan"),
            "d_linear": float("nan"),
            "d_quadratic": float("nan"),
            "restoring_stiffness": float("nan"),
            "bias": float("nan"),
            "rmse": float("nan"),
            "rmse_unit": "Nm",
            "max_abs_angle_rad": float(max((abs(v) for v in angles), default=0.0)),
            "max_abs_rate_radps": float(max((abs(v) for v in rates), default=0.0)),
            "max_abs_tau_nm": float(max((abs(v) for v in torques), default=0.0)),
            "note": "not enough attitude pulse samples with valid IMU orientation",
            "bounded_fit": constrain_physical_coefficients,
        }

    x = np.asarray(rows, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if constrain_physical_coefficients:
        coeff, active_bounds = _bounded_lstsq(
            x,
            y,
            lower=[0.0, 0.0, 0.0, 0.0, -max_abs_bias],
            upper=[max_effective_inertia, max_linear_damping, max_quadratic_damping, max_restoring_stiffness, max_abs_bias],
            param_names=["m_eff", "d_linear", "d_quadratic", "restoring_stiffness", "bias"],
        )
    else:
        coeff, *_ = np.linalg.lstsq(x, y, rcond=None)
        active_bounds = []
    residual = y - x @ coeff
    return {
        "fit_type": "attitude_pulse",
        "bounded_fit": constrain_physical_coefficients,
        "active_bounds": ",".join(active_bounds),
        "raw_sample_count": float(raw_count),
        "sample_count": float(len(rows)),
        "rejected_outlier_count": float(raw_count - len(rows)),
        "rejected_abs_accel_count": float(absolute_rejected),
        "rejected_mad_accel_count": float(mad_rejected),
        "rejected_local_spike_count": float(local_rejected),
        "m_eff": float(coeff[0]),
        "d_linear": float(coeff[1]),
        "d_quadratic": float(coeff[2]),
        "restoring_stiffness": float(coeff[3]),
        "bias": float(coeff[4]),
        "rmse": float(math.sqrt(float(np.mean(residual * residual)))),
        "rmse_unit": "Nm",
        "max_abs_angle_rad": float(max(abs(v) for v in angles)),
        "max_abs_rate_radps": float(max(abs(v) for v in rates)),
        "max_abs_tau_nm": float(max(abs(v) for v in torques)),
    }


def _axis_motion_labels(axis: str) -> tuple[str, str, str]:
    labels = {
        "surge_x": ("u velocity", "u acceleration", "m/s"),
        "heave_y": ("y velocity", "y acceleration", "m/s"),
        "sway_z": ("z velocity", "z acceleration", "m/s"),
        "roll_x": ("roll rate", "roll angular acceleration", "rad/s"),
        "yaw_y": ("yaw rate", "yaw angular acceleration", "rad/s"),
        "pitch_z": ("pitch rate", "pitch angular acceleration", "rad/s"),
    }
    return labels.get(axis, (f"{axis} velocity", f"{axis} acceleration", "unit/s"))


def _write_motion_png(samples: Sequence[Sample], axis: str, path: Path, *, fit_only: bool = False) -> None:
    from PIL import Image, ImageDraw

    axis_samples = [
        sample
        for sample in samples
        if sample.axis == axis
        and np.isfinite(sample.time_sec)
        and (not fit_only or sample.sample_window)
    ]
    if not axis_samples:
        return

    index = AXIS_TO_NU_INDEX[axis]
    times = [sample.time_sec - axis_samples[0].time_sec for sample in axis_samples]
    velocity = [float(sample.nu[index]) for sample in axis_samples]
    acceleration = [float(sample.nu_dot[index]) for sample in axis_samples]

    velocity_label, acceleration_label, velocity_unit = _axis_motion_labels(axis)
    acceleration_unit = "rad/s^2" if velocity_unit == "rad/s" else "m/s^2"
    width, height = 1100, 760
    left, right = 90, 40
    top_panel = (105, 355)
    bottom_panel = (440, 690)
    plot_width = width - left - right
    colors = {
        "background": (255, 255, 255),
        "title": (24, 33, 47),
        "meta": (83, 96, 113),
        "axis": (123, 135, 148),
        "grid": (216, 222, 232),
        "velocity": (23, 105, 170),
        "acceleration": (194, 65, 12),
    }

    def scale_points(values: Sequence[float], panel: tuple[int, int]) -> tuple[list[tuple[float, float]], float, float]:
        y_top, y_bottom = panel
        finite_values = [v for v in values if math.isfinite(v)]
        if not finite_values:
            finite_values = [0.0]
        y_min = min(finite_values)
        y_max = max(finite_values)
        if abs(y_max - y_min) < 1e-9:
            pad = max(abs(y_max) * 0.1, 1.0)
            y_min -= pad
            y_max += pad
        else:
            pad = (y_max - y_min) * 0.12
            y_min -= pad
            y_max += pad

        t_max = max(max(times), 1e-6)
        points: list[tuple[float, float]] = []
        for t, value in zip(times, values):
            if not math.isfinite(value):
                continue
            x = left + (t / t_max) * plot_width
            y = y_bottom - ((value - y_min) / (y_max - y_min)) * (y_bottom - y_top)
            points.append((x, y))
        return points, y_min, y_max

    velocity_points, velocity_min, velocity_max = scale_points(velocity, top_panel)
    accel_points, accel_min, accel_max = scale_points(acceleration, bottom_panel)
    t_max = max(max(times), 1e-6)

    image = Image.new("RGB", (width, height), colors["background"])
    draw = ImageDraw.Draw(image)

    def text(x: float, y: float, value: object, fill: tuple[int, int, int] = colors["meta"]) -> None:
        draw.text((int(round(x)), int(round(y))), str(value), fill=fill)

    title_suffix = "fit samples only" if fit_only else "all recorded phases"
    text(left, 28, f"{axis} velocity and acceleration", colors["title"])
    text(left, 54, f"Samples: {len(axis_samples)} | time span: {t_max:.2f}s | {title_suffix}")

    def draw_panel(
        panel: tuple[int, int],
        y_min: float,
        y_max: float,
        title: str,
        unit: str,
        color: tuple[int, int, int],
        points: list[tuple[float, float]],
    ) -> None:
        y_top, y_bottom = panel
        x_left = left
        x_right = left + plot_width
        y_mid = (y_top + y_bottom) / 2.0
        text(x_left, y_top - 24, f"{title} ({unit})", colors["title"])
        draw.line((x_left, y_bottom, x_right, y_bottom), fill=colors["axis"], width=1)
        draw.line((x_left, y_top, x_left, y_bottom), fill=colors["axis"], width=1)
        for offset in range(0, int(plot_width), 14):
            if offset % 28 == 0:
                draw.point((int(x_left + offset), int(y_mid)), fill=colors["grid"])
        text(x_left - 76, y_top - 7, f"{y_max:.4g}")
        text(x_left - 76, y_mid - 7, f"{((y_min + y_max) * 0.5):.4g}")
        text(x_left - 76, y_bottom - 7, f"{y_min:.4g}")
        if len(points) >= 2:
            draw.line([(int(round(x)), int(round(y))) for x, y in points], fill=color, width=3)

    draw_panel(top_panel, velocity_min, velocity_max, velocity_label, velocity_unit, colors["velocity"], velocity_points)
    draw_panel(bottom_panel, accel_min, accel_max, acceleration_label, acceleration_unit, colors["acceleration"], accel_points)

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        x = left + fraction * plot_width
        label = t_max * fraction
        draw.line((x, bottom_panel[1], x, bottom_panel[1] + 7), fill=colors["axis"], width=1)
        text(x - 12, bottom_panel[1] + 12, f"{label:.2f}")

    text(left + plot_width / 2 - 130, height - 35, f"time since first {axis} sample (s)")
    image.save(path, format="PNG")


class HydrodynamicIdentifierNode(Node):
    def __init__(self) -> None:
        super().__init__("hydrodynamic_identifier")

        self._auto_start = _bool_value(self.declare_parameter("auto_start", False).value)
        self._rate_hz = float(self.declare_parameter("publish_rate_hz", 50.0).value)
        self._output_directory = str(self.declare_parameter("output_directory", "data/hydrodynamic_identification").value)
        self._output_prefix = str(self.declare_parameter("output_prefix", "finsrov_fossen_6dof").value)
        self._run_role = str(self.declare_parameter("run_role", "identification").value).strip().lower()
        if self._run_role not in {"identification", "held_out_fossen_validation"}:
            raise ValueError(
                "run_role must be `identification` or `held_out_fossen_validation`, "
                f"got {self._run_role!r}"
            )
        self._thruster_topic = str(self.declare_parameter("thruster_topic", "/finsrov/thrusters_out").value)
        self._motor_rpm_topic = str(self.declare_parameter("motor_rpm_topic", "/finsrov/hardware/motor_rpm_raw").value)
        self._hardware_telemetry_topic = str(
            self.declare_parameter("hardware_telemetry_topic", "/finsrov/hardware/telemetry").value
        )
        self._use_hardware_telemetry = _bool_value(
            self.declare_parameter("use_hardware_telemetry", True).value
        )
        self._sync_max_telemetry_age_sec = float(
            self.declare_parameter("sync_max_telemetry_age_sec", 0.02).value
        )
        self._sync_max_dvl_age_sec = float(self.declare_parameter("sync_max_dvl_age_sec", 0.03).value)
        self._sync_buffer_sec = float(self.declare_parameter("sync_buffer_sec", 2.0).value)
        self._imu_topic = str(self.declare_parameter("imu_topic", "/finsrov/imu_link").value)
        self._dvl_topic = str(self.declare_parameter("dvl_topic", "/finsrov/dvl_link").value)
        self._hardware_config = str(self.declare_parameter("hardware_bridge_config", "").value)
        self._basis_indices = [int(v) for v in self.declare_parameter("ros_to_body_basis_indices", [0, 2, 1]).value]
        self._basis_signs = [float(v) for v in self.declare_parameter("ros_to_body_basis_signs", [1.0, 1.0, 1.0]).value]
        self._basis_matrix = _build_basis_matrix(self._basis_indices, self._basis_signs)
        self._axes = tuple(
            axis
            for axis in _string_list(
                self.declare_parameter(
                    "axes",
                    list(AXIS_ORDER),
                    ParameterDescriptor(dynamic_typing=True),
                ).value
            )
            if axis in AXIS_ORDER
        )
        self._include_negative = _bool_value(self.declare_parameter("include_negative", True).value)
        self._repeat_per_trial = max(1, int(self.declare_parameter("repeat_per_trial", 1).value))
        self._pause_between_trials = _bool_value(
            self.declare_parameter(
                "pause_between_trials",
                False,
                ParameterDescriptor(
                    description=(
                        "Pause with zero thrust after each signed trial until Enter is pressed "
                        "or the continue service is called."
                    )
                ),
            ).value
        )
        self._pre_zero_sec = float(self.declare_parameter("pre_zero_sec", 2.0).value)
        self._baseline_sec = float(self.declare_parameter("baseline_sec", 2.0).value)
        self._hold_sec = float(self.declare_parameter("hold_sec", 6.0).value)
        self._sample_sec = float(self.declare_parameter("sample_sec", 4.0).value)
        self._step_fit_start_sec = float(self.declare_parameter("step_fit_start_sec", 0.0).value)
        self._rest_sec = float(self.declare_parameter("rest_sec", 4.0).value)
        self._post_zero_sec = float(self.declare_parameter("post_zero_sec", 2.0).value)
        self._heave_y_identification_mode = _normalize_heave_identification_mode(
            self.declare_parameter("heave_y_identification_mode", "step").value
        )
        self._heave_y_coast_sec = float(self.declare_parameter("heave_y_coast_sec", 8.0).value)
        self._heave_y_ascent_velocity_threshold = max(
            0.0, float(self.declare_parameter("heave_y_ascent_velocity_threshold_mps", 0.01).value)
        )
        self._heave_y_ascent_sample_sec = max(
            0.0, float(self.declare_parameter("heave_y_ascent_sample_sec", 2.0).value)
        )
        self._max_wrench_scale = float(self.declare_parameter("max_wrench_scale", 0.75).value)
        self._gravity_compensation_enabled = _bool_value(
            self.declare_parameter("gravity_compensation_enabled", True).value
        )
        self._gravity_mps2 = float(self.declare_parameter("gravity_mps2", 9.80665).value)
        self._rpm_stale_sec = float(self.declare_parameter("rpm_stale_sec", 0.3).value)
        self._min_fit_abs_tau = float(self.declare_parameter("min_fit_abs_tau", 0.05).value)
        self._fit_velocity_deadband = float(self.declare_parameter("fit_velocity_deadband", 0.005).value)
        self._fit_max_abs_linear_accel = float(self.declare_parameter("fit_max_abs_linear_accel_mps2", 5.0).value)
        self._fit_max_abs_angular_accel = float(self.declare_parameter("fit_max_abs_angular_accel_radps2", 3.0).value)
        self._fit_accel_outlier_mad_threshold = float(
            self.declare_parameter("fit_accel_outlier_mad_threshold", 8.0).value
        )
        self._fit_accel_spike_local_mad_threshold = float(
            self.declare_parameter("fit_accel_spike_local_mad_threshold", 6.0).value
        )
        self._fit_accel_spike_local_window = int(
            self.declare_parameter("fit_accel_spike_local_window", 5).value
        )
        self._fit_constrain_physical_coefficients = _bool_value(
            self.declare_parameter("fit_constrain_physical_coefficients", True).value
        )
        self._fit_max_effective_mass = float(self.declare_parameter("fit_max_effective_mass_kg", 100.0).value)
        self._fit_max_effective_inertia = float(self.declare_parameter("fit_max_effective_inertia_kgm2", 20.0).value)
        self._fit_max_linear_damping = float(self.declare_parameter("fit_max_linear_damping", 500.0).value)
        self._fit_max_quadratic_damping = float(self.declare_parameter("fit_max_quadratic_damping", 1000.0).value)
        self._fit_max_restoring_stiffness = float(
            self.declare_parameter("fit_max_restoring_stiffness_nm_per_rad", 100.0).value
        )
        self._fit_max_abs_bias = float(self.declare_parameter("fit_max_abs_bias", 100.0).value)
        self._accel_alpha = float(np.clip(float(self.declare_parameter("acceleration_filter_alpha", 0.35).value), 0.0, 1.0))
        self._attitude_pulse_sec = float(self.declare_parameter("attitude_pulse_sec", 0.6).value)
        self._attitude_ringdown_sec = float(self.declare_parameter("attitude_ringdown_sec", 3.0).value)
        self._attitude_fit_start_sec = float(self.declare_parameter("attitude_fit_start_sec", 0.05).value)
        self._attitude_min_abs_tau = float(self.declare_parameter("attitude_min_abs_tau_nm", 0.005).value)
        self._attitude_min_abs_angle_rad = float(self.declare_parameter("attitude_min_abs_angle_rad", 0.003).value)
        self._attitude_min_abs_rate_radps = float(self.declare_parameter("attitude_min_abs_rate_radps", 0.003).value)
        self._attitude_max_abs_angle_rad = {
            "roll_x": math.radians(float(self.declare_parameter("roll_x_max_abs_angle_deg", 5.0).value)),
            "pitch_z": math.radians(float(self.declare_parameter("pitch_z_max_abs_angle_deg", 5.0).value)),
        }
        self._attitude_max_abs_rate_radps = {
            "roll_x": float(self.declare_parameter("roll_x_max_abs_rate_radps", 0.4).value),
            "pitch_z": float(self.declare_parameter("pitch_z_max_abs_rate_radps", 0.4).value),
        }
        self._ignore_attitude_safety_checks = _bool_value(
            self.declare_parameter(
                "ignore_attitude_safety_checks",
                False,
                ParameterDescriptor(
                    description=(
                        "Disable roll/pitch angle and angular-rate cutoffs for a supervised diagnostic run. "
                        "False is required for normal pool testing."
                    )
                ),
            ).value
        )
        self._equilibrium_angles = np.zeros(6, dtype=np.float64)
        self._equilibrium_angles[AXIS_TO_NU_INDEX["roll_x"]] = float(
            self.declare_parameter("roll_x_equilibrium_angle_rad", 0.0).value
        )
        self._equilibrium_angles[AXIS_TO_NU_INDEX["pitch_z"]] = float(
            self.declare_parameter("pitch_z_equilibrium_angle_rad", 0.0).value
        )

        self._levels = {
            "surge_x": _as_vec(self.declare_parameter("surge_x_levels", [1.5, 3.0, 4.5, 6.0]).value, 64),
            "heave_y": _as_vec(self.declare_parameter("heave_y_levels", [1.0, 2.0, 3.0, 4.0]).value, 64),
            "sway_z": _as_vec(self.declare_parameter("sway_z_levels", [1.0, 2.0, 3.0, 4.0]).value, 64),
            "roll_x": _as_vec(self.declare_parameter("roll_x_levels", [0.03, 0.06]).value, 64),
            "pitch_z": _as_vec(self.declare_parameter("pitch_z_levels", [0.03, 0.06]).value, 64),
            "yaw_y": _as_vec(self.declare_parameter("yaw_y_levels", [0.15, 0.30, 0.45, 0.60]).value, 64),
        }
        self._apply_cli_amplitude_override()

        self._curve, self._motor_order, self._motor_signs, self._resolved_hardware_config = _load_hardware_bridge_config(
            self._hardware_config
        )
        self._force_limit_pos, self._force_limit_neg = self._curve.force_limits(self._max_wrench_scale)

        self._thruster_pub = self.create_publisher(Float32MultiArray, self._thruster_topic, 10)
        self.create_subscription(TwistWithCovarianceStamped, self._dvl_topic, self._dvl_callback, 10)
        if self._use_hardware_telemetry:
            self.create_subscription(HardwareTelemetry, self._hardware_telemetry_topic, self._hardware_telemetry_callback, 100)
        else:
            self.create_subscription(Imu, self._imu_topic, self._imu_callback, 10)
            self.create_subscription(Float32MultiArray, self._motor_rpm_topic, self._motor_rpm_callback, 10)
        self.create_service(Trigger, "~/start", self._start_service)
        self.create_service(Trigger, "~/stop", self._stop_service)
        self.create_service(Trigger, "~/continue", self._continue_service)

        self._pause_condition = threading.Condition()
        self._awaiting_continue = False
        self._continue_requested = False
        self._stdin_stop_event = threading.Event()
        self._stdin_thread: threading.Thread | None = None
        if self._pause_between_trials:
            self._stdin_thread = threading.Thread(
                target=self._stdin_continue_loop,
                name="hydrodynamic-identifier-stdin",
                daemon=True,
            )
            self._stdin_thread.start()

        self._trials = self._build_trials()
        self._phases = (
            Phase("baseline", max(self._baseline_sec, 0.0)),
            Phase("excitation", max(self._hold_sec, 0.0)),
            Phase("coast", 0.0),
            Phase("rest", max(self._rest_sec, 0.0)),
        )
        self._timer = self.create_timer(1.0 / max(self._rate_hz, 1.0), self._tick)

        self._running = False
        self._trial_index = -1
        self._phase_index = 0
        self._phase_start_sec = 0.0
        self._run_start_sec = 0.0
        self._post_zero_active = False
        self._latest_linear_body = np.zeros(3, dtype=np.float64)
        self._latest_angular_body = np.zeros(3, dtype=np.float64)
        self._latest_linear_body_dot = np.zeros(3, dtype=np.float64)
        self._latest_angular_body_dot = np.zeros(3, dtype=np.float64)
        self._latest_dvl_frame_id = ""
        self._latest_dvl_linear_raw = np.zeros(3, dtype=np.float64)
        self._latest_dvl_linear_controller = np.zeros(3, dtype=np.float64)
        self._latest_imu_frame_id = ""
        self._latest_imu_linear_accel_raw = np.zeros(3, dtype=np.float64)
        self._latest_imu_linear_accel_controller = np.zeros(3, dtype=np.float64)
        self._latest_gravity_controller = np.zeros(3, dtype=np.float64)
        self._latest_imu_linear_accel_kinematic = np.zeros(3, dtype=np.float64)
        self._latest_imu_angular_raw = np.zeros(3, dtype=np.float64)
        self._latest_imu_angular_controller = np.zeros(3, dtype=np.float64)
        self._latest_imu_quat_raw = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._latest_imu_quat_controller = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        self._previous_angular_body: np.ndarray | None = None
        self._previous_angular_body_sec: float | None = None
        self._latest_angle_body = np.zeros(6, dtype=np.float64)
        self._attitude_safety_cutoff_sec: float | None = None
        self._attitude_safety_cutoffs: list[dict[str, float | int | str]] = []
        self._heave_y_coast_seen_downward = False
        self._heave_y_ascent_sample_start_sec: float | None = None
        self._heave_y_ascent_events: list[dict[str, float | int | str | bool]] = []
        self._latest_orientation_valid = False
        self._latest_rpm: np.ndarray | None = None
        self._latest_rpm_sec: float | None = None
        self._telemetry_buffer: deque[TelemetryBufferSample] = deque(maxlen=256)
        self._dvl_buffer: deque[DvlBufferSample] = deque(maxlen=256)
        self._latest_sample_time_sec = self._now_sec()
        self._latest_imu_mcu_time_ms = -1
        self._latest_imu_ros_stamp_sec = float("nan")
        self._latest_dvl_stamp_sec = float("nan")
        self._latest_host_receive_time_ns = 0
        self._latest_imu_age_sec = float("nan")
        self._latest_dvl_age_sec = float("nan")
        self._latest_telemetry_sequence = -1
        self._latest_synchronization_valid = not self._use_hardware_telemetry
        self._samples: list[Sample] = []
        self._csv_file = None
        self._csv_writer: csv.writer | None = None
        self._current_output_base: Path | None = None
        self._current_output_axis = ""
        self._current_output_run_index = 0
        self._last_command_forces = np.zeros(THRUSTER_COUNT, dtype=np.float64)
        self._last_tau_command = np.zeros(6, dtype=np.float64)
        self._last_tau_actual = np.zeros(6, dtype=np.float64)

        self.get_logger().info(
            "hydrodynamic identifier ready: "
            f"run_role={self._run_role}, axes={self._axes}, trials={len(self._trials)}, "
            f"repeat_per_trial={self._repeat_per_trial}, thruster_topic={self._thruster_topic}, "
            f"imu_topic={self._imu_topic}, dvl_topic={self._dvl_topic}, "
            f"hardware_telemetry_topic={self._hardware_telemetry_topic}, "
            f"use_hardware_telemetry={self._use_hardware_telemetry}, "
            f"heave_y_identification_mode={self._heave_y_identification_mode}, "
            f"hardware_config={self._resolved_hardware_config or 'defaults'}, motor_order={self._motor_order}, "
            f"motor_signs={[round(v, 3) for v in self._motor_signs]}"
        )
        if self._auto_start:
            self._start_run()

    def destroy_node(self) -> bool:
        self._stdin_stop_event.set()
        with self._pause_condition:
            self._pause_condition.notify_all()
        if rclpy.ok():
            try:
                self._publish_thrusters(np.zeros(THRUSTER_COUNT, dtype=np.float64))
            except Exception as exc:
                self.get_logger().warn(f"failed to publish shutdown zero thrust: {exc}")
        self._close_csv()
        return super().destroy_node()

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _body_from_ros(self, vector: Sequence[float], frame_id: str | None = None) -> np.ndarray:
        return _vector_to_controller_body(vector, self._basis_matrix, frame_id)

    def _angular_body_from_ros(self, vector: Sequence[float], frame_id: str | None = None) -> np.ndarray:
        return _axial_vector_to_controller_body(vector, self._basis_matrix, frame_id)

    @staticmethod
    def _is_attitude_pulse_axis(axis: str) -> bool:
        return axis in ATTITUDE_PULSE_AXIS_ORDER

    def _is_heave_dive_and_coast_trial(self, trial: Trial) -> bool:
        return trial.axis == "heave_y" and self._heave_y_identification_mode == "dive_and_coast"

    def _update_heave_y_coast_sampling(self, now: float, trial: Trial, elapsed: float) -> bool:
        """Return true when the zero-thrust coast phase can advance to rest."""
        threshold = self._heave_y_ascent_velocity_threshold
        velocity_y = float(self._latest_linear_body[AXIS_TO_NU_INDEX["heave_y"]])
        if not self._heave_y_coast_seen_downward and velocity_y <= -threshold:
            self._heave_y_coast_seen_downward = True

        if self._heave_y_ascent_sample_start_sec is None:
            if self._heave_y_coast_seen_downward and velocity_y >= threshold:
                self._heave_y_ascent_sample_start_sec = now
                self._heave_y_ascent_events.append(
                    {
                        "trial_index": int(self._trial_index),
                        "axis": trial.axis,
                        "level": float(trial.level),
                        "coast_detection_sec": float(elapsed),
                        "ascent_velocity_mps": velocity_y,
                        "sample_duration_sec": self._heave_y_ascent_sample_sec,
                        "detected": True,
                    }
                )
                self.get_logger().warn(
                    f"heave_y ascent detected at coast={elapsed:.2f}s, "
                    f"vy={velocity_y:+.3f}m/s; recording {self._heave_y_ascent_sample_sec:.1f}s"
                )
                return False
            if elapsed >= max(self._heave_y_coast_sec, 0.0):
                self._heave_y_ascent_events.append(
                    {
                        "trial_index": int(self._trial_index),
                        "axis": trial.axis,
                        "level": float(trial.level),
                        "coast_detection_sec": -1.0,
                        "ascent_velocity_mps": velocity_y,
                        "sample_duration_sec": 0.0,
                        "detected": False,
                    }
                )
                self.get_logger().warn(
                    f"heave_y coast timed out after {elapsed:.2f}s without upward velocity "
                    f">= {threshold:.3f}m/s; no coast samples will enter the fit"
                )
                return True
            return False

        return now - self._heave_y_ascent_sample_start_sec >= self._heave_y_ascent_sample_sec

    def _phase_duration(self, trial: Trial, phase_name: str) -> float:
        if self._is_attitude_pulse_axis(trial.axis):
            if phase_name == "excitation":
                return max(self._attitude_pulse_sec, 0.0) + max(self._attitude_ringdown_sec, 0.0)
        if self._is_heave_dive_and_coast_trial(trial) and phase_name == "coast":
            return max(self._heave_y_coast_sec, 0.0)
        for phase in self._phases:
            if phase.name == phase_name:
                return phase.duration
        return 0.0

    def _attitude_limit_exceeded(self, trial: Trial) -> tuple[bool, str]:
        if self._ignore_attitude_safety_checks or not self._is_attitude_pulse_axis(trial.axis):
            return False, ""
        if not self._latest_orientation_valid:
            return False, ""
        index = AXIS_TO_NU_INDEX[trial.axis]
        angle = abs(float(self._latest_angle_body[index]))
        rate = abs(float(self._latest_angular_body[index - 3]))
        max_angle = self._attitude_max_abs_angle_rad[trial.axis]
        max_rate = self._attitude_max_abs_rate_radps[trial.axis]
        if max_angle > 0.0 and angle > max_angle:
            return True, (
                f"relative angle {math.degrees(angle):.2f}deg > "
                f"{math.degrees(max_angle):.2f}deg"
            )
        if max_rate > 0.0 and rate > max_rate:
            return True, f"rate {rate:.3f}rad/s > {max_rate:.3f}rad/s"
        return False, ""

    def _build_trials(self) -> list[Trial]:
        trials: list[Trial] = []
        for axis in self._axes:
            values = [float(v) for v in self._levels[axis] if math.isfinite(float(v)) and abs(float(v)) > 1e-9]
            for value in values:
                for repeat_index in range(1, getattr(self, "_repeat_per_trial", 1) + 1):
                    if axis == "heave_y" and self._heave_y_identification_mode == "dive_and_coast":
                        # Controller body +y is up. This mode always drives down, then coasts upward.
                        trials.append(Trial(axis, -abs(value), repeat_index))
                        continue
                    if self._include_negative:
                        trials.append(Trial(axis, abs(value), repeat_index))
                        trials.append(Trial(axis, -abs(value), repeat_index))
                    else:
                        trials.append(Trial(axis, value, repeat_index))
        return trials

    def _apply_cli_amplitude_override(self) -> None:
        amplitude = float(self.declare_parameter("amplitude", 0.0).value)
        amplitudes = _as_vec(self.declare_parameter("amplitudes", [0.0]).value, 64)

        override_values: list[float] = []
        if math.isfinite(amplitude) and abs(amplitude) > 1e-9:
            override_values.append(amplitude)
        override_values.extend(float(v) for v in amplitudes if math.isfinite(float(v)) and abs(float(v)) > 1e-9)

        if not override_values:
            return

        if len(self._axes) != 1:
            raise RuntimeError(
                "amplitude/amplitudes override is only allowed for a single selected axis; "
                "set axes:='[surge_x]' or use axis-specific *_levels parameters."
            )

        axis = self._axes[0]
        self._levels[axis] = np.array(override_values, dtype=np.float64)
        self.get_logger().info(
            f"overriding {axis} levels from command line: "
            f"{[round(v, 6) for v in self._levels[axis].tolist()]}"
        )

    def _start_service(self, request, response):
        del request
        if self._running:
            response.success = False
            response.message = "hydrodynamic identification is already running"
            return response
        self._start_run()
        response.success = True
        response.message = f"started {len(self._trials)} trials"
        return response

    def _stop_service(self, request, response):
        del request
        if not self._running:
            response.success = False
            response.message = "hydrodynamic identification is not running"
            return response
        self._finish_run(stopped=True)
        response.success = True
        response.message = "stopped and sent zero thrust"
        return response

    def _continue_service(self, request, response):
        del request
        with self._pause_condition:
            if not self._running or not self._awaiting_continue:
                response.success = False
                response.message = "hydrodynamic identification is not waiting between trials"
                return response
            self._continue_requested = True
            self._pause_condition.notify_all()
        response.success = True
        response.message = "continue requested; next trial will start on the next timer tick"
        return response

    def _stdin_continue_loop(self) -> None:
        while not self._stdin_stop_event.is_set():
            with self._pause_condition:
                self._pause_condition.wait_for(
                    lambda: self._awaiting_continue or self._stdin_stop_event.is_set()
                )
                if self._stdin_stop_event.is_set():
                    return

            try:
                input()
            except (EOFError, OSError):
                self.get_logger().warn(
                    "stdin is unavailable; use "
                    "/hydrodynamic_identifier/continue to resume the next trial"
                )
                return

            with self._pause_condition:
                if self._awaiting_continue:
                    self._continue_requested = True
                    self._pause_condition.notify_all()

    def _start_run(self) -> None:
        if not self._trials:
            raise RuntimeError("no hydrodynamic identification trials configured")
        now = self._now_sec()
        self._current_output_axis = self._output_axis_label()
        self._current_output_base, self._current_output_run_index = self._next_output_base(self._current_output_axis)
        self._csv_file = (self._current_output_base.with_suffix(".csv")).open("w", newline="", encoding="utf-8")
        self._csv_writer = csv.writer(self._csv_file)
        self._write_header()
        self._samples.clear()
        self._attitude_safety_cutoffs.clear()
        self._heave_y_ascent_events.clear()
        self._attitude_safety_cutoff_sec = None
        self._heave_y_coast_seen_downward = False
        self._heave_y_ascent_sample_start_sec = None
        self._trial_index = -1
        self._phase_index = 0
        self._phase_start_sec = now
        self._run_start_sec = now
        self._post_zero_active = False
        with self._pause_condition:
            self._awaiting_continue = False
            self._continue_requested = False
            self._pause_condition.notify_all()
        self._previous_angular_body = None
        self._previous_angular_body_sec = None
        self._latest_linear_body_dot[:] = 0.0
        self._latest_angular_body_dot[:] = 0.0
        self._running = True
        self._publish_thrusters(np.zeros(THRUSTER_COUNT, dtype=np.float64))
        self.get_logger().warn(
            f"starting hydrodynamic identification: output={self._current_output_base}, "
            f"pre_zero={self._pre_zero_sec:.1f}s, trials={len(self._trials)}"
        )

    def _output_axis_label(self) -> str:
        trial_axes = tuple(dict.fromkeys(trial.axis for trial in self._trials))
        if len(trial_axes) == 1:
            return trial_axes[0]
        return "multi_axis"

    def _next_output_base(self, axis_label: str) -> tuple[Path, int]:
        axis_dir = Path(self._output_directory).expanduser() / axis_label
        axis_dir.mkdir(parents=True, exist_ok=True)
        prefix = f"{axis_label}_"
        max_index = 0
        for child in axis_dir.iterdir():
            name = child.name
            if not name.startswith(prefix):
                continue
            suffix = name[len(prefix) :]
            if len(suffix) >= 3 and suffix[:3].isdigit():
                max_index = max(max_index, int(suffix[:3]))

        index = max_index + 1
        while True:
            run_id = f"{axis_label}_{index:03d}"
            run_dir = axis_dir / run_id
            try:
                run_dir.mkdir()
            except FileExistsError:
                index += 1
                continue
            return run_dir / run_id, index

    def _finish_run(self, *, stopped: bool = False) -> None:
        self._publish_thrusters(np.zeros(THRUSTER_COUNT, dtype=np.float64))
        with self._pause_condition:
            self._awaiting_continue = False
            self._continue_requested = False
            self._pause_condition.notify_all()
        self._running = False
        plot_files = self._write_motion_plots()
        validation_run = self._run_role == "held_out_fossen_validation"
        result = (
            self._validation_run_results(stopped=stopped, plot_files=plot_files)
            if validation_run
            else self._fit_results(stopped=stopped, plot_files=plot_files)
        )
        if self._current_output_base is not None:
            result_path = self._current_output_base.with_suffix(
                ".validation.json" if validation_run else ".fit.json"
            )
            result_path.write_text(
                json.dumps(_json_safe(result), indent=2, sort_keys=True, allow_nan=False),
                encoding="utf-8",
            )
            result_name = "held-out hydrodynamic validation" if validation_run else "hydrodynamic identification"
            self.get_logger().warn(f"{result_name} {'stopped' if stopped else 'finished'}: {result_path}")
        self._close_csv()

    def _write_motion_plots(self) -> list[str]:
        if self._current_output_base is None:
            return []
        axes = [axis for axis in AXIS_ORDER if any(sample.axis == axis for sample in self._samples)]
        plot_files: list[str] = []
        for axis in axes:
            if self._current_output_axis == axis:
                plot_path = self._current_output_base.with_name(
                    f"{self._current_output_base.name}_velocity_acceleration.png"
                )
                fit_plot_path = self._current_output_base.with_name(
                    f"{self._current_output_base.name}_fit_velocity_acceleration.png"
                )
            else:
                plot_path = self._current_output_base.with_name(
                    f"{self._current_output_base.name}_{axis}_velocity_acceleration.png"
                )
                fit_plot_path = self._current_output_base.with_name(
                    f"{self._current_output_base.name}_{axis}_fit_velocity_acceleration.png"
                )
            try:
                _write_motion_png(self._samples, axis, plot_path)
                _write_motion_png(self._samples, axis, fit_plot_path, fit_only=True)
            except Exception as exc:
                self.get_logger().warn(f"failed to write {axis} motion plot: {exc}")
                continue
            if plot_path.exists():
                plot_files.append(str(plot_path))
                self.get_logger().info(f"wrote {axis} motion plot: {plot_path}")
            if fit_plot_path.exists():
                plot_files.append(str(fit_plot_path))
                self.get_logger().info(f"wrote {axis} fit motion plot: {fit_plot_path}")
        return plot_files

    def _close_csv(self) -> None:
        if self._csv_file is not None:
            self._csv_file.flush()
            self._csv_file.close()
        self._csv_file = None
        self._csv_writer = None

    def _write_header(self) -> None:
        assert self._csv_writer is not None
        self._csv_writer.writerow(
            [
                "time_sec",
                "sample_time_ros_sec",
                "trial_index",
                "repeat_index",
                "axis",
                "phase",
                "phase_elapsed_sec",
                "level",
                "sample_window",
                "tau_target_fx_n",
                "tau_target_fy_n",
                "tau_target_fz_n",
                "tau_target_mx_nm",
                "tau_target_my_yaw_nm",
                "tau_target_mz_nm",
                "tau_fit_fx_n",
                "tau_fit_fy_n",
                "tau_fit_fz_n",
                "tau_fit_mx_nm",
                "tau_fit_my_yaw_nm",
                "tau_fit_mz_nm",
                "nu_x_mps",
                "nu_y_mps",
                "nu_z_mps",
                "nu_yaw_radps",
                "nudot_x_mps2",
                "nudot_y_mps2",
                "nudot_z_mps2",
                "nudot_yaw_radps2",
                "nu_roll_x_radps",
                "nu_pitch_z_radps",
                "nudot_roll_x_radps2",
                "nudot_pitch_z_radps2",
                "angle_roll_x_rad",
                "angle_pitch_z_rad",
                "angle_yaw_y_rad",
                "orientation_valid",
                "imu_mcu_time_ms",
                "imu_ros_stamp_sec",
                "dvl_ros_stamp_sec",
                "host_receive_time_ns",
                "imu_age_sec",
                "dvl_age_sec",
                "telemetry_sequence",
                "synchronization_valid",
                "dvl_frame_id",
                "dvl_raw_linear_x",
                "dvl_raw_linear_y",
                "dvl_raw_linear_z",
                "dvl_controller_linear_x",
                "dvl_controller_linear_y",
                "dvl_controller_linear_z",
                "imu_frame_id",
                "imu_raw_angular_x",
                "imu_raw_angular_y",
                "imu_raw_angular_z",
                "imu_controller_angular_x",
                "imu_controller_angular_y",
                "imu_controller_angular_z",
                "imu_raw_linear_accel_x",
                "imu_raw_linear_accel_y",
                "imu_raw_linear_accel_z",
                "imu_controller_linear_accel_x",
                "imu_controller_linear_accel_y",
                "imu_controller_linear_accel_z",
                "imu_controller_gravity_x",
                "imu_controller_gravity_y",
                "imu_controller_gravity_z",
                "imu_controller_kinematic_accel_x",
                "imu_controller_kinematic_accel_y",
                "imu_controller_kinematic_accel_z",
                "imu_raw_quat_x",
                "imu_raw_quat_y",
                "imu_raw_quat_z",
                "imu_raw_quat_w",
                "imu_controller_quat_x",
                "imu_controller_quat_y",
                "imu_controller_quat_z",
                "imu_controller_quat_w",
                *[f"force_cmd_{name}_n" for name in THRUSTER_NAMES],
                *[f"rpm_canonical_{name}" for name in THRUSTER_NAMES],
                *[f"force_from_rpm_{name}_n" for name in THRUSTER_NAMES],
            ]
        )

    def _tick(self) -> None:
        if not self._running:
            return

        now = self._now_sec()
        if self._use_hardware_telemetry:
            self._update_synchronized_state(now)
        if self._trial_index < 0:
            if now - self._phase_start_sec < max(self._pre_zero_sec, 0.0):
                self._publish_and_log("pre_zero", None, 0.0, False)
                return
            self._advance_trial(now)

        with self._pause_condition:
            if self._awaiting_continue:
                if self._continue_requested:
                    self._awaiting_continue = False
                    self._continue_requested = False
                    self._phase_start_sec = now
                    should_resume = True
                else:
                    should_resume = False
            else:
                should_resume = False
        if not should_resume and self._awaiting_continue:
            self._publish_thrusters(np.zeros(THRUSTER_COUNT, dtype=np.float64))
            return
        if should_resume:
            self.get_logger().warn(
                f"continuing trial {self._trial_index + 1}/{len(self._trials)}"
            )

        if self._post_zero_active:
            self._post_zero_or_finish(now)
            return

        trial = self._trials[self._trial_index]
        phase = self._phases[self._phase_index]
        elapsed = now - self._phase_start_sec
        if (
            self._is_attitude_pulse_axis(trial.axis)
            and phase.name == "excitation"
            and self._attitude_safety_cutoff_sec is None
        ):
            exceeded, reason = self._attitude_limit_exceeded(trial)
            if exceeded:
                self._attitude_safety_cutoff_sec = elapsed
                self._attitude_safety_cutoffs.append(
                    {
                        "trial_index": int(self._trial_index),
                        "axis": trial.axis,
                        "level": float(trial.level),
                        "cutoff_sec": float(elapsed),
                        "reason": reason,
                    }
                )
                self.get_logger().warn(
                    f"cutting {trial.axis} pulse early for safety: {reason}; "
                    "recording the remaining excitation window as zero-thrust ringdown"
                )
        if self._is_heave_dive_and_coast_trial(trial) and phase.name == "coast":
            if self._update_heave_y_coast_sampling(now, trial, elapsed):
                self._advance_phase(now)
                return
        elif elapsed >= self._phase_duration(trial, phase.name):
            self._advance_phase(now)
            return

        if self._is_attitude_pulse_axis(trial.axis):
            sample_window = phase.name == "excitation" and elapsed >= max(0.0, self._attitude_fit_start_sec)
        elif self._is_heave_dive_and_coast_trial(trial):
            sample_window = (
                (phase.name == "excitation" and elapsed >= max(0.0, self._step_fit_start_sec))
                or (phase.name == "coast" and self._heave_y_ascent_sample_start_sec is not None)
            )
        else:
            sample_window = phase.name == "excitation" and elapsed >= max(0.0, self._step_fit_start_sec)
        self._publish_and_log(phase.name, trial, elapsed, sample_window)

    def _advance_trial(self, now: float) -> None:
        previous_trial_index = self._trial_index
        self._trial_index += 1
        if self._trial_index >= len(self._trials):
            self._post_zero_active = True
            self._phase_start_sec = now
            self.get_logger().warn(f"all trials complete; sending zero thrust for {self._post_zero_sec:.1f}s")
            return
        self._phase_index = 0
        self._phase_start_sec = now
        self._attitude_safety_cutoff_sec = None
        self._heave_y_coast_seen_downward = False
        self._heave_y_ascent_sample_start_sec = None
        if self._pause_between_trials and previous_trial_index >= 0:
            with self._pause_condition:
                self._awaiting_continue = True
                self._continue_requested = False
                self._pause_condition.notify_all()
            self._publish_thrusters(np.zeros(THRUSTER_COUNT, dtype=np.float64))
            self.get_logger().warn(
                f"trial {previous_trial_index + 1}/{len(self._trials)} complete; "
                f"zero thrust active. Manually reposition the vehicle, then press Enter "
                f"or call /hydrodynamic_identifier/continue for trial "
                f"{self._trial_index + 1}/{len(self._trials)}."
            )
        trial = self._trials[self._trial_index]
        if self._is_attitude_pulse_axis(trial.axis):
            self.get_logger().warn(
                f"trial {self._trial_index + 1}/{len(self._trials)} axis={trial.axis} "
                f"pulse_torque={trial.level:+.3f}Nm; "
                f"baseline={self._baseline_sec:.1f}s pulse={self._attitude_pulse_sec:.1f}s "
                f"ringdown={self._attitude_ringdown_sec:.1f}s"
            )
        else:
            if self._is_heave_dive_and_coast_trial(trial):
                self.get_logger().warn(
                    f"trial {self._trial_index + 1}/{len(self._trials)} axis=heave_y "
                    f"dive_force={trial.level:+.3f}N; baseline={self._baseline_sec:.1f}s "
                    f"dive={self._hold_sec:.1f}s wait_ascent={self._heave_y_coast_sec:.1f}s "
                    f"threshold={self._heave_y_ascent_velocity_threshold:.3f}m/s "
                    f"ascent_sample={self._heave_y_ascent_sample_sec:.1f}s rest={self._rest_sec:.1f}s"
                )
                return
            self.get_logger().info(
                f"trial {self._trial_index + 1}/{len(self._trials)} axis={trial.axis} level={trial.level:+.3f}"
            )

    def _advance_phase(self, now: float) -> None:
        self._phase_index += 1
        self._phase_start_sec = now
        if self._phase_index >= len(self._phases):
            self._advance_trial(now)
            return

    def _post_zero_or_finish(self, now: float) -> None:
        if now - self._phase_start_sec < max(self._post_zero_sec, 0.0):
            self._publish_and_log("post_zero", None, now - self._phase_start_sec, False)
            return
        self._finish_run(stopped=False)

    def _publish_and_log(self, phase: str, trial: Trial | None, phase_elapsed: float, sample_window: bool) -> None:
        tau_target = np.zeros(6, dtype=np.float64)
        if (
            phase == "excitation"
            and trial is not None
            and trial.axis in STEP_AXIS_ORDER
        ):
            tau_target[AXIS_TO_TAU_INDEX[trial.axis]] = trial.level
        elif (
            phase == "excitation"
            and trial is not None
            and self._is_attitude_pulse_axis(trial.axis)
            and phase_elapsed <= max(self._attitude_pulse_sec, 0.0)
            and self._attitude_safety_cutoff_sec is None
        ):
            tau_target[AXIS_TO_TAU_INDEX[trial.axis]] = trial.level

        forces = self._allocate_forces(tau_target)
        tau_command = WRENCH_FROM_THRUST @ forces
        self._publish_thrusters(forces)
        self._last_command_forces = forces
        self._last_tau_command = tau_command

        rpm_is_fresh = (
            self._latest_rpm is not None
            and self._latest_rpm_sec is not None
            and (self._rpm_stale_sec <= 0.0 or self._now_sec() - self._latest_rpm_sec <= self._rpm_stale_sec)
        )
        rpm = self._latest_rpm if rpm_is_fresh else np.zeros(THRUSTER_COUNT, dtype=np.float64)
        force_from_rpm = self._curve.rpm_to_force(rpm) if rpm_is_fresh else forces
        tau_actual = WRENCH_FROM_THRUST @ force_from_rpm
        self._last_tau_actual = tau_actual

        nu, nu_dot = self._current_nu_and_derivative()
        active_axis = trial.axis if trial is not None else "none"
        if active_axis in AXIS_TO_NU_INDEX:
            self._samples.append(
                Sample(
                    time_sec=self._latest_sample_time_sec,
                    axis=active_axis,
                    phase=phase,
                    phase_elapsed_sec=phase_elapsed,
                    sample_window=sample_window,
                    tau=tau_actual.astype(np.float64, copy=True),
                    nu=nu.astype(np.float64, copy=True),
                    nu_dot=nu_dot.astype(np.float64, copy=True),
                    angle=self._latest_angle_body.astype(np.float64, copy=True),
                    orientation_valid=self._latest_orientation_valid,
                    synchronization_valid=self._latest_synchronization_valid,
                )
            )

        if self._csv_writer is not None:
            self._csv_writer.writerow(
                [
                    f"{self._now_sec():.6f}",
                    f"{self._latest_sample_time_sec:.9f}",
                    self._trial_index,
                    trial.repeat_index if trial is not None else 0,
                    active_axis,
                    phase,
                    f"{phase_elapsed:.6f}",
                    f"{trial.level if trial is not None else 0.0:.6f}",
                    int(sample_window),
                    *[f"{v:.9g}" for v in tau_target],
                    *[f"{v:.9g}" for v in tau_actual],
                    f"{nu[0]:.9g}",
                    f"{nu[1]:.9g}",
                    f"{nu[2]:.9g}",
                    f"{nu[4]:.9g}",
                    f"{nu_dot[0]:.9g}",
                    f"{nu_dot[1]:.9g}",
                    f"{nu_dot[2]:.9g}",
                    f"{nu_dot[4]:.9g}",
                    f"{nu[3]:.9g}",
                    f"{nu[5]:.9g}",
                    f"{nu_dot[3]:.9g}",
                    f"{nu_dot[5]:.9g}",
                    f"{self._latest_angle_body[3]:.9g}",
                    f"{self._latest_angle_body[5]:.9g}",
                    f"{self._latest_angle_body[4]:.9g}",
                    int(self._latest_orientation_valid),
                    self._latest_imu_mcu_time_ms,
                    f"{self._latest_imu_ros_stamp_sec:.9f}",
                    f"{self._latest_dvl_stamp_sec:.9f}",
                    self._latest_host_receive_time_ns,
                    f"{self._latest_imu_age_sec:.9f}",
                    f"{self._latest_dvl_age_sec:.9f}",
                    self._latest_telemetry_sequence,
                    int(self._latest_synchronization_valid),
                    self._latest_dvl_frame_id,
                    *[f"{v:.9g}" for v in self._latest_dvl_linear_raw],
                    *[f"{v:.9g}" for v in self._latest_dvl_linear_controller],
                    self._latest_imu_frame_id,
                    *[f"{v:.9g}" for v in self._latest_imu_angular_raw],
                    *[f"{v:.9g}" for v in self._latest_imu_angular_controller],
                    *[f"{v:.9g}" for v in self._latest_imu_linear_accel_raw],
                    *[f"{v:.9g}" for v in self._latest_imu_linear_accel_controller],
                    *[f"{v:.9g}" for v in self._latest_gravity_controller],
                    *[f"{v:.9g}" for v in self._latest_imu_linear_accel_kinematic],
                    *[f"{v:.9g}" for v in self._latest_imu_quat_raw],
                    *[f"{v:.9g}" for v in self._latest_imu_quat_controller],
                    *[f"{v:.9g}" for v in forces],
                    *[f"{v:.9g}" for v in rpm],
                    *[f"{v:.9g}" for v in force_from_rpm],
                ]
            )

    def _allocate_forces(self, tau: np.ndarray) -> np.ndarray:
        raw = ALLOCATOR_A @ tau.reshape(6)
        clipped = raw.copy()
        for i, force in enumerate(raw):
            if force >= 0.0:
                clipped[i] = min(force, self._force_limit_pos[i])
            else:
                clipped[i] = max(force, -self._force_limit_neg[i])
        return clipped

    def _publish_thrusters(self, forces: np.ndarray) -> None:
        msg = Float32MultiArray()
        msg.data = [float(v) for v in _as_vec(forces, THRUSTER_COUNT)]
        self._thruster_pub.publish(msg)

    def _current_nu_and_derivative(self) -> tuple[np.ndarray, np.ndarray]:
        nu = np.asarray(
            [
                self._latest_linear_body[0],
                self._latest_linear_body[1],
                self._latest_linear_body[2],
                self._latest_angular_body[0],
                self._latest_angular_body[1],
                self._latest_angular_body[2],
            ],
            dtype=np.float64,
        )
        nu_dot = np.asarray(
            [
                self._latest_linear_body_dot[0],
                self._latest_linear_body_dot[1],
                self._latest_linear_body_dot[2],
                self._latest_angular_body_dot[0],
                self._latest_angular_body_dot[1],
                self._latest_angular_body_dot[2],
            ],
            dtype=np.float64,
        )
        return nu, nu_dot

    def _validation_run_results(self, *, stopped: bool, plot_files: Sequence[str] | None = None) -> dict[str, object]:
        """Write audit metadata for a held-out trial without refitting any coefficient.

        The CSV has the same schema as an identification run so that the Unity
        replay path can consume it directly.  A separate ``.validation.json``
        makes its non-fitting role explicit and prevents accidental use as a
        new source of profile parameters.
        """
        return {
            "schema_version": 1,
            "run_role": "held_out_fossen_validation",
            "fit_performed": False,
            "stopped": stopped,
            "model_scope": "axiswise_diagonal_simplified_fossen_validation",
            "coordinate_contract": "controller_x_forward_y_up_z_left",
            "wrench_order": ["Fx", "Fy", "Fz", "Mx_roll", "My_yaw", "Mz_pitch"],
            "velocity_order": [
                "linear_x",
                "linear_y",
                "linear_z",
                "angular_roll_x",
                "angular_yaw_y",
                "angular_pitch_z",
            ],
            "wrench_matrix_version": "right_hand_roll_v2",
            "output_axis": self._current_output_axis,
            "output_run_index": self._current_output_run_index,
            "output_run_id": f"{self._current_output_axis}_{self._current_output_run_index:03d}",
            "output_prefix": self._output_prefix,
            "sample_count": len(self._samples),
            "synchronized_sample_count": sum(1 for sample in self._samples if sample.synchronization_valid),
            "unsynchronized_sample_count": sum(1 for sample in self._samples if not sample.synchronization_valid),
            "trials": [
                {
                    "trial_index": index,
                    "axis": trial.axis,
                    "level": float(trial.level),
                    "repeat_index": int(trial.repeat_index),
                }
                for index, trial in enumerate(self._trials)
            ],
            "phase_protocol": {
                "pre_zero_sec": self._pre_zero_sec,
                "baseline_sec": self._baseline_sec,
                "excitation_sec": self._hold_sec,
                "rest_sec": self._rest_sec,
                "post_zero_sec": self._post_zero_sec,
                "heave_y_identification_mode": self._heave_y_identification_mode,
                "heave_y_coast_sec": self._heave_y_coast_sec,
            },
            "imu_topic": self._imu_topic,
            "dvl_topic": self._dvl_topic,
            "hardware_telemetry_topic": self._hardware_telemetry_topic,
            "use_hardware_telemetry": self._use_hardware_telemetry,
            "hardware_bridge_config": self._resolved_hardware_config,
            "motor_order": self._motor_order,
            "motor_signs": self._motor_signs,
            "thruster_names": list(THRUSTER_NAMES),
            "force_limit_positive_n": [float(v) for v in self._force_limit_pos],
            "force_limit_negative_n": [float(v) for v in self._force_limit_neg],
            "plot_files": list(plot_files or []),
        }

    def _fit_results(self, *, stopped: bool, plot_files: Sequence[str] | None = None) -> dict[str, object]:
        fits: dict[str, object] = {}
        for axis in AXIS_ORDER:
            if axis in ATTITUDE_PULSE_AXIS_ORDER:
                fits[axis] = _fit_attitude_pulse_axis(
                    self._samples,
                    axis,
                    min_abs_angle=max(self._attitude_min_abs_angle_rad, 0.0),
                    min_abs_rate=max(self._attitude_min_abs_rate_radps, 0.0),
                    min_abs_tau=max(self._attitude_min_abs_tau, 0.0),
                    fit_start_sec=max(self._attitude_fit_start_sec, 0.0),
                    max_abs_acceleration=max(self._fit_max_abs_angular_accel, 0.0),
                    accel_outlier_mad_threshold=max(self._fit_accel_outlier_mad_threshold, 0.0),
                    accel_spike_local_mad_threshold=max(self._fit_accel_spike_local_mad_threshold, 0.0),
                    accel_spike_local_window=max(self._fit_accel_spike_local_window, 0),
                    constrain_physical_coefficients=self._fit_constrain_physical_coefficients,
                    max_effective_inertia=max(self._fit_max_effective_inertia, 0.0),
                    max_linear_damping=max(self._fit_max_linear_damping, 0.0),
                    max_quadratic_damping=max(self._fit_max_quadratic_damping, 0.0),
                    max_restoring_stiffness=max(self._fit_max_restoring_stiffness, 0.0),
                    max_abs_bias=max(self._fit_max_abs_bias, 0.0),
                )
            else:
                max_abs_accel = self._fit_max_abs_angular_accel if axis == "yaw_y" else self._fit_max_abs_linear_accel
                is_heave_dive_and_coast = axis == "heave_y" and self._heave_y_identification_mode == "dive_and_coast"
                fits[axis] = _fit_axis(
                    self._samples,
                    axis,
                    min_abs_tau=max(self._min_fit_abs_tau, 0.0),
                    velocity_deadband=max(self._fit_velocity_deadband, 0.0),
                    max_abs_acceleration=max(max_abs_accel, 0.0),
                    accel_outlier_mad_threshold=max(self._fit_accel_outlier_mad_threshold, 0.0),
                    accel_spike_local_mad_threshold=max(self._fit_accel_spike_local_mad_threshold, 0.0),
                    accel_spike_local_window=max(self._fit_accel_spike_local_window, 0),
                    constrain_physical_coefficients=self._fit_constrain_physical_coefficients,
                    max_effective_mass=max(self._fit_max_effective_mass, 0.0),
                    max_effective_inertia=max(self._fit_max_effective_inertia, 0.0),
                    max_linear_damping=max(self._fit_max_linear_damping, 0.0),
                    max_quadratic_damping=max(self._fit_max_quadratic_damping, 0.0),
                    max_abs_bias=max(self._fit_max_abs_bias, 0.0),
                    fit_phases=("excitation", "coast") if is_heave_dive_and_coast else ("excitation",),
                    allow_zero_tau_samples=is_heave_dive_and_coast,
                    fit_type="dive_and_coast" if is_heave_dive_and_coast else "forced_axis",
                )
        return {
            "schema_version": 1,
            "run_role": "identification",
            "fit_performed": True,
            "model": "decoupled_fossen_6dof_with_roll_pitch_attitude_pulse",
            "step_axis_equation": "tau_i = m_eff_i * nu_dot_i + d_linear_i * nu_i + d_quadratic_i * abs(nu_i) * nu_i + bias_i",
            "heave_y_dive_and_coast_equation": "tau_thruster_y = m_eff_y * y_ddot + d_linear_y * y_dot + d_quadratic_y * abs(y_dot) * y_dot + bias_y; bias_y approximates negative net upward buoyancy in controller +y-up coordinates",
            "attitude_pulse_equation": "tau_i = m_eff_i * angle_ddot_i + d_linear_i * angle_dot_i + d_quadratic_i * abs(angle_dot_i) * angle_dot_i + restoring_stiffness_i * sin(angle_i) + bias_i",
            "stopped": stopped,
            "axes": list(AXIS_ORDER),
            "step_axes": list(STEP_AXIS_ORDER),
            "attitude_pulse_axes": list(ATTITUDE_PULSE_AXIS_ORDER),
            "wrench_order": ["Fx", "Fy", "Fz", "Mx_roll", "My_yaw", "Mz_pitch"],
            "velocity_order": ["linear_x", "linear_y", "linear_z", "angular_roll_x", "angular_yaw_y", "angular_pitch_z"],
            "angle_order": ["roll_x", "yaw_y", "pitch_z"],
            "coordinate_contract": "controller_x_forward_y_up_z_left",
            "wrench_matrix_version": "right_hand_roll_v2",
            "roll_wrench_sign_contract": "left_vertical=-Mx_right_vertical=+Mx",
            "imu_input_frame": self._latest_imu_frame_id or "unknown",
            "imu_expected_input_frame": "finsrov_base_link",
            "imu_mounting_correction": "applied_upstream_by_hardware_bridge",
            "angular_vector_transform": "axial_det_times_basis",
            "imu_topic": self._imu_topic,
            "dvl_topic": self._dvl_topic,
            "hardware_telemetry_topic": self._hardware_telemetry_topic,
            "use_hardware_telemetry": self._use_hardware_telemetry,
            "sync_max_telemetry_age_sec": self._sync_max_telemetry_age_sec,
            "sync_max_dvl_age_sec": self._sync_max_dvl_age_sec,
            "sync_buffer_sec": self._sync_buffer_sec,
            "synchronized_sample_count": sum(1 for sample in self._samples if sample.synchronization_valid),
            "unsynchronized_sample_count": sum(1 for sample in self._samples if not sample.synchronization_valid),
            "controller_frame_passthrough": ["controller_world", "controller_body"],
            "raw_to_controller_basis_indices": self._basis_indices,
            "raw_to_controller_basis_signs": self._basis_signs,
            "sample_count": len(self._samples),
            "repeat_per_trial": self._repeat_per_trial,
            "trials": [
                {
                    "trial_index": index,
                    "axis": trial.axis,
                    "level": float(trial.level),
                    "repeat_index": int(trial.repeat_index),
                }
                for index, trial in enumerate(self._trials)
            ],
            "output_axis": self._current_output_axis,
            "output_run_index": self._current_output_run_index,
            "output_run_id": f"{self._current_output_axis}_{self._current_output_run_index:03d}",
            "output_prefix": self._output_prefix,
            "plot_files": list(plot_files or []),
            "step_fit_start_sec": self._step_fit_start_sec,
            "heave_y_identification_mode": self._heave_y_identification_mode,
            "heave_y_coast_sec": self._heave_y_coast_sec,
            "heave_y_ascent_velocity_threshold_mps": self._heave_y_ascent_velocity_threshold,
            "heave_y_ascent_sample_sec": self._heave_y_ascent_sample_sec,
            "heave_y_dive_direction": "controller_negative_y_down" if self._heave_y_identification_mode == "dive_and_coast" else "not_applicable",
            "heave_y_ascent_events": list(self._heave_y_ascent_events),
            "gravity_compensation_enabled": self._gravity_compensation_enabled,
            "gravity_mps2": self._gravity_mps2,
            "fit_max_abs_linear_accel_mps2": self._fit_max_abs_linear_accel,
            "fit_max_abs_angular_accel_radps2": self._fit_max_abs_angular_accel,
            "fit_accel_outlier_mad_threshold": self._fit_accel_outlier_mad_threshold,
            "fit_accel_spike_local_mad_threshold": self._fit_accel_spike_local_mad_threshold,
            "fit_accel_spike_local_window": self._fit_accel_spike_local_window,
            "fit_constrain_physical_coefficients": self._fit_constrain_physical_coefficients,
            "fit_max_effective_mass_kg": self._fit_max_effective_mass,
            "fit_max_effective_inertia_kgm2": self._fit_max_effective_inertia,
            "fit_max_linear_damping": self._fit_max_linear_damping,
            "fit_max_quadratic_damping": self._fit_max_quadratic_damping,
            "fit_max_restoring_stiffness_nm_per_rad": self._fit_max_restoring_stiffness,
            "fit_max_abs_bias": self._fit_max_abs_bias,
            "attitude_pulse_sec": self._attitude_pulse_sec,
            "attitude_ringdown_sec": self._attitude_ringdown_sec,
            "attitude_fit_start_sec": self._attitude_fit_start_sec,
            "attitude_min_abs_tau_nm": self._attitude_min_abs_tau,
            "attitude_min_abs_angle_rad": self._attitude_min_abs_angle_rad,
            "attitude_min_abs_rate_radps": self._attitude_min_abs_rate_radps,
            "roll_x_max_abs_angle_rad": self._attitude_max_abs_angle_rad["roll_x"],
            "pitch_z_max_abs_angle_rad": self._attitude_max_abs_angle_rad["pitch_z"],
            "roll_x_max_abs_rate_radps": self._attitude_max_abs_rate_radps["roll_x"],
            "pitch_z_max_abs_rate_radps": self._attitude_max_abs_rate_radps["pitch_z"],
            "ignore_attitude_safety_checks": self._ignore_attitude_safety_checks,
            "roll_x_equilibrium_angle_rad": float(self._equilibrium_angles[AXIS_TO_NU_INDEX["roll_x"]]),
            "pitch_z_equilibrium_angle_rad": float(self._equilibrium_angles[AXIS_TO_NU_INDEX["pitch_z"]]),
            "attitude_safety_cutoffs": list(self._attitude_safety_cutoffs),
            "hardware_bridge_config": self._resolved_hardware_config,
            "motor_order": self._motor_order,
            "motor_signs": self._motor_signs,
            "thruster_names": list(THRUSTER_NAMES),
            "force_limit_positive_n": [float(v) for v in self._force_limit_pos],
            "force_limit_negative_n": [float(v) for v in self._force_limit_neg],
            "fits": fits,
        }

    @staticmethod
    def _header_stamp_sec(msg: object, fallback: float) -> float:
        header = getattr(msg, "header", None)
        stamp = getattr(header, "stamp", None)
        if stamp is None:
            return fallback
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return value if math.isfinite(value) and value > 0.0 else fallback

    @staticmethod
    def _interpolate_values(first: np.ndarray, second: np.ndarray, alpha: float) -> np.ndarray:
        return np.asarray(first + float(alpha) * (second - first), dtype=np.float64)

    def _telemetry_at(self, sample_time_sec: float) -> TelemetryBufferSample:
        records = list(self._telemetry_buffer)
        if not records:
            raise RuntimeError("telemetry buffer is empty")
        if sample_time_sec <= records[0].time_sec:
            return records[0]
        if sample_time_sec >= records[-1].time_sec:
            return records[-1]
        for first, second in zip(records, records[1:]):
            if first.time_sec <= sample_time_sec <= second.time_sec:
                span = max(second.time_sec - first.time_sec, 1e-9)
                alpha = (sample_time_sec - first.time_sec) / span
                nearest = first if alpha < 0.5 else second
                orientation = self._interpolate_values(first.orientation_xyzw, second.orientation_xyzw, alpha)
                norm = float(np.linalg.norm(orientation))
                if norm > 1e-9:
                    orientation /= norm
                else:
                    orientation = nearest.orientation_xyzw.copy()
                return TelemetryBufferSample(
                    time_sec=float(sample_time_sec),
                    frame_id=nearest.frame_id,
                    mcu_time_ms=int(nearest.mcu_time_ms),
                    telemetry_sequence=int(nearest.telemetry_sequence),
                    host_receive_time_ns=int(nearest.host_receive_time_ns),
                    orientation_xyzw=orientation,
                    angular_velocity_xyz=self._interpolate_values(
                        first.angular_velocity_xyz, second.angular_velocity_xyz, alpha
                    ),
                    linear_acceleration_xyz=self._interpolate_values(
                        first.linear_acceleration_xyz, second.linear_acceleration_xyz, alpha
                    ),
                    rpm=self._interpolate_values(first.rpm, second.rpm, alpha),
                )
        return records[-1]

    def _dvl_at(self, sample_time_sec: float) -> DvlBufferSample:
        records = list(self._dvl_buffer)
        if not records:
            raise RuntimeError("DVL buffer is empty")
        if sample_time_sec <= records[0].time_sec:
            return records[0]
        if sample_time_sec >= records[-1].time_sec:
            return records[-1]
        for first, second in zip(records, records[1:]):
            if first.time_sec <= sample_time_sec <= second.time_sec:
                span = max(second.time_sec - first.time_sec, 1e-9)
                alpha = (sample_time_sec - first.time_sec) / span
                nearest = first if alpha < 0.5 else second
                return DvlBufferSample(
                    time_sec=float(sample_time_sec),
                    raw_linear=self._interpolate_values(first.raw_linear, second.raw_linear, alpha),
                    controller_linear=self._interpolate_values(
                        first.controller_linear, second.controller_linear, alpha
                    ),
                    frame_id=nearest.frame_id,
                )
        return records[-1]

    def _update_synchronized_state(self, now: float) -> None:
        if not self._telemetry_buffer or not self._dvl_buffer:
            self._latest_synchronization_valid = False
            return

        latest_telemetry = self._telemetry_buffer[-1]
        latest_dvl = self._dvl_buffer[-1]
        sample_time = min(latest_telemetry.time_sec, latest_dvl.time_sec)
        telemetry = self._telemetry_at(sample_time)
        dvl = self._dvl_at(sample_time)

        raw_angular = self._angular_body_from_ros(telemetry.angular_velocity_xyz, telemetry.frame_id)
        raw_linear_accel = self._body_from_ros(telemetry.linear_acceleration_xyz, telemetry.frame_id)
        controller_quat, orientation_valid = _quat_to_controller(
            telemetry.orientation_xyzw, self._basis_matrix, telemetry.frame_id
        )
        roll_x, pitch_z, yaw_y, rpy_valid = _quat_to_controller_rpy_rad(controller_quat)
        orientation_valid = orientation_valid and rpy_valid
        gravity_controller = (
            _gravity_controller_body_from_quaternion(controller_quat, self._gravity_mps2)
            if orientation_valid
            else np.zeros(3, dtype=np.float64)
        )
        if self._gravity_compensation_enabled and orientation_valid:
            linear_accel_kinematic = raw_linear_accel - gravity_controller
        else:
            linear_accel_kinematic = raw_linear_accel.copy()

        if self._previous_angular_body is not None and self._previous_angular_body_sec is not None:
            dt = max(sample_time - self._previous_angular_body_sec, 1e-4)
            raw_dot = (raw_angular - self._previous_angular_body) / dt
            self._latest_angular_body_dot = (
                self._accel_alpha * raw_dot + (1.0 - self._accel_alpha) * self._latest_angular_body_dot
            )
        self._previous_angular_body = raw_angular.copy()
        self._previous_angular_body_sec = sample_time
        self._latest_angular_body = raw_angular
        self._latest_angular_body_dot = np.asarray(self._latest_angular_body_dot, dtype=np.float64)
        self._latest_linear_body_dot = (
            self._accel_alpha * linear_accel_kinematic
            + (1.0 - self._accel_alpha) * self._latest_linear_body_dot
        )
        self._latest_linear_body = dvl.controller_linear.copy()
        self._latest_dvl_frame_id = dvl.frame_id
        self._latest_dvl_linear_raw = dvl.raw_linear.copy()
        self._latest_dvl_linear_controller = dvl.controller_linear.copy()
        self._latest_imu_frame_id = telemetry.frame_id
        self._latest_imu_linear_accel_raw = telemetry.linear_acceleration_xyz.copy()
        self._latest_imu_linear_accel_controller = raw_linear_accel
        self._latest_imu_angular_raw = telemetry.angular_velocity_xyz.copy()
        self._latest_imu_angular_controller = raw_angular
        self._latest_gravity_controller = gravity_controller
        self._latest_imu_linear_accel_kinematic = linear_accel_kinematic
        self._latest_imu_quat_raw = telemetry.orientation_xyzw.copy()
        self._latest_imu_quat_controller = controller_quat
        self._latest_orientation_valid = orientation_valid
        if orientation_valid:
            angle = np.zeros(6, dtype=np.float64)
            angle[AXIS_TO_NU_INDEX["roll_x"]] = roll_x
            angle[AXIS_TO_NU_INDEX["yaw_y"]] = yaw_y
            angle[AXIS_TO_NU_INDEX["pitch_z"]] = pitch_z
            self._latest_angle_body = angle - self._equilibrium_angles

        self._latest_rpm = telemetry.rpm.copy()
        self._latest_rpm_sec = telemetry.time_sec
        self._latest_sample_time_sec = sample_time
        self._latest_imu_mcu_time_ms = telemetry.mcu_time_ms
        self._latest_imu_ros_stamp_sec = telemetry.time_sec
        self._latest_dvl_stamp_sec = dvl.time_sec
        self._latest_host_receive_time_ns = telemetry.host_receive_time_ns
        self._latest_imu_age_sec = max(0.0, now - latest_telemetry.time_sec)
        self._latest_dvl_age_sec = max(0.0, now - latest_dvl.time_sec)
        self._latest_telemetry_sequence = telemetry.telemetry_sequence
        self._latest_synchronization_valid = (
            self._latest_imu_age_sec <= max(self._sync_max_telemetry_age_sec, 0.0)
            and self._latest_dvl_age_sec <= max(self._sync_max_dvl_age_sec, 0.0)
        )

    def _dvl_callback(self, msg: TwistWithCovarianceStamped) -> None:
        raw_linear = _as_vec([msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z], 3)
        linear_body = self._body_from_ros(raw_linear, msg.header.frame_id)
        now = self._now_sec()
        sample_time = self._header_stamp_sec(msg, now)
        self._dvl_buffer.append(
            DvlBufferSample(
                time_sec=sample_time,
                raw_linear=raw_linear.copy(),
                controller_linear=linear_body.copy(),
                frame_id=msg.header.frame_id or "",
            )
        )
        cutoff = sample_time - max(self._sync_buffer_sec, 0.1)
        while self._dvl_buffer and self._dvl_buffer[0].time_sec < cutoff:
            self._dvl_buffer.popleft()
        self._latest_dvl_frame_id = msg.header.frame_id or ""
        self._latest_dvl_linear_raw = raw_linear
        self._latest_dvl_linear_controller = linear_body
        self._latest_linear_body = linear_body
        self._latest_dvl_stamp_sec = sample_time
        self._latest_dvl_age_sec = max(0.0, now - sample_time)

    def _hardware_telemetry_callback(self, msg: HardwareTelemetry) -> None:
        """Store one atomic MCU packet; processing happens on the common timer."""
        if len(msg.rpm) != THRUSTER_COUNT:
            return
        now = self._now_sec()
        sample_time = self._header_stamp_sec(msg, now)
        self._telemetry_buffer.append(
            TelemetryBufferSample(
                time_sec=sample_time,
                frame_id=msg.header.frame_id or "",
                mcu_time_ms=int(msg.mcu_time_ms),
                telemetry_sequence=int(msg.telemetry_sequence),
                host_receive_time_ns=int(msg.host_receive_time_ns),
                orientation_xyzw=_as_vec(msg.orientation_xyzw, 4),
                angular_velocity_xyz=_as_vec(msg.angular_velocity_xyz, 3),
                linear_acceleration_xyz=_as_vec(msg.linear_acceleration_xyz, 3),
                rpm=_as_vec(msg.rpm, THRUSTER_COUNT),
            )
        )
        cutoff = sample_time - max(self._sync_buffer_sec, 0.1)
        while self._telemetry_buffer and self._telemetry_buffer[0].time_sec < cutoff:
            self._telemetry_buffer.popleft()

        # The authoritative DVL timestamp is still the one on dvl_link. This
        # callback only updates the atomic IMU/RPM history.
        self._latest_imu_mcu_time_ms = int(msg.mcu_time_ms)
        self._latest_telemetry_sequence = int(msg.telemetry_sequence)

    def _imu_callback(self, msg: Imu) -> None:
        now = self._now_sec()
        frame_id = msg.header.frame_id
        raw_linear_accel = _as_vec(
            [msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z],
            3,
        )
        linear_accel_body = self._body_from_ros(raw_linear_accel, frame_id)
        self._latest_imu_frame_id = frame_id or ""
        self._latest_imu_linear_accel_raw = raw_linear_accel
        self._latest_imu_linear_accel_controller = linear_accel_body
        raw_angular = _as_vec(
            [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z],
            3,
        )
        angular_body = self._angular_body_from_ros(raw_angular, frame_id)
        self._latest_imu_angular_raw = raw_angular
        self._latest_imu_angular_controller = angular_body
        if self._previous_angular_body is not None and self._previous_angular_body_sec is not None:
            dt = max(now - self._previous_angular_body_sec, 1e-4)
            raw_dot = (angular_body - self._previous_angular_body) / dt
            self._latest_angular_body_dot = (
                self._accel_alpha * raw_dot + (1.0 - self._accel_alpha) * self._latest_angular_body_dot
            )
        self._latest_angular_body = angular_body
        self._previous_angular_body = angular_body.copy()
        self._previous_angular_body_sec = now
        raw_quat = np.asarray(
            [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w],
            dtype=np.float64,
        )
        controller_quat, valid = _quat_to_controller(raw_quat, self._basis_matrix, frame_id)
        self._latest_imu_quat_raw = raw_quat
        self._latest_imu_quat_controller = controller_quat
        roll_x, pitch_z, yaw_y, rpy_valid = _quat_to_controller_rpy_rad(controller_quat)
        valid = valid and rpy_valid
        if msg.orientation_covariance[0] < 0.0:
            valid = False
        self._latest_orientation_valid = valid
        if valid:
            gravity_controller = _gravity_controller_body_from_quaternion(
                controller_quat,
                self._gravity_mps2,
            )
        else:
            gravity_controller = np.zeros(3, dtype=np.float64)
        self._latest_gravity_controller = gravity_controller
        if self._gravity_compensation_enabled and valid:
            linear_accel_kinematic = linear_accel_body - gravity_controller
        else:
            linear_accel_kinematic = linear_accel_body.copy()
        self._latest_imu_linear_accel_kinematic = linear_accel_kinematic
        self._latest_linear_body_dot = (
            self._accel_alpha * linear_accel_kinematic
            + (1.0 - self._accel_alpha) * self._latest_linear_body_dot
        )
        if valid:
            angle = np.zeros(6, dtype=np.float64)
            angle[AXIS_TO_NU_INDEX["roll_x"]] = roll_x
            angle[AXIS_TO_NU_INDEX["yaw_y"]] = yaw_y
            angle[AXIS_TO_NU_INDEX["pitch_z"]] = pitch_z
            self._latest_angle_body = angle - self._equilibrium_angles
        self._latest_sample_time_sec = now
        self._latest_imu_ros_stamp_sec = self._header_stamp_sec(msg, now)
        self._latest_host_receive_time_ns = int(now * 1e9)
        self._latest_imu_age_sec = 0.0
        self._latest_synchronization_valid = True

    def _motor_rpm_callback(self, msg: Float32MultiArray) -> None:
        self._latest_rpm = _as_vec(msg.data, THRUSTER_COUNT)
        self._latest_rpm_sec = self._now_sec()


def main(args: Sequence[str] | None = None) -> None:
    rclpy.init(args=args)
    node = HydrodynamicIdentifierNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
