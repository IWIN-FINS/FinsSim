from __future__ import annotations

import math

import numpy as np


def normalize_quat_xyzw(quat) -> np.ndarray:
    q = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm <= 1e-12:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    return q / norm


def quat_xyzw_to_rotation_matrix(quat) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def inverse_rotate_vector(quat_xyzw, vector_world) -> np.ndarray:
    rotation = quat_xyzw_to_rotation_matrix(quat_xyzw)
    return rotation.T @ np.asarray(vector_world, dtype=np.float64).reshape(3)


def wrap_angle(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def quat_multiply_xyzw(lhs, rhs) -> np.ndarray:
    ax, ay, az, aw = normalize_quat_xyzw(lhs)
    bx, by, bz, bw = normalize_quat_xyzw(rhs)
    return normalize_quat_xyzw(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def quat_conjugate_xyzw(quat) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat)
    return np.array([-x, -y, -z, w], dtype=np.float64)


def quat_to_rpy_xyzw(quat) -> tuple[float, float, float]:
    x, y, z, w = normalize_quat_xyzw(quat)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1.0:
        pitch = math.copysign(math.pi / 2.0, sinp)
    else:
        pitch = math.asin(sinp)

    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def rpy_to_quat_xyzw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    return normalize_quat_xyzw(
        [
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        ]
    )


def pose_matrix(translation_xyz, rotation_xyzw) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = quat_xyzw_to_rotation_matrix(rotation_xyzw)
    matrix[:3, 3] = np.asarray(translation_xyz, dtype=np.float64).reshape(3)
    return matrix


def matrix_to_quat_xyzw(rotation_matrix) -> np.ndarray:
    rotation = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(rotation))
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (rotation[2, 1] - rotation[1, 2]) / scale
        y = (rotation[0, 2] - rotation[2, 0]) / scale
        z = (rotation[1, 0] - rotation[0, 1]) / scale
    else:
        diagonal = np.diag(rotation)
        if diagonal[0] > diagonal[1] and diagonal[0] > diagonal[2]:
            scale = math.sqrt(1.0 + rotation[0, 0] - rotation[1, 1] - rotation[2, 2]) * 2.0
            w = (rotation[2, 1] - rotation[1, 2]) / scale
            x = 0.25 * scale
            y = (rotation[0, 1] + rotation[1, 0]) / scale
            z = (rotation[0, 2] + rotation[2, 0]) / scale
        elif diagonal[1] > diagonal[2]:
            scale = math.sqrt(1.0 + rotation[1, 1] - rotation[0, 0] - rotation[2, 2]) * 2.0
            w = (rotation[0, 2] - rotation[2, 0]) / scale
            x = (rotation[0, 1] + rotation[1, 0]) / scale
            y = 0.25 * scale
            z = (rotation[1, 2] + rotation[2, 1]) / scale
        else:
            scale = math.sqrt(1.0 + rotation[2, 2] - rotation[0, 0] - rotation[1, 1]) * 2.0
            w = (rotation[1, 0] - rotation[0, 1]) / scale
            x = (rotation[0, 2] + rotation[2, 0]) / scale
            y = (rotation[1, 2] + rotation[2, 1]) / scale
            z = 0.25 * scale
    return normalize_quat_xyzw([x, y, z, w])


def matrix_to_pose(matrix) -> tuple[np.ndarray, np.ndarray]:
    transform = np.asarray(matrix, dtype=np.float64).reshape(4, 4)
    return transform[:3, 3].astype(np.float64, copy=True), matrix_to_quat_xyzw(transform[:3, :3])
