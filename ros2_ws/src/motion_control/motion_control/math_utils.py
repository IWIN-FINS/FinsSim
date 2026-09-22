from __future__ import annotations

from typing import Sequence

import numpy as np


DEFAULT_BASIS_INDICES: tuple[int, int, int] = (0, 2, 1)
DEFAULT_BASIS_SIGNS: tuple[float, float, float] = (1.0, 1.0, 1.0)


def build_basis_matrix(
    indices: Sequence[int] = DEFAULT_BASIS_INDICES,
    signs: Sequence[float] = DEFAULT_BASIS_SIGNS,
) -> np.ndarray:
    """Build a 3x3 axis-remap matrix from ROS xyz to controller xyz."""
    rows = []
    resolved_indices = tuple(int(v) for v in indices)
    resolved_signs = tuple(float(v) for v in signs)
    if len(resolved_indices) != 3 or len(resolved_signs) != 3:
        raise ValueError("basis indices/signs must both have length 3")

    for axis_idx, sign in zip(resolved_indices, resolved_signs):
        if axis_idx < 0 or axis_idx > 2:
            raise ValueError(f"basis axis index must be in [0, 2], got {axis_idx}")
        row = np.zeros(3, dtype=np.float64)
        row[axis_idx] = sign
        rows.append(row)
    return np.stack(rows, axis=0)


def reorder_vector(vec_xyz: Sequence[float], basis_matrix: np.ndarray) -> np.ndarray:
    vector = np.asarray(vec_xyz, dtype=np.float64).reshape(3)
    return (basis_matrix @ vector).astype(np.float32)


def axial_basis_matrix(basis_matrix: np.ndarray) -> np.ndarray:
    """Return the basis transform for an axial vector such as angular velocity.

    A ROS-to-Unity axis map may include a reflection (the default x/y/z swap
    has determinant -1). Angular velocity is an axial vector, so under an
    improper basis transform it needs the determinant factor in addition to
    the ordinary vector transform.
    """
    basis = np.asarray(basis_matrix, dtype=np.float64).reshape(3, 3)
    determinant = float(np.linalg.det(basis))
    if not np.isclose(abs(determinant), 1.0, atol=1e-6):
        raise ValueError(
            "axial-vector basis transform requires an orthogonal basis with determinant +/-1, "
            f"got determinant={determinant}"
        )
    return (determinant * basis).astype(np.float64, copy=False)


def transform_axial_vector(vec_xyz: Sequence[float], basis_matrix: np.ndarray) -> np.ndarray:
    """Transform an axial vector, for example angular velocity or rotation rate."""
    vector = np.asarray(vec_xyz, dtype=np.float64).reshape(3)
    return (axial_basis_matrix(basis_matrix) @ vector).astype(np.float32)


def transform_axial_covariance(covariance: Sequence[float], basis_matrix: np.ndarray) -> list[float]:
    """Transform a 3x3 covariance belonging to an axial vector."""
    matrix = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
    transform = axial_basis_matrix(basis_matrix)
    return (transform @ matrix @ transform.T).reshape(9).astype(float).tolist()


def quat_normalize(quat_xyzw: Sequence[float]) -> np.ndarray:
    quat = np.asarray(quat_xyzw, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(quat)
    if norm <= 1e-8:
        return np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (quat / norm).astype(np.float32)


def quat_multiply(lhs_xyzw: Sequence[float], rhs_xyzw: Sequence[float]) -> np.ndarray:
    lx, ly, lz, lw = quat_normalize(lhs_xyzw).astype(np.float64)
    rx, ry, rz, rw = quat_normalize(rhs_xyzw).astype(np.float64)
    return quat_normalize(
        [
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
            lw * rw - lx * rx - ly * ry - lz * rz,
        ]
    )


def quat_to_matrix(quat_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = quat_normalize(quat_xyzw).astype(np.float64)
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


def matrix_to_quat(rotation_matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation_matrix, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))

    if trace > 0.0:
        scale = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * scale
        x = (matrix[2, 1] - matrix[1, 2]) / scale
        y = (matrix[0, 2] - matrix[2, 0]) / scale
        z = (matrix[1, 0] - matrix[0, 1]) / scale
    elif matrix[0, 0] > matrix[1, 1] and matrix[0, 0] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
        w = (matrix[2, 1] - matrix[1, 2]) / scale
        x = 0.25 * scale
        y = (matrix[0, 1] + matrix[1, 0]) / scale
        z = (matrix[0, 2] + matrix[2, 0]) / scale
    elif matrix[1, 1] > matrix[2, 2]:
        scale = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
        w = (matrix[0, 2] - matrix[2, 0]) / scale
        x = (matrix[0, 1] + matrix[1, 0]) / scale
        y = 0.25 * scale
        z = (matrix[1, 2] + matrix[2, 1]) / scale
    else:
        scale = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
        w = (matrix[1, 0] - matrix[0, 1]) / scale
        x = (matrix[0, 2] + matrix[2, 0]) / scale
        y = (matrix[1, 2] + matrix[2, 1]) / scale
        z = 0.25 * scale

    return quat_normalize([x, y, z, w])


def quat_angle_error_rad(lhs_xyzw: Sequence[float], rhs_xyzw: Sequence[float]) -> float:
    lhs = quat_normalize(lhs_xyzw).astype(np.float64)
    rhs = quat_normalize(rhs_xyzw).astype(np.float64)
    dot = float(np.clip(np.abs(np.dot(lhs, rhs)), 0.0, 1.0))
    return float(2.0 * np.arccos(dot))


def quat_from_controller_ypr(
    yaw_rad: float,
    pitch_rad: float,
    roll_rad: float,
) -> np.ndarray:
    """Build controller-frame quaternion from yaw(Y), pitch(Z), roll(X)."""
    cy = np.cos(float(yaw_rad))
    sy = np.sin(float(yaw_rad))
    cp = np.cos(float(pitch_rad))
    sp = np.sin(float(pitch_rad))
    cr = np.cos(float(roll_rad))
    sr = np.sin(float(roll_rad))

    rot_yaw = np.array(
        [
            [cy, 0.0, sy],
            [0.0, 1.0, 0.0],
            [-sy, 0.0, cy],
        ],
        dtype=np.float64,
    )
    rot_pitch = np.array(
        [
            [cp, -sp, 0.0],
            [sp, cp, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    rot_roll = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cr, -sr],
            [0.0, sr, cr],
        ],
        dtype=np.float64,
    )

    rotation = rot_yaw @ rot_pitch @ rot_roll
    return matrix_to_quat(rotation)


def transform_quat_basis(quat_xyzw: Sequence[float], basis_matrix: np.ndarray) -> np.ndarray:
    """Convert a quaternion from ROS xyz basis into controller xyz basis."""
    rotation_ros = quat_to_matrix(quat_xyzw)
    rotation_controller = basis_matrix @ rotation_ros @ basis_matrix.T
    return matrix_to_quat(rotation_controller)


def rotate_vector(quat_xyzw: Sequence[float], vec_xyz: Sequence[float]) -> np.ndarray:
    rotation = quat_to_matrix(quat_xyzw)
    vector = np.asarray(vec_xyz, dtype=np.float64).reshape(3)
    return (rotation @ vector).astype(np.float32)


def inverse_rotate_vector(quat_xyzw: Sequence[float], vec_xyz: Sequence[float]) -> np.ndarray:
    rotation = quat_to_matrix(quat_xyzw)
    vector = np.asarray(vec_xyz, dtype=np.float64).reshape(3)
    return (rotation.T @ vector).astype(np.float32)
