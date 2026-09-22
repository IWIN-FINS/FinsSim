from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _normalize_quaternion(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    quaternion = np.asarray(quaternion_xyzw, dtype=np.float64).reshape(-1)
    if quaternion.size != 4:
        raise ValueError("IMU mounting quaternion must contain exactly four xyzw values")
    if not np.all(np.isfinite(quaternion)):
        raise ValueError("IMU mounting quaternion must contain only finite values")
    norm = float(np.linalg.norm(quaternion))
    if norm <= 1e-9:
        raise ValueError("IMU mounting quaternion norm must be greater than zero")
    return quaternion / norm


def _quaternion_to_matrix(quaternion_xyzw: Sequence[float]) -> np.ndarray:
    x, y, z, w = _normalize_quaternion(quaternion_xyzw)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y)],
            [2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x)],
            [2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quaternion(rotation: np.ndarray) -> np.ndarray:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
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
    return _normalize_quaternion([x, y, z, w])


@dataclass(frozen=True)
class CorrectedImuSample:
    orientation_xyzw: tuple[float, float, float, float]
    angular_velocity_xyz: tuple[float, float, float]
    linear_acceleration_xyz: tuple[float, float, float]


class ImuMountingTransform:
    """Rotate MCU sensor-frame IMU data into the ROS base_link frame."""

    def __init__(self, sensor_to_base_quaternion_xyzw: Sequence[float]) -> None:
        quaternion = _normalize_quaternion(sensor_to_base_quaternion_xyzw)
        self._quaternion_xyzw = tuple(float(value) for value in quaternion)
        self._rotation_base_sensor = _quaternion_to_matrix(quaternion)

    @property
    def quaternion_xyzw(self) -> tuple[float, float, float, float]:
        return self._quaternion_xyzw

    def transform_vector(self, vector_sensor: Sequence[float]) -> tuple[float, float, float]:
        vector = np.asarray(vector_sensor, dtype=np.float64).reshape(-1)
        if vector.size != 3 or not np.all(np.isfinite(vector)):
            raise ValueError("IMU vector must contain exactly three finite values")
        transformed = self._rotation_base_sensor @ vector
        return tuple(float(value) for value in transformed)

    def transform_orientation(self, orientation_world_sensor_xyzw: Sequence[float]) -> tuple[float, float, float, float]:
        rotation_world_sensor = _quaternion_to_matrix(orientation_world_sensor_xyzw)
        rotation_world_base = rotation_world_sensor @ self._rotation_base_sensor.T
        quaternion = _matrix_to_quaternion(rotation_world_base)
        return tuple(float(value) for value in quaternion)

    def transform_covariance(self, covariance_sensor: Sequence[float]) -> list[float]:
        covariance = np.asarray(covariance_sensor, dtype=np.float64).reshape(-1)
        if covariance.size != 9 or not np.all(np.isfinite(covariance)):
            raise ValueError("IMU covariance must contain exactly nine finite values")
        matrix = covariance.reshape(3, 3)
        transformed = self._rotation_base_sensor @ matrix @ self._rotation_base_sensor.T
        return transformed.reshape(9).astype(float).tolist()

    def correct_sample(
        self,
        *,
        orientation_world_sensor_xyzw: Sequence[float],
        angular_velocity_sensor_xyz: Sequence[float],
        linear_acceleration_sensor_xyz: Sequence[float],
    ) -> CorrectedImuSample:
        return CorrectedImuSample(
            orientation_xyzw=self.transform_orientation(orientation_world_sensor_xyzw),
            angular_velocity_xyz=self.transform_vector(angular_velocity_sensor_xyz),
            linear_acceleration_xyz=self.transform_vector(linear_acceleration_sensor_xyz),
        )
