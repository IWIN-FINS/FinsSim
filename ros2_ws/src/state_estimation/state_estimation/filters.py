from __future__ import annotations

import numpy as np

from .transforms import wrap_angle


class VectorLowPass:
    def __init__(self, alpha: float, size: int) -> None:
        self.alpha = float(max(0.0, min(1.0, alpha)))
        self.value = np.zeros(size, dtype=np.float64)
        self.initialized = False

    def reset(self) -> None:
        self.value[:] = 0.0
        self.initialized = False

    def update(self, sample) -> np.ndarray:
        sample_array = np.asarray(sample, dtype=np.float64)
        if not self.initialized:
            self.value = sample_array.astype(np.float64, copy=True)
            self.initialized = True
        else:
            self.value = self.alpha * sample_array + (1.0 - self.alpha) * self.value
        return self.value.astype(np.float64, copy=True)


class PositionVelocityEKF:
    def __init__(self, process_noise_position: float, process_noise_velocity: float) -> None:
        self.x = np.zeros(6, dtype=np.float64)
        self.p = np.diag([100.0, 100.0, 100.0, 10.0, 10.0, 10.0]).astype(np.float64)
        self.process_noise_position = float(max(process_noise_position, 1e-9))
        self.process_noise_velocity = float(max(process_noise_velocity, 1e-9))
        self.last_time: float | None = None

    def predict(self, stamp_sec: float) -> None:
        if self.last_time is None:
            self.last_time = float(stamp_sec)
            return
        dt = max(0.0, min(1.0, float(stamp_sec) - self.last_time))
        self.last_time = float(stamp_sec)
        if dt <= 0.0:
            return
        f = np.eye(6, dtype=np.float64)
        f[0, 3] = dt
        f[1, 4] = dt
        f[2, 5] = dt
        q = np.diag(
            [
                self.process_noise_position * dt,
                self.process_noise_position * dt,
                self.process_noise_position * dt,
                self.process_noise_velocity * dt,
                self.process_noise_velocity * dt,
                self.process_noise_velocity * dt,
            ]
        )
        self.x = f @ self.x
        self.p = f @ self.p @ f.T + q

    def update_indices(
        self,
        indices: list[int],
        measurement,
        covariance,
        *,
        gate_mahalanobis: float | None = None,
    ) -> tuple[bool, float]:
        idx = list(indices)
        z = np.asarray(measurement, dtype=np.float64).reshape(len(idx))
        r = np.asarray(covariance, dtype=np.float64).reshape(len(idx), len(idx))
        h = np.zeros((len(idx), 6), dtype=np.float64)
        for row, index in enumerate(idx):
            h[row, index] = 1.0
        residual = z - h @ self.x
        s = h @ self.p @ h.T + r
        try:
            s_inv = np.linalg.inv(s)
        except np.linalg.LinAlgError:
            s_inv = np.linalg.pinv(s)
        distance = float(residual.T @ s_inv @ residual)
        if gate_mahalanobis is not None and distance > float(gate_mahalanobis):
            return False, distance
        k = self.p @ h.T @ s_inv
        self.x = self.x + k @ residual
        identity = np.eye(6, dtype=np.float64)
        self.p = (identity - k @ h) @ self.p @ (identity - k @ h).T + k @ r @ k.T
        return True, distance

    def cap_covariance(self, indices: list[int], max_variance: float) -> None:
        limit = float(max(max_variance, 1e-12))
        for index in indices:
            idx = int(index)
            if self.p[idx, idx] <= limit:
                continue
            scale = (limit / max(float(self.p[idx, idx]), 1e-12)) ** 0.5
            self.p[idx, :] *= scale
            self.p[:, idx] *= scale
            self.p[idx, idx] = limit

    def decay_velocity(self, indices: list[int], decay: float, *, zero_epsilon: float = 1e-4) -> None:
        factor = float(min(max(decay, 0.0), 1.0))
        for index in indices:
            idx = int(index)
            self.x[idx] *= factor
            if abs(float(self.x[idx])) < float(zero_epsilon):
                self.x[idx] = 0.0


class YawEKF:
    def __init__(self, process_noise: float = 0.02) -> None:
        self.yaw = 0.0
        self.p = 10.0
        self.process_noise = float(max(process_noise, 1e-9))
        self.last_time: float | None = None
        self.initialized = False

    def predict(self, stamp_sec: float, yaw_rate: float = 0.0) -> None:
        if self.last_time is None:
            self.last_time = float(stamp_sec)
            return
        dt = max(0.0, min(1.0, float(stamp_sec) - self.last_time))
        self.last_time = float(stamp_sec)
        if dt <= 0.0:
            return
        self.yaw = wrap_angle(self.yaw + float(yaw_rate) * dt)
        self.p += self.process_noise * dt

    def update(self, yaw: float, covariance: float, *, gate_mahalanobis: float | None = None) -> tuple[bool, float]:
        measurement = wrap_angle(yaw)
        if not self.initialized:
            self.yaw = measurement
            self.p = float(max(covariance, 1e-9))
            self.initialized = True
            return True, 0.0
        r = float(max(covariance, 1e-9))
        residual = wrap_angle(measurement - self.yaw)
        s = self.p + r
        distance = float(residual * residual / s)
        if gate_mahalanobis is not None and distance > float(gate_mahalanobis):
            return False, distance
        k = self.p / s
        self.yaw = wrap_angle(self.yaw + k * residual)
        self.p = (1.0 - k) * self.p
        return True, distance
