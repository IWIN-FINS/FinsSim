from __future__ import annotations

from dataclasses import dataclass
import math
from collections.abc import Sequence


THRUSTER_NAMES = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")
THRUSTER_COUNT = len(THRUSTER_NAMES)


@dataclass(frozen=True)
class SignedQuadraticThrusterCurve:
    c1_positive: tuple[float, ...]
    c1_negative: tuple[float, ...]
    rpm_min: tuple[float, ...]
    rpm_max: tuple[float, ...]
    force_deadband_n: tuple[float, ...]
    min_effective_rpm_positive: tuple[float, ...]
    min_effective_rpm_negative: tuple[float, ...]
    firmware_max_rpm: float

    def __post_init__(self) -> None:
        for name, values in (
            ("c1_positive", self.c1_positive),
            ("c1_negative", self.c1_negative),
            ("rpm_min", self.rpm_min),
            ("rpm_max", self.rpm_max),
            ("force_deadband_n", self.force_deadband_n),
            ("min_effective_rpm_positive", self.min_effective_rpm_positive),
            ("min_effective_rpm_negative", self.min_effective_rpm_negative),
        ):
            if len(values) != THRUSTER_COUNT:
                raise ValueError(f"{name} must contain {THRUSTER_COUNT} values")
            if any(not math.isfinite(float(v)) for v in values):
                raise ValueError(f"{name} must contain finite values")
        if not math.isfinite(self.firmware_max_rpm) or self.firmware_max_rpm <= 0.0:
            raise ValueError("firmware_max_rpm must be positive")
        for index, (c1_pos, c1_neg, rpm_min, rpm_max) in enumerate(
            zip(self.c1_positive, self.c1_negative, self.rpm_min, self.rpm_max)
        ):
            if abs(c1_pos) <= 0.0:
                raise ValueError(f"c1_positive[{index}] for {THRUSTER_NAMES[index]} must be non-zero")
            if abs(c1_neg) <= 0.0:
                raise ValueError(f"c1_negative[{index}] for {THRUSTER_NAMES[index]} must be non-zero")
            if rpm_min > 0.0:
                raise ValueError(f"rpm_min[{index}] for {THRUSTER_NAMES[index]} must be <= 0")
            if rpm_max < 0.0:
                raise ValueError(f"rpm_max[{index}] for {THRUSTER_NAMES[index]} must be >= 0")
            if self.force_deadband_n[index] < 0.0:
                raise ValueError(f"force_deadband_n[{index}] for {THRUSTER_NAMES[index]} must be >= 0")
            if self.min_effective_rpm_positive[index] < 0.0:
                raise ValueError(
                    f"min_effective_rpm_positive[{index}] for {THRUSTER_NAMES[index]} must be >= 0"
                )
            if self.min_effective_rpm_negative[index] < 0.0:
                raise ValueError(
                    f"min_effective_rpm_negative[{index}] for {THRUSTER_NAMES[index]} must be >= 0"
                )

    def force_to_rpm(self, force_n: float, index: int) -> float:
        force = float(force_n)
        if not math.isfinite(force):
            raise ValueError(f"force at index {index} must be finite")
        if index < 0 or index >= THRUSTER_COUNT:
            raise ValueError(f"thruster index out of range: {index}")
        if abs(force) <= float(self.force_deadband_n[index]):
            return 0.0
        c1 = abs(self.c1_positive[index] if force > 0.0 else self.c1_negative[index])
        omega_rad_s = math.copysign(math.sqrt(abs(force) / c1), force)
        rpm = omega_rad_s * 60.0 / (2.0 * math.pi)
        if force > 0.0:
            rpm = max(rpm, float(self.min_effective_rpm_positive[index]))
        else:
            rpm = min(rpm, -float(self.min_effective_rpm_negative[index]))
        return max(float(self.rpm_min[index]), min(float(self.rpm_max[index]), rpm))

    def forces_to_rpm(self, forces_n: Sequence[float]) -> list[float]:
        if len(forces_n) != THRUSTER_COUNT:
            raise ValueError(f"expected {THRUSTER_COUNT} force values, got {len(forces_n)}")
        return [self.force_to_rpm(float(force), index) for index, force in enumerate(forces_n)]

    def rpm_to_normalized(self, rpm: float) -> float:
        value = float(rpm) / self.firmware_max_rpm
        if not math.isfinite(value):
            raise ValueError(f"rpm must be finite, got {rpm!r}")
        return max(-1.0, min(1.0, value))

    def forces_to_normalized_rpm(self, forces_n: Sequence[float]) -> tuple[list[float], list[float]]:
        target_rpm = self.forces_to_rpm(forces_n)
        normalized = [self.rpm_to_normalized(rpm) for rpm in target_rpm]
        return normalized, target_rpm
