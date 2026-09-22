"""Pure state and detector primitives for T1 manual-disturbance trials.

The hardware experiment deliberately detects *observable motion* rather than
claiming to measure an uninstrumented external force.  Keeping the detector in
this dependency-free module makes its thresholds testable without ROS2 and
allows the online runner and offline analyser to use the same error convention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any


DISTURBANCE_AXES = ("x", "z")


def wrap_degrees(value: float) -> float:
    """Return the shortest signed angular difference in ``[-180, 180]``."""

    return math.degrees(math.atan2(math.sin(math.radians(value)), math.cos(math.radians(value))))


@dataclass(frozen=True)
class T1State:
    """A controller-world state relative to one frozen T1 target."""

    timestamp_sec: float
    x_m: float
    depth_m: float
    z_m: float
    yaw_deg: float
    x_error_m: float
    depth_error_m: float
    z_error_m: float
    yaw_error_deg: float
    x_velocity_mps: float | None
    depth_velocity_mps: float | None
    z_velocity_mps: float | None
    yaw_rate_deg_s: float | None
    health_ok: bool

    def error_for_axis(self, axis: str) -> float:
        values = {
            "x": self.x_error_m,
            "depth": self.depth_error_m,
            "z": self.z_error_m,
            "yaw": self.yaw_error_deg,
        }
        return values[axis]

    def velocity_for_axis(self, axis: str) -> float | None:
        values = {
            "x": self.x_velocity_mps,
            "depth": self.depth_velocity_mps,
            "z": self.z_velocity_mps,
            "yaw": self.yaw_rate_deg_s,
        }
        return values[axis]


def make_state(
    *,
    timestamp_sec: float,
    pose: tuple[float, float, float, float],
    target: tuple[float, float, float, float],
    velocity: tuple[float | None, float | None, float | None, float | None],
    health_ok: bool,
) -> T1State:
    """Build a state using the repository's controller-world convention."""

    x_m, depth_m, z_m, yaw_deg = pose
    target_x, target_depth, target_z, target_yaw = target
    return T1State(
        timestamp_sec=float(timestamp_sec),
        x_m=float(x_m),
        depth_m=float(depth_m),
        z_m=float(z_m),
        yaw_deg=float(yaw_deg),
        x_error_m=float(x_m) - float(target_x),
        depth_error_m=float(depth_m) - float(target_depth),
        z_error_m=float(z_m) - float(target_z),
        yaw_error_deg=wrap_degrees(float(yaw_deg) - float(target_yaw)),
        x_velocity_mps=velocity[0],
        depth_velocity_mps=velocity[1],
        z_velocity_mps=velocity[2],
        yaw_rate_deg_s=velocity[3],
        health_ok=bool(health_ok),
    )


def _positive_float(mapping: dict[str, Any], key: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"detection.{key} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"detection.{key} must be a positive finite number")
    return value


class DisturbanceDetector:
    """Debounced horizontal-position detector for one manual perturbation.

    E8 deliberately treats an observed horizontal displacement from the centre
    target as the manual-perturbation event.  It does not infer external force,
    use finite-difference velocity, or use depth/yaw departures.  Persistence
    rejects a single pose sample but does not make this a calibrated-force test.
    """

    def __init__(self, config: dict[str, Any]) -> None:
        self.horizontal_error_threshold = {
            "x": _positive_float(config, "x_error_m"),
            "z": _positive_float(config, "z_error_m"),
        }
        try:
            self.persistence_sec = float(config.get("persistence_sec", 0.30))
        except (TypeError, ValueError) as exc:
            raise ValueError("detection.persistence_sec must be a non-negative number") from exc
        if not math.isfinite(self.persistence_sec) or self.persistence_sec < 0.0:
            raise ValueError("detection.persistence_sec must be a non-negative finite number")
        self._candidate_axis: str | None = None
        self._candidate_started_sec: float | None = None

    def reset(self) -> None:
        self._candidate_axis = None
        self._candidate_started_sec = None

    def observe(self, state: T1State) -> dict[str, Any] | None:
        """Return immutable detection metadata once a candidate is confirmed."""

        if not state.health_ok:
            self.reset()
            return None
        candidates: list[tuple[float, str]] = []
        for axis in DISTURBANCE_AXES:
            error = state.error_for_axis(axis)
            error_ratio = abs(error) / self.horizontal_error_threshold[axis]
            if error_ratio >= 1.0:
                candidates.append((error_ratio, axis))
        if not candidates:
            self.reset()
            return None

        _, axis = max(candidates)
        if axis != self._candidate_axis:
            self._candidate_axis = axis
            self._candidate_started_sec = state.timestamp_sec
            return None

        assert self._candidate_started_sec is not None
        elapsed = state.timestamp_sec - self._candidate_started_sec
        if elapsed < self.persistence_sec:
            return None

        payload = {
            "detection_source": "controller_world_horizontal_position_error_debounced",
            "detected_axis": axis,
            "candidate_started_sec_monotonic": self._candidate_started_sec,
            "confirmed_sec_monotonic": state.timestamp_sec,
            "candidate_persistence_sec": elapsed,
            "pose_controller_world": [state.x_m, state.depth_m, state.z_m, state.yaw_deg],
            "error": {
                "x_m": state.x_error_m,
                "depth_m": state.depth_error_m,
                "z_m": state.z_error_m,
                "yaw_deg": state.yaw_error_deg,
            },
            "thresholds": {
                "horizontal_error_m": self.horizontal_error_threshold,
                "persistence_sec": self.persistence_sec,
            },
        }
        self.reset()
        return payload


def state_within_thresholds(
    state: T1State,
    thresholds: dict[str, Any],
) -> bool:
    """Return whether all controlled pose-error axes are inside their bounds."""

    try:
        position_ok = (
            abs(state.x_error_m) <= float(thresholds["x_m"])
            and abs(state.depth_error_m) <= float(thresholds["depth_m"])
            and abs(state.z_error_m) <= float(thresholds["z_m"])
            and abs(state.yaw_error_deg) <= float(thresholds["yaw_deg"])
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("thresholds must contain x_m, depth_m, z_m, and yaw_deg") from exc
    return position_ok and state.health_ok
