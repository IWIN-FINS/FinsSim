"""Deterministic, validated T2 reference trajectory generation.

All T2 geometry is declared in ``t2_hardware_experiment.yaml``.  This module
turns the declared straight, circle, vertex-slowed ellipse, or Gerono
figure-eight into a sampled ``controller_world`` reference with explicit
position and velocity.  T2 intentionally carries no yaw reference: its
current hardware protocol tracks translation only and disables all rotational
wrench components.  This module contains no ROS dependency so geometry and
safety checks are unit-testable.
"""

from __future__ import annotations

import csv
from bisect import bisect_left
from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class ReferencePoint:
    time_sec: float
    x_m: float
    y_m: float
    z_m: float
    vx_mps: float
    vy_mps: float
    vz_mps: float


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a YAML mapping")
    return value


def _float(value: Any, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def _vector(value: Any, size: int, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or len(value) != size:
        raise ValueError(f"{label} must contain {size} numeric values")
    return tuple(_float(item, f"{label}[{index}]") for index, item in enumerate(value))


def _waypoints(value: Any, label: str) -> list[tuple[float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{label} must contain at least two [x, z] waypoints")
    return [_vector(point, 2, f"{label}[{index}]") for index, point in enumerate(value)]


def _smoothstep(u: float) -> tuple[float, float]:
    """Return s(u) and ds/dt multiplier before division by duration.

    The cubic profile starts and ends at zero velocity, which avoids a step
    command at the physical trajectory boundary.
    """

    clipped = min(max(float(u), 0.0), 1.0)
    return 3.0 * clipped * clipped - 2.0 * clipped * clipped * clipped, 6.0 * clipped * (1.0 - clipped)


def _sample_times(duration_sec: float, sample_rate_hz: float) -> list[float]:
    if duration_sec <= 0.0 or sample_rate_hz <= 0.0:
        raise ValueError("duration_sec and sample_rate_hz must be positive")
    count = max(2, int(math.ceil(duration_sec * sample_rate_hz)) + 1)
    return [duration_sec * index / (count - 1) for index in range(count)]


def _ellipse_phase_lut(
    *,
    semi_major_x_m: float,
    semi_minor_z_m: float,
    initial_phase_rad: float,
    direction_sign: float,
    turns: float,
    vertex_speed_scale: float,
) -> tuple[list[float], list[float]]:
    """Build a monotone, arc-length-aware phase lookup for a slowed ellipse.

    With ``x=a cos(theta)`` and ``z=b sin(theta)``, a uniform phase rate would
    make the physical reference speed depend on the semi-axis lengths.  We
    instead parameterize phase by ``integral(q(theta) / g(theta)) dtheta``,
    where ``q`` is the tangent metric and ``g`` is a declared speed multiplier.
    This yields a physical speed proportional to ``g``: it is
    ``vertex_speed_scale`` at all four cardinal vertices and reaches its local
    maximum midway between adjacent vertices.  The outer cubic smoothstep is
    retained to start and finish a trial at zero command velocity.
    """

    total_phase = 2.0 * math.pi * turns
    # 4096 nodes per revolution makes the quadrature/inversion error far below
    # the 10 Hz trajectory transport resolution, while keeping dry-runs fast.
    count = max(1025, int(math.ceil(4096.0 * turns)) + 1)
    phase_offsets = [total_phase * index / (count - 1) for index in range(count)]
    cumulative = [0.0]

    def reciprocal_speed_shape(offset: float) -> float:
        phase = initial_phase_rad + direction_sign * offset
        sine = math.sin(phase)
        cosine = math.cos(phase)
        tangent_metric = math.hypot(semi_major_x_m * sine, semi_minor_z_m * cosine)
        speed_shape = vertex_speed_scale + (1.0 - vertex_speed_scale) * math.sin(2.0 * phase) ** 2
        return tangent_metric / speed_shape

    previous = reciprocal_speed_shape(phase_offsets[0])
    for left, right in zip(phase_offsets, phase_offsets[1:]):
        current = reciprocal_speed_shape(right)
        cumulative.append(cumulative[-1] + 0.5 * (previous + current) * (right - left))
        previous = current
    return phase_offsets, cumulative


def _ellipse_phase_at_progress(
    phase_offsets: list[float],
    cumulative: list[float],
    progress: float,
) -> float:
    """Invert the normalized arc-length lookup without a ROS/numpy dependency."""

    target = min(max(progress, 0.0), 1.0) * cumulative[-1]
    index = bisect_left(cumulative, target)
    if index <= 0:
        return phase_offsets[0]
    if index >= len(cumulative):
        return phase_offsets[-1]
    lower, upper = cumulative[index - 1], cumulative[index]
    fraction = 0.0 if upper <= lower else (target - lower) / (upper - lower)
    return phase_offsets[index - 1] + fraction * (phase_offsets[index] - phase_offsets[index - 1])


def _validate_declared_peak_speed(
    spec: dict[str, Any],
    *,
    label: str,
    raw: list[tuple[float, float, float, float, float, float, float]],
) -> None:
    """Fail closed when a YAML speed contract no longer matches its timing."""

    target_peak = spec.get("target_peak_speed_mps")
    if target_peak is None:
        return
    target_peak = _float(target_peak, f"{label}.target_peak_speed_mps")
    tolerance = _float(
        spec.get("target_peak_speed_tolerance_mps", 0.002),
        f"{label}.target_peak_speed_tolerance_mps",
    )
    if target_peak <= 0.0 or tolerance < 0.0:
        raise ValueError(f"{label} target peak speed must be positive and its tolerance non-negative")
    achieved_peak = max(math.hypot(point[4], point[6]) for point in raw)
    if abs(achieved_peak - target_peak) > tolerance:
        raise ValueError(
            f"{label}.duration_sec does not satisfy {label}.target_peak_speed_mps: "
            f"achieved {achieved_peak:.6f} m/s, target {target_peak:.6f} +/- {tolerance:.6f} m/s"
        )


def generate_trajectory(spec: dict[str, Any], *, fixed_depth_m: float, sample_rate_hz: float) -> list[ReferencePoint]:
    """Generate a trajectory declared by one catalog entry in the T2 YAML."""

    generator = str(spec.get("generator", "")).strip().lower()
    duration = _float(spec.get("duration_sec"), "trajectory.duration_sec")
    times = _sample_times(duration, _float(sample_rate_hz, "trajectory.sample_rate_hz"))
    raw: list[tuple[float, float, float, float, float, float, float]] = []

    if generator == "line":
        start = _vector(spec.get("start_xz_m"), 2, "line.start_xz_m")
        end = _vector(spec.get("end_xz_m"), 2, "line.end_xz_m")
        dx, dz = end[0] - start[0], end[1] - start[1]
        for time_sec in times:
            progress, ds_du = _smoothstep(time_sec / duration)
            derivative = ds_du / duration
            raw.append((time_sec, start[0] + dx * progress, fixed_depth_m, start[1] + dz * progress, dx * derivative, 0.0, dz * derivative))
    elif generator == "polyline_xz":
        waypoints = _waypoints(spec.get("waypoints_xz_m"), "polyline.waypoints_xz_m")
        segment_durations = _vector(
            spec.get("segment_durations_sec"),
            len(waypoints) - 1,
            "polyline.segment_durations_sec",
        )
        if any(value <= 0.0 for value in segment_durations):
            raise ValueError("polyline.segment_durations_sec entries must be positive")
        declared_duration = sum(segment_durations)
        if not math.isclose(duration, declared_duration, rel_tol=0.0, abs_tol=1e-6):
            raise ValueError(
                "polyline.duration_sec must equal the sum of "
                "polyline.segment_durations_sec"
            )
        # Include exact corners in addition to the uniform output grid.  Each
        # segment has its own smooth start/stop, so velocity is zero at a
        # waypoint and the requested direction change is never discontinuous.
        boundaries = [0.0]
        for segment_duration in segment_durations:
            boundaries.append(boundaries[-1] + segment_duration)
        times = sorted({*times, *boundaries})
        for time_sec in times:
            segment_index = min(
                len(segment_durations) - 1,
                next(
                    (
                        index
                        for index, boundary in enumerate(boundaries[1:])
                        if time_sec <= boundary + 1e-9
                    ),
                    len(segment_durations) - 1,
                ),
            )
            start_time = boundaries[segment_index]
            segment_duration = segment_durations[segment_index]
            progress, ds_du = _smoothstep((time_sec - start_time) / segment_duration)
            start, end = waypoints[segment_index], waypoints[segment_index + 1]
            dx, dz = end[0] - start[0], end[1] - start[1]
            derivative = ds_du / segment_duration
            raw.append((
                time_sec,
                start[0] + dx * progress,
                fixed_depth_m,
                start[1] + dz * progress,
                dx * derivative,
                0.0,
                dz * derivative,
            ))
    elif generator == "circle_xz":
        center = _vector(spec.get("center_xz_m"), 2, "circle.center_xz_m")
        radius = _float(spec.get("radius_m"), "circle.radius_m")
        if radius <= 0.0:
            raise ValueError("circle.radius_m must be positive")
        turns = _float(spec.get("turns", 1.0), "circle.turns")
        if turns <= 0.0:
            raise ValueError("circle.turns must be positive")
        initial_phase = math.radians(_float(spec.get("initial_phase_deg", 0.0), "circle.initial_phase_deg"))
        direction = str(spec.get("direction", "counterclockwise")).lower()
        sign = 1.0 if direction == "counterclockwise" else -1.0 if direction == "clockwise" else None
        if sign is None:
            raise ValueError("circle.direction must be clockwise or counterclockwise")
        for time_sec in times:
            progress, ds_du = _smoothstep(time_sec / duration)
            phase = initial_phase + sign * 2.0 * math.pi * turns * progress
            phase_rate = sign * 2.0 * math.pi * turns * ds_du / duration
            raw.append((time_sec, center[0] + radius * math.cos(phase), fixed_depth_m, center[1] + radius * math.sin(phase), -radius * math.sin(phase) * phase_rate, 0.0, radius * math.cos(phase) * phase_rate))
        _validate_declared_peak_speed(spec, label="circle", raw=raw)
    elif generator == "ellipse_xz_vertex_slow":
        center = _vector(spec.get("center_xz_m"), 2, "ellipse.center_xz_m")
        semi_major_x = _float(spec.get("semi_major_x_m"), "ellipse.semi_major_x_m")
        semi_minor_z = _float(spec.get("semi_minor_z_m"), "ellipse.semi_minor_z_m")
        if semi_major_x <= 0.0 or semi_minor_z <= 0.0:
            raise ValueError("ellipse semi-axis lengths must be positive")
        turns = _float(spec.get("turns", 1.0), "ellipse.turns")
        if turns <= 0.0:
            raise ValueError("ellipse.turns must be positive")
        initial_phase = math.radians(_float(spec.get("initial_phase_deg", 0.0), "ellipse.initial_phase_deg"))
        direction = str(spec.get("direction", "counterclockwise")).lower()
        sign = 1.0 if direction == "counterclockwise" else -1.0 if direction == "clockwise" else None
        if sign is None:
            raise ValueError("ellipse.direction must be clockwise or counterclockwise")
        vertex_speed_scale = _float(spec.get("vertex_speed_scale", 0.5), "ellipse.vertex_speed_scale")
        if not 0.0 < vertex_speed_scale <= 1.0:
            raise ValueError("ellipse.vertex_speed_scale must be in (0, 1]")

        phase_offsets, cumulative = _ellipse_phase_lut(
            semi_major_x_m=semi_major_x,
            semi_minor_z_m=semi_minor_z,
            initial_phase_rad=initial_phase,
            direction_sign=sign,
            turns=turns,
            vertex_speed_scale=vertex_speed_scale,
        )
        arc_time_scale = cumulative[-1] / duration
        for time_sec in times:
            progress, ds_du = _smoothstep(time_sec / duration)
            phase_offset = _ellipse_phase_at_progress(phase_offsets, cumulative, progress)
            phase = initial_phase + sign * phase_offset
            sine = math.sin(phase)
            cosine = math.cos(phase)
            tangent_metric = math.hypot(semi_major_x * sine, semi_minor_z * cosine)
            speed_shape = vertex_speed_scale + (1.0 - vertex_speed_scale) * math.sin(2.0 * phase) ** 2
            phase_rate = sign * arc_time_scale * ds_du * speed_shape / tangent_metric
            raw.append((
                time_sec,
                center[0] + semi_major_x * cosine,
                fixed_depth_m,
                center[1] + semi_minor_z * sine,
                -semi_major_x * sine * phase_rate,
                0.0,
                semi_minor_z * cosine * phase_rate,
            ))

        # The duration in YAML is deliberately derived from this physical
        # profile.  Fail closed if a later geometry/timing edit silently
        # violates the declared peak-speed contract.
        _validate_declared_peak_speed(spec, label="ellipse", raw=raw)
    elif generator == "gerono_xz":
        center = _vector(spec.get("center_xz_m"), 2, "lemniscate.center_xz_m")
        amplitude_x = _float(spec.get("amplitude_x_m"), "lemniscate.amplitude_x_m")
        amplitude_z = _float(spec.get("amplitude_z_m"), "lemniscate.amplitude_z_m")
        if amplitude_x <= 0.0 or amplitude_z <= 0.0:
            raise ValueError("lemniscate amplitudes must be positive")
        initial_phase = math.radians(_float(spec.get("initial_phase_deg", 0.0), "lemniscate.initial_phase_deg"))
        direction = str(spec.get("direction", "forward")).lower()
        sign = 1.0 if direction == "forward" else -1.0 if direction == "reverse" else None
        if sign is None:
            raise ValueError("lemniscate.direction must be forward or reverse")
        for time_sec in times:
            progress, ds_du = _smoothstep(time_sec / duration)
            phase = initial_phase + sign * 2.0 * math.pi * progress
            phase_rate = sign * 2.0 * math.pi * ds_du / duration
            raw.append((time_sec, center[0] + amplitude_x * math.sin(phase), fixed_depth_m, center[1] + amplitude_z * math.sin(phase) * math.cos(phase), amplitude_x * math.cos(phase) * phase_rate, 0.0, amplitude_z * math.cos(2.0 * phase) * phase_rate))
        _validate_declared_peak_speed(spec, label="gerono", raw=raw)
    else:
        raise ValueError(f"unsupported T2 trajectory generator: {generator!r}")
    return [ReferencePoint(*point) for point in raw]


def validate_reference(points: Iterable[ReferencePoint], workspace: dict[str, Any]) -> dict[str, float]:
    """Validate sampled geometry and return peak speed/acceleration diagnostics."""

    values = list(points)
    if len(values) < 2:
        raise ValueError("a T2 trajectory requires at least two sampled points")
    limits = _mapping(workspace, "surveyed_workspace")
    margin = _float(limits.get("clearance_margin_m"), "surveyed_workspace.clearance_margin_m")
    x_min, x_max = _vector(limits.get("x_m"), 2, "surveyed_workspace.x_m")
    y_min, y_max = _vector(limits.get("y_m"), 2, "surveyed_workspace.y_m")
    z_min, z_max = _vector(limits.get("z_m"), 2, "surveyed_workspace.z_m")
    if not (x_min + margin < x_max - margin and y_min + margin < y_max - margin and z_min + margin < z_max - margin):
        raise ValueError("workspace clearance margin leaves no valid volume")
    for index, point in enumerate(values):
        if not (x_min + margin <= point.x_m <= x_max - margin and y_min + margin <= point.y_m <= y_max - margin and z_min + margin <= point.z_m <= z_max - margin):
            raise ValueError(f"trajectory sample {index} is outside the surveyed clearance volume")
    speeds = [math.sqrt(point.vx_mps**2 + point.vy_mps**2 + point.vz_mps**2) for point in values]
    accelerations: list[float] = []
    for left, right in zip(values, values[1:]):
        dt = max(right.time_sec - left.time_sec, 1e-9)
        accelerations.append(math.sqrt((right.vx_mps - left.vx_mps) ** 2 + (right.vy_mps - left.vy_mps) ** 2 + (right.vz_mps - left.vz_mps) ** 2) / dt)
    max_speed = _float(limits.get("maximum_horizontal_speed_mps"), "surveyed_workspace.maximum_horizontal_speed_mps")
    max_acceleration = _float(limits.get("maximum_horizontal_acceleration_mps2"), "surveyed_workspace.maximum_horizontal_acceleration_mps2")
    if max(speeds) > max_speed + 1e-9:
        raise ValueError(f"trajectory peak speed {max(speeds):.4f} m/s exceeds declared limit {max_speed:.4f} m/s")
    if accelerations and max(accelerations) > max_acceleration + 1e-9:
        raise ValueError(f"trajectory peak acceleration {max(accelerations):.4f} m/s^2 exceeds declared limit {max_acceleration:.4f} m/s^2")
    return {
        "peak_speed_mps": max(speeds),
        "peak_acceleration_mps2": max(accelerations) if accelerations else 0.0,
    }


def write_reference_csv(path: Path, points: Iterable[ReferencePoint]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["time_sec", "x_m", "y_m", "z_m", "vx_mps", "vy_mps", "vz_mps"])
        writer.writeheader()
        for point in points:
            writer.writerow({
                "time_sec": f"{point.time_sec:.9f}", "x_m": f"{point.x_m:.9f}", "y_m": f"{point.y_m:.9f}", "z_m": f"{point.z_m:.9f}",
                "vx_mps": f"{point.vx_mps:.9f}", "vy_mps": f"{point.vy_mps:.9f}", "vz_mps": f"{point.vz_mps:.9f}",
            })
