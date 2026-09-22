from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from hardware_bridge.direct_thruster_sender import CANONICAL_NAMES
from hardware_bridge.thruster_test_sender import _zeros

from .force_sensor_modbus import ForceReading, ModbusForceSensor
from .thruster_curve_sweep import (
    DEFAULT_THRUSTER_TOPIC,
    SweepConfig,
    command_at_elapsed,
    total_duration,
    validate_sweep_config,
    values_for_sample,
)


CONFIG_SECTION = "thruster_curve_measure"


@dataclass(frozen=True)
class StepConfig:
    index: int
    amplitude: float
    step: float
    commands: tuple[float, ...]
    pre_zero_sec: float
    settle_sec: float
    hold_sec: float
    sample_sec: float
    rest_sec: float
    repeat_rest_sec: float
    post_zero_sec: float
    repeat_count: int


@dataclass(frozen=True)
class StepSample:
    phase: str
    command: float
    point_index: int | None = None
    repeat_index: int | None = None
    sample_window: bool = False
    done: bool = False


@dataclass(frozen=True)
class SummaryPoint:
    command: float
    force_mean_n: float
    force_median_n: float
    force_std_n: float
    sample_count: int
    repeat_index: int
    point_index: int
    rpm_mean: float | None = None
    rpm_std: float | None = None
    omega_mean_rad_s: float | None = None


@dataclass(frozen=True)
class SignedQuadraticFit:
    side: str
    quadratic_coeff: float
    linear_coeff: float
    rmse_n: float
    sample_count: int
    command_min: float
    command_max: float


@dataclass(frozen=True)
class OmegaQuadraticFit:
    side: str
    c1: float
    rmse_n: float
    sample_count: int
    omega_min_rad_s: float
    omega_max_rad_s: float
    rpm_min: float
    rpm_max: float


class LatestRpm:
    def __init__(self, *, count: int, clock) -> None:
        self.values: list[float] | None = None
        self.stamp_sec: float | None = None
        self._count = count
        self._clock = clock

    def callback(self, msg) -> None:
        data = [float(value) if math.isfinite(float(value)) else 0.0 for value in msg.data]
        if len(data) < self._count:
            data.extend([0.0] * (self._count - len(data)))
        self.values = data[: self._count]
        self.stamp_sec = self._now_sec()

    def latest(self, *, max_age_sec: float) -> tuple[list[float] | None, float | None]:
        if self.values is None or self.stamp_sec is None:
            return None, None
        age = self._now_sec() - self.stamp_sec
        if max_age_sec > 0.0 and age > max_age_sec:
            return None, age
        return self.values, age

    def _now_sec(self) -> float:
        return self._clock.now().nanoseconds * 1e-9


def rpm_to_omega_rad_s(rpm: float) -> float:
    return float(rpm) * (2.0 * math.pi / 60.0)


def make_default_step_commands(amplitude: float, step: float) -> tuple[float, ...]:
    amplitude = abs(float(amplitude))
    step = abs(float(step))
    if not math.isfinite(amplitude):
        raise ValueError("--amplitude must be finite")
    if not math.isfinite(step):
        raise ValueError("--step must be finite")
    if amplitude <= 0.0 or amplitude > 1.0:
        raise ValueError("--amplitude must be in (0, 1]")
    if step <= 0.0 or step > amplitude * 2.0:
        raise ValueError("--step must be in (0, 2 * amplitude]")

    values: list[float] = []
    current = -amplitude
    epsilon = step * 1e-6
    while current <= amplitude + epsilon:
        values.append(max(-amplitude, min(amplitude, current)))
        current += step
    if values[-1] < amplitude - epsilon:
        values.append(amplitude)

    cleaned: list[float] = []
    for value in values:
        rounded = 0.0 if abs(value) < 1e-9 else round(value, 6)
        if not cleaned or abs(cleaned[-1] - rounded) > 1e-9:
            cleaned.append(rounded)
    return tuple(cleaned)


def parse_step_commands(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("--step-commands must contain at least one value")
    for value in values:
        if not math.isfinite(value):
            raise ValueError("--step-commands values must be finite")
        if value < -1.0 or value > 1.0:
            raise ValueError("--step-commands values must be in [-1, 1]")
    return values


def validate_step_config(config: StepConfig) -> None:
    if config.index < 0 or config.index >= len(CANONICAL_NAMES):
        raise ValueError(f"--index must be in [0, {len(CANONICAL_NAMES) - 1}], got {config.index}")
    if not math.isfinite(config.amplitude) or config.amplitude <= 0.0 or config.amplitude > 1.0:
        raise ValueError("--amplitude must be in (0, 1]")
    if not math.isfinite(config.step) or config.step <= 0.0:
        raise ValueError("--step must be > 0")
    if not math.isfinite(config.pre_zero_sec) or config.pre_zero_sec < 0.0:
        raise ValueError("--pre-zero must be >= 0")
    if not math.isfinite(config.settle_sec) or config.settle_sec < 0.0:
        raise ValueError("--settle must be >= 0")
    if not math.isfinite(config.hold_sec) or config.hold_sec <= 0.0:
        raise ValueError("--step-hold must be > 0")
    if not math.isfinite(config.sample_sec) or config.sample_sec <= 0.0 or config.sample_sec > config.hold_sec:
        raise ValueError("--step-sample must be in (0, step-hold]")
    if not math.isfinite(config.rest_sec) or config.rest_sec < 0.0:
        raise ValueError("--step-rest must be >= 0")
    if not math.isfinite(config.repeat_rest_sec) or config.repeat_rest_sec < 0.0:
        raise ValueError("--repeat-rest must be >= 0")
    if not math.isfinite(config.post_zero_sec) or config.post_zero_sec < 0.0:
        raise ValueError("--post-zero must be >= 0")
    if config.repeat_count < 1:
        raise ValueError("--repeat-count must be >= 1")
    for command in config.commands:
        if not math.isfinite(command):
            raise ValueError("step commands must be finite")
        if command < -1.0 or command > 1.0:
            raise ValueError("step commands must be in [-1, 1]")


def step_total_duration(config: StepConfig) -> float:
    validate_step_config(config)
    per_point = config.hold_sec + config.rest_sec
    repeated_points = config.repeat_count * len(config.commands) * per_point
    repeated_rests = max(0, config.repeat_count - 1) * config.repeat_rest_sec
    return (
        config.pre_zero_sec
        + config.settle_sec
        + repeated_points
        + repeated_rests
        + config.post_zero_sec
    )


def step_sample_at_elapsed(elapsed_sec: float, config: StepConfig) -> StepSample:
    validate_step_config(config)
    t = max(0.0, float(elapsed_sec))

    if t < config.pre_zero_sec:
        return StepSample("pre_zero", 0.0)
    t -= config.pre_zero_sec

    if t < config.settle_sec:
        return StepSample("settle_zero", 0.0)
    t -= config.settle_sec

    per_point = config.hold_sec + config.rest_sec
    repeat_step_region = len(config.commands) * per_point
    for repeat_index in range(config.repeat_count):
        if t < repeat_step_region:
            point_index = int(t // per_point)
            local_t = t - point_index * per_point
            command = config.commands[point_index]
            if local_t < config.hold_sec:
                sample_start = max(0.0, config.hold_sec - config.sample_sec)
                in_sample_window = local_t >= sample_start
                phase = "step_sample" if in_sample_window else "step_settle"
                return StepSample(
                    phase,
                    command,
                    point_index=point_index,
                    repeat_index=repeat_index,
                    sample_window=in_sample_window,
                )
            return StepSample("step_rest_zero", 0.0, point_index=point_index, repeat_index=repeat_index)
        t -= repeat_step_region

        if repeat_index < config.repeat_count - 1:
            if t < config.repeat_rest_sec:
                return StepSample("repeat_rest_zero", 0.0, repeat_index=repeat_index)
            t -= config.repeat_rest_sec

    if t < config.post_zero_sec:
        return StepSample("post_zero", 0.0)

    return StepSample("done", 0.0, done=True)


def values_for_index_command(index: int, command: float) -> list[float]:
    values = _zeros()
    values[index] = command
    return values


def _write_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "wall_time_unix",
            "monotonic_time",
            "elapsed_sec",
            "phase",
            "target_index",
            "target_name",
            "target_command",
            *CANONICAL_NAMES,
            "sensor_ok",
            "sensor_error",
            "raw_value",
            "signed_value",
            "force_n",
            "zero_offset_n",
            "force_zeroed_n",
            "target_rpm",
            "target_omega_rad_s",
            "rpm_age_sec",
            *(f"rpm_{name}" for name in CANONICAL_NAMES),
            "sample_window",
            "point_index",
            "repeat_index",
        ]
    )


def _write_row(
    writer: csv.writer,
    *,
    wall_time: float,
    monotonic_time: float,
    elapsed_sec: float,
    phase: str,
    index: int,
    values: Sequence[float],
    reading: ForceReading,
    zero_offset_n: float,
    rpm_values: Sequence[float] | None = None,
    rpm_age_sec: float | None = None,
    sample_window: bool = False,
    point_index: int | None = None,
    repeat_index: int | None = None,
) -> None:
    force = reading.force_n
    force_zeroed = force - zero_offset_n if force is not None else None
    target_rpm = None if rpm_values is None else float(rpm_values[index])
    target_omega = None if target_rpm is None else rpm_to_omega_rad_s(target_rpm)
    writer.writerow(
        [
            f"{wall_time:.6f}",
            f"{monotonic_time:.6f}",
            f"{elapsed_sec:.6f}",
            phase,
            index,
            CANONICAL_NAMES[index],
            f"{values[index]:.6f}",
            *(f"{value:.6f}" for value in values),
            1 if reading.ok else 0,
            reading.error,
            "" if reading.raw_value is None else reading.raw_value,
            "" if reading.signed_value is None else reading.signed_value,
            "" if force is None else f"{force:.6f}",
            f"{zero_offset_n:.6f}",
            "" if force_zeroed is None else f"{force_zeroed:.6f}",
            "" if target_rpm is None else f"{target_rpm:.6f}",
            "" if target_omega is None else f"{target_omega:.6f}",
            "" if rpm_age_sec is None else f"{rpm_age_sec:.6f}",
            *(("" if rpm_values is None else f"{float(value):.6f}") for value in (rpm_values or [None] * len(CANONICAL_NAMES))),
            1 if sample_window else 0,
            "" if point_index is None else point_index,
            "" if repeat_index is None else repeat_index,
        ]
    )


def _write_summary_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "target_index",
            "target_name",
            "repeat_index",
            "point_index",
            "target_command",
            "sample_count",
            "force_mean_n",
            "force_median_n",
            "force_std_n",
            "force_min_n",
            "force_max_n",
            "rpm_mean",
            "rpm_median",
            "rpm_std",
            "rpm_min",
            "rpm_max",
            "omega_mean_rad_s",
        ]
    )


def _write_summary_rows(
    writer: csv.writer,
    *,
    index: int,
    samples: dict[tuple[int, int], list[float]],
    rpm_samples: dict[tuple[int, int], list[float]] | None = None,
    commands: Sequence[float],
) -> None:
    for repeat_index, point_index in sorted(samples):
        forces = samples[(repeat_index, point_index)]
        if not forces:
            continue
        command = commands[point_index]
        std = statistics.stdev(forces) if len(forces) > 1 else 0.0
        rpms = [] if rpm_samples is None else rpm_samples.get((repeat_index, point_index), [])
        rpm_std = statistics.stdev(rpms) if len(rpms) > 1 else 0.0
        rpm_mean = statistics.mean(rpms) if rpms else None
        writer.writerow(
            [
                index,
                CANONICAL_NAMES[index],
                repeat_index,
                point_index,
                f"{command:.6f}",
                len(forces),
                f"{statistics.mean(forces):.6f}",
                f"{statistics.median(forces):.6f}",
                f"{std:.6f}",
                f"{min(forces):.6f}",
                f"{max(forces):.6f}",
                "" if rpm_mean is None else f"{rpm_mean:.6f}",
                "" if not rpms else f"{statistics.median(rpms):.6f}",
                "" if not rpms else f"{rpm_std:.6f}",
                "" if not rpms else f"{min(rpms):.6f}",
                "" if not rpms else f"{max(rpms):.6f}",
                "" if rpm_mean is None else f"{rpm_to_omega_rad_s(rpm_mean):.6f}",
            ]
        )


def signed_quadratic_force(command: float, fit: SignedQuadraticFit) -> float:
    return fit.quadratic_coeff * command * abs(command) + fit.linear_coeff * command


def omega_quadratic_force(omega_rad_s: float, fit: OmegaQuadraticFit) -> float:
    return fit.c1 * omega_rad_s * abs(omega_rad_s)


def fit_signed_quadratic(points: Sequence[SummaryPoint], *, side: str, min_abs_command: float = 0.0) -> SignedQuadraticFit:
    if side not in {"positive", "negative"}:
        raise ValueError("side must be 'positive' or 'negative'")
    min_abs_command = abs(float(min_abs_command))
    if side == "positive":
        selected = [point for point in points if point.command > min_abs_command]
    else:
        selected = [point for point in points if point.command < -min_abs_command]
    if len(selected) < 2:
        raise ValueError(f"not enough {side} points for signed quadratic fit")

    s11 = s12 = s22 = y1 = y2 = 0.0
    for point in selected:
        x1 = point.command * abs(point.command)
        x2 = point.command
        y = point.force_mean_n
        s11 += x1 * x1
        s12 += x1 * x2
        s22 += x2 * x2
        y1 += x1 * y
        y2 += x2 * y

    determinant = s11 * s22 - s12 * s12
    if abs(determinant) < 1e-12:
        raise ValueError(f"{side} fit is ill-conditioned")

    quadratic = (y1 * s22 - y2 * s12) / determinant
    linear = (s11 * y2 - s12 * y1) / determinant
    fit = SignedQuadraticFit(
        side=side,
        quadratic_coeff=quadratic,
        linear_coeff=linear,
        rmse_n=0.0,
        sample_count=len(selected),
        command_min=min(point.command for point in selected),
        command_max=max(point.command for point in selected),
    )
    residuals = [signed_quadratic_force(point.command, fit) - point.force_mean_n for point in selected]
    rmse = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    return SignedQuadraticFit(
        side=side,
        quadratic_coeff=quadratic,
        linear_coeff=linear,
        rmse_n=rmse,
        sample_count=len(selected),
        command_min=fit.command_min,
        command_max=fit.command_max,
    )


def fit_omega_quadratic(
    points: Sequence[SummaryPoint],
    *,
    side: str,
    min_abs_omega_rad_s: float = 0.0,
) -> OmegaQuadraticFit:
    if side not in {"all", "positive", "negative"}:
        raise ValueError("side must be 'all', 'positive', or 'negative'")
    min_abs_omega_rad_s = abs(float(min_abs_omega_rad_s))
    candidates = [
        point
        for point in points
        if point.omega_mean_rad_s is not None
        and point.rpm_mean is not None
        and math.isfinite(point.omega_mean_rad_s)
        and math.isfinite(point.force_mean_n)
        and abs(point.omega_mean_rad_s) > min_abs_omega_rad_s
    ]
    if side == "positive":
        selected = [point for point in candidates if point.omega_mean_rad_s is not None and point.omega_mean_rad_s > 0.0]
    elif side == "negative":
        selected = [point for point in candidates if point.omega_mean_rad_s is not None and point.omega_mean_rad_s < 0.0]
    else:
        selected = candidates
    if len(selected) < 2:
        raise ValueError(f"not enough {side} points for omega quadratic fit")

    xx = xy = 0.0
    for point in selected:
        omega = float(point.omega_mean_rad_s)
        x = omega * abs(omega)
        xx += x * x
        xy += x * point.force_mean_n
    if xx <= 1e-18:
        raise ValueError(f"{side} omega fit is ill-conditioned")

    c1 = xy / xx
    residuals = [omega_quadratic_force(float(point.omega_mean_rad_s), OmegaQuadraticFit(side, c1, 0.0, 0, 0.0, 0.0, 0.0, 0.0)) - point.force_mean_n for point in selected]
    rmse = math.sqrt(sum(residual * residual for residual in residuals) / len(residuals))
    omegas = [float(point.omega_mean_rad_s) for point in selected]
    rpms = [float(point.rpm_mean) for point in selected if point.rpm_mean is not None]
    return OmegaQuadraticFit(
        side=side,
        c1=c1,
        rmse_n=rmse,
        sample_count=len(selected),
        omega_min_rad_s=min(omegas),
        omega_max_rad_s=max(omegas),
        rpm_min=min(rpms),
        rpm_max=max(rpms),
    )


def load_raw_rpm_force_points(csv_path: Path, *, phases: set[str] | None = None) -> list[SummaryPoint]:
    points: list[SummaryPoint] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if phases is not None and row.get("phase", "") not in phases:
                continue
            if (
                row.get("sensor_ok") != "1"
                or row.get("force_zeroed_n", "") == ""
                or row.get("target_rpm", "") == ""
                or row.get("target_omega_rad_s", "") == ""
            ):
                continue
            command = float(row["target_command"])
            force = float(row["force_zeroed_n"])
            rpm = float(row["target_rpm"])
            omega = float(row["target_omega_rad_s"])
            if not all(math.isfinite(value) for value in (command, force, rpm, omega)):
                continue
            points.append(
                SummaryPoint(
                    command=command,
                    force_mean_n=force,
                    force_median_n=force,
                    force_std_n=0.0,
                    sample_count=1,
                    repeat_index=0,
                    point_index=len(points),
                    rpm_mean=rpm,
                    rpm_std=0.0,
                    omega_mean_rad_s=omega,
                )
            )
    return points


def load_summary_points(summary_csv: Path) -> list[SummaryPoint]:
    points: list[SummaryPoint] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("target_command") or not row.get("force_mean_n"):
                continue
            points.append(
                SummaryPoint(
                    command=float(row["target_command"]),
                    force_mean_n=float(row["force_mean_n"]),
                    force_median_n=float(row["force_median_n"]),
                    force_std_n=float(row["force_std_n"]),
                    sample_count=int(row["sample_count"]),
                    repeat_index=int(row["repeat_index"]),
                    point_index=int(row["point_index"]),
                    rpm_mean=float(row["rpm_mean"]) if row.get("rpm_mean") else None,
                    rpm_std=float(row["rpm_std"]) if row.get("rpm_std") else None,
                    omega_mean_rad_s=float(row["omega_mean_rad_s"]) if row.get("omega_mean_rad_s") else None,
                )
            )
    return points


def write_fit_csv(fit_csv: Path, fits: Sequence[SignedQuadraticFit]) -> None:
    fit_csv.parent.mkdir(parents=True, exist_ok=True)
    with fit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "side",
                "model",
                "quadratic_coeff",
                "linear_coeff",
                "rmse_n",
                "sample_count",
                "command_min",
                "command_max",
            ]
        )
        for fit in fits:
            writer.writerow(
                [
                    fit.side,
                    "force_n = quadratic_coeff * command * abs(command) + linear_coeff * command",
                    f"{fit.quadratic_coeff:.9g}",
                    f"{fit.linear_coeff:.9g}",
                    f"{fit.rmse_n:.9g}",
                    fit.sample_count,
                    f"{fit.command_min:.6f}",
                    f"{fit.command_max:.6f}",
                ]
            )


def write_omega_fit_csv(fit_csv: Path, fits: Sequence[OmegaQuadraticFit]) -> None:
    fit_csv.parent.mkdir(parents=True, exist_ok=True)
    with fit_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "side",
                "model",
                "c1",
                "rmse_n",
                "sample_count",
                "omega_min_rad_s",
                "omega_max_rad_s",
                "rpm_min",
                "rpm_max",
            ]
        )
        for fit in fits:
            writer.writerow(
                [
                    fit.side,
                    "force_n = c1 * omega_rad_s * abs(omega_rad_s)",
                    f"{fit.c1:.9g}",
                    f"{fit.rmse_n:.9g}",
                    fit.sample_count,
                    f"{fit.omega_min_rad_s:.6f}",
                    f"{fit.omega_max_rad_s:.6f}",
                    f"{fit.rpm_min:.6f}",
                    f"{fit.rpm_max:.6f}",
                ]
            )


def save_discrete_plot(points: Sequence[SummaryPoint], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not points:
        raise RuntimeError("no summary points available for discrete plotting")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    commands = [point.command for point in points]
    forces = [point.force_mean_n for point in points]
    stds = [point.force_std_n for point in points]

    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.errorbar(commands, forces, yerr=stds, fmt="o", markersize=4, capsize=2, label="steady samples")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.axvline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("command")
    ax.set_ylabel("zeroed force (N)")
    ax.grid(True)
    ax.legend()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_omega_fit_plot(points: Sequence[SummaryPoint], fits: Sequence[OmegaQuadraticFit], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    valid = [
        point
        for point in points
        if point.rpm_mean is not None and point.omega_mean_rad_s is not None and math.isfinite(point.omega_mean_rad_s)
    ]
    if not valid:
        raise RuntimeError("no RPM summary points available for omega fit plotting")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

    marker_size = 8 if len(valid) > 200 else 20
    marker_alpha = 0.45 if len(valid) > 200 else 0.8

    axes[0].scatter(
        [point.command for point in valid],
        [float(point.rpm_mean) for point in valid],
        s=marker_size,
        alpha=marker_alpha,
    )
    axes[0].axhline(0.0, color="0.5", linewidth=0.8)
    axes[0].axvline(0.0, color="0.5", linewidth=0.8)
    axes[0].set_xlabel("command")
    axes[0].set_ylabel("RPM")
    axes[0].grid(True)

    axes[1].scatter(
        [float(point.omega_mean_rad_s) for point in valid],
        [point.force_mean_n for point in valid],
        s=marker_size,
        alpha=marker_alpha,
        label="samples",
    )
    for fit in fits:
        if fit.sample_count < 2:
            continue
        xs = [fit.omega_min_rad_s + (fit.omega_max_rad_s - fit.omega_min_rad_s) * i / 120.0 for i in range(121)]
        ys = [omega_quadratic_force(x, fit) for x in xs]
        axes[1].plot(xs, ys, linewidth=2, label=f"{fit.side}: c1={fit.c1:.3g}")
    axes[1].axhline(0.0, color="0.5", linewidth=0.8)
    axes[1].axvline(0.0, color="0.5", linewidth=0.8)
    axes[1].set_xlabel("omega (rad/s)")
    axes[1].set_ylabel("zeroed force (N)")
    axes[1].grid(True)
    axes[1].legend()

    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def save_fit_plot(points: Sequence[SummaryPoint], fits: Sequence[SignedQuadraticFit], plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not points:
        raise RuntimeError("no summary points available for fit plotting")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    ax.scatter([point.command for point in points], [point.force_mean_n for point in points], s=20, label="steady mean")
    for fit in fits:
        if fit.sample_count < 2:
            continue
        xs = [fit.command_min + (fit.command_max - fit.command_min) * i / 120.0 for i in range(121)]
        ys = [signed_quadratic_force(x, fit) for x in xs]
        ax.plot(xs, ys, linewidth=2, label=f"{fit.side} fit")
    ax.axhline(0.0, color="0.5", linewidth=0.8)
    ax.axvline(0.0, color="0.5", linewidth=0.8)
    ax.set_xlabel("command")
    ax.set_ylabel("zeroed force (N)")
    ax.grid(True)
    ax.legend()
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _config_key_to_dest(key: str) -> str:
    return key.replace("-", "_")


def _parser_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}


def load_config_defaults(path: Path, parser: argparse.ArgumentParser) -> dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a YAML mapping")

    data = raw.get(CONFIG_SECTION, raw)
    if not isinstance(data, dict):
        raise ValueError(f"config section '{CONFIG_SECTION}' must be a YAML mapping")

    valid_dests = _parser_dests(parser)
    defaults: dict[str, object] = {}
    unknown: list[str] = []
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("config keys must be strings")
        dest = _config_key_to_dest(key)
        if dest == "config":
            continue
        if dest not in valid_dests:
            unknown.append(key)
            continue
        defaults[dest] = value

    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config key(s): {joined}")
    return defaults


def _path_from_config(value: object) -> Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _validate_motor_name(value: object) -> str:
    name = str(value).strip()
    if not name or name in {".", ".."} or Path(name).name != name:
        raise ValueError("--motor-name must be a single directory name, not a path")
    return name


def _directory_has_data(path: Path) -> bool:
    return path.exists() and any(path.iterdir())


def _resolve_output_paths(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.output_dir = _path_from_config(args.output_dir)
    args.csv = _path_from_config(args.csv)
    args.summary_csv = _path_from_config(args.summary_csv)
    args.plot = _path_from_config(args.plot)
    args.points_plot = _path_from_config(args.points_plot)
    args.fit_plot = _path_from_config(args.fit_plot)
    args.fit_csv = _path_from_config(args.fit_csv)
    args.rpm_fit_plot = _path_from_config(args.rpm_fit_plot)
    args.rpm_fit_csv = _path_from_config(args.rpm_fit_csv)

    using_output_dir = args.output_dir is not None or args.motor_name not in (None, "")
    args.run_dir = None
    if using_output_dir:
        if args.output_dir is None or args.motor_name in (None, ""):
            parser.error("--output-dir and --motor-name must be provided together")
        try:
            args.motor_name = _validate_motor_name(args.motor_name)
        except ValueError as exc:
            parser.error(str(exc))
        args.run_dir = args.output_dir / args.motor_name
        if _directory_has_data(args.run_dir) and not args.overwrite:
            parser.error(f"output directory already contains data: {args.run_dir}; use --overwrite to overwrite")
        args.csv = args.csv or args.run_dir / "raw.csv"
        args.summary_csv = args.summary_csv or args.run_dir / "summary.csv"
        args.plot = args.plot or args.run_dir / "curve.png"
        args.points_plot = args.points_plot or args.run_dir / "points.png"
        args.fit_csv = args.fit_csv or args.run_dir / "fit.csv"
        args.fit_plot = args.fit_plot or args.run_dir / "fit.png"
        args.rpm_fit_csv = args.rpm_fit_csv or args.run_dir / "rpm_fit.csv"
        args.rpm_fit_plot = args.rpm_fit_plot or args.run_dir / "rpm_fit.png"
        return

    args.motor_name = None
    if args.csv is None:
        parser.error("--output-dir and --motor-name are required, or provide legacy --csv")
    args.summary_csv = args.summary_csv or args.csv.with_name(f"{args.csv.stem}_summary{args.csv.suffix}")
    args.points_plot = args.points_plot or args.summary_csv.with_name(f"{args.summary_csv.stem}_points.png")
    args.fit_csv = args.fit_csv or args.summary_csv.with_name(f"{args.summary_csv.stem}_fit.csv")
    args.fit_plot = args.fit_plot or args.summary_csv.with_name(f"{args.summary_csv.stem}_fit.png")
    args.rpm_fit_csv = args.rpm_fit_csv or args.summary_csv.with_name(f"{args.summary_csv.stem}_rpm_fit.csv")
    args.rpm_fit_plot = args.rpm_fit_plot or args.summary_csv.with_name(f"{args.summary_csv.stem}_rpm_fit.png")


def _remove_ros_args(argv: Sequence[str]) -> list[str]:
    try:
        from rclpy.utilities import remove_ros_args

        return remove_ros_args(args=list(argv))
    except ModuleNotFoundError:
        pass

    cleaned: list[str] = []
    iterator = iter(argv)
    for item in iterator:
        if item != "--ros-args":
            cleaned.append(item)
            continue
        for ros_item in iterator:
            if ros_item == "--":
                break
    return cleaned


def publish_values(pub, values: Sequence[float]) -> None:
    from std_msgs.msg import Float32MultiArray

    msg = Float32MultiArray()
    msg.data = list(values)
    pub.publish(msg)


def collect_zero_offset(
    *,
    node,
    pub,
    sensor: ModbusForceSensor,
    samples: int,
    sample_rate_hz: float,
) -> float:
    if samples <= 0:
        return 0.0

    period = 1.0 / max(sample_rate_hz, 0.1)
    forces: list[float] = []
    node.get_logger().warn(f"collecting zero offset from {samples} sensor samples")
    for _ in range(samples):
        publish_values(pub, _zeros())
        reading = sensor.read()
        if reading.ok and reading.force_n is not None:
            forces.append(reading.force_n)
        else:
            node.get_logger().warn(f"zero sample failed: {reading.error}")
        time.sleep(period)

    if not forces:
        raise RuntimeError("failed to collect any valid zero samples")
    zero = statistics.median(forces)
    node.get_logger().warn(f"zero offset set to {zero:+.6f} N from {len(forces)} valid samples")
    return zero


def save_plot(csv_path: Path, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elapsed: list[float] = []
    command: list[float] = []
    force: list[float] = []
    phase: list[str] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sensor_ok") != "1" or row.get("force_zeroed_n", "") == "":
                continue
            elapsed.append(float(row["elapsed_sec"]))
            command.append(float(row["target_command"]))
            force.append(float(row["force_zeroed_n"]))
            phase.append(row.get("phase", ""))

    if not elapsed:
        raise RuntimeError("no valid force samples available for plotting")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(elapsed, command, label="command")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("command")
    axes[0].grid(True)
    axes[0].legend()

    has_loop_phases = any(item in {"ramp_negative_to_positive", "ramp_positive_to_negative"} for item in phase)
    if has_loop_phases:
        phase_styles = [
            ("ramp_negative_to_positive", "negative to positive", "tab:blue"),
            ("ramp_positive_to_negative", "positive to negative", "tab:orange"),
        ]
        plotted_phases = set()
        for phase_name, label, color in phase_styles:
            xs = [value for value, item in zip(command, phase) if item == phase_name]
            ys = [value for value, item in zip(force, phase) if item == phase_name]
            if xs:
                axes[1].scatter(xs, ys, s=10, alpha=0.75, label=label, color=color)
                plotted_phases.add(phase_name)
        other_xs = [
            value for value, item in zip(command, phase) if item not in plotted_phases and "zero" not in item
        ]
        other_ys = [value for value, item in zip(force, phase) if item not in plotted_phases and "zero" not in item]
        if other_xs:
            axes[1].scatter(other_xs, other_ys, s=8, alpha=0.35, label="hold/other", color="tab:gray")
        axes[1].legend()
    else:
        axes[1].scatter(command, force, s=8, alpha=0.7)
    axes[1].set_xlabel("command")
    axes[1].set_ylabel("zeroed force (N)")
    axes[1].grid(True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a thruster command sweep while reading a Modbus force transmitter and logging a thrust curve CSV."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML defaults file; command-line arguments override it")
    parser.add_argument("--index", type=int, default=None, help="canonical thruster index [0..7]")
    parser.add_argument("--topic", default=DEFAULT_THRUSTER_TOPIC, help=f"output topic, default: {DEFAULT_THRUSTER_TOPIC}")
    parser.add_argument(
        "--mode",
        choices=("step", "ramp", "loop", "rpm-sweep"),
        default="step",
        help="measurement input profile: step, ramp, loop (-A -> +A -> -A), or rpm-sweep, default: step",
    )
    parser.add_argument("--amplitude", type=float, default=1.0, help="sweep limit, default: 1.0")
    parser.add_argument("--rate", type=float, default=10.0, help="publish/read/log rate in Hz, default: 10")
    parser.add_argument("--pre-zero", type=float, default=5.0, help="initial zero-command duration, seconds")
    parser.add_argument("--marker-amplitude", type=float, default=0.0, help="disabled by default for integrated logging")
    parser.add_argument("--marker-on", type=float, default=0.5, help="sync marker on duration, seconds")
    parser.add_argument("--marker-off", type=float, default=0.5, help="sync marker off duration, seconds")
    parser.add_argument("--marker-count", type=int, default=0, help="sync marker count, default: 0")
    parser.add_argument("--settle", type=float, default=3.0, help="zero-command settling time before sweep")
    parser.add_argument("--hold-start", type=float, default=2.0, help="hold negative max before ramp, seconds")
    parser.add_argument("--ramp", type=float, default=60.0, help="linear ramp duration, seconds")
    parser.add_argument("--hold-end", type=float, default=2.0, help="hold positive max after ramp, seconds")
    parser.add_argument("--post-zero", type=float, default=5.0, help="final zero-command duration, seconds")
    parser.add_argument("--bidirectional", action="store_true", help="also sweep back from positive max to negative max")
    parser.add_argument("--step", type=float, default=0.1, help="step-mode command interval, default: 0.1")
    parser.add_argument("--step-hold", type=float, default=5.0, help="step-mode hold duration per command, seconds")
    parser.add_argument("--step-sample", type=float, default=2.0, help="step-mode steady sample window at end of each hold, seconds")
    parser.add_argument("--step-rest", type=float, default=1.0, help="step-mode zero-command rest after each command, seconds")
    parser.add_argument("--step-commands", default="", help="optional comma-separated command list, overrides --amplitude/--step")
    parser.add_argument("--repeat-count", type=int, default=1, help="step-mode repeats of the full command list")
    parser.add_argument("--repeat-rest", type=float, default=0.0, help="zero-command rest between repeated full command lists, seconds")
    parser.add_argument("--output-dir", type=Path, default=None, help="root output directory; files are written under <output-dir>/<motor-name>")
    parser.add_argument("--motor-name", default=None, help="motor identifier used as the per-run output directory name, e.g. M001")
    parser.add_argument("--overwrite", action="store_true", help="allow writing into an existing non-empty output directory")
    parser.add_argument("--csv", type=Path, default=None, help="legacy override for raw CSV path")
    parser.add_argument("--summary-csv", type=Path, default=None, help="legacy override for step-mode summary CSV path")
    parser.add_argument("--plot", type=Path, default=None, help="legacy override for output curve plot path")
    parser.add_argument("--points-plot", type=Path, default=None, help="legacy override for step-mode steady point plot path")
    parser.add_argument("--fit-plot", type=Path, default=None, help="legacy override for step-mode signed quadratic fit plot path")
    parser.add_argument("--fit-csv", type=Path, default=None, help="legacy override for step-mode signed quadratic fit parameters CSV path")
    parser.add_argument("--rpm-fit-plot", type=Path, default=None, help="legacy override for step-mode omega-force fit plot path")
    parser.add_argument("--rpm-fit-csv", type=Path, default=None, help="legacy override for step-mode omega-force c1 fit CSV path")
    parser.add_argument("--fit-min-abs-command", type=float, default=0.0, help="ignore commands with abs(command) <= this value when fitting")
    parser.add_argument("--rpm-fit-min-abs-rpm", type=float, default=0.0, help="ignore RPM values with abs(rpm) <= this value when fitting c1")
    parser.add_argument("--no-fit", action="store_true", help="do not create fitting outputs")
    parser.add_argument("--force-topic", default="/finsrov/thruster_curve/force_zeroed_n", help="real-time zeroed force topic; set empty to disable")
    parser.add_argument("--force-log-period", type=float, default=1.0, help="terminal force log period in seconds; set 0 to disable")
    parser.add_argument("--rpm-topic", default="/finsrov/hardware/motor_rpm_raw", help="canonical motor RPM topic; set empty to disable RPM logging/fitting")
    parser.add_argument("--rpm-stale-sec", type=float, default=0.5, help="max RPM sample age before treating it as unavailable; set 0 to never expire")
    parser.add_argument("--sensor-port", default=None, help="Modbus serial port, e.g. /dev/ttyUSB0 or COM4")
    parser.add_argument("--sensor-baudrate", type=int, default=115200)
    parser.add_argument("--sensor-bytesize", type=int, default=8)
    parser.add_argument("--sensor-parity", default="N")
    parser.add_argument("--sensor-stopbits", type=int, default=1)
    parser.add_argument("--sensor-timeout", type=float, default=0.2)
    parser.add_argument("--sensor-address", type=int, default=205)
    parser.add_argument("--sensor-count", type=int, default=1)
    parser.add_argument("--sensor-device-id", type=int, default=1)
    parser.add_argument("--sensor-divisor", type=float, default=100.0)
    parser.add_argument("--sensor-scale", type=float, default=0.98)
    parser.add_argument("--sensor-offset-n", type=float, default=0.0)
    parser.add_argument("--zero-samples", type=int, default=20, help="initial zero samples before sweep; set 0 to disable")
    parser.add_argument("--no-stop-on-exit", action="store_true", help="do not publish a final all-zero frame")
    parser.add_argument("--print-plan", action="store_true", help="print plan and exit without opening ROS or sensor")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)
    argv = _remove_ros_args(argv)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = _build_parser()
    if config_args.config is not None:
        try:
            defaults = load_config_defaults(config_args.config, parser)
        except ValueError as exc:
            parser.error(f"{config_args.config}: {exc}")
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    _resolve_output_paths(args, parser)
    if args.index is None:
        parser.error("--index is required, either on the command line or in --config")
    if args.sensor_port in (None, ""):
        parser.error("--sensor-port is required, either on the command line or in --config")
    if args.step_commands is None:
        args.step_commands = ""
    elif isinstance(args.step_commands, (list, tuple)):
        args.step_commands = ",".join(str(value) for value in args.step_commands)
    else:
        args.step_commands = str(args.step_commands)
    return args


def _config_from_args(args: argparse.Namespace) -> SweepConfig:
    loop_mode = args.mode in {"loop", "rpm-sweep"}
    return SweepConfig(
        index=args.index,
        amplitude=args.amplitude,
        pre_zero_sec=args.pre_zero,
        marker_amplitude=args.marker_amplitude,
        marker_on_sec=args.marker_on,
        marker_off_sec=args.marker_off,
        marker_count=args.marker_count,
        settle_sec=args.settle,
        hold_start_sec=0.0 if loop_mode else args.hold_start,
        ramp_sec=args.ramp,
        hold_end_sec=0.0 if loop_mode else args.hold_end,
        post_zero_sec=args.post_zero,
        bidirectional=True if loop_mode else args.bidirectional,
    )


def _step_config_from_args(args: argparse.Namespace) -> StepConfig:
    commands = parse_step_commands(args.step_commands) if args.step_commands else make_default_step_commands(args.amplitude, args.step)
    return StepConfig(
        index=args.index,
        amplitude=args.amplitude,
        step=args.step,
        commands=commands,
        pre_zero_sec=args.pre_zero,
        settle_sec=args.settle,
        hold_sec=args.step_hold,
        sample_sec=args.step_sample,
        rest_sec=args.step_rest,
        repeat_rest_sec=args.repeat_rest,
        post_zero_sec=args.post_zero,
        repeat_count=args.repeat_count,
    )


def main() -> None:
    parser = _build_parser()
    args = parse_args()
    try:
        if args.mode == "step":
            step_config = _step_config_from_args(args)
            validate_step_config(step_config)
            duration = step_total_duration(step_config)
            sweep_config = None
        else:
            sweep_config = _config_from_args(args)
            validate_sweep_config(sweep_config)
            duration = total_duration(sweep_config)
            step_config = None
    except ValueError as exc:
        parser.error(str(exc))

    if args.print_plan:
        active_index = step_config.index if step_config is not None else sweep_config.index  # type: ignore[union-attr]
        print(f"index={active_index} name={CANONICAL_NAMES[active_index]}")
        print(f"topic={args.topic}")
        print(f"mode={args.mode}")
        print(f"duration_sec={duration:.3f}")
        if step_config is not None:
            print(f"commands={','.join(f'{value:.3f}' for value in step_config.commands)}")
            print(f"step_hold_sec={step_config.hold_sec:.3f}")
            print(f"step_sample_sec={step_config.sample_sec:.3f}")
            print(f"step_rest_sec={step_config.rest_sec:.3f}")
            print(f"repeat_count={step_config.repeat_count}")
            print(f"repeat_rest_sec={step_config.repeat_rest_sec:.3f}")
        else:
            if args.mode in {"loop", "rpm-sweep"}:
                print(
                    f"sweep=-{sweep_config.amplitude:.3f} -> +{sweep_config.amplitude:.3f} -> "
                    f"-{sweep_config.amplitude:.3f}"
                )
            else:
                print(f"sweep=-{sweep_config.amplitude:.3f} -> +{sweep_config.amplitude:.3f}")  # type: ignore[union-attr]
            print(f"ramp_sec={sweep_config.ramp_sec:.3f}")  # type: ignore[union-attr]
            print(f"bidirectional={sweep_config.bidirectional}")  # type: ignore[union-attr]
        if args.run_dir is not None:
            print(f"output_dir={args.output_dir}")
            print(f"motor_name={args.motor_name}")
            print(f"run_dir={args.run_dir}")
            print(f"overwrite={args.overwrite}")
        print(f"csv={args.csv}")
        if args.force_topic:
            print(f"force_topic={args.force_topic}")
        print(f"force_log_period_sec={args.force_log_period:.3f}")
        if args.mode == "step":
            print(f"summary_csv={args.summary_csv}")
            if not args.no_fit:
                print(f"points_plot={args.points_plot}")
                print(f"fit_csv={args.fit_csv}")
                print(f"fit_plot={args.fit_plot}")
                print(f"rpm_fit_csv={args.rpm_fit_csv}")
                print(f"rpm_fit_plot={args.rpm_fit_plot}")
                print(f"fit_min_abs_command={args.fit_min_abs_command:.6f}")
                print(f"rpm_fit_min_abs_rpm={args.rpm_fit_min_abs_rpm:.6f}")
        elif args.mode == "rpm-sweep" and not args.no_fit:
            print(f"rpm_fit_csv={args.rpm_fit_csv}")
            print(f"rpm_fit_plot={args.rpm_fit_plot}")
            print(f"rpm_fit_min_abs_rpm={args.rpm_fit_min_abs_rpm:.6f}")
            print("rpm_fit_source=raw_csv")
        if args.rpm_topic:
            print(f"rpm_topic={args.rpm_topic}")
            print(f"rpm_stale_sec={args.rpm_stale_sec:.3f}")
        print(f"sensor_port={args.sensor_port}")
        return

    import rclpy
    from std_msgs.msg import Float32, Float32MultiArray

    sensor = ModbusForceSensor(
        port=args.sensor_port,
        baudrate=args.sensor_baudrate,
        bytesize=args.sensor_bytesize,
        parity=args.sensor_parity,
        stopbits=args.sensor_stopbits,
        timeout=args.sensor_timeout,
        address=args.sensor_address,
        count=args.sensor_count,
        device_id=args.sensor_device_id,
        divisor=args.sensor_divisor,
        scale=args.sensor_scale,
        offset_n=args.sensor_offset_n,
    )
    if not sensor.connect():
        raise SystemExit(f"failed to open force sensor serial port: {args.sensor_port}")

    rclpy.init()
    node = rclpy.create_node("thruster_curve_measure")
    pub = node.create_publisher(Float32MultiArray, args.topic, 10)
    force_pub = node.create_publisher(Float32, args.force_topic, 10) if args.force_topic else None
    latest_rpm = LatestRpm(count=len(CANONICAL_NAMES), clock=node.get_clock())
    rpm_sub = (
        node.create_subscription(Float32MultiArray, args.rpm_topic, latest_rpm.callback, 10)
        if args.rpm_topic
        else None
    )
    if args.run_dir is not None:
        args.run_dir.mkdir(parents=True, exist_ok=True)
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        active_index = step_config.index if step_config is not None else sweep_config.index  # type: ignore[union-attr]
        node.get_logger().warn(
            "starting integrated thrust curve measurement: "
            f"mode={args.mode}, topic={args.topic}, index={active_index}({CANONICAL_NAMES[active_index]}), "
            f"sensor={args.sensor_port}, duration={duration:.2f}s, csv={args.csv}"
        )
        zero_offset_n = collect_zero_offset(
            node=node,
            pub=pub,
            sensor=sensor,
            samples=args.zero_samples,
            sample_rate_hz=args.rate,
        )

        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            _write_header(writer)
            period = 1.0 / max(args.rate, 0.1)
            start = time.monotonic()
            next_tick = start
            next_force_log = start
            step_summary_samples: dict[tuple[int, int], list[float]] = {}
            step_rpm_samples: dict[tuple[int, int], list[float]] = {}

            while rclpy.ok():
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                now = time.monotonic()
                elapsed = now - start
                if step_config is not None:
                    sample = step_sample_at_elapsed(elapsed, step_config)
                    if sample.done:
                        break
                    values = values_for_index_command(step_config.index, sample.command)
                    index = step_config.index
                    phase = sample.phase
                    sample_window = sample.sample_window
                    point_index = sample.point_index
                    repeat_index = sample.repeat_index
                else:
                    ramp_sample = command_at_elapsed(elapsed, sweep_config)  # type: ignore[arg-type]
                    if ramp_sample.done:
                        break
                    values = values_for_sample(ramp_sample, sweep_config.index)  # type: ignore[union-attr]
                    index = sweep_config.index  # type: ignore[union-attr]
                    phase = ramp_sample.phase
                    sample_window = False
                    point_index = None
                    repeat_index = None

                publish_values(pub, values)
                rclpy.spin_once(node, timeout_sec=0.0)
                reading = sensor.read()
                rpm_values, rpm_age_sec = latest_rpm.latest(max_age_sec=args.rpm_stale_sec)
                force_zeroed = reading.force_n - zero_offset_n if reading.ok and reading.force_n is not None else None
                if force_zeroed is not None and force_pub is not None:
                    force_msg = Float32()
                    force_msg.data = float(force_zeroed)
                    force_pub.publish(force_msg)
                if args.force_log_period > 0.0 and now >= next_force_log:
                    if force_zeroed is None:
                        node.get_logger().warn(
                            f"force read failed: phase={phase}, command={values[index]:+.3f}, error={reading.error}"
                        )
                    else:
                        node.get_logger().info(
                            f"force_zeroed={force_zeroed:+.3f} N, raw_force={reading.force_n:+.3f} N, "
                            f"command={values[index]:+.3f}, "
                            f"rpm={'n/a' if rpm_values is None else f'{rpm_values[index]:+.1f}'}, phase={phase}"
                        )
                    next_force_log = now + args.force_log_period
                _write_row(
                    writer,
                    wall_time=time.time(),
                    monotonic_time=now,
                    elapsed_sec=elapsed,
                    phase=phase,
                    index=index,
                    values=values,
                    reading=reading,
                    zero_offset_n=zero_offset_n,
                    rpm_values=rpm_values,
                    rpm_age_sec=rpm_age_sec,
                    sample_window=sample_window,
                    point_index=point_index,
                    repeat_index=repeat_index,
                )
                if (
                    step_config is not None
                    and sample_window
                    and point_index is not None
                    and repeat_index is not None
                    and reading.ok
                    and reading.force_n is not None
                ):
                    step_summary_samples.setdefault((repeat_index, point_index), []).append(force_zeroed)
                    if rpm_values is not None:
                        step_rpm_samples.setdefault((repeat_index, point_index), []).append(float(rpm_values[index]))
                next_tick += period

        if not args.no_stop_on_exit:
            publish_values(pub, _zeros())
        if step_config is not None:
            summary_csv = args.summary_csv
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                summary_writer = csv.writer(f)
                _write_summary_header(summary_writer)
                _write_summary_rows(
                    summary_writer,
                    index=step_config.index,
                    samples=step_summary_samples,
                    rpm_samples=step_rpm_samples,
                    commands=step_config.commands,
                )
            node.get_logger().info(f"saved step summary CSV: {summary_csv}")
            if not args.no_fit:
                points = load_summary_points(summary_csv)
                save_discrete_plot(points, args.points_plot)
                fits = []
                for side in ("negative", "positive"):
                    try:
                        fits.append(
                            fit_signed_quadratic(points, side=side, min_abs_command=args.fit_min_abs_command)
                        )
                    except ValueError as exc:
                        node.get_logger().warn(f"skipping {side} fit: {exc}")
                if fits:
                    write_fit_csv(args.fit_csv, fits)
                    save_fit_plot(points, fits, args.fit_plot)
                    node.get_logger().info(f"saved signed quadratic fit CSV: {args.fit_csv}")
                    node.get_logger().info(f"saved signed quadratic fit plot: {args.fit_plot}")
                else:
                    node.get_logger().warn("skipping fit outputs: no side had enough valid points")
                node.get_logger().info(f"saved steady point plot: {args.points_plot}")
                omega_fits = []
                min_abs_omega = rpm_to_omega_rad_s(args.rpm_fit_min_abs_rpm)
                for side in ("all", "negative", "positive"):
                    try:
                        omega_fits.append(fit_omega_quadratic(points, side=side, min_abs_omega_rad_s=min_abs_omega))
                    except ValueError as exc:
                        node.get_logger().warn(f"skipping {side} omega fit: {exc}")
                if omega_fits:
                    write_omega_fit_csv(args.rpm_fit_csv, omega_fits)
                    save_omega_fit_plot(points, omega_fits, args.rpm_fit_plot)
                    node.get_logger().info(f"saved omega-force c1 fit CSV: {args.rpm_fit_csv}")
                    node.get_logger().info(f"saved omega-force c1 fit plot: {args.rpm_fit_plot}")
                else:
                    node.get_logger().warn("skipping omega-force fit outputs: not enough valid RPM summary points")
        elif args.mode == "rpm-sweep" and not args.no_fit:
            points = load_raw_rpm_force_points(
                args.csv,
                phases={"ramp_negative_to_positive", "ramp_positive_to_negative"},
            )
            omega_fits = []
            min_abs_omega = rpm_to_omega_rad_s(args.rpm_fit_min_abs_rpm)
            for side in ("all", "negative", "positive"):
                try:
                    omega_fits.append(fit_omega_quadratic(points, side=side, min_abs_omega_rad_s=min_abs_omega))
                except ValueError as exc:
                    node.get_logger().warn(f"skipping {side} raw omega fit: {exc}")
            if omega_fits:
                write_omega_fit_csv(args.rpm_fit_csv, omega_fits)
                save_omega_fit_plot(points, omega_fits, args.rpm_fit_plot)
                node.get_logger().info(
                    f"saved raw omega-force c1 fit CSV: {args.rpm_fit_csv} from {len(points)} samples"
                )
                node.get_logger().info(f"saved raw omega-force c1 fit plot: {args.rpm_fit_plot}")
            else:
                node.get_logger().warn("skipping raw omega-force fit outputs: not enough valid RPM/force samples")
        if args.plot is not None:
            save_plot(args.csv, args.plot)
            node.get_logger().info(f"saved thrust curve plot: {args.plot}")
        node.get_logger().info(f"saved thrust curve CSV: {args.csv}")
    except KeyboardInterrupt:
        if not args.no_stop_on_exit:
            publish_values(pub, _zeros())
        node.get_logger().warn("thruster curve measurement interrupted")
    finally:
        sensor.close()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
