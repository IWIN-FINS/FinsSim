"""Plot per-force excitation segments from a hydrodynamic identifier CSV.

The command is intentionally offline: it only reads a completed CSV and
writes deterministic PNG diagnostics next to that run.  It never creates ROS
publishers or sends a thruster command.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .hydrodynamic_identifier_node import AXIS_ORDER


@dataclass(frozen=True)
class AxisColumns:
    velocity: str
    acceleration: str
    velocity_label: str
    acceleration_label: str
    velocity_unit: str
    acceleration_unit: str
    force_unit: str


AXIS_COLUMNS = {
    "surge_x": AxisColumns("nu_x_mps", "nudot_x_mps2", "surge velocity u", "kinematic acceleration a", "m/s", "m/s^2", "N"),
    "heave_y": AxisColumns("nu_y_mps", "nudot_y_mps2", "heave velocity w", "kinematic acceleration a", "m/s", "m/s^2", "N"),
    "sway_z": AxisColumns("nu_z_mps", "nudot_z_mps2", "sway velocity v", "kinematic acceleration a", "m/s", "m/s^2", "N"),
    "roll_x": AxisColumns("nu_roll_x_radps", "nudot_roll_x_radps2", "roll rate p", "angular acceleration alpha", "rad/s", "rad/s^2", "Nm"),
    "pitch_z": AxisColumns("nu_pitch_z_radps", "nudot_pitch_z_radps2", "pitch rate q", "angular acceleration alpha", "rad/s", "rad/s^2", "Nm"),
    "yaw_y": AxisColumns("nu_yaw_radps", "nudot_yaw_radps2", "yaw rate r", "angular acceleration alpha", "rad/s", "rad/s^2", "Nm"),
}


@dataclass(frozen=True)
class Segment:
    level: float
    trial_index: int
    time_sec: tuple[float, ...]
    velocity: tuple[float, ...]
    acceleration: tuple[float, ...]


def _float(row: dict[str, str], name: str, default: float = float("nan")) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _integer(row: dict[str, str], name: str, default: int = -1) -> int:
    try:
        return int(row.get(name, default))
    except (TypeError, ValueError):
        return default


def _sample_time(row: dict[str, str]) -> float:
    synchronized = _float(row, "sample_time_ros_sec")
    return synchronized if math.isfinite(synchronized) else _float(row, "time_sec", 0.0)


def _force_token(level: float, unit: str) -> str:
    magnitude = f"{abs(level):g}".replace(".", "p")
    return f"{'p' if level >= 0.0 else 'm'}{magnitude}{unit}"


def _force_label(level: float, unit: str) -> str:
    return f"{level:+g} {unit}"


def _axis_from_rows(rows: Iterable[dict[str, str]], axis_override: str | None) -> str:
    if axis_override is not None:
        return axis_override
    axes = {str(row.get("axis", "")).strip() for row in rows}
    supported = sorted(axis for axis in axes if axis in AXIS_COLUMNS)
    if len(supported) != 1:
        raise ValueError(f"Cannot infer exactly one supported axis; found {supported or 'none'}")
    return supported[0]


def load_segments(csv_path: Path, axis_override: str | None = None) -> tuple[str, list[Segment]]:
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    axis = _axis_from_rows(rows, axis_override)
    columns = AXIS_COLUMNS[axis]
    grouped: dict[tuple[int, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if str(row.get("axis", "")).strip() != axis or str(row.get("phase", "")).strip() != "excitation":
            continue
        level = _float(row, "level")
        time_sec = _sample_time(row)
        velocity = _float(row, columns.velocity)
        acceleration = _float(row, columns.acceleration)
        if not all(math.isfinite(value) for value in (level, time_sec, velocity, acceleration)):
            continue
        grouped[(_integer(row, "trial_index"), level)].append(row)

    segments: list[Segment] = []
    for (trial_index, level), entries in grouped.items():
        entries.sort(key=_sample_time)
        start = _sample_time(entries[0])
        segments.append(
            Segment(
                level=level,
                trial_index=trial_index,
                time_sec=tuple(_sample_time(row) - start for row in entries),
                velocity=tuple(_float(row, columns.velocity) for row in entries),
                acceleration=tuple(_float(row, columns.acceleration) for row in entries),
            )
        )
    if not segments:
        raise ValueError(f"No finite {axis} excitation samples found in {csv_path}")
    return axis, sorted(segments, key=lambda segment: (segment.level, segment.trial_index))


def _style_axis(axis, ylabel: str, xlabel: str | None = None) -> None:
    axis.axhline(0.0, color="0.35", linewidth=0.8)
    axis.grid(True, alpha=0.30)
    axis.set_ylabel(ylabel)
    if xlabel is not None:
        axis.set_xlabel(xlabel)


def _plot_segment(segment: Segment, axis: str, run_id: str, output_path: Path) -> None:
    columns = AXIS_COLUMNS[axis]
    fig, (velocity_axis, acceleration_axis) = plt.subplots(2, 1, figsize=(10, 6.5), dpi=160, sharex=True, constrained_layout=True)
    title = f"{run_id} | applied force {_force_label(segment.level, columns.force_unit)} | excitation only | {len(segment.time_sec)} samples"
    fig.suptitle(title)
    velocity_axis.plot(segment.time_sec, segment.velocity, color="tab:blue", linewidth=2.0)
    acceleration_axis.plot(segment.time_sec, segment.acceleration, color="tab:orange", linewidth=2.0)
    _style_axis(velocity_axis, f"{columns.velocity_label} ({columns.velocity_unit})")
    _style_axis(
        acceleration_axis,
        f"{columns.acceleration_label} ({columns.acceleration_unit})",
        "time since excitation start (s)",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def _plot_all_segments(segments: list[Segment], axis: str, run_id: str, output_path: Path) -> None:
    columns = AXIS_COLUMNS[axis]
    figure, axes = plt.subplots(
        len(segments),
        2,
        figsize=(14, max(3.0, 2.5 * len(segments))),
        dpi=160,
        squeeze=False,
        constrained_layout=True,
    )
    figure.suptitle(f"{run_id}: velocity and gravity-compensated acceleration per force segment")
    for index, segment in enumerate(segments):
        velocity_axis, acceleration_axis = axes[index]
        velocity_axis.plot(segment.time_sec, segment.velocity, color="tab:blue", linewidth=1.5)
        acceleration_axis.plot(segment.time_sec, segment.acceleration, color="tab:orange", linewidth=1.5)
        _style_axis(velocity_axis, f"{_force_label(segment.level, columns.force_unit)}\n{columns.velocity_label} ({columns.velocity_unit})")
        _style_axis(acceleration_axis, f"{columns.acceleration_label} ({columns.acceleration_unit})")
        if index == len(segments) - 1:
            velocity_axis.set_xlabel("time since excitation start (s)")
            acceleration_axis.set_xlabel("time since excitation start (s)")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def plot_force_segments(csv_path: Path, *, axis_override: str | None = None, output_dir: Path | None = None) -> list[Path]:
    csv_path = csv_path.expanduser().resolve()
    axis, segments = load_segments(csv_path, axis_override=axis_override)
    destination = (output_dir or csv_path.parent).expanduser().resolve()
    run_id = csv_path.stem
    columns = AXIS_COLUMNS[axis]
    outputs: list[Path] = []
    all_segments_path = destination / f"{run_id}_all_force_segments_velocity_acceleration.png"
    _plot_all_segments(segments, axis, run_id, all_segments_path)
    outputs.append(all_segments_path)
    for segment in segments:
        path = destination / f"{run_id}_force_{_force_token(segment.level, columns.force_unit)}_velocity_acceleration.png"
        _plot_segment(segment, axis, run_id, path)
        outputs.append(path)
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="Completed hydrodynamic identifier CSV.")
    parser.add_argument("--axis", choices=AXIS_ORDER, default=None, help="Override axis inferred from CSV.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Directory for PNG files; defaults to the CSV directory.")
    args = parser.parse_args()
    for output in plot_force_segments(args.csv, axis_override=args.axis, output_dir=args.output_dir):
        print(output)


if __name__ == "__main__":
    main()
