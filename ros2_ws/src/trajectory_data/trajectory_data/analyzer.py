from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from .alignment import TimedRecord, rate_summary, safe_json, sorted_records


POOL_BOUNDS_M = {
    "x": {"min": -2.0, "max": 2.0},
    # The pool bottom is y=-1 m. No y upper limit was specified, so a
    # positive y estimate is reported but is not classified as out of bounds.
    "y": {"min": -1.0, "max": None},
    "z": {"min": -1.0, "max": 1.0},
}
AXIS_NAMES = ("x", "y", "z")


def _load_rows(path: Path) -> list[dict[str, Any]]:
    return pq.read_table(path).to_pylist() if path.is_file() else []


def _records(rows: list[dict[str, Any]]) -> list[TimedRecord]:
    return sorted_records(
        TimedRecord(
            time_sec=float(row["source_time_sec"]),
            bag_time_sec=float(row.get("bag_time_sec", row["source_time_sec"])),
            payload=row,
        )
        for row in rows
        if row.get("source_time_sec") is not None
    )


def _array(row: dict[str, Any], field: str, size: int) -> np.ndarray | None:
    values = row.get(field)
    if values is None or len(values) < size or any(value is None for value in values[:size]):
        return None
    return np.asarray(values[:size], dtype=float)


def _candidate_jumps(records: Sequence[TimedRecord], field: str, size: int, signal: str, z_score: float) -> list[dict[str, Any]]:
    times: list[float] = []
    values: list[np.ndarray] = []
    for record in records:
        value = _array(record.payload, field, size)
        if value is not None and np.all(np.isfinite(value)):
            times.append(record.time_sec)
            values.append(value)
    if len(times) < 4:
        return []
    value_matrix = np.stack(values)
    dt = np.diff(np.asarray(times))
    delta = np.linalg.norm(np.diff(value_matrix, axis=0), axis=1)
    valid = dt > 1e-6
    rate = delta[valid] / dt[valid]
    if rate.size < 3:
        return []
    median = float(np.median(rate))
    mad = float(np.median(np.abs(rate - median)))
    threshold = max(median + float(z_score) * 1.4826 * mad, 1e-8)
    events: list[dict[str, Any]] = []
    for index, (step, change, derivative) in enumerate(zip(dt, delta, np.divide(delta, dt, out=np.zeros_like(delta), where=dt > 1e-6))):
        if step > 1e-6 and derivative > threshold:
            events.append(
                {
                    "event_time_sec": times[index + 1],
                    "signal": signal,
                    "delta": float(change),
                    "delta_per_sec": float(derivative),
                    "threshold_per_sec": threshold,
                }
            )
    return events


def _nearest_status(records: Sequence[TimedRecord], time_sec: float) -> dict[str, Any]:
    previous: TimedRecord | None = None
    for record in records:
        if record.time_sec > time_sec:
            break
        previous = record
    if previous is None:
        return {}
    payload = previous.payload.get("json", {})
    return payload if isinstance(payload, dict) else {}


def _sequence_gaps(records: Sequence[TimedRecord]) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    for first, second in zip(records, records[1:]):
        first_sequence = first.payload.get("telemetry_sequence")
        second_sequence = second.payload.get("telemetry_sequence")
        if first_sequence is None or second_sequence is None:
            continue
        expected = (int(first_sequence) + 1) & 0xFFFF
        if int(second_sequence) != expected:
            events.append(
                {
                    "event_time_sec": second.time_sec,
                    "signal": "telemetry_sequence_gap",
                    "delta": int(second_sequence) - int(first_sequence),
                    "delta_per_sec": None,
                    "threshold_per_sec": None,
                }
            )
    return events


def _pool_bounds_report(records: Sequence[TimedRecord], stream_name: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    positions: list[np.ndarray] = []
    valid_records: list[TimedRecord] = []
    for record in records:
        position = _array(record.payload, "position_xyz", 3)
        if position is not None and np.all(np.isfinite(position)):
            positions.append(position)
            valid_records.append(record)

    limits = {axis: dict(bounds) for axis, bounds in POOL_BOUNDS_M.items()}
    if not positions:
        return {
            "stream": stream_name,
            "samples": 0,
            "in_bounds_samples": 0,
            "out_of_bounds_samples": 0,
            "out_of_bounds_fraction": None,
            "out_of_bounds_segments": 0,
            "coordinate_min_m": None,
            "coordinate_max_m": None,
            "limits_m": limits,
            "by_axis": {},
        }, []

    matrix = np.stack(positions)
    out_of_bounds = np.zeros(len(valid_records), dtype=bool)
    per_axis: dict[str, dict[str, Any]] = {}
    event_details: list[list[tuple[str, str, float]]] = [[] for _ in valid_records]
    for index, axis_name in enumerate(AXIS_NAMES):
        values = matrix[:, index]
        bounds = POOL_BOUNDS_M[axis_name]
        lower = bounds["min"]
        upper = bounds["max"]
        below = np.zeros(len(values), dtype=bool) if lower is None else values < float(lower)
        above = np.zeros(len(values), dtype=bool) if upper is None else values > float(upper)
        out_of_bounds |= below | above
        for sample_index in np.flatnonzero(below):
            event_details[int(sample_index)].append((axis_name, "below_min", float(lower) - float(values[sample_index])))
        for sample_index in np.flatnonzero(above):
            event_details[int(sample_index)].append((axis_name, "above_max", float(values[sample_index]) - float(upper)))
        per_axis[axis_name] = {
            "min_m": None if lower is None else float(lower),
            "max_m": None if upper is None else float(upper),
            "observed_min_m": float(np.min(values)),
            "observed_max_m": float(np.max(values)),
            "below_min_samples": int(np.count_nonzero(below)),
            "above_max_samples": int(np.count_nonzero(above)),
            "max_below_min_m": None if not np.any(below) else float(np.max(float(lower) - values[below])),
            "max_above_max_m": None if not np.any(above) else float(np.max(values[above] - float(upper))),
        }

    segments = int(np.count_nonzero(out_of_bounds & np.concatenate(([True], ~out_of_bounds[:-1]))))
    events: list[dict[str, Any]] = []
    for index in np.flatnonzero(out_of_bounds):
        record = valid_records[int(index)]
        details = event_details[int(index)]
        events.append(
            {
                "event_time_sec": float(record.time_sec),
                "bag_time_sec": float(record.bag_time_sec),
                "stream": stream_name,
                "position_xyz": [float(value) for value in matrix[int(index)]],
                "violated_axes": [axis_name for axis_name, _, _ in details],
                "violation_directions": [direction for _, direction, _ in details],
                "max_violation_m": max(amount for _, _, amount in details),
            }
        )
    sample_count = len(valid_records)
    out_count = int(np.count_nonzero(out_of_bounds))
    return {
        "stream": stream_name,
        "samples": sample_count,
        "in_bounds_samples": sample_count - out_count,
        "out_of_bounds_samples": out_count,
        "out_of_bounds_fraction": float(out_count / sample_count),
        "out_of_bounds_segments": segments,
        "coordinate_min_m": [float(value) for value in np.min(matrix, axis=0)],
        "coordinate_max_m": [float(value) for value in np.max(matrix, axis=0)],
        "limits_m": limits,
        "by_axis": per_axis,
    }, events


def _plot_pose(path: Path, fused: Sequence[TimedRecord], vision: Sequence[TimedRecord]) -> None:
    if not fused and not vision:
        return
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for label, records, color in (("fused", fused, "tab:blue"), ("vision", vision, "tab:orange")):
        if not records:
            continue
        first_time = records[0].time_sec
        time = [record.time_sec - first_time for record in records]
        values = [_array(record.payload, "position_xyz", 3) for record in records]
        for axis_index in range(3):
            series = [value[axis_index] if value is not None else math.nan for value in values]
            axes[axis_index].plot(time, series, label=label, color=color, alpha=0.8)
            axes[axis_index].set_ylabel(f"position[{axis_index}] m")
            axes[axis_index].grid(alpha=0.25)
    axes[0].legend()
    axes[-1].set_xlabel("time since first sample [s]")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _plot_depth(path: Path, raw_depth: Sequence[TimedRecord], fused_depth: Sequence[TimedRecord]) -> None:
    if not raw_depth and not fused_depth:
        return
    figure, axis = plt.subplots(figsize=(14, 5))
    for label, records, invert_z in (
        ("-hardware raw depth.z (z-up)", raw_depth, True),
        ("fused depth.z (z-up)", fused_depth, False),
    ):
        if not records:
            continue
        first_time = records[0].time_sec
        time = [record.time_sec - first_time for record in records]
        values = []
        for record in records:
            position = _array(record.payload, "position_xyz", 3)
            values.append(-position[2] if position is not None and invert_z else position[2] if position is not None else math.nan)
        axis.plot(time, values, label=label)
    axis.set_xlabel("time since first sample [s]")
    axis.set_ylabel("z / depth [m]")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _pose_arrays(records: Sequence[TimedRecord]) -> tuple[np.ndarray, np.ndarray]:
    times: list[float] = []
    positions: list[np.ndarray] = []
    for record in records:
        position = _array(record.payload, "position_xyz", 3)
        if position is not None and np.all(np.isfinite(position)):
            times.append(record.time_sec)
            positions.append(position)
    if not positions:
        return np.empty(0, dtype=float), np.empty((0, 3), dtype=float)
    return np.asarray(times, dtype=float), np.stack(positions)


def _plot_trajectory_xyz(path: Path, fused_pose: Sequence[TimedRecord]) -> None:
    times, positions = _pose_arrays(fused_pose)
    if len(times) == 0:
        return
    times -= times[0]
    figure, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    for index, axis_name in enumerate(("x", "y", "z")):
        axis = axes[index]
        bounds = POOL_BOUNDS_M[axis_name]
        low, high = bounds["min"], bounds["max"]
        if low is not None and high is not None:
            axis.axhspan(low, high, color="tab:green", alpha=0.08, label="pool range")
        if low is not None:
            axis.axhline(low, color="tab:green", linestyle="--", linewidth=0.8, label="pool lower bound")
        if high is not None:
            axis.axhline(high, color="tab:green", linestyle="--", linewidth=0.8, label="pool upper bound")
        axis.plot(times, positions[:, index], color="tab:blue", linewidth=1.0, label="fused pose")
        axis.set_ylabel(f"{axis_name} [m]")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")
    axes[-1].set_xlabel("time since first fused pose [s]")
    figure.suptitle("Fused trajectory components in pool_world")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _expanded_limits(target: tuple[float, float], values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    low = min(float(target[0]), float(np.min(finite))) if finite.size else float(target[0])
    high = max(float(target[1]), float(np.max(finite))) if finite.size else float(target[1])
    margin = max((high - low) * 0.05, 0.05)
    return low - margin, high + margin


def _plot_trajectory_3d_pool(path: Path, fused_pose: Sequence[TimedRecord]) -> None:
    _, positions = _pose_arrays(fused_pose)
    if len(positions) == 0:
        return
    figure = plt.figure(figsize=(12, 8))
    axis = figure.add_subplot(111, projection="3d")
    x_low, x_high = POOL_BOUNDS_M["x"]["min"], POOL_BOUNDS_M["x"]["max"]
    y_low = POOL_BOUNDS_M["y"]["min"]
    y_high = max(0.0, float(np.max(positions[:, 1])))
    z_low, z_high = POOL_BOUNDS_M["z"]["min"], POOL_BOUNDS_M["z"]["max"]

    walls = [
        [(x_low, y_low, z_low), (x_high, y_low, z_low), (x_high, y_low, z_high), (x_low, y_low, z_high)],
        [(x_low, y_low, z_low), (x_low, y_high, z_low), (x_low, y_high, z_high), (x_low, y_low, z_high)],
        [(x_high, y_low, z_low), (x_high, y_high, z_low), (x_high, y_high, z_high), (x_high, y_low, z_high)],
        [(x_low, y_low, z_low), (x_high, y_low, z_low), (x_high, y_high, z_low), (x_low, y_high, z_low)],
    ]
    axis.add_collection3d(
        Poly3DCollection(walls, facecolors="tab:cyan", edgecolors="tab:gray", linewidths=0.8, alpha=0.10)
    )
    axis.plot(
        [x_low, x_high, x_high, x_low, x_low],
        [y_high] * 5,
        [z_low, z_low, z_high, z_high, z_low],
        color="tab:cyan",
        linestyle="--",
        linewidth=1.0,
        label="water surface y=0",
    )
    axis.plot(positions[:, 0], positions[:, 1], positions[:, 2], color="tab:blue", linewidth=1.4, label="fused pose trajectory")
    axis.scatter(*positions[0], color="tab:green", s=35, label="start")
    axis.scatter(*positions[-1], color="tab:red", s=35, label="end")
    axis.set_xlim(*_expanded_limits((x_low, x_high), positions[:, 0]))
    axis.set_ylim(*_expanded_limits((y_low, y_high), positions[:, 1]))
    axis.set_zlim(*_expanded_limits((z_low, z_high), positions[:, 2]))
    axis.set_box_aspect((4.0, 1.0, 2.0))
    axis.set_xlabel("x [m]")
    axis.set_ylabel("y [m], depth")
    axis.set_zlabel("z [m]")
    axis.set_title("Fused trajectory: x[-2,2], y>=-1, z[-1,1]")
    axis.view_init(elev=24.0, azim=-58.0)
    axis.legend(loc="upper left")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_action_rpm(path: Path, wrench: Sequence[TimedRecord], telemetry: Sequence[TimedRecord]) -> None:
    if not wrench and not telemetry:
        return
    figure, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=False)
    if wrench:
        start = wrench[0].time_sec
        time = [record.time_sec - start for record in wrench]
        values = [_array(record.payload, "wrench4", 4) for record in wrench]
        for index, name in enumerate(("surge", "sway", "heave", "yaw")):
            axes[0].plot(time, [value[index] if value is not None else math.nan for value in values], label=name)
        axes[0].legend(ncol=4)
    axes[0].set_ylabel("teleop wrench")
    axes[0].grid(alpha=0.25)
    if telemetry:
        start = telemetry[0].time_sec
        time = [record.time_sec - start for record in telemetry]
        values = [_array(record.payload, "rpm", 8) for record in telemetry]
        for index in range(8):
            axes[1].plot(time, [value[index] if value is not None else math.nan for value in values], label=f"M{index + 1}")
        axes[1].legend(ncol=4, fontsize=8)
    axes[1].set_ylabel("RPM")
    axes[1].set_xlabel("time since first sample [s]")
    axes[1].grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)


def _write_summary_csv(path: Path, report: dict[str, Any]) -> None:
    rows = []
    for stream, summary in report["streams"].items():
        rows.append({"stream": stream, **summary})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stream", "samples", "rate_hz", "period_median_sec", "period_p95_sec"])
        writer.writeheader()
        writer.writerows(rows)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze native-rate and aligned FinsROV teleoperation trajectory data.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--jump-z-score", type=float, default=8.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    session_dir = args.session_dir.resolve()
    raw_dir = session_dir / "raw_streams"
    if not raw_dir.is_dir():
        raise SystemExit(f"raw streams not found: {raw_dir}; run export_teleop_trajectory first")
    streams = {path.stem: _records(_load_rows(path)) for path in raw_dir.glob("*.parquet")}
    report = {"streams": {name: rate_summary(records) for name, records in sorted(streams.items())}}
    events: list[dict[str, Any]] = []
    events.extend(_candidate_jumps(streams.get("vision_refracted", []), "position_xyz", 3, "vision_position", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("fused_pose", []), "position_xyz", 3, "fused_position", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("hardware_depth_raw", []), "position_xyz", 3, "raw_depth", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("fused_depth", []), "position_xyz", 3, "fused_depth", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("hardware_imu_raw", []), "angular_velocity_xyz", 3, "raw_imu_angular_velocity", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("hardware_imu_raw", []), "linear_acceleration_xyz", 3, "raw_imu_acceleration", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("fused_imu", []), "angular_velocity_xyz", 3, "fused_imu_angular_velocity", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("fused_imu", []), "linear_acceleration_xyz", 3, "fused_imu_acceleration", args.jump_z_score))
    events.extend(_candidate_jumps(streams.get("fused_dvl", []), "linear_velocity_xyz", 3, "fused_velocity", args.jump_z_score))
    events.extend(_sequence_gaps(streams.get("hardware_telemetry", [])))
    status_records = streams.get("state_status", [])
    for event in events:
        status = _nearest_status(status_records, float(event["event_time_sec"]))
        event["vision_mode"] = str(status.get("vision_mode", ""))
        event["state_ready"] = bool(status.get("ready", False))
        event["reject_reason"] = str(status.get("reject_reason", ""))
    events.sort(key=lambda event: float(event["event_time_sec"]))
    pq.write_table(pa.Table.from_pylist(events) if events else pa.table({"event_time_sec": pa.array([], type=pa.float64())}), session_dir / "jump_events.parquet", compression="zstd")
    bounds_reports: dict[str, dict[str, Any]] = {}
    out_of_bounds_events: list[dict[str, Any]] = []
    for stream_name in ("fused_pose", "vision_refracted"):
        bounds_report, stream_events = _pool_bounds_report(streams.get(stream_name, []), stream_name)
        bounds_reports[stream_name] = bounds_report
        out_of_bounds_events.extend(stream_events)
    for event in out_of_bounds_events:
        status = _nearest_status(status_records, float(event["event_time_sec"]))
        event["vision_mode"] = str(status.get("vision_mode", ""))
        event["state_ready"] = bool(status.get("ready", False))
        event["reject_reason"] = str(status.get("reject_reason", ""))
    out_of_bounds_events.sort(key=lambda event: float(event["event_time_sec"]))
    pq.write_table(
        pa.Table.from_pylist(out_of_bounds_events)
        if out_of_bounds_events
        else pa.table({"event_time_sec": pa.array([], type=pa.float64())}),
        session_dir / "out_of_bounds_events.parquet",
        compression="zstd",
    )
    report["jump_event_count"] = len(events)
    report["jump_event_count_by_signal"] = {
        signal: sum(1 for event in events if event["signal"] == signal)
        for signal in sorted({str(event["signal"]) for event in events})
    }
    report["pool_bounds"] = {
        "frame_id": "pool_world",
        "definition": "x and z have closed min/max bounds; y has only the configured lower bound.",
        "limits_m": {axis_name: dict(bounds) for axis_name, bounds in POOL_BOUNDS_M.items()},
        "streams": bounds_reports,
        "out_of_bounds_event_count": len(out_of_bounds_events),
    }
    (session_dir / "quality_report.json").write_text(
        json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    _write_summary_csv(session_dir / "trajectory_summary.csv", report)
    plots = session_dir / "plots"
    plots.mkdir(exist_ok=True)
    _plot_pose(plots / "vision_vs_fused_pose.png", streams.get("fused_pose", []), streams.get("vision_refracted", []))
    _plot_trajectory_xyz(plots / "trajectory_xyz.png", streams.get("fused_pose", []))
    _plot_trajectory_3d_pool(plots / "trajectory_3d_pool.png", streams.get("fused_pose", []))
    _plot_depth(plots / "raw_vs_fused_depth.png", streams.get("hardware_depth_raw", []), streams.get("fused_depth", []))
    _plot_action_rpm(plots / "teleop_wrench_and_rpm.png", streams.get("teleop_wrench", []), streams.get("hardware_telemetry", []))
    print(f"analyzed {len(streams)} raw streams; found {len(events)} candidate jump events")


if __name__ == "__main__":
    main()
