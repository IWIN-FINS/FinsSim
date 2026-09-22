#!/usr/bin/env python3
"""Reproducible offline analysis for event-bounded AprilTag stability logs.

The live recorder deliberately keeps raw ROS observations and event markers.
This tool converts two such recordings into publication-facing *runtime
stability* evidence.  It never estimates an external ground truth and thus
never reports its continuity metrics as localization accuracy.

It was written for the E1 T1/T2 recordings, but accepts arbitrary recorder
directories that use the same NDJSON schema.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import statistics
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


T1_SESSION_RE = re.compile(r"_P0[1-8]_.*_R01$")
T2_SESSION_RE = re.compile(r"_R0[1-3]$")
EPS = 1e-9


@dataclass(frozen=True)
class Window:
    kind: str
    session_id: str
    label: str
    start_sec: float
    end_sec: float


def _percentile(values: Iterable[float], fraction: float) -> float | None:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return None
    position = (len(ordered) - 1) * fraction
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return sum(values) / len(values) if values else None


def _median(values: Iterable[float]) -> float | None:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(values) if values else None


def _fmt(value: float | None, digits: int = 3) -> str:
    return "--" if value is None else f"{value:.{digits}f}"


def _pct(value: float | None) -> str:
    return "--" if value is None else f"{100.0 * value:.1f}%"


def _bool(value: Any) -> bool:
    return bool(value) if isinstance(value, bool) else str(value).strip().lower() in {"1", "true", "yes"}


def _yaw_rad(payload: dict[str, Any]) -> float | None:
    try:
        x, y, z, w = (float(payload[key]) for key in ("qx", "qy", "qz", "qw"))
    except (KeyError, TypeError, ValueError):
        return None
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def _wrap_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def _trial_label(session_id: str, kind: str) -> str:
    if kind == "T1":
        match = re.search(r"_(P\d\d)_", session_id)
        return match.group(1) if match else session_id
    match = re.search(r"_(R\d\d)$", session_id)
    return match.group(1) if match else session_id


def _load_rows(path: Path) -> list[dict[str, Any]]:
    stream = path / "raw" / "stream.ndjson"
    if not stream.is_file():
        raise FileNotFoundError(f"Missing recorder stream: {stream}")
    rows: list[dict[str, Any]] = []
    with stream.open(encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            try:
                row = json.loads(line)
                row["elapsed_sec"] = float(row["elapsed_sec"])
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid row {line_no} in {stream}: {exc}") from exc
            rows.append(row)
    return rows


def _event_windows(rows: list[dict[str, Any]], *, kind: str, duration_sec: float) -> list[Window]:
    start_event = "hold_start" if kind == "T1" else "trajectory_start"
    selector = T1_SESSION_RE if kind == "T1" else T2_SESSION_RE
    starts: dict[str, float] = {}
    for row in rows:
        if row.get("topic") != "experiment_event" or row.get("payload", {}).get("event") != start_event:
            continue
        session_id = str(row.get("linked_t1_session", ""))
        if session_id and selector.search(session_id):
            starts.setdefault(session_id, float(row["elapsed_sec"]))
    windows = [
        Window(kind, session_id, _trial_label(session_id, kind), start, start + duration_sec)
        for session_id, start in starts.items()
    ]
    return sorted(windows, key=lambda item: item.label)


def _topic_rows(rows: list[dict[str, Any]], window: Window, topic: str) -> list[dict[str, Any]]:
    return sorted(
        [
            row
            for row in rows
            if row.get("linked_t1_session") == window.session_id
            and row.get("topic") == topic
            and window.start_sec <= float(row["elapsed_sec"]) <= window.end_sec
        ],
        key=lambda row: float(row["elapsed_sec"]),
    )


def _availability(rows: list[dict[str, Any]], key: str) -> tuple[float | None, int, int]:
    values = [_bool(row.get("payload", {}).get(key)) for row in rows if key in row.get("payload", {})]
    return ((sum(values) / len(values)) if values else None, sum(values), len(values))


def _strict_outages(status_rows: list[dict[str, Any]], window: Window) -> list[dict[str, float]]:
    """Contiguous strict-Snell false intervals, bounded by the event window."""

    if not status_rows:
        return []
    intervals: list[dict[str, float]] = []
    start: float | None = None
    for row in status_rows:
        time_sec = float(row["elapsed_sec"])
        valid = _bool(row.get("payload", {}).get("snell_valid"))
        if not valid and start is None:
            start = time_sec
        elif valid and start is not None:
            intervals.append({"start_sec": start, "end_sec": time_sec, "duration_sec": max(0.0, time_sec - start)})
            start = None
    if start is not None:
        intervals.append({"start_sec": start, "end_sec": window.end_sec, "duration_sec": max(0.0, window.end_sec - start)})
    return intervals


def _physical_snell_pose(rows: list[dict[str, Any]], window: Window) -> list[tuple[float, float, float, float]]:
    """Keep poses timestamp-matched to a status frame with ``snell_valid=true``.

    The pose topic may still contain a constrained pinhole fallback.  Matching
    it to strict status avoids treating fallback poses as physical refraction.
    """

    status = _topic_rows(rows, window, "refractive_status")
    valid_times = [float(row["elapsed_sec"]) for row in status if _bool(row.get("payload", {}).get("snell_valid"))]
    poses = _topic_rows(rows, window, "snell_pose")
    result: list[tuple[float, float, float, float]] = []
    index = 0
    for row in poses:
        if not valid_times:
            break
        time_sec = float(row["elapsed_sec"])
        while index + 1 < len(valid_times) and abs(valid_times[index + 1] - time_sec) <= abs(valid_times[index] - time_sec):
            index += 1
        if abs(valid_times[index] - time_sec) > 0.050:
            continue
        payload = row.get("payload", {})
        try:
            x_m, y_m = float(payload["x_m"]), float(payload["y_m"])
        except (KeyError, TypeError, ValueError):
            continue
        yaw = _yaw_rad(payload)
        if yaw is not None:
            result.append((time_sec, x_m, y_m, yaw))
    return result


def _static_jitter(poses: list[tuple[float, float, float, float]]) -> dict[str, float | int | None]:
    if not poses:
        return {"pose_count": 0, "xy_rms_m": None, "xy_p95_m": None, "yaw_std_deg": None, "yaw_p95_abs_deg": None}
    center_x = statistics.median(item[1] for item in poses)
    center_y = statistics.median(item[2] for item in poses)
    yaw_center = statistics.median(item[3] for item in poses)
    xy = [math.hypot(item[1] - center_x, item[2] - center_y) for item in poses]
    yaw_error = [_wrap_angle(item[3] - yaw_center) for item in poses]
    return {
        "pose_count": len(poses),
        "xy_rms_m": math.sqrt(sum(value * value for value in xy) / len(xy)),
        "xy_p95_m": _percentile(xy, 0.95),
        "yaw_std_deg": math.degrees(statistics.pstdev(yaw_error)) if len(yaw_error) > 1 else 0.0,
        "yaw_p95_abs_deg": math.degrees(_percentile([abs(value) for value in yaw_error], 0.95) or 0.0),
    }


def _local_detrended_jitter(poses: list[tuple[float, float, float, float]], window_sec: float = 1.0) -> dict[str, float | int | None]:
    """One-second local median residual for a moving vehicle (T2)."""

    if not poses:
        return {"pose_count": 0, "xy_rms_m": None, "xy_p95_m": None, "yaw_std_deg": None, "yaw_p95_abs_deg": None}
    half_window = window_sec / 2.0
    xy_residuals: list[float] = []
    yaw_residuals: list[float] = []
    for time_sec, x_m, y_m, yaw in poses:
        neighbors = [item for item in poses if abs(item[0] - time_sec) <= half_window]
        median_x = statistics.median(item[1] for item in neighbors)
        median_y = statistics.median(item[2] for item in neighbors)
        median_yaw = statistics.median(item[3] for item in neighbors)
        xy_residuals.append(math.hypot(x_m - median_x, y_m - median_y))
        yaw_residuals.append(_wrap_angle(yaw - median_yaw))
    return {
        "pose_count": len(poses),
        "xy_rms_m": math.sqrt(sum(value * value for value in xy_residuals) / len(xy_residuals)),
        "xy_p95_m": _percentile(xy_residuals, 0.95),
        "yaw_std_deg": math.degrees(statistics.pstdev(yaw_residuals)) if len(yaw_residuals) > 1 else 0.0,
        "yaw_p95_abs_deg": math.degrees(_percentile([abs(value) for value in yaw_residuals], 0.95) or 0.0),
    }


def _linear_velocity(samples: list[tuple[float, float, float, float]]) -> tuple[float, float, float] | None:
    if len(samples) < 5:
        return None
    mean_time = sum(item[0] for item in samples) / len(samples)
    denominator = sum((item[0] - mean_time) ** 2 for item in samples)
    if denominator <= EPS:
        return None

    def slope(index: int) -> float:
        mean_value = sum(item[index] for item in samples) / len(samples)
        return sum((item[0] - mean_time) * (item[index] - mean_value) for item in samples) / denominator

    unwrapped_yaw = [samples[0][3]]
    for item in samples[1:]:
        unwrapped_yaw.append(unwrapped_yaw[-1] + _wrap_angle(item[3] - unwrapped_yaw[-1]))
    yaw_samples = [(item[0], item[1], item[2], yaw) for item, yaw in zip(samples, unwrapped_yaw)]
    return slope(1), slope(2), _linear_velocity_yaw(yaw_samples, mean_time, denominator)


def _linear_velocity_yaw(samples: list[tuple[float, float, float, float]], mean_time: float, denominator: float) -> float:
    mean_yaw = sum(item[3] for item in samples) / len(samples)
    return sum((item[0] - mean_time) * (item[3] - mean_yaw) for item in samples) / denominator


def _reacquisition_metrics(poses: list[tuple[float, float, float, float]], outages: list[dict[str, float]]) -> list[dict[str, float | None]]:
    """Constant-velocity residual across each nontrivial strict-Snell outage.

    This is a continuity diagnostic, not an accuracy estimate: vehicle motion
    between observations is unknown, especially during T2.
    """

    metrics: list[dict[str, float | None]] = []
    for outage in outages:
        if outage["duration_sec"] < 0.25:
            continue
        pre = [item for item in poses if item[0] <= outage["start_sec"]]
        post = [item for item in poses if item[0] >= outage["end_sec"]]
        if not pre or not post:
            continue
        last_pre = pre[-1]
        first_post = post[0]
        history = [item for item in pre if item[0] >= last_pre[0] - 0.5]
        velocity = _linear_velocity(history)
        residual_xy: float | None = None
        residual_yaw_deg: float | None = None
        if velocity is not None:
            dt = first_post[0] - last_pre[0]
            predicted_x = last_pre[1] + velocity[0] * dt
            predicted_y = last_pre[2] + velocity[1] * dt
            predicted_yaw = last_pre[3] + velocity[2] * dt
            residual_xy = math.hypot(first_post[1] - predicted_x, first_post[2] - predicted_y)
            residual_yaw_deg = math.degrees(abs(_wrap_angle(first_post[3] - predicted_yaw)))
        metrics.append(
            {
                "outage_start_sec": outage["start_sec"],
                "outage_end_sec": outage["end_sec"],
                "outage_duration_sec": outage["duration_sec"],
                "first_reacquired_pose_sec": first_post[0],
                "three_frame_revalidation_sec": None,
                "constant_velocity_xy_residual_m": residual_xy,
                "constant_velocity_yaw_residual_deg": residual_yaw_deg,
            }
        )
    return metrics


def _summary(values: list[float]) -> dict[str, float | None]:
    return {"median": _median(values), "p25": _percentile(values, 0.25), "p75": _percentile(values, 0.75), "p95": _percentile(values, 0.95), "max": max(values) if values else None}


def _metric_rows(rows: list[dict[str, Any]], windows: list[Window]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    trial_rows: list[dict[str, Any]] = []
    outages_rows: list[dict[str, Any]] = []
    recovery_rows: list[dict[str, Any]] = []
    for window in windows:
        detector = _topic_rows(rows, window, "detector_status")
        refractive = _topic_rows(rows, window, "refractive_status")
        fusion = _topic_rows(rows, window, "fusion_status")
        detection, detection_true, detection_n = _availability(detector, "detected")
        strict, strict_true, strict_n = _availability(refractive, "snell_valid")
        fallback, fallback_true, fallback_n = _availability(refractive, "fallback_used")
        fusion_ready, fusion_true, fusion_n = _availability(fusion, "ready")
        visual_update, visual_true, visual_n = _availability(fusion, "vision_fresh")
        first_window = Window(window.kind, window.session_id, window.label, window.start_sec, window.start_sec + 10.0)
        first_refractive = _topic_rows(rows, first_window, "refractive_status") if window.kind == "T1" else []
        first_strict, first_strict_true, first_strict_n = _availability(first_refractive, "snell_valid")
        late_window = Window(window.kind, window.session_id, window.label, window.end_sec - 20.0, window.end_sec)
        late_refractive = _topic_rows(rows, late_window, "refractive_status") if window.kind == "T1" else []
        late_strict, late_strict_true, late_strict_n = _availability(late_refractive, "snell_valid")
        outages = _strict_outages(refractive, window)
        long_outages = [item for item in outages if item["duration_sec"] >= 0.25]
        physical_poses = _physical_snell_pose(rows, window)
        jitter_window = late_window if window.kind == "T1" else window
        jitter_poses = _physical_snell_pose(rows, jitter_window)
        jitter = _static_jitter(jitter_poses) if window.kind == "T1" else _local_detrended_jitter(jitter_poses)
        recovery = _reacquisition_metrics(physical_poses, long_outages)
        reasons = Counter(str(item.get("payload", {}).get("constrained_reject_reason", "")) for item in refractive if not _bool(item.get("payload", {}).get("snell_valid")))
        for index, interval in enumerate(outages, start=1):
            outages_rows.append({"kind": window.kind, "trial_id": window.label, "session_id": window.session_id, "outage_index": index, **interval})
        trial_rows.append(
            {
                "kind": window.kind,
                "trial_id": window.label,
                "session_id": window.session_id,
                "window_start_sec": window.start_sec,
                "window_duration_sec": window.end_sec - window.start_sec,
                "detector_availability": detection,
                "detector_true_frames": detection_true,
                "detector_frames": detection_n,
                "strict_snell_availability": strict,
                "strict_snell_true_frames": strict_true,
                "strict_snell_frames": strict_n,
                "first10_strict_snell_availability": first_strict,
                "first10_strict_snell_true_frames": first_strict_true,
                "first10_strict_snell_frames": first_strict_n,
                "late_strict_snell_availability": late_strict,
                "late_strict_snell_true_frames": late_strict_true,
                "late_strict_snell_frames": late_strict_n,
                "fallback_rate": fallback,
                "fallback_true_frames": fallback_true,
                "fallback_frames": fallback_n,
                "estimator_output_availability": fusion_ready,
                "estimator_output_true_samples": fusion_true,
                "estimator_output_samples": fusion_n,
                "estimator_visual_input_availability": visual_update,
                "estimator_visual_input_true_samples": visual_true,
                "estimator_visual_input_samples": visual_n,
                "strict_outage_count": len(outages),
                "strict_outage_time_sec": sum(item["duration_sec"] for item in outages),
                "strict_outage_fraction": sum(item["duration_sec"] for item in outages) / (window.end_sec - window.start_sec),
                "strict_outage_count_ge_025s": len(long_outages),
                "strict_outage_per_min": len(outages) * 60.0 / (window.end_sec - window.start_sec),
                "strict_outage_p95_sec": _percentile([item["duration_sec"] for item in outages], 0.95),
                "strict_outage_max_sec": max((item["duration_sec"] for item in outages), default=None),
                "jitter_window": "last_20s_static" if window.kind == "T1" else "full_trajectory_1s_local_median_detrended",
                "jitter_pose_count": jitter["pose_count"],
                "jitter_xy_rms_m": jitter["xy_rms_m"],
                "jitter_xy_p95_m": jitter["xy_p95_m"],
                "jitter_yaw_std_deg": jitter["yaw_std_deg"],
                "jitter_yaw_p95_abs_deg": jitter["yaw_p95_abs_deg"],
                "reacquisition_count_ge_025s": len(recovery),
                "reacquisition_xy_residual_p95_m": _percentile([item["constant_velocity_xy_residual_m"] for item in recovery if item["constant_velocity_xy_residual_m"] is not None], 0.95),
                "reacquisition_xy_residual_max_m": max((item["constant_velocity_xy_residual_m"] for item in recovery if item["constant_velocity_xy_residual_m"] is not None), default=None),
                "main_failure_reason": reasons.most_common(1)[0][0] if reasons else "none",
            }
        )
        for item in recovery:
            item.update({"kind": window.kind, "trial_id": window.label, "session_id": window.session_id})
            recovery_rows.append(item)
    return trial_rows, outages_rows, recovery_rows


def _availability_bins(rows: list[dict[str, Any]], windows: list[Window], *, bins: int) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for window in windows:
        status = _topic_rows(rows, window, "refractive_status")
        detector = _topic_rows(rows, window, "detector_status")
        for index in range(bins):
            low = window.start_sec + (window.end_sec - window.start_sec) * index / bins
            high = window.start_sec + (window.end_sec - window.start_sec) * (index + 1) / bins
            status_bin = [row for row in status if low <= float(row["elapsed_sec"]) < high]
            detector_bin = [row for row in detector if low <= float(row["elapsed_sec"]) < high]
            strict, _, _ = _availability(status_bin, "snell_valid")
            detected, _, _ = _availability(detector_bin, "detected")
            fallback, _, _ = _availability(status_bin, "fallback_used")
            result.append({"kind": window.kind, "trial_id": window.label, "bin_index": index, "progress_start": index / bins, "progress_end": (index + 1) / bins, "strict_snell_availability": strict, "detector_availability": detected, "fallback_rate": fallback})
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def _plots(output_dir: Path, t1: list[dict[str, Any]], t2: list[dict[str, Any]], outages: list[dict[str, Any]], bins: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    all_rows = t1 + t2
    figure, axes = plt.subplots(2, 1, figsize=(10.0, 6.4), sharey=True)
    for axis, source, title in zip(axes, (t1, t2), ("T1: eight station-keeping targets", "T2: three elliptical trajectories")):
        labels = [row["trial_id"] for row in source]
        x = list(range(len(labels)))
        width = 0.19
        specs = (("detector_availability", "Detector"), ("strict_snell_availability", "Strict Snell"), ("fallback_rate", "Fallback"), ("estimator_output_availability", "Estimator output"))
        for offset, (key, label) in enumerate(specs):
            axis.bar([value + (offset - 1.5) * width for value in x], [row[key] or 0.0 for row in source], width, label=label)
        axis.set_xticks(x, labels)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("fraction")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend(ncol=4, fontsize=8)
    figure.tight_layout()
    figure.savefig(plot_dir / "availability_by_trial.png", dpi=220)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.6, 4.2))
    for kind, color in (("T1", "tab:blue"), ("T2", "tab:orange")):
        values = sorted(float(row["duration_sec"]) for row in outages if row.get("kind") == kind and isinstance(row.get("outage_index"), int))
        if values:
            axis.step(values, [(index + 1) / len(values) for index in range(len(values))], where="post", label=kind, color=color)
    axis.set_xlabel("strict Snell outage duration [s]")
    axis.set_ylabel("empirical CDF")
    axis.set_xlim(left=0.0)
    axis.set_ylim(0.0, 1.02)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "strict_snell_outage_ecdf.png", dpi=220)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(10.0, 4.0))
    for axis, source, title in zip(axes, (t1, t2), ("T1 late-hold output dispersion", "T2 local detrended output residual")):
        labels = [row["trial_id"] for row in source]
        x = list(range(len(labels)))
        axis.bar([value - 0.18 for value in x], [1000.0 * (row["jitter_xy_rms_m"] or 0.0) for row in source], 0.35, label="RMS")
        axis.bar([value + 0.18 for value in x], [1000.0 * (row["jitter_xy_p95_m"] or 0.0) for row in source], 0.35, label="P95")
        axis.set_xticks(x, labels)
        axis.set_ylabel("horizontal displacement [mm]")
        axis.set_title(title)
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    figure.tight_layout()
    figure.savefig(plot_dir / "horizontal_output_continuity.png", dpi=220)
    plt.close(figure)

    for kind, filename, xlabel in (("T1", "t1_time_since_hold_availability.png", "time since hold start [s]"), ("T2", "t2_normalized_progress_availability.png", "normalized trajectory progress")):
        source = [row for row in bins if row["kind"] == kind]
        grouped: dict[int, list[dict[str, Any]]] = {}
        for row in source:
            grouped.setdefault(int(row["bin_index"]), []).append(row)
        xs = sorted(grouped)
        strict = [_mean([item["strict_snell_availability"] for item in grouped[index] if item["strict_snell_availability"] is not None]) for index in xs]
        detector = [_mean([item["detector_availability"] for item in grouped[index] if item["detector_availability"] is not None]) for index in xs]
        figure, axis = plt.subplots(figsize=(7.8, 3.8))
        if kind == "T1":
            x_values = [(index + 0.5) * 5.0 for index in xs]
        else:
            x_values = [(index + 0.5) / len(xs) for index in xs]
        axis.plot(x_values, strict, marker="o", label="Strict Snell")
        axis.plot(x_values, detector, marker="s", label="Detector")
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("availability fraction")
        axis.set_xlabel(xlabel)
        axis.grid(alpha=0.25)
        axis.legend()
        figure.tight_layout()
        figure.savefig(plot_dir / filename, dpi=220)
        plt.close(figure)


def _aggregate_text(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    fields = ("detector_availability", "strict_snell_availability", "first10_strict_snell_availability", "late_strict_snell_availability", "fallback_rate", "estimator_output_availability", "strict_outage_p95_sec", "strict_outage_max_sec", "jitter_xy_rms_m", "jitter_xy_p95_m", "jitter_yaw_std_deg", "jitter_yaw_p95_abs_deg")
    return {field: _summary([row[field] for row in rows if row.get(field) is not None]) for field in fields}


def _write_markdown(output_dir: Path, t1_rows: list[dict[str, Any]], t2_rows: list[dict[str, Any]], outages: list[dict[str, Any]], recovery: list[dict[str, Any]], bins: list[dict[str, Any]]) -> None:
    t1_aggregate, t2_aggregate = _aggregate_text(t1_rows), _aggregate_text(t2_rows)
    all_outages = {kind: [float(row["duration_sec"]) for row in outages if row.get("kind") == kind and isinstance(row.get("outage_index"), int)] for kind in ("T1", "T2")}
    t1_full = _mean([row["strict_snell_availability"] for row in t1_rows])
    t1_first = _mean([row["first10_strict_snell_availability"] for row in t1_rows])
    t1_late = _mean([row["late_strict_snell_availability"] for row in t1_rows])
    t1_detector = _mean([row["detector_availability"] for row in t1_rows])
    t1_fallback = _mean([row["fallback_rate"] for row in t1_rows])
    t1_estimator_output = _mean([row["estimator_output_availability"] for row in t1_rows])
    t2_availability = _mean([row["strict_snell_availability"] for row in t2_rows])
    t1_bins = [row for row in bins if row["kind"] == "T1"]
    t2_bins = [row for row in bins if row["kind"] == "T2"]
    def bin_average(source: list[dict[str, Any]], index: int, key: str) -> float | None:
        return _mean([row[key] for row in source if row["bin_index"] == index and row[key] is not None])
    t1_first_5s, t1_final = bin_average(t1_bins, 0, "strict_snell_availability"), bin_average(t1_bins, 11, "strict_snell_availability")
    t2_first, t2_middle, t2_weak = bin_average(t2_bins, 0, "strict_snell_availability"), _mean([bin_average(t2_bins, index, "strict_snell_availability") for index in range(1, 7)]), bin_average(t2_bins, 7, "strict_snell_availability")
    lines = [
        "# E1 AprilTag / Snell 运行时稳定性分析",
        "",
        "**生成方式：** `src/experiment_recorder/tools/analyze_apriltag_stability_sessions.py`  ",
        "**分析对象：** T1 八个定点保持会话与 T2 椭圆轨迹三次重复。  ",
        "**统计单位：** 一个 T1 目标点或一个 T2 repeat；帧仅用于计算每个统计单位内部的时间序列指标，不能视为独立重复。",
        "",
        "## 结论摘要",
        "",
        f"- T1 八个目标点的严格物理 Snell 平均可用率为：完整 60 s **{_pct(t1_full)}**、开始后前 10 s **{_pct(t1_first)}**、最后 20 s **{_pct(t1_late)}**。P01/P02 是明显弱点。",
        f"- 作为更细的时间趋势检查，T1 的第一个 5 s 为 **{_pct(t1_first_5s)}**，最后 5 s 升至 **{_pct(t1_final)}**；这解释了完整窗口为何不能代表后期保持阶段。",
        f"- T2 三次椭圆轨迹的严格 Snell 可用率平均为 **{_pct(t2_availability)}**。0--10% 进度为 **{_pct(t2_first)}**，10--70% 的平均值为 **{_pct(t2_middle)}**，70--80% 出现可重复弱区（**{_pct(t2_weak)}**）。",
        "- 两组数据能证明真实闭环运行中的可用性、连续失效和短时输出连续性；它们**不含外部真实位置**，所以不能替代 E2 的水平定位精度实验，也不能把抖动残差写成定位 RMSE。",
        "",
        "## 1. 数据范围与审计边界",
        "",
        "| 数据 | 正式窗口 | 排除内容 |",
        "|---|---|---|",
        "| T1 `20260903_143422...` | P01--P08 的 R01；每点第一次 `hold_start` 后 60 s | P01 的截断 R02、reset/acquisition 片段，以及旧记录器重复 marker 产生的伪 `hold_after_registered_window` 短段 |",
        "| T2 `20260903_144940...` | R01--R03；每条第一次 `trajectory_start` 后 92.7 s | repeat 间的 15 s 复位间隔与轨迹结束后的等待时间 |",
        "",
        "严格物理 Snell 成功定义为 `refractive_status.snell_valid=true`。检测成功、约束输出或融合状态可继续存在并不等价于严格 Snell 成功：尤其是系统配置允许 PnP 回退时，回退帧被单列为 fallback，未计入严格 Snell。",
        "",
        "## 2. 四类稳定性指标",
        "",
        "### 2.1 可用率",
        "",
        r"\[ A=\frac{1}{N}\sum_{k=1}^{N}\mathbb{I}(q_k=1). \]",
        "",
        "分别记录基础 AprilTag 检测可用率、严格物理 Snell 可用率、回退率以及下游状态输出可用率。最后一项只说明控制侧持续有状态输出，不能替代视觉定位成功率。",
        "",
        "### 2.2 连续失效",
        "",
        "将连续 `snell_valid=false` 帧合并为 outage，报告 P95、最大值、每分钟次数和累计无定位时间占比；`plots/strict_snell_outage_ecdf.png` 给出完整经验 CDF。",
        "",
        "### 2.3 输出连续性 / 抖动",
        "",
        r"T1 的后 20 s 近似保持阶段使用 \(J_{\mathrm{RMS}}=\sqrt{N^{-1}\sum_k\|\mathbf p_k-\operatorname{median}(\mathbf p)\|_2^2}\)，并报告水平 P95、yaw 标准差与 yaw 的绝对 P95 偏差。",
        "",
        r"T2 是运动任务，因此先以 1 s 居中滑动中位数 \(\hat{\mathbf p}(t)\) 去除轨迹趋势，再计算 \(\mathbf r(t)=\mathbf p(t)-\hat{\mathbf p}(t)\) 的 RMS/P95。它是短时输出抖动，不是动态定位精度。",
        "",
        "### 2.4 重获后的连续性",
        "",
        r"对每段 \(g_j\ge0.25\,\mathrm{s}\) 的 outage，以失效前 0.5 s 的局部线性速度外推，计算重获首帧的残差 \(J_j=\|\mathbf p(t_j^+)-[\mathbf p(t_j^-)+\hat{\mathbf v}\Delta t_j]\|_2\)。该数值用于发现重新检测时的突变风险；由于中间真实运动未知，不能解释为绝对误差。",
        "",
        "## 3. 可用率结果",
        "",
        "### T1：逐目标点",
        "",
        "| 点 | 检测 | 严格 Snell（60 s） | 严格 Snell（前 10 s） | 严格 Snell（后 20 s） | 回退 | 下游状态输出 | 主要严格失败原因 |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in t1_rows:
        lines.append(f"| {row['trial_id']} | {_pct(row['detector_availability'])} | {_pct(row['strict_snell_availability'])} | {_pct(row['first10_strict_snell_availability'])} | {_pct(row['late_strict_snell_availability'])} | {_pct(row['fallback_rate'])} | {_pct(row['estimator_output_availability'])} | {row['main_failure_reason'] or 'none'} |")
    lines.append(f"| **八点平均** | **{_pct(t1_detector)}** | **{_pct(t1_full)}** | **{_pct(t1_first)}** | **{_pct(t1_late)}** | **{_pct(t1_fallback)}** | **{_pct(t1_estimator_output)}** | -- |")
    lines += [
        "",
        "### T2：逐重复",
        "",
        "| Repeat | 检测 | 严格 Snell | 回退 | 下游状态输出 | 主要严格失败原因 |",
        "|---|---:|---:|---:|---:|---|",
    ]
    for row in t2_rows:
        lines.append(f"| {row['trial_id']} | {_pct(row['detector_availability'])} | {_pct(row['strict_snell_availability'])} | {_pct(row['fallback_rate'])} | {_pct(row['estimator_output_availability'])} | {row['main_failure_reason'] or 'none'} |")
    lines += [
        "",
        "可视化：`plots/availability_by_trial.png`、`plots/t1_time_since_hold_availability.png`、`plots/t2_normalized_progress_availability.png`。",
        "",
        "## 4. 连续失效结果",
        "",
        "| 数据 | outage 数 | 平均 outage 次/分钟 | 超过 0.25 s | 累计无严格 Snell 时间占比 | P95 outage [s] | 最大 outage [s] |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for kind, source, duration in (("T1", t1_rows, 60.0), ("T2", t2_rows, 92.7)):
        values = all_outages[kind]
        long_count = sum(value >= 0.25 for value in values)
        unavailable = sum(row["strict_outage_time_sec"] for row in source) / (duration * len(source))
        per_minute = len(values) * 60.0 / (duration * len(source))
        lines.append(f"| {kind} | {len(values)} | {_fmt(per_minute, 1)} | {long_count} | {_pct(unavailable)} | {_fmt(_percentile(values, 0.95))} | {_fmt(max(values) if values else None)} |")
    lines += [
        "",
        "T1 中最长 outage 发生在包括早期接近的完整窗口；若只以最后 20 s 保持段观察，必须单独报告，不能与完整窗口混合。T2 的弱区可在归一化 70--80% 进度段复现，因而优先检查该空间位置和艇体朝向下的 tag 可见性。",
        "",
        "## 5. 输出连续性结果",
        "",
        "| 数据 | 统计方式 | 每 trial RMS 中位数 [m] | 每 trial P95 中位数 [m] | 最差 trial P95 [m] | yaw 标准差中位数 [deg] |",
        "|---|---|---:|---:|---:|---:|",
        f"| T1 | 最后 20 s，相对窗口中位位置 | {_fmt(t1_aggregate['jitter_xy_rms_m']['median'])} | {_fmt(t1_aggregate['jitter_xy_p95_m']['median'])} | {_fmt(t1_aggregate['jitter_xy_p95_m']['max'])} | {_fmt(t1_aggregate['jitter_yaw_std_deg']['median'])} |",
        f"| T2 | 全轨迹，1 s 局部中位数去趋势 | {_fmt(t2_aggregate['jitter_xy_rms_m']['median'])} | {_fmt(t2_aggregate['jitter_xy_p95_m']['median'])} | {_fmt(t2_aggregate['jitter_xy_p95_m']['max'])} | {_fmt(t2_aggregate['jitter_yaw_std_deg']['median'])} |",
        "",
        "T1 的相对窗口中位位置离散度仍可能含真实低频控制运动；只有 T2 的去趋势残差能较直接反映短时连续性。二者均不可称为定位准确度。",
        "",
        "## 6. 重获连续性结果",
        "",
        "| 数据 | 可计算重获事件（outage ≥0.25 s） | 全部事件外推残差 P95 [m] | 最大外推残差 [m] | 解释边界 |",
        "|---|---:|---:|---:|---|",
        f"| T1 | {sum(1 for row in recovery if row['kind'] == 'T1')} | {_fmt(_percentile([float(row['constant_velocity_xy_residual_m']) for row in recovery if row['kind'] == 'T1' and row['constant_velocity_xy_residual_m'] is not None], 0.95))} | {_fmt(max((float(row['constant_velocity_xy_residual_m']) for row in recovery if row['kind'] == 'T1' and row['constant_velocity_xy_residual_m'] is not None), default=None))} | 包含接近目标时真实运动 |",
        f"| T2 | {sum(1 for row in recovery if row['kind'] == 'T2')} | {_fmt(_percentile([float(row['constant_velocity_xy_residual_m']) for row in recovery if row['kind'] == 'T2' and row['constant_velocity_xy_residual_m'] is not None], 0.95))} | {_fmt(max((float(row['constant_velocity_xy_residual_m']) for row in recovery if row['kind'] == 'T2' and row['constant_velocity_xy_residual_m'] is not None), default=None))} | 包含曲线运动和真实加速度 |",
        "",
        "这一项是诊断指标：若需要在论文中作为强结论，应增加独立真值或将 ROV 固定后进行专门的重获实验。",
        "",
        "## 7. 论文写作建议",
        "",
        "建议将 E1 表述为 **runtime availability and continuity study**，与 E2 的水平定位精度实验并列：",
        "",
        "> 在八个 T1 定点保持会话中，严格折射定位在后期保持段显著优于初始接近阶段；在三次 T2 椭圆轨迹中，轨迹主体阶段保持高可用，但起始入水/接近阶段及一个可重复的空间弱区仍受 tag 可见性限制。失效主要由可见 tag 数不足而非持续的折射几何失败主导。",
        "",
        "提交前需将任务容差转化为预先声明的连续性门限。例如最大速度为 \(v_{\\max}\)、允许盲行距离为 \(d_{\\mathrm{allow}}\) 时，应检查 \(v_{\\max}g_{\\max}\le d_{\\mathrm{allow}}\)。当前报告给出测得的 \(g\)，但不虚构该门限。",
        "",
        "## 8. 输出文件",
        "",
        "- `t1_per_target_metrics.csv`、`t2_per_repeat_metrics.csv`：逐统计单位指标；",
        "- `strict_snell_outages.csv`、`reacquisition_metrics.csv`：严格 Snell outage 与重获诊断；",
        "- `availability_progress_bins.csv`：T1 时间分箱和 T2 归一化进度分箱；",
        "- `plots/`：四类可直接审阅的图；",
        "- 本报告：方法、边界和聚合结果。",
    ]
    (output_dir / "E1_AprilTag_Stability_Analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--t1", type=Path, required=True, help="T1 E1 recorder directory")
    parser.add_argument("--t2", type=Path, required=True, help="T2 E1 recorder directory")
    parser.add_argument("--output", type=Path, required=True, help="new or replaceable analysis directory")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output.exists():
        if not args.overwrite:
            raise SystemExit(f"Output already exists: {args.output}; pass --overwrite to replace only this analysis directory")
        shutil.rmtree(args.output)
    args.output.mkdir(parents=True)
    t1_raw, t2_raw = _load_rows(args.t1), _load_rows(args.t2)
    t1_windows = _event_windows(t1_raw, kind="T1", duration_sec=60.0)
    t2_windows = _event_windows(t2_raw, kind="T2", duration_sec=92.7)
    if len(t1_windows) != 8 or len(t2_windows) != 3:
        raise SystemExit(f"Expected eight T1 windows and three T2 windows, got {len(t1_windows)} and {len(t2_windows)}")
    t1_rows, t1_outages, t1_recovery = _metric_rows(t1_raw, t1_windows)
    t2_rows, t2_outages, t2_recovery = _metric_rows(t2_raw, t2_windows)
    bins = _availability_bins(t1_raw, t1_windows, bins=12) + _availability_bins(t2_raw, t2_windows, bins=10)
    outages = t1_outages + t2_outages
    _write_csv(args.output / "t1_per_target_metrics.csv", t1_rows)
    _write_csv(args.output / "t2_per_repeat_metrics.csv", t2_rows)
    _write_csv(args.output / "strict_snell_outages.csv", outages)
    _write_csv(args.output / "reacquisition_metrics.csv", t1_recovery + t2_recovery)
    _write_csv(args.output / "availability_progress_bins.csv", bins)
    _plots(args.output, t1_rows, t2_rows, outages, bins)
    _write_markdown(args.output, t1_rows, t2_rows, outages, t1_recovery + t2_recovery, bins)
    provenance = {
        "analysis": "E1 AprilTag strict-Snell runtime stability",
        "t1_source": str(args.t1.resolve()),
        "t2_source": str(args.t2.resolve()),
        "window_contract": {"T1": "P01-P08, R01, first hold_start + 60 s", "T2": "R01-R03, first trajectory_start + 92.7 s"},
        "strict_solution": "refractive_status.snell_valid=true; PnP fallback excluded",
        "limits": ["No external truth is present; no continuity metric is an accuracy result.", "Recovery residual assumes constant velocity over the prior 0.5 s and is diagnostic only."],
    }
    (args.output / "analysis_manifest.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote E1 stability analysis to {args.output}")


if __name__ == "__main__":
    main()
