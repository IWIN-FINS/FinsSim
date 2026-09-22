"""Offline analysis for the independent E8 disturbance-recovery protocol."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from .analysis import _angle_error, _select_control_pose_topic, _write_csv
from .manifest import now_iso, read_json, write_json


def _target(metadata: dict[str, Any]) -> tuple[float, float, float, float] | None:
    raw = metadata.get("target_controller_world")
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    try:
        return float(raw[0]), float(raw[1]), float(raw[2]), math.radians(float(raw[3]))
    except (TypeError, ValueError):
        return None


def _event_time(events: list[dict[str, Any]], name: str) -> float | None:
    matches = [float(event["timestamp_sec"]) for event in events if event.get("event") == name]
    return min(matches) if matches else None


def _as_float(mapping: dict[str, Any], key: str, default: float) -> float:
    try:
        value = float(mapping.get(key, default))
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def _rows(
    poses: list[dict[str, Any]],
    *,
    target: tuple[float, float, float, float],
    detection_time: float | None,
    normalisation: dict[str, float],
) -> list[dict[str, Any]]:
    tx, ty, tz, tyaw = target
    rows: list[dict[str, Any]] = []
    previous: dict[str, Any] | None = None
    for pose in sorted(poses, key=lambda item: float(item["timestamp_sec"])):
        timestamp = float(pose["timestamp_sec"])
        yaw = pose.get("yaw_rad")
        yaw_error = _angle_error(float(yaw), tyaw) if yaw is not None else None
        row: dict[str, Any] = {
            "timestamp_sec": timestamp,
            "time_after_detection_sec": timestamp - detection_time if detection_time is not None else None,
            "x_m": float(pose["x_m"]),
            "depth_m": float(pose["y_m"]),
            "z_m": float(pose["z_m"]),
            "yaw_deg": math.degrees(float(yaw)) if yaw is not None else None,
            "x_error_m": float(pose["x_m"]) - tx,
            "depth_error_m": float(pose["y_m"]) - ty,
            "z_error_m": float(pose["z_m"]) - tz,
            "yaw_error_deg": math.degrees(yaw_error) if yaw_error is not None else None,
            "x_velocity_mps": None,
            "depth_velocity_mps": None,
            "z_velocity_mps": None,
            "yaw_rate_deg_s": None,
        }
        if previous is not None:
            delta = timestamp - float(previous["timestamp_sec"])
            if 0.005 <= delta <= 0.50:
                row["x_velocity_mps"] = (row["x_m"] - previous["x_m"]) / delta
                row["depth_velocity_mps"] = (row["depth_m"] - previous["depth_m"]) / delta
                row["z_velocity_mps"] = (row["z_m"] - previous["z_m"]) / delta
                if row["yaw_error_deg"] is not None and previous["yaw_error_deg"] is not None:
                    delta_yaw = math.degrees(math.atan2(
                        math.sin(math.radians(row["yaw_error_deg"] - previous["yaw_error_deg"])),
                        math.cos(math.radians(row["yaw_error_deg"] - previous["yaw_error_deg"])),
                    ))
                    row["yaw_rate_deg_s"] = delta_yaw / delta
        values = (
            row["x_error_m"] / normalisation["x_m"],
            row["depth_error_m"] / normalisation["depth_m"],
            row["z_error_m"] / normalisation["z_m"],
            (row["yaw_error_deg"] / normalisation["yaw_deg"]) if row["yaw_error_deg"] is not None else math.nan,
        )
        row["normalised_error"] = math.sqrt(sum(value * value for value in values)) if all(math.isfinite(value) for value in values) else None
        rows.append(row)
        previous = row
    return rows


def _inside_recovery_thresholds(row: dict[str, Any], thresholds: dict[str, float]) -> bool:
    required = ("x_error_m", "depth_error_m", "z_error_m", "yaw_error_deg")
    if any(row.get(key) is None for key in required):
        return False
    return (
        abs(float(row["x_error_m"])) <= thresholds["x_m"]
        and abs(float(row["depth_error_m"])) <= thresholds["depth_m"]
        and abs(float(row["z_error_m"])) <= thresholds["z_m"]
        and abs(float(row["yaw_error_deg"])) <= thresholds["yaw_deg"]
    )


def _integral_abs_normalised_error(rows: list[dict[str, Any]], start: float, end: float) -> float | None:
    selected = [row for row in rows if start <= float(row["timestamp_sec"]) <= end and row.get("normalised_error") is not None]
    if len(selected) < 2:
        return None
    total = 0.0
    for left, right in zip(selected, selected[1:]):
        delta = float(right["timestamp_sec"]) - float(left["timestamp_sec"])
        if 0.0 < delta <= 0.50:
            total += delta * (float(left["normalised_error"]) + float(right["normalised_error"])) * 0.5
    return total


def _plot(
    derived: Path,
    rows: list[dict[str, Any]],
    *,
    detection_time: float,
    peak_time: float | None,
    recovery_entry_time: float | None,
    thresholds: dict[str, float],
    pre_detection_window_sec: float,
) -> tuple[list[str], list[str]]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional plotting dependency.
        return [], [f"matplotlib unavailable; no E8 disturbance plot generated: {exc}"]
    # Keep a short pre-event window so the detected disturbance is visible as
    # an interior, common reference line rather than being hidden at the left
    # edge of a post-event-only graph.
    selected = [
        row for row in rows
        if float(row["timestamp_sec"]) >= detection_time - max(pre_detection_window_sec, 0.0)
    ]
    if not selected:
        return [], ["no controller-world samples at or after disturbance_detected event"]
    fig, axes = plt.subplots(4, 1, figsize=(7.2, 7.8), sharex=True)
    series = (
        ("x_error_m", "x error [m]", thresholds["x_m"]),
        ("depth_error_m", "depth error [m]", thresholds["depth_m"]),
        ("z_error_m", "z error [m]", thresholds["z_m"]),
        ("yaw_error_deg", "yaw error [deg]", thresholds["yaw_deg"]),
    )
    times = [float(row["timestamp_sec"]) - detection_time for row in selected]
    for index, (key, label, threshold) in enumerate(series):
        axis = axes[index]
        axis.plot(times, [row.get(key) for row in selected], linewidth=0.9, label="error")
        axis.axhline(0.0, color="tab:red", linestyle="--", linewidth=0.8, label="target" if index == 0 else None)
        axis.axhspan(-threshold, threshold, color="tab:green", alpha=0.12, label="recovery bound" if index == 0 else None)
        axis.axvline(
            0.0,
            color="tab:purple",
            linestyle="-.",
            linewidth=1.2,
            label="disturbance detected (t=0)" if index == 0 else None,
        )
        if peak_time is not None:
            axis.axvline(peak_time - detection_time, color="tab:orange", linestyle=":", linewidth=1.0, label="peak" if index == 0 else None)
        if recovery_entry_time is not None:
            axis.axvline(recovery_entry_time - detection_time, color="tab:green", linestyle=":", linewidth=1.0, label="recovery entry" if index == 0 else None)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("time relative to detected perturbation [s]")
    fig.suptitle("E8 manual perturbation recovery (state-observed)")
    fig.tight_layout()
    path = derived / "plots" / "disturbance_recovery_xyzyaw.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return [str(path.relative_to(derived))], []


def augment_t1_disturbance_report(
    session_dir: Path,
    derived: Path,
    poses: list[dict[str, Any]],
    events: list[dict[str, Any]],
    report: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Write E8 recovery artefacts and attach a report section.

    The function is called only after the generic analyser has copied raw bag
    samples to ``derived``.  It always uses the immutable bag event stream and
    the frozen metadata embedded in the session manifest.
    """

    manifest = read_json(session_dir / "manifest.json")
    metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
    target = _target(metadata)
    protocol = metadata.get("disturbance_protocol") if isinstance(metadata.get("disturbance_protocol"), dict) else {}
    section: dict[str, Any] = {
        "analyzer": "experiment_recorder.t1_disturbance_analysis",
        "analysis_basis": "state-observed manual perturbation; no external force magnitude inferred",
        "target_controller_world": metadata.get("target_controller_world"),
        "detected": False,
        "warnings": [],
    }
    warnings: list[str] = []
    if target is None:
        section["warnings"].append("missing target_controller_world metadata")
        report["t1_disturbance_recovery"] = section
        write_json(derived / "disturbance_analysis_report.json", section)
        return [], section["warnings"]
    recovery_raw = protocol.get("recovery_thresholds") if isinstance(protocol.get("recovery_thresholds"), dict) else {}
    thresholds = {
        "x_m": _as_float(recovery_raw, "x_m", 0.10),
        "depth_m": _as_float(recovery_raw, "depth_m", 0.10),
        "z_m": _as_float(recovery_raw, "z_m", 0.10),
        "yaw_deg": _as_float(recovery_raw, "yaw_deg", 10.0),
    }
    detection_time = _event_time(events, "disturbance_detected")
    if detection_time is None:
        section.update({"reason": "no disturbance_detected event in bag", "finished_at": now_iso()})
        report["t1_disturbance_recovery"] = section
        write_json(derived / "disturbance_analysis_report.json", section)
        _write_csv(derived / "disturbance_metrics.csv", [section], list(section.keys()))
        return [], [section["reason"]]
    pose_topic = _select_control_pose_topic(poses)
    selected = [row for row in poses if row.get("topic") == pose_topic] if pose_topic else []
    if not selected:
        section.update({"reason": "no controller-world pose samples", "finished_at": now_iso()})
        report["t1_disturbance_recovery"] = section
        write_json(derived / "disturbance_analysis_report.json", section)
        _write_csv(derived / "disturbance_metrics.csv", [section], list(section.keys()))
        return [], [section["reason"]]
    rows = _rows(selected, target=target, detection_time=detection_time, normalisation=thresholds)
    _write_csv(
        derived / "disturbance_samples.csv",
        rows,
        [
            "timestamp_sec", "time_after_detection_sec", "x_m", "depth_m", "z_m", "yaw_deg",
            "x_error_m", "depth_error_m", "z_error_m", "yaw_error_deg", "x_velocity_mps",
            "depth_velocity_mps", "z_velocity_mps", "yaw_rate_deg_s", "normalised_error",
        ],
    )
    post_timeout = _as_float(protocol, "post_detection_timeout_sec", 45.0)
    peak_search = _as_float(protocol, "peak_search_sec", min(5.0, post_timeout))
    pre_detection_window = _as_float(protocol, "plot_pre_detection_window_sec", 5.0)
    post_rows = [row for row in rows if detection_time <= float(row["timestamp_sec"]) <= detection_time + peak_search]
    peak = max(post_rows, key=lambda row: float(row.get("normalised_error") or -math.inf)) if post_rows else None
    peak_time = float(peak["timestamp_sec"]) if peak is not None else None
    required_hold = _as_float(protocol, "required_stable_hold_sec", 10.0)
    recovery_entry_time: float | None = None
    recovery_confirmed_time: float | None = None
    candidate_start: float | None = None
    if peak_time is not None:
        for row in rows:
            timestamp = float(row["timestamp_sec"])
            if timestamp < peak_time or timestamp > detection_time + post_timeout:
                continue
            if _inside_recovery_thresholds(row, thresholds):
                candidate_start = candidate_start or timestamp
                if timestamp - candidate_start >= required_hold:
                    recovery_entry_time = candidate_start
                    recovery_confirmed_time = timestamp
                    break
            else:
                candidate_start = None
    response_end = min(
        max((float(row["timestamp_sec"]) for row in rows), default=detection_time),
        detection_time + post_timeout,
    )
    section.update({
        "detected": True,
        "pose_topic": pose_topic,
        "detection_event_sec": detection_time,
        "peak_time_sec": peak_time,
        "peak_after_detection_sec": peak_time - detection_time if peak_time is not None else None,
        "peak_errors": {
            "x_m": peak.get("x_error_m") if peak else None,
            "depth_m": peak.get("depth_error_m") if peak else None,
            "z_m": peak.get("z_error_m") if peak else None,
            "yaw_deg": peak.get("yaw_error_deg") if peak else None,
            "normalised": peak.get("normalised_error") if peak else None,
        },
        "recovery_entry_time_sec": recovery_entry_time,
        "recovery_confirmed_time_sec": recovery_confirmed_time,
        "recovery_time_from_peak_sec": recovery_entry_time - peak_time if recovery_entry_time is not None and peak_time is not None else None,
        "recovery_confirmed": recovery_confirmed_time is not None,
        "observation_timeout_sec": post_timeout,
        "normalised_error_iae_from_peak": _integral_abs_normalised_error(rows, peak_time, response_end) if peak_time is not None else None,
        "thresholds": thresholds,
        "required_stable_hold_sec": required_hold,
        "finished_at": now_iso(),
    })
    if recovery_confirmed_time is None:
        warnings.append("recovery not confirmed inside the frozen post-detection observation window")
    plots, plot_warnings = _plot(
        derived,
        rows,
        detection_time=detection_time,
        peak_time=peak_time,
        recovery_entry_time=recovery_entry_time,
        thresholds=thresholds,
        pre_detection_window_sec=pre_detection_window,
    )
    warnings.extend(plot_warnings)
    section["warnings"] = warnings
    report["t1_disturbance_recovery"] = section
    _write_csv(
        derived / "disturbance_metrics.csv",
        [{
            "detected": section["detected"],
            "pose_topic": section["pose_topic"],
            "detection_event_sec": section["detection_event_sec"],
            "peak_after_detection_sec": section["peak_after_detection_sec"],
            "peak_normalised_error": section["peak_errors"]["normalised"],
            "peak_x_error_m": section["peak_errors"]["x_m"],
            "peak_depth_error_m": section["peak_errors"]["depth_m"],
            "peak_z_error_m": section["peak_errors"]["z_m"],
            "peak_yaw_error_deg": section["peak_errors"]["yaw_deg"],
            "recovery_time_from_peak_sec": section["recovery_time_from_peak_sec"],
            "recovery_confirmed": section["recovery_confirmed"],
            "normalised_error_iae_from_peak": section["normalised_error_iae_from_peak"],
        }],
        [
            "detected", "pose_topic", "detection_event_sec", "peak_after_detection_sec", "peak_normalised_error",
            "peak_x_error_m", "peak_depth_error_m", "peak_z_error_m", "peak_yaw_error_deg",
            "recovery_time_from_peak_sec", "recovery_confirmed", "normalised_error_iae_from_peak",
        ],
    )
    write_json(derived / "disturbance_analysis_report.json", section)
    return plots, warnings
