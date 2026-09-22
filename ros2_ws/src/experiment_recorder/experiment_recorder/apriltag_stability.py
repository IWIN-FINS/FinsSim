"""Subscriber-only AprilTag/Snell/fusion runtime-stability recorder.

The command deliberately has no publishers, service clients, subprocesses, or
parameter writes.  It can run next to a T1 hardware experiment without
changing the controller, hardware bridge, or experiment runner.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import String
from msgs.msg import TrajectoryEvent
import yaml

from .manifest import file_sha256, git_revision, now_iso, safe_session_dir, write_json


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _default_config() -> Path:
    source = Path(__file__).resolve().parents[1] / "config" / "apriltag_stability.yaml"
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("experiment_recorder")) / "config" / source.name
        if installed.is_file():
            return installed
    except Exception:
        pass
    return source


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"stability configuration not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid stability YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"stability configuration must be a mapping: {path}")
    return payload


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"configuration field {key!r} must be a mapping")
    return value


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_cli_path(value: str | Path, repo_root: Path) -> Path:
    """Accept a path relative to either the caller's cwd or repository root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    return cwd_path if cwd_path.exists() else _resolve_path(path, repo_root)


def _new_session_id(experiment_id: str) -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{experiment_id}_Stability"


def _stamp_ns(message: Any) -> int | None:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return None
    value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
    return value if value > 0 else None


def _json_payload(data: str) -> dict[str, Any]:
    try:
        parsed = json.loads(data)
    except (TypeError, ValueError):
        return {"parse_error": True, "raw": str(data)}
    return parsed if isinstance(parsed, dict) else {"parse_error": True, "raw": str(data)}


def _as_bool(value: object) -> bool | None:
    return value if isinstance(value, bool) else None


def _finite(value: object) -> float | None:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _phase_for_hold_age(age_sec: float, windows: dict[str, Any]) -> str:
    """Split an event-defined hold period without guessing a vehicle arrival time."""

    if age_sec < float(windows.get("transient_end_sec", 40.0)):
        return "hold_transient"
    if age_sec < float(windows.get("late_end_sec", 60.0)):
        return "late_hold"
    return "hold_after_registered_window"


def _capture_observation(mode: str, event_phase: str) -> bool:
    """Return whether a non-event sample belongs to the requested window.

    T1 evidence is restricted to each target's ``hold_start`` through
    ``phase_end``. T2 is
    restricted to ``trajectory_start`` through ``trajectory_end``. Acquisition,
    post-recording, and repeat-reset gaps are not merely excluded from an
    aggregate; they are not written as observation samples at all. Event
    markers remain in the raw stream so the boundaries are auditable.
    """

    return (
        mode == "free_drift"
        or (mode == "t1" and event_phase == "hold")
        or (mode == "t2" and event_phase == "trajectory")
    )


def _false_intervals(samples: Iterable[dict[str, Any]], *, value_key: str, end_sec: float) -> list[dict[str, float]]:
    """Return contiguous false-status intervals from ordered normalized rows."""

    intervals: list[dict[str, float]] = []
    started: float | None = None
    last_time: float | None = None
    for sample in samples:
        timestamp = _finite(sample.get("elapsed_sec"))
        if timestamp is None:
            continue
        value = _as_bool(sample.get("payload", {}).get(value_key))
        if value is False and started is None:
            started = timestamp
        if value is True and started is not None:
            intervals.append({"start_sec": started, "end_sec": timestamp, "duration_sec": max(0.0, timestamp - started)})
            started = None
        last_time = timestamp
    if started is not None:
        finish = max(end_sec, last_time if last_time is not None else started)
        intervals.append({"start_sec": started, "end_sec": finish, "duration_sec": max(0.0, finish - started)})
    return intervals


def _rate_summary(samples: list[dict[str, Any]], *, value_key: str) -> dict[str, Any]:
    observed = [_as_bool(row.get("payload", {}).get(value_key)) for row in samples]
    observed = [value for value in observed if value is not None]
    if not observed:
        return {"sample_count": 0, "true_count": 0, "availability": None}
    true_count = sum(observed)
    return {"sample_count": len(observed), "true_count": true_count, "availability": true_count / len(observed)}


def _stream_rate_hz(samples: list[dict[str, Any]]) -> float | None:
    values = [_finite(row.get("elapsed_sec")) for row in samples]
    times = [value for value in values if value is not None]
    if len(times) < 2 or times[-1] <= times[0]:
        return None
    return (len(times) - 1) / (times[-1] - times[0])


def _pose_increment_summary(samples: list[dict[str, Any]]) -> dict[str, Any]:
    distances: list[float] = []
    previous: tuple[float, float] | None = None
    for row in samples:
        payload = row.get("payload", {})
        x_m, y_m = _finite(payload.get("x_m")), _finite(payload.get("y_m"))
        if x_m is None or y_m is None:
            continue
        if previous is not None:
            distances.append(math.hypot(x_m - previous[0], y_m - previous[1]))
        previous = (x_m, y_m)
    if not distances:
        return {"increment_count": 0, "median_step_xy_m": None, "p95_step_xy_m": None}
    ordered = sorted(distances)
    return {
        "increment_count": len(distances),
        "median_step_xy_m": ordered[len(ordered) // 2],
        "p95_step_xy_m": ordered[min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)],
    }


@dataclass
class _EventState:
    session_id: str = "unlinked"
    setpoint_id: str = ""
    phase: str = "waiting_for_t1"
    hold_start_sec: float | None = None


class AprilTagStabilityRecorder(Node):
    """A passive observer for production perception and fusion interfaces."""

    def __init__(
        self,
        *,
        config: dict[str, Any],
        mode: str,
        session_id: str,
        linked_t1_session: str | None,
        raw_path: Path,
    ) -> None:
        super().__init__("apriltag_stability_recorder")
        self._config = config
        self._mode = mode
        self._linked_t1_session = linked_t1_session
        self._start_monotonic = time.monotonic()
        self._event_state = _EventState(
            session_id=session_id if mode == "free_drift" else "unlinked",
            phase="free_drift" if mode == "free_drift" else "waiting_for_t1",
        )
        t1_config = _mapping(config, "t1")
        self._hold_windows = t1_config.get("hold_windows", {})
        self._t1_capture_scope = str(t1_config.get("capture_scope", "all_target_sessions"))
        if self._t1_capture_scope != "all_target_sessions":
            raise ValueError("t1.capture_scope must be all_target_sessions")
        self._rows: list[dict[str, Any]] = []
        self._raw_stream = raw_path.open("w", encoding="utf-8")
        self._topic_counts: dict[str, int] = {}
        self._ignored_counts: dict[str, int] = {}
        topics = _mapping(config, "topics")

        # These subscriptions are intentionally the node's only ROS actions.
        self.create_subscription(String, str(topics["detector_status"]), self._detector_status_callback, 50)
        self.create_subscription(String, str(topics["refractive_status"]), self._refractive_status_callback, 50)
        self.create_subscription(String, str(topics["fusion_status"]), self._fusion_status_callback, 50)
        self.create_subscription(PoseWithCovarianceStamped, str(topics["snell_pose"]), self._snell_pose_callback, 50)
        self.create_subscription(PoseWithCovarianceStamped, str(topics["pinhole_pose"]), self._pinhole_pose_callback, 50)
        self.create_subscription(PoseWithCovarianceStamped, str(topics["fused_pose"]), self._fused_pose_callback, 50)
        if mode in {"t1", "t2"}:
            self.create_subscription(TrajectoryEvent, str(topics["experiment_event"]), self._event_callback, 50)
        period = max(float(_mapping(config, "sampling").get("info_period_sec", 5.0)), 0.5)
        self.create_timer(period, self._log_progress)

    @property
    def rows(self) -> list[dict[str, Any]]:
        return self._rows

    @property
    def elapsed_sec(self) -> float:
        return max(0.0, time.monotonic() - self._start_monotonic)

    def close(self) -> None:
        if not self._raw_stream.closed:
            self._raw_stream.flush()
            self._raw_stream.close()

    def _current_phase(self) -> str:
        if self._mode == "free_drift":
            return "free_drift"
        if self._mode == "t1" and self._event_state.phase == "hold" and self._event_state.hold_start_sec is not None:
            return _phase_for_hold_age(self.elapsed_sec - self._event_state.hold_start_sec, self._hold_windows)
        return self._event_state.phase

    def _record(self, topic: str, payload: dict[str, Any], *, source_stamp_ns: int | None = None) -> None:
        if topic != "experiment_event" and not _capture_observation(self._mode, self._event_state.phase):
            self._ignored_counts[topic] = self._ignored_counts.get(topic, 0) + 1
            return
        row = {
            "elapsed_sec": self.elapsed_sec,
            "wall_time": now_iso(),
            "topic": topic,
            "source_stamp_ns": source_stamp_ns,
            "mode": self._mode,
            "linked_t1_session": self._event_state.session_id,
            "setpoint_id": self._event_state.setpoint_id,
            "phase": self._current_phase(),
            "payload": payload,
        }
        self._rows.append(row)
        self._topic_counts[topic] = self._topic_counts.get(topic, 0) + 1
        self._raw_stream.write(json.dumps(row, ensure_ascii=True, sort_keys=True) + "\n")

    def _detector_status_callback(self, message: String) -> None:
        payload = _json_payload(message.data)
        source_stamp = payload.get("frame_stamp_ns")
        self._record("detector_status", payload, source_stamp_ns=source_stamp if isinstance(source_stamp, int) else None)

    def _refractive_status_callback(self, message: String) -> None:
        payload = _json_payload(message.data)
        source_stamp = payload.get("input_stamp_ns")
        self._record("refractive_status", payload, source_stamp_ns=source_stamp if isinstance(source_stamp, int) else None)

    def _fusion_status_callback(self, message: String) -> None:
        self._record("fusion_status", _json_payload(message.data))

    @staticmethod
    def _pose_payload(message: PoseWithCovarianceStamped) -> dict[str, Any]:
        pose = message.pose.pose
        return {
            "frame_id": str(message.header.frame_id),
            "x_m": float(pose.position.x),
            "y_m": float(pose.position.y),
            "z_m": float(pose.position.z),
            "qx": float(pose.orientation.x),
            "qy": float(pose.orientation.y),
            "qz": float(pose.orientation.z),
            "qw": float(pose.orientation.w),
        }

    def _snell_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        self._record("snell_pose", self._pose_payload(message), source_stamp_ns=_stamp_ns(message))

    def _pinhole_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        self._record("pinhole_pose", self._pose_payload(message), source_stamp_ns=_stamp_ns(message))

    def _fused_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        self._record("fused_pose", self._pose_payload(message), source_stamp_ns=_stamp_ns(message))

    def _event_callback(self, message: TrajectoryEvent) -> None:
        if self._linked_t1_session and message.session_id != self._linked_t1_session:
            return
        payload = {
            "event": message.event,
            "label": message.label,
            "session_id": message.session_id,
            "metadata": _json_payload(message.metadata_json),
        }
        if message.event == "trial_start":
            self._event_state = _EventState(session_id=message.session_id, setpoint_id=message.label, phase="acquisition")
        elif message.event == "hold_start":
            # The T1 runner deliberately repeats a marker three times. The
            # first receipt defines the time origin; later duplicates must not
            # shift the transient/late-hold boundary.
            if self._event_state.session_id != message.session_id or self._event_state.phase != "hold":
                self._event_state.session_id = message.session_id
                self._event_state.setpoint_id = message.label
                self._event_state.phase = "hold"
                self._event_state.hold_start_sec = self.elapsed_sec
        elif message.event == "trajectory_start":
            self._event_state.session_id = message.session_id
            self._event_state.setpoint_id = message.label
            self._event_state.phase = "trajectory"
            self._event_state.hold_start_sec = None
        elif message.event in {"trajectory_end", "phase_end", "trial_end", "operator_abort", "safety_abort"}:
            self._event_state.phase = "post_trial"
        self._record("experiment_event", payload, source_stamp_ns=_stamp_ns(message))

    def _log_progress(self) -> None:
        counts = ", ".join(f"{key}={value}" for key, value in sorted(self._topic_counts.items())) or "no samples"
        skipped = sum(self._ignored_counts.values())
        self.get_logger().info(
            f"AprilTag stability: mode={self._mode} phase={self._current_phase()} "
            f"session={self._event_state.session_id} elapsed={self.elapsed_sec:.1f}s "
            f"recorded=({counts}) skipped_outside_window={skipped}"
        )


def _group_rows(rows: list[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("topic") == "experiment_event":
            continue
        key = (str(row.get("linked_t1_session", "unlinked")), str(row.get("phase", "unknown")))
        grouped.setdefault(key, []).append(row)
    return grouped


def _rows_for_topic(rows: list[dict[str, Any]], topic: str) -> list[dict[str, Any]]:
    return [row for row in rows if row.get("topic") == topic]


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_summaries(summary_rows: list[dict[str, Any]], dropout_rows: list[dict[str, Any]], raw_rows: list[dict[str, Any]], plots_dir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics = ("detector", "constrained_output", "snell_physical_solution", "fusion_vision_fresh", "fusion_ready")
    if summary_rows:
        labels = [f"{row['t1_session_id']}\n{row['phase']}" for row in summary_rows]
        figure, axis = plt.subplots(figsize=(max(7.0, len(labels) * 1.3), 4.2))
        width = 0.8 / len(metrics)
        for index, metric in enumerate(metrics):
            values = [row.get(f"{metric}_availability") for row in summary_rows]
            axis.bar([i + (index - (len(metrics) - 1) / 2) * width for i in range(len(labels))], [v or 0.0 for v in values], width=width, label=metric)
        axis.set_ylim(0.0, 1.05)
        axis.set_ylabel("availability fraction")
        axis.set_xticks(range(len(labels)), labels, rotation=20, ha="right")
        axis.legend(fontsize=8, ncol=2)
        axis.grid(axis="y", alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots_dir / "availability_by_phase.png", dpi=180)
        plt.close(figure)
    if dropout_rows:
        figure, axis = plt.subplots(figsize=(7.0, 4.0))
        by_stream: dict[str, list[float]] = {}
        for row in dropout_rows:
            by_stream.setdefault(str(row["stream"]), []).append(float(row["duration_sec"]))
        for stream, values in sorted(by_stream.items()):
            axis.hist(values, bins=min(20, max(4, len(values))), alpha=0.45, label=stream)
        axis.set_xlabel("continuous unavailable interval [s]")
        axis.set_ylabel("count")
        axis.legend(fontsize=8)
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(plots_dir / "dropout_duration_histogram.png", dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(2, 1, figsize=(8.0, 5.5), sharex=True)
    plotted = False
    for topic, label in (("snell_pose", "Snell/constrained"), ("pinhole_pose", "Pinhole"), ("fused_pose", "Fusion")):
        samples = _rows_for_topic(raw_rows, topic)
        if not samples:
            continue
        times = [float(row["elapsed_sec"]) for row in samples]
        xs = [_finite(row.get("payload", {}).get("x_m")) for row in samples]
        ys = [_finite(row.get("payload", {}).get("y_m")) for row in samples]
        axes[0].plot(times, xs, linewidth=0.9, label=label)
        axes[1].plot(times, ys, linewidth=0.9, label=label)
        plotted = True
    if plotted:
        axes[0].set_ylabel("pool x [m]")
        axes[1].set_ylabel("pool y [m]")
        axes[1].set_xlabel("recorder elapsed time [s]")
        for axis in axes:
            axis.grid(alpha=0.25)
            axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(plots_dir / "horizontal_pose_timeline.png", dpi=180)
    plt.close(figure)


def _analyse(rows: list[dict[str, Any]], *, duration_sec: float, output_dir: Path) -> dict[str, Any]:
    """Generate per-session availability summaries; frames are not trials."""

    summary_rows: list[dict[str, Any]] = []
    dropout_rows: list[dict[str, Any]] = []
    metric_specs = (
        ("detector", "detector_status", "detected"),
        ("constrained_output", "refractive_status", "constrained_valid"),
        ("snell_physical_solution", "refractive_status", "snell_valid"),
        ("fusion_vision_fresh", "fusion_status", "vision_fresh"),
        ("fusion_ready", "fusion_status", "ready"),
    )
    for (session_id, phase), subset in sorted(_group_rows(rows).items()):
        row: dict[str, Any] = {"t1_session_id": session_id, "phase": phase}
        for metric, topic, value_key in metric_specs:
            samples = _rows_for_topic(subset, topic)
            values = _rate_summary(samples, value_key=value_key)
            row.update({f"{metric}_{key}": value for key, value in values.items()})
            row[f"{metric}_rate_hz"] = _stream_rate_hz(samples)
            for interval in _false_intervals(samples, value_key=value_key, end_sec=duration_sec):
                dropout_rows.append({"t1_session_id": session_id, "phase": phase, "stream": metric, **interval})
        for stream_name, topic in (("snell", "snell_pose"), ("pinhole", "pinhole_pose"), ("fusion", "fused_pose")):
            increments = _pose_increment_summary(_rows_for_topic(subset, topic))
            row.update({f"{stream_name}_{key}": value for key, value in increments.items()})
        summary_rows.append(row)
    derived = output_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    _write_csv(derived / "phase_summary.csv", summary_rows)
    _write_csv(derived / "dropout_intervals.csv", dropout_rows)
    _plot_summaries(summary_rows, dropout_rows, rows, derived / "plots")
    return {
        "analysis_schema_version": 1,
        "duration_sec": duration_sec,
        "statistical_unit": "one free-drift recording or one event-defined T1 session/phase; frames are not independent trials",
        "availability_definitions": {
            "detector": "detector status frames with detected=true",
            "constrained_output": "refractive status frames with constrained_valid=true; may include configured pinhole fallback",
            "snell_physical_solution": "refractive status frames with snell_valid=true",
            "fusion_vision_fresh": "fusion status samples with vision_fresh=true",
            "fusion_ready": "fusion status samples with ready=true",
            "pose_increment": "consecutive horizontal output displacement, not absolute localization error or a noise-only estimate",
        },
        "phase_summary": summary_rows,
        "dropout_interval_count": len(dropout_rows),
        "warnings": [
            "This observer measures runtime availability and output continuity. It does not create localization ground truth.",
            "A false constrained output is not necessarily a detector miss: geometry/depth/IMU constraints may reject it.",
            "In T1 mode, every event-defined target session retains only samples between hold_start and phase_end. In T2 mode, only samples between trajectory_start and trajectory_end are retained; acquisition and reset intervals are excluded.",
        ],
    }


def _copy_config(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _write_readme(session_dir: Path) -> None:
    (session_dir / "README.md").write_text(
        "# AprilTag 感知稳定性记录\n\n"
        "本目录由 `record_apriltag_stability` 生成。记录器仅订阅 ROS2 话题；"
        "不启动控制器/桥接、不发布推进器命令、不修改 T1。\n\n"
        "- `raw/stream.ndjson`：带接收时间、事件阶段和原始状态字段的规范化流。\n"
        "- `derived/phase_summary.csv`：按自由漂移记录或 T1 session/phase 汇总的可用率。\n"
        "- `derived/dropout_intervals.csv`：连续不可用区间。\n"
        "- `derived/plots/`：可用率、丢失时长和水平输出时间线。\n"
        "- `analysis_report.json`：指标定义与分析警告。\n\n"
        "这些结果没有外部真值，不能替代 E2 定位精度指标。\n",
        encoding="utf-8",
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Read-only AprilTag/Snell/fusion runtime stability recorder")
    parser.add_argument("--config", type=Path, default=_default_config())
    parser.add_argument("--mode", choices=("free_drift", "t1", "t2"), required=True)
    parser.add_argument("--duration-sec", type=float, default=None, help="required for free_drift unless supplied by YAML")
    parser.add_argument("--session-id", default="", help="optional safe output-directory name")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--linked-t1-session", default="", help="optional TrajectoryEvent session ID filter")
    parser.add_argument("--label", default="", help="operator label stored only in the manifest")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    config_path = _resolve_cli_path(args.config, repo_root)
    config = _load_yaml(config_path)
    experiment = _mapping(config, "experiment")
    output_root = _resolve_path(args.output_root or experiment["output_root"], repo_root)
    experiment_id = str(experiment.get("experiment_id", "E1-APRILTAG-AVAILABILITY"))
    session_id = str(args.session_id).strip() or _new_session_id(experiment_id)
    session_dir = safe_session_dir(output_root, session_id)
    duration = args.duration_sec
    if duration is None:
        duration = float(experiment.get("default_free_drift_duration_sec", 60.0)) if args.mode == "free_drift" else 0.0
    if args.mode == "free_drift" and duration <= 0.0:
        raise SystemExit("free_drift requires --duration-sec > 0 or a positive YAML default")
    if args.mode == "t1" and duration < 0.0:
        raise SystemExit("--duration-sec must be non-negative")
    if args.dry_run:
        print(json.dumps({"mode": args.mode, "session_dir": str(session_dir), "duration_sec": duration, "subscriber_only": True, "linked_t1_session": args.linked_t1_session or None}, ensure_ascii=False, indent=2))
        return
    if session_dir.exists():
        if not args.overwrite:
            raise SystemExit(f"session directory already exists: {session_dir}; choose --session-id or pass --overwrite")
        shutil.rmtree(session_dir)
    raw_dir = session_dir / "raw"
    raw_dir.mkdir(parents=True)
    _copy_config(config_path, session_dir / "resolved_config.yaml")
    _write_readme(session_dir)
    manifest = {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "mode": args.mode,
        "session_id": session_id,
        "linked_t1_session": args.linked_t1_session or None,
        "operator_label": args.label or None,
        "started_at": now_iso(),
        "requested_duration_sec": duration if duration > 0.0 else None,
        "safety_contract": {"subscriber_only": True, "publishes_control": False, "launches_processes": False, "changes_parameters": False, "touches_hardware_bridge": False},
        "capture_contract": "free_drift: entire requested duration; t1: every event-defined target session, non-event samples only between hold_start and phase_end; t2: non-event samples only between trajectory_start and trajectory_end",
        "config_source": str(config_path),
        "config_sha256": file_sha256(config_path),
        "code_commit": git_revision(repo_root),
    }
    write_json(session_dir / "manifest.json", manifest)
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node = AprilTagStabilityRecorder(config=config, mode=args.mode, session_id=session_id, linked_t1_session=args.linked_t1_session or None, raw_path=raw_dir / "stream.ndjson")
    interrupted = False
    try:
        deadline = time.monotonic() + duration if duration > 0.0 else None
        node.get_logger().info(f"AprilTag stability recording started: mode={args.mode}, output={session_dir}. This node is subscriber-only and does not alter the active experiment.")
        while rclpy.ok() and (deadline is None or time.monotonic() < deadline):
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        interrupted = True
        node.get_logger().info("Ctrl-C received: finalizing the stability recorder only; T1 is not interrupted.")
    finally:
        duration_actual = node.elapsed_sec
        node.close()
        report = _analyse(node.rows, duration_sec=duration_actual, output_dir=session_dir)
        report.update({"ended_at": now_iso(), "interrupted": interrupted, "sample_count": len(node.rows)})
        write_json(session_dir / "derived" / "analysis_report.json", report)
        manifest.update({"ended_at": now_iso(), "actual_duration_sec": duration_actual, "interrupted": interrupted, "sample_count": len(node.rows), "analysis_report": "derived/analysis_report.json"})
        write_json(session_dir / "manifest.json", manifest)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    print(f"AprilTag stability session finalized: {session_dir}")


if __name__ == "__main__":
    main()
