"""Automatic post-run analysis for recorder sessions.

The recorder calls :func:`analyze_session` after rosbag2 has been stopped.  The
analyser deliberately keeps the raw bag immutable: it writes normalized CSV
samples, a machine-readable summary, and plots below ``derived/``.  Message
schemas are discovered from rosbag2 at runtime so the package can analyse both
the real and ROS2 simulation profiles without duplicating custom message
definitions.

Analysis is best-effort.  A missing optional plotting dependency or an empty
topic does not destroy a valid bag; the reason is recorded in
``derived/analysis_report.json`` and in the session manifest.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any, Iterable

from .manifest import now_iso, read_json, write_json
from .progress import log_stage


POSE_TOPIC_HINTS = (
    "/finsrov/pose",
    "/finsrov/evaluation/pose",
    "/finsrov/ablation/no_ekf/pose",
    "/finsrov/vision/refracted_pose_6d",
    "/finsrov/vision/refracted_pose_6d_pure",
    "/sim/finsrov/controller/pose",
    "/finsrov/controller/pose",
)
COMMAND_TOPIC_HINTS = (
    "/finsrov/hardware/thruster_cmd_echo",
    "/finsrov/thrusters_out",
    "/finsrov/controller/debug/thruster_command",
    "/motion_controller/debug/action",
    "/motion_controller/debug/policy_action",
    "/motion_controller/debug/wrench6d",
    "/motion_controller/debug/thruster_command",
    "/sim/finsrov/thrusters_out",
    "/sim/motion_controller/debug/thruster_command",
)
POSITION_GOAL_TOPIC_HINTS = frozenset({
    "/motion_controller/command/position_controller_world",
    "/sim/motion_controller/command/position_controller_world",
})
ACTIVE_POSITION_GOAL_TOPIC_HINTS = frozenset({
    "/motion_controller/status/active_pose",
    "/sim/motion_controller/status/active_pose",
})
STAMPED_THRUSTER_COMMAND_TOPIC_HINTS = frozenset({
    "/sim/motion_controller/debug/thruster_command_stamped",
})
LOCKSTEP_ACK_TOPIC_HINTS = frozenset({
    "/sim/motion_controller/debug/control_tick_complete",
})
GOAL_REACHED_TOPIC_HINTS = frozenset({
    "/motion_controller/status/reached",
    "/sim/motion_controller/status/reached",
})


def _stamp_seconds(message: Any, fallback_ns: int) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is not None and hasattr(stamp, "sec"):
        return float(stamp.sec) + float(getattr(stamp, "nanosec", 0)) * 1e-9
    return float(fallback_ns) * 1e-9


def _first_attr(value: Any, names: Iterable[str]) -> Any:
    for name in names:
        if hasattr(value, name):
            return getattr(value, name)
    return None


def _xyz(value: Any) -> tuple[float, float, float] | None:
    if value is None:
        return None
    nested = _first_attr(value, ("pose", "position", "translation"))
    if nested is not None and nested is not value:
        result = _xyz(nested)
        if result is not None:
            return result
    if all(hasattr(value, name) for name in ("x", "y", "z")):
        try:
            return float(value.x), float(value.y), float(value.z)
        except (TypeError, ValueError):
            return None
    return None


def _yaw(value: Any, *, controller_world: bool = False) -> float | None:
    """Extract yaw in the orientation convention declared by the pose frame.

    Real ``pool_world`` ROS poses use the conventional Z-up yaw.  Unity
    sim-truth poses are explicitly tagged ``controller_world`` and retain the
    Unity controller basis, where yaw is a rotation about +Y.  Treating the
    latter as Z-up makes a real yaw rotation appear to be zero in the
    experiment plots and metrics.
    """
    if value is None:
        return None
    nested = _first_attr(value, ("pose", "orientation"))
    if nested is not None and nested is not value:
        result = _yaw(nested, controller_world=controller_world)
        if result is not None:
            return result
    if all(hasattr(value, name) for name in ("x", "y", "z", "w")):
        try:
            qx, qy, qz, qw = (float(value.x), float(value.y), float(value.z), float(value.w))
            if controller_world:
                return math.atan2(2.0 * (qw * qy + qx * qz), 1.0 - 2.0 * (qx * qx + qy * qy))
            return math.atan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
        except (TypeError, ValueError, ZeroDivisionError):
            return None
    return None


def _pose_row(message: Any, timestamp_ns: int, topic: str) -> dict[str, Any] | None:
    point = _xyz(message)
    if point is None:
        return None
    header = getattr(message, "header", None)
    frame_id = str(getattr(header, "frame_id", ""))
    return {
        "timestamp_sec": _stamp_seconds(message, timestamp_ns),
        "topic": topic,
        # T2 must never align a pool_world state with a controller_world
        # reference without an explicit transform.  Preserve the recorded
        # frame instead of losing that information during CSV normalization.
        "frame_id": frame_id,
        "x_m": point[0],
        "y_m": point[1],
        "z_m": point[2],
        "yaw_rad": _yaw(message, controller_world=frame_id.strip().lower() == "controller_world"),
    }


def _numeric_sequence(message: Any) -> list[float] | None:
    for name in ("data", "values", "thrust", "rpm", "commands", "forces"):
        value = getattr(message, name, None)
        if value is None or isinstance(value, (str, bytes)):
            continue
        try:
            result = [float(item) for item in value]
        except (TypeError, ValueError):
            continue
        if result:
            return result
    # Wrench-like messages are converted to a fixed six-vector for plotting.
    wrench = getattr(message, "wrench", None)
    if wrench is not None:
        force = getattr(wrench, "force", None)
        torque = getattr(wrench, "torque", None)
        f = _xyz(force)
        t = _xyz(torque)
        if f is not None and t is not None:
            return [*f, *t]
    return None


def _bool_sample(message: Any, timestamp_ns: int, topic: str) -> dict[str, Any] | None:
    """Normalize a recorded ``std_msgs/Bool`` controller-status sample."""

    value = getattr(message, "data", None)
    if not isinstance(value, bool):
        return None
    return {"timestamp_sec": _stamp_seconds(message, timestamp_ns), "topic": topic, "reached": value}


def _read_bag(
    session_dir: Path,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[str],
]:
    """Read pose, command, goal, status, lockstep-audit, and event samples from rosbag2.

    A T1 ``hold_start`` marker is normally the immutable timing source.  Some
    early sessions lost that marker while rosbag2 was still discovering the
    event publisher. Keeping the controller-world position-goal command as a
    separate, typed stream permits the highest-priority recovery path without
    modifying the raw bag.
    """

    bag_dir = session_dir / "raw" / "rosbag2"
    if not (bag_dir / "metadata.yaml").is_file():
        raise FileNotFoundError(f"rosbag metadata not found: {bag_dir / 'metadata.yaml'}")
    try:
        import rosbag2_py  # type: ignore
        from rclpy.serialization import deserialize_message  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
    except Exception as exc:  # pragma: no cover - depends on ROS installation
        raise RuntimeError(f"ROS2 bag analysis dependencies are unavailable: {exc}") from exc

    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_by_topic = {item.name: item.type for item in reader.get_all_topics_and_types()}
    message_types = {topic: get_message(type_name) for topic, type_name in type_by_topic.items()}
    poses: list[dict[str, Any]] = []
    commands: list[dict[str, Any]] = []
    position_goals: list[dict[str, Any]] = []
    active_position_goals: list[dict[str, Any]] = []
    stamped_commands: list[dict[str, Any]] = []
    lockstep_acks: list[dict[str, Any]] = []
    goal_reached: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        message = deserialize_message(serialized, message_types[topic])
        if any(topic == hint or topic.endswith(hint) for hint in POSE_TOPIC_HINTS):
            row = _pose_row(message, timestamp_ns, topic)
            if row is not None:
                poses.append(row)
        if any(topic == hint or topic.endswith(hint) for hint in COMMAND_TOPIC_HINTS):
            values = _numeric_sequence(message)
            if values is not None:
                commands.append({"timestamp_sec": _stamp_seconds(message, timestamp_ns), "topic": topic, "values": values})
        if topic in POSITION_GOAL_TOPIC_HINTS:
            row = _pose_row(message, timestamp_ns, topic)
            if row is not None:
                position_goals.append(row)
        if topic in ACTIVE_POSITION_GOAL_TOPIC_HINTS:
            row = _pose_row(message, timestamp_ns, topic)
            if row is not None:
                active_position_goals.append(row)
        if topic in STAMPED_THRUSTER_COMMAND_TOPIC_HINTS:
            try:
                payload = json.loads(str(getattr(message, "data", "{}")))
                values = [float(value) for value in payload.get("values", [])]
            except (TypeError, ValueError, json.JSONDecodeError):
                values = []
            if values:
                stamped_commands.append(
                    {
                        "timestamp_sec": _stamp_seconds(message, timestamp_ns),
                        "topic": topic,
                        "values": values,
                        "frame_id": str(getattr(getattr(message, "header", None), "frame_id", "")),
                    }
                )
        if topic in LOCKSTEP_ACK_TOPIC_HINTS:
            try:
                tick_ns = int(getattr(message, "data"))
            except (TypeError, ValueError):
                tick_ns = 0
            if tick_ns > 0:
                lockstep_acks.append({"timestamp_sec": tick_ns * 1e-9, "topic": topic, "tick_ns": tick_ns})
        if topic in GOAL_REACHED_TOPIC_HINTS:
            row = _bool_sample(message, timestamp_ns, topic)
            if row is not None:
                goal_reached.append(row)
        if topic.endswith("/experiment/event") or topic == "/finsrov/experiment/event":
            event = {"timestamp_sec": _stamp_seconds(message, timestamp_ns), "event": str(getattr(message, "event", "")), "label": str(getattr(message, "label", ""))}
            raw_metadata = str(getattr(message, "metadata_json", "{}"))
            try:
                event["metadata"] = json.loads(raw_metadata)
            except json.JSONDecodeError:
                event["metadata"] = {"raw": raw_metadata}
            events.append(event)
    return (
        poses,
        commands,
        position_goals,
        active_position_goals,
        stamped_commands,
        lockstep_acks,
        goal_reached,
        events,
        sorted(type_by_topic),
    )


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _truth_xy(session_dir: Path) -> tuple[float, float] | None:
    path = session_dir / "external_truth" / "truth_xy.csv"
    if not path.is_file():
        return None


def _target_from_metadata(metadata: dict[str, Any]) -> tuple[float, float, float, float] | None:
    """Decode a controller-world T1 target from frozen session metadata."""

    value = metadata.get("target_controller_world")
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        return float(value[0]), float(value[1]), float(value[2]), math.radians(float(value[3]))
    except (TypeError, ValueError):
        return None


def _target_from_events(events: list[dict[str, Any]]) -> tuple[float, float, float, float] | None:
    """Return the first registered T1 target, if the event metadata has one."""

    for event in events:
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            target = _target_from_metadata(metadata)
            if target is not None:
                return target
    return None


def _angle_error(angle: float | None, target: float) -> float | None:
    if angle is None:
        return None
    return math.atan2(math.sin(angle - target), math.cos(angle - target))


def _window_settings(metadata: dict[str, Any], start_sec: float, end_sec: float | None) -> dict[str, float]:
    """Resolve immutable T1 response and primary-evaluation windows.

    Sessions recorded before the protocol change have no window metadata; for
    those, preserve their historical whole-response analysis rather than
    silently relabelling it as a late-hold metric.
    """

    available_duration = max(0.0, (end_sec - start_sec) if end_sec is not None else 0.0)
    primary = metadata.get("primary_evaluation_window")
    response = metadata.get("response_window")
    if not isinstance(primary, dict):
        return {
            "response_start_sec": start_sec,
            "response_end_sec": end_sec if end_sec is not None else start_sec,
            "evaluation_start_sec": start_sec,
            "evaluation_end_sec": end_sec if end_sec is not None else start_sec,
            "resample_hz": 0.0,
            "max_interpolation_gap_sec": 0.0,
            "minimum_valid_fraction": 0.0,
            "protocol_version": "legacy_whole_response",
        }
    response_duration = float(response.get("duration_sec", available_duration)) if isinstance(response, dict) else available_duration
    response_end = start_sec + max(response_duration, 0.0)
    if end_sec is not None:
        response_end = min(response_end, end_sec)
    evaluation_start = start_sec + max(float(primary.get("start_offset_sec", 0.0)), 0.0)
    evaluation_end = evaluation_start + max(float(primary.get("duration_sec", 0.0)), 0.0)
    evaluation_end = min(evaluation_end, response_end)
    return {
        "response_start_sec": start_sec,
        "response_end_sec": response_end,
        "evaluation_start_sec": evaluation_start,
        "evaluation_end_sec": evaluation_end,
        "resample_hz": max(float(primary.get("resample_hz", 0.0)), 0.0),
        "max_interpolation_gap_sec": max(float(primary.get("max_interpolation_gap_sec", 0.0)), 0.0),
        "minimum_valid_fraction": min(max(float(primary.get("minimum_valid_fraction", 0.0)), 0.0), 1.0),
        "protocol_version": "primary_late_hold",
    }


def _interpolate_pose_row(left: dict[str, Any], right: dict[str, Any], timestamp_sec: float) -> dict[str, Any]:
    """Linearly interpolate translation and shortest-path interpolate yaw."""

    left_time = float(left["timestamp_sec"])
    right_time = float(right["timestamp_sec"])
    alpha = 0.0 if right_time <= left_time else (timestamp_sec - left_time) / (right_time - left_time)
    row: dict[str, Any] = {"timestamp_sec": timestamp_sec, "topic": left["topic"]}
    for key in ("x_m", "y_m", "z_m"):
        left_value, right_value = left.get(key), right.get(key)
        row[key] = (
            float(left_value) + alpha * (float(right_value) - float(left_value))
            if left_value is not None and right_value is not None
            else None
        )
    left_yaw, right_yaw = left.get("yaw_rad"), right.get("yaw_rad")
    if left_yaw is None or right_yaw is None:
        row["yaw_rad"] = None
    else:
        row["yaw_rad"] = float(left_yaw) + alpha * _angle_error(float(right_yaw), float(left_yaw))
    return row


def _resample_pose_segment(
    segment: list[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
    rate_hz: float,
    max_interpolation_gap_sec: float,
) -> tuple[list[dict[str, Any]], int]:
    """Return fixed-rate in-window samples and the requested grid size.

    No sample is invented across a large acquisition gap.  This makes the
    valid fraction meaningful when vision/fusion becomes stale.
    """

    ordered = sorted(segment, key=lambda row: float(row["timestamp_sec"]))
    if end_sec < start_sec:
        return [], 0
    if rate_hz <= 0.0:
        return [row for row in ordered if start_sec <= float(row["timestamp_sec"]) <= end_sec], len(ordered)
    step = 1.0 / rate_hz
    expected = int(math.floor((end_sec - start_sec) * rate_hz + 1e-9)) + 1
    rows: list[dict[str, Any]] = []
    right_index = 0
    for sample_index in range(expected):
        timestamp = start_sec + sample_index * step
        while right_index < len(ordered) and float(ordered[right_index]["timestamp_sec"]) < timestamp:
            right_index += 1
        if right_index < len(ordered) and math.isclose(float(ordered[right_index]["timestamp_sec"]), timestamp, abs_tol=1e-6):
            rows.append(dict(ordered[right_index], timestamp_sec=timestamp))
            continue
        if right_index == 0 or right_index >= len(ordered):
            continue
        left, right = ordered[right_index - 1], ordered[right_index]
        if max_interpolation_gap_sec > 0.0 and float(right["timestamp_sec"]) - float(left["timestamp_sec"]) > max_interpolation_gap_sec:
            continue
        rows.append(_interpolate_pose_row(left, right, timestamp))
    return rows, expected


def _matching_position_goal_start(
    position_goals: list[dict[str, Any]], target: tuple[float, float, float, float]
) -> float | None:
    """Return the first recorded controller-world position-goal matching ``target``.

    This is deliberately a narrow recovery rule: a goal must match all three
    translation components and yaw in the frozen session target.  It therefore
    cannot turn an unrelated preflight or cancellation message into a T1
    timing anchor.
    """

    tx, ty, tz, tyaw = target
    translation_tolerance_m = 1e-4
    yaw_tolerance_rad = 1e-4
    for goal in sorted(position_goals, key=lambda row: float(row["timestamp_sec"])):
        if str(goal.get("frame_id", "")).strip().lower() != "controller_world":
            continue
        try:
            translation_matches = (
                abs(float(goal["x_m"]) - tx) <= translation_tolerance_m
                and abs(float(goal["y_m"]) - ty) <= translation_tolerance_m
                and abs(float(goal["z_m"]) - tz) <= translation_tolerance_m
            )
        except (KeyError, TypeError, ValueError):
            continue
        yaw_error = _angle_error(goal.get("yaw_rad"), tyaw)
        if translation_matches and yaw_error is not None and abs(yaw_error) <= yaw_tolerance_rad:
            return float(goal["timestamp_sec"])
    return None


def _run_calibrated_hold_end_delay(session_dir: Path, response_duration_sec: float) -> float | None:
    """Estimate the runner's post-hold marker delay from sibling raw-event exports.

    In the affected accelerated simulation batches, some trial bags contain
    ``phase_end`` but no early event burst and also missed the one-shot target
    command.  Sibling trials in the *same frozen run* retain both markers.
    Their ``phase_end - hold_start - declared_duration`` is the deterministic
    event-delivery overhead to remove before reconstructing the missing start.
    The existing ``derived/events.json`` files are normalized copies of each
    immutable rosbag event stream, not manually edited annotations.
    """

    if response_duration_sec <= 0.0:
        return None
    trial_root = session_dir.parent
    values: list[float] = []
    for event_path in sorted(trial_root.glob("*/derived/events.json")):
        try:
            loaded = json.loads(event_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(loaded, list):
            continue
        starts = [
            float(event["timestamp_sec"])
            for event in loaded
            if isinstance(event, dict) and event.get("event") == "hold_start"
        ]
        if not starts:
            continue
        start = min(starts)
        ends = [
            float(event["timestamp_sec"])
            for event in loaded
            if isinstance(event, dict)
            and event.get("event") == "phase_end"
            and float(event.get("timestamp_sec", 0.0)) > start
        ]
        if ends:
            values.append(min(ends) - start - response_duration_sec)
    return statistics.median(values) if values else None


def _trial_windows(
    events: list[dict[str, Any]],
    *,
    manifest_metadata: dict[str, Any] | None = None,
    position_goals: list[dict[str, Any]] | None = None,
    active_position_goals: list[dict[str, Any]] | None = None,
    phase_end_delay_sec: float | None = None,
) -> list[dict[str, Any]]:
    """Build target-point windows from the recorder's T1 event markers."""

    starts = [event for event in events if event.get("event") in {"hold_start", "trajectory_start"}]
    windows: list[dict[str, Any]] = []
    # Both hardware and simulation runners publish each event a few times to
    # make the event marker robust to discovery/transient transport.  A bag
    # therefore contains a short burst of otherwise identical hold_start
    # messages.  Treat that burst as one semantic window rather than writing
    # duplicate metrics or overwriting the same target plot repeatedly.
    last_start_by_key: dict[tuple[object, ...], float] = {}
    duplicate_burst_sec = 1.0
    active_goals = active_position_goals or []
    for start in starts:
        target = _target_from_events([start])
        if target is None:
            continue
        metadata = start.get("metadata") if isinstance(start.get("metadata"), dict) else {}
        label = str(metadata.get("setpoint_id") or start.get("label") or f"target_{len(windows) + 1:02d}")
        timing_contract = metadata.get("timing_contract") if isinstance(metadata.get("timing_contract"), dict) else {}
        hold_start_mode = str(timing_contract.get("hold_start_mode", "experiment_event_v1"))
        if hold_start_mode == "first_controller_active_goal_tick_v1":
            controller_tick = _matching_position_goal_start(active_goals, target)
            if controller_tick is None:
                # A lockstep protocol must never fall back to the runner's
                # arrival-time event. The audit section will mark the session
                # needs_review with the missing controller-tick evidence.
                continue
            timestamp = controller_tick
            timing_provenance = "first_controller_active_goal_tick"
        else:
            timestamp = float(start["timestamp_sec"])
            timing_provenance = "hold_start_event"
        event_key = (
            str(start.get("event", "")),
            label,
            *(round(float(value), 9) for value in target),
        )
        previous_timestamp = last_start_by_key.get(event_key)
        if previous_timestamp is not None and timestamp - previous_timestamp <= duplicate_burst_sec:
            continue
        last_start_by_key[event_key] = timestamp
        end_event = next(
            (
                event
                for event in events
                if event.get("timestamp_sec", 0.0) > timestamp
                and event.get("event") in {"phase_end", "trial_end", "safety_abort"}
            ),
            None,
        )
        end_time = float(end_event["timestamp_sec"]) if end_event else None
        settings = _window_settings(metadata, timestamp, end_time)
        windows.append(
            {
                "label": label,
                "start_sec": timestamp,
                "end_sec": end_time,
                "target": target,
                "end_event": end_event.get("event") if end_event else None,
                "timing_provenance": timing_provenance,
                **settings,
            }
        )
    if windows:
        return windows

    # T1 simulation sessions collected before the recorder-ready fix can have
    # a complete controller-world target command and a normal phase_end while
    # all early event markers are absent.  Treat the recorded, target-matched
    # command as the same timing boundary for every method, but retain its
    # provenance in all derived products.
    metadata = manifest_metadata if isinstance(manifest_metadata, dict) else {}
    timing_contract = metadata.get("timing_contract") if isinstance(metadata.get("timing_contract"), dict) else {}
    if str(timing_contract.get("hold_start_mode", "")) == "first_controller_active_goal_tick_v1":
        return windows
    target = _target_from_metadata(metadata)
    start_sec = (
        _matching_position_goal_start(position_goals or [], target)
        if target is not None
        else None
    )
    if target is None:
        return windows
    label = str(metadata.get("setpoint_id") or "target_01")
    end_event = next(
        (
            event
            for event in events
            if (start_sec is None or float(event.get("timestamp_sec", 0.0)) > start_sec)
            and event.get("event") in {"phase_end", "trial_end", "safety_abort"}
        ),
        None,
    )
    end_time = float(end_event["timestamp_sec"]) if end_event else None
    if start_sec is not None:
        timing_provenance = "position_goal_inferred_missing_hold_start"
    else:
        response = metadata.get("response_window")
        response_duration = float(response.get("duration_sec", 0.0)) if isinstance(response, dict) else 0.0
        if end_event is None or end_event.get("event") != "phase_end" or phase_end_delay_sec is None:
            return windows
        start_sec = end_time - response_duration - phase_end_delay_sec
        timing_provenance = "phase_end_calibrated_missing_hold_start"
    settings = _window_settings(metadata, start_sec, end_time)
    return [{
        "label": label,
        "start_sec": start_sec,
        "end_sec": end_time,
        "target": target,
        "end_event": end_event.get("event") if end_event else None,
        "timing_provenance": timing_provenance,
        **settings,
    }]


def _select_control_pose_topic(
    poses: list[dict[str, Any]], *, required_topic: str | None = None
) -> str | None:
    """Select the controller-world pose used for T1 target comparisons.

    ``required_topic`` is used by localization-noise trials: the policy sees
    perturbed controller state, whereas metrics must use the parallel clean
    raw-fusion conversion recorded on the declared evaluation topic.
    """

    topics = {str(row["topic"]) for row in poses}
    if required_topic:
        return required_topic if required_topic in topics else None
    for preferred in ("/finsrov/controller/pose", "/sim/finsrov/controller/pose", "/finsrov/pose"):
        if preferred in topics:
            return preferred
    return sorted(topics)[0] if topics else None


def _t1_success_contract(metadata: dict[str, Any]) -> dict[str, float]:
    """Resolve the pose-only T1 completion contract recorded with a trial.

    Historical trials did not embed this contract in their event metadata, so
    the protocol defaults preserve the published T1 thresholds. Velocity and
    angular-rate thresholds are deliberately absent: success is a sustained
    pose-and-yaw condition, not a low-speed condition.
    """

    raw = metadata.get("t1_success_definition", {})
    raw = raw if isinstance(raw, dict) else {}
    limits = raw.get("all_controlled_errors_within", {})
    limits = limits if isinstance(limits, dict) else {}
    return {
        "x_m": float(limits.get("x_m", 0.10)),
        "y_m": float(limits.get("depth_m", 0.10)),
        "z_m": float(limits.get("z_m", 0.10)),
        "yaw_rad": math.radians(float(limits.get("yaw_deg", 10.0))),
        "continuous_hold_sec": float(raw.get("continuous_hold_sec", 10.0)),
        # A missing pose sample cannot demonstrate continuous hold. Match the
        # T1 primary-window interpolation limit unless recorded otherwise.
        "maximum_sample_gap_sec": float(raw.get("maximum_sample_gap_sec", 0.25)),
    }


def _first_t1_protocol_success_time(
    rows: list[dict[str, Any]],
    target: tuple[float, float, float, float],
    contract: dict[str, float],
) -> float | None:
    """Return the first sample completing the sustained pose-only T1 goal."""

    tx, ty, tz, tyaw = target
    entered_sec: float | None = None
    previous_sec: float | None = None
    for row in sorted(rows, key=lambda item: float(item["timestamp_sec"])):
        now_sec = float(row["timestamp_sec"])
        values = (row.get("x_m"), row.get("y_m"), row.get("z_m"), row.get("yaw_rad"))
        if any(value is None for value in values):
            entered_sec, previous_sec = None, now_sec
            continue
        yaw_error = _angle_error(values[3], tyaw)
        within = (
            abs(float(values[0]) - tx) <= contract["x_m"]
            and abs(float(values[1]) - ty) <= contract["y_m"]
            and abs(float(values[2]) - tz) <= contract["z_m"]
            and yaw_error is not None
            and abs(yaw_error) <= contract["yaw_rad"]
        )
        if (
            not within
            or (
                previous_sec is not None
                and now_sec - previous_sec > contract["maximum_sample_gap_sec"]
            )
        ):
            entered_sec = None
        elif entered_sec is None:
            entered_sec = now_sec
        if entered_sec is not None and now_sec - entered_sec >= contract["continuous_hold_sec"]:
            return now_sec
        previous_sec = now_sec
    return None


def _write_target_plots(
    derived: Path,
    poses: list[dict[str, Any]],
    windows: list[dict[str, Any]],
    pose_topic: str | None,
    goal_reached: list[dict[str, Any]],
    success_contract: dict[str, float],
) -> tuple[list[str], list[dict[str, Any]], list[dict[str, Any]], list[str]]:
    """Write one x/y/z/yaw plot per target and return samples/metrics/warnings."""

    if pose_topic is None:
        return [], [], [], ["no pose topic available for target-point plots"]
    selected = [row for row in poses if row["topic"] == pose_topic]
    if not selected or not windows:
        return [], [], [], ["no T1 target event windows were found; target-point plots not generated"]
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        return [], [], [], [f"matplotlib unavailable; target-point plots not generated: {exc}"]

    warnings: list[str] = []
    if pose_topic == "/finsrov/pose":
        warnings.append("target plots fell back to /finsrov/pose; verify it is expressed in controller_world")
    target_dir = derived / "plots" / "targets"
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    generated: list[str] = []
    for window in windows:
        start = float(window["start_sec"])
        recorded_end = float(window["end_sec"]) if window["end_sec"] is not None else max(row["timestamp_sec"] for row in selected)
        response_start = max(start, float(window.get("response_start_sec", start)))
        response_end = min(recorded_end, float(window.get("response_end_sec", recorded_end)))
        evaluation_start = max(response_start, float(window.get("evaluation_start_sec", response_start)))
        evaluation_end = min(response_end, float(window.get("evaluation_end_sec", response_end)))
        response_segment = [row for row in selected if response_start <= row["timestamp_sec"] <= response_end]
        if not response_segment:
            continue
        # Preserve the legacy controller status for audit, but calculate the
        # reported success from the pose-only protocol below. Earlier runs
        # used a speed gate inside the controller, which is not part of the
        # T1 success definition.
        reached_times = sorted(
            float(row["timestamp_sec"])
            for row in goal_reached
            if bool(row.get("reached")) and response_start <= float(row["timestamp_sec"]) <= response_end
        )
        first_reached = reached_times[0] if reached_times else None
        early_reached = first_reached is not None and first_reached < evaluation_start
        tx, ty, tz, tyaw = window["target"]
        first_success = _first_t1_protocol_success_time(
            response_segment, (tx, ty, tz, tyaw), success_contract
        )
        early_success = first_success is not None and first_success < evaluation_start
        safe_label = "".join(char if char.isalnum() or char in "-_" else "_" for char in window["label"]).strip("_") or "target"
        fig, axes = plt.subplots(4, 1, figsize=(7.0, 7.5), sharex=True)
        values = (("x_m", tx, "x [m]"), ("y_m", ty, "y [m]"), ("z_m", tz, "z [m]"), ("yaw_rad", tyaw, "yaw [deg]"))
        for axis_index, (key, target_value, axis_label) in enumerate(values):
            times = [(row["timestamp_sec"] - start) for row in response_segment]
            series: list[float | None] = []
            for row in response_segment:
                value = row.get(key)
                if key == "yaw_rad":
                    # Yaw is a circular quantity.  Do not subtract the two
                    # displayed Euler angles directly: +179 deg and -179 deg
                    # are physically only 2 deg apart.  For the plot, express
                    # the measured yaw in the target-centred branch so that a
                    # trace crossing the +/-180 deg display boundary remains
                    # visually continuous.  The same wrapped residual is used
                    # for RMSE and P95 below.
                    yaw_error = _angle_error(value, target_value)
                    value = math.degrees(target_value + yaw_error) if yaw_error is not None else None
                    target_plot = math.degrees(target_value)
                else:
                    target_plot = target_value
                series.append(value)
            axes[axis_index].plot(times, series, linewidth=0.8, label="measured")
            axes[axis_index].axhline(target_plot, color="tab:red", linestyle="--", linewidth=1.0, label="target")
            axes[axis_index].axvspan(
                evaluation_start - start,
                evaluation_end - start,
                color="tab:green",
                alpha=0.12,
                label="primary evaluation window" if axis_index == 0 else None,
            )
            if first_success is not None:
                axes[axis_index].axvline(
                    first_success - start,
                    color="tab:blue",
                    linestyle=":",
                    linewidth=1.3,
                    label="protocol success" if axis_index == 0 else None,
                )
            axes[axis_index].set_ylabel(axis_label)
            axes[axis_index].grid(True, alpha=0.3)
        axes[0].legend(fontsize=8)
        axes[-1].set_xlabel("time after hold start [s]")
        title_suffix = "before primary window" if early_success else "during response"
        success_label = "none" if first_success is None else title_suffix
        fig.suptitle(
            "T1 response (green: primary 40--60 s; blue: pose-only protocol "
            f"success {success_label})"
        )
        fig.tight_layout()
        path = target_dir / f"{safe_label}_xyzyaw.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(derived)))

        metric_segment, expected_samples = _resample_pose_segment(
            response_segment,
            start_sec=evaluation_start,
            end_sec=evaluation_end,
            rate_hz=float(window.get("resample_hz", 0.0)),
            max_interpolation_gap_sec=float(window.get("max_interpolation_gap_sec", 0.0)),
        )
        valid_fraction = len(metric_segment) / expected_samples if expected_samples else 0.0
        minimum_valid_fraction = float(window.get("minimum_valid_fraction", 0.0))
        metric: dict[str, Any] = {
            "target_id": window["label"],
            "pose_topic": pose_topic,
            "response_start_sec": response_start,
            "response_end_sec": response_end,
            "evaluation_start_sec": evaluation_start,
            "evaluation_end_sec": evaluation_end,
            "evaluation_protocol": window.get("protocol_version", "unknown"),
            "resample_hz": float(window.get("resample_hz", 0.0)),
            "samples": len(metric_segment),
            "expected_samples": expected_samples,
            "valid_fraction": valid_fraction,
            "minimum_valid_fraction": minimum_valid_fraction,
            "metrics_valid": valid_fraction >= minimum_valid_fraction,
            "timing_provenance": str(window.get("timing_provenance", "unknown")),
            "goal_reached_time_after_hold_start_sec": first_reached - start if first_reached is not None else None,
            "goal_reached_before_primary_window": early_reached,
            "protocol_success": first_success is not None,
            "protocol_success_time_after_hold_start_sec": first_success - start if first_success is not None else None,
            "protocol_success_before_primary_window": early_success,
            "protocol_success_hold_sec": success_contract["continuous_hold_sec"],
            "protocol_success_position_tolerance_m": success_contract["x_m"],
            "protocol_success_yaw_tolerance_rad": success_contract["yaw_rad"],
        }
        target_by_key = {key: target for key, target, _ in values}
        for key, _, _ in values:
            errors: list[float] = []
            for row in metric_segment:
                value = row.get(key)
                if key == "yaw_rad":
                    error = _angle_error(value, tyaw)
                else:
                    error = float(value) - float(target_by_key[key]) if value is not None else None
                if error is not None:
                    errors.append(error)
            metric[f"{key}_rmse"] = math.sqrt(sum(item * item for item in errors) / len(errors)) if errors else None
            metric[f"{key}_p95_abs"] = sorted(abs(item) for item in errors)[max(0, math.ceil(0.95 * len(errors)) - 1)] if errors else None
            metric[f"{key}_signed_error_mean"] = statistics.fmean(errors) if errors else None
            metric[f"{key}_abs_error_mean"] = statistics.fmean(abs(item) for item in errors) if errors else None
        metrics.append(metric)
        for row in response_segment:
            rows.append({
                "target_id": window["label"],
                "phase": "primary_evaluation" if evaluation_start <= row["timestamp_sec"] <= evaluation_end else "response",
                "time_after_hold_start_sec": row["timestamp_sec"] - start,
                "timestamp_sec": row["timestamp_sec"],
                "x_m": row.get("x_m"),
                "y_m": row.get("y_m"),
                "z_m": row.get("z_m"),
                "yaw_deg": math.degrees(row["yaw_rad"]) if row.get("yaw_rad") is not None else None,
                "yaw_error_deg": (
                    math.degrees(_angle_error(row.get("yaw_rad"), tyaw))
                    if _angle_error(row.get("yaw_rad"), tyaw) is not None
                    else None
                ),
                "target_x_m": tx,
                "target_y_m": ty,
                "target_z_m": tz,
                "target_yaw_deg": math.degrees(tyaw),
            })
    return generated, metrics, rows, warnings
    with path.open(encoding="utf-8", newline="") as stream:
        row = next(csv.DictReader(stream), None)
    if not row:
        return None
    try:
        return float(row["truth_x_m"]), float(row["truth_y_m"])
    except (KeyError, TypeError, ValueError):
        return None


def _trajectory_xy_poses(poses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return poses that share the pool-world horizontal ``x-y`` plane.

    ``/finsrov/controller/pose`` is deliberately excluded here.  Its second
    coordinate is the Unity/controller vertical axis (depth), not the
    pool-world horizontal ``y`` axis.  T1 controller-frame responses are
    instead represented by the per-target ``x/y/z/yaw`` plots; T2 has its own
    frame-aware trajectory plotter.
    """

    return [row for row in poses if "/controller/pose" not in str(row.get("topic", ""))]


def _plot(derived: Path, poses: list[dict[str, Any]], commands: list[dict[str, Any]], events: list[dict[str, Any]], *, horizontal_only: bool = False) -> tuple[list[str], list[str]]:
    generated: list[str] = []
    warnings: list[str] = []
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        warnings.append(f"matplotlib unavailable; no PNG plots generated: {exc}")
        return generated, warnings
    plots = derived / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    trajectory_poses = _trajectory_xy_poses(poses)
    if trajectory_poses:
        fig, ax = plt.subplots(figsize=(6.0, 4.5))
        for topic in sorted({row["topic"] for row in trajectory_poses}):
            subset = [row for row in trajectory_poses if row["topic"] == topic]
            ax.plot([row["x_m"] for row in subset], [row["y_m"] for row in subset], label=topic)
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        ax.set_title("Recorded horizontal trajectory")
        ax.grid(True, alpha=0.3)
        if len({row["topic"] for row in trajectory_poses}) <= 4:
            ax.legend(fontsize=7)
        fig.tight_layout()
        path = plots / "trajectory_xy.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(derived)))

    if poses:
        if not horizontal_only:
            fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
            time0 = min(row["timestamp_sec"] for row in poses)
            t = [(row["timestamp_sec"] - time0) for row in poses]
            for key, axis, label in (("x_m", axes[0], "x [m]"), ("y_m", axes[1], "y [m]"), ("z_m", axes[2], "z [m]")):
                axis.plot(t, [row[key] for row in poses], linewidth=0.8)
                axis.set_ylabel(label)
                axis.grid(True, alpha=0.3)
            axes[-1].set_xlabel("time [s]")
            fig.suptitle("Recorded pose components")
            fig.tight_layout()
            path = plots / "pose_components.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            generated.append(str(path.relative_to(derived)))
    if commands:
        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        time0 = min(row["timestamp_sec"] for row in commands)
        for index in range(max(len(row["values"]) for row in commands)):
            subset = [row for row in commands if len(row["values"]) > index]
            ax.plot([row["timestamp_sec"] - time0 for row in subset], [row["values"][index] for row in subset], linewidth=0.7, label=f"u[{index}]")
        ax.set_xlabel("time [s]")
        ax.set_ylabel("command (native units)")
        ax.set_title("Recorded actuator/control command")
        ax.grid(True, alpha=0.3)
        if max(len(row["values"]) for row in commands) <= 8:
            ax.legend(ncol=2, fontsize=7)
        fig.tight_layout()
        path = plots / "commands.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(derived)))
    if not generated and events:
        fig, ax = plt.subplots(figsize=(7.0, 2.5))
        t0 = events[0]["timestamp_sec"]
        for event in events:
            x = event["timestamp_sec"] - t0
            ax.axvline(x, color="tab:blue", alpha=0.5)
            ax.text(x, 0.5, event["event"], rotation=90, va="center", fontsize=7)
        ax.set_yticks([])
        ax.set_xlabel("time [s]")
        ax.set_title("Experiment events")
        fig.tight_layout()
        path = plots / "events.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        generated.append(str(path.relative_to(derived)))
    return generated, warnings


def _coverage_audit(
    rows: list[dict[str, Any]],
    *,
    start_sec: float,
    end_sec: float,
    edge_tolerance_sec: float,
    max_gap_sec: float,
) -> dict[str, Any]:
    """Audit a header-stamped stream against one simulation-time interval."""

    timestamps = sorted(float(row["timestamp_sec"]) for row in rows)
    if not timestamps:
        return {
            "samples": 0,
            "first_sec": None,
            "last_sec": None,
            "max_gap_sec": None,
            "passes": False,
            "reason": "no_samples",
        }
    in_or_near_window = [value for value in timestamps if start_sec - edge_tolerance_sec <= value <= end_sec + edge_tolerance_sec]
    if not in_or_near_window:
        return {
            "samples": len(timestamps),
            "first_sec": timestamps[0],
            "last_sec": timestamps[-1],
            "max_gap_sec": None,
            "passes": False,
            "reason": "no_samples_near_response_window",
        }
    gaps = [right - left for left, right in zip(in_or_near_window, in_or_near_window[1:])]
    max_gap = max(gaps) if gaps else 0.0
    starts_in_time = in_or_near_window[0] <= start_sec + edge_tolerance_sec
    ends_in_time = in_or_near_window[-1] >= end_sec - edge_tolerance_sec
    passes = starts_in_time and ends_in_time and max_gap <= max_gap_sec
    reason = "ok" if passes else ",".join(
        part
        for part, condition in (
            ("late_start", not starts_in_time),
            ("early_end", not ends_in_time),
            ("gap_exceeds_limit", max_gap > max_gap_sec),
        )
        if condition
    )
    return {
        "samples": len(in_or_near_window),
        "first_sec": in_or_near_window[0],
        "last_sec": in_or_near_window[-1],
        "max_gap_sec": max_gap,
        "passes": passes,
        "reason": reason,
    }


def _t1_lockstep_alignment_audit(
    *,
    metadata: dict[str, Any],
    poses: list[dict[str, Any]],
    active_position_goals: list[dict[str, Any]],
    stamped_commands: list[dict[str, Any]],
    lockstep_acks: list[dict[str, Any]],
) -> dict[str, Any] | None:
    """Validate the evidence needed to trust a 10x lockstep T1 trial.

    No conclusion is drawn from a nominally complete rosbag unless the state,
    controller-applied target, stamped command and Unity--ROS2 tick handoff
    all cover the same response interval.  This deliberately invalidates old
    free-running T1 bags whose commands were recorded before their goal.
    """

    clock = metadata.get("simulation_clock") if isinstance(metadata.get("simulation_clock"), dict) else {}
    timing = metadata.get("timing_contract") if isinstance(metadata.get("timing_contract"), dict) else {}
    if (
        str(clock.get("mode", "")) != "ros2_control_lockstep"
        or str(timing.get("hold_start_mode", "")) != "first_controller_active_goal_tick_v1"
    ):
        return None
    target = _target_from_metadata(metadata)
    response = metadata.get("response_window") if isinstance(metadata.get("response_window"), dict) else {}
    duration_sec = max(0.0, float(response.get("duration_sec", 0.0)))
    if target is None or duration_sec <= 0.0:
        return {
            "required": True,
            "passes": False,
            "reason": "missing_target_or_response_duration",
        }
    start_sec = _matching_position_goal_start(active_position_goals, target)
    if start_sec is None:
        return {
            "required": True,
            "passes": False,
            "reason": "missing_matching_controller_active_goal",
            "target": list(target),
        }
    end_sec = start_sec + duration_sec
    pose_topic = _select_control_pose_topic(poses)
    control_poses = [row for row in poses if row.get("topic") == pose_topic] if pose_topic else []
    audits = {
        "pose": _coverage_audit(
            control_poses,
            start_sec=start_sec,
            end_sec=end_sec,
            edge_tolerance_sec=0.15,
            max_gap_sec=0.25,
        ),
        "stamped_controller_command": _coverage_audit(
            stamped_commands,
            start_sec=start_sec,
            end_sec=end_sec,
            edge_tolerance_sec=0.15,
            max_gap_sec=0.25,
        ),
        "lockstep_ack": _coverage_audit(
            lockstep_acks,
            start_sec=start_sec,
            end_sec=end_sec,
            edge_tolerance_sec=0.06,
            max_gap_sec=0.10,
        ),
    }
    failed = [name for name, value in audits.items() if not bool(value["passes"])]
    return {
        "required": True,
        "passes": not failed,
        "response_start_sec": start_sec,
        "response_end_sec": end_sec,
        "timing_authority": "controller_active_pose_header_and_unity_ros2_ack",
        "pose_topic": pose_topic,
        "streams": audits,
        "failed_streams": failed,
    }


def analyze_session(session_dir: Path, *, force: bool = False) -> dict[str, Any]:
    """Analyse one completed recorder session and return its report."""

    session_dir = session_dir.expanduser().resolve()
    manifest_path = session_dir / "manifest.json"
    manifest = read_json(manifest_path)
    derived = session_dir / "derived"
    derived.mkdir(parents=True, exist_ok=True)
    horizontal_only = str(manifest.get("profile", "")).lower() == "apriltag" or str(manifest.get("experiment_id", "")) in {"E1", "E2"}
    report: dict[str, Any] = {
        "schema_version": 1,
        "analyzer": "experiment_recorder.analysis",
        "started_at": now_iso(),
        "session_id": manifest.get("session_id", session_dir.name),
        "experiment_id": manifest.get("experiment_id"),
        "profile": manifest.get("profile"),
        "horizontal_only": horizontal_only,
        "status": "complete",
        "warnings": [],
        "plots": [],
    }
    log_stage(
        f"analysis processing: session={report['session_id']} profile={report['profile']} "
        f"raw={session_dir / 'raw'}"
    )
    try:
        (
            poses,
            commands,
            position_goals,
            active_position_goals,
            stamped_commands,
            lockstep_acks,
            goal_reached,
            events,
            topics,
        ) = _read_bag(session_dir)
        report["bag_topics"] = topics
        report["sample_counts"] = {
            "pose": len(poses),
            "commands": len(commands),
            "position_goals": len(position_goals),
            "active_position_goals": len(active_position_goals),
            "stamped_commands": len(stamped_commands),
            "lockstep_acks": len(lockstep_acks),
            "goal_reached": len(goal_reached),
            "events": len(events),
        }
        if horizontal_only:
            pose_rows = [{key: row[key] for key in ("timestamp_sec", "topic", "frame_id", "x_m", "y_m")} for row in poses]
            _write_csv(derived / "pose_samples.csv", pose_rows, ["timestamp_sec", "topic", "frame_id", "x_m", "y_m"])
        else:
            _write_csv(
                derived / "pose_samples.csv",
                poses,
                ["timestamp_sec", "topic", "frame_id", "x_m", "y_m", "z_m", "yaw_rad"],
            )
        command_rows = [{"timestamp_sec": row["timestamp_sec"], "topic": row["topic"], "values_json": json.dumps(row["values"])} for row in commands]
        _write_csv(derived / "command_samples.csv", command_rows, ["timestamp_sec", "topic", "values_json"])
        _write_csv(
            derived / "position_goal_samples.csv",
            position_goals,
            ["timestamp_sec", "topic", "frame_id", "x_m", "y_m", "z_m", "yaw_rad"],
        )
        _write_csv(
            derived / "active_position_goal_samples.csv",
            active_position_goals,
            ["timestamp_sec", "topic", "frame_id", "x_m", "y_m", "z_m", "yaw_rad"],
        )
        _write_csv(
            derived / "stamped_command_samples.csv",
            [
                {
                    "timestamp_sec": row["timestamp_sec"],
                    "topic": row["topic"],
                    "frame_id": row.get("frame_id", ""),
                    "values_json": json.dumps(row["values"]),
                }
                for row in stamped_commands
            ],
            ["timestamp_sec", "topic", "frame_id", "values_json"],
        )
        _write_csv(
            derived / "lockstep_ack_samples.csv",
            lockstep_acks,
            ["timestamp_sec", "topic", "tick_ns"],
        )
        _write_csv(
            derived / "goal_reached_samples.csv",
            goal_reached,
            ["timestamp_sec", "topic", "reached"],
        )
        (derived / "events.json").write_text(json.dumps(events, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        truth = _truth_xy(session_dir)
        if truth and poses:
            per_topic: dict[str, dict[str, Any]] = {}
            metric_rows: list[dict[str, Any]] = []
            for topic in sorted({row["topic"] for row in poses}):
                topic_poses = [row for row in poses if row["topic"] == topic]
                errors = [math.hypot(row["x_m"] - truth[0], row["y_m"] - truth[1]) for row in topic_poses]
                metrics = {
                    "truth_x_m": truth[0],
                    "truth_y_m": truth[1],
                    "rmse_m": math.sqrt(sum(item * item for item in errors) / len(errors)),
                    "p95_m": sorted(errors)[max(0, math.ceil(0.95 * len(errors)) - 1)],
                    "mean_m": statistics.fmean(errors),
                    "samples": len(errors),
                }
                per_topic[topic] = metrics
                metric_rows.append({"topic": topic, **metrics})
            report["horizontal_truth"] = {"truth_x_m": truth[0], "truth_y_m": truth[1], "by_topic": per_topic}
            _write_csv(derived / "metrics.csv", metric_rows, ["topic", "truth_x_m", "truth_y_m", "rmse_m", "p95_m", "mean_m", "samples"])
        session_metadata = manifest.get("metadata") if isinstance(manifest.get("metadata"), dict) else {}
        response = session_metadata.get("response_window") if isinstance(session_metadata, dict) else None
        response_duration = float(response.get("duration_sec", 0.0)) if isinstance(response, dict) else 0.0
        # E8 uses a detection/peak/recovery timing contract, not the static
        # T1 hold-start evaluation window. Its dedicated analyser below owns
        # all target-relative metrics and plots.
        is_t1_disturbance = str(session_metadata.get("analysis_kind", "")) == "t1_disturbance_recovery"
        target_windows = (
            _trial_windows(
                events,
                manifest_metadata=session_metadata,
                position_goals=position_goals,
                active_position_goals=active_position_goals,
                phase_end_delay_sec=_run_calibrated_hold_end_delay(session_dir, response_duration),
            )
            if not horizontal_only and not is_t1_disturbance
            else []
        )
        # In both localization-noise and no-EKF state-input ablations, the
        # controller state is intentionally perturbed/replaced while metrics
        # must be computed on the parallel normal-EKF evaluation stream.
        ablation_metadata = session_metadata.get("localization_noise_ablation")
        if not isinstance(ablation_metadata, dict):
            ablation_metadata = session_metadata.get("state_estimation_ablation")
        required_evaluation_pose_topic = (
            str(ablation_metadata.get("evaluation_pose_topic", "")).strip()
            if isinstance(ablation_metadata, dict)
            else ""
        )
        target_pose_topic = _select_control_pose_topic(
            poses,
            required_topic=required_evaluation_pose_topic or None,
        )
        success_contract = _t1_success_contract(session_metadata)
        target_plot_paths, target_metrics, target_samples, target_warnings = _write_target_plots(
            derived, poses, target_windows, target_pose_topic, goal_reached, success_contract
        )
        if required_evaluation_pose_topic and target_pose_topic is None:
            target_warnings.append(
                "ablation evaluation pose is missing from rosbag; refusing to fall back to controller "
                f"controller pose: {required_evaluation_pose_topic}"
            )
        if target_metrics:
            report["target_point_metrics"] = target_metrics
            _write_csv(
                derived / "tracking_metrics.csv",
                [
                    {
                        "target_id": metric["target_id"],
                        "pose_topic": metric["pose_topic"],
                        "x_rmse_m": metric.get("x_m_rmse"),
                        "x_p95_abs_m": metric.get("x_m_p95_abs"),
                        "x_signed_error_mean_m": metric.get("x_m_signed_error_mean"),
                        "x_abs_error_mean_m": metric.get("x_m_abs_error_mean"),
                        "y_rmse_m": metric.get("y_m_rmse"),
                        "y_p95_abs_m": metric.get("y_m_p95_abs"),
                        "y_signed_error_mean_m": metric.get("y_m_signed_error_mean"),
                        "y_abs_error_mean_m": metric.get("y_m_abs_error_mean"),
                        "z_rmse_m": metric.get("z_m_rmse"),
                        "z_p95_abs_m": metric.get("z_m_p95_abs"),
                        "z_signed_error_mean_m": metric.get("z_m_signed_error_mean"),
                        "z_abs_error_mean_m": metric.get("z_m_abs_error_mean"),
                        "yaw_rmse_rad": metric.get("yaw_rad_rmse"),
                        "yaw_p95_abs_rad": metric.get("yaw_rad_p95_abs"),
                        "yaw_signed_error_mean_rad": metric.get("yaw_rad_signed_error_mean"),
                        "yaw_abs_error_mean_rad": metric.get("yaw_rad_abs_error_mean"),
                        "evaluation_start_sec": metric.get("evaluation_start_sec"),
                        "evaluation_end_sec": metric.get("evaluation_end_sec"),
                        "resample_hz": metric.get("resample_hz"),
                        "samples": metric.get("samples"),
                        "expected_samples": metric.get("expected_samples"),
                        "valid_fraction": metric.get("valid_fraction"),
                        "metrics_valid": metric.get("metrics_valid"),
                        "goal_reached_time_after_hold_start_sec": metric.get("goal_reached_time_after_hold_start_sec"),
                        "goal_reached_before_primary_window": metric.get("goal_reached_before_primary_window"),
                        "protocol_success": metric.get("protocol_success"),
                        "protocol_success_time_after_hold_start_sec": metric.get("protocol_success_time_after_hold_start_sec"),
                        "protocol_success_before_primary_window": metric.get("protocol_success_before_primary_window"),
                        "protocol_success_hold_sec": metric.get("protocol_success_hold_sec"),
                        "protocol_success_position_tolerance_m": metric.get("protocol_success_position_tolerance_m"),
                        "protocol_success_yaw_tolerance_rad": metric.get("protocol_success_yaw_tolerance_rad"),
                        "timing_provenance": metric.get("timing_provenance"),
                    }
                    for metric in target_metrics
                ],
                ["target_id", "pose_topic", "x_rmse_m", "x_p95_abs_m", "x_signed_error_mean_m", "x_abs_error_mean_m", "y_rmse_m", "y_p95_abs_m", "y_signed_error_mean_m", "y_abs_error_mean_m", "z_rmse_m", "z_p95_abs_m", "z_signed_error_mean_m", "z_abs_error_mean_m", "yaw_rmse_rad", "yaw_p95_abs_rad", "yaw_signed_error_mean_rad", "yaw_abs_error_mean_rad", "evaluation_start_sec", "evaluation_end_sec", "resample_hz", "samples", "expected_samples", "valid_fraction", "metrics_valid", "goal_reached_time_after_hold_start_sec", "goal_reached_before_primary_window", "protocol_success", "protocol_success_time_after_hold_start_sec", "protocol_success_before_primary_window", "protocol_success_hold_sec", "protocol_success_position_tolerance_m", "protocol_success_yaw_tolerance_rad", "timing_provenance"],
            )
            _write_csv(
                derived / "target_point_samples.csv",
                [
                    {
                        key: row.get(key)
                        for key in (
                            "target_id", "phase", "time_after_hold_start_sec", "timestamp_sec",
                            "x_m", "y_m", "z_m", "yaw_deg", "target_x_m", "target_y_m",
                            "yaw_error_deg", "target_z_m", "target_yaw_deg",
                        )
                    }
                    for row in target_samples
                ],
                ["target_id", "phase", "time_after_hold_start_sec", "timestamp_sec", "x_m", "y_m", "z_m", "yaw_deg", "yaw_error_deg", "target_x_m", "target_y_m", "target_z_m", "target_yaw_deg"],
            )
        report["target_point_plot_count"] = len(target_plot_paths)
        report["warnings"].extend(target_warnings)
        lockstep_audit = _t1_lockstep_alignment_audit(
            metadata=session_metadata,
            poses=poses,
            active_position_goals=active_position_goals,
            stamped_commands=stamped_commands,
            lockstep_acks=lockstep_acks,
        )
        if lockstep_audit is not None:
            report["t1_lockstep_alignment"] = lockstep_audit
            if not bool(lockstep_audit.get("passes", False)):
                report["status"] = "needs_review"
                failed_streams = lockstep_audit.get("failed_streams", [])
                suffix = f" failed_streams={failed_streams}" if failed_streams else ""
                report["warnings"].append(
                    "T1 lockstep timestamp audit failed; do not use this trial for controller comparison."
                    + suffix
                )
        report["duration_sec"] = ((max(row["timestamp_sec"] for row in poses + commands + events) - min(row["timestamp_sec"] for row in poses + commands + events)) if poses or commands or events else 0.0)
        if commands:
            effort: dict[str, float] = {}
            for topic in sorted({row["topic"] for row in commands}):
                subset = [row for row in commands if row["topic"] == topic]
                total = 0.0
                for previous, current in zip(subset, subset[1:]):
                    dt = max(0.0, float(current["timestamp_sec"]) - float(previous["timestamp_sec"]))
                    # This is a dimensionless command effort unless the topic
                    # explicitly carries physical units; no force/RPM claim is
                    # made by the generic analyser.
                    total += dt * sum(float(value) ** 2 for value in previous["values"])
                effort[topic] = total
            report["command_effort_integral_native_squared_sec"] = effort
        base_plots, plot_warnings = _plot(derived, poses, commands, events, horizontal_only=horizontal_only)
        t2_plots: list[str] = []
        t2_warnings: list[str] = []
        if str(manifest.get("profile", "")).lower() in {"t2", "t2_sim"}:
            from .t2_analysis import augment_t2_report

            t2_plots, t2_warnings = augment_t2_report(session_dir, derived, poses, commands, events, report)
            # Generic command/pose plots do not prove that a trajectory trial
            # has an auditable reference-versus-state alignment.  Keep the
            # raw bag valid, but make the missing T2 evidence visible to the
            # runner instead of reporting an apparently complete experiment.
            if "t2" not in report:
                report["status"] = "needs_review"
                t2_warnings.append(
                    "T2-specific reference alignment was not produced; generic plots are diagnostic only"
                )
        disturbance_plots: list[str] = []
        disturbance_warnings: list[str] = []
        if str(session_metadata.get("analysis_kind", "")) == "t1_disturbance_recovery":
            from .t1_disturbance_analysis import augment_t1_disturbance_report

            disturbance_plots, disturbance_warnings = augment_t1_disturbance_report(
                session_dir,
                derived,
                poses,
                events,
                report,
            )
            # An E8 session may legitimately contain no manually applied
            # perturbation, but it must never look like a successful recovery
            # experiment. Keep the raw recording while making that distinction
            # explicit in the report and manifest.
            disturbance = report.get("t1_disturbance_recovery", {})
            if not isinstance(disturbance, dict) or not bool(disturbance.get("detected", False)):
                report["status"] = "needs_review"
        report["plots"] = target_plot_paths + base_plots + t2_plots + disturbance_plots
        report["warnings"].extend(plot_warnings)
        report["warnings"].extend(t2_warnings)
        report["warnings"].extend(disturbance_warnings)
        if not poses and not commands and not events:
            report["status"] = "needs_review"
            report["warnings"].append("bag contains no recognized pose, command, or event samples")
    except Exception as exc:
        report["status"] = "needs_review"
        report["error"] = repr(exc)
    report["finished_at"] = now_iso()
    write_json(derived / "analysis_report.json", report)
    log_stage(
        f"analysis artifact written: session={report['session_id']} status={report['status']} "
        f"report={derived / 'analysis_report.json'} plots={len(report['plots'])}"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyse a completed FinsROV recorder session and generate derived CSV/plots.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--force", action="store_true", help="re-run even when analysis_report.json already exists")
    args = parser.parse_args()
    report_path = args.session_dir.expanduser().resolve() / "derived" / "analysis_report.json"
    if report_path.is_file() and not args.force:
        print(report_path)
        return
    report = analyze_session(args.session_dir, force=args.force)
    print(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
