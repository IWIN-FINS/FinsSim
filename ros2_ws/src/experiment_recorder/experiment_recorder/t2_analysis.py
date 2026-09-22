"""T2 trajectory-reference alignment, metrics, and plots.

The controller publishes its applied trajectory reference as
``[elapsed, x, y, z, vx, vy, vz, duration]``.  This module uses that runtime
stream rather than guessing reference timing from the sender command.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
from pathlib import Path
from statistics import fmean, median
from typing import Any

from .analysis import _stamp_seconds
from .manifest import write_json


LEGACY_REFERENCE_TOPICS = frozenset({
    "/motion_controller/debug/trajectory_reference",
    "/sim/motion_controller/debug/trajectory_reference",
})
STAMPED_REFERENCE_TOPICS = frozenset({
    "/sim/motion_controller/debug/trajectory_reference_stamped",
})
LEGACY_STATE_STATUS_TOPICS = frozenset({
    "/finsrov/controller/state/status",
    "/sim/finsrov/controller/state/status",
})
STAMPED_STATE_STATUS_TOPICS = frozenset({
    "/sim/finsrov/controller/state/status_stamped",
})


def _read_t2_runtime(session_dir: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    bag_dir = session_dir / "raw" / "rosbag2"
    try:
        import rosbag2_py  # type: ignore
        from rclpy.serialization import deserialize_message  # type: ignore
        from rosidl_runtime_py.utilities import get_message  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(f"ROS2 bag analysis dependencies are unavailable: {exc}") from exc
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    types = {item.name: item.type for item in reader.get_all_topics_and_types()}
    message_types = {topic: get_message(type_name) for topic, type_name in types.items()}
    legacy_reference: list[dict[str, Any]] = []
    stamped_reference: list[dict[str, Any]] = []
    legacy_status: list[dict[str, Any]] = []
    stamped_status: list[dict[str, Any]] = []
    while reader.has_next():
        topic, serialized, timestamp_ns = reader.read_next()
        if topic not in (
            LEGACY_REFERENCE_TOPICS
            | STAMPED_REFERENCE_TOPICS
            | LEGACY_STATE_STATUS_TOPICS
            | STAMPED_STATE_STATUS_TOPICS
        ):
            continue
        message = deserialize_message(serialized, message_types[topic])
        timestamp = _stamp_seconds(message, timestamp_ns)
        if topic in LEGACY_REFERENCE_TOPICS:
            try:
                data = [float(value) for value in message.data]
            except (AttributeError, TypeError, ValueError):
                continue
            if len(data) < 8:
                continue
            legacy_reference.append({
                "timestamp_sec": timestamp, "elapsed_sec": data[0],
                "ref_x_m": data[1], "ref_y_m": data[2], "ref_z_m": data[3],
                "ref_vx_mps": data[4], "ref_vy_mps": data[5], "ref_vz_mps": data[6],
                "duration_sec": data[7],
                "timing_source": "rosbag_receive_time_legacy_array",
            })
        elif topic in STAMPED_REFERENCE_TOPICS:
            try:
                position = message.pose.position
                stamped_reference.append({
                    "timestamp_sec": timestamp,
                    # Filled from the frozen trajectory_start event below.
                    "elapsed_sec": math.nan,
                    "ref_x_m": float(position.x),
                    "ref_y_m": float(position.y),
                    "ref_z_m": float(position.z),
                    "ref_vx_mps": 0.0,
                    "ref_vy_mps": 0.0,
                    "ref_vz_mps": 0.0,
                    "duration_sec": math.nan,
                    "timing_source": "message_header_stamped",
                })
            except (AttributeError, TypeError, ValueError):
                continue
        elif topic in LEGACY_STATE_STATUS_TOPICS:
            try:
                payload = json.loads(str(message.data))
            except (AttributeError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                legacy_status.append({"timestamp_sec": timestamp, **payload})
        elif topic in STAMPED_STATE_STATUS_TOPICS:
            try:
                payload = json.loads(str(message.data))
            except (AttributeError, TypeError, json.JSONDecodeError):
                continue
            if isinstance(payload, dict):
                stamped_status.append({"timestamp_sec": timestamp, **payload})
    # The stamped stream is authoritative for accelerated Unity evaluation:
    # its Header is on the same /clock timebase as the controller-world pose.
    # Retain the legacy Float32MultiArray path for hardware and old bags.
    return (
        stamped_reference if stamped_reference else legacy_reference,
        stamped_status if stamped_status else legacy_status,
    )


def _resolve_reference_timing(
    samples: list[dict[str, Any]], *, start_sec: float, duration_sec: float
) -> list[dict[str, Any]]:
    """Fill elapsed/duration for a header-stamped T2 reference stream.

    The legacy Float32MultiArray is already self-timed.  PoseStamped carries
    the trustworthy simulation clock but intentionally only the geometric
    reference, so derive elapsed from the frozen trajectory_start marker.
    """

    resolved: list[dict[str, Any]] = []
    for sample in samples:
        row = dict(sample)
        if str(row.get("timing_source", "")) == "message_header_stamped":
            row["elapsed_sec"] = max(0.0, float(row["timestamp_sec"]) - float(start_sec))
            row["duration_sec"] = float(duration_sec)
        resolved.append(row)
    return resolved


def _event_window(events: list[dict[str, Any]], *, duration_sec: float) -> tuple[float, float]:
    """Resolve the frozen T2 metric window from runtime events and metadata.

    T2 is a tracking task, so its primary metric intentionally includes the
    initial tracking transient.  The window starts at the recorded
    ``trajectory_start`` event, ends at the configured trajectory duration,
    and is shortened only by an abort/early terminal event.
    """

    start = next((event for event in events if event.get("event") == "trajectory_start"), None)
    if start is None:
        raise ValueError("T2 session is missing trajectory_start event")
    start_sec = float(start["timestamp_sec"])
    if duration_sec <= 0.0:
        raise ValueError("T2 evaluation duration_sec must be positive")
    end = next(
        (
            event for event in events
            if event.get("timestamp_sec", 0.0) > start_sec
            and event.get("event") in {"trajectory_end", "safety_abort", "operator_abort", "trial_end"}
        ),
        None,
    )
    configured_end_sec = start_sec + duration_sec
    terminal_end_sec = float(end["timestamp_sec"]) if end else configured_end_sec
    return start_sec, min(configured_end_sec, terminal_end_sec)


def _lockstep_controller_window(
    samples: list[dict[str, Any]], *, duration_sec: float
) -> tuple[float, float]:
    """Resolve a T2 window from the controller's first exact lockstep tick.

    The simulation runner is intentionally not part of the controller's ACK
    path.  At accelerated rates its ordinary ``use_sim_time`` callback can be
    behind the exact ``/clock`` value that caused the controller to apply a
    trajectory.  New lockstep trials therefore declare
    ``first_controller_tick_v1`` and use the first stamped runtime reference
    (which is published with the controller's exact tick timestamp) as t=0.
    The immutable ``trajectory_start`` recorder event is still required by
    the completion audit; it is simply not used to interpolate across two
    clocks with different delivery latency.
    """

    if duration_sec <= 0.0:
        raise ValueError("T2 evaluation duration_sec must be positive")
    timestamps = [
        float(row["timestamp_sec"])
        for row in samples
        if str(row.get("timing_source", "")) == "message_header_stamped"
        and math.isfinite(float(row.get("timestamp_sec", math.nan)))
    ]
    if not timestamps:
        raise ValueError("lockstep T2 session has no stamped controller reference")
    start_sec = min(timestamps)
    return start_sec, start_sec + duration_sec


def _runtime_reference_inferred_window(
    samples: list[dict[str, float]],
    events: list[dict[str, Any]],
    *,
    duration_sec: float,
) -> tuple[float, float]:
    """Recover a plotting window when a legacy bag lacks trajectory_start.

    The controller's runtime reference carries both ROS timestamp and elapsed
    trajectory time.  Their difference is the actual start-time estimate.  A
    median over the recorded stream is robust to the final held reference.
    This is a recovery path for retained trials: the controller's own runtime
    reference is the timing authority, while callers preserve an explicit
    inferred-timing provenance instead of silently treating it as a marker.
    """

    if duration_sec <= 0.0:
        raise ValueError("T2 evaluation duration_sec must be positive")
    candidates = [
        float(row["timestamp_sec"]) - float(row["elapsed_sec"])
        for row in samples
        if math.isfinite(float(row.get("timestamp_sec", math.nan)))
        and math.isfinite(float(row.get("elapsed_sec", math.nan)))
        and -1e-6 <= float(row["elapsed_sec"]) <= duration_sec + 1.0
    ]
    if not candidates:
        raise ValueError("T2 session is missing trajectory_start and has no usable runtime reference")
    start_sec = median(candidates)
    terminal = next(
        (
            event
            for event in events
            if float(event.get("timestamp_sec", 0.0)) > start_sec
            and event.get("event") in {"trajectory_end", "safety_abort", "operator_abort", "trial_end"}
        ),
        None,
    )
    configured_end_sec = start_sec + duration_sec
    terminal_end_sec = float(terminal["timestamp_sec"]) if terminal else configured_end_sec
    return start_sec, min(configured_end_sec, terminal_end_sec)


def _interpolate_reference(
    samples: list[dict[str, float]], timestamp_sec: float, max_gap_sec: float,
) -> dict[str, float] | None:
    """Interpolate the controller's *runtime-applied* reference at one time."""

    stamps = [float(row["timestamp_sec"]) for row in samples]
    index = bisect.bisect_left(stamps, timestamp_sec)
    if index < len(samples) and math.isclose(stamps[index], timestamp_sec, abs_tol=1e-6):
        return {**samples[index], "timestamp_sec": timestamp_sec}
    if index == 0 or index >= len(samples):
        return None
    left, right = samples[index - 1], samples[index]
    interval = float(right["timestamp_sec"]) - float(left["timestamp_sec"])
    if interval <= 0.0 or interval > max_gap_sec:
        return None
    alpha = (timestamp_sec - float(left["timestamp_sec"])) / interval
    result = {"timestamp_sec": timestamp_sec}
    for key in (
        "elapsed_sec", "ref_x_m", "ref_y_m", "ref_z_m",
        "ref_vx_mps", "ref_vy_mps", "ref_vz_mps", "duration_sec",
    ):
        result[key] = float(left[key]) + alpha * (float(right[key]) - float(left[key]))
    # Keep provenance through resampling.  The newer PoseStamped stream needs
    # a position-difference velocity reconstruction downstream; dropping its
    # source label here would make it indistinguishable from a legacy array.
    result["timing_source"] = str(left.get("timing_source", "unknown"))
    return result


def _resample_runtime_reference(
    samples: list[dict[str, float]],
    *,
    start_sec: float,
    end_sec: float,
    rate_hz: float,
    max_gap_sec: float,
) -> tuple[list[dict[str, float]], int]:
    """Return fixed-rate runtime-reference samples and the requested count."""

    if end_sec < start_sec:
        return [], 0
    ordered = sorted(samples, key=lambda row: float(row["timestamp_sec"]))
    if rate_hz <= 0.0:
        selected = [row for row in ordered if start_sec <= float(row["timestamp_sec"]) <= end_sec]
        return selected, len(selected)
    expected = int(math.floor((end_sec - start_sec) * rate_hz + 1e-9)) + 1
    step = 1.0 / rate_hz
    result: list[dict[str, float]] = []
    for index in range(expected):
        timestamp_sec = start_sec + index * step
        row = _interpolate_reference(ordered, timestamp_sec, max_gap_sec)
        if row is not None:
            result.append(row)
    return result, expected


def _restore_stamped_reference_velocities(reference: list[dict[str, Any]]) -> None:
    """Recover velocity from geometric PoseStamped references when needed.

    The modern Unity lockstep stream deliberately uses PoseStamped for an
    exact /clock-aligned geometric reference and does not serialize velocity.
    Legacy hardware Float32MultiArray messages already carry controller
    velocities.  Reconstructing only the stamped case avoids reporting a
    misleading zero reference speed in trajectory diagnostics.
    """

    if len(reference) < 2:
        return
    axes = (("ref_x_m", "ref_vx_mps"), ("ref_y_m", "ref_vy_mps"), ("ref_z_m", "ref_vz_mps"))
    for index, row in enumerate(reference):
        if str(row.get("timing_source", "")) != "message_header_stamped":
            continue
        left_index = max(0, index - 1)
        right_index = min(len(reference) - 1, index + 1)
        if left_index == right_index:
            continue
        left, right = reference[left_index], reference[right_index]
        dt = float(right["timestamp_sec"]) - float(left["timestamp_sec"])
        if dt <= 1e-9:
            continue
        for position_key, velocity_key in axes:
            row[velocity_key] = (float(right[position_key]) - float(left[position_key])) / dt


def _interpolate_pose(samples: list[dict[str, Any]], timestamp_sec: float, max_gap_sec: float) -> dict[str, float] | None:
    stamps = [float(row["timestamp_sec"]) for row in samples]
    index = bisect.bisect_left(stamps, timestamp_sec)
    if index == 0 or index >= len(samples):
        return None
    left, right = samples[index - 1], samples[index]
    duration = float(right["timestamp_sec"]) - float(left["timestamp_sec"])
    if duration <= 0.0 or duration > max_gap_sec:
        return None
    alpha = (timestamp_sec - float(left["timestamp_sec"])) / duration
    return {
        "x_m": float(left["x_m"]) + alpha * (float(right["x_m"]) - float(left["x_m"])),
        "y_m": float(left["y_m"]) + alpha * (float(right["y_m"]) - float(left["y_m"])),
        "z_m": float(left["z_m"]) + alpha * (float(right["z_m"]) - float(left["z_m"])),
    }


def _legacy_coordinate_transform() -> dict[str, Any]:
    """Return the documented pre-metadata FinsROV controller convention.

    Old bags did not freeze the coordinate transform in their event metadata.
    This fallback permits a corrective re-analysis of those bags, while the
    resulting report explicitly marks the transform as inferred rather than
    silently treating pool-world values as controller-world values.
    """

    return {
        "schema": "legacy_inferred_pool_world_to_controller_world_v1",
        "pool_world_frame": "pool_world",
        "controller_world_frame": "controller_world",
        # controller = [pool.x, pool.z - water_surface_z, pool.y]
        "basis_indices": [0, 2, 1],
        "basis_signs": [1.0, 1.0, 1.0],
        "post_basis_offset_m": [0.0, -0.96, 0.0],
        "water_surface_z_m": 0.96,
        "provenance": "legacy_inferred_from_documented_adapter_default",
    }


def _coordinate_transform_from_metadata(metadata: dict[str, Any], warnings: list[str]) -> dict[str, Any]:
    candidate = metadata.get("coordinate_transform")
    if not isinstance(candidate, dict):
        warnings.append(
            "T2 coordinate transform was absent from this legacy bag; applied the documented "
            "[pool.x, pool.z-0.96, pool.y] controller mapping and marked the report inferred"
        )
        return _legacy_coordinate_transform()
    try:
        indices = [int(value) for value in candidate["basis_indices"]]
        signs = [float(value) for value in candidate["basis_signs"]]
        offset = [float(value) for value in candidate["post_basis_offset_m"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("T2 metadata has an invalid coordinate_transform") from exc
    if sorted(indices) != [0, 1, 2] or len(signs) != 3 or len(offset) != 3:
        raise ValueError("T2 coordinate_transform must contain a three-axis signed permutation")
    if any(not math.isfinite(value) or abs(value) < 1e-9 for value in signs) or any(
        not math.isfinite(value) for value in offset
    ):
        raise ValueError("T2 coordinate_transform contains invalid basis values")
    return {
        **candidate,
        "basis_indices": indices,
        "basis_signs": signs,
        "post_basis_offset_m": offset,
        "provenance": "frozen_trial_metadata",
    }


def _pool_to_controller(position_pool: dict[str, float], transform: dict[str, Any]) -> dict[str, float]:
    pool = [float(position_pool[axis]) for axis in ("x_m", "y_m", "z_m")]
    indices = [int(value) for value in transform["basis_indices"]]
    signs = [float(value) for value in transform["basis_signs"]]
    offset = [float(value) for value in transform["post_basis_offset_m"]]
    controller = [signs[index] * pool[indices[index]] + offset[index] for index in range(3)]
    return dict(zip(("x_m", "y_m", "z_m"), controller, strict=True))


def _controller_to_pool(position_controller: dict[str, float], transform: dict[str, Any]) -> dict[str, float]:
    controller = [float(position_controller[axis]) for axis in ("x_m", "y_m", "z_m")]
    indices = [int(value) for value in transform["basis_indices"]]
    signs = [float(value) for value in transform["basis_signs"]]
    offset = [float(value) for value in transform["post_basis_offset_m"]]
    pool = [0.0, 0.0, 0.0]
    for controller_axis, pool_axis in enumerate(indices):
        pool[pool_axis] = (controller[controller_axis] - offset[controller_axis]) / signs[controller_axis]
    return dict(zip(("x_m", "y_m", "z_m"), pool, strict=True))


def _select_t2_pose_source(poses: list[dict[str, Any]]) -> tuple[str, str] | None:
    """Choose an explicitly typed T2 state source; never mix frames silently."""

    by_topic: dict[str, list[dict[str, Any]]] = {}
    for row in poses:
        by_topic.setdefault(str(row["topic"]), []).append(row)
    for topic in ("/finsrov/controller/pose", "/sim/finsrov/controller/pose"):
        rows = by_topic.get(topic, [])
        frames = {str(row.get("frame_id", "")).strip().lower() for row in rows}
        if rows and frames == {"controller_world"}:
            return topic, "controller_world_direct"
    # Physical T2 bags retain fusion output in pool_world. It is valid only
    # after the frozen transform above maps it into the controller contract.
    rows = by_topic.get("/finsrov/pose", [])
    if rows:
        frames = {str(row.get("frame_id", "")).strip().lower() for row in rows}
        if frames and frames - {"", "pool_world"}:
            return None
        return "/finsrov/pose", "pool_world_transformed"
    return None


def _distance_to_segment(px: float, pz: float, ax: float, az: float, bx: float, bz: float) -> float:
    dx, dz = bx - ax, bz - az
    denominator = dx * dx + dz * dz
    if denominator <= 1e-12:
        return math.hypot(px - ax, pz - az)
    fraction = min(1.0, max(0.0, ((px - ax) * dx + (pz - az) * dz) / denominator))
    return math.hypot(px - (ax + fraction * dx), pz - (az + fraction * dz))


def _cross_track_error(rows: list[dict[str, Any]], index: int, radius: int = 20) -> float:
    row = rows[index]
    first = max(0, index - radius)
    last = min(len(rows) - 1, index + radius)
    if last <= first:
        return math.hypot(row["actual_x_m"] - row["ref_x_m"], row["actual_z_m"] - row["ref_z_m"])
    return min(
        _distance_to_segment(
            float(row["actual_x_m"]), float(row["actual_z_m"]),
            float(rows[segment]["ref_x_m"]), float(rows[segment]["ref_z_m"]),
            float(rows[segment + 1]["ref_x_m"]), float(rows[segment + 1]["ref_z_m"]),
        )
        for segment in range(first, last)
    )


def _percentile(values: list[float], fraction: float = 0.95) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def _error_summary(values: list[float]) -> dict[str, float | None]:
    """Return signed-error metrics while making the p95 convention explicit."""

    if not values:
        return {"rmse": None, "mae": None, "p95_absolute": None}
    absolute = [abs(value) for value in values]
    return {
        "rmse": math.sqrt(fmean(value * value for value in values)),
        "mae": fmean(absolute),
        "p95_absolute": _percentile(absolute),
    }


def _latest_status_before(
    statuses: list[dict[str, Any]], timestamp_sec: float, max_age_sec: float,
) -> tuple[dict[str, Any] | None, float | None]:
    """Return the last controller-state status valid at ``timestamp_sec``.

    Status messages do not carry a ROS header, so their bag timestamp is the
    only timing source.  A future status is never used to validate an earlier
    reference sample, and a stale status is explicitly rejected.
    """

    if not statuses:
        return None, None
    stamps = [float(row["timestamp_sec"]) for row in statuses]
    index = bisect.bisect_right(stamps, timestamp_sec) - 1
    if index < 0:
        return None, None
    candidate = statuses[index]
    age = max(0.0, timestamp_sec - float(candidate["timestamp_sec"]))
    if max_age_sec > 0.0 and age > max_age_sec:
        return None, age
    return candidate, age


def _controller_state_is_fresh(status: dict[str, Any] | None, contract: dict[str, Any]) -> tuple[bool, str]:
    """Apply the frozen trial freshness contract to one controller status."""

    if status is None:
        return False, "missing_or_stale_status"
    checks = (
        ("ready", bool(contract.get("require_ready", True))),
        ("initialized", bool(contract.get("require_initialized", True))),
        ("imu_fresh", bool(contract.get("require_imu_fresh", True))),
        ("depth_fresh", bool(contract.get("require_depth_fresh", True))),
    )
    for key, required in checks:
        if required and not bool(status.get(key, False)):
            return False, key
    allowed_modes = {str(value) for value in contract.get("allowed_vision_modes", ("fresh", "coast"))}
    mode = str(status.get("vision_mode", "unknown"))
    if allowed_modes and mode not in allowed_modes:
        return False, f"vision_mode:{mode}"
    # ``coast`` is intentionally accepted when the controller profile permits
    # it.  Requiring vision_fresh here would silently contradict that runtime
    # contract; ``vision_mode`` remains in the audit row for later reporting.
    return True, mode


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _split_circle_loops(rows: list[dict[str, Any]], turns: int) -> list[list[dict[str, Any]]]:
    """Split a circle trial into complete reference-loop display windows.

    The circle generator applies the cubic smoothstep to the total trial, so
    equal wall-clock thirds do not coincide with completed circles.  Reuse the
    same progress law here to split plots by reference-loop phase instead.
    """

    if turns < 1 or not rows:
        return []
    duration = max(float(row["duration_sec"]) for row in rows)
    if duration <= 0.0:
        return []
    result: list[list[dict[str, Any]]] = [[] for _ in range(turns)]
    for row in rows:
        normalized_time = min(max(float(row["elapsed_sec"]) / duration, 0.0), 1.0)
        progress = 3.0 * normalized_time * normalized_time - 2.0 * normalized_time**3
        loop_index = min(int(math.floor(turns * progress)), turns - 1)
        result[loop_index].append(row)
    return result


def _plot(
    derived: Path,
    rows: list[dict[str, Any]],
    *,
    circle_turns: int | None = None,
    horizontal_plane: str = "pool_xy",
) -> list[str]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except Exception as exc:  # pragma: no cover - optional dependency
        raise RuntimeError(f"matplotlib unavailable for T2 plots: {exc}") from exc
    if not rows:
        return []
    output = derived / "plots" / "t2"
    output.mkdir(parents=True, exist_ok=True)
    # These names were generated by the pre-fix analyser, which compared raw
    # pool-world y/z values against controller-world y/z references. Remove
    # only those known derived artifacts during a re-analysis so they cannot
    # be mistaken for a valid result.
    for obsolete in (
        output / "path_xz.png",
        output / "state_vs_reference_xyz.png",
        output / "tracking_error_xyz.png",
    ):
        if obsolete.is_file():
            obsolete.unlink()
    for obsolete in output.glob("circle_loop_*_path_xz.png"):
        if obsolete.is_file():
            obsolete.unlink()
    for obsolete in output.glob("circle_loop_*_path_controller_xz.png"):
        if obsolete.is_file():
            obsolete.unlink()
    for obsolete in (output / "path_controller_xz.png",):
        if obsolete.is_file():
            obsolete.unlink()
    time = [float(row["elapsed_sec"]) for row in rows]
    produced: list[str] = []
    use_controller_plane = horizontal_plane == "controller_xz"
    reference_x = "ref_x_m" if use_controller_plane else "ref_pool_x_m"
    reference_y = "ref_z_m" if use_controller_plane else "ref_pool_y_m"
    actual_x = "actual_x_m" if use_controller_plane else "actual_pool_x_m"
    actual_y = "actual_z_m" if use_controller_plane else "actual_pool_y_m"
    x_label = "controller x [m]" if use_controller_plane else "pool x [m]"
    y_label = "controller z [m]" if use_controller_plane else "pool y [m]"
    reference_label = "runtime reference (controller_world)" if use_controller_plane else "reference mapped to pool_world"
    actual_label = "simulation truth (controller_world)" if use_controller_plane else "fused measurement (pool_world)"
    plane_title = "Unity horizontal plane (controller x-z)" if use_controller_plane else "physical horizontal plane (pool_world)"
    path_suffix = "controller_xz" if use_controller_plane else "pool_xy"

    if circle_turns is not None and circle_turns > 0:
        for loop_index, loop_rows in enumerate(_split_circle_loops(rows, circle_turns), start=1):
            if not loop_rows:
                continue
            fig, axis = plt.subplots(figsize=(6.2, 5.0))
            axis.plot(
                [row[reference_x] for row in loop_rows],
                [row[reference_y] for row in loop_rows],
                "--",
                label=reference_label,
            )
            axis.plot(
                [row[actual_x] for row in loop_rows],
                [row[actual_y] for row in loop_rows],
                label=actual_label,
            )
            axis.set_xlabel(x_label)
            axis.set_ylabel(y_label)
            axis.set_title(f"T2 circle loop {loop_index}/{circle_turns} ({plane_title})")
            axis.grid(True, alpha=0.3)
            axis.axis("equal")
            axis.legend()
            fig.tight_layout()
            path = output / f"circle_loop_{loop_index:02d}_path_{path_suffix}.png"
            fig.savefig(path, dpi=180)
            plt.close(fig)
            produced.append(str(path.relative_to(derived)))
    else:
        fig, axis = plt.subplots(figsize=(6.2, 5.0))
        axis.plot(
            [row[reference_x] for row in rows],
            [row[reference_y] for row in rows],
            "--",
            label=reference_label,
        )
        axis.plot(
            [row[actual_x] for row in rows],
            [row[actual_y] for row in rows],
            label=actual_label,
        )
        axis.set_xlabel(x_label)
        axis.set_ylabel(y_label)
        axis.set_title(f"T2 {plane_title}")
        axis.grid(True, alpha=0.3)
        axis.axis("equal")
        axis.legend()
        fig.tight_layout()
        path = output / f"path_{path_suffix}.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        produced.append(str(path.relative_to(derived)))

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
    components = (
        ("x", "actual_x_m", "ref_x_m", "controller x [m]"),
        ("y", "actual_y_m", "ref_y_m", "controller y (up) [m]"),
        ("z", "actual_z_m", "ref_z_m", "controller z [m]"),
    )
    for axis, (_, actual, reference, label) in zip(axes, components):
        axis.plot(time, [row[actual] for row in rows], label="measurement transformed to controller_world")
        axis.plot(time, [row[reference] for row in rows], "--", label="runtime reference")
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[0].legend(fontsize=8)
    axes[-1].set_xlabel("trajectory elapsed time [s]")
    fig.suptitle("T2 state and runtime reference (controller_world)")
    fig.tight_layout()
    path = output / "state_vs_reference_controller_xyz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    produced.append(str(path.relative_to(derived)))

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 6.0), sharex=True)
    component_errors = (
        ("controller x error [m]", "error_x_m"),
        ("controller y error [m]", "error_y_m"),
        ("controller z error [m]", "error_z_m"),
    )
    for axis, (label, field) in zip(axes, component_errors):
        axis.plot(time, [row[field] for row in rows])
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
        axis.set_ylabel(label)
        axis.grid(True, alpha=0.3)
    axes[-1].set_xlabel("trajectory elapsed time [s]")
    fig.suptitle("T2 signed tracking error by controlled degree of freedom (controller_world)")
    fig.tight_layout()
    path = output / "tracking_error_controller_xyz.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    produced.append(str(path.relative_to(derived)))

    fig, axes = plt.subplots(3, 1, figsize=(7.0, 5.8), sharex=True)
    axes[0].plot(time, [row["cross_track_error_m"] for row in rows])
    axes[0].set_ylabel("cross-track [m]")
    axes[1].plot(time, [row["time_aligned_position_error_m"] for row in rows])
    axes[1].set_ylabel("3D error [m]")
    axes[2].plot(time, [row["horizontal_time_aligned_error_m"] for row in rows])
    axes[2].set_ylabel("horizontal x-z error [m]")
    axes[2].set_xlabel("trajectory elapsed time [s]")
    for axis in axes:
        axis.grid(True, alpha=0.3)
    fig.suptitle("T2 tracking errors")
    fig.tight_layout()
    path = output / "tracking_error.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    produced.append(str(path.relative_to(derived)))
    return produced


def _command_diagnostics(commands: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    """Report reproducible native-unit effort/smoothness without inventing units."""

    result: dict[str, dict[str, float]] = {}
    for topic in sorted({str(row["topic"]) for row in commands}):
        samples = sorted((row for row in commands if row["topic"] == topic), key=lambda row: float(row["timestamp_sec"]))
        effort = 0.0
        variation = 0.0
        for left, right in zip(samples, samples[1:]):
            dt = max(float(right["timestamp_sec"]) - float(left["timestamp_sec"]), 1e-9)
            left_values = [float(value) for value in left["values"]]
            right_values = [float(value) for value in right["values"]]
            effort += dt * sum(value * value for value in left_values)
            variation += sum((right_values[index] - left_values[index]) ** 2 for index in range(min(len(left_values), len(right_values)))) / dt
        result[topic] = {
            "samples": float(len(samples)),
            "effort_native_squared_sec": effort,
            "smoothness_native_squared_per_sec": variation,
        }
    return result


def augment_t2_report(
    session_dir: Path,
    derived: Path,
    poses: list[dict[str, Any]],
    commands: list[dict[str, Any]],
    events: list[dict[str, Any]],
    report: dict[str, Any],
) -> tuple[list[str], list[str]]:
    """Add T2-specific outputs to the generic automatic analysis report."""

    warnings: list[str] = []
    metadata = next(
        (
            event.get("metadata")
            for event in events
            if event.get("event") == "trajectory_start" and isinstance(event.get("metadata"), dict)
        ),
        next((event.get("metadata") for event in events if isinstance(event.get("metadata"), dict)), {}),
    )
    if not isinstance(metadata, dict):
        metadata = {}
    source = _select_t2_pose_source(poses)
    if source is None:
        return [], [
            "T2 analysis skipped: no pose source with an auditable controller_world or pool_world frame; "
            "raw coordinates will not be compared to the trajectory reference"
        ]
    pose_topic, pose_source_kind = source
    coordinate_transform = _coordinate_transform_from_metadata(metadata, warnings)
    evaluation = metadata.get("evaluation_window", {}) if isinstance(metadata, dict) else {}
    max_gap = float(evaluation.get("max_interpolation_gap_sec", 0.25)) if isinstance(evaluation, dict) else 0.25
    duration_sec = float(evaluation.get("duration_sec", 30.0)) if isinstance(evaluation, dict) else 30.0
    resample_hz = float(evaluation.get("resample_hz", 10.0)) if isinstance(evaluation, dict) else 10.0
    raw_reference, status = _read_t2_runtime(session_dir)
    timing_contract = metadata.get("timing_contract", {}) if isinstance(metadata, dict) else {}
    if not isinstance(timing_contract, dict):
        timing_contract = {}
    start_mode = str(timing_contract.get("trajectory_start_mode", "experiment_event_v1"))
    timing_provenance = "trajectory_start_event"
    try:
        if start_mode == "first_controller_tick_v1":
            # Verify that the recorder event is present even though the
            # controller reference provides the exact timebase.
            if not any(event.get("event") == "trajectory_start" for event in events):
                raise ValueError("T2 session is missing trajectory_start event")
            start, end = _lockstep_controller_window(raw_reference, duration_sec=duration_sec)
            timing_provenance = "lockstep_first_controller_tick"
        else:
            start, end = _event_window(events, duration_sec=duration_sec)
    except ValueError as exc:
        start, end = _runtime_reference_inferred_window(
            raw_reference,
            events,
            duration_sec=duration_sec,
        )
        timing_provenance = "runtime_reference_inferred_missing_trajectory_start"
        warnings.append(
            "T2 trajectory_start marker is missing; the analysis window is reconstructed from the "
            "controller's recorded runtime reference. Per the retained-trial policy, this trial remains "
            "eligible for tracking metrics, with inferred timing provenance recorded: "
            f"{exc}"
        )
    raw_reference = _resolve_reference_timing(
        raw_reference,
        start_sec=start,
        duration_sec=duration_sec,
    )
    raw_reference = [row for row in raw_reference if start <= row["timestamp_sec"] <= end]
    status = [row for row in status if start <= row["timestamp_sec"] <= end]
    reference, requested_reference_samples = _resample_runtime_reference(
        raw_reference,
        start_sec=start,
        end_sec=end,
        rate_hz=resample_hz,
        max_gap_sec=max_gap,
    )
    _restore_stamped_reference_velocities(reference)
    if len(reference) < requested_reference_samples:
        warnings.append(
            "T2 runtime reference has gaps inside the frozen evaluation window; "
            "missing grid points are excluded and reduce the valid fraction"
        )
    selected_pose = sorted((row for row in poses if row["topic"] == pose_topic), key=lambda row: float(row["timestamp_sec"]))
    completion_contract = metadata.get("completion_contract", {}) if isinstance(metadata, dict) else {}
    if not isinstance(completion_contract, dict):
        completion_contract = {}
    status_max_age = float(completion_contract.get("status_max_age_sec", max_gap))
    ordered_status = sorted(status, key=lambda row: float(row["timestamp_sec"]))
    reference_status = [
        _latest_status_before(ordered_status, float(row["timestamp_sec"]), status_max_age)
        for row in reference
    ]
    fresh_reference = [
        _controller_state_is_fresh(sample, completion_contract)
        for sample, _ in reference_status
    ]
    if not ordered_status:
        warnings.append("T2 state-status audit is unavailable: no real or /sim controller state-status topic was recorded")
    aligned: list[dict[str, Any]] = []
    for row, (state, state_age), (state_fresh, state_reason) in zip(reference, reference_status, fresh_reference):
        interpolated_pose = _interpolate_pose(selected_pose, float(row["timestamp_sec"]), max_gap)
        if interpolated_pose is None:
            continue
        if pose_source_kind == "pool_world_transformed":
            actual_pool = interpolated_pose
            actual = _pool_to_controller(actual_pool, coordinate_transform)
        else:
            actual = interpolated_pose
            actual_pool = _controller_to_pool(actual, coordinate_transform)
        reference_controller = {
            "x_m": float(row["ref_x_m"]),
            "y_m": float(row["ref_y_m"]),
            "z_m": float(row["ref_z_m"]),
        }
        reference_pool = _controller_to_pool(reference_controller, coordinate_transform)
        error_x = float(actual["x_m"]) - reference_controller["x_m"]
        error_y = float(actual["y_m"]) - reference_controller["y_m"]
        error_z = float(actual["z_m"]) - reference_controller["z_m"]
        aligned.append({
            **row,
            "actual_x_m": actual["x_m"], "actual_y_m": actual["y_m"], "actual_z_m": actual["z_m"],
            "actual_pool_x_m": actual_pool["x_m"],
            "actual_pool_y_m": actual_pool["y_m"],
            "actual_pool_z_m": actual_pool["z_m"],
            "ref_pool_x_m": reference_pool["x_m"],
            "ref_pool_y_m": reference_pool["y_m"],
            "ref_pool_z_m": reference_pool["z_m"],
            "error_x_m": error_x, "error_y_m": error_y, "error_z_m": error_z,
            "horizontal_time_aligned_error_m": math.hypot(error_x, error_z),
            "time_aligned_position_error_m": math.sqrt(error_x * error_x + error_y * error_y + error_z * error_z),
            "state_fresh": state_fresh,
            "state_fresh_reason": state_reason,
            "state_status_age_sec": state_age,
            "state_vision_mode": str(state.get("vision_mode", "unknown")) if state is not None else "unknown",
        })
    for index, row in enumerate(aligned):
        row["cross_track_error_m"] = _cross_track_error(aligned, index)
    valid_fraction = len(aligned) / requested_reference_samples if requested_reference_samples else 0.0
    runtime_reference_fraction = len(reference) / requested_reference_samples if requested_reference_samples else 0.0
    if not aligned:
        return [], [*warnings, "T2 analysis found no reference/pose pairs within the interpolation gap"]
    fields = list(aligned[0].keys())
    _write_csv(derived / "reference_aligned_samples.csv", aligned, fields)
    horizontal = [float(row["horizontal_time_aligned_error_m"]) for row in aligned]
    position = [float(row["time_aligned_position_error_m"]) for row in aligned]
    error_x = [float(row["error_x_m"]) for row in aligned]
    error_y = [float(row["error_y_m"]) for row in aligned]
    error_z = [float(row["error_z_m"]) for row in aligned]
    depth = [abs(float(row["error_y_m"])) for row in aligned]
    cross_track = [float(row["cross_track_error_m"]) for row in aligned]
    duration = max(float(row["duration_sec"]) for row in reference)
    progress = max(float(row["elapsed_sec"]) for row in reference) / duration if duration > 1e-9 else 0.0
    forbidden = {
        str(value)
        for value in completion_contract.get(
            "forbidden_events",
            ("safety_abort", "operator_abort", "state_timeout", "workspace_exit", "controller_fault"),
        )
    }
    events_seen = [str(event.get("event", "")) for event in events]
    min_fraction = float(evaluation.get("minimum_valid_fraction", 0.90)) if isinstance(evaluation, dict) else 0.90
    required_progress = float(completion_contract.get("required_progress_fraction", 0.98))
    required_state_fresh = float(completion_contract.get("required_state_fresh_fraction", 0.90))
    state_fresh_fraction = sum(1 for fresh, _ in fresh_reference if fresh) / len(reference) if reference else 0.0
    completion = (
        progress >= required_progress
        and valid_fraction >= min_fraction
        and state_fresh_fraction >= required_state_fresh
        and not any(event in forbidden for event in events_seen)
    )
    metrics = {
        "pose_topic": pose_topic,
        "pose_source_kind": pose_source_kind,
        "coordinate_transform": coordinate_transform,
        "evaluation_window": {
            "start_event": str(evaluation.get("start_event", "trajectory_start")),
            "timing_provenance": timing_provenance,
            "start_timestamp_sec": start,
            "end_timestamp_sec": end,
            "requested_duration_sec": duration_sec,
            "resample_hz": resample_hz,
            "requested_reference_samples": requested_reference_samples,
            "max_interpolation_gap_sec": max_gap,
        },
        "raw_reference_samples": len(raw_reference),
        "reference_timing_source": (
            str(raw_reference[0].get("timing_source", "unknown")) if raw_reference else "unknown"
        ),
        "reference_samples": len(reference),
        "aligned_samples": len(aligned),
        "reference_valid_fraction": valid_fraction,
        "runtime_reference_fraction": runtime_reference_fraction,
        "progress_fraction": progress,
        "completion_ratio": 1.0 if completion else 0.0,
        "completion_thresholds": {
            "required_progress_fraction": required_progress,
            "required_reference_valid_fraction": min_fraction,
            "required_state_fresh_fraction": required_state_fresh,
            "status_max_age_sec": status_max_age,
            "allowed_vision_modes": list(completion_contract.get("allowed_vision_modes", ("fresh", "coast"))),
        },
        "cross_track_rmse_m": math.sqrt(fmean(value * value for value in cross_track)),
        "cross_track_p95_m": _percentile(cross_track),
        "time_aligned_position_rmse_m": math.sqrt(fmean(value * value for value in position)),
        "horizontal_time_aligned_rmse_m": math.sqrt(fmean(value * value for value in horizontal)),
        "depth_rmse_m": math.sqrt(fmean(value * value for value in depth)),
        "per_dof_error": {
            "x_m": _error_summary(error_x),
            "y_depth_m": _error_summary(error_y),
            "z_m": _error_summary(error_z),
        },
        "forbidden_events": [event for event in events_seen if event in forbidden],
        "state_status_samples": len(status),
        "state_fresh_reference_samples": sum(1 for fresh, _ in fresh_reference if fresh),
        "state_fresh_fraction": state_fresh_fraction,
        "command_diagnostics": _command_diagnostics(commands),
        "thruster_saturation_fraction": None,
        "allocation_residual": None,
        "completion_scope": (
            "runtime reference progress + pose/reference validity + frozen controller-state freshness contract + "
            "forbidden-event audit; a missing trajectory_start marker is recovered from the recorded runtime "
            "reference and its timing provenance remains explicitly recorded"
        ),
    }
    write_json(derived / "t2_metrics.json", metrics)
    _write_csv(derived / "t2_metrics.csv", [metrics], list(metrics.keys()))
    trajectory_geometry = metadata.get("trajectory_geometry", {}) if isinstance(metadata, dict) else {}
    circle_turns: int | None = None
    if isinstance(trajectory_geometry, dict) and str(trajectory_geometry.get("generator", "")).lower() == "circle_xz":
        try:
            candidate_turns = int(trajectory_geometry.get("turns", 0))
        except (TypeError, ValueError):
            candidate_turns = 0
        if candidate_turns > 0:
            circle_turns = candidate_turns
        else:
            warnings.append("T2 circle plot was not split: missing/invalid frozen trajectory_geometry.turns")
    plot_paths = _plot(
        derived,
        aligned,
        circle_turns=circle_turns,
        horizontal_plane="controller_xz" if bool(metadata.get("simulation", False)) else "pool_xy",
    )
    report["t2"] = metrics
    return plot_paths, warnings
