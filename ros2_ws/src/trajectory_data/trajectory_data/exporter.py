from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import rosbag2_py
import yaml
from rosidl_runtime_py.utilities import get_message
from rclpy.serialization import deserialize_message

from .alignment import TimedRecord, interpolate_payload, previous_record, safe_json, sorted_records, stamp_to_sec


STREAM_BY_TOPIC = {
    "/finsrov/hardware/telemetry": "hardware_telemetry",
    "/finsrov/hardware/imu_raw": "hardware_imu_raw",
    "/finsrov/hardware/depth_raw": "hardware_depth_raw",
    "/finsrov/hardware/motor_rpm_raw": "hardware_rpm_raw",
    "/finsrov/hardware/thruster_cmd_echo": "hardware_thruster_echo",
    "/finsrov/hardware/status": "hardware_status",
    "/finsrov/vision/tag_poses_3d_camera": "vision_tags",
    "/finsrov/vision/refracted_pose_6d": "vision_refracted",
    "/finsrov/vision/refracted_pose_6d_pure": "vision_refracted_pure",
    "/finsrov/vision/status": "vision_status",
    "/finsrov/pose": "fused_pose",
    "/finsrov/imu_link": "fused_imu",
    "/finsrov/depth_link": "fused_depth",
    "/finsrov/dvl_link": "fused_dvl",
    "/finsrov/state/status": "state_status",
    "/finsrov/teleop/body_wrench_cmd": "teleop_wrench",
    "/finsrov/teleop/status": "teleop_status",
    "/finsrov/teleop/enabled": "teleop_enabled",
    "/finsrov/thrusters_out": "thruster_command",
    "/finsrov/trajectory/event": "trajectory_event",
}


def _header_stamp(message: Any, bag_time_sec: float) -> float:
    header = getattr(message, "header", None)
    stamp = getattr(header, "stamp", None)
    if stamp is None:
        return bag_time_sec
    value = stamp_to_sec(stamp)
    return value if math.isfinite(value) and value > 0.0 else bag_time_sec


def _vector(vector: Any) -> list[float]:
    return [float(vector.x), float(vector.y), float(vector.z)]


def _quaternion(quaternion: Any) -> list[float]:
    return [float(quaternion.x), float(quaternion.y), float(quaternion.z), float(quaternion.w)]


def _pose_payload(message: Any) -> dict[str, Any]:
    pose = message.pose.pose if hasattr(message.pose, "pose") else message.pose
    return {
        "frame_id": str(message.header.frame_id),
        "position_xyz": _vector(pose.position),
        "orientation_xyzw": _quaternion(pose.orientation),
        "covariance": [float(value) for value in getattr(message.pose, "covariance", [])],
    }


def _imu_payload(message: Any) -> dict[str, Any]:
    return {
        "frame_id": str(message.header.frame_id),
        "orientation_xyzw": _quaternion(message.orientation),
        "angular_velocity_xyz": _vector(message.angular_velocity),
        "linear_acceleration_xyz": _vector(message.linear_acceleration),
    }


def _dvl_payload(message: Any) -> dict[str, Any]:
    twist = message.twist.twist if hasattr(message.twist, "twist") else message.twist
    return {
        "frame_id": str(message.header.frame_id),
        "linear_velocity_xyz": _vector(twist.linear),
        "angular_velocity_xyz": _vector(twist.angular),
        "covariance": [float(value) for value in getattr(message.twist, "covariance", [])],
    }


def _payload(topic: str, message: Any) -> dict[str, Any]:
    if topic == "/finsrov/hardware/telemetry":
        return {
            "frame_id": str(message.header.frame_id),
            "telemetry_sequence": int(message.telemetry_sequence),
            "mcu_time_ms": int(message.mcu_time_ms),
            "status_flags": int(message.status_flags),
            "orientation_xyzw": [float(value) for value in message.orientation_xyzw],
            "angular_velocity_xyz": [float(value) for value in message.angular_velocity_xyz],
            "linear_acceleration_xyz": [float(value) for value in message.linear_acceleration_xyz],
            "depth_m": float(message.depth_m),
            "pressure_pa": float(message.pressure_pa),
            "rpm": [float(value) for value in message.rpm],
            "host_receive_time_ns": int(message.host_receive_time_ns),
        }
    if topic in {"/finsrov/hardware/depth_raw", "/finsrov/depth_link", "/finsrov/vision/refracted_pose_6d", "/finsrov/vision/refracted_pose_6d_pure", "/finsrov/pose"}:
        return _pose_payload(message)
    if topic in {"/finsrov/hardware/imu_raw", "/finsrov/imu_link"}:
        return _imu_payload(message)
    if topic == "/finsrov/dvl_link":
        return _dvl_payload(message)
    if topic == "/finsrov/teleop/body_wrench_cmd":
        wrench = message.wrench
        return {
            "frame_id": str(message.header.frame_id),
            "wrench4": [float(wrench.force.x), float(wrench.force.y), float(wrench.force.z), float(wrench.torque.z)],
        }
    if topic in {"/finsrov/thrusters_out", "/finsrov/hardware/motor_rpm_raw"}:
        return {"values": [float(value) for value in message.data]}
    if topic == "/finsrov/hardware/thruster_cmd_echo":
        return {
            "telemetry_sequence": int(message.telemetry_sequence),
            "mcu_time_ms": int(message.mcu_time_ms),
            "command_count": int(message.command_count),
            "enabled": bool(message.enabled),
            "accepted": bool(message.accepted),
            "reject_flags": int(message.reject_flags),
            "received_thrust": [float(value) for value in message.received_thrust],
            "applied_thrust": [float(value) for value in message.applied_thrust],
        }
    if topic in {"/finsrov/hardware/status", "/finsrov/vision/status", "/finsrov/state/status", "/finsrov/teleop/status"}:
        return {"data": str(message.data), "json": safe_json(str(message.data))}
    if topic == "/finsrov/teleop/enabled":
        return {"enabled": bool(message.data)}
    if topic == "/finsrov/trajectory/event":
        return {
            "session_id": str(message.session_id),
            "event": str(message.event),
            "label": str(message.label),
            "metadata": safe_json(str(message.metadata_json)),
        }
    return {"message": str(message)}


def _read_bag(bag_dir: Path) -> dict[str, list[TimedRecord]]:
    metadata = yaml.safe_load((bag_dir / "metadata.yaml").read_text(encoding="utf-8")) or {}
    bag_metadata = metadata.get("rosbag2_bagfile_information", {})
    compression_mode = str(bag_metadata.get("compression_mode") or "NONE").upper()
    # SequentialCompressionReader handles historical file-compressed bags;
    # it rejects a standard uncompressed SQLite bag, which uses SequentialReader.
    reader = rosbag2_py.SequentialCompressionReader() if compression_mode != "NONE" else rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=str(bag_dir), storage_id="sqlite3"),
        rosbag2_py.ConverterOptions(input_serialization_format="cdr", output_serialization_format="cdr"),
    )
    type_by_topic = {metadata.name: metadata.type for metadata in reader.get_all_topics_and_types()}
    streams: dict[str, list[TimedRecord]] = {stream: [] for stream in STREAM_BY_TOPIC.values()}
    while reader.has_next():
        topic, serialized, bag_timestamp_ns = reader.read_next()
        stream = STREAM_BY_TOPIC.get(topic)
        if stream is None or topic not in type_by_topic:
            continue
        message = deserialize_message(serialized, get_message(type_by_topic[topic]))
        bag_time_sec = float(bag_timestamp_ns) * 1e-9
        streams[stream].append(
            TimedRecord(
                time_sec=_header_stamp(message, bag_time_sec),
                bag_time_sec=bag_time_sec,
                payload=_payload(topic, message),
            )
        )
    return {name: sorted_records(records) for name, records in streams.items() if records}


def _select_bag_directory(session_dir: Path, requested_bag_dir: Path | None) -> tuple[Path, str]:
    if requested_bag_dir is not None:
        bag_dir = requested_bag_dir.expanduser().resolve()
        if not (bag_dir / "metadata.yaml").is_file():
            raise ValueError(f"rosbag metadata not found: {bag_dir / 'metadata.yaml'}")
        return bag_dir, "explicit"
    recovered_dir = session_dir / "rosbag_recovered"
    if (recovered_dir / "metadata.yaml").is_file():
        return recovered_dir, "recovered"
    bag_dir = session_dir / "rosbag"
    if not (bag_dir / "metadata.yaml").is_file():
        raise ValueError(f"rosbag metadata not found: {bag_dir / 'metadata.yaml'}")
    return bag_dir, "original"


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), path, compression="zstd")


def _raw_rows(records: list[TimedRecord]) -> list[dict[str, Any]]:
    return [
        {"source_time_sec": record.time_sec, "bag_time_sec": record.bag_time_sec, **record.payload}
        for record in records
    ]


def _continuous_at(
    records: list[TimedRecord] | None,
    time_sec: float,
    vector_fields: tuple[str, ...],
    quaternion_fields: tuple[str, ...] = (),
) -> tuple[dict[str, Any] | None, float, float | None]:
    if not records:
        return None, math.inf, None
    payload, age = interpolate_payload(records, time_sec, vector_fields=vector_fields, quaternion_fields=quaternion_fields)
    previous = previous_record(records, time_sec)
    return payload, age, previous.time_sec if previous is not None else None


def _held_at(records: list[TimedRecord] | None, time_sec: float) -> tuple[dict[str, Any] | None, float, float | None]:
    if not records:
        return None, math.inf, None
    previous = previous_record(records, time_sec)
    if previous is None:
        return None, math.inf, None
    return previous.payload, max(0.0, time_sec - previous.time_sec), previous.time_sec


def _phase_at(events: list[TimedRecord] | None, time_sec: float) -> str:
    phase = ""
    if not events:
        return phase
    for record in events:
        if record.time_sec > time_sec:
            break
        event = str(record.payload.get("event", ""))
        label = str(record.payload.get("label", ""))
        if event == "phase_start":
            phase = label
        elif event == "phase_end" and (not label or label == phase):
            phase = ""
    return phase


def _state_status(payload: dict[str, Any] | None) -> dict[str, Any]:
    if not payload:
        return {}
    value = payload.get("json", {})
    return value if isinstance(value, dict) else {}


def _as_list(payload: dict[str, Any] | None, field: str, size: int) -> list[float | None]:
    if payload is None:
        return [None] * size
    values = list(payload.get(field, []))
    return [float(values[index]) if index < len(values) else None for index in range(size)]


def _build_trajectory(streams: dict[str, list[TimedRecord]], rate_hz: float, thresholds: dict[str, float]) -> list[dict[str, Any]]:
    anchor = streams.get("hardware_telemetry") or streams.get("fused_pose") or streams.get("teleop_wrench")
    if not anchor:
        raise ValueError("bag has no telemetry, fused pose, or teleop wrench stream")
    start = anchor[0].time_sec
    end = anchor[-1].time_sec
    step = 1.0 / rate_hz
    rows: list[dict[str, Any]] = []
    sample_count = max(0, int(math.floor((end - start) / step)) + 1)
    for index in range(sample_count):
        time_sec = start + index * step
        telemetry, telemetry_age, telemetry_stamp = _continuous_at(
            streams.get("hardware_telemetry"), time_sec, ("angular_velocity_xyz", "linear_acceleration_xyz", "rpm"), ("orientation_xyzw",)
        )
        hardware_imu, hardware_imu_age, hardware_imu_stamp = _continuous_at(
            streams.get("hardware_imu_raw"), time_sec, ("angular_velocity_xyz", "linear_acceleration_xyz"), ("orientation_xyzw",)
        )
        fused_imu, fused_imu_age, fused_imu_stamp = _continuous_at(
            streams.get("fused_imu"), time_sec, ("angular_velocity_xyz", "linear_acceleration_xyz"), ("orientation_xyzw",)
        )
        pose, pose_age, pose_stamp = _continuous_at(
            streams.get("fused_pose"), time_sec, ("position_xyz",), ("orientation_xyzw",)
        )
        dvl, dvl_age, dvl_stamp = _continuous_at(streams.get("fused_dvl"), time_sec, ("linear_velocity_xyz", "angular_velocity_xyz"))
        depth_raw, depth_raw_age, _ = _continuous_at(streams.get("hardware_depth_raw"), time_sec, ("position_xyz",), ("orientation_xyzw",))
        depth_fused, depth_fused_age, _ = _continuous_at(streams.get("fused_depth"), time_sec, ("position_xyz",), ("orientation_xyzw",))
        vision, vision_age, vision_stamp = _continuous_at(
            streams.get("vision_refracted"), time_sec, ("position_xyz",), ("orientation_xyzw",)
        )
        wrench, action_age, action_stamp = _held_at(streams.get("teleop_wrench"), time_sec)
        thrust, thrust_age, thrust_stamp = _held_at(streams.get("thruster_command"), time_sec)
        echo, echo_age, echo_stamp = _held_at(streams.get("hardware_thruster_echo"), time_sec)
        status_payload, status_age, _ = _held_at(streams.get("state_status"), time_sec)
        state_status = _state_status(status_payload)
        row = {
            "timestamp_ros_sec": time_sec,
            "phase": _phase_at(streams.get("trajectory_event"), time_sec),
            "pose_position_world_xyz": _as_list(pose, "position_xyz", 3),
            "pose_orientation_xyzw": _as_list(pose, "orientation_xyzw", 4),
            "dvl_linear_velocity_body_xyz": _as_list(dvl, "linear_velocity_xyz", 3),
            "dvl_angular_velocity_body_xyz": _as_list(dvl, "angular_velocity_xyz", 3),
            "hardware_orientation_xyzw": _as_list(telemetry, "orientation_xyzw", 4),
            "hardware_angular_velocity_xyz": _as_list(telemetry, "angular_velocity_xyz", 3),
            "hardware_linear_acceleration_xyz": _as_list(telemetry, "linear_acceleration_xyz", 3),
            "hardware_raw_imu_orientation_xyzw": _as_list(hardware_imu, "orientation_xyzw", 4),
            "hardware_raw_imu_angular_velocity_xyz": _as_list(hardware_imu, "angular_velocity_xyz", 3),
            "hardware_raw_imu_linear_acceleration_xyz": _as_list(hardware_imu, "linear_acceleration_xyz", 3),
            "fused_imu_orientation_xyzw": _as_list(fused_imu, "orientation_xyzw", 4),
            "fused_imu_angular_velocity_xyz": _as_list(fused_imu, "angular_velocity_xyz", 3),
            "fused_imu_linear_acceleration_xyz": _as_list(fused_imu, "linear_acceleration_xyz", 3),
            "hardware_depth_m": telemetry.get("depth_m") if telemetry else None,
            "hardware_rpm": _as_list(telemetry, "rpm", 8),
            "depth_raw_sensor_z": _as_list(depth_raw, "position_xyz", 3)[2],
            "depth_fused_world_z": _as_list(depth_fused, "position_xyz", 3)[2],
            "vision_position_xyz": _as_list(vision, "position_xyz", 3),
            "vision_orientation_xyzw": _as_list(vision, "orientation_xyzw", 4),
            "teleop_wrench4": _as_list(wrench, "wrench4", 4),
            "thruster_command_force_n": _as_list(thrust, "values", 8),
            "thruster_echo_applied_force_n": _as_list(echo, "applied_thrust", 8),
            "telemetry_sequence": telemetry.get("telemetry_sequence") if telemetry else None,
            "mcu_time_ms": telemetry.get("mcu_time_ms") if telemetry else None,
            "source_stamp_telemetry_sec": telemetry_stamp,
            "source_stamp_hardware_imu_sec": hardware_imu_stamp,
            "source_stamp_fused_imu_sec": fused_imu_stamp,
            "source_stamp_pose_sec": pose_stamp,
            "source_stamp_dvl_sec": dvl_stamp,
            "source_stamp_vision_sec": vision_stamp,
            "source_stamp_action_sec": action_stamp,
            "source_stamp_thruster_sec": thrust_stamp,
            "source_stamp_echo_sec": echo_stamp,
            "telemetry_age_sec": telemetry_age,
            "hardware_imu_age_sec": hardware_imu_age,
            "fused_imu_age_sec": fused_imu_age,
            "pose_age_sec": pose_age,
            "dvl_age_sec": dvl_age,
            "vision_age_sec": vision_age,
            "action_age_sec": action_age,
            "thruster_age_sec": thrust_age,
            "echo_age_sec": echo_age,
            "depth_raw_age_sec": depth_raw_age,
            "depth_fused_age_sec": depth_fused_age,
            "state_status_age_sec": status_age,
            "telemetry_valid": telemetry_age <= thresholds["telemetry"],
            "hardware_imu_valid": hardware_imu_age <= thresholds["imu"],
            "fused_imu_valid": fused_imu_age <= thresholds["imu"],
            "pose_valid": pose_age <= thresholds["pose"],
            "dvl_valid": dvl_age <= thresholds["dvl"],
            "vision_valid": vision_age <= thresholds["vision"],
            "action_valid": action_age <= thresholds["action"],
            "state_ready": bool(state_status.get("ready", False)),
            "state_imu_fresh": bool(state_status.get("imu_fresh", False)),
            "state_depth_fresh": bool(state_status.get("depth_fresh", False)),
            "state_vision_fresh": bool(state_status.get("vision_fresh", False)),
            "vision_mode": str(state_status.get("vision_mode", "")),
            "reject_reason": str(state_status.get("reject_reason", "")),
        }
        row["synchronization_valid"] = bool(
            row["telemetry_valid"]
            and row["hardware_imu_valid"]
            and row["pose_valid"]
            and row["dvl_valid"]
            and row["action_valid"]
            and row["state_imu_fresh"]
            and row["state_depth_fresh"]
        )
        rows.append(row)
    return rows


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export a recorded FinsROV teleoperation rosbag to raw and aligned Parquet data.")
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--bag-dir", type=Path, default=None, help="Override the rosbag directory. Default: rosbag_recovered if present, otherwise rosbag.")
    parser.add_argument("--rate", type=float, default=30.0)
    parser.add_argument("--telemetry-max-age", type=float, default=0.02)
    parser.add_argument("--imu-max-age", type=float, default=0.02)
    parser.add_argument("--pose-max-age", type=float, default=0.05)
    parser.add_argument("--dvl-max-age", type=float, default=0.03)
    parser.add_argument("--vision-max-age", type=float, default=0.10)
    parser.add_argument("--action-max-age", type=float, default=0.05)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.rate <= 0.0:
        raise SystemExit("--rate must be > 0")
    session_dir = args.session_dir.resolve()
    try:
        bag_dir, bag_source = _select_bag_directory(session_dir, args.bag_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    streams = _read_bag(bag_dir)
    raw_dir = session_dir / "raw_streams"
    for name, records in streams.items():
        _write_rows(raw_dir / f"{name}.parquet", _raw_rows(records))
    thresholds = {
        "telemetry": float(args.telemetry_max_age),
        "imu": float(args.imu_max_age),
        "pose": float(args.pose_max_age),
        "dvl": float(args.dvl_max_age),
        "vision": float(args.vision_max_age),
        "action": float(args.action_max_age),
    }
    trajectory = _build_trajectory(streams, float(args.rate), thresholds)
    _write_rows(session_dir / "trajectory.parquet", trajectory)
    summary = {
        "rate_hz": float(args.rate),
        "bag_directory": str(bag_dir),
        "bag_source": bag_source,
        "thresholds_sec": thresholds,
        "raw_stream_samples": {name: len(records) for name, records in streams.items()},
        "trajectory_samples": len(trajectory),
    }
    (session_dir / "export_summary.json").write_text(
        json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"exported {len(trajectory)} aligned samples from {bag_source} bag into {session_dir}")


if __name__ == "__main__":
    main()
