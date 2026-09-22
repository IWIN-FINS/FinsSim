from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import rclpy
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, TwistStamped, TwistWithCovarianceStamped, Vector3Stamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Bool, Float32MultiArray, String
from msgs.msg import HardwareTelemetry, ThrusterCommandEcho


def _safe_name(topic: str) -> str:
    name = topic.strip("/") or "root"
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def _stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def _finite_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _header_payload(msg: Any) -> dict[str, Any]:
    header = getattr(msg, "header", None)
    if header is None:
        return {}
    return {
        "frame_id": str(header.frame_id),
        "stamp_sec": _stamp_to_sec(header.stamp),
    }


def _vector_payload(vector: Any) -> dict[str, float | None]:
    return {
        "x": _finite_float(vector.x),
        "y": _finite_float(vector.y),
        "z": _finite_float(vector.z),
    }


def _quaternion_payload(quaternion: Any) -> dict[str, float | None]:
    return {
        "x": _finite_float(quaternion.x),
        "y": _finite_float(quaternion.y),
        "z": _finite_float(quaternion.z),
        "w": _finite_float(quaternion.w),
    }


def _pose_payload(msg: PoseWithCovarianceStamped) -> dict[str, Any]:
    pose = msg.pose.pose
    return {
        **_header_payload(msg),
        "position": _vector_payload(pose.position),
        "orientation": _quaternion_payload(pose.orientation),
    }


def _pose_stamped_payload(msg: PoseStamped) -> dict[str, Any]:
    return {
        **_header_payload(msg),
        "position": _vector_payload(msg.pose.position),
        "orientation": _quaternion_payload(msg.pose.orientation),
    }


def _vector3_stamped_payload(msg: Vector3Stamped) -> dict[str, Any]:
    return {
        **_header_payload(msg),
        "vector": _vector_payload(msg.vector),
    }


def _imu_payload(msg: Imu) -> dict[str, Any]:
    return {
        **_header_payload(msg),
        "orientation": _quaternion_payload(msg.orientation),
        "angular_velocity": _vector_payload(msg.angular_velocity),
        "linear_acceleration": _vector_payload(msg.linear_acceleration),
    }


def _twist_payload(msg: TwistWithCovarianceStamped) -> dict[str, Any]:
    twist = msg.twist.twist
    return {
        **_header_payload(msg),
        "linear": _vector_payload(twist.linear),
        "angular": _vector_payload(twist.angular),
    }


def _twist_stamped_payload(msg: TwistStamped) -> dict[str, Any]:
    return {
        **_header_payload(msg),
        "linear": _vector_payload(msg.twist.linear),
        "angular": _vector_payload(msg.twist.angular),
    }


def _array_payload(msg: Float32MultiArray) -> dict[str, Any]:
    return {"data": [_finite_float(v) for v in msg.data]}


def _thruster_echo_payload(msg: ThrusterCommandEcho) -> dict[str, Any]:
    return {
        **_header_payload(msg),
        "telemetry_sequence": int(msg.telemetry_sequence),
        "mcu_time_ms": int(msg.mcu_time_ms),
        "command_count": int(msg.command_count),
        "receive_time_ms": int(msg.receive_time_ms),
        "command_crc": int(msg.command_crc),
        "enabled": bool(msg.enabled),
        "accepted": bool(msg.accepted),
        "reject_flags": int(msg.reject_flags),
        "received_thrust": [_finite_float(v) for v in msg.received_thrust],
        "applied_thrust": [_finite_float(v) for v in msg.applied_thrust],
        "debug_mask": int(msg.debug_mask),
        "echo_decimation": int(msg.echo_decimation),
    }


def _hardware_telemetry_payload(msg: HardwareTelemetry) -> dict[str, Any]:
    return {
        **_header_payload(msg),
        "telemetry_sequence": int(msg.telemetry_sequence),
        "mcu_time_ms": int(msg.mcu_time_ms),
        "status_flags": int(msg.status_flags),
        "orientation_xyzw": [_finite_float(v) for v in msg.orientation_xyzw],
        "angular_velocity_xyz": [_finite_float(v) for v in msg.angular_velocity_xyz],
        "linear_acceleration_xyz": [_finite_float(v) for v in msg.linear_acceleration_xyz],
        "depth_m": _finite_float(msg.depth_m),
        "pressure_pa": _finite_float(msg.pressure_pa),
        "rpm": [_finite_float(v) for v in msg.rpm],
        "host_receive_time_ns": int(msg.host_receive_time_ns),
    }


def _string_payload(msg: String) -> dict[str, Any]:
    data = str(msg.data)
    try:
        parsed = json.loads(data)
    except json.JSONDecodeError:
        parsed = None
    return {"data": data, "json": parsed}


def _bool_payload(msg: Bool) -> dict[str, Any]:
    return {"data": bool(msg.data)}


@dataclass(frozen=True)
class TopicSpec:
    label: str
    topic: str
    msg_type: type
    extractor: Callable[[Any], dict[str, Any]]
    stale_warn_sec: float = 0.0


class _TopicCsv:
    def __init__(self, output_dir: Path, spec: TopicSpec) -> None:
        self.spec = spec
        self.path = output_dir / f"{_safe_name(spec.label)}__{_safe_name(spec.topic)}.csv"
        self._file = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=["wall_time_sec", "ros_time_sec", "label", "topic", "msg_type", "payload_json"],
        )
        self._writer.writeheader()
        self._closed = False

    def write(self, wall_time_sec: float, ros_time_sec: float, payload: dict[str, Any]) -> None:
        self._writer.writerow(
            {
                "wall_time_sec": f"{wall_time_sec:.9f}",
                "ros_time_sec": f"{ros_time_sec:.9f}",
                "label": self.spec.label,
                "topic": self.spec.topic,
                "msg_type": self.spec.msg_type.__name__,
                "payload_json": json.dumps(payload, ensure_ascii=True, sort_keys=True),
            }
        )

    def close(self) -> None:
        if self._closed:
            return
        self._file.flush()
        self._file.close()
        self._closed = True


class RealSimDiagnosticsRecorder(Node):
    def __init__(self, output_dir: Path, duration_sec: float, stale_warn_sec: float) -> None:
        super().__init__("real_sim_diagnostics_recorder")
        self._output_dir = output_dir
        self._duration_sec = duration_sec
        self._start_wall = self.get_clock().now().nanoseconds * 1e-9
        self._writers: dict[str, _TopicCsv] = {}
        self._last_seen: dict[str, float] = {}
        self._stale_warned: set[str] = set()
        self._samples: dict[str, int] = {}
        self.done = False
        self._closed = False

        output_dir.mkdir(parents=True, exist_ok=True)

        specs = _default_specs(stale_warn_sec)
        self._write_manifest(specs)
        for spec in specs:
            self._writers[spec.topic] = _TopicCsv(output_dir, spec)
            self._samples[spec.topic] = 0
            self.create_subscription(
                spec.msg_type,
                spec.topic,
                self._make_callback(spec),
                10,
            )

        self.create_timer(1.0, self._tick)
        self.get_logger().info(
            f"recording {len(specs)} diagnostic topics for {duration_sec:.1f}s into `{output_dir}`"
        )

    def _make_callback(self, spec: TopicSpec) -> Callable[[Any], None]:
        def _callback(msg: Any) -> None:
            now = self.get_clock().now().nanoseconds * 1e-9
            payload = spec.extractor(msg)
            self._last_seen[spec.topic] = now
            self._samples[spec.topic] += 1
            self._writers[spec.topic].write(now, now, payload)

        return _callback

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        elapsed = now - self._start_wall
        if self._duration_sec > 0.0 and elapsed >= self._duration_sec:
            self.get_logger().info("diagnostic recording duration reached")
            self.done = True
            return

        for writer in self._writers.values():
            spec = writer.spec
            if spec.stale_warn_sec <= 0.0:
                continue
            last = self._last_seen.get(spec.topic)
            stale_for = elapsed if last is None else now - last
            if stale_for >= spec.stale_warn_sec and spec.topic not in self._stale_warned:
                self._stale_warned.add(spec.topic)
                self.get_logger().warn(
                    f"`{spec.topic}` has no recent data for {stale_for:.1f}s; "
                    "real vehicle may be outside perception range or this link is not running"
                )

    def _write_manifest(self, specs: list[TopicSpec]) -> None:
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "duration_sec": self._duration_sec,
            "topics": [
                {
                    "label": spec.label,
                    "topic": spec.topic,
                    "msg_type": spec.msg_type.__name__,
                    "stale_warn_sec": spec.stale_warn_sec,
                }
                for spec in specs
            ],
        }
        (self._output_dir / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    def close(self) -> None:
        if self._closed:
            return
        for writer in self._writers.values():
            writer.close()
        summary = {
            topic: {"samples": count, "file": self._writers[topic].path.name}
            for topic, count in sorted(self._samples.items())
        }
        (self._output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=True, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        self._closed = True


def _default_specs(stale_warn_sec: float) -> list[TopicSpec]:
    return [
        TopicSpec("real_controller_pose", "/finsrov/controller/pose", PoseWithCovarianceStamped, _pose_payload, stale_warn_sec),
        TopicSpec("real_controller_imu", "/finsrov/controller/imu", Imu, _imu_payload, stale_warn_sec),
        TopicSpec("real_controller_depth", "/finsrov/controller/depth", PoseWithCovarianceStamped, _pose_payload, stale_warn_sec),
        TopicSpec("real_controller_dvl", "/finsrov/controller/dvl", TwistWithCovarianceStamped, _twist_payload, stale_warn_sec),
        TopicSpec("real_controller_state_status", "/finsrov/controller/state/status", String, _string_payload, stale_warn_sec),
        TopicSpec("real_fused_pose", "/finsrov/pose", PoseWithCovarianceStamped, _pose_payload, stale_warn_sec),
        TopicSpec("real_fused_imu", "/finsrov/imu_link", Imu, _imu_payload, stale_warn_sec),
        TopicSpec("real_fused_status", "/finsrov/state/status", String, _string_payload, stale_warn_sec),
        TopicSpec("real_vision_status", "/finsrov/vision/status", String, _string_payload, stale_warn_sec),
        TopicSpec("real_command_position_world", "/motion_controller/command/position_controller_world", PoseStamped, _pose_stamped_payload),
        TopicSpec("real_command_position_body", "/motion_controller/command/position_controller_body", Vector3Stamped, _vector3_stamped_payload),
        TopicSpec("real_command_pose_body", "/motion_controller/command/pose_controller_body", PoseStamped, _pose_stamped_payload),
        TopicSpec("real_command_velocity_body", "/motion_controller/command/velocity_controller_body", TwistStamped, _twist_stamped_payload),
        TopicSpec("real_status_active_pose", "/motion_controller/status/active_pose", PoseStamped, _pose_stamped_payload),
        TopicSpec("real_status_error_body", "/motion_controller/status/error_body", Vector3Stamped, _vector3_stamped_payload),
        TopicSpec("real_status_reached", "/motion_controller/status/reached", Bool, _bool_payload),
        TopicSpec("real_debug_observation", "/motion_controller/debug/observation", Float32MultiArray, _array_payload),
        TopicSpec("real_debug_action", "/motion_controller/debug/action", Float32MultiArray, _array_payload),
        TopicSpec("real_debug_wrench6d", "/motion_controller/debug/wrench6d", Float32MultiArray, _array_payload),
        TopicSpec("real_debug_thruster_command", "/motion_controller/debug/thruster_command", Float32MultiArray, _array_payload),
        TopicSpec("real_debug_policy_info", "/motion_controller/debug/policy_info", String, _string_payload),
        TopicSpec("real_thruster_command", "/finsrov/thrusters_out", Float32MultiArray, _array_payload),
        TopicSpec("real_hardware_imu_raw", "/finsrov/hardware/imu_raw", Imu, _imu_payload, stale_warn_sec),
        TopicSpec("real_hardware_rpm", "/finsrov/hardware/motor_rpm_raw", Float32MultiArray, _array_payload),
        TopicSpec(
            "real_hardware_telemetry",
            "/finsrov/hardware/telemetry",
            HardwareTelemetry,
            _hardware_telemetry_payload,
            stale_warn_sec,
        ),
        TopicSpec(
            "real_hardware_echo",
            "/finsrov/hardware/thruster_cmd_echo",
            ThrusterCommandEcho,
            _thruster_echo_payload,
        ),
        TopicSpec("real_hardware_status", "/finsrov/hardware/status", String, _string_payload),
        TopicSpec("sim_controller_pose", "/sim/finsrov/controller/pose", PoseWithCovarianceStamped, _pose_payload, stale_warn_sec),
        TopicSpec("sim_controller_imu", "/sim/finsrov/controller/imu", Imu, _imu_payload, stale_warn_sec),
        TopicSpec("sim_controller_depth", "/sim/finsrov/controller/depth", PoseWithCovarianceStamped, _pose_payload, stale_warn_sec),
        TopicSpec("sim_controller_dvl", "/sim/finsrov/controller/dvl", TwistWithCovarianceStamped, _twist_payload, stale_warn_sec),
        TopicSpec(
            "sim_thruster_applied_wrench",
            "/sim/finsrov/debug/thruster_applied_wrench",
            TwistWithCovarianceStamped,
            _twist_payload,
            stale_warn_sec,
        ),
        TopicSpec("sim_controller_state_status", "/sim/finsrov/controller/state/status", String, _string_payload),
        TopicSpec("sim_command_position_world", "/sim/motion_controller/command/position_controller_world", PoseStamped, _pose_stamped_payload),
        TopicSpec("sim_command_position_body", "/sim/motion_controller/command/position_controller_body", Vector3Stamped, _vector3_stamped_payload),
        TopicSpec("sim_command_pose_body", "/sim/motion_controller/command/pose_controller_body", PoseStamped, _pose_stamped_payload),
        TopicSpec("sim_command_velocity_body", "/sim/motion_controller/command/velocity_controller_body", TwistStamped, _twist_stamped_payload),
        TopicSpec("sim_status_active_pose", "/sim/motion_controller/status/active_pose", PoseStamped, _pose_stamped_payload),
        TopicSpec("sim_status_error_body", "/sim/motion_controller/status/error_body", Vector3Stamped, _vector3_stamped_payload),
        TopicSpec("sim_status_reached", "/sim/motion_controller/status/reached", Bool, _bool_payload),
        TopicSpec("sim_debug_observation", "/sim/motion_controller/debug/observation", Float32MultiArray, _array_payload),
        TopicSpec("sim_debug_action", "/sim/motion_controller/debug/action", Float32MultiArray, _array_payload),
        TopicSpec("sim_debug_wrench6d", "/sim/motion_controller/debug/wrench6d", Float32MultiArray, _array_payload),
        TopicSpec("sim_debug_thruster_command", "/sim/motion_controller/debug/thruster_command", Float32MultiArray, _array_payload),
        TopicSpec("sim_debug_policy_info", "/sim/motion_controller/debug/policy_info", String, _string_payload),
        TopicSpec("sim_thruster_command", "/sim/finsrov/thrusters_out", Float32MultiArray, _array_payload),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record real-vs-sim ROS diagnostics for FinsROV RL deployment.")
    parser.add_argument("--duration", type=float, default=30.0, help="Recording duration in seconds. Use <=0 to run until Ctrl+C.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Default: ros2_ws/data/rl_real_sim_diagnostics/<timestamp>",
    )
    parser.add_argument(
        "--stale-warn-sec",
        type=float,
        default=5.0,
        help="Warn if perception/state topics have no samples for this many seconds.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.output_dir is None:
        repo_root = Path(__file__).resolve()
        env_root = os.environ.get("FINSSIM_REPO_ROOT")
        if env_root:
            workspace = Path(env_root) / "ros2_ws"
        else:
            for parent in repo_root.parents:
                if parent.name == "ros2_ws":
                    workspace = parent
                    break
            else:
                workspace = Path.cwd()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = workspace / "data" / "rl_real_sim_diagnostics" / timestamp
    else:
        output_dir = args.output_dir

    rclpy.init()
    node = RealSimDiagnosticsRecorder(output_dir, float(args.duration), float(args.stale_warn_sec))
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        node.get_logger().info("diagnostic recording interrupted")
    finally:
        node.close()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
