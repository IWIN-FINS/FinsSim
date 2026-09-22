from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import rclpy
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from msgs.msg import TrajectoryEvent

from .session import prepare_session_directory, session_directory


TOPICS = (
    "/finsrov/teleop/body_wrench_cmd",
    "/finsrov/teleop/status",
    "/finsrov/teleop/enabled",
    "/finsrov/thrusters_out",
    "/finsrov/hardware/telemetry",
    "/finsrov/hardware/imu_raw",
    "/finsrov/hardware/depth_raw",
    "/finsrov/hardware/motor_rpm_raw",
    "/finsrov/hardware/thruster_cmd_echo",
    "/finsrov/hardware/status",
    "/finsrov/vision/tag_poses_3d_camera",
    "/finsrov/vision/refracted_pose_6d",
    "/finsrov/vision/refracted_pose_6d_pure",
    "/finsrov/vision/status",
    "/finsrov/pose",
    "/finsrov/imu_link",
    "/finsrov/depth_link",
    "/finsrov/dvl_link",
    "/finsrov/state/status",
    "/finsrov/trajectory/event",
    "/tf_static",
)


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_revision(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


class SessionEventPublisher(Node):
    def __init__(self) -> None:
        super().__init__("teleop_trajectory_session_recorder")
        self._publisher = self.create_publisher(TrajectoryEvent, "/finsrov/trajectory/event", 10)

    def publish(self, session_id: str, event: str, label: str = "", metadata: dict[str, Any] | None = None) -> bool:
        if not rclpy.ok():
            return False
        message = TrajectoryEvent()
        try:
            message.header.stamp = self.get_clock().now().to_msg()
            message.session_id = session_id
            message.event = event
            message.label = label
            message.metadata_json = json.dumps(metadata or {}, ensure_ascii=True, sort_keys=True)
            self._publisher.publish(message)
        except (RCLError, ExternalShutdownException):
            return False
        return True


def _publish_events(node: SessionEventPublisher, session_id: str, event: str, label: str = "", metadata=None) -> None:
    """Publish repeated lifecycle markers while the ROS context is usable."""

    for _ in range(3):
        if not rclpy.ok() or not node.publish(session_id, event, label, metadata):
            return
        try:
            rclpy.spin_once(node, timeout_sec=0.05)
        except (RCLError, ExternalShutdownException):
            return
        time.sleep(0.1)


def _stop_bag_recorder(child: subprocess.Popen[str]) -> None:
    """Stop the isolated bag process once and wait for its storage flush."""

    if child.poll() is not None:
        return
    try:
        os.killpg(child.pid, signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGTERM)
        try:
            child.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait(timeout=5.0)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record passive FinsROV teleoperation trajectory rosbag data.")
    parser.add_argument("--session-id", default=None, help="Session directory name. Default: timestamped identifier.")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Default: ros2_ws/data/teleop_trajectories.",
    )
    parser.add_argument("--label", default="", help="Optional free-form session label stored in manifest metadata.")
    parser.add_argument("--max-bag-duration", type=float, default=0.0, help="Split bag files after this many seconds; 0 disables splitting.")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing session directory before recording. Ignored by --dry-run.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Write no data; print the rosbag command only.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = _repo_root()
    workspace = repo_root / "ros2_ws"
    session_id = args.session_id or datetime.now().strftime("%Y%m%d_%H%M%S_teleop")
    output_root = args.output_root or workspace / "data" / "teleop_trajectories"
    try:
        session_dir = session_directory(output_root, session_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    bag_dir = session_dir / "rosbag"
    if session_dir.exists() and not args.overwrite:
        raise SystemExit(f"session directory already exists: {session_dir}")

    command = [
        "ros2",
        "bag",
        "record",
        "--output",
        str(bag_dir),
        "--storage",
        "sqlite3",
    ]
    if args.max_bag_duration > 0.0:
        command.extend(["--max-bag-duration", str(args.max_bag_duration)])
    command.extend(TOPICS)
    if args.dry_run:
        print(" ".join(command))
        return

    try:
        prepare_session_directory(session_dir, overwrite=bool(args.overwrite))
    except (FileExistsError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    manifest_path = session_dir / "manifest.json"
    manifest = {
        "session_id": session_id,
        "label": args.label,
        "overwritten": bool(args.overwrite),
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "recording": {"storage": "sqlite3", "compression": "none", "topics": list(TOPICS)},
        "git_revision": _git_revision(repo_root),
        "config_sha256": {
            "teleop": _file_sha256(workspace / "src/teleop/config/finsrov_gamepad_v4_pro1.yaml"),
            "hardware_bridge": _file_sha256(workspace / "src/hardware_bridge/config/finsrov_hardware_bridge_v4_pro1.yaml"),
            "state_fusion": _file_sha256(workspace / "src/state_estimation/config/state_fusion.yaml"),
        },
        "topics": list(TOPICS),
        "ended_at": None,
    }
    _write_json(manifest_path, manifest)

    rclpy.init()
    node = SessionEventPublisher()
    child: subprocess.Popen[str] | None = None
    try:
        # Keep rosbag in its own foreground process group. A terminal Ctrl-C
        # reaches this recorder only; this process then asks rosbag to flush
        # and close exactly once instead of racing two SIGINT handlers.
        child = subprocess.Popen(command, text=True, start_new_session=True)
        time.sleep(0.75)
        _publish_events(node, session_id, "session_start", args.label, {"recorder": "record_teleop_trajectory"})
        if rclpy.ok():
            node.get_logger().info(f"recording session `{session_id}` into `{session_dir}`")
        while child.poll() is None:
            time.sleep(0.2)
    except KeyboardInterrupt:
        if rclpy.ok():
            node.get_logger().info("stopping trajectory recording")
    finally:
        _publish_events(node, session_id, "session_stop")
        if child is not None:
            _stop_bag_recorder(child)
        manifest["ended_at"] = datetime.now().astimezone().isoformat(timespec="seconds")
        _write_json(manifest_path, manifest)
        try:
            node.destroy_node()
        except RCLError:
            pass
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
