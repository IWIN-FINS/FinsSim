"""Publish controller-world ``MultiDOFJointTrajectory`` commands.

``send_trajectory`` keeps the small JSON interface useful for ad-hoc ROS
debugging.  ``send_t2_trajectory_goal`` reads a named, validated trajectory
from the recorder's T2 experiment YAML instead, so a manual controller trial
uses exactly the same geometry and timing as a later recorded T2 run.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time
from typing import Any, Sequence

import rclpy
from geometry_msgs.msg import Transform, Twist
from rclpy.node import Node
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
import yaml

from .goal_sender import _wait_for_subscribers


def _build_message(points: Sequence[Sequence[float]], *, label: str) -> MultiDOFJointTrajectory:
    """Build the common translation-only T2 transport message.

    Each row has ``[time_sec, x, y, z, vx, vy, vz]``.  The identity
    quaternion is transport-only: the translation-only PID profile and T2
    PPO profile both keep attitude control disabled.
    """

    if len(points) < 2:
        raise ValueError(f"{label} requires at least two trajectory points")
    message = MultiDOFJointTrajectory()
    message.header.frame_id = "controller_world"
    previous_time = -1.0
    for index, raw in enumerate(points):
        if len(raw) != 7:
            raise ValueError(f"{label}[{index}] must be [time_sec,x,y,z,vx,vy,vz]")
        time_sec, x_m, y_m, z_m, vx_mps, vy_mps, vz_mps = (float(value) for value in raw)
        if time_sec < 0.0 or time_sec < previous_time:
            raise ValueError(f"{label} times must be non-negative and nondecreasing")
        previous_time = time_sec
        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = x_m
        transform.translation.y = y_m
        transform.translation.z = z_m
        transform.rotation.w = 1.0
        point.transforms.append(transform)
        velocity = Twist()
        velocity.linear.x = vx_mps
        velocity.linear.y = vy_mps
        velocity.linear.z = vz_mps
        point.velocities.append(velocity)
        point.time_from_start.sec = int(time_sec)
        point.time_from_start.nanosec = int(round((time_sec - int(time_sec)) * 1e9))
        message.points.append(point)
    return message


def _publish_message(
    message: MultiDOFJointTrajectory,
    *,
    topic: str,
    count: int,
    interval_sec: float,
    match_timeout_sec: float,
    node_name: str,
) -> None:
    rclpy.init(args=None)
    node = Node(node_name)
    publisher = node.create_publisher(MultiDOFJointTrajectory, topic, 10)
    try:
        _wait_for_subscribers(node, publisher, topic=topic, timeout_sec=match_timeout_sec)
        for _ in range(max(int(count), 1)):
            message.header.stamp = node.get_clock().now().to_msg()
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(max(float(interval_sec), 0.0))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Send a controller-world T2 trajectory.")
    parser.add_argument(
        "--points-json",
        required=True,
        help="JSON [[time_sec,x,y,z], ...] or [[time_sec,x,y,z,vx,vy,vz], ...].",
    )
    parser.add_argument("--topic", default="/motion_controller/command/trajectory")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--match-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)
    raw_points = json.loads(args.points_json)
    if not isinstance(raw_points, list) or len(raw_points) < 2:
        raise ValueError("--points-json requires at least two trajectory points")
    normalized: list[list[float]] = []
    for raw in raw_points:
        if not isinstance(raw, list) or len(raw) not in {4, 7}:
            raise ValueError("each point must be [time_sec,x,y,z] or [time_sec,x,y,z,vx,vy,vz]")
        row = [float(value) for value in raw]
        if len(raw) == 7:
            normalized.append(row)
        else:
            normalized.append([*row, 0.0, 0.0, 0.0])
    _publish_message(
        _build_message(normalized, label="--points-json"),
        topic=args.topic,
        count=args.count,
        interval_sec=args.interval,
        match_timeout_sec=args.match_timeout,
        node_name="send_trajectory",
    )


def _load_t2_trajectory(config_path: Path, trajectory_id: str) -> tuple[list[list[float]], dict[str, Any]]:
    """Resolve one enabled, validated T2 path without starting an experiment.

    The pure geometry generator lives with the experiment protocol.  Reusing
    it prevents a manual PID pilot from silently using different radii,
    depths, durations, or velocity samples than the recorder.
    """

    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"T2 config does not exist: {config_path}") from exc
    except yaml.YAMLError as exc:
        raise ValueError(f"invalid T2 config {config_path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"T2 config must contain a mapping: {config_path}")
    try:
        coordinate = payload["coordinate_contract"]
        contract = payload["trajectory_contract"]
        workspace = payload["surveyed_workspace"]
        trajectories = payload["trajectories"]
    except KeyError as exc:
        raise ValueError(f"T2 config is missing required section `{exc.args[0]}`") from exc
    if not all(isinstance(value, dict) for value in (coordinate, contract, workspace, trajectories)):
        raise ValueError("T2 coordinate_contract, trajectory_contract, surveyed_workspace, and trajectories must be mappings")
    spec = trajectories.get(trajectory_id)
    if not isinstance(spec, dict) or not bool(spec.get("enabled", False)):
        available = sorted(
            name for name, candidate in trajectories.items()
            if isinstance(candidate, dict) and bool(candidate.get("enabled", False))
        )
        raise ValueError(f"T2 trajectory `{trajectory_id}` is disabled or unknown; available={available}")
    fixed_depth = coordinate.get("fixed_depth")
    if not isinstance(fixed_depth, dict):
        raise ValueError("T2 coordinate_contract.fixed_depth must be a mapping")

    # Keep the manual sender tied to the same deterministic generator and
    # workspace/speed/acceleration validation used by run_t2_experiment.
    from experiment_recorder.t2_trajectory import generate_trajectory, validate_reference

    points = generate_trajectory(
        spec,
        fixed_depth_m=float(fixed_depth["y_m"]),
        sample_rate_hz=float(contract["sample_rate_hz"]),
    )
    diagnostics = validate_reference(points, workspace)
    rows = [
        [
            point.time_sec,
            point.x_m,
            point.y_m,
            point.z_m,
            point.vx_mps,
            point.vy_mps,
            point.vz_mps,
        ]
        for point in points
    ]
    summary = {
        "config": str(config_path),
        "trajectory": trajectory_id,
        "frame_id": "controller_world",
        "point_count": len(rows),
        "duration_sec": rows[-1][0],
        "start_controller_world": rows[0][1:4],
        "end_controller_world": rows[-1][1:4],
        "peak_diagnostics": diagnostics,
        "orientation_control": "disabled_identity_quaternion_transport",
    }
    return rows, summary


def t2_main(argv: Sequence[str] | None = None) -> None:
    """Publish one named trajectory from ``t2_hardware_experiment.yaml``."""

    parser = argparse.ArgumentParser(
        description="Send one validated T2 YAML trajectory without launching an experiment recorder."
    )
    parser.add_argument("--config", required=True, type=Path, help="T2 experiment YAML path.")
    parser.add_argument("--trajectory", required=True, help="One enabled trajectory ID from the T2 YAML.")
    parser.add_argument("--topic", default="/motion_controller/command/trajectory")
    parser.add_argument("--count", type=int, default=3, help="How many times to publish the complete trajectory message.")
    parser.add_argument("--interval", type=float, default=0.2, help="Seconds between complete-message publications.")
    parser.add_argument("--match-timeout", type=float, default=2.0)
    parser.add_argument("--dry-run", action="store_true", help="Print the resolved geometry and do not initialize ROS2.")
    args = parser.parse_args(argv)

    config_path = args.config.expanduser().resolve()
    rows, summary = _load_t2_trajectory(config_path, str(args.trajectory).strip())
    if args.dry_run:
        print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
        return
    _publish_message(
        _build_message(rows, label=f"T2 trajectory `{args.trajectory}`"),
        topic=str(args.topic),
        count=args.count,
        interval_sec=args.interval,
        match_timeout_sec=args.match_timeout,
        node_name="send_t2_trajectory_goal",
    )
