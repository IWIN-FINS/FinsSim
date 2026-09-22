"""Publish explicit GoalYaw episode boundaries without commanding hardware.

The node deliberately publishes only a goal pose and an auditable event.  It
does not subscribe to a joystick, enable teleoperation, or write thruster
commands; the established teleop and safety chain remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import math

import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node

from msgs.msg import TrajectoryEvent

from .contracts import EVENT_TOPIC, GOAL_POSE_TOPIC, PROTOCOL_VERSION


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a GoalYaw AIRL/GAIL episode event; this never commands vehicle motion.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--episode-id", required=True)
    parser.add_argument("--event", required=True, choices=("episode_start", "episode_stop"))
    parser.add_argument("--goal-x", type=float, default=None)
    parser.add_argument("--goal-y", type=float, default=None)
    parser.add_argument("--goal-z", type=float, default=None)
    parser.add_argument("--goal-yaw-deg", type=float, default=None)
    parser.add_argument("--frame-id", default="map")
    parser.add_argument("--outcome", choices=("success", "abort", "timeout"), default="success")
    return parser.parse_args()


def _goal_metadata(args: argparse.Namespace) -> dict[str, object]:
    if any(value is None for value in (args.goal_x, args.goal_y, args.goal_z, args.goal_yaw_deg)):
        raise ValueError("episode_start requires --goal-x, --goal-y, --goal-z, and --goal-yaw-deg")
    yaw_rad = math.radians(float(args.goal_yaw_deg))
    return {
        "protocol": PROTOCOL_VERSION,
        "episode_id": args.episode_id,
        "goal_frame_id": args.frame_id,
        "goal_position_world_xyz": [float(args.goal_x), float(args.goal_y), float(args.goal_z)],
        "goal_orientation_xyzw": [0.0, 0.0, math.sin(yaw_rad * 0.5), math.cos(yaw_rad * 0.5)],
    }


class GoalYawProtocolPublisher(Node):
    def __init__(self) -> None:
        super().__init__("goal_yaw_irl_protocol")
        self._event_publisher = self.create_publisher(TrajectoryEvent, EVENT_TOPIC, 10)
        self._goal_publisher = self.create_publisher(PoseStamped, GOAL_POSE_TOPIC, 10)

    def publish(self, args: argparse.Namespace) -> None:
        metadata: dict[str, object] = {
            "protocol": PROTOCOL_VERSION,
            "episode_id": args.episode_id,
        }
        event = "goal_yaw_episode_stop"
        if args.event == "episode_start":
            metadata = _goal_metadata(args)
            event = "goal_yaw_episode_start"
            self._publish_goal_pose(metadata)
        else:
            metadata["outcome"] = args.outcome

        for _ in range(3):
            message = TrajectoryEvent()
            message.header.stamp = self.get_clock().now().to_msg()
            message.session_id = args.session_id
            message.event = event
            message.label = args.episode_id
            message.metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
            self._event_publisher.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)

    def _publish_goal_pose(self, metadata: dict[str, object]) -> None:
        position = metadata["goal_position_world_xyz"]
        orientation = metadata["goal_orientation_xyzw"]
        pose = PoseStamped()
        pose.header.stamp = self.get_clock().now().to_msg()
        pose.header.frame_id = str(metadata["goal_frame_id"])
        pose.pose.position.x, pose.pose.position.y, pose.pose.position.z = (float(value) for value in position)  # type: ignore[arg-type]
        pose.pose.orientation.x, pose.pose.orientation.y, pose.pose.orientation.z, pose.pose.orientation.w = (float(value) for value in orientation)  # type: ignore[arg-type]
        self._goal_publisher.publish(pose)


def main() -> None:
    args = _parse_args()
    try:
        rclpy.init()
        node = GoalYawProtocolPublisher()
        try:
            node.publish(args)
        finally:
            node.destroy_node()
            rclpy.shutdown()
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc


if __name__ == "__main__":
    main()
