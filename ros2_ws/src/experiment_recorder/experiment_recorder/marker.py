from __future__ import annotations

import argparse
import json
import time

import rclpy
from rclpy.node import Node

from msgs.msg import TrajectoryEvent


EVENTS = (
    "trial_start",
    "phase_start",
    "phase_end",
    "hold_start",
    "safety_abort",
    "trial_end",
    "operator_mark",
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Publish a generic FinsROV experiment event marker.")
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--event", required=True, choices=EVENTS)
    parser.add_argument("--label", default="")
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--topic", default="/finsrov/experiment/event")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    try:
        metadata = json.loads(args.metadata_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--metadata-json must be valid JSON: {exc}") from exc
    if not isinstance(metadata, dict):
        raise SystemExit("--metadata-json must be a JSON object")

    rclpy.init()
    node = Node("finsrov_experiment_marker")
    publisher = node.create_publisher(TrajectoryEvent, args.topic, 10)
    try:
        for _ in range(3):
            message = TrajectoryEvent()
            message.header.stamp = node.get_clock().now().to_msg()
            message.session_id = args.session_id
            message.event = args.event
            message.label = args.label
            message.metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
            publisher.publish(message)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(0.1)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
