from __future__ import annotations

import argparse
import time
import numpy as np

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped, Vector3Stamped
from rclpy.node import Node
from std_msgs.msg import Empty, String
from std_srvs.srv import Trigger

from .math_utils import quat_from_controller_ypr


def _build_goal_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Publish a FinsSim motion goal.")
    parser.add_argument("--frame", choices=("controller_world", "controller_body"), default="controller_world")
    parser.add_argument("--x", type=float, required=True)
    parser.add_argument("--y", type=float, required=True)
    parser.add_argument("--z", type=float, required=True)
    parser.add_argument("--yaw", type=float, default=None, help="Target yaw in degrees.")
    parser.add_argument("--pitch", type=float, default=None, help="Target pitch in degrees.")
    parser.add_argument("--roll", type=float, default=None, help="Target roll in degrees.")
    parser.add_argument(
        "--topic",
        type=str,
        default="",
        help="Override topic. Defaults to /motion_controller/command/position_controller_world, "
        "/motion_controller/command/position_controller_body, or /motion_controller/command/pose_controller_body.",
    )
    parser.add_argument("--count", type=int, default=3, help="How many times to publish the command.")
    parser.add_argument("--interval", type=float, default=0.2, help="Delay between publications in seconds.")
    parser.add_argument(
        "--match-timeout",
        type=float,
        default=2.0,
        help="Seconds to wait for a matching subscriber before publishing.",
    )
    return parser


def _wait_for_subscribers(node: Node, pub, *, topic: str, timeout_sec: float) -> None:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while pub.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)
    if pub.get_subscription_count() == 0:
        node.get_logger().warning(
            f"publishing `{topic}` without a matched subscriber; command may be missed"
        )


def _publish_goal(node: Node, args: argparse.Namespace) -> None:
    has_orientation_target = any(value is not None for value in (args.yaw, args.pitch, args.roll))
    target_quat = (
        quat_from_controller_ypr(
            np.deg2rad(float(args.yaw or 0.0)),
            np.deg2rad(float(args.pitch or 0.0)),
            np.deg2rad(float(args.roll or 0.0)),
        )
        if has_orientation_target
        else None
    )

    if args.frame == "controller_world":
        topic = args.topic or "/motion_controller/command/position_controller_world"
        pub = node.create_publisher(PoseStamped, topic, 10)
        msg = PoseStamped()
        msg.header.frame_id = "controller_world"
        msg.pose.position.x = float(args.x)
        msg.pose.position.y = float(args.y)
        msg.pose.position.z = float(args.z)
        if target_quat is not None:
            msg.pose.orientation.x = float(target_quat[0])
            msg.pose.orientation.y = float(target_quat[1])
            msg.pose.orientation.z = float(target_quat[2])
            msg.pose.orientation.w = float(target_quat[3])
    else:
        if target_quat is None:
            topic = args.topic or "/motion_controller/command/position_controller_body"
            pub = node.create_publisher(Vector3Stamped, topic, 10)
            msg = Vector3Stamped()
            msg.header.frame_id = "controller_body"
            msg.vector.x = float(args.x)
            msg.vector.y = float(args.y)
            msg.vector.z = float(args.z)
        else:
            topic = args.topic or "/motion_controller/command/pose_controller_body"
            pub = node.create_publisher(PoseStamped, topic, 10)
            msg = PoseStamped()
            msg.header.frame_id = "controller_body"
            msg.pose.position.x = float(args.x)
            msg.pose.position.y = float(args.y)
            msg.pose.position.z = float(args.z)
            msg.pose.orientation.x = float(target_quat[0])
            msg.pose.orientation.y = float(target_quat[1])
            msg.pose.orientation.z = float(target_quat[2])
            msg.pose.orientation.w = float(target_quat[3])

    yaw_suffix = f", target_yaw_deg={float(args.yaw):+.2f}" if args.yaw is not None else ", target_yaw_deg=<none>"
    node.get_logger().info(
        "publishing position goal: "
        f"topic={topic}, frame_id={msg.header.frame_id}, "
        f"target=[{float(args.x):+.3f}, {float(args.y):+.3f}, {float(args.z):+.3f}]{yaw_suffix}"
    )
    _wait_for_subscribers(node, pub, topic=topic, timeout_sec=float(args.match_timeout))
    for _ in range(max(int(args.count), 1)):
        now = node.get_clock().now().to_msg()
        msg.header.stamp = now
        pub.publish(msg)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(max(float(args.interval), 0.0))


def main(argv: list[str] | None = None) -> None:
    parser = _build_goal_parser()
    args = parser.parse_args(argv)
    rclpy.init(args=None)
    node = Node("send_position_goal")
    try:
        _publish_goal(node, args)
    finally:
        node.destroy_node()
        rclpy.shutdown()


def velocity_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Publish a FinsSim controller-frame velocity command.")
    parser.add_argument("--frame", choices=("controller_body", "controller_world"), default="controller_body")
    parser.add_argument("--vx", type=float, required=True)
    parser.add_argument("--vy", type=float, required=True)
    parser.add_argument("--vz", type=float, required=True)
    parser.add_argument("--wy", type=float, default=0.0, help="Yaw-rate command in controller frame.")
    parser.add_argument("--wp", type=float, default=0.0, help="Pitch-rate command in controller frame.")
    parser.add_argument("--wr", type=float, default=0.0, help="Roll-rate command in controller frame.")
    parser.add_argument(
        "--topic",
        type=str,
        default="/motion_controller/command/velocity_controller_body",
    )
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--match-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = Node("send_velocity_command")
    pub = node.create_publisher(TwistStamped, args.topic, 10)
    msg = TwistStamped()
    msg.header.frame_id = args.frame
    msg.twist.linear.x = float(args.vx)
    msg.twist.linear.y = float(args.vy)
    msg.twist.linear.z = float(args.vz)
    msg.twist.angular.x = float(args.wr)
    msg.twist.angular.y = float(args.wy)
    msg.twist.angular.z = float(args.wp)

    try:
        _wait_for_subscribers(node, pub, topic=args.topic, timeout_sec=float(args.match_timeout))
        for _ in range(max(int(args.count), 1)):
            msg.header.stamp = node.get_clock().now().to_msg()
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(max(float(args.interval), 0.0))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def cancel_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cancel the active FinsSim motion goal.")
    parser.add_argument("--topic", type=str, default="/motion_controller/command/cancel")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--match-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = Node("cancel_goal")
    pub = node.create_publisher(Empty, args.topic, 10)
    msg = Empty()
    try:
        _wait_for_subscribers(node, pub, topic=args.topic, timeout_sec=float(args.match_timeout))
        for _ in range(max(int(args.count), 1)):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(max(float(args.interval), 0.0))
    finally:
        node.destroy_node()
        rclpy.shutdown()


def reset_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Call the FinsSim vehicle reset service.")
    parser.add_argument("--service", type=str, default="/motion_controller/reset_vehicle")
    parser.add_argument("--timeout", type=float, default=5.0)
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = Node("reset_vehicle")
    client = node.create_client(Trigger, args.service)

    try:
        if not client.wait_for_service(timeout_sec=max(float(args.timeout), 0.0)):
            raise RuntimeError(
                f"reset service `{args.service}` is not available; "
                "start `ros2 run motion_control reset_service` first"
            )

        future = client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(node, future, timeout_sec=max(float(args.timeout), 0.0))
        if not future.done():
            raise RuntimeError(f"timed out while waiting for reset service `{args.service}`")
        if future.exception() is not None:
            raise RuntimeError(f"reset service `{args.service}` failed: {future.exception()}")

        response = future.result()
        if response is None:
            raise RuntimeError(f"reset service `{args.service}` returned no response")
        if not response.success:
            raise RuntimeError(f"reset service `{args.service}` returned failure: {response.message}")

        node.get_logger().info(response.message or f"reset service `{args.service}` succeeded")
    finally:
        node.destroy_node()
        rclpy.shutdown()


def control_mode_main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Switch the FinsSim motion controller control source.")
    parser.add_argument("mode", choices=("ros_manual", "unity_random", "ros", "manual", "unity", "random"))
    parser.add_argument("--topic", type=str, default="/motion_controller/control_mode")
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--interval", type=float, default=0.2)
    parser.add_argument("--match-timeout", type=float, default=2.0)
    args = parser.parse_args(argv)

    rclpy.init(args=None)
    node = Node("set_control_mode")
    pub = node.create_publisher(String, args.topic, 10)
    msg = String()
    msg.data = str(args.mode)

    try:
        _wait_for_subscribers(node, pub, topic=args.topic, timeout_sec=float(args.match_timeout))
        for _ in range(max(int(args.count), 1)):
            pub.publish(msg)
            rclpy.spin_once(node, timeout_sec=0.05)
            time.sleep(max(float(args.interval), 0.0))
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
