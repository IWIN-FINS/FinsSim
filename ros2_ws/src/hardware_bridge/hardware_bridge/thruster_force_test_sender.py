from __future__ import annotations

import argparse
import math
from typing import List

from .thruster_curve import THRUSTER_COUNT, THRUSTER_NAMES
from .thruster_test_sender import DEFAULT_TOPIC, _zeros


def make_force_pattern(mode: str, force: float, *, index: int | None = None) -> List[float]:
    force = float(force)
    if not math.isfinite(force):
        raise ValueError("--force must be finite")
    values = _zeros()
    mode_normalized = str(mode).strip().lower().replace("-", "_")
    if mode_normalized == "zero":
        return values
    if mode_normalized == "single":
        if index is None or index < 0 or index >= THRUSTER_COUNT:
            raise ValueError("--index must be in [0, 7] for mode=single")
        values[index] = force
        return values
    if mode_normalized == "horizontal":
        for i in range(4, 8):
            values[i] = force
        return values
    if mode_normalized == "vertical":
        for i in range(4):
            values[i] = force
        return values
    raise ValueError("mode must be one of: zero, single, horizontal, vertical")


def create_thruster_force_test_node(
    *,
    topic: str,
    values: List[float],
    rate_hz: float,
    duration_sec: float,
    send_stop_on_exit: bool,
):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    class ThrusterForceTestSender(Node):
        def __init__(self) -> None:
            super().__init__("send_thruster_force_test")
            if rate_hz <= 0.0:
                raise ValueError("--rate must be > 0")
            if duration_sec < 0.0:
                raise ValueError("--duration must be >= 0")
            self._pub = self.create_publisher(Float32MultiArray, topic, 10)
            self._values = list(values)
            self._rate_hz = float(rate_hz)
            self._duration_sec = float(duration_sec)
            self._send_stop_on_exit = bool(send_stop_on_exit)
            self._start_time = self.get_clock().now()
            self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

            named_values = ", ".join(f"{name}={value:+.3f}N" for name, value in zip(THRUSTER_NAMES, self._values))
            self.get_logger().warn(
                "publishing thruster force test command: "
                f"topic={topic}, rate={self._rate_hz:.1f}Hz, duration={self._duration_sec:.2f}s, "
                f"{named_values}"
            )

        def _on_timer(self) -> None:
            elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
            if self._duration_sec > 0.0 and elapsed >= self._duration_sec:
                if self._send_stop_on_exit:
                    self.publish_values(_zeros())
                self.get_logger().info(f"thruster force test finished after {elapsed:.2f}s")
                rclpy.shutdown()
                return
            self.publish_values(self._values)

        def publish_values(self, values: List[float]) -> None:
            msg = Float32MultiArray()
            msg.data = list(values)
            self._pub.publish(msg)

    return ThrusterForceTestSender()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish 8 force_N commands to /finsrov/thrusters_out. "
            "Use only when hardware_bridge command_mode=force_n. "
            "Order: [V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]."
        )
    )
    parser.add_argument("--mode", choices=("zero", "single", "horizontal", "vertical"), default="single")
    parser.add_argument("--index", type=int, default=None, help="canonical thruster index for mode=single")
    parser.add_argument("--force", type=float, default=0.2, help="force command in N, default: 0.2")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"output topic, default: {DEFAULT_TOPIC}")
    parser.add_argument("--rate", type=float, default=20.0, help="publish rate in Hz, default: 20")
    parser.add_argument("--duration", type=float, default=3.0, help="duration in seconds; 0 means run until Ctrl-C")
    parser.add_argument("--no-stop-on-exit", action="store_true", help="do not publish a final all-zero frame")
    parser.add_argument("--print-only", action="store_true", help="print the force command and exit without ROS publishing")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        values = make_force_pattern(args.mode, args.force, index=args.index)
    except ValueError as exc:
        parser.error(str(exc))

    if args.print_only:
        named_values = ", ".join(f"{name}={value:+.6g}N" for name, value in zip(THRUSTER_NAMES, values))
        print(f"[{', '.join(f'{value:.6g}' for value in values)}]")
        print(named_values)
        return

    import rclpy

    rclpy.init()
    node = create_thruster_force_test_node(
        topic=args.topic,
        values=values,
        rate_hz=args.rate,
        duration_sec=args.duration,
        send_stop_on_exit=not args.no_stop_on_exit,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if not args.no_stop_on_exit:
            node.publish_values(_zeros())
        node.get_logger().info("thruster force test interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
