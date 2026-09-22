from __future__ import annotations

import argparse
import math
from typing import Iterable, List

from .thruster_test_sender import DEFAULT_TOPIC, THRUSTER_COUNT, _zeros


CANONICAL_NAMES = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")


def clamp_thruster_values(values: Iterable[float], *, limit: float | None = None) -> List[float]:
    resolved_limit = None if limit is None else abs(float(limit))
    out: List[float] = []
    for value in values:
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"thruster values must be finite, got {value!r}")
        if resolved_limit is not None:
            value_f = max(-resolved_limit, min(resolved_limit, value_f))
        out.append(value_f)
    if len(out) != THRUSTER_COUNT:
        raise ValueError(f"expected {THRUSTER_COUNT} values, got {len(out)}")
    return out


def create_direct_thruster_node(
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

    class DirectThrusterSender(Node):
        def __init__(self) -> None:
            super().__init__("send_direct_thrusters")
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
            self._count = 0
            self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

            named_values = ", ".join(f"{name}={value:+.3f}" for name, value in zip(CANONICAL_NAMES, self._values))
            self.get_logger().warn(
                "publishing direct thruster command: "
                f"topic={topic}, rate={self._rate_hz:.1f}Hz, duration={self._duration_sec:.2f}s, "
                f"{named_values}"
            )

        def _on_timer(self) -> None:
            elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
            if self._duration_sec > 0.0 and elapsed >= self._duration_sec:
                if self._send_stop_on_exit:
                    self.publish_values(_zeros())
                self.get_logger().info(f"direct thruster command finished after {elapsed:.2f}s")
                rclpy.shutdown()
                return
            self.publish_values(self._values)

        def publish_values(self, values: List[float]) -> None:
            msg = Float32MultiArray()
            msg.data = list(values)
            self._pub.publish(msg)
            self._count += 1

    return DirectThrusterSender()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish 8 direct Unity-canonical FinsROV thruster values to /finsrov/thrusters_out. "
            "Order: [V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]."
        )
    )
    parser.add_argument(
        "values",
        nargs=THRUSTER_COUNT,
        type=float,
        metavar="VALUE",
        help="8 finite canonical thruster values; values are clamped only when --limit is supplied",
    )
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"output topic, default: {DEFAULT_TOPIC}")
    parser.add_argument(
        "--limit",
        type=float,
        default=None,
        help="optional absolute command limit; omitted by default so ForceN tests are not sender-clamped",
    )
    parser.add_argument("--rate", type=float, default=20.0, help="publish rate in Hz, default: 20")
    parser.add_argument("--duration", type=float, default=3.0, help="duration in seconds; 0 means run until Ctrl-C")
    parser.add_argument("--no-stop-on-exit", action="store_true", help="do not publish a final all-zero frame")
    parser.add_argument("--print-only", action="store_true", help="print the clamped command and exit without ROS publishing")
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    try:
        values = clamp_thruster_values(args.values, limit=args.limit)
    except ValueError as exc:
        parser.error(str(exc))

    if args.print_only:
        named_values = ", ".join(f"{name}={value:+.3f}" for name, value in zip(CANONICAL_NAMES, values))
        print(f"[{', '.join(f'{value:.6g}' for value in values)}]")
        print(named_values)
        return

    import rclpy

    rclpy.init()
    node = create_direct_thruster_node(
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
        node.get_logger().info("direct thruster command interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
