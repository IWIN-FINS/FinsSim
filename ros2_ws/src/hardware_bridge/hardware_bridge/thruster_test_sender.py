from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List


THRUSTER_COUNT = 8
DEFAULT_TOPIC = "/finsrov/thrusters_out"
COMMAND_MODE_NORMALIZED_RPM = "normalized_rpm"
COMMAND_MODE_FORCE_N = "force_n"
COMMAND_MODE_NORMALIZED_THROTTLE = "normalized_throttle"
COMMAND_MODES = (COMMAND_MODE_NORMALIZED_RPM, COMMAND_MODE_FORCE_N, COMMAND_MODE_NORMALIZED_THROTTLE)


@dataclass(frozen=True)
class ThrusterPattern:
    name: str
    values: List[float]
    description: str
    command_mode: str = COMMAND_MODE_NORMALIZED_RPM
    unit: str = "normalized"


def _zeros() -> List[float]:
    return [0.0] * THRUSTER_COUNT


def _clamp(value: float, limit: float) -> float:
    if not math.isfinite(value):
        raise ValueError(f"value must be finite, got {value!r}")
    limit = abs(float(limit))
    return max(-limit, min(limit, float(value)))


def _normalize_command_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "rpm": COMMAND_MODE_NORMALIZED_RPM,
        "normalized": COMMAND_MODE_NORMALIZED_RPM,
        "normalized_rpm": COMMAND_MODE_NORMALIZED_RPM,
        "force": COMMAND_MODE_FORCE_N,
        "force_n": COMMAND_MODE_FORCE_N,
        "throttle": COMMAND_MODE_NORMALIZED_THROTTLE,
        "normalized_throttle": COMMAND_MODE_NORMALIZED_THROTTLE,
    }
    if mode not in aliases:
        raise ValueError(f"unknown command mode {value!r}; valid modes: {', '.join(COMMAND_MODES)}")
    return aliases[mode]


def make_thruster_pattern(
    mode: str,
    amplitude: float,
    *,
    turn_limit: float = 0.1,
    index: int | None = None,
    command_mode: str = COMMAND_MODE_NORMALIZED_RPM,
) -> ThrusterPattern:
    """Return Unity-canonical thruster values for hardware mapping tests.

    Canonical order:
      [V_LF, V_LB, V_RB, V_RF, H_LF, H_LB, H_RB, H_RF]

    The aggregate modes follow ControlForPosition.Heuristic:
      vertical: indices 0..3 same sign
      forward/backward: indices 4,5 same sign and 6,7 opposite sign
      turn: indices 4..7 same sign, clamped by turn_limit
    """

    mode = mode.strip().lower().replace("-", "_")
    command_mode = _normalize_command_mode(command_mode)
    if command_mode == COMMAND_MODE_FORCE_N:
        if not math.isfinite(float(amplitude)):
            raise ValueError(f"force amplitude must be finite, got {amplitude!r}")
        amp = float(amplitude)
        unit = "N"
    else:
        amp = _clamp(amplitude, 1.0)
        unit = "normalized"
    values = _zeros()

    if mode == "stop":
        return ThrusterPattern(mode, values, "all thrusters zero", command_mode, unit)

    if mode == "index":
        if index is None:
            raise ValueError("--index is required when mode is index")
        if index < 0 or index >= THRUSTER_COUNT:
            raise ValueError(f"--index must be in [0, {THRUSTER_COUNT - 1}], got {index}")
        values[index] = amp
        return ThrusterPattern(mode, values, f"single canonical thruster index {index}", command_mode, unit)

    if mode in ("up", "ascend"):
        for i in range(4):
            values[i] = amp
        return ThrusterPattern("up", values, "vertical thrusters same positive sign", command_mode, unit)

    if mode in ("down", "descend", "dive"):
        for i in range(4):
            values[i] = -amp
        return ThrusterPattern("down", values, "vertical thrusters same negative sign", command_mode, unit)

    if mode in ("forward", "ahead"):
        values[4] = amp
        values[5] = amp
        values[6] = -amp
        values[7] = -amp
        return ThrusterPattern("forward", values, "ControlForPosition positive horizontal pattern", command_mode, unit)

    if mode in ("backward", "back", "reverse"):
        values[4] = -amp
        values[5] = -amp
        values[6] = amp
        values[7] = amp
        return ThrusterPattern("backward", values, "ControlForPosition negative horizontal pattern", command_mode, unit)

    turn_amp = _clamp(amp, turn_limit) if command_mode != COMMAND_MODE_FORCE_N else math.copysign(
        min(abs(amp), abs(float(turn_limit))),
        amp,
    )
    if mode in ("turn_left", "left", "yaw_left"):
        for i in range(4, 8):
            values[i] = turn_amp
        return ThrusterPattern("turn_left", values, "ControlForPosition positive steering pattern", command_mode, unit)

    if mode in ("turn_right", "right", "yaw_right"):
        for i in range(4, 8):
            values[i] = -turn_amp
        return ThrusterPattern("turn_right", values, "ControlForPosition negative steering pattern", command_mode, unit)

    valid = ", ".join(pattern_names())
    raise ValueError(f"unknown mode {mode!r}; valid modes: {valid}")


def pattern_names() -> Iterable[str]:
    return (
        "stop",
        "forward",
        "backward",
        "turn_left",
        "turn_right",
        "up",
        "down",
        "index",
    )


def pattern_table(amplitude: float, *, turn_limit: float = 0.1) -> Dict[str, List[float]]:
    return {
        name: make_thruster_pattern(name, amplitude, turn_limit=turn_limit, index=0 if name == "index" else None).values
        for name in pattern_names()
    }


def create_thruster_test_node(
    *,
    topic: str,
    mode: str,
    amplitude: float,
    command_mode: str,
    rate_hz: float,
    duration_sec: float,
    turn_limit: float,
    index: int | None,
    send_stop_on_exit: bool,
):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    class ThrusterTestSender(Node):
        def __init__(self) -> None:
            super().__init__("send_thruster_test")
            if rate_hz <= 0.0:
                raise ValueError("--rate must be > 0")
            if duration_sec < 0.0:
                raise ValueError("--duration must be >= 0")

            self._pub = self.create_publisher(Float32MultiArray, topic, 10)
            self._pattern = make_thruster_pattern(
                mode,
                amplitude,
                turn_limit=turn_limit,
                index=index,
                command_mode=command_mode,
            )
            self._rate_hz = float(rate_hz)
            self._duration_sec = float(duration_sec)
            self._send_stop_on_exit = bool(send_stop_on_exit)
            self._start_time = self.get_clock().now()
            self._count = 0
            self._done = False
            self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

            self.get_logger().warn(
                "publishing thruster test pattern: "
                f"topic={topic}, pattern={self._pattern.name}, command_mode={self._pattern.command_mode}, "
                f"unit={self._pattern.unit}, rate={self._rate_hz:.1f}Hz, "
                f"duration={self._duration_sec:.2f}s, values={self._pattern.values}, "
                f"description={self._pattern.description}"
            )

        def _on_timer(self) -> None:
            elapsed = (self.get_clock().now() - self._start_time).nanoseconds * 1e-9
            if self._duration_sec > 0.0 and elapsed >= self._duration_sec:
                if self._send_stop_on_exit:
                    self.publish_values(_zeros())
                self.get_logger().info(f"thruster test finished after {elapsed:.2f}s, published {self._count} frames")
                self._done = True
                self._timer.cancel()
                return

            self.publish_values(self._pattern.values)

        def publish_values(self, values: List[float]) -> None:
            msg = Float32MultiArray()
            msg.data = list(values)
            self._pub.publish(msg)
            self._count += 1

    return ThrusterTestSender()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish Unity-canonical FinsROV thruster test patterns to /finsrov/thrusters_out. "
            "Use --command-mode normalized_rpm for motor_order/sign tests, or --command-mode force_n "
            "to test the force_N -> RPM curve path."
        )
    )
    parser.add_argument("mode", choices=tuple(pattern_names()), help="test pattern to publish")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help=f"output topic, default: {DEFAULT_TOPIC}")
    parser.add_argument(
        "--command-mode",
        choices=COMMAND_MODES,
        default=COMMAND_MODE_NORMALIZED_RPM,
        help="semantic mode expected by hardware_bridge command_mode, default: normalized_rpm",
    )
    parser.add_argument(
        "--amplitude",
        type=float,
        default=0.2,
        help="test magnitude; [-1,1] for normalized modes, N for force_n unless --force is set",
    )
    parser.add_argument("--force", type=float, default=None, help="force magnitude in N; implies --command-mode force_n")
    parser.add_argument("--rate", type=float, default=20.0, help="publish rate in Hz, default: 20")
    parser.add_argument("--duration", type=float, default=3.0, help="duration in seconds; 0 means run until Ctrl-C")
    parser.add_argument(
        "--turn-limit",
        type=float,
        default=0.1,
        help="turn mode clamp; normalized value in normalized modes, N in force_n mode, default: 0.1",
    )
    parser.add_argument("--index", type=int, default=None, help="canonical thruster index for mode=index")
    parser.add_argument("--no-stop-on-exit", action="store_true", help="do not publish a final all-zero frame")
    parser.add_argument("--print-only", action="store_true", help="print the pattern and exit without ROS publishing")
    return parser


def main() -> None:
    import rclpy

    parser = _build_parser()
    args = parser.parse_args()
    command_mode = COMMAND_MODE_FORCE_N if args.force is not None else args.command_mode
    amplitude = args.force if args.force is not None else args.amplitude

    try:
        pattern = make_thruster_pattern(
            args.mode,
            amplitude,
            turn_limit=args.turn_limit,
            index=args.index,
            command_mode=command_mode,
        )
    except ValueError as exc:
        parser.error(str(exc))

    if args.print_only:
        print(f"{pattern.name} command_mode={pattern.command_mode} unit={pattern.unit}: {pattern.values}")
        return

    rclpy.init()
    node = create_thruster_test_node(
        topic=args.topic,
        mode=args.mode,
        amplitude=amplitude,
        command_mode=command_mode,
        rate_hz=args.rate,
        duration_sec=args.duration,
        turn_limit=args.turn_limit,
        index=args.index,
        send_stop_on_exit=not args.no_stop_on_exit,
    )
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        if not args.no_stop_on_exit:
            node.publish_values(_zeros())
        node.get_logger().info("thruster test interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
