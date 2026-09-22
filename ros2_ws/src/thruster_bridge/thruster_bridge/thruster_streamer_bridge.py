import math
from typing import List

import rclpy
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .protocol import THRUSTER_COUNT, build_thruster_frame
from .transports import Transport, TransportError, make_transport


def _float_list_param(node: Node, name: str, default: List[float]) -> List[float]:
    value = node.declare_parameter(name, default).value
    return [float(v) for v in value]


def _int_list_param(node: Node, name: str, default: List[int]) -> List[int]:
    value = node.declare_parameter(name, default).value
    return [int(v) for v in value]


class ThrusterStreamerBridge(Node):
    def __init__(self) -> None:
        super().__init__("thruster_streamer_bridge")

        self._input_topic = str(self.declare_parameter("input_topic", "/finsrov/thrusters_out").value)
        self._transport_type = str(self.declare_parameter("transport", "serial").value)
        self._serial_port = str(self.declare_parameter("serial_port", "/dev/ttyUSB0").value)
        self._baudrate = int(self.declare_parameter("baudrate", 115200).value)
        self._host = str(self.declare_parameter("host", "192.168.1.10").value)
        self._port = int(self.declare_parameter("port", 9000).value)
        self._timeout_sec = float(self.declare_parameter("timeout_sec", 0.2).value)
        self._reconnect_interval_sec = float(self.declare_parameter("reconnect_interval_sec", 1.0).value)
        self._clamp = bool(self.declare_parameter("clamp", True).value)
        self._output_scale = float(self.declare_parameter("output_scale", 1.0).value)
        self._enabled = bool(self.declare_parameter("enabled", True).value)
        self._timeout_enabled = bool(self.declare_parameter("timeout_enabled", False).value)
        self._safety_timeout_sec = float(self.declare_parameter("safety_timeout_sec", 0.5).value)
        self._send_rate_limit_hz = float(self.declare_parameter("send_rate_limit_hz", 0.0).value)
        self._log_every_n = int(self.declare_parameter("log_every_n", 100).value)
        self._motor_order = _int_list_param(self, "motor_order", list(range(THRUSTER_COUNT)))
        self._motor_signs = _float_list_param(self, "motor_signs", [1.0] * THRUSTER_COUNT)

        if len(self._motor_order) != THRUSTER_COUNT or sorted(self._motor_order) != list(range(THRUSTER_COUNT)):
            raise ValueError("motor_order must be a permutation of [0, 1, 2, 3, 4, 5, 6, 7]")
        if len(self._motor_signs) != THRUSTER_COUNT:
            raise ValueError("motor_signs must contain 8 values")

        self._transport: Transport = make_transport(
            self._transport_type,
            serial_port=self._serial_port,
            baudrate=self._baudrate,
            host=self._host,
            port=self._port,
            timeout_sec=self._timeout_sec,
            reconnect_interval_sec=self._reconnect_interval_sec,
        )
        self._last_msg_time = None
        self._last_send_time = None
        self._timed_out = False
        self._sent_count = 0
        self._dropped_count = 0

        self._subscription = self.create_subscription(
            Float32MultiArray,
            self._input_topic,
            self._on_thrusters,
            10,
        )
        self._watchdog = self.create_timer(0.05, self._on_watchdog)

        try:
            self._transport.open()
            self.get_logger().info(self._transport_description("opened"))
        except Exception as exc:
            self.get_logger().warn(f"{self._transport_description('open failed')}: {exc}")

        self.get_logger().info(
            "thruster streamer bridge started: "
            f"input_topic={self._input_topic}, transport={self._transport_type}, "
            f"clamp={self._clamp}, output_scale={self._output_scale}, "
            f"safety_timeout_sec={self._safety_timeout_sec}"
        )

    def destroy_node(self) -> bool:
        try:
            self._send_frame([0.0] * THRUSTER_COUNT, enabled=False, force=True)
        except Exception as exc:
            self.get_logger().warn(f"failed to send shutdown zero frame: {exc}")
        self._transport.close()
        return super().destroy_node()

    def _transport_description(self, status: str) -> str:
        if self._transport_type == "serial":
            return f"{status}: serial_port={self._serial_port}, baudrate={self._baudrate}"
        if self._transport_type in {"tcp", "udp"}:
            return f"{status}: {self._transport_type}://{self._host}:{self._port}"
        return f"{status}: transport={self._transport_type}"

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_thrusters(self, msg: Float32MultiArray) -> None:
        self._last_msg_time = self._now_sec()
        self._timed_out = False

        if len(msg.data) != THRUSTER_COUNT:
            self._dropped_count += 1
            self.get_logger().warn(
                f"drop thruster command: expected 8 values, got {len(msg.data)} "
                f"(dropped={self._dropped_count})"
            )
            return

        try:
            values = self._map_values([float(v) for v in msg.data])
            self._send_frame(values, enabled=self._enabled)
        except Exception as exc:
            self._dropped_count += 1
            self.get_logger().warn(f"drop/send thruster command failed: {exc} (dropped={self._dropped_count})")

    def _map_values(self, values: List[float]) -> List[float]:
        mapped = []
        for output_index, input_index in enumerate(self._motor_order):
            value = values[input_index] * self._motor_signs[output_index] * self._output_scale
            if not math.isfinite(value):
                raise ValueError(f"thruster value at index {input_index} is not finite: {values[input_index]!r}")
            if self._clamp:
                value = max(-1.0, min(1.0, value))
            mapped.append(value)
        return mapped

    def _send_frame(self, values: List[float], *, enabled: bool, force: bool = False) -> None:
        now = self._now_sec()
        if not force and self._send_rate_limit_hz > 0.0 and self._last_send_time is not None:
            min_interval = 1.0 / self._send_rate_limit_hz
            if now - self._last_send_time < min_interval:
                return

        frame = build_thruster_frame(values, enabled=enabled)
        try:
            self._transport.send(frame)
        except TransportError:
            raise
        except Exception as exc:
            raise TransportError(str(exc)) from exc

        self._last_send_time = now
        self._sent_count += 1
        if self._log_every_n > 0 and self._sent_count % self._log_every_n == 0:
            self.get_logger().info(
                f"sent {self._sent_count} thruster frames; last enabled={enabled}, "
                f"values={[round(v, 3) for v in values]}"
            )

    def _on_watchdog(self) -> None:
        if self._safety_timeout_sec <= 0.0 or self._last_msg_time is None or self._timed_out:
            return
        elapsed = self._now_sec() - self._last_msg_time
        if elapsed < self._safety_timeout_sec:
            return
        try:
            self._send_frame([0.0] * THRUSTER_COUNT, enabled=self._timeout_enabled, force=True)
            self._timed_out = True
            self.get_logger().warn(
                f"thruster command timeout after {elapsed:.3f}s; sent zero frame "
                f"enabled={self._timeout_enabled}"
            )
        except Exception as exc:
            self.get_logger().warn(f"failed to send timeout zero frame: {exc}")


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ThrusterStreamerBridge()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
