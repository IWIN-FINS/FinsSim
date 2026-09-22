from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import List

from builtin_interfaces.msg import Time
from msgs.msg import HardwareTelemetry, ThrusterCommandEcho
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rcl_interfaces.msg import SetParametersResult
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from rclpy.parameter import Parameter
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray, String

from .protocol import (
    DEBUG_MASK_THRUSTER_ECHO,
    THRUSTER_COUNT,
    TelemetryPacket,
    TelemetryStreamParser,
    ThrusterCommandEchoPacket,
    build_debug_config_frame,
    build_thruster_command_frame,
)
from .imu_mounting import ImuMountingTransform
from .thruster_curve import SignedQuadraticThrusterCurve, THRUSTER_NAMES
from .time_sync import McuClockMapper
from .transports import FullDuplexTransport, TransportError, make_transport


COMMAND_MODE_FORCE_N = "force_n"
COMMAND_MODE_NORMALIZED_RPM = "normalized_rpm"
COMMAND_MODE_NORMALIZED_THROTTLE = "normalized_throttle"
COMMAND_MODES = {COMMAND_MODE_FORCE_N, COMMAND_MODE_NORMALIZED_RPM, COMMAND_MODE_NORMALIZED_THROTTLE}


def _ros_time_from_seconds(seconds: float) -> Time:
    """Convert a floating ROS-clock timestamp without using wall-clock time."""
    value = float(seconds)
    if not math.isfinite(value):
        raise ValueError("mapped ROS time must be finite")
    sec = math.floor(value)
    nanosec = int(round((value - sec) * 1e9))
    if nanosec >= 1_000_000_000:
        sec += 1
        nanosec -= 1_000_000_000
    if nanosec < 0:
        sec -= 1
        nanosec += 1_000_000_000
    stamp = Time()
    stamp.sec = int(sec)
    stamp.nanosec = int(nanosec)
    return stamp


def _float_list_param(node: Node, name: str, default: List[float]) -> List[float]:
    return [float(v) for v in node.declare_parameter(name, default).value]


def _int_list_param(node: Node, name: str, default: List[int]) -> List[int]:
    return [int(v) for v in node.declare_parameter(name, default).value]


def _coerce_int_list(value: object) -> List[int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("value must be an integer array")
    return [int(v) for v in value]


def _coerce_float_list(value: object) -> List[float]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes, bytearray)):
        raise ValueError("value must be a float array")
    return [float(v) for v in value]


def _validate_motor_order(values: List[int]) -> None:
    if len(values) != THRUSTER_COUNT or sorted(values) != list(range(THRUSTER_COUNT)):
        raise ValueError("motor_order must be a permutation of [0, 1, 2, 3, 4, 5, 6, 7]")


def _validate_motor_signs(values: List[float]) -> None:
    if len(values) != THRUSTER_COUNT:
        raise ValueError("motor_signs must contain 8 values")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("motor_signs must contain finite values")


def _normalize_command_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    aliases = {
        "force": COMMAND_MODE_FORCE_N,
        "force_n": COMMAND_MODE_FORCE_N,
        "rpm": COMMAND_MODE_NORMALIZED_RPM,
        "normalized_rpm": COMMAND_MODE_NORMALIZED_RPM,
        "throttle": COMMAND_MODE_NORMALIZED_THROTTLE,
        "normalized_throttle": COMMAND_MODE_NORMALIZED_THROTTLE,
        "normalized_direct": COMMAND_MODE_NORMALIZED_RPM,
    }
    if mode not in aliases:
        raise ValueError(f"unknown command_mode `{value}`; expected one of {sorted(COMMAND_MODES)}")
    return aliases[mode]


def map_motor_rpm_to_canonical(
    values: Sequence[float],
    motor_order: Sequence[int],
    motor_signs: Sequence[float],
) -> List[float]:
    if len(values) != THRUSTER_COUNT:
        raise ValueError(f"motor RPM must contain {THRUSTER_COUNT} values")
    mapped = [0.0] * THRUSTER_COUNT
    for output_index, input_index in enumerate(motor_order):
        value = float(values[output_index]) * float(motor_signs[output_index])
        if not math.isfinite(value):
            value = 0.0
        mapped[int(input_index)] = value
    return mapped


def map_canonical_thrusters_to_mcu_normalized(
    values: Sequence[float],
    *,
    command_mode: str,
    output_scale: float,
    clamp: bool,
    motor_order: Sequence[int],
    motor_signs: Sequence[float],
    firmware_max_rpm: float,
    thruster_curve: SignedQuadraticThrusterCurve,
) -> tuple[List[float], List[float], List[float]]:
    if len(values) != THRUSTER_COUNT:
        raise ValueError(f"expected {THRUSTER_COUNT} thruster values, got {len(values)}")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("thruster command values must be finite")

    scaled_input = [float(value) * float(output_scale) for value in values]
    normalized_mode = _normalize_command_mode(command_mode)
    if normalized_mode == COMMAND_MODE_FORCE_N:
        normalized_canonical, target_rpm = thruster_curve.forces_to_normalized_rpm(scaled_input)
    elif normalized_mode in (COMMAND_MODE_NORMALIZED_RPM, COMMAND_MODE_NORMALIZED_THROTTLE):
        normalized_canonical = list(scaled_input)
        target_rpm = [value * firmware_max_rpm for value in normalized_canonical]
    else:
        raise ValueError(f"unsupported command_mode `{command_mode}`")

    if clamp:
        normalized_canonical = [max(-1.0, min(1.0, value)) for value in normalized_canonical]
        target_rpm = [value * firmware_max_rpm for value in normalized_canonical]

    mapped: List[float] = []
    for output_index, input_index in enumerate(motor_order):
        value = normalized_canonical[int(input_index)] * float(motor_signs[output_index])
        if not math.isfinite(value):
            raise ValueError(f"non-finite thruster value at index {input_index}: {values[int(input_index)]!r}")
        if clamp:
            value = max(-1.0, min(1.0, value))
        mapped.append(value)
    return mapped, normalized_canonical, target_rpm


class HardwareBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("hardware_bridge")

        self._transport_type = str(self.declare_parameter("transport", "serial").value)
        self._serial_port = str(self.declare_parameter("serial_port", "/dev/ttyUSB0").value)
        self._baudrate = int(self.declare_parameter("baudrate", 115200).value)
        self._host = str(self.declare_parameter("host", "192.168.1.10").value)
        self._port = int(self.declare_parameter("port", 9000).value)
        self._udp_bind_host = str(self.declare_parameter("udp_bind_host", "0.0.0.0").value)
        self._udp_bind_port = int(self.declare_parameter("udp_bind_port", 54321).value)
        self._udp_remote_host = str(self.declare_parameter("udp_remote_host", "192.168.0.2").value)
        self._udp_remote_port = int(self.declare_parameter("udp_remote_port", 58766).value)
        self._timeout_sec = float(self.declare_parameter("timeout_sec", 0.2).value)
        self._reconnect_interval_sec = float(self.declare_parameter("reconnect_interval_sec", 1.0).value)
        self._input_thruster_topic = str(
            self.declare_parameter("input_thruster_topic", "/finsrov/thrusters_out").value
        )
        self._imu_topic = str(self.declare_parameter("imu_raw_topic", "/finsrov/hardware/imu_raw").value)
        self._depth_topic = str(self.declare_parameter("depth_raw_topic", "/finsrov/hardware/depth_raw").value)
        self._status_topic = str(self.declare_parameter("status_topic", "/finsrov/hardware/status").value)
        self._thruster_echo_topic = str(
            self.declare_parameter("thruster_echo_topic", "/finsrov/hardware/thruster_cmd_echo").value
        )
        self._motor_rpm_topic = str(
            self.declare_parameter("motor_rpm_raw_topic", "/finsrov/hardware/motor_rpm_raw").value
        )
        self._hardware_telemetry_topic = str(
            self.declare_parameter("hardware_telemetry_topic", "/finsrov/hardware/telemetry").value
        )
        self._frame_id = str(self.declare_parameter("frame_id", "finsrov_base_link").value)
        self._depth_frame_id = str(self.declare_parameter("depth_frame_id", "finsrov_depth_link").value)
        self._imu_mounting = ImuMountingTransform(
            _float_list_param(
                self,
                "imu_sensor_to_base_quaternion_xyzw",
                [0.0, 0.0, 1.0, 0.0],
            )
        )
        self._command_mode = _normalize_command_mode(str(self.declare_parameter("command_mode", COMMAND_MODE_FORCE_N).value))
        self._firmware_max_rpm = float(self.declare_parameter("firmware_max_rpm", 3000.0).value)
        self._thruster_curve = SignedQuadraticThrusterCurve(
            c1_positive=tuple(
                _float_list_param(
                    self,
                    "thruster_curve.c1_positive",
                    [1.0e-4] * THRUSTER_COUNT,
                )
            ),
            c1_negative=tuple(
                _float_list_param(
                    self,
                    "thruster_curve.c1_negative",
                    [1.0e-4] * THRUSTER_COUNT,
                )
            ),
            rpm_min=tuple(
                _float_list_param(
                    self,
                    "thruster_curve.rpm_min",
                    [-3000.0] * THRUSTER_COUNT,
                )
            ),
            rpm_max=tuple(
                _float_list_param(
                    self,
                    "thruster_curve.rpm_max",
                    [3000.0] * THRUSTER_COUNT,
                )
            ),
            force_deadband_n=tuple(
                _float_list_param(
                    self,
                    "thruster_curve.force_deadband_n",
                    [0.0] * THRUSTER_COUNT,
                )
            ),
            min_effective_rpm_positive=tuple(
                _float_list_param(
                    self,
                    "thruster_curve.min_effective_rpm_positive",
                    [0.0] * THRUSTER_COUNT,
                )
            ),
            min_effective_rpm_negative=tuple(
                _float_list_param(
                    self,
                    "thruster_curve.min_effective_rpm_negative",
                    [0.0] * THRUSTER_COUNT,
                )
            ),
            firmware_max_rpm=self._firmware_max_rpm,
        )
        self._clamp = bool(self.declare_parameter("clamp", True).value)
        self._output_scale = float(self.declare_parameter("output_scale", 1.0).value)
        self._enabled = bool(self.declare_parameter("enabled", True).value)
        self._send_zero_on_start = bool(self.declare_parameter("send_zero_on_start", True).value)
        self._timeout_enabled = bool(self.declare_parameter("timeout_enabled", False).value)
        self._safety_timeout_sec = float(self.declare_parameter("safety_timeout_sec", 0.5).value)
        self._telemetry_rate_timeout_sec = float(self.declare_parameter("telemetry_rate_timeout_sec", 0.5).value)
        self._read_rate_hz = float(self.declare_parameter("read_rate_hz", 200.0).value)
        self._send_rate_limit_hz = float(self.declare_parameter("send_rate_limit_hz", 0.0).value)
        self._status_rate_hz = float(self.declare_parameter("status_rate_hz", 1.0).value)
        self._log_every_n = int(self.declare_parameter("log_every_n", 100).value)
        self._debug_mask = int(self.declare_parameter("debug_mask", 0).value)
        self._debug_echo_decimation = int(self.declare_parameter("debug_echo_decimation", 1).value)
        self._debug_config_send_on_start = bool(self.declare_parameter("debug_config_send_on_start", True).value)
        self._debug_config_retry_period_sec = float(
            self.declare_parameter("debug_config_retry_period_sec", 0.1).value
        )
        self._motor_order = _int_list_param(self, "motor_order", list(range(THRUSTER_COUNT)))
        self._motor_signs = _float_list_param(self, "motor_signs", [1.0] * THRUSTER_COUNT)

        _validate_motor_order(self._motor_order)
        _validate_motor_signs(self._motor_signs)

        self._transport: FullDuplexTransport = make_transport(
            self._transport_type,
            serial_port=self._serial_port,
            baudrate=self._baudrate,
            host=self._host,
            port=self._port,
            udp_bind_host=self._udp_bind_host,
            udp_bind_port=self._udp_bind_port,
            udp_remote_host=self._udp_remote_host,
            udp_remote_port=self._udp_remote_port,
            timeout_sec=self._timeout_sec,
            reconnect_interval_sec=self._reconnect_interval_sec,
        )
        self._parser = TelemetryStreamParser()
        self._mcu_clock_mapper = McuClockMapper()

        self._last_thruster_time: float | None = None
        self._last_send_time: float | None = None
        self._last_telemetry_time: float | None = None
        self._thruster_timed_out = False
        self._telemetry_timed_out = False
        self._sent_count = 0
        self._telemetry_count = 0
        self._thruster_echo_count = 0
        self._motor_rpm_count = 0
        self._last_motor_rpm = [0.0] * THRUSTER_COUNT
        self._last_motor_rpm_mcu = [0.0] * THRUSTER_COUNT
        self._last_status_flags = 0
        self._last_input_command = [0.0] * THRUSTER_COUNT
        self._last_target_rpm = [0.0] * THRUSTER_COUNT
        self._last_normalized_command = [0.0] * THRUSTER_COUNT
        self._last_mcu_command = [0.0] * THRUSTER_COUNT
        self._send_error_count = 0
        self._read_error_count = 0
        self._debug_config_sent_count = 0
        self._debug_config_error_count = 0
        self._debug_config_sequence = 0
        self._debug_config_dirty = self._debug_config_send_on_start

        self._imu_pub = self.create_publisher(Imu, self._imu_topic, 10)
        self._depth_pub = self.create_publisher(PoseWithCovarianceStamped, self._depth_topic, 10)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)
        self._thruster_echo_pub = self.create_publisher(ThrusterCommandEcho, self._thruster_echo_topic, 10)
        self._motor_rpm_pub = self.create_publisher(Float32MultiArray, self._motor_rpm_topic, 10)
        self._hardware_telemetry_pub = self.create_publisher(HardwareTelemetry, self._hardware_telemetry_topic, 10)
        self._thruster_sub = self.create_subscription(
            Float32MultiArray,
            self._input_thruster_topic,
            self._thruster_callback,
            10,
        )
        self._read_timer = self.create_timer(1.0 / max(self._read_rate_hz, 1.0), self._read_transport)
        self._watchdog_timer = self.create_timer(0.05, self._watchdog)
        self._status_timer = self.create_timer(1.0 / max(self._status_rate_hz, 0.1), self._publish_status)
        self._debug_config_timer = self.create_timer(
            max(self._debug_config_retry_period_sec, 0.02),
            self._flush_debug_config,
        )
        self.add_on_set_parameters_callback(self._set_parameters_callback)

        try:
            self._transport.open()
            self.get_logger().info(self._transport_description("opened"))
            if self._send_zero_on_start:
                self._send_thrusters([0.0] * THRUSTER_COUNT, enabled=self._enabled, force=True)
                self.get_logger().info(f"sent startup zero frame enabled={self._enabled}")
        except Exception as exc:
            self.get_logger().warn(f"{self._transport_description('open failed')}: {exc}")

        self.get_logger().info(
            "hardware bridge started: "
            f"transport={self._transport_type}, input_thruster_topic={self._input_thruster_topic}, "
            f"imu_topic={self._imu_topic}, depth_topic={self._depth_topic}, "
            f"thruster_echo_topic={self._thruster_echo_topic}, motor_rpm_topic={self._motor_rpm_topic}, "
            f"hardware_telemetry_topic={self._hardware_telemetry_topic}, "
            f"imu_sensor_to_base_quaternion_xyzw={list(self._imu_mounting.quaternion_xyzw)}, "
            f"command_mode={self._command_mode}, firmware_max_rpm={self._firmware_max_rpm:.1f}, "
            f"debug_mask=0x{self._debug_mask:08x}, "
            f"debug_echo_decimation={self._debug_echo_decimation}"
        )

    def destroy_node(self) -> bool:
        try:
            self._send_thrusters([0.0] * THRUSTER_COUNT, enabled=False, force=True)
        except Exception as exc:
            self.get_logger().warn(f"failed to send shutdown zero frame: {exc}")
        self._transport.close()
        return super().destroy_node()

    def _transport_description(self, status: str) -> str:
        if self._transport_type == "serial":
            return f"{status}: serial_port={self._serial_port}, baudrate={self._baudrate}"
        if self._transport_type == "tcp":
            return f"{status}: tcp://{self._host}:{self._port}"
        if self._transport_type == "udp":
            return (
                f"{status}: udp bind={self._udp_bind_host}:{self._udp_bind_port}, "
                f"remote={self._udp_remote_host}:{self._udp_remote_port}"
            )
        return f"{status}: transport={self._transport_type}"

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _set_parameters_callback(self, params: list[Parameter]) -> SetParametersResult:
        next_debug_mask = self._debug_mask
        next_debug_echo_decimation = self._debug_echo_decimation
        next_enabled = self._enabled
        next_command_mode = self._command_mode
        next_motor_order = self._motor_order
        next_motor_signs = self._motor_signs
        for param in params:
            try:
                if param.name == "imu_sensor_to_base_quaternion_xyzw":
                    return SetParametersResult(
                        successful=False,
                        reason="imu_sensor_to_base_quaternion_xyzw is startup-only",
                    )
                if param.name == "debug_mask":
                    next_debug_mask = int(param.value)
                elif param.name == "debug_echo_decimation":
                    next_debug_echo_decimation = int(param.value)
                elif param.name == "enabled":
                    next_enabled = bool(param.value)
                elif param.name == "command_mode":
                    next_command_mode = _normalize_command_mode(str(param.value))
                elif param.name == "motor_order":
                    next_motor_order = _coerce_int_list(param.value)
                elif param.name == "motor_signs":
                    next_motor_signs = _coerce_float_list(param.value)
            except ValueError as exc:
                return SetParametersResult(successful=False, reason=str(exc))

        if next_debug_mask < 0 or next_debug_mask > 0xFFFFFFFF:
            return SetParametersResult(successful=False, reason="debug_mask must fit uint32")
        if next_debug_echo_decimation < 1 or next_debug_echo_decimation > 0xFFFF:
            return SetParametersResult(successful=False, reason="debug_echo_decimation must be in [1, 65535]")
        try:
            _validate_motor_order(next_motor_order)
            _validate_motor_signs(next_motor_signs)
        except ValueError as exc:
            return SetParametersResult(successful=False, reason=str(exc))

        changed = (
            next_debug_mask != self._debug_mask
            or next_debug_echo_decimation != self._debug_echo_decimation
        )
        enabled_changed = next_enabled != self._enabled
        command_mode_changed = next_command_mode != self._command_mode
        motor_mapping_changed = (
            next_motor_order != self._motor_order
            or next_motor_signs != self._motor_signs
        )
        self._debug_mask = next_debug_mask
        self._debug_echo_decimation = next_debug_echo_decimation
        self._enabled = next_enabled
        self._command_mode = next_command_mode
        self._motor_order = next_motor_order
        self._motor_signs = next_motor_signs
        if changed:
            self._debug_config_dirty = True
            self.get_logger().info(
                "queued debug config: "
                f"debug_mask=0x{self._debug_mask:08x}, "
                f"debug_echo_decimation={self._debug_echo_decimation}, "
                f"thruster_echo={'on' if self._debug_mask & DEBUG_MASK_THRUSTER_ECHO else 'off'}"
            )
        if enabled_changed:
            self.get_logger().warn(f"hardware bridge {'armed' if self._enabled else 'disarmed'}")
            if not self._enabled:
                try:
                    self._send_thrusters([0.0] * THRUSTER_COUNT, enabled=False, force=True)
                except Exception as exc:
                    self._send_error_count += 1
                    self.get_logger().warn(f"failed to send disarm zero frame: {exc}")
        if command_mode_changed:
            try:
                self._send_thrusters([0.0] * THRUSTER_COUNT, enabled=False, force=True)
            except Exception as exc:
                self._send_error_count += 1
                self.get_logger().warn(f"failed to send command mode change zero frame: {exc}")
            self.get_logger().warn(f"updated command_mode={self._command_mode}; sent disabled zero frame")
        if motor_mapping_changed:
            try:
                self._send_thrusters([0.0] * THRUSTER_COUNT, enabled=False, force=True)
            except Exception as exc:
                self._send_error_count += 1
                self.get_logger().warn(f"failed to send motor mapping change zero frame: {exc}")
            self.get_logger().warn(
                "updated motor mapping; sent disabled zero frame: "
                f"motor_order={self._motor_order}, motor_signs={self._motor_signs}"
            )
        return SetParametersResult(successful=True)

    def _thruster_callback(self, msg: Float32MultiArray) -> None:
        self._last_thruster_time = self._now_sec()
        self._thruster_timed_out = False
        if len(msg.data) != THRUSTER_COUNT:
            self.get_logger().warn(f"drop thruster command: expected 8 values, got {len(msg.data)}")
            return
        try:
            values = self._map_thrusters([float(v) for v in msg.data])
            self._send_thrusters(values, enabled=self._enabled)
        except Exception as exc:
            self._send_error_count += 1
            self.get_logger().warn(f"failed to send thruster command: {exc}")

    def _map_thrusters(self, values: List[float]) -> List[float]:
        mapped, normalized_canonical, target_rpm = map_canonical_thrusters_to_mcu_normalized(
            values,
            command_mode=self._command_mode,
            output_scale=self._output_scale,
            clamp=self._clamp,
            motor_order=self._motor_order,
            motor_signs=self._motor_signs,
            firmware_max_rpm=self._firmware_max_rpm,
            thruster_curve=self._thruster_curve,
        )
        self._last_input_command = [float(value) * self._output_scale for value in values]
        self._last_target_rpm = list(target_rpm)
        self._last_normalized_command = list(normalized_canonical)
        self._last_mcu_command = list(mapped)
        return mapped

    def _map_motor_rpm_to_canonical(self, values: Sequence[float]) -> List[float]:
        return map_motor_rpm_to_canonical(values, self._motor_order, self._motor_signs)

    def _send_thrusters(self, values: List[float], *, enabled: bool, force: bool = False) -> None:
        now = self._now_sec()
        if not force and self._send_rate_limit_hz > 0.0 and self._last_send_time is not None:
            if now - self._last_send_time < 1.0 / self._send_rate_limit_hz:
                return
        frame = build_thruster_command_frame(values, enabled=enabled)
        self._transport.write(frame)
        self._last_send_time = now
        self._sent_count += 1
        if self._log_every_n > 0 and self._sent_count % self._log_every_n == 0:
            self.get_logger().info(
                f"sent {self._sent_count} thruster frames; enabled={enabled}, "
                f"values={[round(v, 3) for v in values]}"
            )

    def _flush_debug_config(self) -> None:
        if not self._debug_config_dirty:
            return
        try:
            frame = build_debug_config_frame(
                self._debug_mask,
                self._debug_echo_decimation,
                sequence=self._debug_config_sequence,
            )
            self._transport.write(frame)
            self._debug_config_sequence = (self._debug_config_sequence + 1) & 0xFFFF
            self._debug_config_sent_count += 1
            self._debug_config_dirty = False
            self.get_logger().info(
                "sent debug config: "
                f"debug_mask=0x{self._debug_mask:08x}, "
                f"debug_echo_decimation={self._debug_echo_decimation}"
            )
        except Exception as exc:
            self._debug_config_error_count += 1
            if self._debug_config_error_count <= 5 or self._debug_config_error_count % 50 == 0:
                self.get_logger().warn(f"failed to send debug config: {exc}")

    def _read_transport(self) -> None:
        try:
            data = self._transport.read(512)
        except TransportError as exc:
            self._read_error_count += 1
            if self._read_error_count <= 5 or self._read_error_count % 50 == 0:
                self.get_logger().warn(f"telemetry read failed: {exc}")
            return
        for packet in self._parser.feed(data):
            if isinstance(packet, TelemetryPacket):
                self._handle_telemetry(packet)
            elif isinstance(packet, ThrusterCommandEchoPacket):
                self._handle_thruster_echo(packet)

    def _handle_telemetry(self, packet: TelemetryPacket) -> None:
        now_sec = self._now_sec()
        self._last_telemetry_time = now_sec
        self._telemetry_timed_out = False
        self._telemetry_count += 1
        self._last_status_flags = int(packet.status_flags) & 0xFFFFFFFF

        receive_stamp = self.get_clock().now().to_msg()
        clock_estimate = self._mcu_clock_mapper.update(packet.mcu_time_ms, now_sec)
        telemetry_stamp = _ros_time_from_seconds(clock_estimate.mapped_ros_time_sec)
        imu = Imu()
        # Keep this host-time stamp unchanged: state_fusion freshness checks
        # currently use /finsrov/hardware/imu_raw and must not change.
        imu.header.stamp = receive_stamp
        imu.header.frame_id = self._frame_id
        quat_w, quat_x, quat_y, quat_z = packet.quat_wxyz
        try:
            corrected_imu = self._imu_mounting.correct_sample(
                orientation_world_sensor_xyzw=[quat_x, quat_y, quat_z, quat_w],
                angular_velocity_sensor_xyz=packet.angular_velocity_xyz,
                linear_acceleration_sensor_xyz=packet.linear_acceleration_xyz,
            )
        except ValueError as exc:
            self._read_error_count += 1
            if self._read_error_count <= 5 or self._read_error_count % 50 == 0:
                self.get_logger().warn(f"discarding telemetry with invalid IMU sample: {exc}")
            return
        quat_x, quat_y, quat_z, quat_w = corrected_imu.orientation_xyzw
        imu.orientation.x = float(quat_x)
        imu.orientation.y = float(quat_y)
        imu.orientation.z = float(quat_z)
        imu.orientation.w = float(quat_w)
        imu.angular_velocity.x = corrected_imu.angular_velocity_xyz[0]
        imu.angular_velocity.y = corrected_imu.angular_velocity_xyz[1]
        imu.angular_velocity.z = corrected_imu.angular_velocity_xyz[2]
        imu.linear_acceleration.x = corrected_imu.linear_acceleration_xyz[0]
        imu.linear_acceleration.y = corrected_imu.linear_acceleration_xyz[1]
        imu.linear_acceleration.z = corrected_imu.linear_acceleration_xyz[2]
        imu.orientation_covariance = self._imu_mounting.transform_covariance(
            [0.02, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.05]
        )
        imu.angular_velocity_covariance = self._imu_mounting.transform_covariance(
            [0.01, 0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.01]
        )
        imu.linear_acceleration_covariance = self._imu_mounting.transform_covariance(
            [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
        )
        self._imu_pub.publish(imu)

        depth = PoseWithCovarianceStamped()
        depth.header.stamp = receive_stamp
        depth.header.frame_id = self._depth_frame_id
        depth.pose.pose.position.z = float(packet.depth_m)
        depth.pose.covariance[14] = 0.0025
        self._depth_pub.publish(depth)

        rpm_msg = Float32MultiArray()
        self._last_motor_rpm_mcu = [float(v) if math.isfinite(float(v)) else 0.0 for v in packet.motor_rpm]
        rpm_msg.data = self._map_motor_rpm_to_canonical(self._last_motor_rpm_mcu)
        self._last_motor_rpm = list(rpm_msg.data)
        self._motor_rpm_count += 1
        self._motor_rpm_pub.publish(rpm_msg)

        telemetry = HardwareTelemetry()
        telemetry.header.stamp = telemetry_stamp
        telemetry.header.frame_id = self._frame_id
        telemetry.telemetry_sequence = int(packet.sequence) & 0xFFFF
        telemetry.mcu_time_ms = int(packet.mcu_time_ms) & 0xFFFFFFFF
        telemetry.status_flags = int(packet.status_flags) & 0xFFFFFFFF
        telemetry.orientation_xyzw = [float(quat_x), float(quat_y), float(quat_z), float(quat_w)]
        telemetry.angular_velocity_xyz = list(corrected_imu.angular_velocity_xyz)
        telemetry.linear_acceleration_xyz = list(corrected_imu.linear_acceleration_xyz)
        telemetry.depth_m = float(packet.depth_m)
        telemetry.pressure_pa = float(packet.pressure_pa)
        telemetry.rpm = list(self._last_motor_rpm)
        telemetry.host_receive_time_ns = int(now_sec * 1e9)
        self._hardware_telemetry_pub.publish(telemetry)

    def _handle_thruster_echo(self, packet: ThrusterCommandEchoPacket) -> None:
        self._thruster_echo_count += 1
        msg = ThrusterCommandEcho()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.telemetry_sequence = int(packet.sequence) & 0xFFFF
        msg.mcu_time_ms = int(packet.mcu_time_ms) & 0xFFFFFFFF
        msg.command_count = int(packet.command_count) & 0xFFFFFFFF
        msg.receive_time_ms = int(packet.receive_time_ms) & 0xFFFFFFFF
        msg.command_crc = int(packet.command_crc) & 0xFFFF
        msg.enabled = bool(packet.enabled)
        msg.accepted = bool(packet.accepted)
        msg.reject_flags = int(packet.reject_flags) & 0xFFFFFFFF
        msg.received_thrust = [float(v) for v in packet.received_thrust]
        msg.applied_thrust = [float(v) for v in packet.applied_thrust]
        msg.debug_mask = int(packet.debug_mask) & 0xFFFFFFFF
        msg.echo_decimation = int(packet.echo_decimation) & 0xFFFF
        self._thruster_echo_pub.publish(msg)

    def _watchdog(self) -> None:
        now = self._now_sec()
        if (
            self._safety_timeout_sec > 0.0
            and self._last_thruster_time is not None
            and not self._thruster_timed_out
            and now - self._last_thruster_time >= self._safety_timeout_sec
        ):
            try:
                self._send_thrusters([0.0] * THRUSTER_COUNT, enabled=self._timeout_enabled, force=True)
                self._thruster_timed_out = True
                self.get_logger().warn(
                    f"thruster command timeout after {now - self._last_thruster_time:.3f}s; "
                    f"sent zero frame enabled={self._timeout_enabled}"
                )
            except Exception as exc:
                self._send_error_count += 1
                self.get_logger().warn(f"failed to send timeout zero frame: {exc}")

        if (
            self._telemetry_rate_timeout_sec > 0.0
            and self._last_telemetry_time is not None
            and not self._telemetry_timed_out
            and now - self._last_telemetry_time >= self._telemetry_rate_timeout_sec
        ):
            self._telemetry_timed_out = True
            self.get_logger().warn(f"telemetry timeout after {now - self._last_telemetry_time:.3f}s")

    def _publish_status(self) -> None:
        mcu_diagnostics_supported = bool(self._last_status_flags & (1 << 31))
        payload = {
            "transport": self._transport_type,
            "udp_bind": f"{self._udp_bind_host}:{self._udp_bind_port}",
            "udp_remote": f"{self._udp_remote_host}:{self._udp_remote_port}",
            "sent_thruster_frames": self._sent_count,
            "telemetry_frames": self._telemetry_count,
            "thruster_echo_frames": self._thruster_echo_count,
            "motor_rpm_frames": self._motor_rpm_count,
            "last_motor_rpm": self._last_motor_rpm,
            "last_motor_rpm_mcu": self._last_motor_rpm_mcu,
            "last_status_flags": self._last_status_flags,
            "mcu_diagnostics_supported": mcu_diagnostics_supported,
            "mcu_command_count_low8": ((self._last_status_flags >> 16) & 0xFF)
            if mcu_diagnostics_supported
            else None,
            "mcu_has_received_command": bool(self._last_status_flags & (1 << 24))
            if mcu_diagnostics_supported
            else None,
            "mcu_debug_echo_enabled": bool(self._last_status_flags & (1 << 25))
            if mcu_diagnostics_supported
            else None,
            "mcu_direct_thrusters_enabled": bool(self._last_status_flags & (1 << 26))
            if mcu_diagnostics_supported
            else None,
            "mcu_target_rpm_nonzero": bool(self._last_status_flags & (1 << 27))
            if mcu_diagnostics_supported
            else None,
            "mcu_target_throttle_nonzero": bool(self._last_status_flags & (1 << 28))
            if mcu_diagnostics_supported
            else None,
            "mcu_motor_handle_ran": bool(self._last_status_flags & (1 << 29))
            if mcu_diagnostics_supported
            else None,
            "mcu_rpm_control_enabled": bool(self._last_status_flags & (1 << 30))
            if mcu_diagnostics_supported
            else None,
            "command_mode": self._command_mode,
            "firmware_max_rpm": self._firmware_max_rpm,
            "thruster_names": list(THRUSTER_NAMES),
            "last_input_command": self._last_input_command,
            "last_target_rpm": self._last_target_rpm,
            "last_normalized_command": self._last_normalized_command,
            "last_mcu_command": self._last_mcu_command,
            "enabled": self._enabled,
            "motor_order": self._motor_order,
            "motor_signs": self._motor_signs,
            "imu_sensor_frame_convention": "right_handed_x_back_y_right_z_up",
            "imu_output_frame": self._frame_id,
            "imu_output_frame_convention": "ros_flu_right_handed_x_forward_y_left_z_up",
            "imu_sensor_to_base_quaternion_xyzw": list(self._imu_mounting.quaternion_xyzw),
            "debug_mask": self._debug_mask,
            "debug_echo_decimation": self._debug_echo_decimation,
            "debug_config_sent_frames": self._debug_config_sent_count,
            "debug_config_send_errors": self._debug_config_error_count,
            "send_errors": self._send_error_count,
            "read_errors": self._read_error_count,
            "parser_crc_errors": self._parser.crc_errors,
            "parser_decode_errors": self._parser.decode_errors,
            "parser_dropped_bytes": self._parser.dropped_bytes,
            "thruster_timed_out": self._thruster_timed_out,
            "telemetry_timed_out": self._telemetry_timed_out,
            "mcu_clock": self._mcu_clock_mapper.diagnostics(),
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = HardwareBridgeNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
