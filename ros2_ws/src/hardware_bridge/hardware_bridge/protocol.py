from __future__ import annotations

from dataclasses import dataclass
import math
import struct
from collections.abc import Sequence
from typing import TypeAlias


COMMAND_HEAD = 0xAA
COMMAND_TAIL = 0xBB
THRUSTER_COUNT = 8
COMMAND_BODY_FORMAT = "<B8f"
COMMAND_BODY_SIZE = struct.calcsize(COMMAND_BODY_FORMAT)
COMMAND_FRAME_SIZE = 1 + COMMAND_BODY_SIZE + 2 + 1

TELEMETRY_MAGIC = b"\xA5\x5A"
TELEMETRY_TYPE_IMU_DEPTH_V1 = 0x01
TELEMETRY_TYPE_THRUSTER_ECHO_V1 = 0x02
TELEMETRY_VERSION = 0x01
TELEMETRY_HEADER_FORMAT = "<2sBBHHI"
TELEMETRY_BODY_HEADER_FORMAT = "<BBHHI"
TELEMETRY_PAYLOAD_BASE_FORMAT = "<12fI"
TELEMETRY_PAYLOAD_FORMAT = "<12fI8f"
THRUSTER_ECHO_PAYLOAD_FORMAT = "<IIHBBI8f8fIH"
TELEMETRY_HEADER_SIZE = struct.calcsize(TELEMETRY_HEADER_FORMAT)
TELEMETRY_BODY_HEADER_SIZE = struct.calcsize(TELEMETRY_BODY_HEADER_FORMAT)
TELEMETRY_PAYLOAD_BASE_SIZE = struct.calcsize(TELEMETRY_PAYLOAD_BASE_FORMAT)
TELEMETRY_PAYLOAD_SIZE = struct.calcsize(TELEMETRY_PAYLOAD_FORMAT)
THRUSTER_ECHO_PAYLOAD_SIZE = struct.calcsize(THRUSTER_ECHO_PAYLOAD_FORMAT)
TELEMETRY_FRAME_SIZE = TELEMETRY_HEADER_SIZE + TELEMETRY_PAYLOAD_SIZE + 2
THRUSTER_ECHO_FRAME_SIZE = TELEMETRY_HEADER_SIZE + THRUSTER_ECHO_PAYLOAD_SIZE + 2

HOST_FRAME_MAGIC = b"\xAC\xCA"
HOST_FRAME_TYPE_DEBUG_CONFIG = 0x01
HOST_FRAME_VERSION = 0x01
HOST_FRAME_HEADER_FORMAT = "<2sBBHH"
DEBUG_CONFIG_PAYLOAD_FORMAT = "<IHH"
HOST_FRAME_HEADER_SIZE = struct.calcsize(HOST_FRAME_HEADER_FORMAT)
DEBUG_CONFIG_PAYLOAD_SIZE = struct.calcsize(DEBUG_CONFIG_PAYLOAD_FORMAT)
DEBUG_CONFIG_FRAME_SIZE = HOST_FRAME_HEADER_SIZE + DEBUG_CONFIG_PAYLOAD_SIZE + 2

DEBUG_MASK_THRUSTER_ECHO = 1 << 0


@dataclass(frozen=True)
class TelemetryPacket:
    sequence: int
    mcu_time_ms: int
    quat_wxyz: tuple[float, float, float, float]
    angular_velocity_xyz: tuple[float, float, float]
    linear_acceleration_xyz: tuple[float, float, float]
    depth_m: float
    pressure_pa: float
    status_flags: int
    motor_rpm: tuple[float, float, float, float, float, float, float, float] = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class ThrusterCommandEchoPacket:
    sequence: int
    mcu_time_ms: int
    command_count: int
    receive_time_ms: int
    command_crc: int
    enabled: bool
    accepted: bool
    reject_flags: int
    received_thrust: tuple[float, float, float, float, float, float, float, float]
    applied_thrust: tuple[float, float, float, float, float, float, float, float]
    debug_mask: int
    echo_decimation: int


TelemetryMessage: TypeAlias = TelemetryPacket | ThrusterCommandEchoPacket


def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS, polynomial 0xA001, init 0xFFFF."""
    crc = 0xFFFF
    for byte in data:
        crc ^= byte
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
            crc &= 0xFFFF
    return crc


def build_thruster_command_frame(throttles: Sequence[float], *, enabled: bool = True) -> bytes:
    if len(throttles) != THRUSTER_COUNT:
        raise ValueError(f"expected {THRUSTER_COUNT} thruster values, got {len(throttles)}")
    values: list[float] = []
    for value in throttles:
        value_f = float(value)
        if not math.isfinite(value_f):
            raise ValueError(f"thruster values must be finite, got {value!r}")
        values.append(value_f)
    body = struct.pack(COMMAND_BODY_FORMAT, 1 if enabled else 0, *values)
    crc = crc16_modbus(body)
    return bytes([COMMAND_HEAD]) + body + struct.pack("<H", crc) + bytes([COMMAND_TAIL])


def build_debug_config_frame(debug_mask: int, echo_decimation: int, *, sequence: int = 0) -> bytes:
    mask = int(debug_mask) & 0xFFFFFFFF
    decimation = max(1, min(0xFFFF, int(echo_decimation)))
    payload = struct.pack(DEBUG_CONFIG_PAYLOAD_FORMAT, mask, decimation, 0)
    header = struct.pack(
        HOST_FRAME_HEADER_FORMAT,
        HOST_FRAME_MAGIC,
        HOST_FRAME_TYPE_DEBUG_CONFIG,
        HOST_FRAME_VERSION,
        len(payload),
        int(sequence) & 0xFFFF,
    )
    body = header[2:] + payload
    return header + payload + struct.pack("<H", crc16_modbus(body))


def build_telemetry_payload_frame(
    msg_type: int,
    sequence: int,
    mcu_time_ms: int,
    payload: bytes,
) -> bytes:
    body_header = struct.pack(
        TELEMETRY_BODY_HEADER_FORMAT,
        int(msg_type) & 0xFF,
        TELEMETRY_VERSION,
        len(payload),
        int(sequence) & 0xFFFF,
        int(mcu_time_ms) & 0xFFFFFFFF,
    )
    body = body_header + payload
    return TELEMETRY_MAGIC + body + struct.pack("<H", crc16_modbus(body))


def build_telemetry_frame(packet: TelemetryPacket) -> bytes:
    payload = struct.pack(
        TELEMETRY_PAYLOAD_FORMAT,
        *packet.quat_wxyz,
        *packet.angular_velocity_xyz,
        *packet.linear_acceleration_xyz,
        float(packet.depth_m),
        float(packet.pressure_pa),
        int(packet.status_flags) & 0xFFFFFFFF,
        *packet.motor_rpm,
    )
    return build_telemetry_payload_frame(TELEMETRY_TYPE_IMU_DEPTH_V1, packet.sequence, packet.mcu_time_ms, payload)


def build_thruster_command_echo_frame(packet: ThrusterCommandEchoPacket) -> bytes:
    payload = struct.pack(
        THRUSTER_ECHO_PAYLOAD_FORMAT,
        int(packet.command_count) & 0xFFFFFFFF,
        int(packet.receive_time_ms) & 0xFFFFFFFF,
        int(packet.command_crc) & 0xFFFF,
        1 if packet.enabled else 0,
        1 if packet.accepted else 0,
        int(packet.reject_flags) & 0xFFFFFFFF,
        *packet.received_thrust,
        *packet.applied_thrust,
        int(packet.debug_mask) & 0xFFFFFFFF,
        int(packet.echo_decimation) & 0xFFFF,
    )
    return build_telemetry_payload_frame(
        TELEMETRY_TYPE_THRUSTER_ECHO_V1,
        packet.sequence,
        packet.mcu_time_ms,
        payload,
    )


def parse_telemetry_frame(frame: bytes) -> TelemetryMessage:
    if len(frame) < TELEMETRY_HEADER_SIZE + 2:
        raise ValueError("telemetry frame is too short")
    magic, msg_type, version, payload_len, sequence, mcu_time_ms = struct.unpack(
        TELEMETRY_HEADER_FORMAT,
        frame[:TELEMETRY_HEADER_SIZE],
    )
    if magic != TELEMETRY_MAGIC:
        raise ValueError("invalid telemetry magic")
    if version != TELEMETRY_VERSION:
        raise ValueError(f"unsupported telemetry version {version}")
    expected_size = TELEMETRY_HEADER_SIZE + payload_len + 2
    if len(frame) != expected_size:
        raise ValueError(f"unexpected telemetry frame size {len(frame)}")
    body = frame[2:-2]
    expected_crc = struct.unpack("<H", frame[-2:])[0]
    actual_crc = crc16_modbus(body)
    if actual_crc != expected_crc:
        raise ValueError(f"telemetry CRC mismatch: expected 0x{expected_crc:04x}, got 0x{actual_crc:04x}")

    if msg_type == TELEMETRY_TYPE_THRUSTER_ECHO_V1:
        return _parse_thruster_command_echo_payload(
            frame[TELEMETRY_HEADER_SIZE:-2],
            sequence=int(sequence),
            mcu_time_ms=int(mcu_time_ms),
            payload_len=int(payload_len),
        )

    if msg_type != TELEMETRY_TYPE_IMU_DEPTH_V1:
        raise ValueError(f"unsupported telemetry type {msg_type}")
    if payload_len not in (TELEMETRY_PAYLOAD_BASE_SIZE, TELEMETRY_PAYLOAD_SIZE):
        raise ValueError(f"unexpected telemetry payload size {payload_len}")

    payload = frame[TELEMETRY_HEADER_SIZE:-2]
    unpacked = struct.unpack(
        TELEMETRY_PAYLOAD_FORMAT if payload_len == TELEMETRY_PAYLOAD_SIZE else TELEMETRY_PAYLOAD_BASE_FORMAT,
        payload,
    )
    quat = tuple(float(v) for v in unpacked[0:4])
    angular = tuple(float(v) for v in unpacked[4:7])
    accel = tuple(float(v) for v in unpacked[7:10])
    motor_rpm = tuple(float(v) for v in unpacked[13:21]) if payload_len == TELEMETRY_PAYLOAD_SIZE else (0.0,) * 8
    return TelemetryPacket(
        sequence=int(sequence),
        mcu_time_ms=int(mcu_time_ms),
        quat_wxyz=quat,  # type: ignore[arg-type]
        angular_velocity_xyz=angular,  # type: ignore[arg-type]
        linear_acceleration_xyz=accel,  # type: ignore[arg-type]
        depth_m=float(unpacked[10]),
        pressure_pa=float(unpacked[11]),
        status_flags=int(unpacked[12]),
        motor_rpm=motor_rpm,  # type: ignore[arg-type]
    )


def _parse_thruster_command_echo_payload(
    payload: bytes,
    *,
    sequence: int,
    mcu_time_ms: int,
    payload_len: int,
) -> ThrusterCommandEchoPacket:
    if payload_len != THRUSTER_ECHO_PAYLOAD_SIZE:
        raise ValueError(f"unexpected thruster echo payload size {payload_len}")
    unpacked = struct.unpack(THRUSTER_ECHO_PAYLOAD_FORMAT, payload)
    received = tuple(float(v) for v in unpacked[6:14])
    applied = tuple(float(v) for v in unpacked[14:22])
    return ThrusterCommandEchoPacket(
        sequence=int(sequence),
        mcu_time_ms=int(mcu_time_ms),
        command_count=int(unpacked[0]),
        receive_time_ms=int(unpacked[1]),
        command_crc=int(unpacked[2]),
        enabled=bool(unpacked[3]),
        accepted=bool(unpacked[4]),
        reject_flags=int(unpacked[5]),
        received_thrust=received,  # type: ignore[arg-type]
        applied_thrust=applied,  # type: ignore[arg-type]
        debug_mask=int(unpacked[22]),
        echo_decimation=int(unpacked[23]),
    )


class TelemetryStreamParser:
    def __init__(self, *, max_payload_size: int = 256) -> None:
        self._buffer = bytearray()
        self._max_payload_size = int(max_payload_size)
        self.dropped_bytes = 0
        self.crc_errors = 0
        self.decode_errors = 0

    def feed(self, data: bytes) -> list[TelemetryMessage]:
        if data:
            self._buffer.extend(data)
        packets: list[TelemetryMessage] = []
        while True:
            magic_index = self._buffer.find(TELEMETRY_MAGIC)
            if magic_index < 0:
                if self._buffer.endswith(TELEMETRY_MAGIC[:1]):
                    self.dropped_bytes += max(0, len(self._buffer) - 1)
                    del self._buffer[:-1]
                else:
                    self.dropped_bytes += len(self._buffer)
                    self._buffer.clear()
                break
            if magic_index > 0:
                self.dropped_bytes += magic_index
                del self._buffer[:magic_index]
            if len(self._buffer) < TELEMETRY_HEADER_SIZE:
                break

            payload_len = struct.unpack("<H", self._buffer[4:6])[0]
            if payload_len > self._max_payload_size:
                self.decode_errors += 1
                del self._buffer[0]
                continue
            frame_size = TELEMETRY_HEADER_SIZE + payload_len + 2
            if len(self._buffer) < frame_size:
                break
            frame = bytes(self._buffer[:frame_size])
            del self._buffer[:frame_size]
            try:
                packets.append(parse_telemetry_frame(frame))
            except ValueError as exc:
                if "CRC mismatch" in str(exc):
                    self.crc_errors += 1
                else:
                    self.decode_errors += 1
        return packets
