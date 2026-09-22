from __future__ import annotations

import math
import struct
import time
from dataclasses import dataclass


HOST_MAGIC = b"\xAC\x54"
DEVICE_MAGIC = b"\xA5\x54"
PROTOCOL_VERSION = 0x01
FRAME_TYPE_COMMAND = 0x01
FRAME_TYPE_STATUS = 0x81
COMMAND_PAYLOAD_FORMAT = "<Bff"
COMMAND_PAYLOAD_SIZE = struct.calcsize(COMMAND_PAYLOAD_FORMAT)
COMMAND_FRAME_SIZE = 8 + COMMAND_PAYLOAD_SIZE + 2
STATUS_PAYLOAD_FORMAT = "<IHIBBIfff"
STATUS_PAYLOAD_SIZE = struct.calcsize(STATUS_PAYLOAD_FORMAT)
STATUS_FRAME_SIZE = 8 + STATUS_PAYLOAD_SIZE + 2


REJECT_NOT_APPLIED = 1 << 0
REJECT_DISABLED = 1 << 1
REJECT_TIMEOUT = 1 << 2
REJECT_BAD_COMMAND = 1 << 3


@dataclass(frozen=True)
class ThrusterBenchStatus:
    sequence: int
    mcu_time_ms: int
    command_sequence: int
    command_count: int
    enabled: bool
    accepted: bool
    reject_flags: int
    command: float
    applied: float
    rpm: float


def crc16_modbus(data: bytes) -> int:
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


def _validate_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


def build_command_frame(
    *,
    sequence: int,
    enabled: bool,
    command: float,
    limit: float,
) -> bytes:
    command = max(-1.0, min(1.0, _validate_finite(command, "command")))
    limit = max(0.0, min(1.0, _validate_finite(limit, "limit")))
    payload = struct.pack(COMMAND_PAYLOAD_FORMAT, 1 if enabled else 0, command, limit)
    header = HOST_MAGIC + struct.pack(
        "<BBHH",
        PROTOCOL_VERSION,
        FRAME_TYPE_COMMAND,
        int(sequence) & 0xFFFF,
        len(payload),
    )
    body = header[2:] + payload
    return header + payload + struct.pack("<H", crc16_modbus(body))


def parse_status_frame(frame: bytes) -> ThrusterBenchStatus:
    if len(frame) != STATUS_FRAME_SIZE:
        raise ValueError(f"expected {STATUS_FRAME_SIZE} status bytes, got {len(frame)}")
    if frame[:2] != DEVICE_MAGIC:
        raise ValueError("invalid status magic")
    version, frame_type, sequence, payload_len = struct.unpack("<BBHH", frame[2:8])
    if version != PROTOCOL_VERSION:
        raise ValueError(f"unsupported status protocol version {version}")
    if frame_type != FRAME_TYPE_STATUS:
        raise ValueError(f"unexpected status frame type {frame_type}")
    if payload_len != STATUS_PAYLOAD_SIZE:
        raise ValueError(f"unexpected status payload size {payload_len}")
    expected_crc = struct.unpack("<H", frame[-2:])[0]
    actual_crc = crc16_modbus(frame[2:-2])
    if expected_crc != actual_crc:
        raise ValueError(f"status CRC mismatch: expected 0x{expected_crc:04x}, got 0x{actual_crc:04x}")

    unpacked = struct.unpack(STATUS_PAYLOAD_FORMAT, frame[8:-2])
    return ThrusterBenchStatus(
        sequence=int(sequence),
        mcu_time_ms=int(unpacked[0]),
        command_sequence=int(unpacked[1]),
        command_count=int(unpacked[2]),
        enabled=bool(unpacked[3]),
        accepted=bool(unpacked[4]),
        reject_flags=int(unpacked[5]),
        command=float(unpacked[6]),
        applied=float(unpacked[7]),
        rpm=float(unpacked[8]),
    )


class StatusStreamParser:
    def __init__(self) -> None:
        self._buffer = bytearray()
        self.crc_errors = 0
        self.decode_errors = 0
        self.dropped_bytes = 0

    def feed(self, data: bytes) -> list[ThrusterBenchStatus]:
        if data:
            self._buffer.extend(data)
        packets: list[ThrusterBenchStatus] = []
        while self._buffer:
            start = self._buffer.find(DEVICE_MAGIC)
            if start < 0:
                self.dropped_bytes += len(self._buffer)
                self._buffer.clear()
                break
            if start > 0:
                self.dropped_bytes += start
                del self._buffer[:start]
            if len(self._buffer) < STATUS_FRAME_SIZE:
                break
            frame = bytes(self._buffer[:STATUS_FRAME_SIZE])
            try:
                packets.append(parse_status_frame(frame))
                del self._buffer[:STATUS_FRAME_SIZE]
            except ValueError as exc:
                if "CRC" in str(exc):
                    self.crc_errors += 1
                else:
                    self.decode_errors += 1
                del self._buffer[0]
        return packets


class SerialThrusterBenchDriver:
    def __init__(
        self,
        *,
        port: str,
        baudrate: int = 115200,
        timeout: float = 0.02,
        write_timeout: float = 0.2,
    ) -> None:
        import serial

        self._serial = serial.Serial(
            port=port,
            baudrate=int(baudrate),
            timeout=float(timeout),
            write_timeout=float(write_timeout),
        )
        self._sequence = 0
        self._parser = StatusStreamParser()
        self.last_status: ThrusterBenchStatus | None = None

    def close(self) -> None:
        self._serial.close()

    def command(self, command: float, *, enabled: bool = True, limit: float = 0.4) -> int:
        self._sequence = (self._sequence + 1) & 0xFFFF
        frame = build_command_frame(
            sequence=self._sequence,
            enabled=enabled,
            command=command,
            limit=limit,
        )
        self._serial.write(frame)
        self._serial.flush()
        return self._sequence

    def stop(self, *, repeat: int = 5, interval_sec: float = 0.02) -> None:
        for _ in range(max(1, int(repeat))):
            self.command(0.0, enabled=False, limit=0.0)
            self.read_status(timeout_sec=interval_sec)
            time.sleep(max(0.0, float(interval_sec)))

    def read_status(self, *, timeout_sec: float = 0.0) -> ThrusterBenchStatus | None:
        deadline = time.monotonic() + max(0.0, float(timeout_sec))
        while True:
            waiting = getattr(self._serial, "in_waiting", 0)
            data = self._serial.read(waiting or 1)
            latest: ThrusterBenchStatus | None = None
            for packet in self._parser.feed(data):
                self.last_status = packet
                latest = packet
            if latest is not None:
                return latest
            if time.monotonic() >= deadline:
                return None
