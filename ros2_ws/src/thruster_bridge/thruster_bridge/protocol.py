import math
import struct
from collections.abc import Sequence


FRAME_HEAD = 0xAA
FRAME_TAIL = 0xBB
THRUSTER_COUNT = 8
BODY_FORMAT = "<B8f"
BODY_SIZE = struct.calcsize(BODY_FORMAT)
FRAME_SIZE = 1 + BODY_SIZE + 2 + 1


def crc16_modbus(data: bytes) -> int:
    """CRC-16/MODBUS with polynomial 0xA001 and initial value 0xFFFF."""
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


def build_thruster_frame(throttles: Sequence[float], *, enabled: bool = True) -> bytes:
    """Build the 37-byte frame decoded by FineSUB Services/V5Streamer/Streamer.hpp."""
    if len(throttles) != THRUSTER_COUNT:
        raise ValueError(f"expected {THRUSTER_COUNT} thruster values, got {len(throttles)}")

    clean_values = []
    for value in throttles:
        float_value = float(value)
        if not math.isfinite(float_value):
            raise ValueError(f"thruster values must be finite, got {value!r}")
        clean_values.append(float_value)

    body = struct.pack(BODY_FORMAT, 1 if enabled else 0, *clean_values)
    crc = crc16_modbus(body)
    return bytes([FRAME_HEAD]) + body + struct.pack("<H", crc) + bytes([FRAME_TAIL])
