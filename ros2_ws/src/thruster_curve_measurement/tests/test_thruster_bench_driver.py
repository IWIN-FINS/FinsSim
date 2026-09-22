import struct

import pytest

from thruster_curve_measurement.thruster_bench_driver import (
    DEVICE_MAGIC,
    FRAME_TYPE_STATUS,
    PROTOCOL_VERSION,
    STATUS_PAYLOAD_FORMAT,
    build_command_frame,
    crc16_modbus,
    parse_status_frame,
)


def test_build_command_frame_uses_single_thruster_protocol():
    frame = build_command_frame(sequence=7, enabled=True, command=0.25, limit=0.4)

    assert frame[:2] == b"\xAC\x54"
    assert frame[2] == PROTOCOL_VERSION
    assert frame[3] == 1
    assert struct.unpack("<H", frame[4:6])[0] == 7
    assert struct.unpack("<H", frame[6:8])[0] == 9
    enabled, command, limit = struct.unpack("<Bff", frame[8:-2])
    assert enabled == 1
    assert command == pytest.approx(0.25)
    assert limit == pytest.approx(0.4)
    assert struct.unpack("<H", frame[-2:])[0] == crc16_modbus(frame[2:-2])


def test_parse_status_frame_round_trip():
    payload = struct.pack(
        STATUS_PAYLOAD_FORMAT,
        1234,
        7,
        42,
        1,
        1,
        0,
        0.25,
        0.1,
        3200.0,
    )
    header = DEVICE_MAGIC + struct.pack("<BBHH", PROTOCOL_VERSION, FRAME_TYPE_STATUS, 7, len(payload))
    frame = header + payload + struct.pack("<H", crc16_modbus(header[2:] + payload))

    status = parse_status_frame(frame)

    assert status.sequence == 7
    assert status.mcu_time_ms == 1234
    assert status.command_sequence == 7
    assert status.command_count == 42
    assert status.enabled
    assert status.accepted
    assert status.reject_flags == 0
    assert status.command == pytest.approx(0.25)
    assert status.applied == pytest.approx(0.1)
    assert status.rpm == pytest.approx(3200.0)
