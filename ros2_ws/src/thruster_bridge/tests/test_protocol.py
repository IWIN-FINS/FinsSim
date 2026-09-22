import struct

import pytest

from thruster_bridge.protocol import BODY_SIZE, FRAME_SIZE, build_thruster_frame, crc16_modbus


def test_crc16_modbus_known_vector():
    assert crc16_modbus(b"123456789") == 0x4B37


def test_build_thruster_frame_matches_streamer_layout():
    values = [0.0, 0.125, -0.25, 0.5, -0.75, 1.0, -1.0, 0.333]
    frame = build_thruster_frame(values, enabled=True)

    assert len(frame) == FRAME_SIZE == 37
    assert BODY_SIZE == 33
    assert frame[0] == 0xAA
    assert frame[-1] == 0xBB

    body = frame[1:34]
    crc_low_high = frame[34:36]
    assert struct.unpack("<H", crc_low_high)[0] == crc16_modbus(body)

    enabled, *decoded = struct.unpack("<B8f", body)
    assert enabled == 1
    assert decoded == pytest.approx(values)


def test_build_thruster_frame_rejects_wrong_count():
    with pytest.raises(ValueError):
        build_thruster_frame([0.0] * 7)
