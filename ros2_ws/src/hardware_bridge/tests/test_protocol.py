import struct

import pytest

from hardware_bridge.protocol import (
    COMMAND_FRAME_SIZE,
    DEBUG_CONFIG_FRAME_SIZE,
    DEBUG_MASK_THRUSTER_ECHO,
    HOST_FRAME_MAGIC,
    THRUSTER_ECHO_FRAME_SIZE,
    TELEMETRY_FRAME_SIZE,
    TelemetryPacket,
    TelemetryStreamParser,
    ThrusterCommandEchoPacket,
    build_debug_config_frame,
    build_telemetry_frame,
    build_thruster_command_echo_frame,
    build_thruster_command_frame,
    crc16_modbus,
    parse_telemetry_frame,
)


def test_crc16_modbus_known_vector():
    assert crc16_modbus(b"123456789") == 0x4B37


def test_thruster_command_frame_layout():
    frame = build_thruster_command_frame([0.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7], enabled=True)
    assert len(frame) == COMMAND_FRAME_SIZE == 37
    assert frame[0] == 0xAA
    assert frame[-1] == 0xBB
    body = frame[1:34]
    assert struct.unpack("<H", frame[34:36])[0] == crc16_modbus(body)
    enabled, *values = struct.unpack("<B8f", body)
    assert enabled == 1
    assert values == pytest.approx([0.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7])


def test_debug_config_frame_layout():
    frame = build_debug_config_frame(DEBUG_MASK_THRUSTER_ECHO, 10, sequence=3)
    assert len(frame) == DEBUG_CONFIG_FRAME_SIZE
    assert frame[:2] == HOST_FRAME_MAGIC
    body = frame[2:-2]
    assert struct.unpack("<H", frame[-2:])[0] == crc16_modbus(body)
    msg_type, version, payload_len, sequence = struct.unpack("<BBHH", body[:6])
    debug_mask, decimation, reserved = struct.unpack("<IHH", body[6:])
    assert msg_type == 1
    assert version == 1
    assert payload_len == 8
    assert sequence == 3
    assert debug_mask == DEBUG_MASK_THRUSTER_ECHO
    assert decimation == 10
    assert reserved == 0


def test_telemetry_frame_round_trip():
    packet = TelemetryPacket(
        sequence=42,
        mcu_time_ms=123456,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.1, 0.2, 0.3),
        linear_acceleration_xyz=(1.0, 2.0, 3.0),
        depth_m=1.25,
        pressure_pa=101325.0,
        status_flags=0xA5,
        motor_rpm=(10.0, -20.0, 30.0, -40.0, 50.0, -60.0, 70.0, -80.0),
    )
    frame = build_telemetry_frame(packet)
    assert len(frame) == TELEMETRY_FRAME_SIZE
    decoded = parse_telemetry_frame(frame)
    assert decoded.sequence == packet.sequence
    assert decoded.mcu_time_ms == packet.mcu_time_ms
    assert decoded.quat_wxyz == pytest.approx(packet.quat_wxyz)
    assert decoded.angular_velocity_xyz == pytest.approx(packet.angular_velocity_xyz)
    assert decoded.linear_acceleration_xyz == pytest.approx(packet.linear_acceleration_xyz)
    assert decoded.depth_m == pytest.approx(packet.depth_m)
    assert decoded.pressure_pa == pytest.approx(packet.pressure_pa)
    assert decoded.status_flags == packet.status_flags
    assert decoded.motor_rpm == pytest.approx(packet.motor_rpm)


def test_legacy_telemetry_frame_without_motor_rpm_is_supported():
    packet = TelemetryPacket(
        sequence=43,
        mcu_time_ms=123457,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.1, 0.2, 0.3),
        linear_acceleration_xyz=(1.0, 2.0, 3.0),
        depth_m=1.25,
        pressure_pa=101325.0,
        status_flags=0xA5,
    )
    payload = struct.pack(
        "<12fI",
        *packet.quat_wxyz,
        *packet.angular_velocity_xyz,
        *packet.linear_acceleration_xyz,
        packet.depth_m,
        packet.pressure_pa,
        packet.status_flags,
    )
    body = struct.pack("<BBHHI", 1, 1, len(payload), packet.sequence, packet.mcu_time_ms) + payload
    frame = b"\xA5\x5A" + body + struct.pack("<H", crc16_modbus(body))

    decoded = parse_telemetry_frame(frame)

    assert isinstance(decoded, TelemetryPacket)
    assert decoded.sequence == packet.sequence
    assert decoded.depth_m == pytest.approx(packet.depth_m)
    assert decoded.motor_rpm == pytest.approx((0.0,) * 8)


def test_thruster_command_echo_frame_round_trip():
    packet = ThrusterCommandEchoPacket(
        sequence=9,
        mcu_time_ms=100,
        command_count=7,
        receive_time_ms=99,
        command_crc=0x1234,
        enabled=True,
        accepted=True,
        reject_flags=0x01,
        received_thrust=(0.0, 0.1, -0.2, 0.3, -0.4, 0.5, -0.6, 0.7),
        applied_thrust=(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0),
        debug_mask=DEBUG_MASK_THRUSTER_ECHO,
        echo_decimation=10,
    )
    frame = build_thruster_command_echo_frame(packet)
    assert len(frame) == THRUSTER_ECHO_FRAME_SIZE
    decoded = parse_telemetry_frame(frame)
    assert isinstance(decoded, ThrusterCommandEchoPacket)
    assert decoded.sequence == packet.sequence
    assert decoded.command_count == packet.command_count
    assert decoded.command_crc == packet.command_crc
    assert decoded.enabled == packet.enabled
    assert decoded.accepted == packet.accepted
    assert decoded.reject_flags == packet.reject_flags
    assert decoded.received_thrust == pytest.approx(packet.received_thrust)
    assert decoded.applied_thrust == pytest.approx(packet.applied_thrust)
    assert decoded.debug_mask == packet.debug_mask
    assert decoded.echo_decimation == packet.echo_decimation


def test_telemetry_stream_parser_resyncs_after_noise_and_bad_crc():
    good = build_telemetry_frame(
        TelemetryPacket(
            sequence=1,
            mcu_time_ms=2,
            quat_wxyz=(1.0, 0.0, 0.0, 0.0),
            angular_velocity_xyz=(0.0, 0.0, 0.0),
            linear_acceleration_xyz=(0.0, 0.0, 9.8),
            depth_m=0.4,
            pressure_pa=1000.0,
            status_flags=0,
        )
    )
    bad = bytearray(good)
    bad[-3] ^= 0xFF
    parser = TelemetryStreamParser()
    packets = parser.feed(b"noise" + bytes(bad) + good[:10])
    assert packets == []
    packets = parser.feed(good[10:])
    assert len(packets) == 1
    assert packets[0].sequence == 1
    assert parser.dropped_bytes == 5
    assert parser.crc_errors == 1


def test_telemetry_stream_parser_keeps_split_magic_prefix():
    packet = TelemetryPacket(
        sequence=7,
        mcu_time_ms=8,
        quat_wxyz=(1.0, 0.0, 0.0, 0.0),
        angular_velocity_xyz=(0.0, 0.0, 0.0),
        linear_acceleration_xyz=(0.0, 0.0, 9.8),
        depth_m=0.5,
        pressure_pa=1000.0,
        status_flags=0,
    )
    frame = build_telemetry_frame(packet)
    parser = TelemetryStreamParser()

    assert parser.feed(frame[:1]) == []
    packets = parser.feed(frame[1:])

    assert len(packets) == 1
    assert packets[0].sequence == 7
    assert parser.dropped_bytes == 0
