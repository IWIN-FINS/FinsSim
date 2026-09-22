from hardware_bridge.time_sync import McuClockMapper


def test_mapper_uses_one_time_for_packet_and_tracks_offset() -> None:
    mapper = McuClockMapper(window_size=16)
    first = mapper.update(1000, 10.5)
    second = mapper.update(1020, 10.52)

    assert first.mapped_ros_time_sec == 10.5
    assert abs(second.mapped_ros_time_sec - 10.52) < 1e-9
    assert abs(second.receive_delay_sec) < 1e-9
    assert mapper.diagnostics()["sample_count"] == 2


def test_mapper_unwraps_uint32_millisecond_rollover() -> None:
    mapper = McuClockMapper(window_size=16)
    before = mapper.update(0xFFFFFFF0, 100.0)
    after = mapper.update(0x00000010, 100.032)

    assert after.mcu_time_ms_unwrapped > before.mcu_time_ms_unwrapped
    assert after.mcu_time_ms_unwrapped - before.mcu_time_ms_unwrapped == 32
    assert mapper.diagnostics()["wrap_count"] == 1


def test_mapper_does_not_move_backwards_for_an_old_packet() -> None:
    mapper = McuClockMapper(window_size=16)
    first = mapper.update(1000, 10.5)
    current = mapper.update(1020, 10.52)
    old = mapper.update(1005, 10.53)

    assert old.mcu_time_ms_unwrapped == current.mcu_time_ms_unwrapped
    assert old.mapped_ros_time_sec == current.mapped_ros_time_sec
    assert mapper.diagnostics()["out_of_order_count"] == 1
