from thruster_curve_measurement.force_sensor_modbus import (
    convert_signed_to_force,
    decode_force_register,
    decode_signed_u16,
)


def test_decode_signed_u16_positive_and_negative():
    assert decode_signed_u16(0) == 0
    assert decode_signed_u16(32767) == 32767
    assert decode_signed_u16(32768) == -32768
    assert decode_signed_u16(65535) == -1


def test_convert_signed_to_force_matches_current_script_formula():
    assert convert_signed_to_force(100, divisor=100, scale=0.98) == 0.98
    assert convert_signed_to_force(-250, divisor=100, scale=0.98) == -2.45


def test_decode_force_register():
    reading = decode_force_register(65535, divisor=100, scale=0.98)
    assert reading.ok
    assert reading.raw_value == 65535
    assert reading.signed_value == -1
    assert reading.force_n == -0.0098
