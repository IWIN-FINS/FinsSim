import pytest

from teleop.mappings import GamepadMapping, apply_deadband, axes_to_wrench, trigger_value


def _command(*, buttons):
    return axes_to_wrench(
        [0.5, -1.0, -1.0, 1.0, 0.0, 1.0],
        buttons,
        mapping=GamepadMapping(),
        deadband=0.0,
        max_surge_force_n=6.0,
        max_sway_force_n=4.0,
        max_heave_force_n=5.0,
        max_yaw_moment_nm=0.8,
        precision_surge_force_n=2.0,
        precision_sway_force_n=1.5,
        precision_heave_force_n=1.5,
        precision_yaw_moment_nm=0.25,
        boost_surge_force_n=10.0,
        boost_sway_force_n=7.0,
        boost_heave_force_n=8.0,
        boost_yaw_moment_nm=1.2,
    )


def test_deadband_scales_remaining_range():
    assert apply_deadband(0.05, 0.1) == 0.0
    assert apply_deadband(1.0, 0.1) == 1.0
    assert apply_deadband(-1.0, 0.1) == -1.0
    assert apply_deadband(0.55, 0.1) == pytest.approx(0.5)


def test_trigger_value_uses_configured_rest_and_pressed_values():
    assert trigger_value([-1.0], 0, rest_value=-1.0, pressed_value=1.0) == 0.0
    assert trigger_value([0.0], 0, rest_value=-1.0, pressed_value=1.0) == 0.5
    assert trigger_value([1.0], 0, rest_value=-1.0, pressed_value=1.0) == 1.0
    assert trigger_value([], 0, rest_value=-1.0, pressed_value=1.0) == 0.0


def test_precision_profile_uses_rb_and_proportional_rt_heave():
    buttons = [0] * 8
    buttons[5] = 1  # RB precision
    command = _command(buttons=buttons)

    assert command.control_profile == "precision"
    assert command.precision_mode is True
    assert command.boost_mode is False
    assert command.surge_force_n == 2.0
    assert command.sway_force_n == 0.75
    assert command.heave_force_n == 1.5
    assert command.yaw_moment_nm == 0.25


def test_boost_profile_uses_x_and_preserves_right_stick_yaw():
    buttons = [0] * 8
    buttons[2] = 1  # X / recommended M1 mapping
    command = _command(buttons=buttons)

    assert command.control_profile == "boost"
    assert command.precision_mode is False
    assert command.boost_mode is True
    assert command.surge_force_n == 10.0
    assert command.sway_force_n == 3.5
    assert command.heave_force_n == 8.0
    assert command.yaw_moment_nm == 1.2


def test_precision_takes_priority_when_rb_and_boost_are_both_pressed():
    buttons = [0] * 8
    buttons[2] = 1
    buttons[5] = 1
    command = _command(buttons=buttons)

    assert command.control_profile == "precision"
    assert command.precision_mode is True
    assert command.boost_mode is True
    assert command.surge_force_n == 2.0
