from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


def apply_deadband(value: float, deadband: float) -> float:
    value = max(-1.0, min(1.0, float(value)))
    deadband = max(0.0, min(0.95, abs(float(deadband))))
    if abs(value) <= deadband:
        return 0.0
    scaled = (abs(value) - deadband) / (1.0 - deadband)
    return scaled if value > 0.0 else -scaled


def axis_value(axes: Sequence[float], index: int, default: float = 0.0) -> float:
    if index < 0 or index >= len(axes):
        return float(default)
    return max(-1.0, min(1.0, float(axes[index])))


def trigger_value(
    axes: Sequence[float],
    index: int,
    *,
    rest_value: float,
    pressed_value: float,
) -> float:
    """Normalize one pygame trigger axis to [0, 1]."""

    if index < 0 or index >= len(axes):
        return 0.0
    raw = float(axes[index])
    denominator = float(pressed_value) - float(rest_value)
    if abs(denominator) <= 1e-6:
        raise ValueError("trigger_pressed_value must differ from trigger_rest_value")
    return max(0.0, min(1.0, (raw - float(rest_value)) / denominator))


def button_value(buttons: Sequence[int], index: int) -> bool:
    return 0 <= index < len(buttons) and int(buttons[index]) != 0


@dataclass(frozen=True)
class GamepadMapping:
    left_x_axis: int = 0
    left_y_axis: int = 1
    right_x_axis: int = 3
    left_trigger_axis: int = 2
    right_trigger_axis: int = 5
    trigger_rest_value: float = -1.0
    trigger_pressed_value: float = 1.0
    mode_switch_button: int = 6
    start_button: int = 7
    emergency_stop_button: int = 1
    precision_button: int = 5
    boost_button: int = 2
    surge_axis_sign: float = -1.0
    sway_axis_sign: float = 1.0
    yaw_axis_sign: float = 1.0


@dataclass(frozen=True)
class TeleopWrenchCommand:
    surge_force_n: float
    sway_force_n: float
    heave_force_n: float
    yaw_moment_nm: float
    surge_force_limit_n: float
    sway_force_limit_n: float
    heave_force_limit_n: float
    yaw_moment_limit_nm: float
    control_profile: str
    precision_mode: bool
    boost_mode: bool


def axes_to_wrench(
    axes: Sequence[float],
    buttons: Sequence[int],
    *,
    mapping: GamepadMapping,
    deadband: float,
    max_surge_force_n: float,
    max_sway_force_n: float,
    max_heave_force_n: float,
    max_yaw_moment_nm: float,
    precision_surge_force_n: float,
    precision_sway_force_n: float,
    precision_heave_force_n: float,
    precision_yaw_moment_nm: float,
    boost_surge_force_n: float,
    boost_sway_force_n: float,
    boost_heave_force_n: float,
    boost_yaw_moment_nm: float,
) -> TeleopWrenchCommand:
    precision_mode = button_value(buttons, mapping.precision_button)
    boost_mode = button_value(buttons, mapping.boost_button)
    if precision_mode:
        control_profile = "precision"
        surge_limit = abs(float(precision_surge_force_n))
        sway_limit = abs(float(precision_sway_force_n))
        heave_limit = abs(float(precision_heave_force_n))
        yaw_limit = abs(float(precision_yaw_moment_nm))
    elif boost_mode:
        control_profile = "boost"
        surge_limit = abs(float(boost_surge_force_n))
        sway_limit = abs(float(boost_sway_force_n))
        heave_limit = abs(float(boost_heave_force_n))
        yaw_limit = abs(float(boost_yaw_moment_nm))
    else:
        control_profile = "cruise"
        surge_limit = abs(float(max_surge_force_n))
        sway_limit = abs(float(max_sway_force_n))
        heave_limit = abs(float(max_heave_force_n))
        yaw_limit = abs(float(max_yaw_moment_nm))

    surge = (
        apply_deadband(axis_value(axes, mapping.left_y_axis), deadband)
        * float(mapping.surge_axis_sign)
        * surge_limit
    )
    sway = (
        apply_deadband(axis_value(axes, mapping.left_x_axis), deadband)
        * float(mapping.sway_axis_sign)
        * sway_limit
    )
    yaw = (
        apply_deadband(axis_value(axes, mapping.right_x_axis), deadband)
        * float(mapping.yaw_axis_sign)
        * yaw_limit
    )
    heave = (
        trigger_value(
            axes,
            mapping.right_trigger_axis,
            rest_value=mapping.trigger_rest_value,
            pressed_value=mapping.trigger_pressed_value,
        )
        - trigger_value(
            axes,
            mapping.left_trigger_axis,
            rest_value=mapping.trigger_rest_value,
            pressed_value=mapping.trigger_pressed_value,
        )
    ) * heave_limit

    return TeleopWrenchCommand(
        surge_force_n=float(surge),
        sway_force_n=float(sway),
        heave_force_n=float(heave),
        yaw_moment_nm=float(yaw),
        surge_force_limit_n=surge_limit,
        sway_force_limit_n=sway_limit,
        heave_force_limit_n=heave_limit,
        yaw_moment_limit_nm=yaw_limit,
        control_profile=control_profile,
        precision_mode=precision_mode,
        boost_mode=boost_mode,
    )
