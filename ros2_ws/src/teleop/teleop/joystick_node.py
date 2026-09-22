from __future__ import annotations

import json
import time
from typing import Any, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import WrenchStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Bool, Float32MultiArray, String

from .mappings import GamepadMapping, axes_to_wrench, axis_value, button_value, trigger_value
from .thrust_allocator import (
    THRUSTER_COUNT,
    ThrustAllocator,
    make_direct_thruster_values,
)


def _float_list_param(node: Node, name: str, default: Sequence[float], expected_len: int | None = None) -> list[float]:
    values = node.declare_parameter(name, list(default)).value
    if isinstance(values, str):
        values = [float(item.strip()) for item in values.split(",") if item.strip()]
    result = [float(v) for v in values]
    if expected_len is not None and len(result) != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {len(result)}")
    if not np.all(np.isfinite(np.asarray(result, dtype=np.float32))):
        raise ValueError(f"{name} must contain finite values")
    return result


def _int_param(node: Node, name: str, default: int) -> int:
    return int(node.declare_parameter(name, int(default)).value)


def _bool_param(node: Node, name: str, default: bool) -> bool:
    return bool(node.declare_parameter(name, bool(default)).value)


def _str_param(node: Node, name: str, default: str) -> str:
    return str(node.declare_parameter(name, default).value).strip()


def _normalize_mode(value: str) -> str:
    mode = str(value).strip().lower().replace("-", "_")
    if mode not in {"wrench", "direct"}:
        raise ValueError(f"mode must be wrench or direct, got {value!r}")
    return mode


class PygameJoystick:
    def __init__(self, index: int) -> None:
        import pygame

        self._pygame = pygame
        self._index = int(index)
        self._joystick: Any | None = None
        pygame.init()
        pygame.joystick.init()

    @property
    def name(self) -> str:
        if self._joystick is None:
            return "<disconnected>"
        return str(self._joystick.get_name())

    def read(self) -> tuple[list[float], list[int], bool, str]:
        self._pygame.event.pump()
        count = self._pygame.joystick.get_count()
        if count <= self._index:
            self._joystick = None
            return [], [], False, f"joystick index {self._index} not found; count={count}"
        if self._joystick is None:
            self._joystick = self._pygame.joystick.Joystick(self._index)
            self._joystick.init()
        axes = [float(self._joystick.get_axis(i)) for i in range(self._joystick.get_numaxes())]
        buttons = [int(self._joystick.get_button(i)) for i in range(self._joystick.get_numbuttons())]
        return axes, buttons, True, "ok"

    def close(self) -> None:
        if self._joystick is not None:
            self._joystick.quit()
            self._joystick = None
        self._pygame.joystick.quit()
        self._pygame.quit()


class JoystickTeleopNode(Node):
    def __init__(self) -> None:
        super().__init__("finsrov_joystick_teleop")

        self._mode = _normalize_mode(_str_param(self, "mode", "wrench"))
        self._rate_hz = float(self.declare_parameter("rate_hz", 30.0).value)
        if self._rate_hz <= 0.0:
            raise ValueError("rate_hz must be > 0")
        self._deadband = float(self.declare_parameter("deadband", 0.08).value)
        self._publish_zero_when_disabled = _bool_param(self, "publish_zero_when_disabled", True)
        self._start_enabled = _bool_param(self, "start_enabled", False)
        self._joystick_index = _int_param(self, "joystick_index", 0)
        self._frame_id = _str_param(self, "frame_id", "finsrov_base_link")

        self._thruster_topic = _str_param(self, "thruster_topic", "/finsrov/thrusters_out")
        self._enabled_topic = _str_param(self, "enabled_topic", "/finsrov/teleop/enabled")
        self._wrench_topic = _str_param(self, "wrench_topic", "/finsrov/teleop/body_wrench_cmd")
        self._status_topic = _str_param(self, "status_topic", "/finsrov/teleop/status")

        self._max_surge_force_n = float(self.declare_parameter("max_surge_force_n", 6.0).value)
        self._max_sway_force_n = float(self.declare_parameter("max_sway_force_n", 4.0).value)
        self._max_heave_force_n = float(self.declare_parameter("max_heave_force_n", 5.0).value)
        self._max_yaw_moment_nm = float(self.declare_parameter("max_yaw_moment_nm", 0.8).value)
        self._precision_surge_force_n = float(self.declare_parameter("precision_surge_force_n", 2.0).value)
        self._precision_sway_force_n = float(self.declare_parameter("precision_sway_force_n", 1.5).value)
        self._precision_heave_force_n = float(self.declare_parameter("precision_heave_force_n", 1.5).value)
        self._precision_yaw_moment_nm = float(self.declare_parameter("precision_yaw_moment_nm", 0.25).value)
        self._boost_surge_force_n = float(self.declare_parameter("boost_surge_force_n", 10.0).value)
        self._boost_sway_force_n = float(self.declare_parameter("boost_sway_force_n", 7.0).value)
        self._boost_heave_force_n = float(self.declare_parameter("boost_heave_force_n", 8.0).value)
        self._boost_yaw_moment_nm = float(self.declare_parameter("boost_yaw_moment_nm", 1.2).value)
        self._direct_yaw_force_n = float(self.declare_parameter("direct_yaw_force_n", 0.8).value)

        self._mapping = GamepadMapping(
            left_x_axis=_int_param(self, "left_x_axis", 0),
            left_y_axis=_int_param(self, "left_y_axis", 1),
            right_x_axis=_int_param(self, "right_x_axis", 3),
            left_trigger_axis=_int_param(self, "left_trigger_axis", 2),
            right_trigger_axis=_int_param(self, "right_trigger_axis", 5),
            trigger_rest_value=float(self.declare_parameter("trigger_rest_value", -1.0).value),
            trigger_pressed_value=float(self.declare_parameter("trigger_pressed_value", 1.0).value),
            mode_switch_button=_int_param(self, "mode_switch_button", 6),
            start_button=_int_param(self, "start_button", 7),
            emergency_stop_button=_int_param(self, "emergency_stop_button", 1),
            precision_button=_int_param(self, "precision_button", 5),
            boost_button=_int_param(self, "boost_button", 2),
            surge_axis_sign=float(self.declare_parameter("surge_axis_sign", -1.0).value),
            sway_axis_sign=float(self.declare_parameter("sway_axis_sign", 1.0).value),
            yaw_axis_sign=float(self.declare_parameter("yaw_axis_sign", 1.0).value),
        )

        self._physical_wrench_limits_policy = _float_list_param(
            self,
            "physical_wrench_limits_policy",
            [18.586585, 18.068824, 25.636077, 3.791353, 2.535180, 6.947713],
            expected_len=6,
        )
        self._force_limits_positive_n = _float_list_param(
            self,
            "force_limits_positive_n",
            [8.4749, 7.3809, 7.3809, 8.4749, 7.3809, 8.4749, 7.3809, 8.4749],
            expected_len=THRUSTER_COUNT,
        )
        self._force_limits_negative_n = _float_list_param(
            self,
            "force_limits_negative_n",
            [7.9750, 5.7618, 5.7618, 7.9750, 5.7618, 7.9750, 5.7618, 7.9750],
            expected_len=THRUSTER_COUNT,
        )
        self._allocator = ThrustAllocator(
            physical_wrench_limits_policy=self._physical_wrench_limits_policy,
            force_limits_positive_n=self._force_limits_positive_n,
            force_limits_negative_n=self._force_limits_negative_n,
        )

        existing_publishers = self.count_publishers(self._thruster_topic)
        self._thruster_pub = self.create_publisher(Float32MultiArray, self._thruster_topic, 10)
        self._enabled_pub = self.create_publisher(Bool, self._enabled_topic, 10)
        self._wrench_pub = self.create_publisher(WrenchStamped, self._wrench_topic, 10)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)

        self._joystick = PygameJoystick(self._joystick_index)
        self._enabled = self._start_enabled
        self._last_mode_switch_pressed = False
        self._last_start_pressed = False
        self._last_emergency_pressed = False
        self._last_log_sec = 0.0
        self._last_thrusters = [0.0] * THRUSTER_COUNT
        self._last_status = "starting"
        self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

        if existing_publishers > 0:
            self.get_logger().warn(
                f"{self._thruster_topic} already has {existing_publishers} publisher(s); "
                "stop motion_controller/test senders before enabling teleop"
            )
        self.get_logger().warn(
            "joystick teleop started: "
            f"mode={self._mode}, joystick_index={self._joystick_index}, rate={self._rate_hz:.1f}Hz, "
            f"thruster_topic={self._thruster_topic}, start_enabled={self._enabled}"
        )

    def _on_timer(self) -> None:
        try:
            axes, buttons, connected, read_status = self._joystick.read()
        except Exception as exc:
            axes, buttons, connected, read_status = [], [], False, f"joystick read error: {exc}"

        start_pressed = button_value(buttons, self._mapping.start_button)
        mode_switch_pressed = button_value(buttons, self._mapping.mode_switch_button)
        emergency_pressed = button_value(buttons, self._mapping.emergency_stop_button)
        mode_switched = False
        if mode_switch_pressed and not self._last_mode_switch_pressed and connected:
            self._mode = "direct" if self._mode == "wrench" else "wrench"
            # A mode change replaces the allocation semantics. Zero the current
            # output frame so a held stick cannot carry a stale command across it.
            self._publish_enabled(False)
            self._publish_thrusters([0.0] * THRUSTER_COUNT)
            self.get_logger().warn(f"teleop output mode switched to {self._mode}")
            mode_switched = True
        if start_pressed and not self._last_start_pressed and connected:
            self._enabled = not self._enabled
        if emergency_pressed and not self._last_emergency_pressed:
            self._enabled = False
            self._publish_thrusters([0.0] * THRUSTER_COUNT)
        self._last_mode_switch_pressed = mode_switch_pressed
        self._last_start_pressed = start_pressed
        self._last_emergency_pressed = emergency_pressed

        command = axes_to_wrench(
            axes,
            buttons,
            mapping=self._mapping,
            deadband=self._deadband,
            max_surge_force_n=self._max_surge_force_n,
            max_sway_force_n=self._max_sway_force_n,
            max_heave_force_n=self._max_heave_force_n,
            max_yaw_moment_nm=self._max_yaw_moment_nm,
            precision_surge_force_n=self._precision_surge_force_n,
            precision_sway_force_n=self._precision_sway_force_n,
            precision_heave_force_n=self._precision_heave_force_n,
            precision_yaw_moment_nm=self._precision_yaw_moment_nm,
            boost_surge_force_n=self._boost_surge_force_n,
            boost_sway_force_n=self._boost_sway_force_n,
            boost_heave_force_n=self._boost_heave_force_n,
            boost_yaw_moment_nm=self._boost_yaw_moment_nm,
        )
        active = bool(
            connected
            and self._enabled
            and not mode_switched
        )

        allocation = None
        if active:
            if self._mode == "direct":
                yaw_scale = command.yaw_moment_limit_nm / max(abs(self._max_yaw_moment_nm), 1e-6)
                values = make_direct_thruster_values(
                    surge=axis_value(axes, self._mapping.left_y_axis) * self._mapping.surge_axis_sign,
                    heave=(
                        trigger_value(
                            axes,
                            self._mapping.right_trigger_axis,
                            rest_value=self._mapping.trigger_rest_value,
                            pressed_value=self._mapping.trigger_pressed_value,
                        )
                        - trigger_value(
                            axes,
                            self._mapping.left_trigger_axis,
                            rest_value=self._mapping.trigger_rest_value,
                            pressed_value=self._mapping.trigger_pressed_value,
                        )
                    ),
                    yaw=axis_value(axes, self._mapping.right_x_axis) * self._mapping.yaw_axis_sign,
                    max_surge_force_n=command.surge_force_limit_n,
                    max_heave_force_n=command.heave_force_limit_n,
                    max_yaw_force_n=abs(self._direct_yaw_force_n) * yaw_scale,
                    force_limits_positive_n=self._force_limits_positive_n,
                    force_limits_negative_n=self._force_limits_negative_n,
                )
                clamped = False
            else:
                allocation = self._allocator.allocate(
                    command.surge_force_n,
                    command.sway_force_n,
                    command.heave_force_n,
                    command.yaw_moment_nm,
                )
                values = allocation.values
                clamped = allocation.clamped
        else:
            values = [0.0] * THRUSTER_COUNT
            clamped = False

        self._publish_enabled(active)
        self._publish_wrench(command, active)
        if active or self._publish_zero_when_disabled:
            self._publish_thrusters(values)

        status = {
            "mode": self._mode,
            "enabled_toggle": self._enabled,
            "active": active,
            "connected": connected,
            "joystick": self._joystick.name,
            "read_status": read_status,
            "control_profile": command.control_profile,
            "precision_mode": command.precision_mode,
            "boost_mode": command.boost_mode,
            "axes": [round(float(v), 4) for v in axes],
            "buttons": [int(v) for v in buttons],
            "wrench": {
                "surge_force_n": command.surge_force_n if active else 0.0,
                "sway_force_n": command.sway_force_n if active else 0.0,
                "heave_force_n": command.heave_force_n if active else 0.0,
                "yaw_moment_nm": command.yaw_moment_nm if active else 0.0,
            },
            "thrusters_force_n": [round(float(v), 4) for v in values],
            "clamped": clamped,
        }
        if allocation is not None:
            status["controller_body_wrench"] = {
                "order": ["Fx", "Fy", "Fz", "Mx", "My", "Mz"],
                "requested": [round(value, 5) for value in allocation.requested_wrench_body],
                "achieved": [round(value, 5) for value in allocation.achieved_wrench_body],
                "residual": [round(value, 5) for value in allocation.residual_wrench_body],
                "saturated_thruster_indices": allocation.saturated_thruster_indices,
            }
        self._last_status = json.dumps(status, ensure_ascii=False)
        msg = String()
        msg.data = self._last_status
        self._status_pub.publish(msg)

        now = time.monotonic()
        if not connected and now - self._last_log_sec > 2.0:
            self.get_logger().warn(read_status)
            self._last_log_sec = now

    def _publish_enabled(self, active: bool) -> None:
        msg = Bool()
        msg.data = bool(active)
        self._enabled_pub.publish(msg)

    def _publish_wrench(self, command, active: bool) -> None:
        msg = WrenchStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        if active:
            msg.wrench.force.x = float(command.surge_force_n)
            msg.wrench.force.y = float(command.sway_force_n)
            msg.wrench.force.z = float(command.heave_force_n)
            msg.wrench.torque.z = float(command.yaw_moment_nm)
        self._wrench_pub.publish(msg)

    def _publish_thrusters(self, values: Sequence[float]) -> None:
        clean = [float(v) if np.isfinite(float(v)) else 0.0 for v in list(values)[:THRUSTER_COUNT]]
        if len(clean) < THRUSTER_COUNT:
            clean.extend([0.0] * (THRUSTER_COUNT - len(clean)))
        msg = Float32MultiArray()
        msg.data = clean
        self._thruster_pub.publish(msg)
        self._last_thrusters = clean

    def publish_zero_frames(self, count: int = 3) -> None:
        for _ in range(max(1, int(count))):
            self._publish_enabled(False)
            self._publish_thrusters([0.0] * THRUSTER_COUNT)
            time.sleep(0.03)

    def destroy_node(self) -> bool:
        try:
            self._joystick.close()
        finally:
            return super().destroy_node()


def main() -> None:
    rclpy.init()
    node = JoystickTeleopNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        try:
            node.publish_zero_frames(3)
        finally:
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()


if __name__ == "__main__":
    main()
