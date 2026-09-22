import numpy as np
from geometry_msgs.msg import TwistStamped

from motion_control.controller_node import CONTROL_MODE_ROS_MANUAL, MotionCommand, MotionControllerNode


class FakeLogger:
    def __init__(self):
        self.messages = []

    def info(self, message):
        self.messages.append(("info", message))

    def warning(self, message):
        self.messages.append(("warning", message))

    def error(self, message):
        self.messages.append(("error", message))


class FakeBackend:
    name = "fake_velocity_backend"
    observation_mode = "velocity_normalized_body"

    def __init__(self):
        self.reset_calls = 0

    def reset(self):
        self.reset_calls += 1


class FakeController:
    def __init__(self):
        self._control_mode = CONTROL_MODE_ROS_MANUAL
        self._backend = FakeBackend()
        self._active_command = None
        self._goal_reached_latched = True
        self._goal_within_tolerance_since_sec = 1.0
        self._stuck_phase = "idle"
        self._logger = FakeLogger()
        self._now = 10.0

    def _now_sec(self):
        self._now += 0.1
        return self._now

    def _reset_stuck_recovery(self, *, phase):
        self._stuck_phase = phase

    def get_logger(self):
        return self._logger


def _velocity_msg(vx, vy=0.0, vz=0.0, *, frame="controller_body"):
    msg = TwistStamped()
    msg.header.frame_id = frame
    msg.twist.linear.x = float(vx)
    msg.twist.linear.y = float(vy)
    msg.twist.linear.z = float(vz)
    return msg


def test_velocity_refresh_updates_command_without_resetting_backend():
    controller = FakeController()

    MotionControllerNode._velocity_command_callback(controller, _velocity_msg(0.1))

    assert controller._backend.reset_calls == 1
    assert controller._active_command.mode == "velocity"
    assert controller._stuck_phase == "new_velocity"

    MotionControllerNode._velocity_command_callback(controller, _velocity_msg(0.2, vz=0.05))

    assert controller._backend.reset_calls == 1
    assert controller._active_command.mode == "velocity"
    np.testing.assert_allclose(
        controller._active_command.desired_linear_velocity_body,
        np.asarray([0.2, 0.0, 0.05], dtype=np.float32).reshape(3),
    )


def test_velocity_command_resets_backend_when_entering_velocity_mode():
    controller = FakeController()
    controller._active_command = MotionCommand(
        mode="position",
        source_frame="controller_world",
        issued_at_sec=1.0,
        target_position_world=np.zeros(3, dtype=np.float32),
    )

    MotionControllerNode._velocity_command_callback(controller, _velocity_msg(0.1, frame="controller_body"))

    assert controller._backend.reset_calls == 1
