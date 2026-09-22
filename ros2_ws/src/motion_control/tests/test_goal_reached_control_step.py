import numpy as np

from motion_control.controller_node import CONTROL_MODE_ROS_MANUAL, MotionCommand, MotionControllerNode
from motion_control.state_estimator import VehicleState


class RecordingPublisher:
    def __init__(self):
        self.messages = []

    def publish(self, msg):
        self.messages.append(list(getattr(msg, "data", msg)))


class FakeLogger:
    def info(self, message):
        pass

    def warning(self, message):
        pass

    def error(self, message):
        pass


class FakeBackend:
    name = "fake_velocity_backend"
    observation_mode = "velocity_normalized_body"
    linear_velocity_scale = np.ones(3, dtype=np.float32)
    angular_velocity_scale = np.ones(3, dtype=np.float32)
    reference_velocity_limit = np.ones(3, dtype=np.float32)

    def __init__(self):
        self.predict_calls = 0

    def predict_action(self, observation):
        self.predict_calls += 1
        return np.full(8, 0.25, dtype=np.float32)


class FakeEstimator:
    def __init__(self, state):
        self._state = state
        self.advance_calls = 0

    def advance(self, now_sec):
        self.advance_calls += 1

    def snapshot(self, now_sec):
        return self._state

    def world_error_to_body(self, error_world):
        return error_world.astype(np.float32, copy=True)


class FakeController:
    def __init__(self, *, stop_on_goal_reached):
        target = np.array([0.0, -2.0, 0.0], dtype=np.float32)
        state = VehicleState(
            position_world=target.copy(),
            position_ros=np.array([0.0, 0.0, -2.0], dtype=np.float32),
            orientation_world_body=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
            linear_velocity_body=np.zeros(3, dtype=np.float32),
            linear_velocity_world=np.zeros(3, dtype=np.float32),
            linear_acceleration_body=np.zeros(3, dtype=np.float32),
            linear_acceleration_world=np.zeros(3, dtype=np.float32),
            angular_velocity_body_xyz=np.zeros(3, dtype=np.float32),
            angular_velocity_body_ypr=np.zeros(3, dtype=np.float32),
            angular_velocity_world=np.zeros(3, dtype=np.float32),
            stamp_sec=20.0,
        )
        self._control_mode = CONTROL_MODE_ROS_MANUAL
        self._active_command = MotionCommand(
            mode="position",
            source_frame="controller_world",
            issued_at_sec=1.0,
            target_position_world=target,
            target_orientation_world=None,
        )
        self._estimator = FakeEstimator(state)
        self._backend = FakeBackend()
        self._position_tolerance = 0.1
        self._velocity_tolerance = 0.05
        self._orientation_tolerance_deg = 10.0
        self._goal_reach_hold_time_sec = 10.0
        self._goal_within_tolerance_since_sec = 5.0
        self._goal_reached_latched = False
        self._stop_on_goal_reached = bool(stop_on_goal_reached)
        self._outer_loop_kp = np.ones(3, dtype=np.float32)
        self._max_body_velocity = np.ones(3, dtype=np.float32)
        self._pwm_pub = RecordingPublisher()
        self._zero_publish_count = 0
        self._stuck_phase = "idle"
        self._stuck_escape_action = np.zeros(8, dtype=np.float32)
        self.statuses = []

    def _now_sec(self):
        return 20.0

    def _check_navigation_ready(self, now_sec):
        return True, "ready"

    def _log_navigation_gate_change(self, ready, reason):
        pass

    def _publish_goal_status(self, **kwargs):
        self.statuses.append(kwargs)

    def _publish_stuck_status(self, **kwargs):
        pass

    def _reset_stuck_recovery(self, *, phase):
        self._stuck_phase = phase

    def _stuck_escape_active(self, now_sec):
        return False

    def _update_stuck_recovery(self, **kwargs):
        return False, 0.0, 0.0

    def _goal_is_reached(self, *args, **kwargs):
        return MotionControllerNode._goal_is_reached(self, *args, **kwargs)

    def _publish_zero_pwm(self):
        self._zero_publish_count += 1
        self._pwm_pub.publish([0.0] * 8)

    def _action_to_thruster_command(self, action):
        return action

    def _publish_thruster_action(self, action):
        self._pwm_pub.publish(self._action_to_thruster_command(action))

    def get_logger(self):
        return FakeLogger()


class VisionHoldController(FakeController):
    def __init__(self, *, vision_age_sec):
        super().__init__(stop_on_goal_reached=False)
        self._navigation_reason = "fusion_vision_mode_not_allowed(mode=hold)"
        self._last_idle_zero_sec = 0.0
        self._last_thruster_command = np.asarray([0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4], dtype=np.float32)
        self._vision_loss_hold_modes = ("hold",)
        self._vision_loss_hold_timeout_sec = 3.0
        self._fusion_initialized = True
        self._fusion_vision_mode = "hold"
        self._fusion_age_vision_sec = float(vision_age_sec)
        self._state_timeout_sec = 0.5
        self._last_fusion_status_sec = 20.0

    def _check_navigation_ready(self, now_sec):
        return False, self._navigation_reason

    def _should_hold_thrusters_for_navigation_loss(self, now_sec, reason):
        return MotionControllerNode._should_hold_thrusters_for_navigation_loss(self, now_sec, reason)

    def _publish_thruster_hold(self):
        return MotionControllerNode._publish_thruster_hold(self)


def test_reached_goal_with_stop_false_keeps_command_and_continues_control_output():
    controller = FakeController(stop_on_goal_reached=False)

    MotionControllerNode._control_step(controller)

    assert controller._zero_publish_count == 0
    assert controller._active_command is not None
    assert controller._goal_reached_latched is True
    assert controller.statuses[-1]["reached"] is True
    assert controller._backend.predict_calls == 1
    assert controller._pwm_pub.messages[-1] == [0.25] * 8


def test_reached_goal_with_stop_true_zeroes_thrusters_and_clears_command():
    controller = FakeController(stop_on_goal_reached=True)

    MotionControllerNode._control_step(controller)

    assert controller._zero_publish_count == 1
    assert controller._active_command is None
    assert controller._goal_within_tolerance_since_sec is None
    assert controller.statuses[-1]["reached"] is True
    assert controller._backend.predict_calls == 0
    assert controller._pwm_pub.messages[-1] == [0.0] * 8


def test_navigation_hold_republishes_last_thruster_command_on_short_vision_loss():
    controller = VisionHoldController(vision_age_sec=1.0)

    MotionControllerNode._control_step(controller)

    assert controller._zero_publish_count == 0
    assert np.allclose(controller._pwm_pub.messages[-1], [0.1, 0.2, 0.3, 0.4, -0.1, -0.2, -0.3, -0.4])


def test_navigation_hold_zeroes_thrusters_after_long_vision_loss():
    controller = VisionHoldController(vision_age_sec=3.5)

    MotionControllerNode._control_step(controller)

    assert controller._zero_publish_count == 1
    assert controller._pwm_pub.messages[-1] == [0.0] * 8
