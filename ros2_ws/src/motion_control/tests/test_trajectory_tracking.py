import math

import numpy as np
import pytest
from geometry_msgs.msg import Transform
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint

from motion_control.controller_node import CONTROL_MODE_ROS_MANUAL, MotionControllerNode
from motion_control.state_estimator import VehicleState
from motion_control.trajectory_tracking import (
    OBSERVATION_SIZE,
    TrajectoryPoint,
    build_trajectory30_observation,
    sample_trajectory,
)


def _state() -> VehicleState:
    return VehicleState(
        position_world=np.zeros(3, dtype=np.float32),
        position_ros=np.zeros(3, dtype=np.float32),
        orientation_world_body=np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32),
        linear_velocity_body=np.array([0.2, 0.0, -0.1], dtype=np.float32),
        linear_velocity_world=np.array([0.2, 0.0, -0.1], dtype=np.float32),
        linear_acceleration_body=np.zeros(3, dtype=np.float32),
        linear_acceleration_world=np.zeros(3, dtype=np.float32),
        angular_velocity_body_xyz=np.array([0.0, 0.3, 0.0], dtype=np.float32),
        angular_velocity_body_ypr=np.array([0.3, 0.0, 0.0], dtype=np.float32),
        angular_velocity_world=np.array([0.0, 0.3, 0.0], dtype=np.float32),
        stamp_sec=0.0,
    )


def _points() -> tuple[TrajectoryPoint, ...]:
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return (
        TrajectoryPoint(0.0, np.array([0.0, 0.0, 0.0], dtype=np.float32), identity, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        TrajectoryPoint(1.0, np.array([1.0, 0.0, 0.0], dtype=np.float32), identity, np.array([1.0, 0.0, 0.0], dtype=np.float32)),
    )


def test_sample_trajectory_linearly_interpolates_position_velocity_and_orientation():
    current = sample_trajectory(_points(), 0.25)

    assert current.time_sec == pytest.approx(0.25)
    assert current.position_world == pytest.approx([0.25, 0.0, 0.0])
    assert current.linear_velocity_world == pytest.approx([1.0, 0.0, 0.0])
    assert current.orientation_world == pytest.approx([0.0, 0.0, 0.0, 1.0])


def test_trajectory30_observation_has_the_unity_order_and_shape():
    observation, reference = build_trajectory30_observation(
        _state(),
        _points(),
        0.25,
        preview_step_sec=0.1,
        preview_offset_scale=1.0,
        linear_velocity_scale=1.0,
        angular_velocity_scale=1.0,
        trajectory_duration_sec=1.0,
    )

    assert observation.shape == (OBSERVATION_SIZE,)
    assert reference.position_world == pytest.approx([0.25, 0.0, 0.0])
    assert observation[0:12] == pytest.approx([0.25, 0.0, 0.0, 0.35, 0.0, 0.0, 0.45, 0.0, 0.0, 0.55, 0.0, 0.0])
    assert observation[12:15] == pytest.approx([1.0, 0.0, 0.0])
    assert observation[15:18] == pytest.approx([0.2, 0.0, -0.1])
    assert observation[18:21] == pytest.approx([0.0, 0.3, 0.0])
    assert observation[21:24] == pytest.approx([0.0, 1.0, 0.0])
    assert observation[24:26] == pytest.approx([0.0, 1.0])
    assert observation[26:30] == pytest.approx([1.0, 0.0, 0.0, -1.0], abs=1e-6)


class _SilentLogger:
    def info(self, message: str) -> None:
        del message

    def warning(self, message: str) -> None:
        del message


class _TraditionalPidBackend:
    name = "traditional_pid_position"
    observation_mode = "pose20"

    def __init__(self) -> None:
        self.reset_calls = 0

    def reset(self) -> None:
        self.reset_calls += 1


class _TraditionalPidTrajectoryReceiver:
    """Minimal non-ROS node surface exercised by the callback."""

    def __init__(self) -> None:
        self._control_mode = CONTROL_MODE_ROS_MANUAL
        self._backend = _TraditionalPidBackend()
        self._frame_id = "controller_world"
        self._command_trajectory_topic = "/motion_controller/command/trajectory"
        self._active_command = None
        self._goal_reached_latched = True
        self._goal_within_tolerance_since_sec = 12.0
        self._last_policy_action = np.ones(8, dtype=np.float32)
        self._last_policy_action_sec = 12.0
        self.stuck_phase = ""

    def _now_sec(self) -> float:
        return 20.0

    def _reset_stuck_recovery(self, *, phase: str) -> None:
        self.stuck_phase = phase

    def get_logger(self) -> _SilentLogger:
        return _SilentLogger()


def test_traditional_position_pid_accepts_a_t2_trajectory_reference():
    message = MultiDOFJointTrajectory()
    message.header.frame_id = "controller_world"
    for time_sec, x_m in ((0.0, -1.0), (1.0, 1.0)):
        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = x_m
        transform.translation.y = -0.5
        transform.rotation.w = 1.0
        point.transforms.append(transform)
        point.time_from_start.sec = int(time_sec)
        message.points.append(point)

    receiver = _TraditionalPidTrajectoryReceiver()
    MotionControllerNode._trajectory_callback(receiver, message)

    assert receiver._backend.reset_calls == 1
    assert receiver.stuck_phase == "new_trajectory"
    assert receiver._active_command is not None
    assert receiver._active_command.trajectory_duration_sec == pytest.approx(1.0)
    assert receiver._active_command.trajectory_points[0].position_world == pytest.approx([-1.0, -0.5, 0.0])


def test_lockstep_trajectory_defers_its_zero_time_to_the_next_control_tick():
    message = MultiDOFJointTrajectory()
    message.header.frame_id = "controller_world"
    for time_sec, x_m in ((0.0, -1.0), (1.0, 1.0)):
        point = MultiDOFJointTrajectoryPoint()
        transform = Transform()
        transform.translation.x = x_m
        transform.translation.y = -0.5
        transform.rotation.w = 1.0
        point.transforms.append(transform)
        point.time_from_start.sec = int(time_sec)
        message.points.append(point)

    receiver = _TraditionalPidTrajectoryReceiver()
    receiver._lockstep_enabled = True
    MotionControllerNode._trajectory_callback(receiver, message)

    assert receiver._active_command is not None
    assert math.isnan(receiver._active_command.issued_at_sec)
