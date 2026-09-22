import math

import pytest
from geometry_msgs.msg import PoseWithCovarianceStamped

from experiment_recorder.runner import _RosExperimentNode


def _pose_with_quaternion(x: float, y: float, z: float, w: float) -> PoseWithCovarianceStamped:
    message = PoseWithCovarianceStamped()
    orientation = message.pose.pose.orientation
    orientation.x = x
    orientation.y = y
    orientation.z = z
    orientation.w = w
    return message


def test_controller_tilt_is_invariant_to_yaw() -> None:
    """A level Y-up vehicle must remain level at every heading."""

    yaw_rad = math.radians(-127.0)
    message = _pose_with_quaternion(0.0, math.sin(yaw_rad / 2.0), 0.0, math.cos(yaw_rad / 2.0))

    assert _RosExperimentNode._controller_tilt_deg(message) == pytest.approx(0.0, abs=1e-9)


def test_controller_tilt_measures_body_y_inclination() -> None:
    tilt_rad = math.radians(8.5)
    message = _pose_with_quaternion(math.sin(tilt_rad / 2.0), 0.0, 0.0, math.cos(tilt_rad / 2.0))

    assert _RosExperimentNode._controller_tilt_deg(message) == pytest.approx(8.5, abs=1e-9)
