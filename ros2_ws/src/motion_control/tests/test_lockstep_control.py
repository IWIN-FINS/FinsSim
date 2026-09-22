from rosgraph_msgs.msg import Clock
import pytest

from motion_control.controller_node import MotionControllerNode


class _RecordingPublisher:
    def __init__(self) -> None:
        self.messages = []

    def publish(self, message) -> None:
        self.messages.append(int(message.data))


class _LockstepControllerSurface:
    """Minimal node surface for the deterministic clock callback contract."""

    def __init__(self) -> None:
        self._lockstep_last_control_tick_ns = None
        self._lockstep_period_ns = 100_000_000  # 10 Hz policy rate
        self._lockstep_clock_override_sec = None
        self._lockstep_ack_pub = _RecordingPublisher()
        self.control_calls = []

    def _safe_control_step(self) -> None:
        self.control_calls.append(self._lockstep_clock_override_sec)


def _clock(seconds: int, nanoseconds: int) -> Clock:
    message = Clock()
    message.clock.sec = seconds
    message.clock.nanosec = nanoseconds
    return message


def test_lockstep_acknowledges_every_physics_clock_but_runs_policy_at_its_configured_rate() -> None:
    controller = _LockstepControllerSurface()

    MotionControllerNode._lockstep_clock_callback(controller, _clock(10, 0))
    MotionControllerNode._lockstep_clock_callback(controller, _clock(10, 20_000_000))
    MotionControllerNode._lockstep_clock_callback(controller, _clock(10, 100_000_000))

    assert controller.control_calls == pytest.approx([10.0, 10.1])
    assert controller._lockstep_ack_pub.messages == [
        10_000_000_000,
        10_020_000_000,
        10_100_000_000,
    ]
    assert controller._lockstep_clock_override_sec is None
