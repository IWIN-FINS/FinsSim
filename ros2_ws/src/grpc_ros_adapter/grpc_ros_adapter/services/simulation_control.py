from protobuf import simulation_control_pb2
from protobuf import simulation_control_pb2_grpc

import threading
import time

import utils.ros_handle as rh
from builtin_interfaces.msg import Time
from rosgraph_msgs.msg import Clock
from std_msgs.msg import UInt64


class SimulationControl(simulation_control_pb2_grpc.SimulationControlServicer):
    """
    Service used to control the flow of the simulation
    Simulator dictates the clock and does a Step request for ROS nodes
    """
    def __init__(self):
        # `use_sim_time` subscribes to the global `/clock` name.  The adapter
        # itself is launched under `/grpc_ros_adapter`; a relative `clock`
        # publisher would resolve to `/grpc_ros_adapter/clock` and leave every
        # controller on the default global clock stalled.
        self.clock_publisher = rh.Publisher(Clock, "/clock", qos_profile=10)
        self.lockstep_enabled = self._as_bool(rh.get_param("~lockstep_enabled", False))
        self.lockstep_ack_topic = str(
            rh.get_param("~lockstep_ack_topic", "/sim/motion_controller/debug/control_tick_complete")
            or ""
        ).strip()
        try:
            self.lockstep_ack_timeout_wall_sec = float(
                rh.get_param("~lockstep_ack_timeout_wall_sec", 15.0)
            )
        except (TypeError, ValueError):
            self.lockstep_ack_timeout_wall_sec = 15.0
        if self.lockstep_ack_timeout_wall_sec <= 0.0:
            raise ValueError("lockstep_ack_timeout_wall_sec must be positive")

        self._ack_condition = threading.Condition()
        self._last_ack_nanoseconds = -1
        self._ack_subscription = None
        if self.lockstep_enabled:
            if not self.lockstep_ack_topic.startswith("/"):
                raise ValueError("lockstep_ack_topic must be an absolute ROS topic")
            self._ack_subscription = rh.Subscription(
                UInt64,
                self.lockstep_ack_topic,
                self._lockstep_ack_callback,
                qos_profile=50,
            )
            rh.loginfo(
                "grpc_ros_adapter lockstep enabled: waiting for controller acknowledgements on "
                f"{self.lockstep_ack_topic} (timeout={self.lockstep_ack_timeout_wall_sec:.1f}s)"
            )
        self.start_time = -1
        self.current_time = -1

    @staticmethod
    def _as_bool(value) -> bool:
        if isinstance(value, str):
            return value.strip().lower() in {"1", "true", "yes", "on"}
        return bool(value)

    @staticmethod
    def _time_nanoseconds(seconds: int, nanoseconds: int) -> int:
        return int(seconds) * 1_000_000_000 + int(nanoseconds)

    @staticmethod
    def _time_message(seconds: int, nanoseconds: int) -> Time:
        """Create the ROS2 timestamp carried by a SimulationControl RPC.

        ``rh.Time`` is a helper instance, not a callable constructor.  The
        real-time Unity path never exercised this code, but stepped
        simulation uses it on every physics step to drive `/clock`.
        """

        total_nanoseconds = SimulationControl._time_nanoseconds(seconds, nanoseconds)
        message = Time()
        message.sec = total_nanoseconds // 1_000_000_000
        message.nanosec = total_nanoseconds % 1_000_000_000
        return message

    def _lockstep_ack_callback(self, message: UInt64) -> None:
        acknowledgement_ns = int(message.data)
        with self._ack_condition:
            if acknowledgement_ns > self._last_ack_nanoseconds:
                self._last_ack_nanoseconds = acknowledgement_ns
            self._ack_condition.notify_all()

    def _wait_for_lockstep_ack(self, target_nanoseconds: int) -> bool:
        """Wait in the gRPC worker while the ROS spin thread receives the ACK."""

        deadline = time.monotonic() + self.lockstep_ack_timeout_wall_sec
        with self._ack_condition:
            while self._last_ack_nanoseconds < target_nanoseconds:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    rh.logerr(
                        "ROS2 lockstep acknowledgement timed out: "
                        f"target_ns={target_nanoseconds}, last_ack_ns={self._last_ack_nanoseconds}, "
                        f"topic={self.lockstep_ack_topic}"
                    )
                    return False
                self._ack_condition.wait(timeout=remaining)
        return True

    def SetStartTime(self, request, context):

        self.start_time = self._time_message(request.timeSecs, request.timeNsecs)
        self.current_time = self.start_time
        with self._ack_condition:
            self._last_ack_nanoseconds = -1
        sim_clock = Clock()
        sim_clock.clock = self.start_time
        self.clock_publisher.publish(sim_clock)
        return simulation_control_pb2.SetStartTimeResponse(success=True)

    def Step(self, request, context):

        self.current_time = self._time_message(request.totalTimeSecs, request.totalTimeNsecs)
        sim_clock = Clock()
        sim_clock.clock = self.current_time
        self.clock_publisher.publish(sim_clock)
        if self.lockstep_enabled:
            target_nanoseconds = self._time_nanoseconds(request.totalTimeSecs, request.totalTimeNsecs)
            if not self._wait_for_lockstep_ack(target_nanoseconds):
                return simulation_control_pb2.StepResponse(success=False)
        return simulation_control_pb2.StepResponse(success=True)
