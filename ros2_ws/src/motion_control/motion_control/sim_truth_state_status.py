from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Mapping, Sequence

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String
from msgs.msg import StampedJson


DEFAULT_REQUIRED_INPUTS = ("pose", "imu", "depth", "dvl")
VALID_INPUTS = frozenset(DEFAULT_REQUIRED_INPUTS)


@dataclass(frozen=True)
class TopicConfig:
    name: str
    topic: str


def _normalize_required_inputs(values: Sequence[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    for value in values:
        name = str(value).strip().lower()
        if not name:
            continue
        if name not in VALID_INPUTS:
            raise ValueError(f"unknown sim truth required topic `{value}`; expected one of {sorted(VALID_INPUTS)}")
        if name not in normalized:
            normalized.append(name)
    return tuple(normalized)


class SimTruthStatusTracker:
    """Build controller-compatible fusion status from simulated truth arrivals."""

    def __init__(
        self,
        *,
        topic_configs: Sequence[TopicConfig],
        required_inputs: Sequence[str] = DEFAULT_REQUIRED_INPUTS,
        state_timeout_sec: float = 0.5,
        vision_mode_when_ready: str = "fresh",
    ) -> None:
        self._topic_configs = tuple(topic_configs)
        self._topic_by_input = {config.name: config.topic for config in self._topic_configs}
        self._required_inputs = _normalize_required_inputs(required_inputs)
        self._state_timeout_sec = float(state_timeout_sec)
        self._vision_mode_when_ready = str(vision_mode_when_ready).strip() or "fresh"
        self._last_stamp_sec: dict[str, float] = {}

    @property
    def required_inputs(self) -> tuple[str, ...]:
        return self._required_inputs

    def observe(self, name: str, observed_sec: float) -> None:
        normalized = str(name).strip().lower()
        if normalized not in VALID_INPUTS:
            raise ValueError(f"unknown sim truth input `{name}`; expected one of {sorted(VALID_INPUTS)}")
        self._last_stamp_sec[normalized] = float(observed_sec)

    def build_payload(self, now_sec: float) -> dict[str, object]:
        now = float(now_sec)
        ages = self._input_ages(now)
        reject_reason = self._reject_reason(ages)
        ready = reject_reason == ""

        payload: dict[str, object] = {
            "ready": ready,
            "initialized": ready,
            "vision_mode": self._vision_mode_when_ready if ready else "lost",
            "coordinate_convention": "sim_truth_unity_bridge_controller_compatible",
            "state_source": "sim_truth",
            "required_topics": list(self._required_inputs),
            "reject_reason": reject_reason,
        }
        for name in sorted(VALID_INPUTS):
            age = ages.get(name)
            fresh = age is not None and (self._state_timeout_sec <= 0.0 or age <= self._state_timeout_sec)
            payload[f"{name}_topic"] = self._topic_by_input.get(name, "")
            payload[f"{name}_fresh"] = fresh
            payload[f"age_{name}"] = age

        # Keep the fields consumed by existing monitoring/fusion tooling present.
        payload.setdefault("vision_fresh", payload.get("pose_fresh", False))
        payload.setdefault("imu_fresh", payload.get("imu_fresh", False))
        payload.setdefault("depth_fresh", payload.get("depth_fresh", False))
        payload["dvl_fresh"] = payload.get("dvl_fresh", False)
        return payload

    def to_json(self, now_sec: float) -> str:
        return json.dumps(self.build_payload(now_sec), separators=(",", ":"))

    def _input_ages(self, now_sec: float) -> dict[str, float | None]:
        return {
            name: None if name not in self._last_stamp_sec else max(0.0, now_sec - self._last_stamp_sec[name])
            for name in VALID_INPUTS
        }

    def _reject_reason(self, ages: Mapping[str, float | None]) -> str:
        for name in self._required_inputs:
            age = ages.get(name)
            topic = self._topic_by_input.get(name, name)
            if age is None:
                return f"sim_truth_missing:{topic}"
            if self._state_timeout_sec > 0.0 and age > self._state_timeout_sec:
                return f"sim_truth_stale:{topic}"
        return ""


class SimTruthStateStatusNode(Node):
    def __init__(self) -> None:
        super().__init__("sim_truth_state_status")

        self._vehicle_name = str(self.declare_parameter("vehicle_name", "FinsROV").value)
        self._status_topic = str(self.declare_parameter("status_topic", "/finsrov/controller/state/status").value)
        self._status_stamped_topic = str(
            self.declare_parameter("status_stamped_topic", "").value
        ).strip()
        self._pose_topic = str(self.declare_parameter("pose_topic", "/finsrov/controller/pose").value)
        self._imu_topic = str(self.declare_parameter("imu_topic", "/finsrov/controller/imu").value)
        self._depth_topic = str(self.declare_parameter("depth_topic", "/finsrov/controller/depth").value)
        self._dvl_topic = str(self.declare_parameter("dvl_topic", "/finsrov/controller/dvl").value)
        required_inputs = self.declare_parameter("required_topics", list(DEFAULT_REQUIRED_INPUTS)).value
        state_timeout_sec = float(self.declare_parameter("state_timeout_sec", 0.5).value)
        status_rate_hz = float(self.declare_parameter("status_rate_hz", 20.0).value)
        vision_mode_when_ready = str(self.declare_parameter("vision_mode_when_ready", "fresh").value)

        topic_configs = (
            TopicConfig("pose", self._pose_topic),
            TopicConfig("imu", self._imu_topic),
            TopicConfig("depth", self._depth_topic),
            TopicConfig("dvl", self._dvl_topic),
        )
        self._tracker = SimTruthStatusTracker(
            topic_configs=topic_configs,
            required_inputs=required_inputs,
            state_timeout_sec=state_timeout_sec,
            vision_mode_when_ready=vision_mode_when_ready,
        )

        self._status_pub = self.create_publisher(String, self._status_topic, 10)
        self._status_stamped_pub = (
            self.create_publisher(StampedJson, self._status_stamped_topic, 10)
            if self._status_stamped_topic
            else None
        )
        self.create_subscription(PoseWithCovarianceStamped, self._pose_topic, self._pose_callback, 10)
        self.create_subscription(Imu, self._imu_topic, self._imu_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, self._depth_topic, self._depth_callback, 10)
        self.create_subscription(TwistWithCovarianceStamped, self._dvl_topic, self._dvl_callback, 10)

        timer_period = 1.0 / max(1.0, status_rate_hz)
        self.create_timer(timer_period, self._publish_status)
        self.get_logger().info(
            "sim_truth_state_status started: "
            f"vehicle_name={self._vehicle_name}, status_topic={self._status_topic}, "
            f"status_stamped_topic={self._status_stamped_topic or '<disabled>'}, "
            f"required_topics={list(self._tracker.required_inputs)}, state_timeout_sec={state_timeout_sec:.3f}"
        )

    def _now_sec(self) -> float:
        now = self.get_clock().now()
        return float(now.nanoseconds) * 1e-9

    def _pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._tracker.observe("pose", self._now_sec())

    def _imu_callback(self, msg: Imu) -> None:
        self._tracker.observe("imu", self._now_sec())

    def _depth_callback(self, msg: PoseWithCovarianceStamped) -> None:
        self._tracker.observe("depth", self._now_sec())

    def _dvl_callback(self, msg: TwistWithCovarianceStamped) -> None:
        self._tracker.observe("dvl", self._now_sec())

    def _publish_status(self) -> None:
        now = self.get_clock().now()
        payload = self._tracker.to_json(float(now.nanoseconds) * 1e-9)
        msg = String()
        msg.data = payload
        self._status_pub.publish(msg)
        if self._status_stamped_pub is not None:
            stamped = StampedJson()
            stamped.header.stamp = now.to_msg()
            stamped.header.frame_id = "controller_world"
            stamped.data = payload
            self._status_stamped_pub.publish(stamped)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SimTruthStateStatusNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
