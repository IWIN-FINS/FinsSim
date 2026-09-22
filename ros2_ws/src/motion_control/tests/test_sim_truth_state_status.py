import json

import pytest
from std_msgs.msg import String

from motion_control.controller_node import MotionControllerNode
from motion_control.sim_truth_state_status import SimTruthStatusTracker, TopicConfig


TOPICS = (
    TopicConfig("pose", "/finsrov/controller/pose"),
    TopicConfig("imu", "/finsrov/controller/imu"),
    TopicConfig("depth", "/finsrov/controller/depth"),
    TopicConfig("dvl", "/finsrov/controller/dvl"),
)


def make_tracker(**kwargs):
    return SimTruthStatusTracker(topic_configs=TOPICS, **kwargs)


def observe_all(tracker, stamp_sec):
    for name in ("pose", "imu", "depth", "dvl"):
        tracker.observe(name, stamp_sec)


def test_status_is_not_ready_before_required_inputs_arrive():
    payload = make_tracker().build_payload(now_sec=10.0)

    assert payload["ready"] is False
    assert payload["initialized"] is False
    assert payload["vision_mode"] == "lost"
    assert payload["reject_reason"] == "sim_truth_missing:/finsrov/controller/pose"


def test_status_is_ready_when_all_required_inputs_are_fresh():
    tracker = make_tracker(state_timeout_sec=0.5)
    observe_all(tracker, stamp_sec=9.8)

    payload = tracker.build_payload(now_sec=10.0)

    assert payload["ready"] is True
    assert payload["initialized"] is True
    assert payload["vision_mode"] == "fresh"
    assert payload["reject_reason"] == ""
    assert payload["pose_fresh"] is True
    assert payload["imu_fresh"] is True
    assert payload["depth_fresh"] is True
    assert payload["dvl_fresh"] is True


def test_status_closes_when_any_required_input_is_stale():
    tracker = make_tracker(state_timeout_sec=0.5)
    observe_all(tracker, stamp_sec=9.8)
    tracker.observe("imu", 9.0)

    payload = tracker.build_payload(now_sec=10.0)

    assert payload["ready"] is False
    assert payload["initialized"] is False
    assert payload["vision_mode"] == "lost"
    assert payload["reject_reason"] == "sim_truth_stale:/finsrov/controller/imu"


def test_required_topics_subset_ignores_unrequired_inputs():
    tracker = make_tracker(required_inputs=["pose", "imu"], state_timeout_sec=0.5)
    tracker.observe("pose", 9.8)
    tracker.observe("imu", 9.8)

    payload = tracker.build_payload(now_sec=10.0)

    assert payload["ready"] is True
    assert payload["required_topics"] == ["pose", "imu"]
    assert payload["depth_fresh"] is False
    assert payload["dvl_fresh"] is False


def test_payload_json_contains_controller_gate_fields():
    tracker = make_tracker()
    observe_all(tracker, stamp_sec=10.0)

    payload = json.loads(tracker.to_json(now_sec=10.1))

    assert payload["ready"] is True
    assert payload["initialized"] is True
    assert payload["vision_mode"] == "fresh"
    assert payload["reject_reason"] == ""


def test_controller_fusion_callback_accepts_sim_truth_payload():
    tracker = make_tracker()
    observe_all(tracker, stamp_sec=10.0)

    fake_controller = type(
        "FakeController",
        (),
        {
            "_now_sec": lambda self: 10.1,
        },
    )()
    msg = String()
    msg.data = tracker.to_json(now_sec=10.1)

    MotionControllerNode._fusion_status_callback(fake_controller, msg)

    assert fake_controller._fusion_status_received is True
    assert fake_controller._last_fusion_status_sec == 10.1
    assert fake_controller._fusion_ready is True
    assert fake_controller._fusion_initialized is True
    assert fake_controller._fusion_vision_mode == "fresh"
    assert fake_controller._fusion_reject_reason == ""


def test_unknown_required_topic_is_rejected():
    with pytest.raises(ValueError, match="unknown sim truth required topic"):
        make_tracker(required_inputs=["pose", "camera"])
