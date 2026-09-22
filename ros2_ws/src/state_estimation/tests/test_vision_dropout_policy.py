import numpy as np

from state_estimation.state_fusion_node import StateFusionNode


class FakePositionEkf:
    def __init__(self):
        self.x = np.zeros(6, dtype=np.float64)
        self.p = np.eye(6, dtype=np.float64)
        self.decay_calls = []
        self.cap_calls = []

    def decay_velocity(self, indices, decay):
        self.decay_calls.append((list(indices), float(decay)))

    def cap_covariance(self, indices, max_value):
        self.cap_calls.append((list(indices), float(max_value)))


def make_node():
    node = StateFusionNode.__new__(StateFusionNode)
    node._position_ekf = FakePositionEkf()
    node._last_vision_time = 0.0
    node._last_vision_position_xy = np.array([1.25, -0.5], dtype=np.float64)
    node._vision_timeout_sec = 0.25
    node._vision_coast_timeout_sec = 0.75
    node._vision_hold_timeout_sec = 3.0
    node._vision_hold_velocity_decay = 0.5
    node._max_position_covariance_xy = 4.0
    node._max_velocity_covariance_xy = 1.0
    return node


def test_vision_dropout_policy_coasts_before_hold_window():
    node = make_node()

    mode = StateFusionNode._apply_vision_dropout_policy(node, now_sec=0.5, vision_fresh=False)

    assert mode == "coast"


def test_vision_dropout_policy_holds_after_coast_before_lost():
    node = make_node()

    mode = StateFusionNode._apply_vision_dropout_policy(node, now_sec=1.0, vision_fresh=False)

    assert mode == "hold"
    assert np.allclose(node._position_ekf.x[0:2], [1.25, -0.5])
    assert node._position_ekf.decay_calls == [([3, 4], 0.5)]


def test_vision_dropout_policy_reports_lost_after_hold_timeout():
    node = make_node()

    mode = StateFusionNode._apply_vision_dropout_policy(node, now_sec=3.1, vision_fresh=False)

    assert mode == "lost"
