import pytest

from hardware_bridge.direct_thruster_sender import clamp_thruster_values


def test_clamp_thruster_values():
    values = clamp_thruster_values([0.4, -0.4, 0.1, 0.0, 1.0, -1.0, 0.29, -0.29], limit=0.3)
    assert values == [0.3, -0.3, 0.1, 0.0, 0.3, -0.3, 0.29, -0.29]


def test_values_are_unbounded_when_limit_is_omitted():
    values = clamp_thruster_values([2.0, -2.0, 5.0, -5.0, 7.0, -7.0, 1.5, -1.5])
    assert values == [2.0, -2.0, 5.0, -5.0, 7.0, -7.0, 1.5, -1.5]


def test_requires_eight_values():
    with pytest.raises(ValueError):
        clamp_thruster_values([0.0] * 7, limit=0.3)


def test_rejects_non_finite_values():
    with pytest.raises(ValueError):
        clamp_thruster_values([0.0] * 7 + [float("nan")])
