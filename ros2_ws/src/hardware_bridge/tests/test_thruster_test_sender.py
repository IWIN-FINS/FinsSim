import pytest

from hardware_bridge.thruster_test_sender import make_thruster_pattern


def values(mode, amplitude=0.2, **kwargs):
    return make_thruster_pattern(mode, amplitude, **kwargs).values


def test_stop_pattern():
    assert values("stop") == [0.0] * 8


def test_vertical_patterns_follow_unity_heuristic():
    assert values("up") == [0.2, 0.2, 0.2, 0.2, 0.0, 0.0, 0.0, 0.0]
    assert values("down") == [-0.2, -0.2, -0.2, -0.2, 0.0, 0.0, 0.0, 0.0]


def test_forward_patterns_follow_unity_heuristic():
    assert values("forward") == [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, -0.2, -0.2]
    assert values("backward") == [0.0, 0.0, 0.0, 0.0, -0.2, -0.2, 0.2, 0.2]


def test_force_mode_forward_patterns_use_newton_units():
    pattern = make_thruster_pattern("forward", 0.6, command_mode="force_n")

    assert pattern.command_mode == "force_n"
    assert pattern.unit == "N"
    assert pattern.values == [0.0, 0.0, 0.0, 0.0, 0.6, 0.6, -0.6, -0.6]


def test_force_mode_turn_patterns_are_clamped_in_newtons():
    pattern = make_thruster_pattern("turn_left", 0.6, command_mode="force_n", turn_limit=0.2)

    assert pattern.values == [0.0, 0.0, 0.0, 0.0, 0.2, 0.2, 0.2, 0.2]


def test_turn_patterns_are_clamped_like_unity_heuristic():
    assert values("turn_left", amplitude=0.2, turn_limit=0.1) == [0.0, 0.0, 0.0, 0.0, 0.1, 0.1, 0.1, 0.1]
    assert values("turn_right", amplitude=0.2, turn_limit=0.1) == [0.0, 0.0, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1]


def test_index_pattern():
    assert values("index", amplitude=-0.3, index=7) == [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -0.3]


def test_index_requires_valid_index():
    with pytest.raises(ValueError):
        values("index")
    with pytest.raises(ValueError):
        values("index", index=8)
