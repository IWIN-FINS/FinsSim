import pytest

from hardware_bridge.thruster_force_test_sender import make_force_pattern


def test_single_force_pattern():
    assert make_force_pattern("single", 0.2, index=4) == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.0, 0.0, 0.0]
    )


def test_horizontal_force_pattern():
    assert make_force_pattern("horizontal", -0.1) == pytest.approx(
        [0.0, 0.0, 0.0, 0.0, -0.1, -0.1, -0.1, -0.1]
    )


def test_single_requires_index():
    with pytest.raises(ValueError):
        make_force_pattern("single", 0.2)
