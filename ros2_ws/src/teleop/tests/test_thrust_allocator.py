import numpy as np
import pytest

from teleop.thrust_allocator import (
    ThrustAllocator,
    make_direct_thruster_values,
    policy_wrench_limits_to_controller_body,
)


REAL_POLICY_WRENCH_LIMITS = [18.586585, 18.068824, 25.636077, 3.791353, 2.535180, 6.947713]


def _allocator(*, positive=None, negative=None):
    return ThrustAllocator(
        physical_wrench_limits_policy=REAL_POLICY_WRENCH_LIMITS,
        force_limits_positive_n=positive or [10.0] * 8,
        force_limits_negative_n=negative or [10.0] * 8,
    )


def test_policy_wrench_limits_are_reordered_for_controller_body():
    assert policy_wrench_limits_to_controller_body(REAL_POLICY_WRENCH_LIMITS) == pytest.approx(
        [18.586585, 25.636077, 18.068824, 3.791353, 6.947713, 2.535180]
    )


def test_allocator_heave_maps_to_controller_body_fy_and_reconstructs_wrench():
    allocator = _allocator()
    result = allocator.allocate(0.0, 0.0, 4.0, 0.0)

    assert result.requested_wrench_body == pytest.approx([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    assert result.achieved_wrench_body == pytest.approx(result.requested_wrench_body, abs=2e-4)
    assert all(value > 0.0 for value in result.values[:4])


def test_allocator_surge_and_sway_use_physical_horizontal_geometry():
    allocator = _allocator()

    surge = allocator.allocate(4.0, 0.0, 0.0, 0.0)
    sway = allocator.allocate(0.0, 4.0, 0.0, 0.0)

    assert surge.achieved_wrench_body == pytest.approx(surge.requested_wrench_body, abs=2e-4)
    assert sway.achieved_wrench_body == pytest.approx(sway.requested_wrench_body, abs=2e-4)
    assert surge.values[4] > 0.0 and surge.values[5] > 0.0
    assert surge.values[6] < 0.0 and surge.values[7] < 0.0
    assert sway.values[4] < 0.0 and sway.values[5] > 0.0
    assert sway.values[6] > 0.0 and sway.values[7] < 0.0


def test_allocator_maps_ros_yaw_to_controller_body_my():
    allocator = _allocator()
    result = allocator.allocate(0.0, 0.0, 0.0, 1.0)

    assert result.requested_wrench_body == pytest.approx([0.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    assert result.achieved_wrench_body == pytest.approx(result.requested_wrench_body, abs=2e-4)


def test_allocator_respects_per_thruster_force_bounds():
    allocator = _allocator(positive=[0.5] * 8, negative=[0.25] * 8)
    result = allocator.allocate(8.0, 0.0, 0.0, 0.0)

    values = np.asarray(result.values)
    assert result.clamped is True
    assert np.all(values <= 0.5 + 1e-6)
    assert np.all(values >= -0.25 - 1e-6)
    assert result.saturated_thruster_indices


def test_direct_mode_forward_and_yaw_patterns():
    values = make_direct_thruster_values(
        surge=1.0,
        heave=0.0,
        yaw=0.5,
        max_surge_force_n=2.0,
        max_heave_force_n=1.0,
        max_yaw_force_n=0.4,
        force_limits_positive_n=[10.0] * 8,
        force_limits_negative_n=[10.0] * 8,
    )
    assert values == pytest.approx([0.0, 0.0, 0.0, 0.0, 2.2, 2.2, -1.8, -1.8])
