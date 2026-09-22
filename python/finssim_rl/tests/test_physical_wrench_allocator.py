from __future__ import annotations

from unittest.mock import patch

import torch

from finssim_rl.models.traditional_hold_position_wrench import SIM_BODY_WRENCH_LIMITS as HOLD_SIM_BODY_WRENCH_LIMITS
from finssim_rl.models.thrust_allocator import ThrustAllocator
from finssim_rl.training.config import get_config


SIM_BODY_WRENCH_LIMITS = torch.tensor(
    [19.528527, 22.501886, 18.415027, 3.863995, 7.080833, 3.079985],
    dtype=torch.float32,
)


def _allocator() -> ThrustAllocator:
    return ThrustAllocator(
        device="cpu",
        allocation_mode="physical_wrench_allocator",
        physical_wrench_limits=SIM_BODY_WRENCH_LIMITS.tolist(),
        thruster_force_limit_positive=[7.0] * 8,
        thruster_force_limit_negative=[7.0] * 8,
        debug_print_interval=0,
    )


def _legacy_active_set_allocate(allocator: ThrustAllocator, tau: torch.Tensor) -> torch.Tensor:
    """Reference implementation of the pre-vectorization allocator."""

    weights = torch.reciprocal(allocator.physical_wrench_limits)
    weighted_b = weights.view(6, 1) * allocator.B
    lower = -allocator.thruster_force_limit_negative
    upper = allocator.thruster_force_limit_positive
    allocated: list[torch.Tensor] = []

    for target in tau:
        weighted_target = weights * target
        force = torch.zeros(8, dtype=target.dtype, device=target.device)
        free = torch.ones(8, dtype=torch.bool, device=target.device)

        for _ in range(8):
            free_indices = torch.nonzero(free, as_tuple=False).flatten()
            if free_indices.numel() == 0:
                break
            fixed_indices = torch.nonzero(~free, as_tuple=False).flatten()
            rhs = weighted_target.clone()
            if fixed_indices.numel() > 0:
                rhs = rhs - weighted_b[:, fixed_indices] @ force[fixed_indices]
            force[free_indices] = torch.linalg.pinv(weighted_b[:, free_indices]) @ rhs

            violations = torch.maximum(
                lower[free_indices] - force[free_indices],
                force[free_indices] - upper[free_indices],
            )
            if not bool(torch.any(violations > 0.0)):
                break

            index = free_indices[torch.argmax(violations)]
            force[index] = torch.clamp(force[index], lower[index], upper[index])
            free[index] = False

        allocated.append(torch.minimum(torch.maximum(force, lower), upper))

    force_n = torch.stack(allocated, dim=0)
    return torch.where(
        force_n >= 0.0,
        force_n / allocator.thruster_force_limit_positive.view(1, 8),
        force_n / allocator.thruster_force_limit_negative.view(1, 8),
    )


def test_physical_wrench_allocator_pure_axis_limits_produce_the_requested_pure_wrench() -> None:
    allocator = _allocator()
    for axis in range(6):
        target = torch.zeros(6, dtype=torch.float32)
        target[axis] = SIM_BODY_WRENCH_LIMITS[axis]
        normalized = allocator(target)
        force_n = normalized * 7.0

        assert torch.max(torch.abs(normalized)) <= 1.0 + 1e-6
        torch.testing.assert_close(allocator.B @ force_n, target, atol=2e-4, rtol=0.0)


def test_physical_wrench_allocator_half_action_keeps_the_pure_wrench_half_scale() -> None:
    allocator = _allocator()
    target = torch.zeros(6, dtype=torch.float32)
    target[0] = SIM_BODY_WRENCH_LIMITS[0] * 0.5

    normalized = allocator(target)
    force_n = normalized * 7.0

    torch.testing.assert_close(allocator.B @ force_n, target, atol=2e-4, rtol=0.0)


def test_physical_wrench_allocator_positive_heave_uses_positive_vertical_commands() -> None:
    allocator = _allocator()
    target = torch.tensor([0.0, 3.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)

    normalized = allocator(target)
    force_n = normalized * 7.0

    assert torch.all(force_n[:4] > 0.0)
    torch.testing.assert_close(allocator.B @ force_n, target, atol=2e-4, rtol=0.0)


def test_physical_wrench_allocator_vectorized_active_set_matches_legacy_reference() -> None:
    allocator = _allocator()
    torch.manual_seed(42)
    target = (torch.rand(257, 6) * 2.0 - 1.0) * SIM_BODY_WRENCH_LIMITS

    expected = _legacy_active_set_allocate(allocator, target)
    actual = allocator(target)

    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=2e-5)


def test_physical_wrench_allocator_does_not_factorize_in_the_hot_path() -> None:
    allocator = _allocator()
    target = torch.zeros(16, 6, dtype=torch.float32)

    with patch("torch.linalg.pinv", side_effect=AssertionError("hot path must not factorize")):
        actual = allocator(target)

    assert actual.shape == (16, 8)


def test_physical_wrench_allocator_loads_the_calibrated_offline_cache() -> None:
    with patch("torch.linalg.pinv", side_effect=AssertionError("calibrated cache must load")):
        allocator = _allocator()

    assert allocator.free_mask_pseudoinverses.shape == (256, 8, 6)


def test_empirical_mixer_does_not_create_an_unused_physical_cache() -> None:
    with patch("torch.linalg.pinv", side_effect=AssertionError("empirical mixer must not factorize")):
        allocator = ThrustAllocator(
            device="cpu",
            allocation_mode="empirical_thruster_mixer",
            debug_print_interval=0,
        )

    assert not hasattr(allocator, "free_mask_pseudoinverses")


def test_empirical_mixer_positive_roll_matches_physical_forward_model_sign() -> None:
    """Keep the legacy roll mixer aligned with the Unity-measured B sign."""

    allocator = ThrustAllocator(
        device="cpu",
        allocation_mode="empirical_thruster_mixer",
        control_axis_ranges=(1.0,) * 6,
        debug_print_interval=0,
    )
    positive_roll = allocator(torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0, 0.0]))
    achieved_wrench = allocator.B @ (positive_roll * 7.0)

    assert torch.all(positive_roll[:2] < 0.0)
    assert torch.all(positive_roll[2:4] > 0.0)
    assert achieved_wrench[3] > 0.0


def test_traditional_hold_position_physical_wrench_baseline_uses_the_bounded_allocator() -> None:
    config = get_config("traditional_hold_position_wrench_physical_wrench_allocator")

    assert config.training_config.allocator_allocation_mode == "physical_wrench_allocator"
    assert config.training_config.wrench_limits == HOLD_SIM_BODY_WRENCH_LIMITS
    assert config.training_config.allocator_physical_wrench_limits == HOLD_SIM_BODY_WRENCH_LIMITS
