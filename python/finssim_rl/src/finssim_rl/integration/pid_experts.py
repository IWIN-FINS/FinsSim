"""Public constructors for controller baselines usable as expert policies."""

from __future__ import annotations

from finssim_rl.models.traditional_hold_position_pid import (
    TraditionalHoldPositionPidModel,
    TraditionalHoldPositionPidTrainingConfig,
)
from typing import Sequence

from finssim_rl.integration.allocation import DEFAULT_PHYSICAL_WRENCH_LIMITS


def build_goal_yaw_thruster8_pid(
    *,
    device: str = "cpu",
    dt: float = 0.1,
    pid_params: Sequence[float] | None = None,
    yaw_pid_params: Sequence[float] = (1e-3, 0.0, 0.0),
    allocator_control_linear_range: float = 10.0,
    allocator_control_axis_ranges: Sequence[float] = DEFAULT_PHYSICAL_WRENCH_LIMITS,
    allocator_allocation_mode: str = "physical_wrench_allocator",
) -> TraditionalHoldPositionPidModel:
    """Build the existing obs16 position-and-yaw PID with an 8D output.

    The GoalYaw IRL task deliberately shares the current 16D pose-control ABI,
    so this controller is a valid Unity expert without altering its historical
    configuration or registration.
    """

    # The historical named configuration still carries the retired ``matrix``
    # allocator string. Keep that configuration untouched for legacy callers.
    # Callers of this integration boundary must instead select their desired
    # allocator explicitly, which lets IRL preserve the deployed sim PID
    # parameters without changing the established RL baseline.
    values = tuple(float(value) for value in allocator_control_axis_ranges)
    config = TraditionalHoldPositionPidTrainingConfig(
        dt=float(dt),
        pid_params=tuple(float(value) for value in pid_params)
        if pid_params is not None
        else TraditionalHoldPositionPidTrainingConfig.pid_params,
        yaw_pid_params=tuple(float(value) for value in yaw_pid_params),
        allocator_control_linear_range=float(allocator_control_linear_range),
        allocator_allocation_mode=str(allocator_allocation_mode),
        allocator_control_axis_ranges=values,
    )
    return TraditionalHoldPositionPidModel(config, device=device)
