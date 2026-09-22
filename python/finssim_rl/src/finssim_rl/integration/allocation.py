"""Public FinsROV allocation helpers shared by adjacent backends."""

from __future__ import annotations

from typing import Sequence

from finssim_rl.models.thrust_allocator import ThrustAllocator

# Unity's names and the Python allocator's geometry use the same canonical
# physical order.  The symbolic names here make the boundary explicit without
# requiring a Unity assembly at Python import time.
CANONICAL_THRUSTER_ORDER = (
    "Vertical1",
    "Vertical2",
    "Vertical3",
    "Vertical4",
    "Horizontal1",
    "Horizontal2",
    "Horizontal3",
    "Horizontal4",
)

# FinsROV_Fossen pure-axis limits in allocator order
# [Fx, Fy, Fz, Mx, My(yaw), Mz].  These match the 16D HoldForPosition physical
# wrench baseline and are deliberately kept in one public location.
DEFAULT_PHYSICAL_WRENCH_LIMITS = (
    19.528527,
    22.501886,
    18.415027,
    3.863995,
    7.080833,
    3.079985,
)


def build_physical_thrust_allocator(
    *,
    device: str = "cpu",
    physical_wrench_limits: Sequence[float] = DEFAULT_PHYSICAL_WRENCH_LIMITS,
    thruster_force_limit_positive: Sequence[float] = (7.0,) * 8,
    thruster_force_limit_negative: Sequence[float] = (7.0,) * 8,
) -> ThrustAllocator:
    """Return the bounded physical FinsROV wrench-to-thruster allocator."""

    return ThrustAllocator(
        device=device,
        allocation_mode=ThrustAllocator.PHYSICAL_WRENCH_ALLOCATOR,
        physical_wrench_limits=physical_wrench_limits,
        thruster_force_limit_positive=thruster_force_limit_positive,
        thruster_force_limit_negative=thruster_force_limit_negative,
        debug_print_interval=0,
    )
