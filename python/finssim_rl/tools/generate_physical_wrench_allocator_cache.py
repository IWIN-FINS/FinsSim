"""Generate the offline active-set pseudo-inverse cache for FinsROV_Fossen."""

from __future__ import annotations

from pathlib import Path

import torch

from finssim_rl.models.traditional_hold_position_wrench import SIM_BODY_WRENCH_LIMITS
from finssim_rl.models.thrust_allocator import ThrustAllocator


def main() -> None:
    allocator = ThrustAllocator(
        device="cpu",
        allocation_mode=ThrustAllocator.PHYSICAL_WRENCH_ALLOCATOR,
        physical_wrench_limits=SIM_BODY_WRENCH_LIMITS,
        thruster_force_limit_positive=(7.0,) * 8,
        thruster_force_limit_negative=(7.0,) * 8,
        debug_print_interval=0,
    )
    destination = (
        Path(__file__).resolve().parents[1]
        / "src/finssim_rl/models/data/finsrov_fossen_physical_allocator_cache.pt"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": ThrustAllocator._OFFLINE_CACHE_FORMAT_VERSION,
            "physical_wrench_matrix": allocator.B.cpu(),
            "physical_wrench_limits": allocator.physical_wrench_limits.cpu(),
            "free_thruster_masks": allocator.free_thruster_masks.cpu(),
            "free_mask_pseudoinverses": allocator.free_mask_pseudoinverses.cpu(),
        },
        destination,
    )
    print(destination)


if __name__ == "__main__":
    main()
