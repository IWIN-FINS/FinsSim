"""Benchmark physical wrench allocation without starting Unity or PPO training."""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from dataclasses import asdict, dataclass
from typing import Callable

import torch

from finssim_rl.models.traditional_hold_position_wrench import SIM_BODY_WRENCH_LIMITS
from finssim_rl.models.thrust_allocator import ThrustAllocator


@dataclass
class Measurement:
    name: str
    mean_ms: float
    samples_per_second: float
    process_cpu_percent: float


def legacy_active_set_allocate(allocator: ThrustAllocator, tau: torch.Tensor) -> torch.Tensor:
    """Pre-vectorization reference, retained only for performance comparison."""

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


def measure(
    name: str,
    fn: Callable[[], torch.Tensor],
    *,
    batch_size: int,
    iterations: int,
    device: torch.device,
) -> Measurement:
    for _ in range(10):
        fn()
    if device.type == "cuda":
        torch.cuda.synchronize(device)

    cpu_start = resource.getrusage(resource.RUSAGE_SELF)
    wall_start = time.perf_counter()
    if device.type == "cuda":
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        stream = torch.cuda.current_stream(device)
        start_event.record(stream)
        for _ in range(iterations):
            fn()
        end_event.record(stream)
        end_event.synchronize()
        elapsed_s = start_event.elapsed_time(end_event) / 1000.0
    else:
        for _ in range(iterations):
            fn()
        elapsed_s = time.perf_counter() - wall_start
    wall_elapsed_s = time.perf_counter() - wall_start
    cpu_end = resource.getrusage(resource.RUSAGE_SELF)
    cpu_elapsed_s = (cpu_end.ru_utime + cpu_end.ru_stime) - (cpu_start.ru_utime + cpu_start.ru_stime)

    return Measurement(
        name=name,
        mean_ms=elapsed_s * 1000.0 / iterations,
        samples_per_second=batch_size * iterations / elapsed_s,
        process_cpu_percent=100.0 * cpu_elapsed_s / wall_elapsed_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--skip-legacy", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    torch.manual_seed(20260816)
    allocator = ThrustAllocator(
        device=str(device),
        allocation_mode=ThrustAllocator.PHYSICAL_WRENCH_ALLOCATOR,
        physical_wrench_limits=SIM_BODY_WRENCH_LIMITS,
        thruster_force_limit_positive=(7.0,) * 8,
        thruster_force_limit_negative=(7.0,) * 8,
        debug_print_interval=0,
    )
    empirical = ThrustAllocator(
        device=str(device),
        allocation_mode=ThrustAllocator.EMPIRICAL_THRUSTER_MIXER,
        control_axis_ranges=SIM_BODY_WRENCH_LIMITS,
        debug_print_interval=0,
    )
    limits = torch.tensor(SIM_BODY_WRENCH_LIMITS, dtype=torch.float32, device=device)
    tau = (torch.rand(args.batch_size, 6, device=device) * 2.0 - 1.0) * limits

    with torch.no_grad():
        vectorized = allocator(tau)
        legacy = legacy_active_set_allocate(allocator, tau) if not args.skip_legacy else None

        results = [
            measure(
                "physical_vectorized_cached",
                lambda: allocator(tau),
                batch_size=args.batch_size,
                iterations=args.iterations,
                device=device,
            ),
            measure(
                "empirical_matrix_mixer",
                lambda: empirical(tau),
                batch_size=args.batch_size,
                iterations=args.iterations,
                device=device,
            ),
        ]
        if not args.skip_legacy:
            results.append(
                measure(
                    "physical_legacy_per_sample",
                    lambda: legacy_active_set_allocate(allocator, tau),
                    batch_size=args.batch_size,
                    iterations=args.iterations,
                    device=device,
                )
            )

    output = {
        "pid": os.getpid(),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "CPU",
        "batch_size": args.batch_size,
        "iterations": args.iterations,
        "cache_entries": int(allocator.free_mask_pseudoinverses.shape[0]),
        "cache_bytes": allocator.free_mask_pseudoinverses.numel() * allocator.free_mask_pseudoinverses.element_size(),
        "max_abs_difference_to_legacy": (
            float(torch.max(torch.abs(vectorized - legacy)).item()) if legacy is not None else None
        ),
        "measurements": [asdict(result) for result in results],
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
