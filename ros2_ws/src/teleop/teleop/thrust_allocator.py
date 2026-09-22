from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
from typing import Sequence

import numpy as np


THRUSTER_COUNT = 8
CANONICAL_THRUSTERS = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")
POLICY_WRENCH_ORDER = ("surge", "sway", "heave", "roll", "pitch", "yaw")
CONTROLLER_BODY_WRENCH_ORDER = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")


@dataclass(frozen=True)
class AllocationResult:
    """Physical allocation result in canonical force-N order."""

    values: list[float]
    requested_wrench_body: list[float]
    achieved_wrench_body: list[float]
    residual_wrench_body: list[float]
    saturated_thruster_indices: list[int]
    clamped: bool


def _as_float_array(values: Sequence[float], expected_len: int, name: str) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float32).reshape(-1)
    if array.shape[0] != expected_len:
        raise ValueError(f"{name} must contain {expected_len} values, got {array.shape[0]}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain finite values")
    return array


def policy_wrench_limits_to_controller_body(values: Sequence[float]) -> np.ndarray:
    """Map policy order to the shared allocator's [Fx, Fy, Fz, Mx, My, Mz]."""

    policy_limits = _as_float_array(values, 6, "physical_wrench_limits_policy")
    return policy_limits[[0, 2, 1, 3, 5, 4]]


def _ensure_finssim_rl_importable() -> None:
    search_roots = [Path.cwd(), Path(__file__).resolve()]
    configured_root = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured_root:
        search_roots.insert(0, Path(configured_root).expanduser().resolve())
    for root in search_roots:
        for parent in [root, *root.parents]:
            rl_src = parent / "python/finssim_rl/src"
            if (rl_src / "finssim_rl").is_dir():
                if str(rl_src) not in sys.path:
                    sys.path.insert(0, str(rl_src))
                return
    raise RuntimeError(
        "Could not locate python/finssim_rl/src. "
        "Set FINSSIM_REPO_ROOT or launch teleop from the FinsSim workspace."
    )


class ThrustAllocator:
    """Hand-command wrapper around the shared physical FinsROV allocator.

    ROS joystick commands use REP-103 body axes (x forward, y left, z up).
    The shared FinsROV allocator uses controller-body axes (x forward, y up,
    z left), so allocation explicitly maps [surge, sway, heave, yaw] to
    [Fx, Fy, Fz, Mx, My, Mz] = [surge, heave, sway, 0, yaw, 0].
    """

    def __init__(
        self,
        *,
        physical_wrench_limits_policy: Sequence[float],
        force_limits_positive_n: Sequence[float],
        force_limits_negative_n: Sequence[float],
    ) -> None:
        self.force_limits_positive_n = np.abs(
            _as_float_array(force_limits_positive_n, THRUSTER_COUNT, "force_limits_positive_n")
        )
        self.force_limits_negative_n = np.abs(
            _as_float_array(force_limits_negative_n, THRUSTER_COUNT, "force_limits_negative_n")
        )
        self.physical_wrench_limits_body = policy_wrench_limits_to_controller_body(
            physical_wrench_limits_policy
        )

        _ensure_finssim_rl_importable()
        import torch

        from finssim_rl.models.thrust_allocator import ThrustAllocator as SharedThrustAllocator

        self._torch = torch
        self._allocator = SharedThrustAllocator(
            device="cpu",
            deadzone_comp=0.0,
            control_linear_range=1.0,
            control_axis_ranges=(1.0,) * 6,
            allocation_mode=SharedThrustAllocator.PHYSICAL_WRENCH_ALLOCATOR,
            physical_wrench_limits=self.physical_wrench_limits_body.tolist(),
            thruster_force_limit_positive=self.force_limits_positive_n.tolist(),
            thruster_force_limit_negative=self.force_limits_negative_n.tolist(),
            debug_print_interval=0,
        )
        self.physical_wrench_matrix = self._allocator.B.detach().cpu().numpy().astype(np.float32)

    def allocate(
        self,
        surge_force_n: float,
        sway_force_n: float,
        heave_force_n: float,
        yaw_moment_nm: float,
    ) -> AllocationResult:
        requested = np.array(
            [surge_force_n, heave_force_n, sway_force_n, 0.0, yaw_moment_nm, 0.0],
            dtype=np.float32,
        )
        if not np.all(np.isfinite(requested)):
            raise ValueError("wrench values must be finite")

        with self._torch.no_grad():
            normalized = (
                self._allocator(self._torch.as_tensor(requested, dtype=self._torch.float32))
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
        values = np.where(
            normalized >= 0.0,
            normalized * self.force_limits_positive_n,
            normalized * self.force_limits_negative_n,
        ).astype(np.float32, copy=False)
        achieved = (self.physical_wrench_matrix @ values).astype(np.float32, copy=False)
        residual = (achieved - requested).astype(np.float32, copy=False)
        saturated = np.flatnonzero(
            np.logical_or(
                np.isclose(values, self.force_limits_positive_n, atol=1e-5),
                np.isclose(values, -self.force_limits_negative_n, atol=1e-5),
            )
        )
        return AllocationResult(
            values=[float(value) for value in values],
            requested_wrench_body=[float(value) for value in requested],
            achieved_wrench_body=[float(value) for value in achieved],
            residual_wrench_body=[float(value) for value in residual],
            saturated_thruster_indices=[int(index) for index in saturated],
            clamped=bool(saturated.size),
        )


def make_direct_thruster_values(
    *,
    surge: float,
    heave: float,
    yaw: float,
    max_surge_force_n: float,
    max_heave_force_n: float,
    max_yaw_force_n: float,
    force_limits_positive_n: Sequence[float],
    force_limits_negative_n: Sequence[float],
) -> list[float]:
    """Return direct-pattern canonical force values for wiring checks only."""

    values = np.zeros(THRUSTER_COUNT, dtype=np.float32)
    surge_force = max(-1.0, min(1.0, float(surge))) * abs(float(max_surge_force_n))
    heave_force = max(-1.0, min(1.0, float(heave))) * abs(float(max_heave_force_n))
    yaw_force = max(-1.0, min(1.0, float(yaw))) * abs(float(max_yaw_force_n))

    values[0:4] += heave_force
    values[4] += surge_force + yaw_force
    values[5] += surge_force + yaw_force
    values[6] += -surge_force + yaw_force
    values[7] += -surge_force + yaw_force

    pos = np.abs(_as_float_array(force_limits_positive_n, THRUSTER_COUNT, "force_limits_positive_n"))
    neg = np.abs(_as_float_array(force_limits_negative_n, THRUSTER_COUNT, "force_limits_negative_n"))
    values = np.minimum(np.maximum(values, -neg), pos)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    return [float(value) if math.isfinite(float(value)) else 0.0 for value in values]
