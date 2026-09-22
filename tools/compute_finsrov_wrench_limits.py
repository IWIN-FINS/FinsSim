#!/usr/bin/env python3
"""Compute pure-axis FinsROV wrench limits from thruster force bounds.

The wrench convention used here is the Unity/FinsROV body convention:
    [Fx, Fy, Fz, Mx, My, Mz]
where x is forward, y is up, and z is left. The requested policy-order
output is:
    [Fx_max, Fz_max, Fy_max, Mx_max, Mz_max, My_max]

For each axis, the script solves two linear programs. The five non-target
wrench components are constrained to zero, so the result is a pure-axis
capability rather than a force that also introduces an unwanted yaw/roll/etc.
The final symmetric limit is the smaller of the positive and negative
capabilities, multiplied by the safety factor.

Example:
    uv run --project python/finssim_rl python tools/compute_finsrov_wrench_limits.py \
      --geometry tools/finsrov_wrench_geometry.yaml \
      --positive 7.3809 7.3809 7.3809 7.3809 7.3809 7.3809 7.3809 7.3809 \
      --negative 5.7618 5.7618 5.7618 5.7618 5.7618 5.7618 5.7618 5.7618 \
      --safety-factor 0.7
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml
from scipy.optimize import linprog


THRUSTER_NAMES = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")
WRENCH_NAMES = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
POLICY_WRENCH_ORDER = ("Fx", "Fz", "Fy", "Mx", "Mz", "My")


def _vector(value: Any, length: int, name: str) -> np.ndarray:
    result = np.asarray(value, dtype=np.float64)
    if result.shape != (length,):
        raise ValueError(f"{name} must contain {length} values, got shape {result.shape}")
    return result


def _load_geometry(path: Path) -> tuple[np.ndarray, tuple[str, ...]]:
    with path.open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}

    reference_point = _vector(root.get("reference_point", [0.0, 0.0, 0.0]), 3, "reference_point")
    entries = root.get("thrusters")
    if not isinstance(entries, list) or len(entries) != len(THRUSTER_NAMES):
        raise ValueError(f"thrusters must contain exactly {len(THRUSTER_NAMES)} entries")

    names: list[str] = []
    wrench_columns: list[np.ndarray] = []
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict):
            raise ValueError(f"thrusters[{index}] must be a mapping")
        name = str(entry.get("name", ""))
        if name != THRUSTER_NAMES[index]:
            raise ValueError(
                f"thrusters[{index}] must be {THRUSTER_NAMES[index]}, got {name or '<missing>'}"
            )
        position = _vector(entry.get("position"), 3, f"{name}.position") - reference_point
        direction = _vector(entry.get("direction"), 3, f"{name}.direction")
        direction_norm = np.linalg.norm(direction)
        if direction_norm <= 1e-12:
            raise ValueError(f"{name}.direction cannot be zero")
        direction = direction / direction_norm
        wrench_columns.append(np.concatenate((direction, np.cross(position, direction))))
        names.append(name)

    return np.stack(wrench_columns, axis=1), tuple(names)


def _solve_pure_axis(
    wrench_matrix: np.ndarray,
    positive_limits: np.ndarray,
    negative_limits: np.ndarray,
    axis_index: int,
    sign: float,
) -> tuple[float, np.ndarray]:
    other_axes = [index for index in range(6) if index != axis_index]
    bounds = [(-float(negative_limits[i]), float(positive_limits[i])) for i in range(8)]

    # Maximize sign * target_wrench. scipy.linprog minimizes, hence -sign.
    result = linprog(
        c=-sign * wrench_matrix[axis_index],
        A_eq=wrench_matrix[other_axes],
        b_eq=np.zeros(len(other_axes), dtype=np.float64),
        bounds=bounds,
        method="highs",
    )
    if not result.success or result.x is None:
        raise RuntimeError(
            f"Could not solve pure {WRENCH_NAMES[axis_index]} {'positive' if sign > 0 else 'negative'} limit: "
            f"{result.message}"
        )

    achieved_wrench = wrench_matrix @ result.x
    magnitude = float(sign * achieved_wrench[axis_index])
    if magnitude < -1e-8:
        raise RuntimeError(
            f"Solver returned an invalid {WRENCH_NAMES[axis_index]} magnitude {magnitude:.6g}"
        )
    return max(magnitude, 0.0), result.x


def _format_list(values: Sequence[float]) -> str:
    return "[" + ", ".join(f"{float(value):.6f}" for value in values) + "]"


def compute_limits(
    wrench_matrix: np.ndarray,
    positive_limits: np.ndarray,
    negative_limits: np.ndarray,
    safety_factor: float,
) -> dict[str, Any]:
    positive = np.zeros(6, dtype=np.float64)
    negative = np.zeros(6, dtype=np.float64)
    positive_forces: list[np.ndarray] = []
    negative_forces: list[np.ndarray] = []

    for axis_index, name in enumerate(WRENCH_NAMES):
        positive[axis_index], positive_force = _solve_pure_axis(
            wrench_matrix, positive_limits, negative_limits, axis_index, +1.0
        )
        negative[axis_index], negative_force = _solve_pure_axis(
            wrench_matrix, positive_limits, negative_limits, axis_index, -1.0
        )
        positive_forces.append(positive_force)
        negative_forces.append(negative_force)

    symmetric = safety_factor * np.minimum(positive, negative)
    policy_symmetric = np.asarray([symmetric[WRENCH_NAMES.index(name)] for name in POLICY_WRENCH_ORDER])

    return {
        "positive": positive,
        "negative": negative,
        "symmetric": symmetric,
        "policy_symmetric": policy_symmetric,
        "positive_forces": positive_forces,
        "negative_forces": negative_forces,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geometry", type=Path, required=True, help="Thruster geometry YAML file.")
    parser.add_argument(
        "--positive",
        type=float,
        nargs=8,
        required=True,
        metavar="F+",
        help="Eight positive force limits in canonical thruster order, unit N.",
    )
    parser.add_argument(
        "--negative",
        type=float,
        nargs=8,
        required=True,
        metavar="F-",
        help="Eight negative force magnitudes in canonical thruster order, unit N.",
    )
    parser.add_argument(
        "--safety-factor",
        type=float,
        default=0.7,
        help="Multiplier applied to the symmetric pure-axis capability; default 0.7.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    if not 0.0 < args.safety_factor <= 1.0:
        raise SystemExit("--safety-factor must be in the interval (0, 1]")

    positive_limits = _vector(args.positive, 8, "--positive")
    negative_limits = _vector(args.negative, 8, "--negative")
    if np.any(positive_limits < 0.0) or np.any(negative_limits < 0.0):
        raise SystemExit("force limits must be non-negative magnitudes")

    wrench_matrix, names = _load_geometry(args.geometry.expanduser().resolve())
    result = compute_limits(wrench_matrix, positive_limits, negative_limits, args.safety_factor)

    np.set_printoptions(precision=6, suppress=True)
    print("Thruster order:", list(names))
    print("Wrench order:", list(WRENCH_NAMES))
    print("Wrench matrix B [Fx,Fy,Fz,Mx,My,Mz] = B @ force_N:")
    print(wrench_matrix)
    print()
    print("Positive pure-axis capability:", _format_list(result["positive"]))
    print("Negative pure-axis capability:", _format_list(result["negative"]))
    print("Symmetric capability before safety factor:", _format_list(result["symmetric"] / args.safety_factor))
    print(f"Safety factor: {args.safety_factor:.6f}")
    print("L in [Fx,Fy,Fz,Mx,My,Mz] order:", _format_list(result["symmetric"]))
    print("L in requested [Fx,Fz,Fy,Mx,Mz,My] order:", _format_list(result["policy_symmetric"]))
    print()

    for axis_index, axis_name in enumerate(WRENCH_NAMES):
        print(f"{axis_name} positive force allocation:", _format_list(result["positive_forces"][axis_index]))
        print(f"{axis_name} negative force allocation:", _format_list(result["negative_forces"][axis_index]))


if __name__ == "__main__":
    main()
