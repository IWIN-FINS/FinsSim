#!/usr/bin/env python3
"""Fit per-axis hydrodynamic coefficients J, B, D from Unity CSV logs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import numpy as np


AXIS_NAMES = ("SurgeX", "HeaveY", "SwayZ")
PHASE_EXCITATION = "excitation"


@dataclass
class FitResult:
    method: str
    axis: str
    sample_count: int
    j: float
    b: float
    d: float
    r2: float
    rmse_tau: float
    cond_x: float
    tau_mean_abs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv_path", type=Path, help="CSV exported by HydrodynamicAxisIdentifier.")
    parser.add_argument(
        "--axis",
        action="append",
        choices=AXIS_NAMES,
        help="Axis names to fit. Repeat for multiple axes. Defaults to all present axes.",
    )
    parser.add_argument(
        "--phase",
        default=PHASE_EXCITATION,
        help="Only use rows from this phase. Default: excitation.",
    )
    parser.add_argument(
        "--min-abs-command",
        type=float,
        default=0.05,
        help="Ignore rows with |command_level| smaller than this threshold.",
    )
    parser.add_argument(
        "--smooth-window",
        type=int,
        default=9,
        help="Odd moving-average window for velocity and acceleration smoothing. Use 1 to disable.",
    )
    parser.add_argument(
        "--recompute-acc",
        action="store_true",
        help="Recompute acceleration from velocity instead of using acc_axis_mps2 from CSV.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.02,
        help="Fixed timestep used when --recompute-acc is enabled.",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        help="Optional JSON file path for structured results.",
    )
    return parser.parse_args()


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or values.size == 0:
        return values.copy()
    if window % 2 == 0:
        window += 1
    pad = window // 2
    padded = np.pad(values, pad_width=pad, mode="edge")
    kernel = np.ones(window, dtype=np.float64) / float(window)
    return np.convolve(padded, kernel, mode="valid")


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def select_axes(rows: Iterable[dict[str, str]], requested_axes: list[str] | None) -> list[str]:
    present_axes = []
    seen = set()
    for row in rows:
        axis = row["axis"]
        if axis not in seen:
            present_axes.append(axis)
            seen.add(axis)

    if requested_axes:
        return [axis for axis in requested_axes if axis in seen]
    return present_axes


def extract_axis_samples(
    rows: Iterable[dict[str, str]],
    axis: str,
    phase: str,
    min_abs_command: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    command = []
    tau = []
    vel = []
    acc = []
    segment = []
    for row in rows:
        if row["axis"] != axis:
            continue
        if phase and row["phase"] != phase:
            continue
        cmd = float(row["command_level"])
        if abs(cmd) < min_abs_command:
            continue
        command.append(cmd)
        tau.append(float(row["tau_axis_N"]))
        vel.append(float(row["vel_axis_mps"]))
        acc.append(float(row["acc_axis_mps2"]))
        segment.append(f"{row['axis']}\0{row['trial_id']}\0{row['phase']}")

    return (
        np.asarray(command, dtype=np.float64),
        np.asarray(tau, dtype=np.float64),
        np.asarray(vel, dtype=np.float64),
        np.asarray(acc, dtype=np.float64),
        np.asarray(segment, dtype=object),
    )


def recompute_acceleration(vel: np.ndarray, dt: float) -> np.ndarray:
    if vel.size == 0:
        return vel.copy()
    if vel.size == 1:
        return np.zeros_like(vel)
    return np.gradient(vel, max(dt, 1e-8))


def transform_by_segment(
    values: np.ndarray,
    segment: np.ndarray,
    transform: Callable[[np.ndarray], np.ndarray],
) -> np.ndarray:
    """Apply a 1-D transform without letting filters or gradients cross trial boundaries."""
    if values.size == 0:
        return values.copy()

    output = np.empty_like(values)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and segment[end] == segment[start]:
            end += 1

        output[start:end] = transform(values[start:end])
        start = end

    return output


def evaluate_tau_fit(
    *,
    method: str,
    axis: str,
    tau: np.ndarray,
    vel: np.ndarray,
    acc: np.ndarray,
    j: float,
    b: float,
    d: float,
) -> FitResult:
    x = np.column_stack([acc, vel, np.abs(vel) * vel])
    coeffs = np.asarray([j, b, d], dtype=np.float64)
    tau_pred = x @ coeffs
    residual = tau - tau_pred
    ss_res = float(np.sum(residual**2))
    ss_tot = float(np.sum((tau - np.mean(tau)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else 1.0
    rmse = math.sqrt(max(ss_res / max(len(tau), 1), 0.0))
    cond_x = float(np.linalg.cond(x))
    return FitResult(
        method=method,
        axis=axis,
        sample_count=int(len(tau)),
        j=float(j),
        b=float(b),
        d=float(d),
        r2=r2,
        rmse_tau=rmse,
        cond_x=cond_x,
        tau_mean_abs=float(np.mean(np.abs(tau))) if tau.size else 0.0,
    )


def fit_direct_least_squares(tau: np.ndarray, vel: np.ndarray, acc: np.ndarray, axis: str) -> FitResult:
    x = np.column_stack([acc, vel, np.abs(vel) * vel])
    coeffs, _, _, _ = np.linalg.lstsq(x, tau, rcond=None)
    return evaluate_tau_fit(
        method="lstsq",
        axis=axis,
        tau=tau,
        vel=vel,
        acc=acc,
        j=float(coeffs[0]),
        b=float(coeffs[1]),
        d=float(coeffs[2]),
    )


def print_result(result: FitResult) -> None:
    print(
        f"{result.axis} [{result.method}]: "
        f"J={result.j:.6g}, B={result.b:.6g}, D={result.d:.6g}, "
        f"R2={result.r2:.5f}, RMSE_tau={result.rmse_tau:.6g}, "
        f"cond(X)={result.cond_x:.3g}, samples={result.sample_count}"
    )


def main() -> int:
    args = parse_args()
    rows = load_rows(args.csv_path)
    axes = select_axes(rows, args.axis)
    if not axes:
        raise SystemExit("No matching axes found in CSV.")

    results: list[FitResult] = []
    for axis in axes:
        _, tau, vel, acc, segment = extract_axis_samples(
            rows=rows,
            axis=axis,
            phase=args.phase,
            min_abs_command=args.min_abs_command,
        )

        if tau.size < 10:
            print(f"{axis}: skipped, not enough samples ({tau.size}).")
            continue

        vel = transform_by_segment(
            vel,
            segment,
            lambda values: moving_average(values, args.smooth_window),
        )
        if args.recompute_acc:
            acc = transform_by_segment(
                vel,
                segment,
                lambda values: recompute_acceleration(values, args.dt),
            )
        else:
            acc = transform_by_segment(
                acc,
                segment,
                lambda values: moving_average(values, args.smooth_window),
            )

        result = fit_direct_least_squares(tau=tau, vel=vel, acc=acc, axis=axis)
        results.append(result)
        print_result(result)

    if args.output_json:
        payload = [asdict(result) for result in results]
        args.output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
