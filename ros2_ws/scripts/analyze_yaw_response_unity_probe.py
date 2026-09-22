#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def load_real_summary(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            target = finite_float(row.get("target_tau_nm"))
            actual = finite_float(row.get("actual_tau_nm"))
            rate = finite_float(row.get("steady_abs_yaw_rate_radps"))
            if abs(target) < 1e-6 or rate < 1e-6:
                continue
            rows.append(
                {
                    "target_tau_nm": abs(target),
                    "actual_tau_nm": abs(actual),
                    "steady_abs_yaw_rate_radps": rate,
                }
            )
    return rows


def load_probe_segments(path: Path, min_abs_tau: float, min_duration_sec: float, steady_window_sec: float) -> list[dict[str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            t = finite_float(row.get("time_sec"))
            target = finite_float(row.get("thruster_target_yaw_tau_nm"))
            applied = finite_float(row.get("unity_actual_yaw_tau_nm"), finite_float(row.get("thruster_applied_yaw_tau_nm")))
            hydro = finite_float(row.get("dwp2_applied_hydro_yaw_tau_nm"))
            rate = finite_float(row.get("body_yaw_rate_radps"))
            if t > 0:
                rows.append((t, target, applied, hydro, rate))

    segments: list[list[tuple[float, float, float, float, float]]] = []
    current: list[tuple[float, float, float, float, float]] = []
    current_key: tuple[int, int] | None = None
    for sample in rows:
        _, target, applied, _, _ = sample
        command = applied if abs(applied) >= min_abs_tau else target
        if abs(command) < min_abs_tau:
            if current:
                segments.append(current)
                current = []
                current_key = None
            continue

        key = (1 if command > 0 else -1, int(round(abs(command) * 100.0)))
        if current_key is None or key == current_key:
            current.append(sample)
            current_key = key
        else:
            segments.append(current)
            current = [sample]
            current_key = key

    if current:
        segments.append(current)

    summaries: list[dict[str, float]] = []
    for index, segment in enumerate(segments):
        duration = segment[-1][0] - segment[0][0]
        if duration < min_duration_sec:
            continue
        steady_start = segment[-1][0] - max(0.0, steady_window_sec)
        steady = [sample for sample in segment if sample[0] >= steady_start]
        if len(steady) < 3:
            steady = segment
        target = np.asarray([sample[1] for sample in steady], dtype=np.float64)
        applied = np.asarray([sample[2] for sample in steady], dtype=np.float64)
        hydro = np.asarray([sample[3] for sample in steady], dtype=np.float64)
        rate = np.asarray([sample[4] for sample in steady], dtype=np.float64)
        summaries.append(
            {
                "segment_index": float(index),
                "duration_sec": float(duration),
                "target_tau_nm": abs(float(np.median(target))),
                "actual_tau_nm": abs(float(np.median(applied))),
                "dwp2_hydro_tau_nm": abs(float(np.median(hydro))),
                "steady_abs_yaw_rate_radps": abs(float(np.median(rate))),
                "sample_count": float(len(segment)),
            }
        )
    return summaries


def fit_damping(points: list[dict[str, float]], tau_key: str, min_rate: float) -> dict[str, float | int | None]:
    valid = [p for p in points if p["steady_abs_yaw_rate_radps"] >= min_rate and p[tau_key] > 0.01]
    if len(valid) < 2:
        return {"d_linear": None, "d_quadratic": None, "rmse_nm": None, "r2": None, "n": len(valid)}
    r = np.asarray([p["steady_abs_yaw_rate_radps"] for p in valid], dtype=np.float64)
    tau = np.asarray([p[tau_key] for p in valid], dtype=np.float64)
    features = np.stack([r, r * r], axis=1)
    coeffs, *_ = np.linalg.lstsq(features, tau, rcond=None)
    coeffs = np.maximum(coeffs, 0.0)
    pred = features @ coeffs
    rmse = float(np.sqrt(np.mean((pred - tau) ** 2)))
    ss_res = float(np.sum((tau - pred) ** 2))
    ss_tot = float(np.sum((tau - np.mean(tau)) ** 2))
    return {
        "d_linear": float(coeffs[0]),
        "d_quadratic": float(coeffs[1]),
        "rmse_nm": rmse,
        "r2": 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None,
        "n": len(valid),
    }


def plot_tau_rate(path: Path, real: list[dict[str, float]], unity: list[dict[str, float]], unity_tau_key: str, title: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for rows, label, key, color, marker in (
        (real, "real actual tau", "actual_tau_nm", "#1f77b4", "o"),
        (unity, f"unity {unity_tau_key}", unity_tau_key, "#d62728", "s"),
    ):
        pairs = sorted((p[key], p["steady_abs_yaw_rate_radps"]) for p in rows)
        ax.scatter([x for x, _ in pairs], [y for _, y in pairs], label=label, color=color, marker=marker, s=42)
        ax.plot([x for x, _ in pairs], [y for _, y in pairs], color=color, alpha=0.5)
    ax.set_xlabel("abs yaw torque (Nm)")
    ax.set_ylabel("steady abs yaw rate (rad/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def write_summary(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "segment_index",
        "duration_sec",
        "target_tau_nm",
        "actual_tau_nm",
        "dwp2_hydro_tau_nm",
        "steady_abs_yaw_rate_radps",
        "sample_count",
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare real yaw response with Unity DWP2 runtime probe CSV.")
    parser.add_argument("--real-summary", type=Path, required=True)
    parser.add_argument("--unity-probe-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-abs-tau", type=float, default=0.05)
    parser.add_argument("--min-duration", type=float, default=1.0)
    parser.add_argument("--steady-window", type=float, default=1.5)
    parser.add_argument("--min-rate", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    real = load_real_summary(args.real_summary)
    unity = load_probe_segments(args.unity_probe_csv, args.min_abs_tau, args.min_duration, args.steady_window)
    write_summary(args.output_dir / "unity_probe_yaw_segment_summary.csv", unity)

    report = {
        "real_summary": str(args.real_summary),
        "unity_probe_csv": str(args.unity_probe_csv),
        "unity_segment_summary": str(args.output_dir / "unity_probe_yaw_segment_summary.csv"),
        "model": "abs_tau = d_linear*abs_r + d_quadratic*abs_r^2",
        "real_fit_actual_tau": fit_damping(real, "actual_tau_nm", args.min_rate),
        "unity_fit_actual_thruster_tau": fit_damping(unity, "actual_tau_nm", args.min_rate),
        "unity_fit_dwp2_hydro_tau": fit_damping(unity, "dwp2_hydro_tau_nm", args.min_rate),
        "unity_fit_target_tau": fit_damping(unity, "target_tau_nm", args.min_rate),
        "unity_segment_count": len(unity),
    }
    (args.output_dir / "unity_probe_yaw_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    plot_tau_rate(args.output_dir / "probe_actual_tau_vs_yaw_rate.png", real, unity, "actual_tau_nm", "actual yaw tau -> steady yaw_rate")
    plot_tau_rate(args.output_dir / "probe_dwp2_hydro_tau_vs_yaw_rate.png", real, unity, "dwp2_hydro_tau_nm", "DWP2 hydro yaw tau -> steady yaw_rate")
    plot_tau_rate(args.output_dir / "probe_target_tau_vs_yaw_rate.png", real, unity, "target_tau_nm", "target yaw tau -> steady yaw_rate")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
