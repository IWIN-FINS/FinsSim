#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
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


def load_real_summary(path: Path) -> list[dict[str, float | int | str]]:
    rows: list[dict[str, float | int | str]] = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            rows.append(
                {
                    "source": "real",
                    "trial_index": int(finite_float(row.get("trial_index"), -1)),
                    "target_tau_nm": finite_float(row.get("target_tau_nm")),
                    "estimated_tau_nm": finite_float(row.get("actual_tau_nm")),
                    "abs_estimated_tau_nm": finite_float(row.get("abs_actual_tau_nm")),
                    "tau_estimate_source": "real_rpm_to_force_summary",
                    "steady_yaw_rate_radps": finite_float(row.get("steady_yaw_rate_radps")),
                    "steady_abs_yaw_rate_radps": finite_float(row.get("steady_abs_yaw_rate_radps")),
                    "peak_abs_yaw_rate_radps": finite_float(row.get("peak_abs_yaw_rate_radps")),
                    "initial_abs_yaw_accel_radps2": finite_float(row.get("initial_abs_yaw_accel_radps2")),
                    "sample_count": int(finite_float(row.get("sample_count"))),
                }
            )
    return rows


def load_unity_identifier_csv(path: Path, steady_window_sec: float, initial_window_sec: float) -> list[dict[str, float | int | str]]:
    rows = list(csv.DictReader(path.open(newline="", encoding="utf-8")))
    by_trial: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row.get("axis") == "yaw_y" and row.get("phase") == "excitation":
            by_trial[int(finite_float(row.get("trial_index"), -1))].append(row)

    summaries: list[dict[str, float | int | str]] = []
    for trial_index, trial_rows in sorted(by_trial.items()):
        trial_rows = sorted(trial_rows, key=lambda row: finite_float(row.get("phase_elapsed_sec")))
        if len(trial_rows) < 3:
            continue

        t_max = max(finite_float(row["phase_elapsed_sec"]) for row in trial_rows)
        steady_rows = [
            row for row in trial_rows
            if finite_float(row["phase_elapsed_sec"]) >= t_max - max(steady_window_sec, 0.0)
        ]
        if len(steady_rows) < 2:
            steady_rows = trial_rows

        initial_rows = [
            row for row in trial_rows
            if finite_float(row["phase_elapsed_sec"]) <= max(initial_window_sec, 0.0)
        ]
        if len(initial_rows) < 2:
            initial_rows = trial_rows[: min(len(trial_rows), 5)]

        estimated_tau = np.asarray([finite_float(row["tau_fit_my_yaw_nm"]) for row in trial_rows], dtype=np.float64)
        target_tau = np.asarray([finite_float(row["tau_target_my_yaw_nm"]) for row in trial_rows], dtype=np.float64)
        rpm_columns = [key for key in trial_rows[0].keys() if key.startswith("rpm_canonical_")]
        has_rpm_feedback = any(
            abs(finite_float(row.get(column))) > 1e-3
            for row in trial_rows
            for column in rpm_columns
        )
        yaw_rate = np.asarray([finite_float(row["nu_yaw_radps"]) for row in trial_rows], dtype=np.float64)
        yaw_accel_initial = np.asarray(
            [finite_float(row["nudot_yaw_radps2"]) for row in initial_rows],
            dtype=np.float64,
        )
        steady_rate = float(np.median([finite_float(row["nu_yaw_radps"]) for row in steady_rows]))

        summaries.append(
            {
                "source": "unity_dwp2_current",
                "trial_index": trial_index,
                "target_tau_nm": float(np.median(target_tau)),
                "estimated_tau_nm": float(np.median(estimated_tau)),
                "abs_estimated_tau_nm": abs(float(np.median(estimated_tau))),
                "tau_estimate_source": "unity_rpm_to_force" if has_rpm_feedback else "unity_command_fallback",
                "steady_yaw_rate_radps": steady_rate,
                "steady_abs_yaw_rate_radps": abs(steady_rate),
                "peak_abs_yaw_rate_radps": float(np.max(np.abs(yaw_rate))),
                "initial_abs_yaw_accel_radps2": abs(float(np.median(yaw_accel_initial))),
                "sample_count": len(trial_rows),
            }
        )
    return summaries


def tau_key(kind: str) -> tuple[str, str]:
    if kind == "target":
        return "target_tau_nm", "target_tau_nm"
    if kind == "estimated":
        return "estimated_tau_nm", "abs_estimated_tau_nm"
    raise ValueError(f"unknown tau kind: {kind}")


def fit_damping_curve(points: list[dict[str, float | int | str]], min_rate: float, kind: str) -> dict[str, float | int | None]:
    signed_key, abs_key = tau_key(kind)
    valid = [
        point for point in points
        if float(point["steady_abs_yaw_rate_radps"]) > min_rate
        and abs(float(point[signed_key])) > 0.01
    ]
    yaw_rate = np.asarray([float(point["steady_abs_yaw_rate_radps"]) for point in valid], dtype=np.float64)
    tau = np.asarray([abs(float(point[signed_key])) if kind == "target" else float(point[abs_key]) for point in valid], dtype=np.float64)
    if len(yaw_rate) < 2:
        return {"d_linear": None, "d_quadratic": None, "rmse_nm": None, "r2": None, "n": len(yaw_rate)}

    features = np.stack([yaw_rate, yaw_rate * yaw_rate], axis=1)
    coeffs, *_ = np.linalg.lstsq(features, tau, rcond=None)
    coeffs = np.maximum(coeffs, 0.0)
    predicted = features @ coeffs
    rmse = float(np.sqrt(np.mean((predicted - tau) ** 2)))
    ss_res = float(np.sum((tau - predicted) ** 2))
    ss_tot = float(np.sum((tau - np.mean(tau)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-12 else None
    return {
        "d_linear": float(coeffs[0]),
        "d_quadratic": float(coeffs[1]),
        "rmse_nm": rmse,
        "r2": r2,
        "n": len(yaw_rate),
    }


def damping_tau(fit: dict[str, float | int | None], yaw_rate: np.ndarray) -> np.ndarray | None:
    d_linear = fit.get("d_linear")
    d_quadratic = fit.get("d_quadratic")
    if d_linear is None or d_quadratic is None:
        return None
    return float(d_linear) * yaw_rate + float(d_quadratic) * yaw_rate * yaw_rate


def write_summary(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    fields = [
        "source",
        "trial_index",
        "target_tau_nm",
        "estimated_tau_nm",
        "abs_estimated_tau_nm",
        "tau_estimate_source",
        "steady_yaw_rate_radps",
        "steady_abs_yaw_rate_radps",
        "peak_abs_yaw_rate_radps",
        "initial_abs_yaw_accel_radps2",
        "sample_count",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def plot_tau_to_rate(
    path: Path,
    real: list[dict[str, float | int | str]],
    unity: list[dict[str, float | int | str]],
    kind: str,
) -> None:
    signed_key, abs_key = tau_key(kind)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for data, label, color, marker in (
        (real, "real", "#1f77b4", "o"),
        (unity, "unity DWP2 current", "#d62728", "s"),
    ):
        pairs = sorted(
            (
                abs(float(row[signed_key])) if kind == "target" else float(row[abs_key]),
                float(row["steady_abs_yaw_rate_radps"]),
            )
            for row in data
        )
        ax.scatter([x for x, _ in pairs], [y for _, y in pairs], label=label, color=color, marker=marker, s=48, alpha=0.9)
        ax.plot([x for x, _ in pairs], [y for _, y in pairs], color=color, alpha=0.45)
    ax.set_xlabel(f"abs {kind} yaw torque (Nm)")
    ax.set_ylabel("steady abs yaw rate (rad/s)")
    ax.set_title(f"{kind}_tau -> steady yaw_rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_damping_curve(
    path: Path,
    real: list[dict[str, float | int | str]],
    unity: list[dict[str, float | int | str]],
    real_fit: dict[str, float | int | None],
    unity_fit: dict[str, float | int | None],
    kind: str,
) -> None:
    signed_key, abs_key = tau_key(kind)
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    for data, label, color, marker, fit in (
        (real, "real", "#1f77b4", "o", real_fit),
        (unity, "unity DWP2 current", "#d62728", "s", unity_fit),
    ):
        rates = np.asarray([float(row["steady_abs_yaw_rate_radps"]) for row in data], dtype=np.float64)
        tau = np.asarray(
            [abs(float(row[signed_key])) if kind == "target" else float(row[abs_key]) for row in data],
            dtype=np.float64,
        )
        ax.scatter(rates, tau, label=f"{label} samples", color=color, marker=marker, s=48, alpha=0.9)
        fitted_tau = damping_tau(fit, np.asarray([0.0, 1.0], dtype=np.float64))
        if fitted_tau is not None:
            curve_rate = np.linspace(0.0, max(float(rates.max()) if len(rates) else 0.0, 0.1), 200)
            curve_tau = damping_tau(fit, curve_rate)
            assert curve_tau is not None
            ax.plot(
                curve_rate,
                curve_tau,
                color=color,
                label=f"{label} fit: {float(fit['d_linear']):.3f}r + {float(fit['d_quadratic']):.3f}r^2",
            )
    ax.set_xlabel("steady abs yaw rate r (rad/s)")
    ax.set_ylabel(f"abs {kind} yaw torque tau (Nm)")
    ax.set_title(f"yaw damping curve using {kind} tau: tau = d1*r + d2*r^2")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def plot_metrics(
    path: Path,
    real: list[dict[str, float | int | str]],
    unity: list[dict[str, float | int | str]],
    kind: str,
) -> None:
    signed_key, abs_key = tau_key(kind)
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = [
        ("steady_abs_yaw_rate_radps", "steady abs yaw rate (rad/s)"),
        ("peak_abs_yaw_rate_radps", "peak abs yaw rate (rad/s)"),
        ("initial_abs_yaw_accel_radps2", "initial abs yaw accel (rad/s^2)"),
        ("steady_yaw_rate_radps", "signed steady yaw rate (rad/s)"),
    ]
    for ax, (metric, title) in zip(axes.ravel(), metrics):
        for data, label, color, marker in (
            (real, "real", "#1f77b4", "o"),
            (unity, "unity DWP2 current", "#d62728", "s"),
        ):
            pairs = sorted(
                (
                    abs(float(row[signed_key])) if kind == "target" else float(row[abs_key]),
                    float(row[metric]),
                )
                for row in data
            )
            ax.plot([x for x, _ in pairs], [y for _, y in pairs], marker + "-", color=color, label=label, alpha=0.85)
        ax.set_xlabel(f"abs {kind} yaw torque (Nm)")
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare real yaw summary with Unity hydrodynamic_identifier yaw_y CSV.")
    parser.add_argument("--real-summary", type=Path, required=True)
    parser.add_argument("--unity-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--steady-window", type=float, default=1.5)
    parser.add_argument("--initial-window", type=float, default=0.5)
    parser.add_argument("--min-rate", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    real = load_real_summary(args.real_summary)
    unity = load_unity_identifier_csv(args.unity_csv, args.steady_window, args.initial_window)
    real_fit_estimated = fit_damping_curve(real, args.min_rate, "estimated")
    unity_fit_estimated = fit_damping_curve(unity, args.min_rate, "estimated")
    real_fit_target = fit_damping_curve(real, args.min_rate, "target")
    unity_fit_target = fit_damping_curve(unity, args.min_rate, "target")

    summary_csv = args.output_dir / "unity_vs_real_yaw_trial_summary.csv"
    report_json = args.output_dir / "unity_vs_real_yaw_report.json"
    write_summary(summary_csv, real + unity)

    report: dict[str, object] = {
        "real_summary": str(args.real_summary),
        "unity_csv": str(args.unity_csv),
        "summary_csv": str(summary_csv),
        "model": "abs_tau = d_linear*abs_r + d_quadratic*abs_r^2",
        "tau_definitions": {
            "target": "Same definition on real and Unity: requested yaw wrench before allocation/clipping, then allocated to 8 thruster force commands.",
            "estimated": "Real uses measured RPM converted back to force; Unity uses RPM when available, otherwise falls back to force commands.",
        },
        "real_fit_estimated_tau": real_fit_estimated,
        "unity_fit_estimated_tau": unity_fit_estimated,
        "real_fit_target_tau": real_fit_target,
        "unity_fit_target_tau": unity_fit_target,
    }

    real_rates = [float(row["steady_abs_yaw_rate_radps"]) for row in real if float(row["steady_abs_yaw_rate_radps"]) > args.min_rate]
    unity_rates = [float(row["steady_abs_yaw_rate_radps"]) for row in unity if float(row["steady_abs_yaw_rate_radps"]) > args.min_rate]
    real_tau_test = damping_tau(real_fit_estimated, np.asarray([0.1], dtype=np.float64))
    unity_tau_test = damping_tau(unity_fit_estimated, np.asarray([0.1], dtype=np.float64))
    if real_rates and unity_rates and real_tau_test is not None and unity_tau_test is not None:
        min_rate = max(min(real_rates), min(unity_rates))
        max_rate = min(max(real_rates), max(unity_rates))
        if max_rate > min_rate:
            rates = np.linspace(min_rate, max_rate, 50)
            real_tau = damping_tau(real_fit_estimated, rates)
            unity_tau = damping_tau(unity_fit_estimated, rates)
            assert real_tau is not None and unity_tau is not None
            report["overlap_rate_range_radps"] = [float(min_rate), float(max_rate)]
            report["unity_damping_to_real_tau_ratio_median_over_overlap"] = float(
                np.median(unity_tau / np.maximum(real_tau, 1e-9))
            )
            report["recommended_extra_scale_if_using_axis_multiplier"] = float(
                np.median(real_tau / np.maximum(unity_tau, 1e-9))
            )

    report_json.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    plot_tau_to_rate(args.output_dir / "target_tau_vs_steady_yaw_rate.png", real, unity, "target")
    plot_tau_to_rate(args.output_dir / "estimated_tau_vs_steady_yaw_rate.png", real, unity, "estimated")
    plot_damping_curve(
        args.output_dir / "target_tau_vs_yaw_rate_damping_curve.png",
        real,
        unity,
        real_fit_target,
        unity_fit_target,
        "target",
    )
    plot_damping_curve(
        args.output_dir / "estimated_tau_vs_yaw_rate_damping_curve.png",
        real,
        unity,
        real_fit_estimated,
        unity_fit_estimated,
        "estimated",
    )
    plot_metrics(args.output_dir / "target_tau_yaw_response_metrics_comparison.png", real, unity, "target")
    plot_metrics(args.output_dir / "estimated_tau_yaw_response_metrics_comparison.png", real, unity, "estimated")

    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
