from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


DEFAULT_REAL_DIR = Path("./ros2_ws/data/hydrodynamic_identification/yaw_y")
DEFAULT_UNITY_DWP2_DIR = Path("./ros2_ws/data/unity_dwp2_waterobject_benchmark/yaw_r")
DEFAULT_OUTPUT_DIR = Path("./ros2_ws/data/yaw_response_comparison")


@dataclass(frozen=True)
class TrialSummary:
    source: str
    run_id: str
    trial_index: int
    signed_command_nm: float
    abs_command_nm: float
    excitation_sec: float
    steady_window_sec: float
    steady_yaw_rate_radps: float
    steady_abs_yaw_rate_radps: float
    peak_abs_yaw_rate_radps: float
    initial_yaw_accel_radps2: float
    initial_abs_yaw_accel_radps2: float
    time_constant_sec: float | None
    sample_count: int
    steady_sample_count: int


def _finite_float(value: object, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _natural_key(path: Path) -> tuple[str, int]:
    match = re.search(r"_(\d+)", path.stem)
    return (path.stem, int(match.group(1)) if match else -1)


def _group_by_trial(rows: Iterable[dict[str, str]]) -> dict[int, list[dict[str, str]]]:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        trial = int(_finite_float(row.get("trial_index"), -1))
        if trial < 0:
            continue
        grouped.setdefault(trial, []).append(row)
    return grouped


def _summarize_trial(
    *,
    source: str,
    run_id: str,
    trial_index: int,
    rows: list[dict[str, str]],
    level_field: str,
    actual_command_field: str | None,
    yaw_rate_field: str,
    yaw_accel_field: str,
    steady_window_sec: float,
    initial_window_sec: float,
) -> TrialSummary | None:
    excitation = [row for row in rows if str(row.get("phase", "")).strip() == "excitation"]
    if len(excitation) < 3:
        return None

    times = np.asarray([_finite_float(row.get("phase_elapsed_sec")) for row in excitation], dtype=np.float64)
    yaw_rate = np.asarray([_finite_float(row.get(yaw_rate_field)) for row in excitation], dtype=np.float64)
    yaw_accel = np.asarray([_finite_float(row.get(yaw_accel_field)) for row in excitation], dtype=np.float64)
    level = np.asarray([_finite_float(row.get(level_field)) for row in excitation], dtype=np.float64)

    command_values = level
    if actual_command_field is not None and actual_command_field in excitation[0]:
        actual = np.asarray([_finite_float(row.get(actual_command_field)) for row in excitation], dtype=np.float64)
        if np.nanmedian(np.abs(actual)) > 1e-6:
            command_values = actual

    finite = np.isfinite(times) & np.isfinite(yaw_rate) & np.isfinite(yaw_accel) & np.isfinite(command_values)
    times = times[finite]
    yaw_rate = yaw_rate[finite]
    yaw_accel = yaw_accel[finite]
    command_values = command_values[finite]
    if len(times) < 3:
        return None

    order = np.argsort(times)
    times = times[order]
    yaw_rate = yaw_rate[order]
    yaw_accel = yaw_accel[order]
    command_values = command_values[order]

    excitation_sec = float(max(times[-1] - times[0], 0.0))
    command = float(np.nanmedian(command_values))
    steady_start = times[-1] - max(steady_window_sec, 0.0)
    steady_mask = times >= steady_start
    if np.count_nonzero(steady_mask) < 2:
        steady_mask = np.ones_like(times, dtype=bool)
    steady_rate = float(np.nanmedian(yaw_rate[steady_mask]))

    initial_end = times[0] + max(initial_window_sec, 0.0)
    initial_mask = times <= initial_end
    if np.count_nonzero(initial_mask) < 2:
        initial_mask = np.arange(len(times)) < min(len(times), 5)
    initial_accel = float(np.nanmedian(yaw_accel[initial_mask]))

    peak_abs_rate = float(np.nanmax(np.abs(yaw_rate)))
    steady_abs_rate = abs(steady_rate)
    initial_abs_accel = abs(initial_accel)

    time_constant: float | None = None
    if steady_abs_rate > 1e-5:
        response_sign = 1.0 if steady_rate >= 0.0 else -1.0
        signed_rate = yaw_rate * response_sign
        threshold = 0.632 * steady_abs_rate
        hits = np.flatnonzero(signed_rate >= threshold)
        if len(hits) > 0:
            time_constant = float(max(times[hits[0]] - times[0], 0.0))

    return TrialSummary(
        source=source,
        run_id=run_id,
        trial_index=trial_index,
        signed_command_nm=command,
        abs_command_nm=abs(command),
        excitation_sec=excitation_sec,
        steady_window_sec=max(steady_window_sec, 0.0),
        steady_yaw_rate_radps=steady_rate,
        steady_abs_yaw_rate_radps=steady_abs_rate,
        peak_abs_yaw_rate_radps=peak_abs_rate,
        initial_yaw_accel_radps2=initial_accel,
        initial_abs_yaw_accel_radps2=initial_abs_accel,
        time_constant_sec=time_constant,
        sample_count=len(times),
        steady_sample_count=int(np.count_nonzero(steady_mask)),
    )


def load_real_yaw_trials(root: Path, steady_window_sec: float, initial_window_sec: float) -> list[TrialSummary]:
    out: list[TrialSummary] = []
    for path in sorted(root.glob("yaw_y_*/yaw_y_*.csv"), key=_natural_key):
        if path.name.endswith(".fit.csv"):
            continue
        rows = [row for row in _read_csv(path) if row.get("axis") == "yaw_y"]
        for trial_index, trial_rows in sorted(_group_by_trial(rows).items()):
            summary = _summarize_trial(
                source="real",
                run_id=path.parent.name,
                trial_index=trial_index,
                rows=trial_rows,
                level_field="level",
                actual_command_field="tau_fit_my_yaw_nm",
                yaw_rate_field="nu_yaw_radps",
                yaw_accel_field="nudot_yaw_radps2",
                steady_window_sec=steady_window_sec,
                initial_window_sec=initial_window_sec,
            )
            if summary is not None:
                out.append(summary)
    return out


def load_unity_dwp2_yaw_trials(root: Path, steady_window_sec: float, initial_window_sec: float) -> list[TrialSummary]:
    out: list[TrialSummary] = []
    for path in sorted(root.glob("yaw_r_*/yaw_r_*_dwp2_motion.csv"), key=_natural_key):
        rows = [row for row in _read_csv(path) if row.get("axis") == "yaw_r"]
        for trial_index, trial_rows in sorted(_group_by_trial(rows).items()):
            summary = _summarize_trial(
                source="unity_dwp2",
                run_id=path.parent.name,
                trial_index=trial_index,
                rows=trial_rows,
                level_field="command_level",
                actual_command_field=None,
                yaw_rate_field="nu_r_radps",
                yaw_accel_field="nudot_r_radps2",
                steady_window_sec=steady_window_sec,
                initial_window_sec=initial_window_sec,
            )
            if summary is not None:
                out.append(summary)
    return out


def _fit_damping_curve(trials: list[TrialSummary], *, min_steady_yaw_rate: float) -> dict[str, float | int | None]:
    points = [
        (trial.steady_abs_yaw_rate_radps, trial.abs_command_nm)
        for trial in trials
        if trial.steady_abs_yaw_rate_radps >= max(min_steady_yaw_rate, 0.0) and trial.abs_command_nm > 1e-5
    ]
    if len(points) < 2:
        return {"sample_count": len(points), "d_linear": None, "d_quadratic": None, "rmse_nm": None}

    omega = np.asarray([p[0] for p in points], dtype=np.float64)
    tau = np.asarray([p[1] for p in points], dtype=np.float64)
    features = np.stack([omega, omega * omega], axis=1)
    coeffs, *_ = np.linalg.lstsq(features, tau, rcond=None)
    coeffs = np.maximum(coeffs, 0.0)
    pred = features @ coeffs
    rmse = float(np.sqrt(np.mean((pred - tau) ** 2)))
    return {
        "sample_count": len(points),
        "d_linear": float(coeffs[0]),
        "d_quadratic": float(coeffs[1]),
        "rmse_nm": rmse,
    }


def _damping_tau(fit: dict[str, float | int | None], omega: np.ndarray) -> np.ndarray | None:
    d_linear = fit.get("d_linear")
    d_quadratic = fit.get("d_quadratic")
    if d_linear is None or d_quadratic is None:
        return None
    return float(d_linear) * omega + float(d_quadratic) * omega * omega


def _recommend_yaw_scale(
    real_fit: dict[str, float | int | None],
    unity_fit: dict[str, float | int | None],
    real_trials: list[TrialSummary],
    unity_trials: list[TrialSummary],
    *,
    min_steady_yaw_rate: float,
) -> dict[str, object]:
    real_tau = _damping_tau(real_fit, np.asarray([0.1], dtype=np.float64))
    unity_tau = _damping_tau(unity_fit, np.asarray([0.1], dtype=np.float64))
    if real_tau is None or unity_tau is None:
        return {"available": False, "reason": "not enough real/unity points to fit both damping curves"}

    real_valid = [t for t in real_trials if t.steady_abs_yaw_rate_radps >= max(min_steady_yaw_rate, 0.0)]
    unity_valid = [t for t in unity_trials if t.steady_abs_yaw_rate_radps >= max(min_steady_yaw_rate, 0.0)]
    real_command_levels = sorted({round(t.abs_command_nm, 3) for t in real_valid})
    unity_command_levels = sorted({round(t.abs_command_nm, 3) for t in unity_valid})
    warnings: list[str] = []
    if len(real_command_levels) < 3:
        warnings.append("real yaw data has fewer than 3 effective command levels")
    if len(unity_command_levels) < 3:
        warnings.append("unity DWP2 yaw data has fewer than 3 effective command levels")
    common_levels = sorted(set(real_command_levels) & set(unity_command_levels))
    if len(common_levels) == 0:
        warnings.append("real and unity yaw command levels do not overlap; recommendation is based on fitted curves, not matched force levels")

    min_rate = max(
        min((t.steady_abs_yaw_rate_radps for t in real_valid), default=0.0),
        min((t.steady_abs_yaw_rate_radps for t in unity_valid), default=0.0),
    )
    max_rate = min(
        max((t.steady_abs_yaw_rate_radps for t in real_valid), default=0.0),
        max((t.steady_abs_yaw_rate_radps for t in unity_valid), default=0.0),
    )
    if max_rate <= min_rate:
        min_rate = 0.02
        max_rate = max(0.2, max(
            max((t.steady_abs_yaw_rate_radps for t in real_trials), default=0.0),
            max((t.steady_abs_yaw_rate_radps for t in unity_trials), default=0.0),
        ))

    rates = np.linspace(max(min_rate, 0.02), max(max_rate, min_rate + 0.02), 20)
    real_curve = _damping_tau(real_fit, rates)
    unity_curve = _damping_tau(unity_fit, rates)
    assert real_curve is not None and unity_curve is not None
    valid = unity_curve > 1e-6
    if np.count_nonzero(valid) == 0:
        return {"available": False, "reason": "unity fitted damping curve is nearly zero"}

    ratios = np.clip(real_curve[valid] / unity_curve[valid], 0.0, 5.0)
    return {
        "available": True,
        "method": "median real_damping(omega) / unity_dwp2_damping(omega) over overlapping steady yaw-rate range",
        "rate_range_radps": [float(rates[valid][0]), float(rates[valid][-1])],
        "recommended_dwp2_body_yaw_torque_scale": float(np.median(ratios)),
        "p25_scale": float(np.percentile(ratios, 25)),
        "p75_scale": float(np.percentile(ratios, 75)),
        "quality_warnings": warnings,
        "real_command_levels_nm": real_command_levels,
        "unity_command_levels_nm": unity_command_levels,
        "common_command_levels_nm": common_levels,
    }


def _write_summary_csv(path: Path, trials: list[TrialSummary]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(TrialSummary.__dataclass_fields__.keys())
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for trial in trials:
            writer.writerow({field: getattr(trial, field) for field in fields})


def _plot(path: Path, trials: list[TrialSummary], report: dict[str, object]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    colors = {"real": "#1f77b4", "unity_dwp2": "#d62728"}
    labels = {"real": "real", "unity_dwp2": "unity DWP2"}

    def series(source: str) -> list[TrialSummary]:
        return sorted([t for t in trials if t.source == source], key=lambda t: (t.abs_command_nm, t.run_id, t.trial_index))

    for source in ("real", "unity_dwp2"):
        xs = np.asarray([t.abs_command_nm for t in series(source)], dtype=np.float64)
        if xs.size == 0:
            continue
        axes[0, 0].plot(xs, [t.steady_abs_yaw_rate_radps for t in series(source)], "o-", color=colors[source], label=labels[source])
        axes[0, 1].plot(xs, [t.peak_abs_yaw_rate_radps for t in series(source)], "o-", color=colors[source], label=labels[source])
        axes[1, 0].plot(xs, [t.initial_abs_yaw_accel_radps2 for t in series(source)], "o-", color=colors[source], label=labels[source])
        axes[1, 1].plot(xs, [math.nan if t.time_constant_sec is None else t.time_constant_sec for t in series(source)], "o-", color=colors[source], label=labels[source])

    axes[0, 0].set_title("steady yaw rate")
    axes[0, 0].set_ylabel("abs yaw rate (rad/s)")
    axes[0, 1].set_title("peak yaw rate")
    axes[0, 1].set_ylabel("abs yaw rate (rad/s)")
    axes[1, 0].set_title("initial yaw acceleration")
    axes[1, 0].set_ylabel("abs yaw accel (rad/s^2)")
    axes[1, 1].set_title("63% time constant")
    axes[1, 1].set_ylabel("time (s)")

    for ax in axes.flat:
        ax.set_xlabel("abs commanded yaw torque (Nm)")
        ax.grid(True, alpha=0.3)
        ax.legend()

    recommendation = report.get("recommended_yaw_scale", {})
    if isinstance(recommendation, dict) and recommendation.get("available"):
        scale = recommendation.get("recommended_dwp2_body_yaw_torque_scale")
        fig.suptitle(f"Yaw response comparison; suggested DWP2 yaw torque scale ~= {float(scale):.3f}")
    else:
        fig.suptitle("Yaw response comparison")

    fig.savefig(path, dpi=160)
    plt.close(fig)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate real yaw_y identification and Unity DWP2 yaw_r benchmark runs into tuning curves."
    )
    parser.add_argument("--real-dir", type=Path, default=DEFAULT_REAL_DIR)
    parser.add_argument("--unity-dwp2-dir", type=Path, default=DEFAULT_UNITY_DWP2_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--steady-window", type=float, default=1.5, help="seconds from the end of excitation used as steady-state window")
    parser.add_argument("--initial-window", type=float, default=0.5, help="seconds from excitation start used for initial acceleration")
    parser.add_argument(
        "--min-steady-yaw-rate",
        type=float,
        default=0.02,
        help="exclude nearly no-motion trials below this steady yaw rate from damping fitting",
    )
    parser.add_argument("--skip-real", action="store_true")
    parser.add_argument("--skip-unity-dwp2", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    trials: list[TrialSummary] = []
    real_trials: list[TrialSummary] = []
    unity_trials: list[TrialSummary] = []
    if not args.skip_real and args.real_dir.exists():
        real_trials = load_real_yaw_trials(args.real_dir, args.steady_window, args.initial_window)
        trials.extend(real_trials)
    if not args.skip_unity_dwp2 and args.unity_dwp2_dir.exists():
        unity_trials = load_unity_dwp2_yaw_trials(args.unity_dwp2_dir, args.steady_window, args.initial_window)
        trials.extend(unity_trials)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "yaw_response_summary.csv"
    report_json = args.output_dir / "yaw_response_report.json"
    plot_png = args.output_dir / "yaw_response_comparison.png"

    real_fit = _fit_damping_curve(real_trials, min_steady_yaw_rate=args.min_steady_yaw_rate)
    unity_fit = _fit_damping_curve(unity_trials, min_steady_yaw_rate=args.min_steady_yaw_rate)
    report = {
        "real_dir": str(args.real_dir),
        "unity_dwp2_dir": str(args.unity_dwp2_dir),
        "steady_window_sec": float(args.steady_window),
        "initial_window_sec": float(args.initial_window),
        "min_steady_yaw_rate_radps": float(args.min_steady_yaw_rate),
        "trial_counts": {
            "real": len(real_trials),
            "unity_dwp2": len(unity_trials),
            "total": len(trials),
        },
        "damping_fit_model": "tau_abs ~= d_linear * abs(yaw_rate) + d_quadratic * abs(yaw_rate)^2",
        "real_damping_fit": real_fit,
        "unity_dwp2_damping_fit": unity_fit,
        "recommended_yaw_scale": _recommend_yaw_scale(
            real_fit,
            unity_fit,
            real_trials,
            unity_trials,
            min_steady_yaw_rate=args.min_steady_yaw_rate,
        ),
        "outputs": {
            "summary_csv": str(summary_csv),
            "plot_png": str(plot_png),
            "report_json": str(report_json),
        },
    }

    _write_summary_csv(summary_csv, trials)
    _plot(plot_png, trials, report)
    report_json.write_text(json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {summary_csv}")
    print(f"wrote {plot_png}")
    print(f"wrote {report_json}")
    recommendation = report["recommended_yaw_scale"]
    if isinstance(recommendation, dict) and recommendation.get("available"):
        print(
            "recommended Dwp2BodyYawTorqueScaler.yawTorqueScale="
            f"{float(recommendation['recommended_dwp2_body_yaw_torque_scale']):.3f} "
            f"(p25={float(recommendation['p25_scale']):.3f}, p75={float(recommendation['p75_scale']):.3f})"
        )


if __name__ == "__main__":
    main()
