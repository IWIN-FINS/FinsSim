#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AXES = ("surge_x", "sway_z", "heave_y", "roll_x", "pitch_z", "yaw_y")
REAL_DIRS = {
    "surge_x": "surge_x/surge_x_015",
    "sway_z": "sway_z/sway_z_004",
    "heave_y": "heave_y/heave_y_015",
    "roll_x": "roll_x/roll_x_009",
    "pitch_z": "pitch_z/pitch_z_003",
    "yaw_y": "yaw_y/yaw_y_007",
}
REAL_RATE = {
    "surge_x": "nu_x_mps",
    "sway_z": "nu_z_mps",
    "heave_y": "nu_y_mps",
    "roll_x": "nu_roll_x_radps",
    "pitch_z": "nu_pitch_z_radps",
    "yaw_y": "nu_yaw_radps",
}
REAL_ACCEL = {
    "surge_x": "nudot_x_mps2",
    "sway_z": "nudot_z_mps2",
    "heave_y": "nudot_y_mps2",
    "roll_x": "nudot_roll_x_radps2",
    "pitch_z": "nudot_pitch_z_radps2",
    "yaw_y": "nudot_yaw_radps2",
}
REAL_TAU = {
    "surge_x": "tau_fit_fx_n",
    "sway_z": "tau_fit_fz_n",
    "heave_y": "tau_fit_fy_n",
    "roll_x": "tau_fit_mx_nm",
    "pitch_z": "tau_fit_mz_nm",
    "yaw_y": "tau_fit_my_yaw_nm",
}

# Controller body is Unity's [x=forward, y=up, z=left].
WRENCH_COMPONENT = {
    "surge_x": ("linear", "x", "N"),
    "sway_z": ("linear", "z", "N"),
    "heave_y": ("linear", "y", "N"),
    "roll_x": ("angular", "x", "N*m"),
    "pitch_z": ("angular", "z", "N*m"),
    "yaw_y": ("angular", "y", "N*m"),
}


def finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def real_roll_tau_sign(real_root: Path, mode: str) -> float:
    """Resolve the roll sign for real CSVs without rewriting historical data.

    CSVs generated before ``right_hand_roll_v2`` used the opposite roll row
    in the force-to-wrench matrix.  New identifier output carries the matrix
    version in its fit JSON, so future data are used as-is.
    """
    if mode == "legacy":
        return -1.0
    if mode == "physical":
        return 1.0
    fit_path = real_root / REAL_DIRS["roll_x"] / "roll_x_009.fit.json"
    try:
        with fit_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)
    except (OSError, json.JSONDecodeError):
        metadata = {}
    return 1.0 if metadata.get("wrench_matrix_version") == "right_hand_roll_v2" else -1.0


def load_json_topic(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in read_csv(path):
        try:
            payload = json.loads(row["payload_json"])
        except (KeyError, json.JSONDecodeError):
            continue
        rows.append({"wall": finite(row.get("wall_time_sec")), "payload": payload})
    return rows


def find_topic(directory: Path, prefix: str) -> Path:
    matches = sorted(directory.glob(prefix + "__*.csv"))
    if not matches:
        raise FileNotFoundError(f"missing topic CSV `{prefix}` under {directory}")
    return matches[0]


def group_commands(path: Path) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for row in read_csv(path):
        trial = int(finite(row.get("trial_index"), -1))
        if trial < 0:
            continue
        row["wall"] = finite(row.get("wall_time_sec"))
        row["level_f"] = finite(row.get("level"))
        grouped.setdefault(trial, []).append(row)
    return grouped


def sample_component(payload: dict[str, Any], kind: str, component: str) -> float:
    value = payload.get(kind, {})
    return finite(value.get(component)) if isinstance(value, dict) else 0.0


def sim_offset(state_rows: list[dict[str, Any]]) -> float:
    offsets = []
    for row in state_rows:
        stamp = finite(row["payload"].get("stamp_sec"), math.nan)
        if math.isfinite(stamp):
            offsets.append(row["wall"] - stamp)
    return float(np.median(offsets)) if offsets else 0.0


def effective_wall(row: dict[str, Any], offset: float) -> float:
    stamp = finite(row["payload"].get("stamp_sec"), math.nan)
    return stamp + offset if math.isfinite(stamp) else row["wall"]


def steady_or_peak(values: np.ndarray, axis: str) -> float:
    if values.size == 0:
        return 0.0
    if axis == "heave_y" or axis in ("roll_x", "pitch_z"):
        return float(values[np.argmax(np.abs(values))])
    count = max(3, int(values.size * 0.25))
    return float(np.median(values[-count:]))


def rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else 0.0


def real_trial_metrics(
    real_rows: list[dict[str, str]], axis: str, trial: int, roll_tau_sign: float
) -> dict[str, Any]:
    rows = [
        row
        for row in real_rows
        if int(finite(row.get("trial_index"), -1)) == trial and row.get("phase") == "excitation"
    ]
    rate_field = REAL_RATE[axis]
    accel_field = REAL_ACCEL[axis]
    tau_field = REAL_TAU[axis]
    rate = np.asarray([finite(row.get(rate_field)) for row in rows], dtype=float)
    accel = np.asarray([finite(row.get(accel_field)) for row in rows], dtype=float)
    tau = np.asarray([finite(row.get(tau_field)) for row in rows], dtype=float)
    if axis == "roll_x":
        tau *= float(roll_tau_sign)
    target = np.asarray([finite(row.get("level")) for row in rows], dtype=float)
    metric_rate = steady_or_peak(rate, axis)
    metric_tau = steady_or_peak(tau, axis)
    return {
        "real_sample_count": int(len(rows)),
        "real_target_level": float(np.median(target)) if target.size else 0.0,
        "real_tau": metric_tau,
        "real_tau_sign": float(roll_tau_sign) if axis == "roll_x" else 1.0,
        "real_rate": metric_rate,
        "real_peak_abs_rate": float(np.max(np.abs(rate))) if rate.size else 0.0,
        "real_accel_rms": rms(accel),
    }


def sim_trial_metrics(
    directory: Path,
    axis: str,
    trial: int,
    command_rows: list[dict[str, Any]],
    applied_rows: list[dict[str, Any]],
    dvl_rows: list[dict[str, Any]],
    imu_rows: list[dict[str, Any]],
    offset: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    start = min(row["wall"] for row in command_rows)
    end = max(row["wall"] for row in command_rows)
    active_command_rows = [row for row in command_rows if row.get("phase") == "excitation"]
    active_start = min(row["wall"] for row in active_command_rows) if active_command_rows else start
    active_end = max(row["wall"] for row in active_command_rows) if active_command_rows else end
    # The topic is asynchronous. A 0.3 s guard keeps the final published state
    # of a trial while excluding the next reset/trial.
    end_guard = end + 0.3

    def select(rows: list[dict[str, Any]], lower: float, upper: float, guard: float = 0.15) -> list[dict[str, Any]]:
        return [
            row
            for row in rows
            if lower - guard <= effective_wall(row, offset) <= upper + guard
        ]

    applied = select(applied_rows, start, end, 0.0)
    dvl = select(dvl_rows, start, end, 0.0)
    imu = select(imu_rows, start, end, 0.0)
    active_applied = select(applied_rows, active_start, active_end, 0.15)
    active_dvl = select(dvl_rows, active_start, active_end, 0.15)
    active_imu = select(imu_rows, active_start, active_end, 0.15)
    wrench_kind, wrench_component, _unit = WRENCH_COMPONENT[axis]
    applied_time = np.asarray([effective_wall(row, offset) - start for row in applied], dtype=float)
    applied_value = np.asarray(
        [sample_component(row["payload"], wrench_kind, wrench_component) for row in applied], dtype=float
    )
    rate_kind, rate_component = ("linear", {"surge_x": "x", "sway_z": "z", "heave_y": "y"}.get(axis, "x")) \
        if axis in ("surge_x", "sway_z", "heave_y") else ("angular_velocity", {"roll_x": "x", "pitch_z": "z", "yaw_y": "y"}[axis])
    rate_rows = dvl if axis in ("surge_x", "sway_z", "heave_y") else imu
    active_rate_rows = active_dvl if axis in ("surge_x", "sway_z", "heave_y") else active_imu
    rate_time = np.asarray([effective_wall(row, offset) - start for row in rate_rows], dtype=float)
    rate_value = np.asarray([sample_component(row["payload"], rate_kind, rate_component) for row in rate_rows], dtype=float)
    active_rate_value = np.asarray(
        [sample_component(row["payload"], rate_kind, rate_component) for row in active_rate_rows], dtype=float
    )
    accel_time = np.asarray([effective_wall(row, offset) - start for row in imu], dtype=float)
    accel_component = {"surge_x": "x", "sway_z": "z", "heave_y": "y"}.get(axis)
    if accel_component is None:
        accel_value = np.gradient(rate_value, rate_time, edge_order=1) if rate_value.size > 1 else np.zeros_like(rate_value)
        active_accel_value = np.gradient(active_rate_value, edge_order=1) if active_rate_value.size > 1 else np.zeros_like(active_rate_value)
    else:
        accel_value = np.asarray(
            [sample_component(row["payload"], "linear_acceleration", accel_component) for row in imu], dtype=float
        )
        active_accel_value = np.asarray(
            [sample_component(row["payload"], "linear_acceleration", accel_component) for row in active_imu], dtype=float
        )
        if axis == "heave_y":
            # VehicleRosBridge publishes specific force, so remove +g at rest.
            accel_value = accel_value - 9.81
            active_accel_value = active_accel_value - 9.81

    level = float(np.median([row["level_f"] for row in command_rows]))
    active_applied_value = np.asarray(
        [sample_component(row["payload"], wrench_kind, wrench_component) for row in active_applied], dtype=float
    )
    metric_rate = steady_or_peak(active_rate_value, axis)
    metric_tau = steady_or_peak(active_applied_value, axis)
    metrics = {
        "trial_index": trial,
        "command_level": level,
        "sim_applied_wrench": metric_tau,
        "sim_rate": metric_rate,
        "sim_peak_abs_rate": float(np.max(np.abs(rate_value))) if rate_value.size else 0.0,
        "sim_accel_rms": rms(active_accel_value),
        "sim_applied_samples": int(active_applied_value.size),
        "sim_rate_samples": int(active_rate_value.size),
        "sim_time_offset_sec": offset,
    }
    series = {
        "applied_time": applied_time,
        "applied_value": applied_value,
        "rate_time": rate_time,
        "rate_value": rate_value,
        "accel_time": accel_time,
        "accel_value": accel_value,
    }
    return metrics, series


def plot_axis(directory: Path, axis: str, summaries: list[dict[str, Any]], series: dict[int, dict[str, np.ndarray]], units: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False, constrained_layout=True)
    for summary in summaries:
        trial = int(summary["trial_index"])
        data = series.get(trial, {})
        label = f"trial {trial}, cmd={summary['command_level']:+g}"
        if data.get("rate_time", np.empty(0)).size:
            axes[0].plot(data["rate_time"], data["rate_value"], label=label, lw=1.0)
        if data.get("accel_time", np.empty(0)).size:
            axes[1].plot(data["accel_time"], data["accel_value"], label=label, lw=0.9)
    axes[0].set_ylabel(f"{axis} rate ({'m/s' if axis in ('surge_x','sway_z','heave_y') else 'rad/s'})")
    axes[1].set_ylabel(f"{axis} acceleration ({'m/s^2' if axis in ('surge_x','sway_z','heave_y') else 'rad/s^2'})")
    axes[1].set_xlabel("time from trial command start (s)")
    axes[0].grid(True, alpha=0.3)
    axes[1].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2)
    fig.suptitle(f"Unity Fossen {axis}: velocity/rate and acceleration")
    fig.savefig(directory / "velocity_acceleration_vs_time.png", dpi=180)
    plt.close(fig)


def plot_tau_rate(directory: Path, axis: str, summaries: list[dict[str, Any]], units: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 6), constrained_layout=True)
    real = [(abs(s["real_tau"]), abs(s["real_rate"])) for s in summaries if s["real_sample_count"] > 0]
    sim = [(abs(s["sim_applied_wrench"]), abs(s["sim_rate"])) for s in summaries if s["sim_applied_samples"] > 0]
    if real:
        x, y = zip(*sorted(real))
        ax.plot(x, y, "o-", label="real tau_fit -> response", color="tab:blue")
    if sim:
        x, y = zip(*sorted(sim))
        ax.plot(x, y, "s-", label="Unity applied wrench -> response", color="tab:orange")
    ax.set_xlabel(f"absolute applied wrench ({units})")
    ax.set_ylabel("absolute steady/peak rate")
    ax.set_title(f"{axis}: applied wrench versus response")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.savefig(directory / "applied_wrench_vs_response.png", dpi=180)
    plt.close(fig)


def plot_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    for axis in AXES:
        selected = [row for row in rows if row["axis"] == axis and row["sim_applied_samples"] > 0]
        if not selected:
            continue
        axes[0].scatter(
            [abs(row["real_tau"]) for row in selected],
            [abs(row["sim_applied_wrench"]) for row in selected],
            label=axis,
        )
        axes[1].scatter(
            [abs(row["real_rate"]) for row in selected],
            [abs(row["sim_rate"]) for row in selected],
            label=axis,
        )
    axes[0].set_xlabel("real |tau_fit|")
    axes[0].set_ylabel("Unity |applied wrench|")
    axes[0].set_title("Input wrench cross-check")
    axes[1].set_xlabel("real |rate|")
    axes[1].set_ylabel("Unity |rate|")
    axes[1].set_title("Response cross-check")
    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze Unity Fossen hydrodynamic replay data against real identification CSVs.")
    parser.add_argument("--unity-root", type=Path, required=True)
    parser.add_argument("--real-root", type=Path, required=True)
    parser.add_argument(
        "--real-roll-sign",
        choices=("auto", "legacy", "physical"),
        default="auto",
        help="Interpret historical roll tau_fit_mx_nm with the old or right-hand-rule sign.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roll_tau_sign = real_roll_tau_sign(args.real_root, args.real_roll_sign)
    all_rows: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "unity_root": str(args.unity_root),
        "real_root": str(args.real_root),
        "real_roll_sign_mode": args.real_roll_sign,
        "real_roll_tau_sign": roll_tau_sign,
        "axes": {},
    }
    for axis in AXES:
        directory = args.unity_root / axis / f"{axis}_001"
        if not directory.exists():
            continue
        commands = group_commands(directory / "replay_commands.csv")
        real_path = args.real_root / REAL_DIRS[axis] / f"{Path(REAL_DIRS[axis]).name}.csv"
        real_rows = read_csv(real_path)
        applied_rows = load_json_topic(find_topic(directory, "sim_thruster_applied_wrench"))
        dvl_rows = load_json_topic(find_topic(directory, "sim_controller_dvl"))
        imu_rows = load_json_topic(find_topic(directory, "sim_controller_imu"))
        offset = sim_offset(imu_rows)
        summaries: list[dict[str, Any]] = []
        series: dict[int, dict[str, np.ndarray]] = {}
        units = WRENCH_COMPONENT[axis][2]
        for trial, command_rows in sorted(commands.items()):
            force_columns = (
                "force_cmd_V_LF_n", "force_cmd_V_LB_n", "force_cmd_V_RB_n", "force_cmd_V_RF_n",
                "force_cmd_H_LF_n", "force_cmd_H_LB_n", "force_cmd_H_RB_n", "force_cmd_H_RF_n",
            )
            if all(
                abs(finite(row.get(column))) < 1e-8
                for row in command_rows
                for column in force_columns
            ):
                continue
            real_metric = real_trial_metrics(real_rows, axis, trial, roll_tau_sign)
            sim_metric, sim_series = sim_trial_metrics(
                directory, axis, trial, command_rows, applied_rows, dvl_rows, imu_rows, offset
            )
            summary = {"axis": axis, **real_metric, **sim_metric}
            summary["rate_error_abs"] = abs(summary["sim_rate"]) - abs(summary["real_rate"])
            summary["tau_error_abs"] = abs(summary["sim_applied_wrench"]) - abs(summary["real_tau"])
            summaries.append(summary)
            all_rows.append(summary)
            series[trial] = sim_series
        output_csv = directory / "unity_vs_real_trial_summary.csv"
        if summaries:
            with output_csv.open("w", newline="", encoding="utf-8") as stream:
                fields = list(summaries[0].keys())
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(summaries)
            plot_axis(directory, axis, summaries, series, units)
            plot_tau_rate(directory, axis, summaries, units)
        report["axes"][axis] = {
            "directory": str(directory),
            "time_offset_sec": offset,
            "trial_count": len(summaries),
            "summary_csv": str(output_csv),
        }

    summary_csv = args.unity_root / "unity_fossen_vs_real_summary.csv"
    if all_rows:
        with summary_csv.open("w", newline="", encoding="utf-8") as stream:
            fields = list(all_rows[0].keys())
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(all_rows)
        plot_summary(args.unity_root / "all_axes_wrench_and_response.png", all_rows)
    report["summary_csv"] = str(summary_csv)
    report["notes"] = [
        "Unity wrench is /sim/finsrov/debug/thruster_applied_wrench, not the ROS command array.",
        "Real tau uses tau_fit_* fields from the real identifier, which are RPM/curve-derived when RPM is fresh.",
        "heave, roll, and pitch use peak response because their excitation is pulse/coast rather than a long steady step.",
        "Unity IMU heave acceleration has 9.81 m/s^2 specific-force offset removed for plots.",
        "Historical real roll tau_fit_mx_nm is auto-inverted unless its fit JSON declares right_hand_roll_v2.",
    ]
    (args.unity_root / "analysis_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=True, sort_keys=True))


if __name__ == "__main__":
    main()
