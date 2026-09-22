"""Held-out, axiswise validation of a Unity Fossen replay against pool data.

The command deliberately separates three stages:

* ``prepare`` freezes the real CSV and Unity hydrodynamics profile in a manifest;
* the existing ``replay_unity_hydrodynamics`` node replays RPM-derived forces;
* ``analyze`` audits the applied Unity wrench and compares the phase-aligned
  velocity/rate responses without refitting any coefficient.

It is not a generic six-degree-of-freedom digital-twin score.  Its scope is
the axiswise diagonal Fossen profile used by the validation run.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


THRUSTER_NAMES = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")
WRENCH_COLUMNS = (
    "tau_fit_fx_n",
    "tau_fit_fy_n",
    "tau_fit_fz_n",
    "tau_fit_mx_nm",
    "tau_fit_my_yaw_nm",
    "tau_fit_mz_nm",
)
NU_COLUMNS = (
    "nu_x_mps",
    "nu_y_mps",
    "nu_z_mps",
    "nu_roll_x_radps",
    "nu_yaw_radps",
    "nu_pitch_z_radps",
)


@dataclass(frozen=True)
class AxisSpec:
    name: str
    response_index: int
    wrench_index: int
    response_unit: str
    wrench_unit: str
    sim_topic: str
    sim_component: str
    coupling_indices: tuple[int, ...]


AXIS_SPECS = {
    "surge_x": AxisSpec("surge_x", 0, 0, "m/s", "N", "dvl", "x", (1, 2)),
    "heave_y": AxisSpec("heave_y", 1, 1, "m/s", "N", "dvl", "y", (0, 2)),
    "sway_z": AxisSpec("sway_z", 2, 2, "m/s", "N", "dvl", "z", (0, 1)),
    "yaw_y": AxisSpec("yaw_y", 4, 4, "rad/s", "N m", "imu", "y", (3, 5)),
}


@dataclass(frozen=True)
class Criteria:
    input_wrench_p95_max: float = 0.05
    response_nrmse_max: float = 0.20
    coast_nrmse_max: float = 0.30
    initial_slope_rel_max: float = 0.30
    steady_state_rel_max: float = 0.20
    coast_time_rel_max: float = 0.25
    analysis_rate_hz: float = 50.0
    initial_window_sec: float = 0.50
    steady_fraction: float = 0.25
    require_coast: bool = True
    wrench_floor: float = 0.05


@dataclass
class RealTrial:
    trial_index: int
    repeat_index: int
    axis: str
    level: float
    time: np.ndarray
    phase: np.ndarray
    wrench: np.ndarray
    nu: np.ndarray


@dataclass
class SimTrial:
    trial_index: int
    level: float
    command_time: np.ndarray
    phase: np.ndarray


@dataclass
class TimeSeries:
    time: np.ndarray
    values: np.ndarray


def _finite(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_commit(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")


def _unique_sorted(time: np.ndarray, values: np.ndarray) -> TimeSeries:
    if time.size == 0:
        return TimeSeries(np.empty(0), np.empty((0, values.shape[1] if values.ndim == 2 else 1)))
    order = np.argsort(time)
    sorted_time = np.asarray(time[order], dtype=float)
    sorted_values = np.asarray(values[order], dtype=float)
    keep = np.r_[True, np.diff(sorted_time) > 1e-9]
    return TimeSeries(sorted_time[keep], sorted_values[keep])


def _infer_axis(rows: Iterable[dict[str, str]], requested: str | None) -> str:
    if requested is not None:
        if requested not in AXIS_SPECS:
            raise ValueError(f"unsupported axis `{requested}`; use one of {list(AXIS_SPECS)}")
        return requested
    axes = sorted({str(row.get("axis", "")).strip() for row in rows if str(row.get("axis", "")).strip() in AXIS_SPECS})
    if len(axes) != 1:
        raise ValueError(f"cannot infer one validation axis from {axes}; pass --axis explicitly")
    return axes[0]


def _load_real_trials(
    csv_path: Path,
    requested_axis: str | None,
    requested_trials: set[int] | None = None,
) -> tuple[str, dict[int, RealTrial]]:
    rows = _read_rows(csv_path)
    if not rows:
        raise ValueError(f"empty real CSV: {csv_path}")
    axis = _infer_axis(rows, requested_axis)
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        trial_index = int(_finite(row.get("trial_index"), -1.0))
        if (
            trial_index < 0
            or str(row.get("axis", "")).strip() != axis
            or (requested_trials is not None and trial_index not in requested_trials)
        ):
            continue
        grouped.setdefault(trial_index, []).append(row)
    if not grouped:
        raise ValueError(f"no `{axis}` trials in {csv_path}")

    trials: dict[int, RealTrial] = {}
    for trial_index, trial_rows in grouped.items():
        trial_rows.sort(key=lambda row: _finite(row.get("time_sec")))
        time = np.asarray([_finite(row.get("time_sec")) for row in trial_rows], dtype=float)
        time -= time[0]
        wrench = np.asarray(
            [[_finite(row.get(column)) for column in WRENCH_COLUMNS] for row in trial_rows], dtype=float
        )
        nu = np.asarray([[_finite(row.get(column)) for column in NU_COLUMNS] for row in trial_rows], dtype=float)
        levels = np.asarray([_finite(row.get("level")) for row in trial_rows], dtype=float)
        repeats = [int(_finite(row.get("repeat_index"), 1.0)) for row in trial_rows]
        trials[trial_index] = RealTrial(
            trial_index=trial_index,
            repeat_index=max(1, int(round(float(np.median(repeats))))) if repeats else 1,
            axis=axis,
            level=float(np.median(levels)),
            time=time,
            phase=np.asarray([str(row.get("phase", "unknown")) for row in trial_rows], dtype=object),
            wrench=wrench,
            nu=nu,
        )
    return axis, trials


def _find_topic(sim_dir: Path, prefix: str) -> Path:
    matches = sorted(sim_dir.glob(f"{prefix}__*.csv"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one `{prefix}__*.csv` under {sim_dir}, found {len(matches)}")
    return matches[0]


def _read_topic_series(path: Path, kind: str) -> TimeSeries:
    """Read Unity diagnostics on the recorder's common ROS wall-clock timeline.

    Unity message headers carry simulation time, whose offset to ROS wall time
    is not constant when the editor frame rate changes.  The replay command is
    intentionally timestamped in recorder wall time, so using a per-topic
    median header-stamp offset distorts phase alignment.  Keep the original
    Unity header stamp in the raw CSV for diagnostics, but use the recorder's
    callback time for all command--wrench--state comparisons.
    """
    parsed: list[tuple[dict[str, str], dict[str, Any]]] = []
    for row in _read_rows(path):
        try:
            payload = json.loads(row["payload_json"])
        except (KeyError, json.JSONDecodeError):
            continue
        if isinstance(payload, dict):
            parsed.append((row, payload))
    time: list[float] = []
    values: list[list[float]] = []
    for row, payload in parsed:
        if kind == "dvl":
            vector = payload.get("linear", {})
            value = [_finite(vector.get(key)) for key in ("x", "y", "z")] if isinstance(vector, dict) else [0.0] * 3
        elif kind == "imu":
            vector = payload.get("angular_velocity", {})
            value = [_finite(vector.get(key)) for key in ("x", "y", "z")] if isinstance(vector, dict) else [0.0] * 3
        elif kind == "wrench":
            linear = payload.get("linear", {})
            angular = payload.get("angular", {})
            value = [
                *([_finite(linear.get(key)) for key in ("x", "y", "z")] if isinstance(linear, dict) else [0.0] * 3),
                *([_finite(angular.get(key)) for key in ("x", "y", "z")] if isinstance(angular, dict) else [0.0] * 3),
            ]
        else:
            raise ValueError(f"unsupported topic kind `{kind}`")
        time.append(_finite(row.get("wall_time_sec")))
        values.append(value)
    return _unique_sorted(np.asarray(time, dtype=float), np.asarray(values, dtype=float))


def _load_sim_trials(sim_dir: Path) -> dict[int, SimTrial]:
    command_path = sim_dir / "replay_commands.csv"
    if not command_path.is_file():
        raise FileNotFoundError(f"missing replay command log: {command_path}")
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in _read_rows(command_path):
        trial_index = int(_finite(row.get("trial_index"), -1.0))
        if trial_index >= 0:
            grouped.setdefault(trial_index, []).append(row)
    trials: dict[int, SimTrial] = {}
    for trial_index, rows in grouped.items():
        rows.sort(key=lambda row: _finite(row.get("wall_time_sec")))
        time = np.asarray([_finite(row.get("wall_time_sec")) for row in rows], dtype=float)
        levels = np.asarray([_finite(row.get("level")) for row in rows], dtype=float)
        trials[trial_index] = SimTrial(
            trial_index=trial_index,
            level=float(np.median(levels)),
            command_time=time,
            phase=np.asarray([str(row.get("phase", "unknown")) for row in rows], dtype=object),
        )
    if not trials:
        raise ValueError(f"no replay trials in {command_path}")
    return trials


def _phase_bounds(time: np.ndarray, phase: np.ndarray, name: str) -> tuple[float, float] | None:
    mask = phase == name
    if not np.any(mask):
        return None
    selected = time[mask]
    return float(np.min(selected)), float(np.max(selected))


def _phase_values(time: np.ndarray, values: np.ndarray, phase: np.ndarray, name: str) -> TimeSeries | None:
    mask = phase == name
    if np.count_nonzero(mask) < 3:
        return None
    selected_time = time[mask]
    return _unique_sorted(selected_time - selected_time[0], values[mask])


def _sim_phase_values(series: TimeSeries, trial: SimTrial, name: str) -> TimeSeries | None:
    bounds = _phase_bounds(trial.command_time, trial.phase, name)
    if bounds is None:
        return None
    lower, upper = bounds
    mask = (series.time >= lower) & (series.time <= upper)
    if np.count_nonzero(mask) < 3:
        return None
    time = series.time[mask]
    return _unique_sorted(time - lower, series.values[mask])


def _interpolate_pair(real: TimeSeries | None, sim: TimeSeries | None, rate_hz: float) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    if real is None or sim is None or real.time.size < 3 or sim.time.size < 3:
        return None
    end = min(float(real.time[-1]), float(sim.time[-1]))
    if end <= 0.0:
        return None
    count = max(3, int(math.floor(end * max(rate_hz, 1.0))) + 1)
    time = np.linspace(0.0, end, count)
    real_values = np.column_stack([np.interp(time, real.time, real.values[:, col]) for col in range(real.values.shape[1])])
    sim_values = np.column_stack([np.interp(time, sim.time, sim.values[:, col]) for col in range(sim.values.shape[1])])
    return time, real_values, sim_values


def _rmse(values: np.ndarray) -> float | None:
    if values.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(values))))


def _nrmse(real: np.ndarray, sim: np.ndarray) -> tuple[float | None, float | None, float | None]:
    if real.size < 3 or sim.size != real.size:
        return None, None, None
    rmse = _rmse(sim - real)
    assert rmse is not None
    span = float(np.percentile(real, 95.0) - np.percentile(real, 5.0))
    if span <= 1e-9:
        return rmse, None, span
    return rmse, rmse / span, span


def _slope(time: np.ndarray, value: np.ndarray, window_sec: float) -> float | None:
    mask = time <= min(float(time[-1]), max(window_sec, 0.0))
    if np.count_nonzero(mask) < 3:
        return None
    return float(np.polyfit(time[mask], value[mask], 1)[0])


def _steady(value: np.ndarray, fraction: float) -> float | None:
    if value.size < 3:
        return None
    count = max(3, int(math.ceil(value.size * min(max(fraction, 0.05), 1.0))))
    return float(np.median(value[-count:]))


def _relative_error(value: float | None, reference: float | None, floor: float = 1e-9) -> float | None:
    if value is None or reference is None:
        return None
    return abs(value - reference) / max(abs(reference), floor)


def _t10(time: np.ndarray, value: np.ndarray) -> float | None:
    if value.size < 5:
        return None
    count = max(2, int(math.ceil(value.size * 0.10)))
    terminal = float(np.median(value[-count:]))
    initial = float(np.median(value[:count]))
    amplitude = abs(initial - terminal)
    if amplitude <= 1e-9:
        return None
    residual = np.abs(value - terminal)
    crossed = np.flatnonzero(residual <= 0.10 * amplitude)
    return float(time[crossed[0]]) if crossed.size else None


def _coupling_ratio(values: np.ndarray, response_index: int, coupling_indices: tuple[int, ...]) -> float | None:
    if values.shape[0] < 3 or not coupling_indices:
        return None
    primary = float(np.sqrt(np.mean(np.square(values[:, response_index]))))
    if primary <= 1e-9:
        return None
    leakage = float(np.sqrt(np.mean(np.square(values[:, coupling_indices]))))
    return leakage / primary


def _bootstrap_median(values: list[float], seed: int = 20260903) -> dict[str, float | int | None]:
    finite = np.asarray([value for value in values if math.isfinite(value)], dtype=float)
    if finite.size == 0:
        return {"count": 0, "median": None, "bootstrap_ci95_low": None, "bootstrap_ci95_high": None}
    if finite.size == 1:
        value = float(finite[0])
        return {"count": 1, "median": value, "bootstrap_ci95_low": None, "bootstrap_ci95_high": None}
    rng = np.random.default_rng(seed)
    sample_index = rng.integers(0, finite.size, size=(2000, finite.size))
    medians = np.median(finite[sample_index], axis=1)
    return {
        "count": int(finite.size),
        "median": float(np.median(finite)),
        "bootstrap_ci95_low": float(np.percentile(medians, 2.5)),
        "bootstrap_ci95_high": float(np.percentile(medians, 97.5)),
    }


def _phase_pair_for_response(
    real_trial: RealTrial,
    sim_trial: SimTrial,
    sim_series: TimeSeries,
    spec: AxisSpec,
    phase: str,
    rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    real = _phase_values(real_trial.time, real_trial.nu[:, [spec.response_index]], real_trial.phase, phase)
    sim = _sim_phase_values(sim_series, sim_trial, phase)
    if sim is not None:
        component = {"x": 0, "y": 1, "z": 2}[spec.sim_component]
        sim = TimeSeries(sim.time, sim.values[:, [component]])
    return _interpolate_pair(real, sim, rate_hz)


def _concatenate_response_pairs(
    *pairs: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Join phase-aligned response pairs while preserving their sample weighting.

    Excitation and coast are independently phase-aligned because their command
    timing may have different offsets.  For a combined response score we retain
    those alignments, concatenate their uniformly sampled response values, and
    weight each phase by its duration.  The returned time vector is only a
    phase-local plotting convenience; RMSE uses the concatenated values.
    """
    if not pairs or any(pair is None for pair in pairs):
        return None
    valid_pairs = [pair for pair in pairs if pair is not None]
    times: list[np.ndarray] = []
    real_values: list[np.ndarray] = []
    sim_values: list[np.ndarray] = []
    offset = 0.0
    for time, real, sim in valid_pairs:
        if time.size < 3 or real.shape != sim.shape or real.shape[0] != time.size:
            return None
        times.append(time + offset)
        real_values.append(real)
        sim_values.append(sim)
        offset = float(times[-1][-1])
    return np.concatenate(times), np.vstack(real_values), np.vstack(sim_values)


def _phase_pair_for_wrench(
    real_trial: RealTrial,
    sim_trial: SimTrial,
    sim_wrench: TimeSeries,
    phase: str,
    rate_hz: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    real = _phase_values(real_trial.time, real_trial.wrench, real_trial.phase, phase)
    sim = _sim_phase_values(sim_wrench, sim_trial, phase)
    return _interpolate_pair(real, sim, rate_hz)


def _wrench_onset(series: TimeSeries | None, component: int, floor: float) -> float | None:
    """Return the 10%-of-step onset time of one applied-wrench component."""
    if series is None or series.time.size < 5:
        return None
    value = np.abs(series.values[:, component])
    count = max(2, int(math.ceil(value.size * 0.10)))
    baseline = float(np.median(value[:count]))
    terminal = float(np.median(value[-count:]))
    amplitude = abs(terminal - baseline)
    if amplitude < floor:
        return None
    crossed = np.flatnonzero(value >= baseline + 0.10 * amplitude)
    return float(series.time[crossed[0]]) if crossed.size else None


def _rebase_at(series: TimeSeries, onset: float) -> TimeSeries | None:
    mask = series.time >= onset
    if np.count_nonzero(mask) < 3:
        return None
    baseline = np.asarray(
        [np.interp(onset, series.time, series.values[:, column]) for column in range(series.values.shape[1])],
        dtype=float,
    )
    return _unique_sorted(series.time[mask] - onset, series.values[mask] - baseline)


def _wrench_onset_aligned_response_pair(
    real_trial: RealTrial,
    sim_trial: SimTrial,
    sim_response: TimeSeries,
    sim_wrench: TimeSeries,
    spec: AxisSpec,
    criteria: Criteria,
) -> tuple[tuple[np.ndarray, np.ndarray, np.ndarray] | None, float | None, float | None]:
    """Compare response changes after each system's measured wrench onset.

    The phase-start comparison remains the execution-chain diagnostic.  This
    companion pairing removes separately measured actuator delays so the
    resulting \u0394-response is attributable more directly to the hydrodynamic
    model rather than to ROS/Unity thruster timing.
    """
    real_response = _phase_values(
        real_trial.time, real_trial.nu[:, [spec.response_index]], real_trial.phase, "excitation"
    )
    sim_response_phase = _sim_phase_values(sim_response, sim_trial, "excitation")
    if sim_response_phase is not None:
        component = {"x": 0, "y": 1, "z": 2}[spec.sim_component]
        sim_response_phase = TimeSeries(sim_response_phase.time, sim_response_phase.values[:, [component]])
    real_wrench = _phase_values(real_trial.time, real_trial.wrench, real_trial.phase, "excitation")
    sim_wrench_phase = _sim_phase_values(sim_wrench, sim_trial, "excitation")
    real_onset = _wrench_onset(real_wrench, spec.wrench_index, criteria.wrench_floor)
    sim_onset = _wrench_onset(sim_wrench_phase, spec.wrench_index, criteria.wrench_floor)
    if real_response is None or sim_response_phase is None or real_onset is None or sim_onset is None:
        return None, real_onset, sim_onset
    return (
        _interpolate_pair(_rebase_at(real_response, real_onset), _rebase_at(sim_response_phase, sim_onset), criteria.analysis_rate_hz),
        real_onset,
        sim_onset,
    )


def _metric_or_none(value: float | None) -> float | None:
    return float(value) if value is not None and math.isfinite(value) else None


def _trial_metrics(
    real_trial: RealTrial,
    sim_trial: SimTrial,
    spec: AxisSpec,
    sim_dvl: TimeSeries,
    sim_imu: TimeSeries,
    sim_wrench: TimeSeries,
    criteria: Criteria,
) -> tuple[dict[str, Any], dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None]]:
    sim_response = sim_dvl if spec.sim_topic == "dvl" else sim_imu
    response_excitation = _phase_pair_for_response(
        real_trial, sim_trial, sim_response, spec, "excitation", criteria.analysis_rate_hz
    )
    response_coast = _phase_pair_for_response(
        real_trial, sim_trial, sim_response, spec, "coast", criteria.analysis_rate_hz
    )
    wrench_excitation = _phase_pair_for_wrench(
        real_trial, sim_trial, sim_wrench, "excitation", criteria.analysis_rate_hz
    )
    onset_aligned_response, real_wrench_onset, sim_wrench_onset = _wrench_onset_aligned_response_pair(
        real_trial, sim_trial, sim_response, sim_wrench, spec, criteria
    )

    result: dict[str, Any] = {
        "trial_index": real_trial.trial_index,
        "repeat_index": real_trial.repeat_index,
        "axis": spec.name,
        "real_level": real_trial.level,
        "sim_level": sim_trial.level,
        "input_wrench_relative_p50": None,
        "input_wrench_relative_p95": None,
        "real_wrench_onset_sec": _metric_or_none(real_wrench_onset),
        "sim_wrench_onset_sec": _metric_or_none(sim_wrench_onset),
        "wrench_onset_lag_sec": (
            _metric_or_none(sim_wrench_onset - real_wrench_onset)
            if real_wrench_onset is not None and sim_wrench_onset is not None
            else None
        ),
        "excitation_rmse": None,
        "excitation_nrmse": None,
        "excitation_span": None,
        "excitation_plus_coast_rmse": None,
        "excitation_plus_coast_nrmse": None,
        "excitation_plus_coast_span": None,
        "onset_aligned_delta_response_rmse": None,
        "onset_aligned_delta_response_nrmse": None,
        "onset_aligned_delta_response_span": None,
        "initial_slope_real": None,
        "initial_slope_sim": None,
        "initial_slope_relative_error": None,
        "steady_state_real": None,
        "steady_state_sim": None,
        "steady_state_relative_error": None,
        "coast_rmse": None,
        "coast_nrmse": None,
        "coast_t10_real_sec": None,
        "coast_t10_sim_sec": None,
        "coast_t10_relative_error": None,
        "real_coupling_ratio": None,
        "sim_coupling_ratio": None,
        "classification": "invalid",
        "reasons": [],
    }
    reasons: list[str] = []

    if wrench_excitation is None:
        reasons.append("missing_excitation_wrench")
    else:
        _, real_wrench, unity_wrench = wrench_excitation
        active = np.linalg.norm(real_wrench, axis=1) >= criteria.wrench_floor
        if np.count_nonzero(active) < 3:
            reasons.append("insufficient_nonzero_wrench_samples")
        else:
            denominator = np.linalg.norm(real_wrench[active], axis=1) + 1e-12
            relative = np.linalg.norm(unity_wrench[active] - real_wrench[active], axis=1) / denominator
            result["input_wrench_relative_p50"] = float(np.percentile(relative, 50.0))
            result["input_wrench_relative_p95"] = float(np.percentile(relative, 95.0))
            if result["input_wrench_relative_p95"] > criteria.input_wrench_p95_max:
                reasons.append("input_wrench_mismatch")

    if response_excitation is None:
        reasons.append("missing_excitation_response")
    else:
        time, real_response_matrix, sim_response_matrix = response_excitation
        real_response = real_response_matrix[:, 0]
        sim_response_value = sim_response_matrix[:, 0]
        rmse, nrmse, span = _nrmse(real_response, sim_response_value)
        result["excitation_rmse"] = _metric_or_none(rmse)
        result["excitation_nrmse"] = _metric_or_none(nrmse)
        result["excitation_span"] = _metric_or_none(span)
        if nrmse is None:
            reasons.append("insufficient_excitation_response_span")
        elif nrmse > criteria.response_nrmse_max:
            reasons.append("excitation_response_mismatch")
        slope_real = _slope(time, real_response, criteria.initial_window_sec)
        slope_sim = _slope(time, sim_response_value, criteria.initial_window_sec)
        result["initial_slope_real"] = _metric_or_none(slope_real)
        result["initial_slope_sim"] = _metric_or_none(slope_sim)
        result["initial_slope_relative_error"] = _metric_or_none(_relative_error(slope_sim, slope_real))
        if result["initial_slope_relative_error"] is not None and result["initial_slope_relative_error"] > criteria.initial_slope_rel_max:
            reasons.append("initial_slope_mismatch")
        steady_real = _steady(real_response, criteria.steady_fraction)
        steady_sim = _steady(sim_response_value, criteria.steady_fraction)
        result["steady_state_real"] = _metric_or_none(steady_real)
        result["steady_state_sim"] = _metric_or_none(steady_sim)
        result["steady_state_relative_error"] = _metric_or_none(_relative_error(steady_sim, steady_real))
        if result["steady_state_relative_error"] is not None and result["steady_state_relative_error"] > criteria.steady_state_rel_max:
            reasons.append("steady_state_mismatch")
        real_phase = _phase_values(real_trial.time, real_trial.nu, real_trial.phase, "excitation")
        sim_phase = _sim_phase_values(sim_response, sim_trial, "excitation")
        if real_phase is not None:
            result["real_coupling_ratio"] = _metric_or_none(
                _coupling_ratio(real_phase.values, spec.response_index, spec.coupling_indices)
            )
        if sim_phase is not None:
            sim_primary = {"x": 0, "y": 1, "z": 2}[spec.sim_component]
            sim_coupling = tuple(index for index in range(3) if index != sim_primary)
            result["sim_coupling_ratio"] = _metric_or_none(
                _coupling_ratio(sim_phase.values, sim_primary, sim_coupling)
            )

    if onset_aligned_response is not None:
        _, real_delta, sim_delta = onset_aligned_response
        rmse, nrmse, span = _nrmse(real_delta[:, 0], sim_delta[:, 0])
        result["onset_aligned_delta_response_rmse"] = _metric_or_none(rmse)
        result["onset_aligned_delta_response_nrmse"] = _metric_or_none(nrmse)
        result["onset_aligned_delta_response_span"] = _metric_or_none(span)

    if response_coast is None:
        if criteria.require_coast:
            reasons.append("missing_coast_response")
    else:
        coast_time, real_coast_matrix, sim_coast_matrix = response_coast
        real_coast = real_coast_matrix[:, 0]
        sim_coast = sim_coast_matrix[:, 0]
        coast_rmse, coast_nrmse, _ = _nrmse(real_coast, sim_coast)
        result["coast_rmse"] = _metric_or_none(coast_rmse)
        result["coast_nrmse"] = _metric_or_none(coast_nrmse)
        if coast_nrmse is None:
            reasons.append("insufficient_coast_response_span")
        elif coast_nrmse > criteria.coast_nrmse_max:
            reasons.append("coast_response_mismatch")
        t10_real = _t10(coast_time, real_coast)
        t10_sim = _t10(coast_time, sim_coast)
        result["coast_t10_real_sec"] = _metric_or_none(t10_real)
        result["coast_t10_sim_sec"] = _metric_or_none(t10_sim)
        result["coast_t10_relative_error"] = _metric_or_none(_relative_error(t10_sim, t10_real))
        if result["coast_t10_relative_error"] is not None and result["coast_t10_relative_error"] > criteria.coast_time_rel_max:
            reasons.append("coast_time_mismatch")

    excitation_plus_coast = _concatenate_response_pairs(response_excitation, response_coast)
    if excitation_plus_coast is not None:
        _, real_active, sim_active = excitation_plus_coast
        active_rmse, active_nrmse, active_span = _nrmse(real_active[:, 0], sim_active[:, 0])
        result["excitation_plus_coast_rmse"] = _metric_or_none(active_rmse)
        result["excitation_plus_coast_nrmse"] = _metric_or_none(active_nrmse)
        result["excitation_plus_coast_span"] = _metric_or_none(active_span)

    result["reasons"] = reasons
    if any(reason.startswith("missing_") or reason.startswith("insufficient_") for reason in reasons):
        result["classification"] = "incomplete"
    elif reasons:
        result["classification"] = "mismatch"
    else:
        result["classification"] = "consistent_with_frozen_profile"
    return result, {
        "excitation_response": response_excitation,
        "onset_aligned_response": onset_aligned_response,
        "coast_response": response_coast,
        "excitation_wrench": wrench_excitation,
    }


def _plot_trial(
    path: Path,
    spec: AxisSpec,
    trial: dict[str, Any],
    series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray] | None],
    model_label: str,
) -> None:
    excitation_response = series["excitation_response"]
    coast_response = series["coast_response"]
    excitation_wrench = series["excitation_wrench"]
    fig, axes = plt.subplots(2, 1, figsize=(9, 6.5), constrained_layout=True)
    if excitation_wrench is not None:
        time, real, sim = excitation_wrench
        axes[0].plot(time, real[:, spec.wrench_index], label="real RPM-derived wrench", color="tab:blue")
        axes[0].plot(time, sim[:, spec.wrench_index], label="Unity applied wrench", color="tab:orange", alpha=0.85)
    axes[0].set_ylabel(f"{spec.name} wrench [{spec.wrench_unit}]")
    axes[0].grid(alpha=0.3)
    axes[0].legend(loc="best", fontsize=8)
    if excitation_response is not None:
        time, real, sim = excitation_response
        axes[1].plot(time, real[:, 0], label="real", color="tab:blue")
        axes[1].plot(time, sim[:, 0], label=model_label, color="tab:orange", alpha=0.85)
    if coast_response is not None:
        time, real, sim = coast_response
        coast_offset = float(axes[1].lines[-1].get_xdata()[-1]) if axes[1].lines else 0.0
        axes[1].plot(time + coast_offset, real[:, 0], color="tab:blue", ls="--", label="real coast")
        axes[1].plot(time + coast_offset, sim[:, 0], color="tab:orange", ls="--", label="Unity coast")
    axes[1].set_ylabel(f"response [{spec.response_unit}]")
    axes[1].set_xlabel("phase-relative time [s]")
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="best", fontsize=8)
    fig.suptitle(
        f"{spec.name}, trial {trial['trial_index']}, {trial['classification']}; "
        f"NRMSE={trial.get('excitation_nrmse')}"
    )
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_summary(
    path: Path,
    summaries: list[dict[str, Any]],
    spec: AxisSpec,
    evaluation_role: str,
    model_label: str,
) -> None:
    valid = [row for row in summaries if row.get("excitation_nrmse") is not None]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5), constrained_layout=True)
    if valid:
        labels = [str(row["trial_index"]) for row in valid]
        axes[0].bar(labels, [row["excitation_nrmse"] for row in valid], color="tab:blue")
        axes[0].axhline(0.20, color="tab:red", ls="--", lw=1.0, label="default gate")
        axes[0].set_ylabel("excitation NRMSE")
        axes[0].set_xlabel("trial")
        axes[0].legend(fontsize=8)
    if valid:
        axes[1].bar(labels, [row.get("input_wrench_relative_p95") or 0.0 for row in valid], color="tab:orange")
        axes[1].axhline(0.05, color="tab:red", ls="--", lw=1.0, label="default gate")
        axes[1].set_ylabel("input wrench relative p95")
        axes[1].set_xlabel("trial")
        axes[1].legend(fontsize=8)
    for axis in axes:
        axis.grid(axis="y", alpha=0.25)
    role_label = "held-out" if evaluation_role == "held_out" else "in-sample"
    fig.suptitle(f"{spec.name}: {role_label} {model_label} replay")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _csv_cell(value: Any) -> Any:
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value if value is not None else ""


def _write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_cell(row.get(field)) for field in fields})


def _criteria_from_args(args: argparse.Namespace) -> Criteria:
    return Criteria(
        input_wrench_p95_max=args.input_wrench_p95_max,
        response_nrmse_max=args.response_nrmse_max,
        coast_nrmse_max=args.coast_nrmse_max,
        initial_slope_rel_max=args.initial_slope_rel_max,
        steady_state_rel_max=args.steady_state_rel_max,
        coast_time_rel_max=args.coast_time_rel_max,
        analysis_rate_hz=args.analysis_rate_hz,
        initial_window_sec=args.initial_window_sec,
        steady_fraction=args.steady_fraction,
        require_coast=not args.allow_missing_coast,
        wrench_floor=args.wrench_floor,
    )


def _add_criteria_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--input-wrench-p95-max", type=float, default=0.05)
    parser.add_argument("--response-nrmse-max", type=float, default=0.20)
    parser.add_argument("--coast-nrmse-max", type=float, default=0.30)
    parser.add_argument("--initial-slope-rel-max", type=float, default=0.30)
    parser.add_argument("--steady-state-rel-max", type=float, default=0.20)
    parser.add_argument("--coast-time-rel-max", type=float, default=0.25)
    parser.add_argument("--analysis-rate-hz", type=float, default=50.0)
    parser.add_argument("--initial-window-sec", type=float, default=0.50)
    parser.add_argument("--steady-fraction", type=float, default=0.25)
    parser.add_argument("--wrench-floor", type=float, default=0.05)
    parser.add_argument("--allow-missing-coast", action="store_true")


def _prepare(args: argparse.Namespace) -> None:
    real_csv = args.real_csv.expanduser().resolve()
    profile = args.profile.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if not real_csv.is_file() or not profile.is_file():
        raise FileNotFoundError("--real-csv and --profile must both name existing files")
    validation_record = real_csv.with_suffix(".validation.json")
    evaluation_role = str(getattr(args, "evaluation_role", "held_out"))
    source_record: dict[str, Any] | None = None
    if validation_record.is_file():
        source_record = json.loads(validation_record.read_text(encoding="utf-8"))
        if (
            evaluation_role == "held_out"
            and (source_record.get("run_role") != "held_out_fossen_validation" or source_record.get("fit_performed") is not False)
        ):
            raise ValueError(f"{validation_record} is not a no-fit held-out validation record")
    elif evaluation_role == "held_out" and not args.allow_unregistered_csv:
        raise ValueError(
            "real CSV has no `.validation.json` record. Record with "
            "run_role:=held_out_fossen_validation, or use --allow-unregistered-csv only for migration."
        )
    requested_trials = getattr(args, "trials", None)
    selected_trials = set(requested_trials) if requested_trials else None
    axis, trials = _load_real_trials(real_csv, args.axis, selected_trials)
    output_dir.mkdir(parents=True, exist_ok=True)
    criteria = _criteria_from_args(args)
    model_scope = str(getattr(args, "model_scope", "axiswise_diagonal_simplified_fossen"))
    model_label = str(getattr(args, "model_label", "Unity Fossen"))
    manifest = {
        "schema_version": 1,
        "evaluation_role": evaluation_role,
        "purpose": f"{evaluation_role}_axiswise_unity_hydrodynamic_replay",
        "model_scope": model_scope,
        "model_label": model_label,
        "real_csv": str(real_csv),
        "real_csv_sha256": _sha256(real_csv),
        "real_validation_record": str(validation_record) if validation_record.is_file() else None,
        "profile": str(profile),
        "profile_sha256": _sha256(profile),
        "repository_commit": _git_commit(Path(__file__).resolve().parents[4]),
        "axis": axis,
        "trial_indices": sorted(trials),
        "trial_count": len(trials),
        "force_source": getattr(args, "replay_force_source", "force_from_rpm"),
        "runtime_override_note": getattr(args, "runtime_override_note", "") or None,
        "unity_requirements": [
            f"selected model scope: {model_scope}",
            "domain randomization disabled",
            "no secondary DWP2/mesh hydrodynamic force source",
            "ForceN bridge mode",
            "replay actual RPM-derived canonical thruster forces without allocator",
        ],
        "criteria": asdict(criteria),
        "source_run": source_record,
    }
    _write_json(output_dir / "validator_manifest.json", manifest)
    print(f"wrote {output_dir / 'validator_manifest.json'}")
    print(f"axis={axis}; evaluation_role={evaluation_role}; trials={sorted(trials)}")


def _analyze(args: argparse.Namespace) -> None:
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    real_csv = Path(manifest["real_csv"]).expanduser().resolve()
    sim_dir = args.sim_dir.expanduser().resolve()
    output_dir = (args.output_dir or manifest_path.parent / "analysis").expanduser().resolve()
    if _sha256(real_csv) != manifest["real_csv_sha256"] and not args.allow_source_changed:
        raise ValueError("real CSV hash changed after prepare; rerun prepare or pass --allow-source-changed with an audit note")
    profile = Path(manifest["profile"]).expanduser().resolve()
    if profile.is_file() and _sha256(profile) != manifest["profile_sha256"] and not args.allow_source_changed:
        raise ValueError("Unity profile hash changed after prepare; rerun prepare before replaying")
    axis, real_trials = _load_real_trials(
        real_csv,
        manifest.get("axis"),
        set(int(index) for index in manifest.get("trial_indices", [])) or None,
    )
    if axis not in AXIS_SPECS:
        raise ValueError(f"axis `{axis}` is not supported by this validator")
    spec = AXIS_SPECS[axis]
    model_label = str(manifest.get("model_label", "Unity hydrodynamics"))
    criteria_values = dict(manifest.get("criteria", {}))
    criteria = Criteria(**{field: criteria_values.get(field, getattr(Criteria(), field)) for field in Criteria.__dataclass_fields__})
    sim_trials = _load_sim_trials(sim_dir)
    sim_dvl = _read_topic_series(_find_topic(sim_dir, "sim_controller_dvl"), "dvl")
    sim_imu = _read_topic_series(_find_topic(sim_dir, "sim_controller_imu"), "imu")
    sim_wrench = _read_topic_series(_find_topic(sim_dir, "sim_thruster_applied_wrench"), "wrench")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for trial_index in manifest.get("trial_indices", sorted(real_trials)):
        if trial_index not in real_trials:
            continue
        if trial_index not in sim_trials:
            summaries.append(
                {
                    "trial_index": trial_index,
                    "axis": axis,
                    "classification": "incomplete",
                    "reasons": ["missing_sim_replay_trial"],
                }
            )
            continue
        summary, series = _trial_metrics(
            real_trials[trial_index], sim_trials[trial_index], spec, sim_dvl, sim_imu, sim_wrench, criteria
        )
        summaries.append(summary)
        _plot_trial(output_dir / f"trial_{trial_index:03d}_response.png", spec, summary, series, model_label)
    _write_summary_csv(output_dir / "trial_summary.csv", summaries)
    evaluation_role = str(manifest.get("evaluation_role", "held_out"))
    _plot_summary(output_dir / "summary_metrics.png", summaries, spec, evaluation_role, model_label)
    complete_summaries = [row for row in summaries if row.get("classification") != "incomplete"]
    response_values = [
        float(row["excitation_nrmse"])
        for row in complete_summaries
        if row.get("excitation_nrmse") is not None
    ]
    onset_response_values = [
        float(row["onset_aligned_delta_response_nrmse"])
        for row in complete_summaries
        if row.get("onset_aligned_delta_response_nrmse") is not None
    ]
    wrench_onset_lags = [
        float(row["wrench_onset_lag_sec"])
        for row in complete_summaries
        if row.get("wrench_onset_lag_sec") is not None
    ]
    coast_values = [
        float(row["coast_nrmse"])
        for row in complete_summaries
        if row.get("coast_nrmse") is not None
    ]
    excitation_plus_coast_values = [
        float(row["excitation_plus_coast_nrmse"])
        for row in complete_summaries
        if row.get("excitation_plus_coast_nrmse") is not None
    ]
    report = {
        "schema_version": 1,
        "manifest": str(manifest_path),
        "axis": axis,
        "evaluation_role": evaluation_role,
        "model_scope": manifest.get("model_scope"),
        "model_label": model_label,
        "sim_dir": str(sim_dir),
        "criteria": asdict(criteria),
        "trial_count": len(summaries),
        "classification_counts": {
            label: sum(1 for row in summaries if row.get("classification") == label)
            for label in sorted({str(row.get("classification")) for row in summaries})
        },
        "excitation_nrmse": _bootstrap_median(response_values),
        "onset_aligned_delta_response_nrmse": _bootstrap_median(onset_response_values),
        "wrench_onset_lag_sec": _bootstrap_median(wrench_onset_lags),
        "coast_nrmse": _bootstrap_median(coast_values),
        "excitation_plus_coast_nrmse": _bootstrap_median(excitation_plus_coast_values),
        "trial_summary_csv": str(output_dir / "trial_summary.csv"),
        "plots": [str(output_dir / "summary_metrics.png"), *[str(output_dir / f"trial_{int(row['trial_index']):03d}_response.png") for row in summaries if "trial_index" in row and (output_dir / f"trial_{int(row['trial_index']):03d}_response.png").is_file()]],
        "claim_boundary": (
            f"This is an in-sample {model_label} replay check using data that contributed to coefficient fitting; "
            "it measures implementation consistency, not model generalization."
            if evaluation_role == "in_sample"
            else "Only axes classified consistent_with_frozen_profile may be described as held-out replay-consistent under this static-water, single-axis protocol."
        ),
    }
    _write_json(output_dir / "analysis_report.json", report)
    print(f"wrote {output_dir / 'analysis_report.json'}")
    print(json.dumps(report["classification_counts"], ensure_ascii=False, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser("prepare", help="freeze a real CSV/profile pair into a validator manifest")
    prepare.add_argument("--real-csv", type=Path, required=True)
    prepare.add_argument("--profile", type=Path, required=True)
    prepare.add_argument("--output-dir", type=Path, required=True)
    prepare.add_argument("--axis", choices=tuple(AXIS_SPECS), default=None)
    prepare.add_argument(
        "--trial",
        type=int,
        action="append",
        dest="trials",
        default=None,
        help="freeze only these complete trial indices; may be repeated",
    )
    prepare.add_argument("--evaluation-role", choices=("held_out", "in_sample"), default="held_out")
    prepare.add_argument("--model-scope", default="axiswise_diagonal_simplified_fossen")
    prepare.add_argument("--model-label", default="Unity Fossen")
    prepare.add_argument("--allow-unregistered-csv", action="store_true")
    prepare.add_argument(
        "--replay-force-source",
        choices=("force_from_rpm", "command"),
        default="force_from_rpm",
        help="record which replay input was sent to Unity",
    )
    prepare.add_argument(
        "--runtime-override-note",
        default="",
        help="audit note for a runtime-only Unity profile or actuator override",
    )
    _add_criteria_args(prepare)

    analyze = subparsers.add_parser("analyze", help="audit one Unity replay against the frozen real CSV")
    analyze.add_argument("--manifest", type=Path, required=True)
    analyze.add_argument("--sim-dir", type=Path, required=True)
    analyze.add_argument("--output-dir", type=Path, default=None)
    analyze.add_argument("--allow-source-changed", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.command == "prepare":
        _prepare(args)
    elif args.command == "analyze":
        _analyze(args)
    else:  # pragma: no cover - argparse keeps this unreachable.
        raise RuntimeError(f"unsupported command: {args.command}")


if __name__ == "__main__":
    main()
