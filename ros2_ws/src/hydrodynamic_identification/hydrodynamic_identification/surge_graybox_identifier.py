"""Audit-friendly grey-box output-error identification for FinsROV surge.

The historical online identifier regresses a numerically estimated acceleration
against an RPM-derived wrench.  This utility instead rolls a small continuous
time model forward and compares its predicted velocity with the recorded
velocity.  It explicitly separates the commanded surge wrench from the
RPM-derived wrench used only as an auxiliary actuator observation:

    tau_a f_dot + f = gain * tau_cmd(t - delay)
    m_eff u_dot + d_1 u + d_2 |u| u + bias = f

The tool is intentionally offline.  It never modifies a Unity profile and it
splits whole excitation trials (rather than random samples) into fit and held-
out sets.  The static RPM--thrust curve remains a source of uncertainty; the
recorded RPM-derived wrench is therefore an audit signal, not force-sensor
ground truth.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class SurgeTrial:
    """One complete commanded surge trial in the controller-body convention."""

    trial_index: int
    level_n: float
    time_sec: np.ndarray
    phase: np.ndarray
    command_n: np.ndarray
    rpm_wrench_n: np.ndarray
    velocity_mps: np.ndarray


@dataclass(frozen=True)
class Parameters:
    effective_mass_kg: float
    linear_damping_ns_per_m: float
    quadratic_damping_ns2_per_m2: float
    bias_n: float
    actuator_gain: float
    actuator_time_constant_sec: float
    actuator_delay_sec: float


PARAMETER_NAMES = (
    "effective_mass_kg",
    "linear_damping_ns_per_m",
    "quadratic_damping_ns2_per_m2",
    "bias_n",
    "actuator_gain",
    "actuator_time_constant_sec",
    "actuator_delay_sec",
)

# Bounds deliberately encode passivity and physically plausible actuator
# dynamics.  They are not a claim that the fitted data identify every bound.
LOWER = np.asarray([5.0, 0.0, 0.0, -10.0, 0.30, 0.005, 0.0], dtype=float)
UPPER = np.asarray([100.0, 250.0, 400.0, 10.0, 1.50, 1.000, 0.500], dtype=float)
INITIAL = np.asarray([27.144699, 39.073595, 0.0, 0.282259, 0.92, 0.080, 0.120], dtype=float)
INITIAL_STEPS = np.asarray([10.0, 25.0, 40.0, 1.0, 0.12, 0.10, 0.060], dtype=float)


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


def _huber(residual: np.ndarray, delta: float = 0.35) -> np.ndarray:
    absolute = np.abs(residual)
    return np.where(absolute <= delta, 0.5 * residual * residual, delta * (absolute - 0.5 * delta))


def _parse_trials(csv_path: Path) -> dict[int, SurgeTrial]:
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"empty CSV: {csv_path}")

    required = {"trial_index", "axis", "time_sec", "phase", "tau_target_fx_n", "tau_fit_fx_n", "nu_x_mps"}
    missing = required - set(rows[0])
    if missing:
        raise ValueError(f"CSV is missing required surge columns: {sorted(missing)}")

    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if str(row.get("axis", "")).strip() != "surge_x":
            continue
        index = int(_finite(row.get("trial_index"), -1.0))
        if index < 0 or str(row.get("phase", "")).strip() == "":
            continue
        if str(row.get("synchronization_valid", "1")).strip().lower() in {"0", "false", "no"}:
            continue
        grouped.setdefault(index, []).append(row)

    trials: dict[int, SurgeTrial] = {}
    for index, values in grouped.items():
        values.sort(key=lambda row: _finite(row.get("time_sec")))
        time = np.asarray([_finite(row.get("time_sec")) for row in values], dtype=float)
        keep = np.r_[True, np.diff(time) > 1e-6]
        time = time[keep]
        if time.size < 8:
            continue
        selected = [row for row, include in zip(values, keep) if include]
        trials[index] = SurgeTrial(
            trial_index=index,
            level_n=float(np.median([_finite(row.get("level")) for row in selected])),
            time_sec=time - time[0],
            phase=np.asarray([str(row.get("phase", "")).strip() for row in selected], dtype=object),
            command_n=np.asarray([_finite(row.get("tau_target_fx_n")) for row in selected], dtype=float),
            rpm_wrench_n=np.asarray([_finite(row.get("tau_fit_fx_n")) for row in selected], dtype=float),
            velocity_mps=np.asarray([_finite(row.get("nu_x_mps")) for row in selected], dtype=float),
        )
    if not trials:
        raise ValueError(f"no synchronized surge trials in {csv_path}")
    return trials


def _parameters(vector: np.ndarray) -> Parameters:
    return Parameters(*[float(value) for value in vector])


def _simulate_trial(trial: SurgeTrial, vector: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Roll the actuator and surge ODE forward using the recorded command."""
    p = _parameters(vector)
    time = trial.time_sec
    command_delayed = np.interp(time - p.actuator_delay_sec, time, trial.command_n, left=0.0, right=0.0)
    force = np.empty_like(time)
    velocity = np.empty_like(time)
    force[0] = trial.rpm_wrench_n[0]
    velocity[0] = trial.velocity_mps[0]

    def acceleration(value_u: float, value_f: float) -> float:
        drag = p.linear_damping_ns_per_m * value_u + p.quadratic_damping_ns2_per_m2 * abs(value_u) * value_u
        return (value_f - drag - p.bias_n) / p.effective_mass_kg

    for index in range(1, time.size):
        dt = float(np.clip(time[index] - time[index - 1], 1e-4, 0.10))
        target = p.actuator_gain * command_delayed[index - 1]
        decay = math.exp(-dt / p.actuator_time_constant_sec)
        force[index] = target + (force[index - 1] - target) * decay
        a0 = acceleration(float(velocity[index - 1]), float(force[index - 1]))
        a1 = acceleration(float(velocity[index - 1] + dt * a0), float(force[index]))
        velocity[index] = velocity[index - 1] + 0.5 * dt * (a0 + a1)
    return force, velocity


def _fit_mask(trial: SurgeTrial, phases: frozenset[str]) -> np.ndarray:
    return np.asarray([str(value) in phases for value in trial.phase], dtype=bool)


def _trial_scales(trial: SurgeTrial, mask: np.ndarray) -> tuple[float, float]:
    velocity = trial.velocity_mps[mask]
    force = trial.rpm_wrench_n[mask]
    velocity_scale = max(0.03, float(np.percentile(velocity, 95) - np.percentile(velocity, 5)))
    force_scale = max(0.50, float(np.percentile(force, 95) - np.percentile(force, 5)))
    return velocity_scale, force_scale


def _objective(
    vector: np.ndarray,
    trials: Iterable[SurgeTrial],
    *,
    phases: frozenset[str],
    force_weight: float,
) -> float:
    if np.any(vector < LOWER) or np.any(vector > UPPER):
        return float("inf")
    total = 0.0
    count = 0
    for trial in trials:
        mask = _fit_mask(trial, phases)
        if int(mask.sum()) < 6:
            continue
        predicted_force, predicted_velocity = _simulate_trial(trial, vector)
        velocity_scale, force_scale = _trial_scales(trial, mask)
        velocity_loss = _huber((predicted_velocity[mask] - trial.velocity_mps[mask]) / velocity_scale)
        force_loss = _huber((predicted_force[mask] - trial.rpm_wrench_n[mask]) / force_scale)
        total += float(np.sum(velocity_loss) + force_weight * np.sum(force_loss))
        count += int(mask.sum())
    return total / max(count, 1)


def _local_pattern_search(
    start: np.ndarray,
    trials: list[SurgeTrial],
    *,
    phases: frozenset[str],
    force_weight: float,
    max_rounds: int = 80,
) -> tuple[np.ndarray, float]:
    """Derivative-free bounded pattern search; keeps the package NumPy-only."""
    current = np.clip(np.asarray(start, dtype=float), LOWER, UPPER)
    value = _objective(current, trials, phases=phases, force_weight=force_weight)
    steps = INITIAL_STEPS.copy()
    for _ in range(max_rounds):
        improved = False
        for dimension in range(current.size):
            candidates: list[tuple[float, np.ndarray]] = []
            for direction in (-1.0, 1.0):
                candidate = current.copy()
                candidate[dimension] = np.clip(candidate[dimension] + direction * steps[dimension], LOWER[dimension], UPPER[dimension])
                candidate_value = _objective(candidate, trials, phases=phases, force_weight=force_weight)
                candidates.append((candidate_value, candidate))
            candidate_value, candidate = min(candidates, key=lambda item: item[0])
            if candidate_value + 1e-12 < value:
                current, value = candidate, candidate_value
                improved = True
        if not improved:
            steps *= 0.5
            if float(np.max(steps / INITIAL_STEPS)) < 1e-3:
                break
    return current, value


def _fit(
    trials: list[SurgeTrial],
    *,
    phases: frozenset[str],
    force_weight: float,
    restarts: int,
    seed: int,
) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    starts = [INITIAL]
    for _ in range(max(0, restarts - 1)):
        starts.append(
            np.asarray(
                [
                    rng.uniform(10.0, 65.0),
                    rng.uniform(0.0, 140.0),
                    rng.uniform(0.0, 220.0),
                    rng.uniform(-3.0, 3.0),
                    rng.uniform(0.65, 1.20),
                    rng.uniform(0.02, 0.35),
                    rng.uniform(0.0, 0.25),
                ],
                dtype=float,
            )
        )
    candidates = [_local_pattern_search(start, trials, phases=phases, force_weight=force_weight) for start in starts]
    return min(candidates, key=lambda item: item[1])


def _metrics(trial: SurgeTrial, vector: np.ndarray, phases: frozenset[str]) -> dict[str, float | int]:
    mask = _fit_mask(trial, phases)
    predicted_force, predicted_velocity = _simulate_trial(trial, vector)
    velocity_error = predicted_velocity[mask] - trial.velocity_mps[mask]
    force_error = predicted_force[mask] - trial.rpm_wrench_n[mask]
    velocity_scale, force_scale = _trial_scales(trial, mask)
    return {
        "trial_index": trial.trial_index,
        "level_n": trial.level_n,
        "sample_count": int(mask.sum()),
        "velocity_rmse_mps": float(np.sqrt(np.mean(velocity_error * velocity_error))),
        "velocity_nrmse": float(np.sqrt(np.mean(velocity_error * velocity_error)) / velocity_scale),
        "force_rmse_n": float(np.sqrt(np.mean(force_error * force_error))),
        "force_nrmse": float(np.sqrt(np.mean(force_error * force_error)) / force_scale),
    }


def _write_plot(
    path: Path,
    trials: list[SurgeTrial],
    vector: np.ndarray,
    *,
    holdout: frozenset[int],
) -> None:
    columns = 2
    rows = int(math.ceil(len(trials) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(10.0, 2.6 * rows), squeeze=False, sharex=False)
    for axis, trial in zip(axes.flat, trials):
        predicted_force, predicted_velocity = _simulate_trial(trial, vector)
        split = "held-out" if trial.trial_index in holdout else "fit"
        axis.plot(trial.time_sec, trial.velocity_mps, color="#1f77b4", linewidth=1.4, label="recorded velocity")
        axis.plot(trial.time_sec, predicted_velocity, color="#d97924", linewidth=1.4, label="grey-box velocity")
        axis.set_title(f"trial {trial.trial_index}: {trial.level_n:+.0f} N ({split})", fontsize=9)
        axis.set_xlabel("trial time [s]")
        axis.set_ylabel("surge velocity [m/s]")
        twin = axis.twinx()
        twin.plot(trial.time_sec, trial.rpm_wrench_n, color="#777777", alpha=0.45, linewidth=0.9, label="RPM-derived wrench")
        twin.plot(trial.time_sec, predicted_force, color="#6a3d9a", alpha=0.75, linewidth=0.9, label="actuator model wrench")
        twin.set_ylabel("surge wrench [N]", color="#555555")
        if trial is trials[0]:
            handles, labels = axis.get_legend_handles_labels()
            handles2, labels2 = twin.get_legend_handles_labels()
            axis.legend(handles + handles2, labels + labels2, fontsize=7, loc="best")
    for axis in axes.flat[len(trials) :]:
        axis.remove()
    figure.suptitle("Surge grey-box output-error fit: velocity and actuator-wrench audit", fontsize=12)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _parse_indices(raw: str) -> frozenset[int]:
    if not raw.strip():
        return frozenset()
    try:
        return frozenset(int(value.strip()) for value in raw.split(",") if value.strip())
    except ValueError as error:
        raise ValueError("--holdout-trials must be a comma-separated list of integer trial indices") from error


def run(
    csv_path: Path,
    *,
    output_dir: Path,
    holdout: frozenset[int],
    phases: frozenset[str],
    force_weight: float,
    restarts: int,
    seed: int,
) -> Path:
    trials_by_index = _parse_trials(csv_path)
    nonzero_trials = [trial for _, trial in sorted(trials_by_index.items()) if abs(trial.level_n) > 1e-6]
    train_trials = [trial for trial in nonzero_trials if trial.trial_index not in holdout]
    heldout_trials = [trial for trial in nonzero_trials if trial.trial_index in holdout]
    if len(train_trials) < 2:
        raise ValueError("at least two nonzero training trials are required")
    if not heldout_trials:
        raise ValueError("at least one held-out complete trial is required")

    vector, objective = _fit(
        train_trials,
        phases=phases,
        force_weight=force_weight,
        restarts=restarts,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "surge_graybox_output_error.fit.json"
    plot_path = output_dir / "surge_graybox_output_error.png"
    trial_metrics = [
        {
            **_metrics(trial, vector, phases),
            "partition": "held_out" if trial.trial_index in holdout else "fit",
        }
        for trial in nonzero_trials
    ]
    report: dict[str, Any] = {
        "schema_version": 1,
        "method": "bounded_greybox_output_error_pattern_search",
        "source_csv": str(csv_path.resolve()),
        "source_csv_sha256": _sha256(csv_path),
        "axis": "surge_x",
        "coordinate_contract": "controller_x_forward_y_up_z_left",
        "model": {
            "actuator": "tau_a * f_dot + f = gain * tau_cmd(t-delay)",
            "hydrodynamics": "m_eff * u_dot + d1 * u + d2 * abs(u) * u + bias = f",
            "input": "recorded tau_target_fx_n",
            "actuator_observation": "recorded tau_fit_fx_n reconstructed from RPM and static thrust curves",
            "output": "recorded nu_x_mps",
        },
        "fit_protocol": {
            "unit_of_split": "complete trial",
            "fit_trial_indices": [trial.trial_index for trial in train_trials],
            "held_out_trial_indices": sorted(holdout),
            "fit_phases": sorted(phases),
            "force_auxiliary_weight": force_weight,
            "restarts": restarts,
            "seed": seed,
            "robust_loss": "Huber(delta=0.35) after per-trial response scaling",
        },
        "parameter_bounds": {name: [float(lower), float(upper)] for name, lower, upper in zip(PARAMETER_NAMES, LOWER, UPPER)},
        "parameters": asdict(_parameters(vector)),
        "objective": float(objective),
        "trial_metrics": trial_metrics,
        "plot": str(plot_path.resolve()),
        "claim_boundary": (
            "This is a single-recording, held-out-trial output-error check. The RPM-derived wrench is not a force-sensor "
            "measurement; the result must not be interpreted as a validated hardware hydrodynamic profile without independent "
            "velocity and thrust evidence."
        ),
    }
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    _write_plot(plot_path, nonzero_trials, vector, holdout=holdout)
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="synchronized surge identifier CSV")
    parser.add_argument("--output-dir", type=Path, default=None, help="defaults to <csv directory>/greybox_output_error")
    parser.add_argument("--holdout-trials", default="8,9", help="comma-separated complete trial indices; default: 8,9")
    parser.add_argument("--fit-phases", default="excitation,rest", help="comma-separated phases used in the loss")
    parser.add_argument("--force-weight", type=float, default=0.25, help="auxiliary RPM-derived wrench loss weight")
    parser.add_argument("--restarts", type=int, default=12, help="deterministic multistart count")
    parser.add_argument("--seed", type=int, default=20260903)
    args = parser.parse_args()
    if args.force_weight < 0.0:
        parser.error("--force-weight must be non-negative")
    phases = frozenset(value.strip() for value in args.fit_phases.split(",") if value.strip())
    if not phases:
        parser.error("--fit-phases must contain at least one phase")
    output_dir = args.output_dir or args.csv.resolve().parent / "greybox_output_error"
    report = run(
        args.csv.resolve(),
        output_dir=output_dir.resolve(),
        holdout=_parse_indices(args.holdout_trials),
        phases=phases,
        force_weight=float(args.force_weight),
        restarts=max(1, int(args.restarts)),
        seed=int(args.seed),
    )
    print(f"wrote {report}")


if __name__ == "__main__":
    main()
