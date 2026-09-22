"""Offline, auditable refit of hydrodynamic-identification CSV files.

Historical CSV files contain both the online identifier's filtered
``nudot_*`` estimate and the IMU quantities needed to replay it.  The default
``logged`` mode uses the former to reproduce the original online fit.  The
explicit ``recompute`` mode reconstructs gravity compensation and the live
first-order acceleration filter over the full recording without commanding the
vehicle again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from .hydrodynamic_identifier_node import (
    AXIS_ORDER,
    AXIS_TO_NU_INDEX,
    AXIS_TO_TAU_INDEX,
    ATTITUDE_PULSE_AXIS_ORDER,
    Sample,
    _fit_attitude_pulse_axis,
    _fit_axis,
    _gravity_controller_body_from_quaternion,
    _json_safe,
    _quat_normalize,
    _write_motion_png,
)


ACCELERATION_SOURCES = ("logged", "recompute")


def _float(row: dict[str, str], name: str, default: float = 0.0) -> float:
    try:
        value = float(row.get(name, default))
    except (TypeError, ValueError):
        return float(default)
    return value if math.isfinite(value) else float(default)


def _bool(row: dict[str, str], name: str, default: bool = False) -> bool:
    value = str(row.get(name, "")).strip().lower()
    if not value:
        return default
    return value in {"1", "true", "yes", "on"}


def _controller_gravity(row: dict[str, str], gravity_mps2: float) -> tuple[np.ndarray, bool]:
    """Return the gravity vector used by the online identifier when available."""
    recorded = np.asarray(
        [_float(row, f"imu_controller_gravity_{axis}", float("nan")) for axis in ("x", "y", "z")],
        dtype=np.float64,
    )
    if np.all(np.isfinite(recorded)):
        return recorded, True

    quaternion = np.asarray(
        [_float(row, f"imu_controller_quat_{axis}") for axis in ("x", "y", "z", "w")],
        dtype=np.float64,
    )
    _, quaternion_valid = _quat_normalize(quaternion)
    orientation_valid = _bool(row, "orientation_valid", default=False)
    if not quaternion_valid or not orientation_valid:
        return np.zeros(3, dtype=np.float64), False
    return _gravity_controller_body_from_quaternion(quaternion, gravity_mps2), True


def _logged_linear_acceleration(row: dict[str, str]) -> np.ndarray | None:
    """Read the exact filtered acceleration emitted by the online identifier."""
    values: list[float] = []
    for axis in ("x", "y", "z"):
        name = f"nudot_{axis}_mps2"
        if name not in row:
            return None
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            return None
        if not math.isfinite(value):
            return None
        values.append(value)
    return np.asarray(values, dtype=np.float64)


def _linear_acceleration_components(
    row: dict[str, str],
    *,
    gravity_mps2: float,
    gravity_compensation_enabled: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool]:
    """Return logged controller acceleration, gravity, and kinematic acceleration."""
    controller_accel = np.asarray(
        [_float(row, f"imu_controller_linear_accel_{axis}") for axis in ("x", "y", "z")],
        dtype=np.float64,
    )
    gravity, gravity_valid = _controller_gravity(row, gravity_mps2)
    if gravity_compensation_enabled and gravity_valid:
        kinematic_accel = controller_accel - gravity
    else:
        kinematic_accel = controller_accel.copy()
    return controller_accel, gravity, kinematic_accel, gravity_valid


def _replay_linear_acceleration_filter(
    rows: list[dict[str, str]],
    *,
    gravity_mps2: float,
    gravity_compensation_enabled: bool,
    acceleration_filter_alpha: float,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray, bool]]]:
    """Replay the live first-order IMU acceleration filter over every CSV row.

    The online node updates this state continuously, including zero-thrust and
    rest phases.  Replaying only fit-window rows would introduce an artificial
    reset at every phase boundary and cannot reproduce the online estimate.
    """
    alpha = float(np.clip(acceleration_filter_alpha, 0.0, 1.0))
    filtered = np.zeros(3, dtype=np.float64)
    accelerations: list[np.ndarray] = []
    diagnostics: list[tuple[np.ndarray, np.ndarray, bool]] = []
    for row in rows:
        _raw, gravity, kinematic, gravity_valid = _linear_acceleration_components(
            row,
            gravity_mps2=gravity_mps2,
            gravity_compensation_enabled=gravity_compensation_enabled,
        )
        filtered = alpha * kinematic + (1.0 - alpha) * filtered
        accelerations.append(filtered.copy())
        diagnostics.append((gravity, kinematic, gravity_valid))
    return accelerations, diagnostics


def _row_sample(
    row: dict[str, str],
    *,
    gravity_mps2: float,
    gravity_compensation_enabled: bool,
    linear_nu_dot: np.ndarray | None = None,
) -> tuple[Sample | None, np.ndarray, np.ndarray, bool]:
    axis = str(row.get("axis", "")).strip()
    if axis not in AXIS_ORDER:
        return None, np.zeros(3), np.zeros(3), False

    _controller_accel, gravity, kinematic_accel, gravity_valid = _linear_acceleration_components(
        row,
        gravity_mps2=gravity_mps2,
        gravity_compensation_enabled=gravity_compensation_enabled,
    )
    if linear_nu_dot is None:
        linear_nu_dot = kinematic_accel
    linear_nu_dot = np.asarray(linear_nu_dot, dtype=np.float64).reshape(3)

    nu = np.asarray(
        [
            _float(row, "nu_x_mps"),
            _float(row, "nu_y_mps"),
            _float(row, "nu_z_mps"),
            _float(row, "nu_roll_x_radps"),
            _float(row, "nu_yaw_radps"),
            _float(row, "nu_pitch_z_radps"),
        ],
        dtype=np.float64,
    )
    nu_dot = np.asarray(
        [
            linear_nu_dot[0],
            linear_nu_dot[1],
            linear_nu_dot[2],
            _float(row, "nudot_roll_x_radps2"),
            _float(row, "nudot_yaw_radps2"),
            _float(row, "nudot_pitch_z_radps2"),
        ],
        dtype=np.float64,
    )
    tau = np.asarray(
        [
            _float(row, "tau_fit_fx_n"),
            _float(row, "tau_fit_fy_n"),
            _float(row, "tau_fit_fz_n"),
            _float(row, "tau_fit_mx_nm"),
            _float(row, "tau_fit_my_yaw_nm"),
            _float(row, "tau_fit_mz_nm"),
        ],
        dtype=np.float64,
    )
    angle = np.asarray(
        [
            0.0,
            0.0,
            0.0,
            _float(row, "angle_roll_x_rad"),
            _float(row, "angle_yaw_y_rad"),
            _float(row, "angle_pitch_z_rad"),
        ],
        dtype=np.float64,
    )
    sample = Sample(
        time_sec=_float(row, "time_sec"),
        axis=axis,
        phase=str(row.get("phase", "")),
        phase_elapsed_sec=_float(row, "phase_elapsed_sec"),
        sample_window=_bool(row, "sample_window"),
        tau=tau,
        nu=nu,
        nu_dot=nu_dot,
        angle=angle,
        orientation_valid=_bool(row, "orientation_valid"),
    )
    return sample, gravity, kinematic_accel, gravity_valid


def _metadata_float(metadata: dict[str, Any], name: str, default: float) -> float:
    value = metadata.get(name, default)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _metadata_bool(metadata: dict[str, Any], name: str, default: bool) -> bool:
    value = metadata.get(name, default)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    return bool(value)


def _fit_one_axis(samples: list[Sample], axis: str, metadata: dict[str, Any]) -> dict[str, Any]:
    constrained = _metadata_bool(metadata, "fit_constrain_physical_coefficients", True)
    max_abs_accel = (
        _metadata_float(metadata, "fit_max_abs_angular_accel_radps2", 3.0)
        if axis == "yaw_y"
        else _metadata_float(metadata, "fit_max_abs_linear_accel_mps2", 5.0)
    )
    common = dict(
        accel_outlier_mad_threshold=max(0.0, _metadata_float(metadata, "fit_accel_outlier_mad_threshold", 8.0)),
        accel_spike_local_mad_threshold=max(
            0.0, _metadata_float(metadata, "fit_accel_spike_local_mad_threshold", 6.0)
        ),
        accel_spike_local_window=max(0, int(_metadata_float(metadata, "fit_accel_spike_local_window", 5.0))),
        constrain_physical_coefficients=constrained,
        max_linear_damping=max(0.0, _metadata_float(metadata, "fit_max_linear_damping", 500.0)),
        max_quadratic_damping=max(0.0, _metadata_float(metadata, "fit_max_quadratic_damping", 1000.0)),
        max_abs_bias=max(0.0, _metadata_float(metadata, "fit_max_abs_bias", 100.0)),
    )
    if axis in ATTITUDE_PULSE_AXIS_ORDER:
        return _fit_attitude_pulse_axis(
            samples,
            axis,
            min_abs_angle=max(0.0, _metadata_float(metadata, "attitude_min_abs_angle_rad", 0.003)),
            min_abs_rate=max(0.0, _metadata_float(metadata, "attitude_min_abs_rate_radps", 0.003)),
            min_abs_tau=max(0.0, _metadata_float(metadata, "attitude_min_abs_tau_nm", 0.005)),
            fit_start_sec=max(0.0, _metadata_float(metadata, "attitude_fit_start_sec", 0.05)),
            max_abs_acceleration=max(0.0, _metadata_float(metadata, "fit_max_abs_angular_accel_radps2", 3.0)),
            max_effective_inertia=max(0.0, _metadata_float(metadata, "fit_max_effective_inertia_kgm2", 20.0)),
            max_restoring_stiffness=max(
                0.0, _metadata_float(metadata, "fit_max_restoring_stiffness_nm_per_rad", 100.0)
            ),
            **common,
        )
    heave_dive_and_coast = (
        axis == "heave_y"
        and str(metadata.get("heave_y_identification_mode", "step")).strip().lower() == "dive_and_coast"
    )
    return _fit_axis(
        samples,
        axis,
        min_abs_tau=max(0.0, _metadata_float(metadata, "min_fit_abs_tau", 0.05)),
        velocity_deadband=max(0.0, _metadata_float(metadata, "fit_velocity_deadband", 0.005)),
        max_abs_acceleration=max(0.0, max_abs_accel),
        max_effective_mass=max(0.0, _metadata_float(metadata, "fit_max_effective_mass_kg", 100.0)),
        max_effective_inertia=max(0.0, _metadata_float(metadata, "fit_max_effective_inertia_kgm2", 20.0)),
        fit_phases=("excitation", "coast") if heave_dive_and_coast else ("excitation",),
        allow_zero_tau_samples=heave_dive_and_coast,
        fit_type="dive_and_coast" if heave_dive_and_coast else "forced_axis",
        **common,
    )


def refit_csv(
    csv_path: Path,
    *,
    gravity_mps2: float,
    axis_override: str | None,
    acceleration_source: str = "logged",
) -> dict[str, Any]:
    if acceleration_source not in ACCELERATION_SOURCES:
        raise ValueError(
            f"acceleration_source must be one of {ACCELERATION_SOURCES}, got {acceleration_source!r}"
        )
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = list(reader.fieldnames or [])
    if not rows:
        raise ValueError(f"CSV is empty: {csv_path}")

    source_fit_path = csv_path.with_name(f"{csv_path.stem}.fit.json")
    metadata: dict[str, Any] = {}
    if source_fit_path.is_file():
        with source_fit_path.open("r", encoding="utf-8") as stream:
            metadata = json.load(stream)

    gravity_compensation_enabled = _metadata_bool(metadata, "gravity_compensation_enabled", True)
    acceleration_filter_alpha = _metadata_float(metadata, "acceleration_filter_alpha", 0.35)

    if acceleration_source == "logged":
        replayed_accelerations = [_logged_linear_acceleration(row) for row in rows]
        missing = [index for index, value in enumerate(replayed_accelerations) if value is None]
        if missing:
            raise ValueError(
                "logged acceleration source requires nudot_{x,y,z}_mps2 in every CSV row; "
                f"missing or invalid rows begin at index {missing[0]}"
            )
        gravity_rows = [
            _linear_acceleration_components(
                row,
                gravity_mps2=gravity_mps2,
                gravity_compensation_enabled=gravity_compensation_enabled,
            )[1:]
            for row in rows
        ]
    else:
        replayed_accelerations, gravity_rows = _replay_linear_acceleration_filter(
            rows,
            gravity_mps2=gravity_mps2,
            gravity_compensation_enabled=gravity_compensation_enabled,
            acceleration_filter_alpha=acceleration_filter_alpha,
        )

    samples: list[Sample] = []
    for row, linear_nu_dot, (gravity, kinematic_accel, gravity_valid) in zip(
        rows, replayed_accelerations, gravity_rows
    ):
        sample, gravity, kinematic_accel, gravity_valid = _row_sample(
            row,
            gravity_mps2=gravity_mps2,
            gravity_compensation_enabled=gravity_compensation_enabled,
            linear_nu_dot=linear_nu_dot,
        )
        if sample is not None:
            samples.append(sample)

    source_axis = axis_override or str(metadata.get("output_axis", "")).strip()
    if source_axis not in AXIS_ORDER:
        available = [sample.axis for sample in samples if sample.axis in AXIS_ORDER]
        source_axis = available[0] if available else ""
    if source_axis not in AXIS_ORDER:
        raise ValueError(f"Cannot determine a supported axis from {csv_path}")

    fit = _fit_one_axis(samples, source_axis, metadata)
    fits = dict(metadata.get("fits", {}))
    fits[source_axis] = fit

    suffix = f"_refit_{acceleration_source}"
    output_csv = csv_path.with_name(f"{csv_path.stem}{suffix}.csv")
    output_fit = csv_path.with_name(f"{csv_path.stem}{suffix}.fit.json")
    output_png = csv_path.with_name(f"{csv_path.stem}{suffix}_velocity_acceleration.png")
    output_fit_png = csv_path.with_name(f"{csv_path.stem}{suffix}_fit_velocity_acceleration.png")

    extra_fields = [
        "imu_controller_gravity_x",
        "imu_controller_gravity_y",
        "imu_controller_gravity_z",
        "imu_controller_kinematic_accel_x",
        "imu_controller_kinematic_accel_y",
        "imu_controller_kinematic_accel_z",
        "gravity_compensation_valid",
    ]
    output_fields = fieldnames + [name for name in extra_fields if name not in fieldnames]
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=output_fields)
        writer.writeheader()
        for row, (gravity, kinematic_accel, gravity_valid) in zip(rows, gravity_rows):
            enriched = dict(row)
            enriched.update(
                {
                    "imu_controller_gravity_x": f"{gravity[0]:.9g}",
                    "imu_controller_gravity_y": f"{gravity[1]:.9g}",
                    "imu_controller_gravity_z": f"{gravity[2]:.9g}",
                    "imu_controller_kinematic_accel_x": f"{kinematic_accel[0]:.9g}",
                    "imu_controller_kinematic_accel_y": f"{kinematic_accel[1]:.9g}",
                    "imu_controller_kinematic_accel_z": f"{kinematic_accel[2]:.9g}",
                    "gravity_compensation_valid": int(gravity_valid),
                }
            )
            writer.writerow(enriched)

    _write_motion_png(samples, source_axis, output_png)
    _write_motion_png(samples, source_axis, output_fit_png, fit_only=True)

    result = dict(metadata)
    result.update(
        {
            "source_csv": str(csv_path),
            "refit_method": "offline_controller_body_acceleration_replay",
            "acceleration_source": acceleration_source,
            "gravity_compensation_enabled": gravity_compensation_enabled,
            "gravity_mps2": gravity_mps2,
            "acceleration_filter_alpha": acceleration_filter_alpha,
            "acceleration_filter_initial_state_mps2": [0.0, 0.0, 0.0],
            "acceleration_filter_replayed_over_all_csv_rows": acceleration_source == "recompute",
            "output_axis": source_axis,
            "sample_count": len(samples),
            "plot_files": [str(output_png), str(output_fit_png)],
            "refit_csv": str(output_csv),
            "fits": fits,
        }
    )
    with output_fit.open("w", encoding="utf-8") as stream:
        json.dump(_json_safe(result), stream, indent=2, ensure_ascii=False, sort_keys=True)
        stream.write("\n")

    old_fit = metadata.get("fits", {}).get(source_axis, {})
    print(f"{csv_path} ({acceleration_source} acceleration)")
    print(f"  output: {output_fit}")
    for key in ("m_eff", "d_linear", "d_quadratic", "bias", "rmse", "sample_count"):
        old_value = old_fit.get(key, "n/a")
        new_value = fit.get(key, "n/a")
        print(f"  {key}: {old_value} -> {new_value}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", nargs="+", type=Path, help="Previously recorded identifier CSV file(s).")
    parser.add_argument("--gravity-mps2", type=float, default=9.80665)
    parser.add_argument("--axis", choices=AXIS_ORDER, default=None, help="Override axis detection.")
    parser.add_argument(
        "--acceleration-source",
        choices=ACCELERATION_SOURCES,
        default="logged",
        help=(
            "`logged` reuses the filtered nudot_* values emitted by the online identifier; "
            "`recompute` rebuilds the recorded first-order IMU acceleration filter."
        ),
    )
    args = parser.parse_args()
    for path in args.csv:
        refit_csv(
            path.expanduser().resolve(),
            gravity_mps2=args.gravity_mps2,
            axis_override=args.axis,
            acceleration_source=args.acceleration_source,
        )


if __name__ == "__main__":
    main()
