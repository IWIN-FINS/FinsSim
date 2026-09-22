import csv
import json
from pathlib import Path
from types import SimpleNamespace

from hydrodynamic_identification.unity_fossen_replay_validator import _analyze, _prepare, _read_topic_series


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _criteria_args() -> dict[str, object]:
    return {
        "input_wrench_p95_max": 0.05,
        "response_nrmse_max": 0.20,
        "coast_nrmse_max": 0.30,
        "initial_slope_rel_max": 0.30,
        "steady_state_rel_max": 0.20,
        "coast_time_rel_max": 0.25,
        "analysis_rate_hz": 50.0,
        "initial_window_sec": 0.50,
        "steady_fraction": 0.25,
        "wrench_floor": 0.05,
        "allow_missing_coast": False,
    }


def _topic_rows(samples: list[tuple[float, dict[str, object]]]) -> list[dict[str, object]]:
    return [
        {
            "wall_time_sec": f"{100.0 + time_sec:.6f}",
            "ros_time_sec": f"{100.0 + time_sec:.6f}",
            "label": "test",
            "topic": "/sim/test",
            "msg_type": "test",
            "payload_json": json.dumps({**payload, "stamp_sec": time_sec}),
        }
        for time_sec, payload in samples
    ]


def test_unity_series_uses_recorder_wall_time_not_drifting_simulation_stamp(tmp_path: Path):
    topic = tmp_path / "sim_controller_dvl__sim_finsrov_controller_dvl.csv"
    fields = ["wall_time_sec", "ros_time_sec", "label", "topic", "msg_type", "payload_json"]
    _write_csv(
        topic,
        fields,
        [
            {
                "wall_time_sec": "100.000",
                "ros_time_sec": "100.000",
                "label": "test",
                "topic": "/sim/test",
                "msg_type": "TwistWithCovarianceStamped",
                "payload_json": json.dumps({"stamp_sec": 1.0, "linear": {"x": 0.0, "y": 0.0, "z": 0.0}}),
            },
            {
                "wall_time_sec": "100.200",
                "ros_time_sec": "100.200",
                "label": "test",
                "topic": "/sim/test",
                "msg_type": "TwistWithCovarianceStamped",
                "payload_json": json.dumps({"stamp_sec": 1.5, "linear": {"x": 1.0, "y": 0.0, "z": 0.0}}),
            },
        ],
    )

    series = _read_topic_series(topic, "dvl")

    assert list(series.time) == [100.0, 100.2]


def test_prepare_and_analyze_accept_phase_matched_held_out_replay(tmp_path: Path):
    real_csv = tmp_path / "surge.csv"
    profile = tmp_path / "FinsROV_HydrodynamicsProfile.asset"
    profile.write_text("frozen test profile\n", encoding="utf-8")
    phases = [
        ("baseline", [0.0, 0.0, 0.0]),
        ("excitation", [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]),
        ("coast", [0.5, 0.4, 0.3, 0.2, 0.1, 0.0]),
        ("rest", [0.0, 0.0, 0.0]),
    ]
    real_rows: list[dict[str, object]] = []
    command_rows: list[dict[str, object]] = []
    dvl_samples: list[tuple[float, dict[str, object]]] = []
    imu_samples: list[tuple[float, dict[str, object]]] = []
    wrench_samples: list[tuple[float, dict[str, object]]] = []
    time_sec = 0.0
    for phase, values in phases:
        for value in values:
            force = 10.0 if phase == "excitation" else 0.0
            real_rows.append(
                {
                    "time_sec": time_sec,
                    "trial_index": 0,
                    "repeat_index": 1,
                    "axis": "surge_x",
                    "phase": phase,
                    "level": 10.0,
                    "tau_fit_fx_n": force,
                    "tau_fit_fy_n": 0.0,
                    "tau_fit_fz_n": 0.0,
                    "tau_fit_mx_nm": 0.0,
                    "tau_fit_my_yaw_nm": 0.0,
                    "tau_fit_mz_nm": 0.0,
                    "nu_x_mps": value,
                    "nu_y_mps": 0.0,
                    "nu_z_mps": 0.0,
                    "nu_roll_x_radps": 0.0,
                    "nu_yaw_radps": 0.0,
                    "nu_pitch_z_radps": 0.0,
                }
            )
            command_rows.append(
                {"wall_time_sec": 100.0 + time_sec, "trial_index": 0, "phase": phase, "level": 10.0}
            )
            dvl_samples.append((time_sec, {"linear": {"x": value, "y": 0.0, "z": 0.0}}))
            imu_samples.append((time_sec, {"angular_velocity": {"x": 0.0, "y": 0.0, "z": 0.0}}))
            wrench_samples.append(
                (time_sec, {"linear": {"x": force, "y": 0.0, "z": 0.0}, "angular": {"x": 0.0, "y": 0.0, "z": 0.0}})
            )
            time_sec += 0.1
    _write_csv(real_csv, list(real_rows[0]), real_rows)
    real_csv.with_suffix(".validation.json").write_text(
        json.dumps({"run_role": "held_out_fossen_validation", "fit_performed": False}), encoding="utf-8"
    )

    sim_dir = tmp_path / "unity"
    sim_dir.mkdir()
    _write_csv(sim_dir / "replay_commands.csv", list(command_rows[0]), command_rows)
    topic_fields = ["wall_time_sec", "ros_time_sec", "label", "topic", "msg_type", "payload_json"]
    _write_csv(sim_dir / "sim_controller_dvl__sim_finsrov_controller_dvl.csv", topic_fields, _topic_rows(dvl_samples))
    _write_csv(sim_dir / "sim_controller_imu__sim_finsrov_controller_imu.csv", topic_fields, _topic_rows(imu_samples))
    _write_csv(
        sim_dir / "sim_thruster_applied_wrench__sim_finsrov_debug_thruster_applied_wrench.csv",
        topic_fields,
        _topic_rows(wrench_samples),
    )

    run_dir = tmp_path / "run"
    _prepare(
        SimpleNamespace(
            real_csv=real_csv,
            profile=profile,
            output_dir=run_dir,
            axis="surge_x",
            allow_unregistered_csv=False,
            **_criteria_args(),
        )
    )
    _analyze(
        SimpleNamespace(
            manifest=run_dir / "validator_manifest.json",
            sim_dir=sim_dir,
            output_dir=run_dir / "analysis",
            allow_source_changed=False,
        )
    )
    report = json.loads((run_dir / "analysis" / "analysis_report.json").read_text(encoding="utf-8"))
    summary = list(csv.DictReader((run_dir / "analysis" / "trial_summary.csv").open(encoding="utf-8")))
    assert report["classification_counts"] == {"consistent_with_frozen_profile": 1}
    assert float(summary[0]["excitation_nrmse"]) < 1e-10
    assert float(summary[0]["input_wrench_relative_p95"]) < 1e-10
