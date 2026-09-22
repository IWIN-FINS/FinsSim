from pathlib import Path

from experiment_recorder.layout import (
    canonical_run_dir,
    create_run_layout,
    ensure_trial_data_dirs,
    prepare_trial_layout,
    register_trial_runtime_artifacts,
)
from experiment_recorder.manifest import read_json, write_json


def test_layout_creates_only_required_and_requested_trial_directories(tmp_path: Path) -> None:
    data_root = tmp_path / "experiments"
    layout = create_run_layout(data_root, domain="hardware", task="T1", method="PID", run_id="run")
    assert layout.run_dir == canonical_run_dir(data_root, domain="hardware", task="T1", method="PID", run_id="run")
    assert layout.trials_dir.is_dir()
    assert layout.derived_dir.is_dir()
    assert not layout.shared_logs_dir.exists()

    trial = prepare_trial_layout(layout.trials_dir, "trial")
    assert trial.logs_dir.is_dir()
    ensure_trial_data_dirs(trial.trial_dir)
    assert trial.raw_dir.is_dir() and trial.derived_dir.is_dir()
    assert not (trial.trial_dir / "external_truth").exists()
    assert not (trial.trial_dir / "video").exists()

    ensure_trial_data_dirs(trial.trial_dir, video=True)
    assert (trial.trial_dir / "video").is_dir()


def test_registers_trial_local_and_shared_logs_without_copying(tmp_path: Path) -> None:
    layout = create_run_layout(tmp_path, domain="simulation", task="T2", method="PID", run_id="run")
    trial = prepare_trial_layout(layout.trials_dir, "trial")
    write_json(trial.trial_dir / "manifest.json", {"schema_version": 2, "artifacts": {}})
    (trial.logs_dir / "recorder.log").write_text("recorder\n", encoding="utf-8")
    layout.shared_logs_dir.mkdir()
    shared = layout.shared_logs_dir / "grpc_ros_adapter.log"
    shared.write_text("shared\n", encoding="utf-8")
    run_manifest: dict[str, object] = {}

    register_trial_runtime_artifacts(
        run_manifest=run_manifest,
        run_dir=layout.run_dir,
        trial=trial,
        local_logs=("recorder.log",),
        shared_logs=(shared,),
    )

    trial_manifest = read_json(trial.trial_dir / "manifest.json")
    assert trial_manifest["artifacts"]["runtime_logs"]["local"] == ["logs/recorder.log"]
    assert run_manifest["trial_artifacts"]["trial"]["shared_runtime_logs"] == ["logs/grpc_ros_adapter.log"]
