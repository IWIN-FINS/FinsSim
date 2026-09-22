"""Canonical experiment artifact layout shared by all runners.

The helpers in this module intentionally distinguish a *run* (one batch
invocation) from a *trial* (one rosbag).  Trial-owned artifacts must live
below the trial directory; only processes shared by multiple trials may write
to the run-level ``logs`` directory.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Sequence

from .manifest import read_json, safe_session_dir, write_json


VALID_DOMAINS = frozenset({"hardware", "simulation"})
VALID_TASKS = frozenset({"T1", "T2"})


@dataclass(frozen=True)
class RunLayout:
    """Paths owned by one T1/T2 runner invocation."""

    run_dir: Path
    trials_dir: Path
    derived_dir: Path
    shared_logs_dir: Path


@dataclass(frozen=True)
class TrialLayout:
    """Paths owned by one recorded rosbag trial."""

    trial_dir: Path
    raw_dir: Path
    derived_dir: Path
    logs_dir: Path


def canonical_run_dir(data_root: Path, *, domain: str, task: str, method: str, run_id: str) -> Path:
    """Return the only supported T1/T2 run location below ``data_root``."""

    if domain not in VALID_DOMAINS:
        raise ValueError(f"unsupported experiment domain `{domain}`")
    if task not in VALID_TASKS:
        raise ValueError(f"unsupported experiment task `{task}`")
    for name, value in (("method", method), ("run_id", run_id)):
        candidate = Path(str(value).strip())
        if not str(value).strip() or candidate.is_absolute() or candidate.name != str(value).strip():
            raise ValueError(f"{name} must be a non-empty single path component")
    root = data_root.expanduser().resolve()
    return root / domain / task / method / run_id


def create_run_layout(data_root: Path, *, domain: str, task: str, method: str, run_id: str) -> RunLayout:
    """Create a run without creating an empty shared-log directory."""

    run_dir = canonical_run_dir(data_root, domain=domain, task=task, method=method, run_id=run_id)
    run_dir.mkdir(parents=True, exist_ok=False)
    trials_dir = run_dir / "trials"
    derived_dir = run_dir / "derived"
    trials_dir.mkdir()
    derived_dir.mkdir()
    return RunLayout(
        run_dir=run_dir,
        trials_dir=trials_dir,
        derived_dir=derived_dir,
        shared_logs_dir=run_dir / "logs",
    )


def prepare_trial_layout(trials_dir: Path, trial_id: str) -> TrialLayout:
    """Prepare the non-destructive shell needed before spawning a recorder."""

    trial_dir = safe_session_dir(trials_dir, trial_id)
    trial_dir.mkdir(parents=True, exist_ok=False)
    logs_dir = trial_dir / "logs"
    logs_dir.mkdir()
    return TrialLayout(
        trial_dir=trial_dir,
        raw_dir=trial_dir / "raw",
        derived_dir=trial_dir / "derived",
        logs_dir=logs_dir,
    )


def ensure_trial_data_dirs(trial_dir: Path, *, external_truth: bool = False, video: bool = False) -> TrialLayout:
    """Create required data directories and requested optional artifact roots."""

    trial_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = trial_dir / "raw"
    derived_dir = trial_dir / "derived"
    raw_dir.mkdir(exist_ok=True)
    derived_dir.mkdir(exist_ok=True)
    if external_truth:
        (trial_dir / "external_truth").mkdir(exist_ok=True)
    if video:
        (trial_dir / "video").mkdir(exist_ok=True)
    return TrialLayout(
        trial_dir=trial_dir,
        raw_dir=raw_dir,
        derived_dir=derived_dir,
        logs_dir=trial_dir / "logs",
    )


def shared_log_path(layout: RunLayout, name: str) -> Path:
    """Return a lazily-created run-level log path for a shared process."""

    path = layout.shared_logs_dir / name
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def register_trial_runtime_artifacts(
    *,
    run_manifest: dict[str, Any],
    run_dir: Path,
    trial: TrialLayout,
    local_logs: Sequence[str],
    shared_logs: Sequence[Path] = (),
) -> None:
    """Record local and run-shared runtime logs without duplicating files."""

    trial_manifest_path = trial.trial_dir / "manifest.json"
    if not trial_manifest_path.is_file():
        return
    trial_manifest = read_json(trial_manifest_path)
    artifacts = trial_manifest.setdefault("artifacts", {})
    artifacts["runtime_logs"] = {
        "local": [f"logs/{name}" for name in local_logs],
        "shared": [os.path.relpath(path, trial.trial_dir) for path in shared_logs],
    }
    write_json(trial_manifest_path, trial_manifest)

    run_manifest.setdefault("trial_artifacts", {})[trial.trial_dir.name] = {
        "directory": str(trial.trial_dir.relative_to(run_dir)),
        "local_runtime_logs": [str((trial.logs_dir / name).relative_to(run_dir)) for name in local_logs],
        "shared_runtime_logs": [str(path.relative_to(run_dir)) for path in shared_logs],
    }
