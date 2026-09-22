from __future__ import annotations

import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "prune_checkpoints.py"


def _write_checkpoint(run_dir: Path, step: int) -> None:
    (run_dir / f"step_{step}.pt").write_text("model", encoding="utf-8")
    (run_dir / f"step_{step}_training_state.pt").write_text("state", encoding="utf-8")


def test_prune_checkpoints_dry_run_keeps_files(tmp_path):
    run_dir = tmp_path / "demo_run" / "checkpoints"
    run_dir.mkdir(parents=True)
    for step in (10_000, 20_000, 100_000, 200_000):
        _write_checkpoint(run_dir, step)
    (run_dir / "best.pt").write_text("best", encoding="utf-8")
    (run_dir / "final.pt").write_text("final", encoding="utf-8")

    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(tmp_path),
            "--min-step",
            "100000",
            "--keep-last-n",
            "1",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "DRY-RUN" in result.stdout
    assert "delete steps: 10000, 20000" in result.stdout
    assert (run_dir / "step_10000.pt").exists()
    assert (run_dir / "step_20000.pt").exists()
    assert (run_dir / "best.pt").exists()
    assert (run_dir / "final.pt").exists()


def test_prune_checkpoints_apply_removes_old_steps_and_training_state(tmp_path):
    run_dir = tmp_path / "demo_run" / "checkpoints"
    run_dir.mkdir(parents=True)
    for step in (10_000, 20_000, 100_000, 200_000):
        _write_checkpoint(run_dir, step)

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(tmp_path),
            "--min-step",
            "100000",
            "--keep-last-n",
            "1",
            "--keep-every",
            "20000",
            "--apply",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert not (run_dir / "step_10000.pt").exists()
    assert not (run_dir / "step_10000_training_state.pt").exists()
    assert (run_dir / "step_20000.pt").exists()
    assert (run_dir / "step_20000_training_state.pt").exists()
    assert (run_dir / "step_100000.pt").exists()
    assert (run_dir / "step_200000.pt").exists()
