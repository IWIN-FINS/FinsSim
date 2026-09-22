from __future__ import annotations

import os
from pathlib import Path

import pytest

from finssim_cli.isaaclab import _build_command, _child_environment, load_isaaclab_config


WORKSPACE = Path(__file__).resolve().parents[3]
PROJECT = WORKSPACE / "simulators/isaaclab/FinsSimIsaacLab"


def _write_config(path: Path, content: str) -> Path:
    path.write_text(content, encoding="utf-8")
    return path


def _config_text(*, library: str = "rsl_rl", task: str = "FinsSim-FinsROV-HoldForPosition-v0", extra: str = "") -> str:
    return f"""
experiment:
  name: smoke
  seed: 11
  tags: [isaaclab]
isaaclab:
  path: /tmp/IsaacLab
  project_path: {PROJECT}
  task: {task}
  rl_library: {library}
  cuda_visible_devices: "1"
  device: cuda:0
  num_envs: 8
  max_iterations: 2
  headless: true
  hydra_overrides: [env.enable_roll_pitch_moments=true]
train:
  resume_checkpoint: null
play:
  checkpoint: null
  num_envs: 1
  headless: false
  real_time: false
  video: false
{extra}
"""


def test_rsl_command_uses_existing_wrapper_and_learning_to_swim_flags(tmp_path: Path) -> None:
    config = load_isaaclab_config(
        _write_config(tmp_path / "run.yaml", _config_text()), workspace_root=WORKSPACE
    )
    command = _build_command(config, action="train", run_id="test-run")

    assert command[:5] == [
        "/tmp/IsaacLab/isaaclab.sh",
        "-p",
        str(PROJECT / "scripts/train.py"),
        "--task",
        "FinsSim-FinsROV-HoldForPosition-v0",
    ]
    assert command[command.index("--num_envs") + 1] == "8"
    assert command[command.index("--max_iterations") + 1] == "2"
    assert command[command.index("--run_name") + 1] == "test-run"
    assert command[command.index("--viz") + 1] == "none"
    assert command[-1] == "env.enable_roll_pitch_moments=true"


def test_skrl_command_selects_finsim_skrl_wrapper(tmp_path: Path) -> None:
    config = load_isaaclab_config(
        _write_config(tmp_path / "run.yaml", _config_text(library="skrl")), workspace_root=WORKSPACE
    )
    command = _build_command(config, action="train", run_id="ignored-by-skrl")

    assert str(PROJECT / "scripts/train_skrl.py") in command
    assert "--run_name" not in command
    assert command[command.index("--max_iterations") + 1] == "2"


def test_play_requires_yaml_checkpoint(tmp_path: Path) -> None:
    config = load_isaaclab_config(
        _write_config(tmp_path / "run.yaml", _config_text()), workspace_root=WORKSPACE
    )
    with pytest.raises(ValueError, match="play.checkpoint is required"):
        _build_command(config, action="play", run_id="unused")


def test_play_uses_kit_visualizer_when_yaml_is_not_headless(tmp_path: Path) -> None:
    checkpoint = Path("/tmp/IsaacLab/logs/rsl_rl/run/model_0.pt")
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config_path = _write_config(
        tmp_path / "run.yaml",
        _config_text(
            extra=(
                "play:\n"
                f"  checkpoint: {checkpoint}\n"
                "  num_envs: 1\n  headless: false\n  real_time: false\n  video: false"
            ),
        ),
    )
    config = load_isaaclab_config(config_path, workspace_root=WORKSPACE)
    command = _build_command(config, action="play", run_id="unused")

    assert command[command.index("--viz") + 1] == "kit"


def test_checkpoint_must_belong_to_selected_backend(tmp_path: Path) -> None:
    checkpoint = tmp_path / "IsaacLab/logs/skrl/run/checkpoints/agent.pt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    config_path = _write_config(
        tmp_path / "run.yaml",
        _config_text(extra=f"play:\n  checkpoint: {checkpoint}\n  num_envs: 1\n  headless: false\n  real_time: false\n  video: false"),
    )
    config = load_isaaclab_config(config_path, workspace_root=WORKSPACE)
    with pytest.raises(ValueError, match="cross-backend checkpoints"):
        _build_command(config, action="play", run_id="unused")


def test_child_environment_removes_parent_virtualenv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("VIRTUAL_ENV", "/bad/venv")
    monkeypatch.setenv("CONDA_PREFIX", "/bad/conda")
    monkeypatch.setenv("PYTHONHOME", "/bad/python")
    config = load_isaaclab_config(
        _write_config(tmp_path / "run.yaml", _config_text()), workspace_root=WORKSPACE
    )
    child = _child_environment(config)

    assert "VIRTUAL_ENV" not in child
    assert "CONDA_PREFIX" not in child
    assert "PYTHONHOME" not in child
    assert child["CUDA_VISIBLE_DEVICES"] == "1"
    assert child["PYTHONPATH"].split(os.pathsep)[0] == str(PROJECT / "source/finssim_isaaclab_tasks")
