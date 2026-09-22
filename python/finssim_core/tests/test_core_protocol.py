from __future__ import annotations

import json
from dataclasses import dataclass

import yaml

from finssim_core.artifacts import create_artifact_layout
from finssim_core.config_loader import load_experiment_config
from finssim_core.metadata import write_run_metadata
from finssim_core.overrides import apply_dataclass_overrides, merge_config_layers
from finssim_core.runtime import UnityRuntimeConfig


def test_load_experiment_config(tmp_path):
    config_path = tmp_path / "rl.yaml"
    config_path.write_text(
        """
base_config: control_for_pose
experiment:
  name: smoke
  seed: 7
  tags: [quick]
env:
  unity:
    env_path: /tmp/env.x86_64
    use_editor: true
    parallel_mode: multi_area
    env_base_port: 5100
    port_offset: 3
    no_graphics: true
    timeout_wait: 120
  train:
    num_envs: 64
    time_scale: 4.5
  eval:
    num_envs: 5
    time_scale: 1.0
    mode: serial
trainer:
  backend: rl
  overrides:
    total_timesteps: 100
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.base_config == "control_for_pose"
    assert config.experiment.name == "smoke"
    assert config.experiment.seed == 7
    assert config.experiment.tags == ("quick",)
    assert config.unity.env_path == "/tmp/env.x86_64"
    assert config.unity.use_editor is True
    assert config.unity.parallel_mode == "multi_area"
    assert config.train.num_envs == 64
    assert config.eval.num_envs == 5
    assert config.eval.mode == "serial"
    assert config.unity.port_for(0) == 5103
    assert config.unity.timeout_wait == 120
    assert config.unity_explicit_fields == (
        "env_path",
        "use_editor",
        "parallel_mode",
        "env_base_port",
        "port_offset",
        "no_graphics",
        "timeout_wait",
    )
    assert config.unity_overrides() == {
        "env_path": "/tmp/env.x86_64",
        "use_editor": True,
        "parallel_mode": "multi_area",
        "env_base_port": 5100,
        "port_offset": 3,
        "no_graphics": True,
        "timeout_wait": 120,
    }
    assert config.trainer.backend == "rl"
    assert config.trainer.overrides == {"total_timesteps": 100}


def test_load_marllib_experiment_config(tmp_path):
    config_path = tmp_path / "marllib.yaml"
    config_path.write_text(
        """
base_config: unity_3chase1
experiment:
  name: marllib_smoke
unity:
  env_base_port: 7100
trainer:
  backend: marllib
  overrides:
    algorithm: mappo
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.backend == "marllib"
    assert config.base_config == "unity_3chase1"
    assert config.trainer.overrides == {"algorithm": "mappo"}


def test_load_benchmarl_experiment_config(tmp_path):
    config_path = tmp_path / "benchmarl.yaml"
    config_path.write_text(
        """
base_config: chasing_3chase1
experiment:
  name: benchmarl_smoke
unity:
  env_base_port: 8100
trainer:
  backend: benchmarl
  overrides:
    dry_run: true
""",
        encoding="utf-8",
    )

    config = load_experiment_config(config_path)

    assert config.backend == "benchmarl"
    assert config.base_config == "chasing_3chase1"
    assert config.trainer.overrides == {"dry_run": True}


def test_override_priority_and_dataclass_application():
    @dataclass(frozen=True)
    class Example:
        num_envs: int = 1
        time_scale: float = 1.0

    merged = merge_config_layers(
        {"num_envs": 1, "time_scale": 1.0},
        {"num_envs": 2},
        {"time_scale": 5.0},
        {"num_envs": 8},
    )

    assert apply_dataclass_overrides(Example(), merged) == Example(num_envs=8, time_scale=5.0)


def test_artifact_layout_generation(tmp_path):
    layout = create_artifact_layout("rl", "demo", tmp_path)

    assert layout.run_dir == tmp_path / "runs" / "rl" / "demo"
    assert layout.logs_dir.is_dir()
    assert layout.checkpoints_dir.is_dir()
    assert layout.tensorboard_dir.is_dir()


def test_metadata_files(tmp_path):
    command = ["finssim-rl", "train", "--config", "control_for_pose"]
    metadata = write_run_metadata(
        tmp_path,
        backend="rl",
        config_path=tmp_path / "config.yaml",
        run_name="demo",
        seed=123,
        unity_env_path="/tmp/env",
        num_envs=4,
        env_base_port=5005,
        resolved_config={"base_config": "control_for_pose"},
        command=command,
        git_root=tmp_path,
    )

    assert metadata["backend"] == "rl"
    assert (tmp_path / "resolved_config.yaml").exists()
    assert yaml.safe_load((tmp_path / "resolved_config.yaml").read_text()) == {
        "base_config": "control_for_pose"
    }
    assert json.loads((tmp_path / "metadata.json").read_text())["run_name"] == "demo"
    assert (tmp_path / "command.txt").read_text().strip() == "finssim-rl train --config control_for_pose"


def test_metadata_files_allow_unknown_runtime_values(tmp_path):
    metadata = write_run_metadata(
        tmp_path,
        backend="marl",
        config_path=tmp_path / "config.yaml",
        run_name="demo",
        seed=123,
        unity_env_path=None,
        num_envs=None,
        env_base_port=None,
        resolved_config={"base_config": "chasing_3_chase_1"},
        command=["finssim-marl", "train", "--config", "chasing_3_chase_1"],
        git_root=tmp_path,
    )

    assert metadata["unity_env_path"] is None
    assert metadata["num_envs"] is None
    assert metadata["env_base_port"] is None


def test_unity_runtime_ports():
    runtime = UnityRuntimeConfig(env_base_port=6000, port_offset=10)

    assert runtime.worker_id(2) == 12
    assert runtime.port_for(2) == 6012


def test_unity_runtime_rejects_removed_area_count():
    import pytest

    with pytest.raises(ValueError, match="Unknown unity"):
        UnityRuntimeConfig.from_mapping({"num_areas": 0})
