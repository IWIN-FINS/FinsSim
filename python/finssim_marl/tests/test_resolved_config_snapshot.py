from __future__ import annotations

from dataclasses import dataclass

import yaml

from finssim_marl.training.artifacts import write_resolved_run_config
from finssim_marl.training.chasing_3_chase_1_config import get_mappo_config
from finssim_marl.training.config import get_config


@dataclass
class DummyArgs:
    config: str = "chasing_3_chase_1"
    exp_name: str = "snapshot_case"


def test_write_resolved_run_config_contains_algorithm_config(tmp_path):
    base_config = get_config("chasing_3_chase_1")
    final_config = base_config.create_train_config()
    final_config.output_dir = str(tmp_path)
    final_config.log_dir = str(tmp_path / "logs")
    final_config.checkpoint_dir = str(tmp_path / "checkpoints")
    final_config.tensorboard_dir = str(tmp_path / "tensorboard")
    algorithm_config = get_mappo_config(final_config)

    snapshot_path = write_resolved_run_config(
        tmp_path,
        config_name=base_config.name,
        run_name="chasing_3_chase_1__snapshot_case",
        base_config=base_config,
        final_config=final_config,
        algorithm_config=algorithm_config,
        cli_args=DummyArgs(),
        input_resolved_config={"unity": {"environment_parameters": {"foo": 1.0}}},
    )

    payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert payload["base_config"] == base_config.name
    assert payload["algorithm_config"]["gamma"] == final_config.gamma
    assert payload["final_config"]["num_envs"] == final_config.num_envs
    assert payload["input_resolved_config"]["unity"]["environment_parameters"]["foo"] == 1.0
