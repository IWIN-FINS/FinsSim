from __future__ import annotations

from dataclasses import dataclass

import yaml

from finssim_rl.training.artifacts import write_resolved_run_config
from finssim_rl.training.config import BaseEnvironmentConfig, BaseTrainingConfig, BaseConfig


@dataclass
class DummyConfig(BaseConfig):
    def create_model(self, env, args, load_checkpoint=None):
        raise NotImplementedError

    def load_model_for_eval(self, model_path: str, device: str = "auto"):
        raise NotImplementedError


@dataclass
class DummyArgs:
    exp_name: str = "snapshot_case"
    output_dir: str | None = None
    log_dir: str = "logs"
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: str = "tensorboard"


def test_write_resolved_run_config_contains_full_sections(tmp_path):
    config = DummyConfig(
        name="dummy_cfg",
        model_type="PPO",
        env_config=BaseEnvironmentConfig(env_path="/tmp/unity.x86_64", num_envs=4),
        training_config=BaseTrainingConfig(gamma=0.95),
    )
    args = DummyArgs(output_dir=str(tmp_path))

    snapshot_path = write_resolved_run_config(
        tmp_path,
        config_name=config.name,
        base_config=config,
        final_config=config,
        cli_args=args,
    )

    payload = yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
    assert payload["base_config"] == "dummy_cfg"
    assert payload["base_config_resolved"]["env_config"]["env_path"] == "/tmp/unity.x86_64"
    assert payload["final_config"]["training_config"]["gamma"] == 0.95
    assert payload["cli_args"]["exp_name"] == "snapshot_case"
