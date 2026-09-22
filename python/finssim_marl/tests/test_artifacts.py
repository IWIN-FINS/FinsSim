from types import SimpleNamespace

from finssim_marl.training.artifacts import build_run_name, resolve_training_output_paths


def test_output_dir_maps_to_standard_subdirectories():
    config = SimpleNamespace(
        output_dir="/tmp/finsim-marl-run",
        log_dir="runs",
        checkpoint_dir="checkpoints",
        tensorboard_dir=None,
    )

    resolve_training_output_paths(config, "chasing_3_chase_1__managed")

    assert config.log_dir == "/tmp/finsim-marl-run/logs"
    assert config.checkpoint_dir == "/tmp/finsim-marl-run/checkpoints"
    assert config.tensorboard_dir == "/tmp/finsim-marl-run/tensorboard"


def test_missing_output_dir_defaults_to_local_artifacts():
    config = SimpleNamespace(
        output_dir=None,
        log_dir="runs",
        checkpoint_dir="checkpoints",
        tensorboard_dir=None,
    )

    resolve_training_output_paths(config, "chasing_3_chase_1__standalone")

    assert config.output_dir == "artifacts/runs/marl/chasing_3_chase_1__standalone"
    assert config.log_dir == "artifacts/runs/marl/chasing_3_chase_1__standalone/logs"
    assert config.checkpoint_dir == "artifacts/runs/marl/chasing_3_chase_1__standalone/checkpoints"
    assert config.tensorboard_dir == "artifacts/runs/marl/chasing_3_chase_1__standalone/tensorboard"


def test_build_run_name_matches_training_script_convention():
    assert build_run_name("chasing_3_chase_1", "exp") == "chasing_3_chase_1__exp"
    assert build_run_name("chasing_3_chase_1", None) == "chasing_3_chase_1"
