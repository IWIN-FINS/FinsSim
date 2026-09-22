from types import SimpleNamespace

from finssim_rl.training.artifacts import resolve_training_output_paths


def test_output_dir_maps_to_standard_subdirectories():
    args = SimpleNamespace(
        config="control_for_pose",
        exp_name="managed",
        output_dir="/tmp/finsim-run",
        log_dir=None,
        checkpoint_dir=None,
        tensorboard_dir=None,
    )

    resolve_training_output_paths(args)

    assert args.log_dir == "/tmp/finsim-run/logs"
    assert args.checkpoint_dir == "/tmp/finsim-run/checkpoints"
    assert args.tensorboard_dir == "/tmp/finsim-run/tensorboard"


def test_missing_output_dir_defaults_to_local_artifacts():
    args = SimpleNamespace(
        config="control_for_pose",
        exp_name="standalone",
        output_dir=None,
        log_dir=None,
        checkpoint_dir=None,
        tensorboard_dir=None,
    )

    resolve_training_output_paths(args)

    assert args.output_dir == "artifacts/runs/rl/standalone"
    assert args.log_dir == "artifacts/runs/rl/standalone/logs"
    assert args.checkpoint_dir == "artifacts/runs/rl/standalone/checkpoints"
    assert args.tensorboard_dir == "artifacts/runs/rl/standalone/tensorboard"
