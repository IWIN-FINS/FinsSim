from finssim_rl.training.eval_args import EvalArgs


def test_eval_uses_explicit_checkpoint_path() -> None:
    args = EvalArgs(
        config="ppo_wrench_for_pose_empirical_thruster_mixer",
        output_dir="/tmp/eval-run",
        checkpoint_path="/tmp/training-run/checkpoints/ppo_underwater_model_123_steps.zip",
    )

    assert args.checkpoint_path == "/tmp/training-run/checkpoints/ppo_underwater_model_123_steps.zip"


def test_eval_does_not_derive_checkpoint_path_from_output_dir() -> None:
    args = EvalArgs(config="ppo_wrench_for_pose_empirical_thruster_mixer", output_dir="/tmp/eval-run")

    assert args.checkpoint_path is None
