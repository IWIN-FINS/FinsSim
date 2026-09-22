from pathlib import Path

from finssim_rl.training.trainer import find_latest_checkpoint


def test_latest_checkpoint_uses_numeric_step_order(tmp_path: Path) -> None:
    (tmp_path / "ppo_underwater_model_99968_steps.zip").touch()
    expected = tmp_path / "ppo_underwater_model_4372992_steps.zip"
    expected.touch()
    (tmp_path / "ppo_underwater_model_1000000_steps.zip").touch()

    assert find_latest_checkpoint(str(tmp_path)) == str(expected)


def test_latest_checkpoint_falls_back_to_final_then_best(tmp_path: Path) -> None:
    best = tmp_path / "best_model.zip"
    best.touch()
    assert find_latest_checkpoint(str(tmp_path)) == str(best)

    final = tmp_path / "final_model.zip"
    final.touch()
    assert find_latest_checkpoint(str(tmp_path)) == str(final)
