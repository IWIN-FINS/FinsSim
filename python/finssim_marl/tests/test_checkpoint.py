from finssim_marl.utils.checkpoint import CheckpointConfig, CheckpointManager


class DummyAlgorithm:
    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            f.write("model")


def test_get_latest_ignores_training_state_files(tmp_path):
    run_dir = tmp_path / "demo_run"
    run_dir.mkdir()

    (run_dir / "step_100.pt").write_text("model")
    (run_dir / "step_100_training_state.pt").write_text("state")
    (run_dir / "step_200.pt").write_text("model")
    (run_dir / "step_200_training_state.pt").write_text("state")

    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(tmp_path)))

    latest = manager.get_latest("demo_run")

    assert latest is not None
    assert latest.name == "step_200.pt"


def test_save_interval_saves_when_called_for_non_multiple_step(tmp_path):
    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(tmp_path), save_freq=50_000))
    algorithm = DummyAlgorithm()

    path = manager.save_interval(
        "demo_run",
        51_723,
        algorithm,
        {"step": 51_723, "training_step": 4},
    )

    assert path is not None
    assert path.endswith("step_51723.pt")
    assert (tmp_path / "demo_run" / "step_51723.pt").exists()
    assert (tmp_path / "demo_run" / "step_51723_training_state.pt").exists()


def test_save_named_writes_named_checkpoint_and_preserves_latest_step_lookup(tmp_path):
    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(tmp_path)))
    algorithm = DummyAlgorithm()

    step_path = manager.save_interval(
        "demo_run",
        1_000,
        algorithm,
        {"step": 1_000, "training_step": 3},
    )
    best_path = manager.save_named(
        "demo_run",
        "best",
        algorithm,
        {"step": 900, "training_step": 2, "best_eval_reward": 12.3},
    )
    final_path = manager.save_named(
        "demo_run",
        "final.pt",
        algorithm,
        {"step": 1_200, "training_step": 4},
    )

    assert step_path.endswith("step_1000.pt")
    assert best_path.endswith("best.pt")
    assert final_path.endswith("final.pt")
    assert (tmp_path / "demo_run" / "best.pt").exists()
    assert (tmp_path / "demo_run" / "best_training_state.pt").exists()
    assert (tmp_path / "demo_run" / "final.pt").exists()
    assert (tmp_path / "demo_run" / "final_training_state.pt").exists()

    latest = manager.get_latest("demo_run")
    assert latest is not None
    assert latest.name == "step_1000.pt"
