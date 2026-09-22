from pathlib import Path

from finssim_marl.training.async_evaluation import AsyncEvaluationWorker
from finssim_marl.training.config import get_config
from finssim_marl.utils.checkpoint import CheckpointConfig, CheckpointManager


class DummyAlgorithm:
    def save(self, path):
        Path(path).write_bytes(b"model")


def test_async_evaluation_worker_returns_snapshot_specific_metrics(tmp_path):
    snapshot_dir = tmp_path / "eval_snapshots"
    evaluated = []
    worker = AsyncEvaluationWorker(
        snapshot_dir=snapshot_dir,
        evaluator=lambda snapshot, parameters: evaluated.append((snapshot.name, parameters)) or {"ep_reward": 4.0},
    )
    worker.start()
    snapshot = snapshot_dir / "snapshot_000000000100.pt"
    snapshot.write_bytes(b"model")
    worker.submit(
        step=100,
        snapshot_path=snapshot,
        training_state={"step": 100},
        environment_parameters={"lesson": 2.0},
    )
    worker.close()

    results = worker.drain_results()
    assert len(results) == 1
    assert results[0].job.step == 100
    assert results[0].metrics == {"ep_reward": 4.0}
    assert evaluated == [(snapshot.name, {"lesson": 2.0})]


def test_marl_train_config_defaults_to_discarding_eval_snapshots():
    config = get_config("chasing_3_chase_1_end_to_end").create_train_config()

    assert config.eval_mode == "asynchronous"
    assert config.keep_eval_snapshots is False


def test_checkpoint_manager_promotes_evaluated_snapshot_and_discards_source_by_default(tmp_path):
    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(tmp_path)))
    snapshot = tmp_path / "run" / "eval_snapshots" / "snapshot_000000000100.pt"
    snapshot.parent.mkdir(parents=True)
    manager.save(str(snapshot), DummyAlgorithm(), {"step": 100, "training_step": 2})

    best_path = Path(
        manager.promote_snapshot(
            "run",
            snapshot,
            {"step": 100, "training_step": 2, "best_eval_reward": 3.5},
            keep_source=False,
        )
    )

    assert best_path.read_bytes() == b"model"
    assert not snapshot.exists()
    assert (tmp_path / "run" / "best_training_state.pt").exists()


def test_checkpoint_manager_keeps_snapshot_when_requested(tmp_path):
    manager = CheckpointManager(CheckpointConfig(checkpoint_dir=str(tmp_path)))
    snapshot = tmp_path / "run" / "eval_snapshots" / "snapshot_000000000100.pt"
    snapshot.parent.mkdir(parents=True)
    manager.save(str(snapshot), DummyAlgorithm(), {"step": 100})

    best_path = Path(manager.promote_snapshot("run", snapshot, {"step": 100}, keep_source=True))

    assert best_path.read_bytes() == b"model"
    assert snapshot.read_bytes() == b"model"
