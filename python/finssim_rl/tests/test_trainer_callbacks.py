import json
from types import SimpleNamespace

from finssim_rl.training import trainer


class _CallbackCapture:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


def test_callbacks_use_separate_eval_env_and_aggregate_step_frequencies(monkeypatch, tmp_path):
    checkpoint_instances = []
    eval_instances = []

    def checkpoint_callback(*args, **kwargs):
        instance = _CallbackCapture(*args, **kwargs)
        checkpoint_instances.append(instance)
        return instance

    def eval_callback(*args, **kwargs):
        instance = _CallbackCapture(*args, **kwargs)
        eval_instances.append(instance)
        return instance

    monkeypatch.setattr(trainer, "CheckpointCallback", checkpoint_callback)
    monkeypatch.setattr(trainer, "EvalCallback", eval_callback)
    monkeypatch.setattr(trainer, "CallbackList", lambda callbacks: callbacks)

    training_env = SimpleNamespace(num_envs=1024)
    evaluation_env = SimpleNamespace(num_envs=1)
    args = SimpleNamespace(checkpoint_dir=str(tmp_path / "checkpoints"), log_dir=str(tmp_path / "logs"))
    config = SimpleNamespace(
        model_type="PPO_WRENCH",
        training_config=SimpleNamespace(checkpoint_freq=819_200, eval_freq=327_680),
    )

    callbacks = trainer.make_callbacks(training_env, args, config, eval_env=evaluation_env)

    assert callbacks[0] is checkpoint_instances[0]
    assert isinstance(callbacks[1], trainer.UnityStatsTensorboardCallback)
    assert callbacks[2] is eval_instances[0]
    assert isinstance(callbacks[3], trainer.UnityStatsTensorboardCallback)
    assert checkpoint_instances[0].kwargs["save_freq"] == 800
    assert eval_instances[0].args[0] is evaluation_env
    assert eval_instances[0].kwargs["eval_freq"] == 320


def test_async_evaluation_writes_tensorboard_metrics_at_snapshot_step(monkeypatch, tmp_path):
    writes = []

    class Writer:
        def add_scalar(self, tag, value, step):
            writes.append((tag, value, step))

        def flush(self):
            writes.append(("flush", None, None))

    callback = trainer.AsyncEvaluationCallback(
        SimpleNamespace(num_envs=1),
        SimpleNamespace(
            log_dir=str(tmp_path / "logs"),
            checkpoint_dir=str(tmp_path / "checkpoints"),
            tensorboard_dir=str(tmp_path / "tensorboard"),
        ),
        SimpleNamespace(),
        eval_freq=1,
    )
    callback.tensorboard_writer = Writer()

    callback._write_tensorboard_metrics({
        "snapshot_timestep": 163_840,
        "mean_reward": 12.5,
        "mean_episode_length": 150.0,
        "queue_wait_seconds": 0.75,
        "started_at": 10.0,
        "completed_at": 13.5,
    }, {"FinsROV/trajectory_tracking/tracking_error_p95_m": 0.12})

    assert ("eval/mean_reward", 12.5, 163_840) in writes
    assert ("eval/mean_episode_length", 150.0, 163_840) in writes
    assert ("eval/queue_wait_seconds", 0.75, 163_840) in writes
    assert ("eval/duration_seconds", 3.5, 163_840) in writes
    assert ("FinsROV/trajectory_tracking/tracking_error_p95_m", 0.12, 163_840) in writes


def test_unity_stats_callback_writes_trajectory_metrics_to_tensorboard(monkeypatch, tmp_path):
    writes = []

    class Writer:
        def add_scalar(self, tag, value, step):
            writes.append((tag, value, step))

        def flush(self):
            pass

    callback = trainer.UnityStatsTensorboardCallback(
        SimpleNamespace(num_envs=2),
        SimpleNamespace(log_dir=str(tmp_path / "logs")),
        "unity_train_metrics",
    )
    callback.tensorboard_writer = Writer()
    callback.num_timesteps = 256
    monkeypatch.setattr(
        trainer,
        "drain_unity_stats",
        lambda _env: {"FinsROV/trajectory_tracking/tracking_error_p95_m": 0.42},
    )

    callback._write_metrics()

    assert writes == [("FinsROV/trajectory_tracking/tracking_error_p95_m", 0.42, 256)]


def test_async_evaluation_promotes_the_first_completed_snapshot_to_best_model(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    snapshot_dir = checkpoint_dir / "eval_snapshots"
    snapshot_dir.mkdir(parents=True)
    log_dir.mkdir()
    snapshot = snapshot_dir / "snapshot_000000000005.zip"
    snapshot.write_bytes(b"evaluated-model")

    callback = trainer.AsyncEvaluationCallback(
        SimpleNamespace(num_envs=1),
        SimpleNamespace(
            log_dir=str(log_dir),
            checkpoint_dir=str(checkpoint_dir),
            tensorboard_dir=str(tmp_path / "tensorboard"),
            device="cpu",
        ),
        SimpleNamespace(
            env_config=SimpleNamespace(eval_num_episodes=1, parallel_mode="multi_area"),
            load_model_for_eval=lambda *_args, **_kwargs: object(),
        ),
        eval_freq=1,
    )
    monkeypatch.setattr(trainer, "evaluate_policy", lambda *_args, **_kwargs: ([3.5], [12]))

    callback.jobs.put((5, snapshot, 0.0))
    callback.jobs.put(None)
    callback._run()

    best_model = checkpoint_dir / "best_model.zip"
    assert best_model.read_bytes() == b"evaluated-model"
    assert not snapshot.exists()
    payload = json.loads((log_dir / "async_evaluations.jsonl").read_text(encoding="utf-8"))
    assert payload["is_best"] is True
    assert payload["mean_reward"] == 3.5


def test_async_evaluation_keeps_completed_snapshots_only_when_configured(monkeypatch, tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"
    log_dir = tmp_path / "logs"
    snapshot_dir = checkpoint_dir / "eval_snapshots"
    snapshot_dir.mkdir(parents=True)
    log_dir.mkdir()
    first_snapshot = snapshot_dir / "snapshot_000000000005.zip"
    second_snapshot = snapshot_dir / "snapshot_000000000010.zip"
    first_snapshot.write_bytes(b"first-model")
    second_snapshot.write_bytes(b"second-model")

    callback = trainer.AsyncEvaluationCallback(
        SimpleNamespace(num_envs=1),
        SimpleNamespace(
            log_dir=str(log_dir),
            checkpoint_dir=str(checkpoint_dir),
            tensorboard_dir=str(tmp_path / "tensorboard"),
            device="cpu",
        ),
        SimpleNamespace(
            env_config=SimpleNamespace(eval_num_episodes=1, parallel_mode="multi_area"),
            training_config=SimpleNamespace(keep_eval_snapshots=True),
            load_model_for_eval=lambda *_args, **_kwargs: object(),
        ),
        eval_freq=1,
    )
    monkeypatch.setattr(trainer, "evaluate_policy", lambda *_args, **_kwargs: ([3.5], [12]))

    callback.jobs.put((5, first_snapshot, 0.0))
    callback.jobs.put((10, second_snapshot, 0.0))
    callback.jobs.put(None)
    callback._run()

    assert (checkpoint_dir / "best_model.zip").read_bytes() == b"first-model"
    assert first_snapshot.read_bytes() == b"first-model"
    assert second_snapshot.read_bytes() == b"second-model"
