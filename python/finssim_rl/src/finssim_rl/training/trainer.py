"""
训练工具函数
"""
import os
import re
import shutil
import json
import queue
import threading
import time
from pathlib import Path
from typing import Any, Optional

from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecEnv
from torch.utils.tensorboard import SummaryWriter

from finssim_rl.training.args import Args
from finssim_rl.training.config import BaseConfig
from finssim_rl.training.one_chase_one_eval_metrics import (
    OneChaseOneDistanceTracker,
    is_one_chase_one_config,
)
from finssim_rl.envs.unity_stats import drain_unity_stats


class UnityStatsTensorboardCallback(BaseCallback):
    """Persist custom Unity train/eval metrics as a dedicated TensorBoard run."""

    def __init__(self, env: VecEnv, args: Args, run_name: str):
        super().__init__()
        self.env = env
        self.args = args
        tensorboard_root = getattr(args, "tensorboard_dir", None) or str(Path(args.log_dir).parent / "tensorboard")
        self.tensorboard_dir = Path(tensorboard_root) / run_name
        self.tensorboard_writer: SummaryWriter | None = None

    def _on_training_start(self) -> None:
        self.tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        drain_unity_stats(self.env)

    def _on_step(self) -> bool:
        self._write_metrics()
        return True

    def _on_training_end(self) -> None:
        self._write_metrics()

    def _write_metrics(self) -> None:
        writer = self.tensorboard_writer
        if writer is None:
            return
        for key, value in drain_unity_stats(self.env).items():
            writer.add_scalar(key, value, self.num_timesteps)
        writer.flush()

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None


class AsyncEvaluationCallback(BaseCallback):
    """Evaluate immutable model snapshots on a dedicated Unity VecEnv.

    The queue intentionally has no size limit: experiment configs request FIFO
    evaluation for every scheduled snapshot, rather than dropping old metrics.
    Snapshots are temporary queue inputs by default. Set
    ``training_config.keep_eval_snapshots`` to retain every evaluated snapshot
    for post-hoc analysis.
    """
    def __init__(self, eval_env: VecEnv, args: Args, config: BaseConfig, eval_freq: int):
        super().__init__()
        self.eval_env, self.args, self.config = eval_env, args, config
        self.eval_freq = max(1, int(eval_freq))
        self.jobs: queue.Queue[tuple[int, Path, float] | None] = queue.Queue()
        self.worker: threading.Thread | None = None
        self.error: BaseException | None = None
        self.metrics_path = Path(args.log_dir) / "async_evaluations.jsonl"
        self.snapshot_dir = Path(args.checkpoint_dir) / "eval_snapshots"
        self.best_model_path = Path(args.checkpoint_dir) / "best_model.zip"
        self.best_mean_reward = float("-inf")
        self.keep_eval_snapshots = bool(
            getattr(getattr(config, "training_config", None), "keep_eval_snapshots", False)
        )
        self.tensorboard_dir = Path(args.tensorboard_dir) / "async_eval"
        self.tensorboard_writer: SummaryWriter | None = None

    def _on_training_start(self) -> None:
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        if not self.keep_eval_snapshots:
            self._discard_snapshots()
        self.tensorboard_writer = SummaryWriter(log_dir=str(self.tensorboard_dir))
        drain_unity_stats(self.eval_env)
        self.worker = threading.Thread(target=self._run, name="finssim-async-eval", daemon=True)
        self.worker.start()

    def _on_step(self) -> bool:
        if self.error is not None:
            raise RuntimeError("asynchronous evaluation worker failed") from self.error
        if self.n_calls % self.eval_freq == 0:
            step = int(self.num_timesteps)
            snapshot = self.snapshot_dir / f"snapshot_{step:012d}.zip"
            self.model.save(str(snapshot))
            self.jobs.put((step, snapshot, time.time()))
        return True

    def _run(self) -> None:
        try:
            while True:
                job = self.jobs.get()
                if job is None:
                    return
                step, snapshot, submitted = job
                started = time.time()
                model = self.config.load_model_for_eval(str(snapshot), device=self.args.device, env=self.eval_env)
                distance_tracker = (
                    OneChaseOneDistanceTracker(self.eval_env.num_envs)
                    if is_one_chase_one_config(self.config)
                    else None
                )
                rewards, lengths = evaluate_policy(
                    model, self.eval_env, n_eval_episodes=self.config.env_config.eval_num_episodes, deterministic=True,
                    callback=(
                        lambda local_vars, _global_vars: distance_tracker.record_transition(
                            int(local_vars["i"]),
                            local_vars["new_observations"],
                            bool(local_vars["done"]),
                            local_vars["info"],
                            previous_observations=local_vars["observations"],
                        )
                        if distance_tracker is not None
                        else None
                    ),
                    return_episode_rewards=True, warn=False,
                )
                completed = time.time()
                unity_metrics = drain_unity_stats(self.eval_env)
                mean_reward = sum(rewards) / len(rewards)
                is_best = self._promote_best_snapshot(snapshot, mean_reward)
                payload = {
                    "snapshot_timestep": step,
                    "submitted_at": submitted,
                    "started_at": started,
                    "completed_at": completed,
                    "queue_wait_seconds": started - submitted,
                    "eval_mode": "asynchronous",
                    "parallel_mode": self.config.env_config.parallel_mode,
                    "effective_eval_slots": self.eval_env.num_envs,
                    "mean_reward": mean_reward,
                    "mean_episode_length": sum(lengths) / len(lengths),
                    "is_best": is_best,
                    "unity_metrics": unity_metrics,
                    "distance_metrics": distance_tracker.summary() if distance_tracker is not None else {},
                }
                with self.metrics_path.open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(payload) + "\n")
                self._write_tensorboard_metrics(payload, unity_metrics)
                if not self.keep_eval_snapshots and not is_best:
                    snapshot.unlink(missing_ok=True)
        except BaseException as exc:  # surfaced from the training callback
            self.error = exc

    def _promote_best_snapshot(self, snapshot: Path, mean_reward: float) -> bool:
        """Atomically update best_model.zip without losing requested history."""
        if mean_reward <= self.best_mean_reward:
            return False
        if self.keep_eval_snapshots:
            temporary_best = self.best_model_path.with_name(f".{self.best_model_path.name}.tmp")
            shutil.copy2(snapshot, temporary_best)
            temporary_best.replace(self.best_model_path)
        else:
            snapshot.replace(self.best_model_path)
        self.best_mean_reward = mean_reward
        return True

    def close(self) -> None:
        self.jobs.put(None)
        if self.worker is not None:
            self.worker.join()
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
            self.tensorboard_writer = None
        if not self.keep_eval_snapshots:
            self._discard_snapshots()
            try:
                self.snapshot_dir.rmdir()
            except OSError:
                # Leave a non-empty directory alone: it may contain a file
                # supplied outside this callback.
                pass
        if self.error is not None:
            raise RuntimeError("asynchronous evaluation worker failed") from self.error

    def _discard_snapshots(self) -> None:
        """Remove callback-owned queue files left by completed or aborted evals."""
        for snapshot in self.snapshot_dir.glob("snapshot_*.zip"):
            snapshot.unlink(missing_ok=True)
        temporary_best = self.best_model_path.with_name(f".{self.best_model_path.name}.tmp")
        temporary_best.unlink(missing_ok=True)

    def _write_tensorboard_metrics(
        self,
        payload: dict[str, Any],
        unity_metrics: dict[str, float],
    ) -> None:
        """Write completed asynchronous evaluations at their original snapshot step."""
        writer = self.tensorboard_writer
        if writer is None:
            return

        step = int(payload["snapshot_timestep"])
        writer.add_scalar("eval/mean_reward", float(payload["mean_reward"]), step)
        writer.add_scalar("eval/mean_episode_length", float(payload["mean_episode_length"]), step)
        writer.add_scalar("eval/is_best", 1.0 if payload.get("is_best") else 0.0, step)
        writer.add_scalar("eval/queue_wait_seconds", float(payload["queue_wait_seconds"]), step)
        writer.add_scalar("eval/duration_seconds", float(payload["completed_at"]) - float(payload["started_at"]), step)
        for key, value in unity_metrics.items():
            writer.add_scalar(key, value, step)
        for key, value in payload.get("distance_metrics", {}).items():
            writer.add_scalar(key, float(value), step)
        writer.flush()


def find_latest_checkpoint(model_save_path: str) -> Optional[str]:
    """
    找到最新的检查点文件
    
    Args:
        model_save_path: 模型保存路径
    
    Returns:
        最新的检查点路径，如果没有则返回 None
    """
    model_save_path_obj = Path(model_save_path)
    if not model_save_path_obj.exists():
        return None

    # CheckpointCallback names checkpoints as
    #   <prefix>_underwater_model_<steps>_steps.zip
    # Lexicographic ordering is wrong once the step counter changes digit width.
    step_pattern = re.compile(r"_underwater_model_(\d+)_steps\.zip$")
    checkpoint_files = []
    for candidate in model_save_path_obj.glob("*_underwater_model_*_steps.zip"):
        match = step_pattern.search(candidate.name)
        if match:
            checkpoint_files.append((int(match.group(1)), candidate))

    if checkpoint_files:
        return str(max(checkpoint_files, key=lambda item: (item[0], item[1].name))[1])

    # A completed/interrupted run may only have an explicit final model or an
    # EvalCallback best model. They are valid fallbacks, in that order.
    for fallback_name in ("final_model.zip", "best_model.zip"):
        fallback = model_save_path_obj / fallback_name
        if fallback.is_file():
            return str(fallback)

    return None


def setup_directories(args: Args) -> None:
    """
    设置日志和检查点目录
    
    Args:
        args: 命令行参数
    """
    if not args.checkpoint_dir or not args.log_dir:
        raise ValueError("checkpoint_dir and log_dir must not be None")
    
    model_dir_exists = os.path.exists(args.checkpoint_dir)
    log_dir_exists = os.path.exists(args.log_dir)

    if (model_dir_exists or log_dir_exists) and not args.resume and not args.overwrite:
        raise RuntimeError(
            f"Error: Experiment '{args.exp_name}' already exists!\n"
            f"Please use one of the following options:\n"
            f"  1. --resume: Resume training from checkpoint\n"
            f"  2. --overwrite: Overwrite existing files\n"
            f"  3. Use a different --exp-name"
        )

    if (model_dir_exists or log_dir_exists) and args.overwrite and not args.resume:
        print(f"Warning: Overwriting existing experiment '{args.exp_name}'")
        if model_dir_exists:
            shutil.rmtree(args.checkpoint_dir)
        if log_dir_exists:
            shutil.rmtree(args.log_dir)

    os.makedirs(args.log_dir, exist_ok=True)
    os.makedirs(args.checkpoint_dir, exist_ok=True)


def make_callbacks(
    env: VecEnv,
    args: Args,
    config: BaseConfig,
    *,
    eval_env: VecEnv | None = None,
) -> CallbackList:
    """
    创建训练回调函数列表
    
    Args:
        env: 向量化环境
        args: 命令行参数
        config: 配置对象
    
    Returns:
        回调函数列表
    """
    if not args.checkpoint_dir or not args.log_dir:
        raise ValueError("checkpoint_dir and log_dir must not be None")
    
    training_config = config.training_config
    num_envs = env.num_envs
    
    # CheckpointCallback：定期保存模型
    checkpoint_callback = CheckpointCallback(
        save_freq=max(int(training_config.checkpoint_freq / num_envs), 1),
        save_path=args.checkpoint_dir,
        name_prefix=f"{config.model_type.lower()}_underwater_model",
    )

    # Historical scripts retain SB3's synchronous fallback.
    callback_eval_env = eval_env or env
    runtime_env_config = getattr(config, "env_config", None)
    train_metrics_callback = UnityStatsTensorboardCallback(env, args, "unity_train_metrics")
    if eval_env is not None and runtime_env_config is not None and getattr(runtime_env_config, "eval_mode", "asynchronous") == "asynchronous":
        return CallbackList([
            checkpoint_callback,
            train_metrics_callback,
            AsyncEvaluationCallback(
                eval_env, args, config,
                eval_freq=max(int(training_config.eval_freq / num_envs), 1),
            ),
        ])
    eval_callback = EvalCallback(
        callback_eval_env,
        best_model_save_path=args.checkpoint_dir,
        log_path=args.log_dir,
        eval_freq=max(int(training_config.eval_freq / num_envs), 1),
        deterministic=True,
        render=False,
    )
    
    return CallbackList([
        checkpoint_callback,
        train_metrics_callback,
        eval_callback,
        UnityStatsTensorboardCallback(callback_eval_env, args, "unity_eval_metrics"),
    ])


def create_model(
    env: VecEnv,
    config: BaseConfig,
    args: Args,
    load_checkpoint: Optional[str] = None,
):
    """
    创建强化学习模型
    
    Args:
        env: 向量化环境
        config: 配置对象
        args: 命令行参数
        load_checkpoint: 要加载的检查点路径
    
    Returns:
        训练好的模型对象
    """
    # 委托给配置对象创建模型（工厂方法由各模型配置实现）
    return config.create_model(env, args, load_checkpoint)


def find_model_for_eval(
    checkpoint_dir: str,
    model_type: str = "best",
) -> Optional[str]:
    """
    查找用于评估的模型文件
    
    Args:
        checkpoint_dir: 检查点目录
        model_type: 模型类型，可选值：
            - 'best': best_model.zip（评估时的最佳模型）
            - 'latest': 最新的检查点
            - 'final': final_model.zip（训练完成后保存的最终模型）
    
    Returns:
        模型路径，如果没有找到则返回 None
    """
    checkpoint_dir_path = Path(checkpoint_dir)
    
    if not checkpoint_dir_path.exists():
        return None
    
    if model_type == "best":
        best_model_path = checkpoint_dir_path / "best_model.zip"
        if best_model_path.exists():
            return str(best_model_path)
    
    elif model_type == "final":
        final_model_path = checkpoint_dir_path / "final_model.zip"
        if final_model_path.exists():
            return str(final_model_path)
    
    elif model_type == "latest":
        checkpoint_files = sorted(checkpoint_dir_path.glob("*_underwater_model_*.zip"))
        if checkpoint_files:
            return str(checkpoint_files[-1])
    
    # 如果指定类型没有找到，尝试其他类型（降级）
    fallback_order = {
        "best": ["final", "latest"],
        "final": ["best", "latest"],
        "latest": ["best", "final"],
    }
    
    for fallback_type in fallback_order.get(model_type, []):
        model_path = find_model_for_eval(checkpoint_dir, fallback_type)
        if model_path:
            print(f"Warning: {model_type} model not found, using {fallback_type} instead")
            return model_path
    
    return None


def load_model_for_eval(
    model_path: str,
    config,
    env=None,
    device: str = "auto",
):
    """
    加载用于评估的模型
    
    Args:
        model_path: 模型文件路径
        config: 配置对象
        device: 计算设备
    
    Returns:
        加载的模型对象
    """
    return config.load_model_for_eval(model_path, device=device, env=env)
