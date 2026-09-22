"""Public Unity VecEnv factories for sibling FinsSim backends."""

from __future__ import annotations

from dataclasses import replace

from finssim_rl.models import make_unity_env, make_unity_env_for_eval, make_unity_training_areas_env
from finssim_rl.training.config import BaseEnvironmentConfig
from finssim_rl.training.managed_subproc_vec_env import ManagedSubprocVecEnv


def make_train_vec_env(env_config: BaseEnvironmentConfig):
    """Create the configured training VecEnv without assuming a model type."""

    if not env_config.env_path and not env_config.use_editor:
        raise ValueError("env_config.env_path is required when Unity Editor mode is disabled")
    if env_config.num_envs < 1:
        raise ValueError("env_config.num_envs must be positive")
    if env_config.parallel_mode == "multi_area":
        return make_unity_training_areas_env(env_config.env_path, env_config)
    if env_config.parallel_mode == "multi_binary":
        return ManagedSubprocVecEnv(
            [make_unity_env(env_config.env_path, rank, env_config) for rank in range(env_config.num_envs)]
        )
    raise ValueError("parallel_mode must be multi_area or multi_binary")


def make_eval_vec_env(
    env_config: BaseEnvironmentConfig,
    *,
    worker_id: int | None = None,
):
    """Create an independent evaluation VecEnv using the configured ABI."""

    if not env_config.env_path and not env_config.use_editor:
        raise ValueError("env_config.env_path is required when Unity Editor mode is disabled")
    if env_config.eval_num_envs < 1:
        raise ValueError("env_config.eval_num_envs must be positive")
    effective = replace(env_config, num_envs=env_config.eval_num_envs, time_scale=env_config.eval_time_scale)
    start_worker = effective.port_offset if worker_id is None else int(worker_id)
    if effective.parallel_mode == "multi_area":
        return make_unity_training_areas_env(
            effective.env_path,
            effective,
            worker_id=start_worker,
            time_scale=effective.eval_time_scale,
        )
    if effective.parallel_mode == "multi_binary":
        return ManagedSubprocVecEnv(
            [
                make_unity_env_for_eval(
                    effective.env_path,
                    effective,
                    no_graphics=effective.no_graphics,
                    time_scale=effective.eval_time_scale,
                    worker_id=start_worker + rank,
                )
                for rank in range(effective.eval_num_envs)
            ]
        )
    raise ValueError("parallel_mode must be multi_area or multi_binary")
