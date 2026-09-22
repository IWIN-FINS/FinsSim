"""BenchMARL task wrappers for FinsSim Unity MARL environments."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from benchmarl.environments import Task, TaskClass
from torchrl.data import Composite
from torchrl.envs import EnvBase

from finssim_marl.envs.torchrl import FinsSimTorchRLEnv, make_unity_torchrl_env
from finssim_marl.envs.unity.worker import EnvConfig


class FinsSimBenchMARLTask(Task, Enum):
    """BenchMARL task enum for FinsSim scenarios."""

    CHASING_3CHASE1 = None

    @staticmethod
    def associated_class() -> type[TaskClass]:
        return FinsSimBenchMARLTaskClass


class FinsSimBenchMARLTaskClass(TaskClass):
    """BenchMARL ``TaskClass`` backed by :class:`FinsSimTorchRLEnv`."""

    @staticmethod
    def env_name() -> str:
        return "finssim"

    def get_env_fun(
        self,
        num_envs: int,
        continuous_actions: bool,
        seed: Optional[int],
        device,
    ):
        if not continuous_actions:
            raise ValueError("FinsSim Unity MARL tasks currently expose continuous actions only.")

        config = dict(self.config)
        env_config = EnvConfig(
            worker_id=int(config.get("worker_id", 0)),
            env_base_port=int(config.get("env_base_port", 5005)),
            unity_env_binary_path=str(
                config.get("unity_env_binary_path")
                or config.get("env_path")
                or ""
            ),
            time_scale=float(config.get("time_scale", 10.0)),
            env_type=str(config.get("env_type", "3chase1")),
            no_graphics=bool(config.get("no_graphics", True)),
            herder_action_source=str(config.get("herder_action_source", "python_policy")),
            netter_action_source=str(config.get("netter_action_source", "python_policy")),
            prey_action_source=str(config.get("prey_action_source", "wrapper_escape")),
        )
        task_seed = 0 if seed is None else int(seed)

        def make_env() -> FinsSimTorchRLEnv:
            return make_unity_torchrl_env(env_config, seed=task_seed, device=device)

        return make_env

    def supports_continuous_actions(self) -> bool:
        return True

    def supports_discrete_actions(self) -> bool:
        return False

    def max_steps(self, env: EnvBase) -> int:
        wrapped = getattr(env, "_env", None)
        return int(getattr(wrapped, "max_steps", self.config.get("episode_limit", 900)))

    def has_render(self, env: EnvBase) -> bool:
        return False

    def group_map(self, env: EnvBase) -> dict[str, list[str]]:
        return getattr(env, "group_map", {"agents": []})

    def observation_spec(self, env: EnvBase) -> Composite:
        return env.observation_spec

    def info_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    def state_spec(self, env: EnvBase) -> Optional[Composite]:
        state = env.observation_spec["state"]
        return Composite({"state": state}, shape=env.batch_size, device=env.device)

    def action_spec(self, env: EnvBase) -> Composite:
        return env.action_spec

    def action_mask_spec(self, env: EnvBase) -> Optional[Composite]:
        return None

    @staticmethod
    def log_info(batch) -> dict[str, float]:
        logs: dict[str, float] = {}
        reward = batch.get(("next", "agents", "reward"), None)
        if reward is not None:
            logs["finssim/reward_mean"] = float(reward.float().mean().item())
        return logs


def build_task_from_config(config: dict[str, Any]) -> FinsSimBenchMARLTaskClass:
    """Build the default FinsSim BenchMARL task class from a config mapping."""
    return FinsSimBenchMARLTask.CHASING_3CHASE1.get_task(config=config)
