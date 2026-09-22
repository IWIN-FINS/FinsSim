from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

ParallelMode = Literal["multi_area", "multi_binary"]
EvalMode = Literal["asynchronous", "serial"]


@dataclass(frozen=True)
class UnityRuntimeConfig:
    """Backend-neutral Unity binary runtime settings."""

    env_path: str | None = None
    use_editor: bool = False
    parallel_mode: ParallelMode = "multi_area"
    env_base_port: int = 5005
    port_offset: int = 0
    no_graphics: bool = True
    timeout_wait: int = 360
    seed: int = 42
    environment_parameters: dict[str, float] = field(default_factory=dict)
    unity_additional_args: list[str] = field(default_factory=list)

    def worker_id(self, index: int) -> int:
        if index < 0:
            raise ValueError("worker index must be non-negative")
        return self.port_offset + index

    def port_for(self, index: int) -> int:
        return self.env_base_port + self.worker_id(index)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "UnityRuntimeConfig":
        """Build a runtime config from a YAML mapping."""
        if data is None:
            return cls()

        allowed = {
            "env_path",
            "use_editor",
            "parallel_mode",
            "env_base_port",
            "port_offset",
            "no_graphics",
            "timeout_wait",
            "seed",
            "environment_parameters",
            "unity_additional_args",
        }
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown unity config fields: {', '.join(unknown)}")

        environment_parameters_raw = data.get("environment_parameters", {})
        if environment_parameters_raw is None:
            environment_parameters = {}
        elif isinstance(environment_parameters_raw, dict):
            environment_parameters = {
                str(key): float(value)
                for key, value in environment_parameters_raw.items()
            }
        else:
            raise ValueError("unity.environment_parameters must be a mapping")

        unity_additional_args_raw = data.get("unity_additional_args", [])
        if unity_additional_args_raw is None:
            unity_additional_args = []
        elif isinstance(unity_additional_args_raw, list):
            unity_additional_args = [str(value) for value in unity_additional_args_raw]
        else:
            raise ValueError("unity.unity_additional_args must be a list")

        runtime = cls(
            env_path=data.get("env_path"),
            use_editor=bool(data.get("use_editor", cls.use_editor)),
            parallel_mode=str(data.get("parallel_mode", cls.parallel_mode)),
            env_base_port=int(data.get("env_base_port", cls.env_base_port)),
            port_offset=int(data.get("port_offset", cls.port_offset)),
            no_graphics=bool(data.get("no_graphics", cls.no_graphics)),
            timeout_wait=int(data.get("timeout_wait", cls.timeout_wait)),
            seed=int(data.get("seed", cls.seed)),
            environment_parameters=environment_parameters,
            unity_additional_args=unity_additional_args,
        )
        if runtime.parallel_mode not in {"multi_area", "multi_binary"}:
            raise ValueError("env.unity.parallel_mode must be 'multi_area' or 'multi_binary'")
        return runtime


@dataclass(frozen=True)
class UnityRoleConfig:
    """Role-specific Unity settings, inherited from ``env.unity`` when omitted."""

    num_envs: int = 1
    time_scale: float = 10.0
    no_graphics: bool | None = None
    timeout_wait: int | None = None
    mode: EvalMode = "asynchronous"
    num_episodes: int = 5

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None, *, role: str) -> "UnityRoleConfig":
        if data is None:
            return cls()
        allowed = {"num_envs", "time_scale", "no_graphics", "timeout_wait"}
        if role == "eval":
            allowed.update({"mode", "num_episodes"})
        unknown = sorted(set(data) - allowed)
        if unknown:
            raise ValueError(f"Unknown env.{role} config fields: {', '.join(unknown)}")
        mode = str(data.get("mode", cls.mode))
        if role == "eval" and mode not in {"asynchronous", "serial"}:
            raise ValueError("env.eval.mode must be 'asynchronous' or 'serial'")
        result = cls(
            num_envs=int(data.get("num_envs", cls.num_envs)),
            time_scale=float(data.get("time_scale", cls.time_scale)),
            no_graphics=(None if "no_graphics" not in data else bool(data["no_graphics"])),
            timeout_wait=(None if "timeout_wait" not in data else int(data["timeout_wait"])),
            mode=mode,
            num_episodes=int(data.get("num_episodes", cls.num_episodes)),
        )
        if result.num_envs < 1:
            raise ValueError(f"env.{role}.num_envs must be positive")
        if role == "eval" and result.num_episodes < 1:
            raise ValueError("env.eval.num_episodes must be positive")
        return result
