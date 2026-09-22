from __future__ import annotations

from dataclasses import dataclass, field, replace
import os
from pathlib import Path
from typing import Any, Literal

import yaml

from finssim_core.experiment import ExperimentConfig
from finssim_core.runtime import UnityRoleConfig, UnityRuntimeConfig

BackendName = Literal["rl", "marl", "benchmarl", "marllib"]


def _workspace_root(config_path: Path) -> Path:
    """Locate the repository root for portable, root-relative YAML paths."""
    for candidate in (config_path.parent, *config_path.parents):
        if (candidate / "pyproject.toml").is_file() and (candidate / "configs").is_dir():
            return candidate
    return config_path.parent


def _resolve_workspace_path(value: str, *, workspace_root: Path) -> str:
    path = Path(os.path.expandvars(value)).expanduser()
    return str(path.resolve() if path.is_absolute() else (workspace_root / path).resolve())


@dataclass(frozen=True)
class TrainerConfig:
    """Backend selector and backend-specific override payload."""

    backend: BackendName
    overrides: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "TrainerConfig":
        if data is None:
            raise ValueError("Missing required 'trainer' section")

        backend = data.get("backend")
        if backend not in {"rl", "marl", "benchmarl", "marllib"}:
            raise ValueError("trainer.backend must be one of: rl, marl, benchmarl, marllib")

        overrides = data.get("overrides") or {}
        if not isinstance(overrides, dict):
            raise ValueError("trainer.overrides must be a mapping")

        return cls(backend=backend, overrides=dict(overrides))


@dataclass(frozen=True)
class FinsSimExperimentConfig:
    """Parsed FinsSim YAML experiment file."""

    path: Path
    base_config: str
    experiment: ExperimentConfig
    unity: UnityRuntimeConfig
    train: UnityRoleConfig
    eval: UnityRoleConfig
    trainer: TrainerConfig
    raw: dict[str, Any]
    unity_explicit_fields: tuple[str, ...] = field(default_factory=tuple)

    @property
    def backend(self) -> BackendName:
        return self.trainer.backend

    def has_unity_override(self, field_name: str) -> bool:
        return field_name in self.unity_explicit_fields

    def unity_overrides(self) -> dict[str, Any]:
        return {
            field_name: getattr(self.unity, field_name)
            for field_name in self.unity_explicit_fields
        }


def load_experiment_config(path: str | Path) -> FinsSimExperimentConfig:
    """Load and validate a FinsSim experiment YAML file."""
    config_path = Path(path).expanduser().resolve()
    workspace_root = _workspace_root(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream) or {}

    if not isinstance(raw, dict):
        raise ValueError(f"Experiment file must contain a YAML mapping: {config_path}")

    base_config = raw.get("base_config")
    if not base_config:
        raise ValueError("base_config is required")

    experiment = ExperimentConfig.from_mapping(raw.get("experiment"))
    env_raw = raw.get("env")
    legacy_unity_explicit_fields: tuple[str, ...] | None = None
    # MARL's current 3Chase1 runtime is a replicated TrainingArea scene. Flat
    # MARL YAML therefore inherits multi_area, while other legacy backends keep
    # their historical multi-binary behavior unless they explicitly opt in.
    if env_raw is None:
        legacy = dict(raw.get("unity") or {})
        legacy_num_envs = legacy.pop("num_envs", 1)
        legacy_num_areas = legacy.pop("num_areas", None)
        legacy_eval_envs = legacy.pop("num_eval_envs", 1)
        legacy_eval_areas = legacy.pop("num_eval_areas", None)
        legacy_time_scale = legacy.pop("time_scale", 10.0)
        legacy_unity_explicit_fields = tuple(legacy.keys())
        legacy_parallel_mode = (
            "multi_area"
            if legacy_num_areas is not None or (raw.get("trainer") or {}).get("backend") == "marl"
            else "multi_binary"
        )
        env_raw = {
            "unity": {**legacy, "parallel_mode": legacy_parallel_mode},
            "train": {"num_envs": legacy_num_areas if legacy_num_areas is not None else legacy_num_envs, "time_scale": legacy_time_scale},
            "eval": {"num_envs": legacy_eval_areas if legacy_eval_areas is not None else legacy_eval_envs, "time_scale": legacy_time_scale},
        }
    if not isinstance(env_raw, dict):
        raise ValueError("env must be a mapping")
    unknown_env = sorted(set(env_raw) - {"unity", "train", "eval"})
    if unknown_env:
        raise ValueError(f"Unknown env config fields: {', '.join(unknown_env)}")
    unity_raw = env_raw.get("unity")
    if isinstance(unity_raw, dict) and ({"num_areas", "num_eval_areas"} & set(unity_raw)):
        raise ValueError("num_areas/num_eval_areas were removed; use env.unity.parallel_mode and env.train/env.eval.num_envs")
    unity = UnityRuntimeConfig.from_mapping(unity_raw)
    if unity.env_path:
        unity = replace(
            unity,
            env_path=_resolve_workspace_path(str(unity.env_path), workspace_root=workspace_root),
        )
    unity_explicit_fields = (
        legacy_unity_explicit_fields
        if legacy_unity_explicit_fields is not None
        else (tuple(unity_raw.keys()) if isinstance(unity_raw, dict) else ())
    )
    train = UnityRoleConfig.from_mapping(env_raw.get("train"), role="train")
    eval_config = UnityRoleConfig.from_mapping(env_raw.get("eval"), role="eval")
    if "seed" not in (unity_raw or {}):
        unity = UnityRuntimeConfig(
            env_path=unity.env_path,
            use_editor=unity.use_editor,
            parallel_mode=unity.parallel_mode,
            env_base_port=unity.env_base_port,
            port_offset=unity.port_offset,
            no_graphics=unity.no_graphics,
            timeout_wait=unity.timeout_wait,
            seed=experiment.seed,
            environment_parameters=unity.environment_parameters,
            unity_additional_args=unity.unity_additional_args,
        )
    trainer = TrainerConfig.from_mapping(raw.get("trainer"))

    return FinsSimExperimentConfig(
        path=config_path,
        base_config=str(base_config),
        experiment=experiment,
        unity=unity,
        train=train,
        eval=eval_config,
        trainer=trainer,
        unity_explicit_fields=unity_explicit_fields,
        raw=raw,
    )
