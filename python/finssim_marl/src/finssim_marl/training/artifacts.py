from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


def build_run_name(config_name: str, exp_name: str | None) -> str:
    return f"{config_name}__{exp_name}" if exp_name else config_name


def resolve_training_output_paths(config: Any, run_name: str, backend: str = "marl") -> Path:
    """Resolve standard output paths on a config-like object."""
    output_dir = Path(config.output_dir) if getattr(config, "output_dir", None) else Path("artifacts") / "runs" / backend / run_name
    config.output_dir = str(output_dir)

    if not getattr(config, "log_dir", None) or config.log_dir == "runs":
        config.log_dir = str(output_dir / "logs")
    if not getattr(config, "checkpoint_dir", None) or config.checkpoint_dir == "checkpoints":
        config.checkpoint_dir = str(output_dir / "checkpoints")
    if getattr(config, "tensorboard_dir", None) is None:
        config.tensorboard_dir = str(output_dir / "tensorboard")

    return output_dir


def _to_serializable(value: Any) -> Any:
    if dataclasses.is_dataclass(value):
        return {
            field.name: _to_serializable(getattr(value, field.name))
            for field in dataclasses.fields(value)
        }
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _to_serializable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_serializable(item) for item in value]
    if callable(value):
        return getattr(value, "__name__", repr(value))
    return value


def write_resolved_run_config(
    output_dir: str | Path,
    *,
    config_name: str,
    run_name: str,
    base_config: Any,
    final_config: Any,
    algorithm_config: Any,
    cli_args: Any,
    input_resolved_config: dict | None = None,
) -> Path:
    """Persist a reproducible snapshot of the fully resolved MARL run config."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_path / "resolved_config.yaml"
    payload = {
        "base_config": config_name,
        "run_name": run_name,
        "base_config_resolved": _to_serializable(base_config),
        "final_config": _to_serializable(final_config),
        "algorithm_config": _to_serializable(algorithm_config),
        "cli_args": _to_serializable(cli_args),
        "input_resolved_config": _to_serializable(input_resolved_config or {}),
        "artifacts": {
            "output_dir": str(output_path),
            "logs_dir": getattr(final_config, "log_dir", None),
            "checkpoints_dir": getattr(final_config, "checkpoint_dir", None),
            "tensorboard_dir": getattr(final_config, "tensorboard_dir", None),
        },
    }
    with snapshot_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return snapshot_path
