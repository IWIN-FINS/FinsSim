from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any

import yaml


def resolve_training_output_paths(args: Any, backend: str = "rl") -> Path:
    """Resolve standard output paths on an args-like object."""
    exp_name = getattr(args, "exp_name", "") or getattr(args, "config", "default")
    output_dir = Path(args.output_dir) if getattr(args, "output_dir", None) else Path("artifacts") / "runs" / backend / exp_name
    args.output_dir = str(output_dir)

    if getattr(args, "log_dir", None) is None:
        args.log_dir = str(output_dir / "logs")
    if getattr(args, "checkpoint_dir", None) is None:
        args.checkpoint_dir = str(output_dir / "checkpoints")
    if getattr(args, "tensorboard_dir", None) is None:
        args.tensorboard_dir = str(output_dir / "tensorboard")

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
    base_config: Any,
    final_config: Any,
    cli_args: Any,
) -> Path:
    """Persist a reproducible snapshot of the fully resolved RL run config."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    snapshot_path = output_path / "resolved_config.yaml"
    payload = {
        "base_config": config_name,
        "base_config_resolved": _to_serializable(base_config),
        "final_config": _to_serializable(final_config),
        "cli_args": _to_serializable(cli_args),
        "artifacts": {
            "output_dir": str(output_path),
            "logs_dir": getattr(cli_args, "log_dir", None),
            "checkpoints_dir": getattr(cli_args, "checkpoint_dir", None),
            "tensorboard_dir": getattr(cli_args, "tensorboard_dir", None),
        },
    }
    with snapshot_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(payload, stream, allow_unicode=True, sort_keys=False)
    return snapshot_path
