from __future__ import annotations

import json
import platform
import shlex
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import yaml


def get_git_commit(cwd: str | Path | None = None) -> str | None:
    """Return the current git commit for cwd, or None outside a git tree."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path.cwd() if cwd is None else Path(cwd),
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def _to_builtin(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_builtin(item) for item in value]
    return value


def write_run_metadata(
    run_dir: str | Path,
    *,
    backend: str,
    config_path: str | Path,
    run_name: str,
    seed: int,
    unity_env_path: str | None,
    num_envs: int | None,
    env_base_port: int | None,
    resolved_config: dict[str, Any],
    command: Sequence[str],
    git_root: str | Path | None = None,
) -> dict[str, Any]:
    """Write resolved_config.yaml, metadata.json, and command.txt for a run."""
    output_dir = Path(run_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_config_path = output_dir / "resolved_config.yaml"
    with resolved_config_path.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(_to_builtin(resolved_config), stream, sort_keys=False)

    metadata = {
        "backend": backend,
        "config_path": str(Path(config_path).expanduser().resolve()),
        "run_name": run_name,
        "python_version": platform.python_version(),
        "seed": seed,
        "unity_env_path": unity_env_path,
        "num_envs": num_envs,
        "env_base_port": env_base_port,
        "git_commit": get_git_commit(git_root),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "resolved_config_path": str(resolved_config_path),
    }
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")

    command_path = output_dir / "command.txt"
    command_path.write_text(shlex.join([str(part) for part in command]) + "\n", encoding="utf-8")

    return metadata
