"""Compile human-friendly reward YAML sections into Unity float parameters."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

import yaml


DEFAULT_REWARD_PROTOCOL = "one_chase_one_reward_v1"


@dataclass(frozen=True)
class CompiledRewardConfig:
    protocol: str
    version: int
    mode_name: str
    mode_id: int
    environment_parameters: dict[str, float]


def _find_workspace_root(start: Path | None = None) -> Path:
    current = (start or Path.cwd()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs" / "reward_protocols").exists():
            return candidate
    return current


def _load_protocol(protocol: str, *, workspace_root: Path | None = None) -> dict[str, Any]:
    root = workspace_root or _find_workspace_root()
    path = root / "configs" / "reward_protocols" / f"{protocol}.yaml"
    if not path.exists():
        raise ValueError(f"Unknown reward protocol {protocol!r}: {path} does not exist")
    with path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Reward protocol must be a YAML mapping: {path}")
    return payload


def _float_mapping(values: Mapping[str, Any], *, context: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for key, value in values.items():
        if not isinstance(value, Real):
            raise TypeError(f"{context}.{key} must be numeric, got {type(value).__name__}")
        result[str(key)] = float(value)
    return result


def compile_reward_config(
    reward: Mapping[str, Any] | None,
    *,
    workspace_root: Path | None = None,
) -> CompiledRewardConfig | None:
    """Compile top-level `reward:` YAML into ML-Agents environment parameters.

    The YAML-facing API uses strings such as `distance_only`. Unity receives
    numeric ids because ML-Agents EnvironmentParametersChannel is float-based.
    """
    if reward is None:
        return None
    if not isinstance(reward, Mapping):
        raise TypeError("reward must be a mapping")

    protocol_name = str(reward.get("protocol", DEFAULT_REWARD_PROTOCOL))
    protocol = _load_protocol(protocol_name, workspace_root=workspace_root)
    prefix = str(protocol.get("prefix", ""))
    if not prefix:
        raise ValueError(f"Reward protocol {protocol_name!r} is missing prefix")

    version = int(protocol.get("version", 0))
    modes_raw = protocol.get("modes") or {}
    if not isinstance(modes_raw, Mapping):
        raise ValueError(f"Reward protocol {protocol_name!r} modes must be a mapping")
    modes = {str(name): int(value) for name, value in modes_raw.items()}

    default_mode = str(protocol.get("default_mode", next(iter(modes), "")))
    mode_name = str(reward.get("mode", default_mode))
    if mode_name not in modes:
        valid = ", ".join(sorted(modes))
        raise ValueError(f"Unknown reward.mode={mode_name!r}; valid modes: {valid}")
    mode_id = modes[mode_name]

    defaults_raw = protocol.get("parameters") or {}
    if not isinstance(defaults_raw, Mapping):
        raise ValueError(f"Reward protocol {protocol_name!r} parameters must be a mapping")
    parameters = _float_mapping(defaults_raw, context=f"reward_protocols.{protocol_name}.parameters")

    overrides_raw = reward.get("parameters") or {}
    if not isinstance(overrides_raw, Mapping):
        raise TypeError("reward.parameters must be a mapping")
    unknown = sorted(set(str(key) for key in overrides_raw) - set(parameters))
    if unknown:
        valid = ", ".join(sorted(parameters))
        raise ValueError(f"Unknown reward.parameters keys: {', '.join(unknown)}. Valid keys: {valid}")
    parameters.update(_float_mapping(overrides_raw, context="reward.parameters"))

    env_params = {
        f"{prefix}protocol_version": float(version),
        f"{prefix}mode": float(mode_id),
    }
    for key, value in parameters.items():
        env_params[f"{prefix}{key}"] = float(value)

    return CompiledRewardConfig(
        protocol=protocol_name,
        version=version,
        mode_name=mode_name,
        mode_id=mode_id,
        environment_parameters=env_params,
    )


def compile_reward_config_from_resolved_config(
    resolved_config: Mapping[str, Any] | None,
    *,
    workspace_root: Path | None = None,
) -> CompiledRewardConfig | None:
    if not resolved_config:
        return None
    return compile_reward_config(
        resolved_config.get("reward"),
        workspace_root=workspace_root,
    )


def resolved_unity_environment_parameters(
    resolved_config: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Extract explicit Unity float parameters from platform or backend snapshots.

    ``finssim rl`` writes the platform launch snapshot with a top-level
    ``unity.environment_parameters`` mapping.  The RL backend subsequently
    replaces that file with its own ``final_config`` snapshot.  Supporting both
    shapes keeps direct backend resume runs reproducible as well.
    """
    if not resolved_config:
        return {}

    candidates: list[Any] = []
    final_config = resolved_config.get("final_config")
    if isinstance(final_config, Mapping):
        env_config = final_config.get("env_config")
        if isinstance(env_config, Mapping):
            candidates.append(env_config.get("environment_parameters"))

    unity = resolved_config.get("unity")
    if isinstance(unity, Mapping):
        candidates.append(unity.get("environment_parameters"))

    parameters: dict[str, float] = {}
    for candidate in candidates:
        if candidate is None:
            continue
        if not isinstance(candidate, Mapping):
            raise TypeError("unity.environment_parameters must be a mapping")
        parameters.update(_float_mapping(candidate, context="unity.environment_parameters"))
    return parameters


def load_resolved_config(path: str | Path | None) -> dict[str, Any] | None:
    if not path:
        return None
    resolved_path = Path(path).expanduser()
    if not resolved_path.exists():
        return None
    with resolved_path.open("r", encoding="utf-8") as stream:
        payload = yaml.safe_load(stream) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Resolved config must contain a YAML mapping: {resolved_path}")
    return payload
