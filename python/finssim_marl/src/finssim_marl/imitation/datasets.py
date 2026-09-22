"""Persistence and validation for FinsSim MARL imitation datasets."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .schemas import MultiAgentDataset


ARRAY_FILE = "episodes.npz"
METADATA_FILE = "metadata.yaml"
STATS_FILE = "stats.json"


def validate_dataset(dataset: MultiAgentDataset) -> None:
    """Validate core multi-agent dataset shape invariants."""
    obs = np.asarray(dataset.obs)
    states = np.asarray(dataset.states)
    actions = np.asarray(dataset.actions)
    rewards = np.asarray(dataset.rewards)
    dones = np.asarray(dataset.dones)
    truncateds = np.asarray(dataset.truncateds)
    role_ids = np.asarray(dataset.role_ids)
    valid_steps = (
        None if dataset.valid_steps is None else np.asarray(dataset.valid_steps)
    )

    if obs.ndim != 4:
        raise ValueError(f"obs must have shape [E, T+1, N, obs_dim], got {obs.shape}")
    if states.ndim != 3:
        raise ValueError(f"states must have shape [E, T+1, state_dim], got {states.shape}")
    if actions.ndim != 4:
        raise ValueError(f"actions must have shape [E, T, N, action_dim], got {actions.shape}")

    episodes, obs_steps, n_agents, _obs_dim = obs.shape
    act_episodes, horizon, act_agents, _action_dim = actions.shape
    if act_episodes != episodes:
        raise ValueError(f"actions episode count {act_episodes} does not match obs {episodes}")
    if obs_steps != horizon + 1:
        raise ValueError(f"obs time dimension must be T+1 ({horizon + 1}), got {obs_steps}")
    if act_agents != n_agents:
        raise ValueError(f"actions agent count {act_agents} does not match obs {n_agents}")
    if states.shape[:2] != (episodes, obs_steps):
        raise ValueError(
            f"states leading dims must be {(episodes, obs_steps)}, got {states.shape[:2]}"
        )
    if rewards.shape not in {(episodes, horizon), (episodes, horizon, n_agents)}:
        raise ValueError(
            f"rewards must have shape [E,T] or [E,T,N], got {rewards.shape}"
        )
    if dones.shape != (episodes, horizon):
        raise ValueError(f"dones must have shape [E,T], got {dones.shape}")
    if truncateds.shape != (episodes, horizon):
        raise ValueError(f"truncateds must have shape [E,T], got {truncateds.shape}")
    if role_ids.shape != (n_agents,):
        raise ValueError(f"role_ids must have shape [N], got {role_ids.shape}")
    if valid_steps is not None and valid_steps.shape != (episodes, horizon):
        raise ValueError(f"valid_steps must have shape [E,T], got {valid_steps.shape}")


def transition_mask(dataset: MultiAgentDataset) -> np.ndarray:
    """Return the valid transition mask, treating older datasets as fully valid."""
    mask = dataset.valid_steps
    if mask is None:
        return np.ones(dataset.dones.shape, dtype=np.bool_)
    return np.asarray(mask, dtype=np.bool_)


def dataset_stats(dataset: MultiAgentDataset) -> dict[str, Any]:
    """Return compact dataset statistics for metadata and quick inspection."""
    validate_dataset(dataset)
    rewards = np.asarray(dataset.rewards, dtype=np.float32)
    mask = transition_mask(dataset)
    valid_count = int(mask.sum())
    if not valid_count:
        valid_rewards = np.asarray([], dtype=np.float32)
    elif rewards.ndim == 3:
        valid_rewards = rewards[mask].reshape(-1)
    else:
        valid_rewards = rewards[mask]
    return {
        "num_episodes": dataset.num_episodes,
        "horizon": dataset.horizon,
        "num_valid_steps": valid_count,
        "num_agents": dataset.num_agents,
        "obs_dim": dataset.obs_dim,
        "action_dim": dataset.action_dim,
        "reward_mean": float(np.mean(valid_rewards)) if valid_rewards.size else 0.0,
        "reward_std": float(np.std(valid_rewards)) if valid_rewards.size else 0.0,
    }


def save_dataset(dataset: MultiAgentDataset, output_dir: str | Path) -> Path:
    """Save a MARL imitation dataset directory."""
    validate_dataset(dataset)
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out / ARRAY_FILE,
        obs=np.asarray(dataset.obs, dtype=np.float32),
        states=np.asarray(dataset.states, dtype=np.float32),
        actions=np.asarray(dataset.actions, dtype=np.float32),
        rewards=np.asarray(dataset.rewards, dtype=np.float32),
        dones=np.asarray(dataset.dones, dtype=np.bool_),
        truncateds=np.asarray(dataset.truncateds, dtype=np.bool_),
        role_ids=np.asarray(dataset.role_ids, dtype=np.int64),
        valid_steps=transition_mask(dataset),
    )
    metadata = dict(dataset.metadata)
    metadata.setdefault("schema", "finssim_marl_imitation_v1")
    metadata.setdefault("stats", dataset_stats(dataset))
    with (out / METADATA_FILE).open("w", encoding="utf-8") as stream:
        yaml.safe_dump(metadata, stream, sort_keys=False, allow_unicode=True)
    with (out / STATS_FILE).open("w", encoding="utf-8") as stream:
        json.dump(dataset_stats(dataset), stream, indent=2)
    return out


def load_dataset(dataset_dir: str | Path) -> MultiAgentDataset:
    """Load and validate a MARL imitation dataset directory."""
    root = Path(dataset_dir).expanduser()
    arrays_path = root / ARRAY_FILE
    metadata_path = root / METADATA_FILE
    if not arrays_path.exists():
        raise FileNotFoundError(f"Missing MARL imitation array file: {arrays_path}")
    with np.load(arrays_path, allow_pickle=False) as arrays:
        metadata: dict[str, Any] = {}
        if metadata_path.exists():
            with metadata_path.open("r", encoding="utf-8") as stream:
                metadata = yaml.safe_load(stream) or {}
        dataset = MultiAgentDataset(
            obs=arrays["obs"],
            states=arrays["states"],
            actions=arrays["actions"],
            rewards=arrays["rewards"],
            dones=arrays["dones"],
            truncateds=arrays["truncateds"],
            role_ids=arrays["role_ids"],
            valid_steps=arrays["valid_steps"] if "valid_steps" in arrays else None,
            metadata=metadata,
        )
    validate_dataset(dataset)
    return dataset
