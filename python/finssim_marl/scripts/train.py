"""
Unified training script for MARL algorithms.

Usage:
    # Start new training (uses config_name as directory name)
    python scripts/train.py

    # Start with specific config and experiment name (creates runs/chasing_3_chase_1__my_experiment/)
    python scripts/train.py --config chasing_3_chase_1 --exp-name my_experiment

    # Resume from latest checkpoint in exp_name directory
    python scripts/train.py --exp-name my_experiment --auto-resume

    # Resume from explicit checkpoint path
    python scripts/train.py --resume-from checkpoints/chasing_3_chase_1__my_experiment/step_1000.pt

    # Overwrite existing run
    python scripts/train.py --exp-name my_experiment --overwrite

For Unity environments with Xvfb (headless):
    xvfb-run --auto-servernum --server-args='-screen 0 1280x1024x24' \
        python scripts/train.py --config chasing_3_chase_1 --env-base-port 9969
"""

from __future__ import annotations

import os
import signal
import sys
import random
import math
import copy
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import tyro
import yaml
from tqdm import tqdm
import debugpy

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Two complete TriNetCapture wrapper episodes (2 x 1200 steps) are enough to
# verify both startup and post-episode reset paths without starting a real run.
SMOKE_TEST_TIMESTEPS = 2_400

from finssim_marl.algorithms.networks.rollout_buffer import RolloutBuffer
from finssim_marl.envs.unity.worker import env_worker, EnvConfig
from finssim_marl.utils.chase_metrics import (
    nearest_chaser_prey_distance_m,
    summarize_episode_distances,
)
from finssim_marl.utils.trinet_metrics import summarize_trinet_episode_metrics


def _summarize_task_distances(config, initial, minimum, final, *, prefix=""):
    if config.env_type == "trinet_capture":
        return summarize_trinet_episode_metrics(initial, minimum, final, prefix=prefix)
    return summarize_episode_distances(initial, minimum, final, prefix=prefix)
from finssim_marl.utils.logging import Logger, LoggerConfig
from finssim_marl.utils.checkpoint import CheckpointManager, CheckpointConfig
from finssim_marl.training.config import merge_args_to_dict, get_config, BaseConfig, TrainConfig
from finssim_marl.training.artifacts import resolve_training_output_paths, write_resolved_run_config
from finssim_marl.training.curriculum import CurriculumScheduler, CurriculumUpdate, normalize_environment_parameters
from finssim_marl.training.async_evaluation import AsyncEvaluationResult, AsyncEvaluationWorker


@dataclass
class TrainArgs:
    """CLI arguments for training script."""
    config: Optional[str] = None # 必须设为None，否则会覆写config里面设定好的配置！
    exp_name: Optional[str] = None
    output_dir: Optional[str] = None
    resolved_config: Optional[str] = None
    env_type: Optional[str] = None
    env_path: Optional[str] = None
    num_envs: Optional[int] = None
    num_eval_envs: Optional[int] = None
    eval_time_scale: Optional[float] = None
    eval_mode: Optional[str] = None
    keep_eval_snapshots: Optional[bool] = None
    test: bool = False
    batch_size: Optional[int] = None
    total_timesteps: Optional[int] = None
    device: Optional[str] = None
    seed: Optional[int] = None
    overwrite: bool = False  # Pass --overwrite on CLI to set True
    auto_resume: bool = False  # Pass --auto-resume to auto-find latest checkpoint and resume
    resume_from: Optional[str] = None  # Explicit checkpoint path to resume from
    env_base_port: Optional[int] = None
    timeout_wait: Optional[int] = None
    time_scale: Optional[float] = None
    parallel_mode: Optional[str] = None
    env_restart_attempts: Optional[int] = None
    show_graphics: bool = False
    controller_backend: Optional[str] = None
    target_body_delta_limits: Optional[Tuple[float, float, float]] = None
    enable_yaw_control: bool = False
    yaw_error_limit_deg: Optional[float] = None
    critic_variant: Optional[str] = None
    critic_attention_heads: Optional[int] = None
    body_pid_control_rate_hz: Optional[float] = None
    body_pid_params: Optional[Tuple[float, float, float, float, float, float, float, float, float]] = None
    body_pid_yaw_pid_params: Optional[Tuple[float, float, float]] = None
    body_pid_integral_decay: Optional[float] = None
    body_pid_yaw_deadband_deg: Optional[float] = None
    debug: bool = False  # Pass --debug to enable debugpy for VS Code attach


@dataclass
class ManagedEnvSlot:
    """Tracks one Unity worker process and its restart metadata."""

    slot_id: int
    group: str
    env_config: EnvConfig
    seed: int
    conn: Optional[Connection] = None
    process: Optional[Process] = None
    disabled: bool = False


def _with_test_suffix(exp_name: str | None) -> str | None:
    if not exp_name:
        return exp_name
    return exp_name if exp_name.endswith("_test") else f"{exp_name}_test"


def merge_cli_args(target, args: TrainArgs, fields: Tuple[str | Tuple[str, str], ...]):
    """Merge CLI overrides into a config-like object.

    Optional arguments only override when they are not None. Boolean flags only
    override when True, because False is also the tyro default for absent flags.
    """
    for field in fields:
        if isinstance(field, tuple):
            field_name, target_name = field
        else:
            field_name = field
            target_name = field

        if not hasattr(target, target_name):
            continue

        value = getattr(args, field_name)
        if value is None:
            continue
        if isinstance(value, bool) and not value:
            continue

        setattr(target, target_name, value)
    return target


def _load_resolved_config(path: Optional[str]) -> dict:
    if not path:
        return {}
    config_path = Path(path).expanduser()
    if not config_path.exists():
        return {}
    with config_path.open("r", encoding="utf-8") as stream:
        resolved = yaml.safe_load(stream) or {}
    if not isinstance(resolved, dict):
        raise ValueError(f"resolved_config must contain a mapping: {config_path}")
    return resolved


def _apply_resolved_config(config: TrainConfig, resolved_config: dict) -> None:
    """Apply nested resolved YAML payload that cannot be represented as CLI flags."""
    if not resolved_config:
        return

    unity = resolved_config.get("unity") or {}
    if not isinstance(unity, dict):
        raise ValueError("resolved_config.unity must be a mapping")
    if "environment_parameters" in unity:
        config.environment_parameters = normalize_environment_parameters(
            unity.get("environment_parameters"),
            context="resolved_config.unity.environment_parameters",
        )
    if "parallel_mode" in unity:
        config.parallel_mode = str(unity["parallel_mode"])
    if "use_editor" in unity:
        config.use_editor = bool(unity["use_editor"])

    if "curriculum" in resolved_config:
        curriculum = resolved_config.get("curriculum") or {}
        if not isinstance(curriculum, dict):
            raise ValueError("resolved_config.curriculum must be a mapping")
        config.curriculum = dict(curriculum)


def apply_position_controller_cli_overrides(mappo_config, args: TrainArgs):
    """Apply CLI-only tuning overrides without changing physical wrench limits."""
    body_pid = getattr(mappo_config, "body_pid_wrench_controller", None)
    if body_pid is None:
        return mappo_config

    if args.body_pid_control_rate_hz is not None:
        body_pid.dt = 1.0 / max(float(args.body_pid_control_rate_hz), 1e-6)
    if args.body_pid_params is not None:
        body_pid.pid_params = tuple(float(value) for value in args.body_pid_params)
    if args.body_pid_yaw_pid_params is not None:
        body_pid.yaw_pid_params = tuple(float(value) for value in args.body_pid_yaw_pid_params)
    if args.body_pid_integral_decay is not None:
        body_pid.integral_decay = float(args.body_pid_integral_decay)
    if args.body_pid_yaw_deadband_deg is not None:
        body_pid.yaw_deadband_deg = float(args.body_pid_yaw_deadband_deg)

    return mappo_config


def resolve_output_paths(config: TrainConfig, run_name: str) -> Path:
    """Resolve standard FinsSim artifact paths for MARL runs."""
    return resolve_training_output_paths(config, run_name, backend="marl")


def setup_run(config_name: str, config: TrainConfig):
    """Setup run directory. Uses exp_name as the stable directory name.

    Checks if run already exists and handles overwrite logic. If --resume is set,
    allows existing directories (checkpoint will be loaded separately).

    Args:
        config_name: Name of the config (e.g., "chasing_3_chase_1")
        config: TrainConfig instance (contains exp_name, overwrite, auto_resume, resume_from, etc.)

    Returns:
        run_name: Name of the run
    """
    if config.exp_name:
        run_name = f"{config_name}__{config.exp_name}"
    elif getattr(config, "test_mode", False):
        run_name = f"{config_name}_test"
    else:
        run_name = config_name

    resolve_output_paths(config, run_name)

    run_dir = config.log_dir
    ckpt_dir = config.checkpoint_dir

    is_resuming = config.auto_resume or config.resume_from is not None

    if os.path.exists(run_dir) or os.path.exists(ckpt_dir):
        if is_resuming:
            # Resume is allowed to use existing directories
            pass
        elif not config.overwrite:
            print(f"[ERROR] Run '{run_name}' already exists at {run_dir} or {ckpt_dir}.")
            print(f"        Use --overwrite to overwrite, or --resume to continue training.")
            sys.exit(1)
        else:
            print(f"[WARN] Overwriting existing run '{run_name}' (--overwrite specified).")

    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(ckpt_dir, exist_ok=True)
    if config.tensorboard_dir:
        os.makedirs(config.tensorboard_dir, exist_ok=True)

    config.run_name = run_name
    return run_name


def _active_slots(slots: Sequence[ManagedEnvSlot]) -> list[ManagedEnvSlot]:
    return [slot for slot in slots if not slot.disabled]


def _start_env_slot(slot: ManagedEnvSlot):
    parent_conn, child_conn = Pipe()
    process = Process(target=env_worker, args=(child_conn, slot.env_config, slot.seed))
    process.daemon = False
    process.start()
    slot.conn = parent_conn
    slot.process = process


def _close_env_slot(slot: ManagedEnvSlot):
    conn = slot.conn
    process = slot.process

    if conn is not None:
        try:
            conn.send(("close", None))
        except Exception:
            pass
        try:
            conn.close()
        except Exception:
            pass
        slot.conn = None

    if process is not None:
        try:
            process.join(timeout=5.0)
        except Exception:
            pass
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                process.terminate()
            process.join(timeout=2.0)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            process.join(timeout=1.0)
        slot.process = None


def _request_env(slot: ManagedEnvSlot, task: str, content=None):
    if slot.disabled:
        raise RuntimeError(f"{slot.group} env slot {slot.slot_id} is disabled.")
    if slot.conn is None:
        raise RuntimeError(f"{slot.group} env slot {slot.slot_id} has no live connection.")

    try:
        slot.conn.send((task, content))
    except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
        raise RuntimeError(
            f"{slot.group} env slot {slot.slot_id} failed to send '{task}': {exc}"
        ) from exc

    try:
        response = slot.conn.recv()
    except (BrokenPipeError, ConnectionResetError, EOFError, OSError) as exc:
        raise RuntimeError(
            f"{slot.group} env slot {slot.slot_id} died while waiting for '{task}': {exc}"
        ) from exc

    if isinstance(response, dict) and "error" in response:
        trace_suffix = ""
        if response.get("traceback"):
            trace_suffix = f"\n{response['traceback']}"
        raise RuntimeError(
            f"{slot.group} env slot {slot.slot_id} returned error on '{task}': "
            f"{response['error']}{trace_suffix}"
        )
    return response


def _set_environment_parameters_for_slots(
    slots: Sequence[ManagedEnvSlot],
    parameters: dict[str, float],
    *,
    label: str,
) -> None:
    normalized = normalize_environment_parameters(parameters)
    if _is_training_areas_env(slots):
        slots.set_environment_parameters(normalized)
        if normalized:
            print(f"[INFO] Applied Unity environment parameters to {label}: {len(normalized)} values")
        return
    for slot in _active_slots(slots):
        slot.env_config.environment_parameters = dict(normalized)
        if slot.conn is None:
            continue
        _request_env(slot, "set_environment_parameters", normalized)
    if normalized:
        print(
            f"[INFO] Applied Unity environment parameters to {label}: "
            f"{len(normalized)} values"
        )


def _apply_curriculum_update(
    slots: Sequence[ManagedEnvSlot],
    update: CurriculumUpdate | None,
    *,
    label: str,
) -> None:
    if update is None:
        return
    _set_environment_parameters_for_slots(slots, update.parameters, label=label)
    if update.changed_lesson:
        print(
            "[INFO] Curriculum lesson active: "
            f"index={update.lesson_index}, name={update.lesson_name}, "
            f"start_step={update.start_step}"
        )


def _disable_env_slot(
    slot: ManagedEnvSlot,
    config: TrainConfig,
    count_attr: str,
    reason: str,
):
    if slot.disabled:
        return

    slot.disabled = True
    _close_env_slot(slot)
    current = max(0, getattr(config, count_attr) - 1)
    setattr(config, count_attr, current)
    print(
        f"[WARN] Disabled {slot.group} env slot {slot.slot_id} after repeated failures "
        f"({reason}). {count_attr} is now {current}."
    )


def _recover_env_slot(
    slot: ManagedEnvSlot,
    config: TrainConfig,
    count_attr: str,
    reason: str,
    reset_after_create: bool = False,
):
    max_attempts = max(1, config.env_restart_attempts)
    last_error = None

    for attempt in range(1, max_attempts + 1):
        print(
            f"[WARN] Recovering {slot.group} env slot {slot.slot_id} "
            f"(attempt {attempt}/{max_attempts}) after {reason}."
        )
        _close_env_slot(slot)
        try:
            _start_env_slot(slot)
            _request_env(slot, "create_env")
            if reset_after_create:
                reset_response = _request_env(slot, "reset")
                print(f"[INFO] Restarted {slot.group} env slot {slot.slot_id}.")
                return reset_response
            print(f"[INFO] Restarted {slot.group} env slot {slot.slot_id}.")
            return True
        except Exception as exc:
            last_error = exc
            print(
                f"[WARN] Restart attempt {attempt}/{max_attempts} failed for "
                f"{slot.group} env slot {slot.slot_id}: {exc}"
            )

    _disable_env_slot(slot, config, count_attr, f"{reason}; last_error={last_error}")
    return None


def _spawn_env_slots(
    slots: Sequence[ManagedEnvSlot],
    config: TrainConfig,
    count_attr: str,
    progress_desc: str,
):
    print(f"Creating {progress_desc.lower()}...")
    pbar = tqdm(total=len(slots), desc=progress_desc)
    pending_create: list[ManagedEnvSlot] = []

    for slot in slots:
        try:
            _start_env_slot(slot)
        except Exception as exc:
            recovered = _recover_env_slot(
                slot,
                config,
                count_attr,
                f"initial worker start failure: {exc}",
                reset_after_create=False,
            )
            if recovered is None and count_attr == "num_envs" and config.num_envs <= 0:
                pbar.close()
                raise RuntimeError("No training environments could be created.") from exc
            pbar.update(1)
            continue

        try:
            if slot.conn is None:
                raise RuntimeError("missing connection")
            slot.conn.send(("create_env", None))
            pending_create.append(slot)
        except Exception as exc:
            recovered = _recover_env_slot(
                slot,
                config,
                count_attr,
                f"initial create_env send failure: {exc}",
                reset_after_create=False,
            )
            if recovered is None and count_attr == "num_envs" and config.num_envs <= 0:
                pbar.close()
                raise RuntimeError("No training environments could be created.") from exc
            pbar.update(1)

    for slot in pending_create:
        try:
            if slot.conn is None:
                raise RuntimeError("missing connection")
            response = slot.conn.recv()
            if isinstance(response, dict) and "error" in response:
                trace_suffix = ""
                if response.get("traceback"):
                    trace_suffix = f"\n{response['traceback']}"
                raise RuntimeError(f"{response['error']}{trace_suffix}")
        except Exception as exc:
            recovered = _recover_env_slot(
                slot,
                config,
                count_attr,
                f"initial create_env recv failure: {exc}",
                reset_after_create=False,
            )
            if recovered is None and count_attr == "num_envs" and config.num_envs <= 0:
                pbar.close()
                raise RuntimeError("No training environments could be created.") from exc
        finally:
            pbar.update(1)
    pbar.close()


def _shutdown_env_slots(slots: Sequence[ManagedEnvSlot]):
    if _is_training_areas_env(slots):
        slots.close()
        return
    for slot in slots:
        _close_env_slot(slot)


def _reset_slots(
    slots: Sequence[ManagedEnvSlot],
    config: TrainConfig,
    count_attr: str,
    reason_prefix: str,
) -> dict[int, dict]:
    reset_results: dict[int, dict] = {}
    pending_reset: list[ManagedEnvSlot] = []

    for slot in slots:
        try:
            if slot.conn is None:
                raise RuntimeError("missing connection")
            slot.conn.send(("reset", None))
            pending_reset.append(slot)
        except Exception as exc:
            content = _recover_env_slot(
                slot,
                config,
                count_attr,
                f"{reason_prefix} send failure: {exc}",
                reset_after_create=True,
            )
            if content is not None:
                reset_results[slot.slot_id] = content

    for slot in pending_reset:
        try:
            if slot.conn is None:
                raise RuntimeError("missing connection")
            content = slot.conn.recv()
            if isinstance(content, dict) and "error" in content:
                trace_suffix = ""
                if content.get("traceback"):
                    trace_suffix = f"\n{content['traceback']}"
                raise RuntimeError(f"{content['error']}{trace_suffix}")
            reset_results[slot.slot_id] = content
        except Exception as exc:
            content = _recover_env_slot(
                slot,
                config,
                count_attr,
                f"{reason_prefix} recv failure: {exc}",
                reset_after_create=True,
            )
            if content is not None:
                reset_results[slot.slot_id] = content

    return reset_results


def create_parallel_envs(config: TrainConfig):
    """Create managed training Unity environments."""
    if config.parallel_mode == "multi_area":
        from finssim_marl.envs.unity.worker import get_unity_training_areas_env
        return get_unity_training_areas_env(
            EnvConfig(
                worker_id=0,
                env_base_port=config.env_base_port,
                unity_env_binary_path=config.unity_env_binary_path,
                timeout_wait=config.timeout_wait,
                time_scale=config.training_time_scale,
                env_type=config.env_type,
                parallel_mode="multi_area",
                num_areas=config.num_envs,
                use_editor=False,
                no_graphics=config.no_graphics,
                herder_action_source=config.herder_action_source,
                netter_action_source=config.netter_action_source,
                prey_action_source=config.prey_action_source,
                environment_parameters=dict(config.environment_parameters),
            ),
            config.seed,
        )
    slots = [
        ManagedEnvSlot(
            slot_id=i,
            group="train",
            env_config=EnvConfig(
                worker_id=i,
                env_base_port=config.env_base_port,
                unity_env_binary_path=config.unity_env_binary_path,
                timeout_wait=config.timeout_wait,
                time_scale=config.training_time_scale,
                env_type=config.env_type,
                no_graphics=config.no_graphics,
                herder_action_source=config.herder_action_source,
                netter_action_source=config.netter_action_source,
                prey_action_source=config.prey_action_source,
                environment_parameters=dict(config.environment_parameters),
            ),
            seed=config.seed + i,
        )
        for i in range(config.num_envs)
    ]

    _spawn_env_slots(slots, config, "num_envs", "Creating Environments")
    active_count = len(_active_slots(slots))
    if active_count <= 0:
        raise RuntimeError("No training environments are available after startup.")
    print(f"Created {active_count} training environments")
    return slots


def _select_env_and_buffer_actions(algorithm, obs, role_ids, deterministic: bool = False):
    """Select actions according to the algorithm's explicit action interface."""
    action_interface = getattr(algorithm, "action_interface", "direct_thruster")
    result = algorithm.select_action(obs, role_ids, deterministic=deterministic)

    if action_interface == "direct_thruster":
        if len(result) == 4:
            env_actions, _aux_actions, log_probs, _chosen_heads = result
            return env_actions, env_actions, log_probs
        env_actions, log_probs, _chosen_heads = result
        return env_actions, env_actions, log_probs

    if action_interface == "position_controller":
        if not (len(result) == 4 and isinstance(result[3], dict) and "buffer_actions" in result[3]):
            raise RuntimeError(
                "position_controller algorithms must return "
                "(env_actions, log_probs, chosen_heads, {'buffer_actions': ...})."
            )
        env_actions, log_probs, _chosen_heads, extra = result
        return env_actions, extra["buffer_actions"], log_probs

    raise ValueError(f"Unknown action_interface: {action_interface}")


def _maybe_reset_controller_state(algorithm, env_mask=None):
    reset_fn = getattr(algorithm, "reset_controller_state", None)
    if callable(reset_fn):
        reset_fn(env_mask)


def _resolve_reward_agent_labels(info: dict) -> tuple[str, ...]:
    labels = info.get("reward_all_agents_labels") or info.get("agent_names")
    if labels is None:
        return tuple(f"agent_{i}" for i in range(info["n_agents"]))
    return tuple(str(label) for label in labels)


def _sanitize_metric_label(label: str) -> str:
    sanitized = "".join(ch.lower() if ch.isalnum() else "_" for ch in label).strip("_")
    while "__" in sanitized:
        sanitized = sanitized.replace("__", "_")
    return sanitized or "agent"


def _extract_reward_all_agents(info: Optional[dict], reward_agent_labels: Sequence[str]) -> np.ndarray:
    num_reward_agents = len(reward_agent_labels)
    if info is None or "reward_all_agents_ordered" not in info:
        return np.zeros(num_reward_agents, dtype=np.float32)

    reward_all_agents = np.asarray(info["reward_all_agents_ordered"], dtype=np.float32)
    if reward_all_agents.shape[0] != num_reward_agents:
        raise ValueError(
            "reward_all_agents_ordered length does not match reward_all_agents_labels: "
            f"{reward_all_agents.shape[0]} vs {num_reward_agents}"
        )

    info_labels = info.get("reward_all_agents_labels")
    if info_labels is not None and tuple(info_labels) != tuple(reward_agent_labels):
        raise ValueError(
            "reward_all_agents_labels changed during rollout: "
            f"{tuple(info_labels)} vs {tuple(reward_agent_labels)}"
        )
    return reward_all_agents


def _build_agent_reward_metrics(
    metric_prefix: str,
    reward_all_agents: np.ndarray,
    reward_agent_labels: Sequence[str],
) -> dict[str, float]:
    return {
        f"{metric_prefix}/ep_reward_by_agent/{_sanitize_metric_label(label)}": float(
            reward_all_agents[:, idx].mean()
        )
        for idx, label in enumerate(reward_agent_labels)
    }


def _is_training_areas_env(env) -> bool:
    return hasattr(env, "num_areas") and hasattr(env, "get_states") and not isinstance(env, (list, tuple))


def _collect_rollout_training_areas(
    env,
    algorithm,
    config: TrainConfig,
    rb: RolloutBuffer,
    reward_agent_labels: Sequence[str],
):
    """Collect one complete episode from every replicated Unity training area."""
    obs, states = env.reset()
    _maybe_reset_controller_state(algorithm)
    num_areas = env.num_areas
    active = np.ones(num_areas, dtype=bool)
    episodes = [
        {"obs": [], "actions": [], "log_prob": [], "reward": [], "states": [], "done": [],
         "obs_chaser_team": [], "reward_chaser_team": []}
        for _ in range(num_areas)
    ]
    reward_sums = np.zeros(num_areas, dtype=np.float64)
    lengths = np.zeros(num_areas, dtype=np.int32)
    team_reward_sums = np.zeros((num_areas, config.chaser_team_reward_dim), dtype=np.float32)
    all_reward_sums = np.zeros((num_areas, len(reward_agent_labels)), dtype=np.float32)
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []
    completed_team_rewards: list[np.ndarray] = []
    completed_all_rewards: list[np.ndarray] = []
    role_ids = np.tile(algorithm.role_ids, (num_areas, 1))
    initial_distances = nearest_chaser_prey_distance_m(obs, role_ids)
    min_distances = initial_distances.copy()
    final_distances = initial_distances.copy()
    completed_initial_distances: list[float] = []
    completed_min_distances: list[float] = []
    completed_final_distances: list[float] = []

    while np.any(active):
        env_actions, buffer_actions, log_probs = _select_env_and_buffer_actions(
            algorithm, obs, role_ids, deterministic=False
        )
        response = env.step(env_actions)
        next_distances = nearest_chaser_prey_distance_m(response["next_obs"], role_ids)
        min_distances = np.minimum(min_distances, next_distances)
        final_distances = next_distances
        for area in np.flatnonzero(active):
            info = response["infos"][area]
            reward_team = np.asarray(info["reward_chaser_team"], dtype=np.float32)
            reward_all = _extract_reward_all_agents(info, reward_agent_labels)
            episodes[area]["obs"].append(obs[area])
            episodes[area]["actions"].append(buffer_actions[area])
            episodes[area]["log_prob"].append(log_probs[area])
            episodes[area]["reward"].append(float(response["reward"][area]))
            episodes[area]["states"].append(states[area])
            episodes[area]["done"].append(bool(response["done"][area]))
            episodes[area]["obs_chaser_team"].append(info["obs_chaser_team"])
            episodes[area]["reward_chaser_team"].append(reward_team)
            reward_sums[area] += float(response["reward"][area])
            lengths[area] += 1
            team_reward_sums[area] += reward_team
            all_reward_sums[area] += reward_all
            if bool(response["terminal"][area]):
                rb.add(episodes[area])
                completed_rewards.append(float(reward_sums[area]))
                completed_lengths.append(int(lengths[area]))
                completed_team_rewards.append(team_reward_sums[area].copy())
                completed_all_rewards.append(all_reward_sums[area].copy())
                completed_initial_distances.append(float(initial_distances[area]))
                completed_min_distances.append(float(min_distances[area]))
                completed_final_distances.append(float(final_distances[area]))
                active[area] = False
        obs = response["next_obs"]
        states = response["next_state"]

    return {
        "ep_reward_team_sum": completed_rewards,
        "ep_length": completed_lengths,
        "ep_reward_chaser_team": np.stack(completed_team_rewards, axis=0),
        "ep_reward_all_agents": np.stack(completed_all_rewards, axis=0),
        "ep_initial_nearest_chaser_distance_to_prey_m": np.asarray(completed_initial_distances),
        "ep_min_nearest_chaser_distance_to_prey_m": np.asarray(completed_min_distances),
        "ep_final_nearest_chaser_distance_to_prey_m": np.asarray(completed_final_distances),
    }


def collect_rollout(
    env_slots,
    algorithm,
    config: TrainConfig,
    rb: RolloutBuffer,
    reward_agent_labels: Sequence[str],
):
    """Collect rollout data from parallel environments.

    Args:
        env_slots: Managed training env slots
        algorithm: Algorithm instance
        config: Training config
        rb: RolloutBuffer to fill

    Returns:
        ep_rewards: List of episode rewards
        ep_lengths: List of episode lengths
    """
    if _is_training_areas_env(env_slots):
        return _collect_rollout_training_areas(env_slots, algorithm, config, rb, reward_agent_labels)

    active_slots = _active_slots(env_slots)
    if not active_slots:
        raise RuntimeError("No active training environments remain.")

    episodes = {
        slot.slot_id: {
            "obs": [],
            "actions": [],
            "log_prob": [],
            "reward": [],
            "states": [],
            "done": [],
            "obs_chaser_team": [],
            "reward_chaser_team": [],
        }
        for slot in active_slots
    }

    reset_results = _reset_slots(
        active_slots,
        config,
        "num_envs",
        "reset before rollout",
    )
    _maybe_reset_controller_state(algorithm)
    obs_by_slot = {
        slot_id: content["obs"]
        for slot_id, content in reset_results.items()
    }
    state_by_slot = {
        slot_id: content["state"]
        for slot_id, content in reset_results.items()
    }
    alive_slots = [slot for slot in active_slots if slot.slot_id in reset_results]

    if not alive_slots:
        raise RuntimeError("All training environments failed during reset/restart.")

    completed_rewards = []
    completed_lengths = []
    completed_reward_chaser_team = []
    num_reward_agents = len(reward_agent_labels)
    completed_reward_all_agents = []
    role_ids = algorithm.role_ids
    ep_reward = {slot.slot_id: 0.0 for slot in alive_slots}
    ep_length = {slot.slot_id: 0 for slot in alive_slots}
    ep_reward_chaser_team = {
        slot.slot_id: np.zeros(config.chaser_team_reward_dim, dtype=np.float32)
        for slot in alive_slots
    }
    ep_reward_all_agents = {
        slot.slot_id: np.zeros(num_reward_agents, dtype=np.float32)
        for slot in alive_slots
    }
    ep_initial_distance = {
        slot.slot_id: float(nearest_chaser_prey_distance_m(obs_by_slot[slot.slot_id], role_ids)[0])
        for slot in alive_slots
    }
    ep_min_distance = dict(ep_initial_distance)
    ep_final_distance = dict(ep_initial_distance)
    completed_initial_distances = []
    completed_min_distances = []
    completed_final_distances = []

    while alive_slots:
        num_envs = len(alive_slots)
        role_ids_current = np.tile(role_ids, (num_envs, 1))
        obs = np.stack([obs_by_slot[slot.slot_id] for slot in alive_slots], axis=0)
        state = np.stack([state_by_slot[slot.slot_id] for slot in alive_slots], axis=0)
        actions_np, buffer_actions_np, log_probs_np = _select_env_and_buffer_actions(
            algorithm, obs, role_ids_current, deterministic=False
        )
        current_distances = nearest_chaser_prey_distance_m(obs, role_ids_current)

        pending = []
        next_alive_slots = []

        for idx, slot in enumerate(alive_slots):
            try:
                if slot.conn is None:
                    raise RuntimeError("missing connection")
                slot.conn.send(("step", actions_np[idx]))
                pending.append((slot, idx))
            except Exception as exc:
                reset_content = _recover_env_slot(
                    slot,
                    config,
                    "num_envs",
                    f"step send failure: {exc}",
                    reset_after_create=True,
                )
                episodes[slot.slot_id] = {
                    "obs": [],
                    "actions": [],
                    "log_prob": [],
                    "reward": [],
                    "states": [],
                    "done": [],
                    "obs_chaser_team": [],
                    "reward_chaser_team": [],
                }
                ep_reward[slot.slot_id] = 0.0
                ep_length[slot.slot_id] = 0
                ep_reward_chaser_team[slot.slot_id] = np.zeros(
                    config.chaser_team_reward_dim,
                    dtype=np.float32,
                )
                ep_reward_all_agents[slot.slot_id] = np.zeros(
                    num_reward_agents,
                    dtype=np.float32,
                )
                if reset_content is not None:
                    env_mask = np.zeros((num_envs,), dtype=bool)
                    env_mask[idx] = True
                    _maybe_reset_controller_state(algorithm, env_mask)
                    obs_by_slot[slot.slot_id] = reset_content["obs"]
                    state_by_slot[slot.slot_id] = reset_content["state"]
                    reset_distance = float(nearest_chaser_prey_distance_m(reset_content["obs"], role_ids)[0])
                    ep_initial_distance[slot.slot_id] = reset_distance
                    ep_min_distance[slot.slot_id] = reset_distance
                    ep_final_distance[slot.slot_id] = reset_distance
                    next_alive_slots.append(slot)

        for slot, idx in pending:
            try:
                if slot.conn is None:
                    raise RuntimeError("missing connection")
                content = slot.conn.recv()
                if isinstance(content, dict) and "error" in content:
                    raise RuntimeError(content["error"])
            except Exception as exc:
                reset_content = _recover_env_slot(
                    slot,
                    config,
                    "num_envs",
                    f"step recv failure: {exc}",
                    reset_after_create=True,
                )
                episodes[slot.slot_id] = {
                    "obs": [],
                    "actions": [],
                    "log_prob": [],
                    "reward": [],
                    "states": [],
                    "done": [],
                    "obs_chaser_team": [],
                    "reward_chaser_team": [],
                }
                ep_reward[slot.slot_id] = 0.0
                ep_length[slot.slot_id] = 0
                ep_reward_chaser_team[slot.slot_id] = np.zeros(
                    config.chaser_team_reward_dim,
                    dtype=np.float32,
                )
                ep_reward_all_agents[slot.slot_id] = np.zeros(
                    num_reward_agents,
                    dtype=np.float32,
                )
                if reset_content is not None:
                    env_mask = np.zeros((num_envs,), dtype=bool)
                    env_mask[idx] = True
                    _maybe_reset_controller_state(algorithm, env_mask)
                    obs_by_slot[slot.slot_id] = reset_content["obs"]
                    state_by_slot[slot.slot_id] = reset_content["state"]
                    reset_distance = float(nearest_chaser_prey_distance_m(reset_content["obs"], role_ids)[0])
                    ep_initial_distance[slot.slot_id] = reset_distance
                    ep_min_distance[slot.slot_id] = reset_distance
                    ep_final_distance[slot.slot_id] = reset_distance
                    next_alive_slots.append(slot)
                continue

            info = content.get("infos")
            if info is not None and "obs_chaser_team" in info:
                obs_chaser_team = info["obs_chaser_team"]
            else:
                obs_chaser_team = np.zeros(config.chaser_team_obs_dim * 3)

            if info is not None and "reward_chaser_team" in info:
                reward_chaser_team = info["reward_chaser_team"]
            else:
                reward_chaser_team = np.zeros(config.chaser_team_reward_dim)

            reward_all_agents = _extract_reward_all_agents(info, reward_agent_labels)
            next_distance = float(nearest_chaser_prey_distance_m(content["next_obs"], role_ids)[0])
            ep_min_distance[slot.slot_id] = min(
                ep_min_distance[slot.slot_id], float(current_distances[idx]), next_distance
            )
            ep_final_distance[slot.slot_id] = next_distance

            episodes[slot.slot_id]["obs"].append(obs[idx])
            episodes[slot.slot_id]["actions"].append(buffer_actions_np[idx])
            episodes[slot.slot_id]["log_prob"].append(log_probs_np[idx])
            episodes[slot.slot_id]["reward"].append(content["reward"])
            episodes[slot.slot_id]["states"].append(state[idx])
            episodes[slot.slot_id]["done"].append(content["done"])
            episodes[slot.slot_id]["obs_chaser_team"].append(obs_chaser_team)
            episodes[slot.slot_id]["reward_chaser_team"].append(reward_chaser_team)

            ep_reward[slot.slot_id] += content["reward"]
            ep_length[slot.slot_id] += 1
            ep_reward_chaser_team[slot.slot_id] += reward_chaser_team
            ep_reward_all_agents[slot.slot_id] += reward_all_agents

            if content["done"] or content["truncated"]:
                rb.add(episodes[slot.slot_id])
                completed_rewards.append(ep_reward[slot.slot_id])
                completed_lengths.append(ep_length[slot.slot_id])
                completed_reward_chaser_team.append(ep_reward_chaser_team[slot.slot_id].copy())
                completed_reward_all_agents.append(ep_reward_all_agents[slot.slot_id].copy())
                completed_initial_distances.append(ep_initial_distance[slot.slot_id])
                completed_min_distances.append(ep_min_distance[slot.slot_id])
                completed_final_distances.append(ep_final_distance[slot.slot_id])
                del obs_by_slot[slot.slot_id]
                del state_by_slot[slot.slot_id]
            else:
                obs_by_slot[slot.slot_id] = content["next_obs"]
                state_by_slot[slot.slot_id] = content["next_state"]
                next_alive_slots.append(slot)

        alive_slots = [slot for slot in next_alive_slots if not slot.disabled]

    if not completed_rewards:
        raise RuntimeError("Rollout ended without any completed episodes.")

    return {
        "ep_reward_team_sum": completed_rewards,
        "ep_length": completed_lengths,
        "ep_reward_chaser_team": np.stack(completed_reward_chaser_team, axis=0),
        "ep_reward_all_agents": np.stack(completed_reward_all_agents, axis=0),
        "ep_initial_nearest_chaser_distance_to_prey_m": np.asarray(completed_initial_distances),
        "ep_min_nearest_chaser_distance_to_prey_m": np.asarray(completed_min_distances),
        "ep_final_nearest_chaser_distance_to_prey_m": np.asarray(completed_final_distances),
    }


def create_eval_envs(config: TrainConfig):
    """Create parallel evaluation Unity environments.

    Args:
        config: TrainConfig instance

    Returns:
        List of managed eval env slots
    """
    if config.eval_mode not in {"asynchronous", "serial"}:
        raise ValueError(f"eval_mode must be 'asynchronous' or 'serial', got {config.eval_mode!r}")
    eval_area_count = config.num_eval_envs if config.eval_mode == "asynchronous" else 1
    if config.parallel_mode == "multi_area":
        from finssim_marl.envs.unity.worker import get_unity_training_areas_env
        return get_unity_training_areas_env(
            EnvConfig(
                worker_id=100,
                env_base_port=config.env_base_port + 100,
                unity_env_binary_path=config.unity_env_binary_path,
                timeout_wait=config.timeout_wait,
                time_scale=config.eval_time_scale,
                env_type=config.env_type,
                parallel_mode="multi_area",
                num_areas=eval_area_count,
                no_graphics=config.no_graphics,
                herder_action_source=config.herder_action_source,
                netter_action_source=config.netter_action_source,
                prey_action_source=config.prey_action_source,
                environment_parameters=dict(config.environment_parameters),
            ),
            config.seed + 1000,
        )

    eval_base_port = config.env_base_port + config.num_envs + 100
    num_eval_envs = eval_area_count

    slots = [
        ManagedEnvSlot(
            slot_id=i,
            group="eval",
            env_config=EnvConfig(
                worker_id=100 + i,
                env_base_port=eval_base_port + i,
                unity_env_binary_path=config.unity_env_binary_path,
                timeout_wait=config.timeout_wait,
                time_scale=config.eval_time_scale,
                env_type=config.env_type,
                no_graphics=config.no_graphics,
                herder_action_source=config.herder_action_source,
                netter_action_source=config.netter_action_source,
                prey_action_source=config.prey_action_source,
                environment_parameters=dict(config.environment_parameters),
            ),
            seed=config.seed + 1000 + i,
        )
        for i in range(num_eval_envs)
    ]

    if not slots:
        return []

    _spawn_env_slots(slots, config, "num_eval_envs", "Creating Eval Environments")
    active_count = len(_active_slots(slots))
    if active_count <= 0:
        print("[WARN] No evaluation environments are available after startup. Evaluation will be disabled.")
        return slots
    print(f"Created {active_count} evaluation environments")
    return slots


def _evaluate_training_areas(env, algorithm, config: TrainConfig, reward_agent_labels: Sequence[str]):
    """Evaluate replicated areas in synchronized cohorts without launching binaries per area."""
    target_total_eps = config.num_eval_ep
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []
    completed_all_rewards: list[np.ndarray] = []
    completed_initial_distances: list[float] = []
    completed_min_distances: list[float] = []
    completed_final_distances: list[float] = []
    pbar = tqdm(total=target_total_eps, desc="Evaluating")
    try:
        while len(completed_rewards) < target_total_eps:
            obs, _states = env.reset()
            _maybe_reset_controller_state(algorithm)
            active = np.ones(env.num_areas, dtype=bool)
            reward_sums = np.zeros(env.num_areas, dtype=np.float64)
            lengths = np.zeros(env.num_areas, dtype=np.int32)
            all_reward_sums = np.zeros((env.num_areas, len(reward_agent_labels)), dtype=np.float32)
            role_ids = np.tile(algorithm.role_ids, (env.num_areas, 1))
            initial_distances = nearest_chaser_prey_distance_m(obs, role_ids)
            min_distances = initial_distances.copy()
            final_distances = initial_distances.copy()
            while np.any(active) and len(completed_rewards) < target_total_eps:
                actions, _buffer_actions, _log_probs = _select_env_and_buffer_actions(
                    algorithm, obs, role_ids, deterministic=True
                )
                response = env.step(actions)
                next_distances = nearest_chaser_prey_distance_m(response["next_obs"], role_ids)
                min_distances = np.minimum(min_distances, next_distances)
                final_distances = next_distances
                for area in np.flatnonzero(active):
                    info = response["infos"][area]
                    reward_sums[area] += float(response["reward"][area])
                    lengths[area] += 1
                    all_reward_sums[area] += _extract_reward_all_agents(info, reward_agent_labels)
                    if bool(response["terminal"][area]):
                        completed_rewards.append(float(reward_sums[area]))
                        completed_lengths.append(int(lengths[area]))
                        completed_all_rewards.append(all_reward_sums[area].copy())
                        completed_initial_distances.append(float(initial_distances[area]))
                        completed_min_distances.append(float(min_distances[area]))
                        completed_final_distances.append(float(final_distances[area]))
                        active[area] = False
                        pbar.update(1)
                        if len(completed_rewards) >= target_total_eps:
                            break
                obs = response["next_obs"]
    finally:
        pbar.close()

    all_rewards = np.stack(completed_all_rewards, axis=0)
    metrics = {
        "ep_reward": float(np.mean(completed_rewards)),
        "std_ep_reward": float(np.std(completed_rewards)),
        "ep_length": float(np.mean(completed_lengths)),
    }
    metrics.update(_build_agent_reward_metrics("ep_reward", all_rewards, reward_agent_labels))
    metrics.update(
        _summarize_task_distances(
            config,
            completed_initial_distances,
            completed_min_distances,
            completed_final_distances,
        )
    )
    return metrics


def evaluate(eval_slots, algorithm, config: TrainConfig, reward_agent_labels: Sequence[str]):
    """Evaluate the current policy.

    Runs config.num_eval_ep complete episodes (until done/truncated or max_steps),
    using num_eval_envs parallel environments.
    """
    if _is_training_areas_env(eval_slots):
        return _evaluate_training_areas(eval_slots, algorithm, config, reward_agent_labels)

    active_slots = _active_slots(eval_slots)
    num_eval_envs = len(active_slots)
    if num_eval_envs <= 0:
        raise RuntimeError("No active evaluation environments remain.")
    target_total_eps = config.num_eval_ep

    reset_results = _reset_slots(
        active_slots,
        config,
        "num_eval_envs",
        "eval reset",
    )
    _maybe_reset_controller_state(algorithm)
    obs_by_slot = {
        slot_id: content["obs"]
        for slot_id, content in reset_results.items()
    }

    active_slots = [slot for slot in active_slots if slot.slot_id in obs_by_slot]
    if not active_slots:
        raise RuntimeError("All evaluation environments failed during reset/restart.")

    # Tracking
    ep_reward = {slot.slot_id: 0.0 for slot in active_slots}
    ep_length = {slot.slot_id: 0 for slot in active_slots}
    completed_ep_rewards = []
    completed_ep_lengths = []
    num_reward_agents = len(reward_agent_labels)
    ep_reward_all_agents = {
        slot.slot_id: np.zeros(num_reward_agents, dtype=np.float32)
        for slot in active_slots
    }
    completed_ep_reward_all_agents = []
    ep_initial_distance = {
        slot.slot_id: float(nearest_chaser_prey_distance_m(obs_by_slot[slot.slot_id], algorithm.role_ids)[0])
        for slot in active_slots
    }
    ep_min_distance = dict(ep_initial_distance)
    ep_final_distance = dict(ep_initial_distance)
    completed_initial_distances = []
    completed_min_distances = []
    completed_final_distances = []

    pbar = tqdm(total=target_total_eps, desc="Evaluating")

    role_ids = algorithm.role_ids

    while len(completed_ep_rewards) < target_total_eps:
        active_slots = [slot for slot in _active_slots(eval_slots) if slot.slot_id in obs_by_slot]
        if not active_slots:
            pbar.close()
            raise RuntimeError("No evaluation environments remain while evaluation is still in progress.")

        num_active = len(active_slots)
        active_role_ids = np.tile(role_ids, (num_active, 1))
        actions_np, _, _ = _select_env_and_buffer_actions(
            algorithm,
            np.stack([obs_by_slot[slot.slot_id] for slot in active_slots], axis=0),
            active_role_ids,
            deterministic=True,
        )
        current_distances = nearest_chaser_prey_distance_m(
            np.stack([obs_by_slot[slot.slot_id] for slot in active_slots], axis=0),
            active_role_ids,
        )

        pending = []
        for idx, slot in enumerate(active_slots):
            try:
                if slot.conn is None:
                    raise RuntimeError("missing connection")
                slot.conn.send(("step", actions_np[idx]))
                pending.append((slot, idx))
            except Exception as exc:
                reset_content = _recover_env_slot(
                    slot,
                    config,
                    "num_eval_envs",
                    f"eval step send failure: {exc}",
                    reset_after_create=True,
                )
                if reset_content is None:
                    obs_by_slot.pop(slot.slot_id, None)
                    continue
                env_mask = np.zeros((num_active,), dtype=bool)
                env_mask[idx] = True
                _maybe_reset_controller_state(algorithm, env_mask)
                obs_by_slot[slot.slot_id] = reset_content["obs"]
                ep_reward[slot.slot_id] = 0.0
                ep_length[slot.slot_id] = 0
                ep_reward_all_agents[slot.slot_id] = np.zeros(num_reward_agents, dtype=np.float32)
                reset_distance = float(nearest_chaser_prey_distance_m(reset_content["obs"], algorithm.role_ids)[0])
                ep_initial_distance[slot.slot_id] = reset_distance
                ep_min_distance[slot.slot_id] = reset_distance
                ep_final_distance[slot.slot_id] = reset_distance

        for slot, idx in pending:
            try:
                if slot.conn is None:
                    raise RuntimeError("missing connection")
                content = slot.conn.recv()
                if isinstance(content, dict) and "error" in content:
                    raise RuntimeError(content["error"])
            except Exception as exc:
                reset_content = _recover_env_slot(
                    slot,
                    config,
                    "num_eval_envs",
                    f"eval step recv failure: {exc}",
                    reset_after_create=True,
                )
                if reset_content is None:
                    obs_by_slot.pop(slot.slot_id, None)
                    continue
                env_mask = np.zeros((num_active,), dtype=bool)
                env_mask[idx] = True
                _maybe_reset_controller_state(algorithm, env_mask)
                obs_by_slot[slot.slot_id] = reset_content["obs"]
                ep_reward[slot.slot_id] = 0.0
                ep_length[slot.slot_id] = 0
                ep_reward_all_agents[slot.slot_id] = np.zeros(num_reward_agents, dtype=np.float32)
                reset_distance = float(nearest_chaser_prey_distance_m(reset_content["obs"], algorithm.role_ids)[0])
                ep_initial_distance[slot.slot_id] = reset_distance
                ep_min_distance[slot.slot_id] = reset_distance
                ep_final_distance[slot.slot_id] = reset_distance
                continue

            info = content.get("infos")
            ep_reward[slot.slot_id] += content["reward"]
            ep_length[slot.slot_id] += 1
            ep_reward_all_agents[slot.slot_id] += _extract_reward_all_agents(info, reward_agent_labels)
            next_distance = float(nearest_chaser_prey_distance_m(content["next_obs"], algorithm.role_ids)[0])
            ep_min_distance[slot.slot_id] = min(
                ep_min_distance[slot.slot_id], float(current_distances[idx]), next_distance
            )
            ep_final_distance[slot.slot_id] = next_distance

            if content["done"] or content["truncated"]:
                completed_ep_rewards.append(ep_reward[slot.slot_id])
                completed_ep_lengths.append(ep_length[slot.slot_id])
                completed_ep_reward_all_agents.append(ep_reward_all_agents[slot.slot_id].copy())
                completed_initial_distances.append(ep_initial_distance[slot.slot_id])
                completed_min_distances.append(ep_min_distance[slot.slot_id])
                completed_final_distances.append(ep_final_distance[slot.slot_id])
                pbar.update(1)

                if len(completed_ep_rewards) < target_total_eps:
                    try:
                        reset_content = _request_env(slot, "reset")
                    except Exception as exc:
                        reset_content = _recover_env_slot(
                            slot,
                            config,
                            "num_eval_envs",
                            f"eval reset after completion failure: {exc}",
                            reset_after_create=True,
                        )
                    if reset_content is None:
                        obs_by_slot.pop(slot.slot_id, None)
                        continue
                    env_mask = np.zeros((num_active,), dtype=bool)
                    env_mask[idx] = True
                    _maybe_reset_controller_state(algorithm, env_mask)
                    obs_by_slot[slot.slot_id] = reset_content["obs"]
                    ep_reward[slot.slot_id] = 0.0
                    ep_length[slot.slot_id] = 0
                    ep_reward_all_agents[slot.slot_id] = np.zeros(num_reward_agents, dtype=np.float32)
                    reset_distance = float(nearest_chaser_prey_distance_m(reset_content["obs"], algorithm.role_ids)[0])
                    ep_initial_distance[slot.slot_id] = reset_distance
                    ep_min_distance[slot.slot_id] = reset_distance
                    ep_final_distance[slot.slot_id] = reset_distance
                else:
                    obs_by_slot.pop(slot.slot_id, None)
            else:
                obs_by_slot[slot.slot_id] = content["next_obs"]

    pbar.close()

    completed_ep_reward_all_agents_np = np.stack(completed_ep_reward_all_agents, axis=0)

    metrics = {
        "ep_reward": np.mean(completed_ep_rewards),
        "std_ep_reward": np.std(completed_ep_rewards),
        "ep_length": np.mean(completed_ep_lengths),
    }
    metrics.update(
        _build_agent_reward_metrics(
            "ep_reward",
            completed_ep_reward_all_agents_np,
            reward_agent_labels,
        )
    )
    metrics.update(
        _summarize_task_distances(
            config,
            completed_initial_distances,
            completed_min_distances,
            completed_final_distances,
        )
    )
    return metrics


def main(config_name: str, base_config: BaseConfig, config, args: TrainArgs):
    """Main training loop.

    Args:
        config_name: Name of the config (for run naming)
        base_config: BaseConfig with factory functions
        config: TrainConfig instance from base_config.create_train_config()
    """
    # Set random seeds
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    def _interrupt_to_keyboard_interrupt(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _interrupt_to_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _interrupt_to_keyboard_interrupt)

    # Setup run
    run_name = setup_run(config_name, config)

    curriculum_scheduler = CurriculumScheduler.from_mapping(
        config.curriculum,
        base_parameters=config.environment_parameters,
    )
    _initial_lesson_index, _initial_lesson_name, _initial_lesson_start, initial_parameters = (
        curriculum_scheduler.parameters_for_step(0)
    )
    if initial_parameters:
        config.environment_parameters = dict(initial_parameters)
        if config.environment_parameters:
            print(
                "[INFO] Initial Unity environment parameters prepared: "
                f"{len(config.environment_parameters)} values"
            )

    # Create parallel environments
    train_env_slots = create_parallel_envs(config)

    # Create evaluation environments
    eval_env_slots = create_eval_envs(config)

    # Get environment info from an active env
    if _is_training_areas_env(eval_env_slots):
        info = eval_env_slots.get_info()
    elif _is_training_areas_env(train_env_slots):
        info = train_env_slots.get_info()
    else:
        info_slot = _active_slots(eval_env_slots)
        if info_slot:
            info_slot = info_slot[0]
        else:
            info_slot = _active_slots(train_env_slots)[0]
        info = _request_env(info_slot, "get_info")
    obs_size = info["obs_size"]
    action_size = info["action_size"]
    n_agents = info["n_agents"]
    state_size = info["state_size"]
    max_steps = info["max_steps"]
    role_ids = info["role_ids"]
    reward_agent_labels = _resolve_reward_agent_labels(info)

    print(f"obs_size: {obs_size}, action_size: {action_size}, n_agents: {n_agents}, state_size: {state_size}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Role IDs: {role_ids}")
    print(f"Reward agent labels: {reward_agent_labels}")

    # Create MAPPO config and algorithm using config factory functions
    mappo_config = base_config.create_mappo_config(config)
    mappo_config = apply_position_controller_cli_overrides(mappo_config, args)
    resolved_snapshot = write_resolved_run_config(
        config.output_dir,
        config_name=config_name,
        run_name=run_name,
        base_config=base_config,
        final_config=config,
        algorithm_config=mappo_config,
        cli_args=args,
        input_resolved_config=_load_resolved_config(getattr(config, "resolved_config", None)),
    )
    print(f"Resolved Run Config: {resolved_snapshot}")
    algorithm = base_config.create_algorithm(
        obs_dim=obs_size,
        action_dim=action_size,
        state_dim=state_size,
        n_agents=n_agents,
        role_ids=role_ids,
        mappo_config=mappo_config,
    )

    # Resume from checkpoint if specified
    start_step = 0
    start_training_step = 0
    best_eval_reward = -math.inf
    best_eval_step = 0
    if config.auto_resume or config.resume_from is not None:
        checkpoint_manager = CheckpointManager(CheckpointConfig(checkpoint_dir=config.checkpoint_dir))
        if config.auto_resume:
            # Auto-find latest checkpoint in the run's directory
            # todo: 没有区分training_state.py和普通的.pt，导致载入失败！
            if not hasattr(config, 'run_name') or not config.run_name:
                print("[ERROR] Cannot auto-resume without a run name. Use --exp-name.")
                sys.exit(1)
            resume_path = checkpoint_manager.get_latest(config.run_name)
            if resume_path is None:
                print(f"[ERROR] No checkpoint found for run '{config.run_name}'. Cannot resume.")
                sys.exit(1)
            print(f"[INFO] Auto-resume from latest: {resume_path}")
        else:
            # Explicit checkpoint path
            resume_path = config.resume_from
        training_state = checkpoint_manager.load(resume_path, algorithm)
        start_step = training_state.get("step", 0)
        start_training_step = int(training_state.get("training_step", 0))
        loaded_best_eval_reward = training_state.get("best_eval_reward")
        if loaded_best_eval_reward is not None:
            best_eval_reward = float(loaded_best_eval_reward)
        best_eval_step = int(training_state.get("best_eval_step", 0))
        print(f"[INFO] Resumed from step {start_step} (training_step={start_training_step})")

    resume_curriculum_update = curriculum_scheduler.maybe_update(start_step, force=True)
    if resume_curriculum_update is not None:
        config.environment_parameters = dict(resume_curriculum_update.parameters)
        if _is_training_areas_env(train_env_slots) and config.eval_mode != "asynchronous":
            _apply_curriculum_update(train_env_slots, resume_curriculum_update, label="training areas")
            _apply_curriculum_update(eval_env_slots, resume_curriculum_update, label="evaluation areas")
        elif _is_training_areas_env(train_env_slots):
            _apply_curriculum_update(train_env_slots, resume_curriculum_update, label="training areas")
        else:
            _apply_curriculum_update(train_env_slots, resume_curriculum_update, label="training environments")
            if config.eval_mode != "asynchronous":
                _apply_curriculum_update(eval_env_slots, resume_curriculum_update, label="evaluation environments")

    # Adjust batch_size to be at least num_envs (one episode per env) and divisible by num_envs
    original_batch_size = config.batch_size
    effective_num_envs = max(1, config.num_envs)
    adjusted_batch_size = max(original_batch_size, effective_num_envs)
    if adjusted_batch_size % effective_num_envs != 0:
        adjusted_batch_size = (adjusted_batch_size // effective_num_envs) * effective_num_envs
    if adjusted_batch_size != original_batch_size:
        print(f"[INFO] batch_size ({original_batch_size}) adjusted to {adjusted_batch_size} "
              f"(min num_envs={effective_num_envs}) to avoid buffer overflow.")
        config.batch_size = adjusted_batch_size

    # Create rollout buffer using config factory function
    # buffer_size = batch_size: 收集够batch_size个episode后才进行训练
    rollout_action_size = getattr(algorithm, "rollout_action_dim", action_size)
    rb = base_config.create_rollout_buffer(
        buffer_size=config.batch_size,
        obs_space=obs_size,
        state_space=state_size,
        action_space=rollout_action_size,
        num_agents=n_agents,
        role_ids=role_ids,
        chaser_team_obs_dim=config.chaser_team_obs_dim,
        chaser_team_reward_dim=config.chaser_team_reward_dim,
        device=device,
    )

    # Initialize logger
    logger_config = LoggerConfig(
        log_dir=config.log_dir,
        use_wandb=config.use_wandb,
        wandb_project=config.wandb_project,
        wandb_entity=config.wandb_entity,
        run_name=run_name,
        tensorboard_dir=config.tensorboard_dir,
    )
    logger = Logger(logger_config)
    experiment_metadata = merge_args_to_dict(config)
    experiment_metadata.update(algorithm.get_experiment_metadata())
    experiment_metadata["reward_all_agents_labels"] = ",".join(reward_agent_labels)
    logger.log_hyperparams(experiment_metadata)
    metadata_scalars = {
        key: float(value)
        for key, value in experiment_metadata.items()
        if isinstance(value, (int, float, bool))
    }
    if metadata_scalars:
        logger.log(metadata_scalars, step=0, prefix="run/")
    if "critic_variant" in experiment_metadata:
        logger.log_text("run/critic_variant", str(experiment_metadata["critic_variant"]), 0)
    if "controller_backend" in experiment_metadata:
        logger.log_text("run/controller_backend", str(experiment_metadata["controller_backend"]), 0)
    logger.log_text("run/reward_all_agents_labels", experiment_metadata["reward_all_agents_labels"], 0)

    # Initialize checkpoint manager
    checkpoint_manager = CheckpointManager(
        CheckpointConfig(
            checkpoint_dir=config.checkpoint_dir,
            save_freq=config.save_freq,
        )
    )

    def _build_training_state() -> dict:
        return {
            "step": step,
            "training_step": training_step,
            "best_eval_reward": None if not math.isfinite(best_eval_reward) else best_eval_reward,
            "best_eval_step": best_eval_step,
        }

    async_evaluator: AsyncEvaluationWorker | None = None
    if config.eval_mode == "asynchronous" and config.num_eval_envs > 0:
        eval_config = copy.deepcopy(config)
        eval_algorithm = None

        def _evaluate_snapshot(snapshot_path: Path, environment_parameters: dict[str, float]) -> dict:
            nonlocal eval_algorithm
            if eval_algorithm is None:
                eval_algorithm = base_config.create_algorithm(
                    obs_dim=obs_size,
                    action_dim=action_size,
                    state_dim=state_size,
                    n_agents=n_agents,
                    role_ids=role_ids,
                    mappo_config=copy.deepcopy(mappo_config),
                )
            eval_config.environment_parameters = dict(environment_parameters)
            _set_environment_parameters_for_slots(
                eval_env_slots,
                eval_config.environment_parameters,
                label="asynchronous evaluation areas",
            )
            eval_algorithm.load(str(snapshot_path))
            eval_algorithm.eval()
            return evaluate(eval_env_slots, eval_algorithm, eval_config, reward_agent_labels)

        async_evaluator = AsyncEvaluationWorker(
            snapshot_dir=Path(config.checkpoint_dir) / run_name / "eval_snapshots",
            evaluator=_evaluate_snapshot,
            keep_snapshots=config.keep_eval_snapshots,
        )
        async_evaluator.start()

    def _process_async_evaluation_results() -> None:
        nonlocal best_eval_reward, best_eval_step
        if async_evaluator is None:
            return
        for result in async_evaluator.drain_results():
            _record_async_evaluation_result(result)

    def _record_async_evaluation_result(result: AsyncEvaluationResult) -> None:
        nonlocal best_eval_reward, best_eval_step
        if result.error is not None or result.metrics is None:
            print(f"[WARN] Async eval @ step {result.job.step} failed: {result.error}")
            if not config.keep_eval_snapshots:
                CheckpointManager.discard_snapshot(result.job.snapshot_path)
            return

        metrics = result.metrics
        logger.log(metrics, result.job.step, prefix="eval/")
        logger.log(
            {
                "queue_wait_seconds": result.started_at - result.job.submitted_at,
                "duration_seconds": result.completed_at - result.started_at,
            },
            result.job.step,
            prefix="eval/",
        )
        print(
            f"[Async eval @ step {result.job.step}] Done - ep_reward: "
            f"{metrics['ep_reward']:.3f} ± {metrics['std_ep_reward']:.3f}"
        )
        current_eval_reward = float(metrics["ep_reward"])
        if current_eval_reward > best_eval_reward:
            best_eval_reward = current_eval_reward
            best_eval_step = result.job.step
            best_state = dict(result.job.training_state)
            best_state["best_eval_reward"] = best_eval_reward
            best_state["best_eval_step"] = best_eval_step
            best_checkpoint_path = checkpoint_manager.promote_snapshot(
                run_name,
                result.job.snapshot_path,
                best_state,
                keep_source=config.keep_eval_snapshots,
            )
            print(
                f"[Checkpoint @ step {result.job.step}] New best eval reward {best_eval_reward:.3f}, "
                f"saved to {best_checkpoint_path}"
            )
        if not config.keep_eval_snapshots:
            CheckpointManager.discard_snapshot(result.job.snapshot_path)

    # Training loop
    ep_rewards = []
    ep_lengths = []
    step = start_step
    training_step = start_training_step
    num_episodes = 0
    last_checkpoint = start_step
    rollout_ep_rewards = []  # 用于累积到batch_size后的日志
    rollout_ep_lengths = []
    rollout_ep_all_agent_rewards = []
    rollout_ep_initial_distances = []
    rollout_ep_min_distances = []
    rollout_ep_final_distances = []

    pbar = tqdm(total=config.total_timesteps, initial=start_step, desc="Training Progress")

    try:
        while step < config.total_timesteps:
            _process_async_evaluation_results()
            curriculum_update = curriculum_scheduler.maybe_update(step)
            if curriculum_update is not None:
                config.environment_parameters = dict(curriculum_update.parameters)
                if _is_training_areas_env(train_env_slots) and config.eval_mode != "asynchronous":
                    _apply_curriculum_update(train_env_slots, curriculum_update, label="training areas")
                    _apply_curriculum_update(eval_env_slots, curriculum_update, label="evaluation areas")
                elif _is_training_areas_env(train_env_slots):
                    _apply_curriculum_update(train_env_slots, curriculum_update, label="training areas")
                else:
                    _apply_curriculum_update(train_env_slots, curriculum_update, label="training environments")
                    if config.eval_mode != "asynchronous":
                        _apply_curriculum_update(eval_env_slots, curriculum_update, label="evaluation environments")

            rollout_stats = collect_rollout(train_env_slots, algorithm, config, rb, reward_agent_labels)
            ep_reward = rollout_stats["ep_reward_team_sum"]
            ep_length = rollout_stats["ep_length"]

            rollout_ep_rewards.extend(ep_reward)
            rollout_ep_lengths.extend(ep_length)
            rollout_ep_all_agent_rewards.append(rollout_stats["ep_reward_all_agents"])
            rollout_ep_initial_distances.append(rollout_stats["ep_initial_nearest_chaser_distance_to_prey_m"])
            rollout_ep_min_distances.append(rollout_stats["ep_min_nearest_chaser_distance_to_prey_m"])
            rollout_ep_final_distances.append(rollout_stats["ep_final_nearest_chaser_distance_to_prey_m"])
            num_episodes += len(ep_reward)
            step += int(np.sum(ep_length))

            pbar.update(int(np.sum(ep_length)))
            pbar.set_postfix({
                'reward': f"{np.mean(ep_reward):.2f}",
                'episodes': num_episodes
            })

            if rb.is_full():
                if len(rollout_ep_rewards) > 0:
                    rollout_ep_all_agent_rewards_np = np.concatenate(rollout_ep_all_agent_rewards, axis=0)
                    rollout_metrics = {
                        "rollout/ep_reward": np.mean(rollout_ep_rewards),
                        "rollout/ep_length": np.mean(rollout_ep_lengths),
                        "rollout/num_episodes": num_episodes,
                    }
                    rollout_metrics.update(
                        _build_agent_reward_metrics(
                            "rollout",
                            rollout_ep_all_agent_rewards_np,
                            reward_agent_labels,
                        )
                    )
                    rollout_metrics.update(
                        _summarize_task_distances(
                            config,
                            np.concatenate(rollout_ep_initial_distances, axis=0),
                            np.concatenate(rollout_ep_min_distances, axis=0),
                            np.concatenate(rollout_ep_final_distances, axis=0),
                            prefix="rollout/",
                        )
                    )
                    logger.log(rollout_metrics, step)

                batch = rb.get_batch()
                metrics = algorithm.update(batch)

                logger.log(metrics, step, prefix="train/")
                logger.log({"train/num_updates": training_step}, step)

                training_step += 1

                rollout_ep_rewards = []
                rollout_ep_lengths = []
                rollout_ep_all_agent_rewards = []
                rollout_ep_initial_distances = []
                rollout_ep_min_distances = []
                rollout_ep_final_distances = []

                if training_step > 0 and training_step % config.eval_steps == 0 and config.num_eval_envs > 0:
                    if async_evaluator is not None:
                        snapshot_path = async_evaluator.snapshot_dir / f"snapshot_{step:012d}.pt"
                        checkpoint_manager.save(str(snapshot_path), algorithm, _build_training_state())
                        async_evaluator.submit(
                            step=step,
                            snapshot_path=snapshot_path,
                            training_state=_build_training_state(),
                            environment_parameters=config.environment_parameters,
                        )
                        print(f"[Async eval @ step {step}] Queued snapshot {snapshot_path.name}")
                    else:
                        print(f"\n[Eval @ step {step}] Running evaluation...")
                        algorithm.eval()
                        try:
                            eval_metrics = evaluate(eval_env_slots, algorithm, config, reward_agent_labels)
                        except Exception as exc:
                            print(f"[WARN] Evaluation skipped after env failures: {exc}")
                            eval_metrics = None
                        if eval_metrics is not None:
                            logger.log(eval_metrics, step, prefix="eval/")
                        algorithm.train()
                        if eval_metrics is not None:
                            print(f"[Eval @ step {step}] Done - ep_reward: {eval_metrics['ep_reward']:.3f} ± {eval_metrics['std_ep_reward']:.3f}")
                            current_eval_reward = float(eval_metrics["ep_reward"])
                            if current_eval_reward > best_eval_reward:
                                best_eval_reward = current_eval_reward
                                best_eval_step = step
                                best_checkpoint_path = checkpoint_manager.save_named(
                                    run_name,
                                    "best",
                                    algorithm,
                                    _build_training_state(),
                                )
                                print(
                                    f"[Checkpoint @ step {step}] New best eval reward {best_eval_reward:.3f}, "
                                    f"saved to {best_checkpoint_path}"
                                )

                if step >= last_checkpoint + config.save_freq:
                    checkpoint_path = checkpoint_manager.save_interval(
                        run_name, step, algorithm, _build_training_state()
                    )
                    if checkpoint_path:
                        print(f"[Checkpoint @ step {step}] Saved to {checkpoint_path}")
                    last_checkpoint = step

        if async_evaluator is not None:
            async_evaluator.close()
            _process_async_evaluation_results()
            if not config.keep_eval_snapshots:
                async_evaluator.discard_residual_snapshots()
                try:
                    async_evaluator.snapshot_dir.rmdir()
                except OSError:
                    pass

        final_checkpoint_path = checkpoint_manager.save_named(
            run_name,
            "final",
            algorithm,
            _build_training_state(),
        )
        print(f"[Checkpoint @ step {step}] Saved final checkpoint to {final_checkpoint_path}")
        print("Training complete!")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user! Cleaning up Unity workers...")
    finally:
        pbar.close()
        if async_evaluator is not None:
            async_evaluator.close()
            _process_async_evaluation_results()
            if not config.keep_eval_snapshots:
                async_evaluator.discard_residual_snapshots()
        logger.close()
        _shutdown_env_slots(eval_env_slots)
        _shutdown_env_slots(train_env_slots)


if __name__ == "__main__":
    # Parse CLI args first to get config name
    args = tyro.cli(TrainArgs)

    # Enable debugpy if --debug flag is set
    if args.debug:
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Waiting for debugger attach on port 10092... Please attach your VS Code debugger.")
        debugpy.wait_for_client()

    # Get BaseConfig (includes factory methods)
    base_config = get_config(args.config)
    config = base_config.create_train_config()

    # Override config fields from CLI args
    merge_cli_args(
        config,
        args,
        (
            "exp_name",
            "output_dir",
            "resolved_config",
            "env_type",
            ("env_path", "unity_env_binary_path"),
            "num_envs",
            "num_eval_envs",
            "eval_time_scale",
            "eval_mode",
            "keep_eval_snapshots",
            "batch_size",
            "total_timesteps",
            "device",
            "seed",
            "overwrite",
            "auto_resume",
            "resume_from",
            "env_base_port",
            "timeout_wait",
            ("time_scale", "training_time_scale"),
            "parallel_mode",
            "env_restart_attempts",
        ),
    )
    if args.show_graphics:
        config.no_graphics = False
    if args.resolved_config is not None:
        config.resolved_config = args.resolved_config
        _apply_resolved_config(config, _load_resolved_config(args.resolved_config))
    if args.test:
        config.test_mode = True
        config.exp_name = _with_test_suffix(config.exp_name)
        config.num_envs = 1
        config.num_eval_envs = 1
        config.batch_size = 1
        config.total_timesteps = (
            args.total_timesteps
            if args.total_timesteps is not None
            else min(config.total_timesteps, SMOKE_TEST_TIMESTEPS)
        )
        print(
            "[Test Mode] Overriding num_envs=1, num_eval_envs=1, "
            f"batch_size=1, and total_timesteps={config.total_timesteps}."
        )
    merge_cli_args(
        base_config,
        args,
        (
            "controller_backend",
            "target_body_delta_limits",
            "enable_yaw_control",
            "yaw_error_limit_deg",
            "critic_variant",
            "critic_attention_heads",
        ),
    )
    if "--no-enable-yaw-control" in sys.argv:
        base_config.enable_yaw_control = False

    main(args.config, base_config, config, args)
