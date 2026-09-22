"""
Unified evaluation script for MARL algorithms.

Usage:
    python scripts/eval.py --config chasing_3_chase_1 --exp-name my_experiment

    python scripts/eval.py \
        --config chasing_3_chase_1 \
        --exp-name my_experiment \
        --num-envs 4 \
        --num-episodes 16 \
        --time-scale 1.0 \
        --visualize True
"""

from __future__ import annotations

import os
import random
import signal
import sys
from dataclasses import dataclass
from multiprocessing import Pipe, Process
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np
import torch
import tyro
import yaml
import debugpy
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finssim_marl.training.config import BaseConfig, TrainConfig, get_config
from finssim_marl.utils.chase_metrics import (
    nearest_chaser_prey_distance_m,
    summarize_episode_distances,
)
from finssim_marl.utils.trinet_metrics import summarize_trinet_episode_metrics
from finssim_marl.training.trinet_subgoal_debug_channel import (
    resolve_trinet_subgoal_debug_channel,
)
from finssim_marl.utils.checkpoint import CheckpointConfig, CheckpointManager


def _summarize_task_distances(config, initial, minimum, final, *, prefix=""):
    if config.env_type == "trinet_capture":
        return summarize_trinet_episode_metrics(initial, minimum, final, prefix=prefix)
    return summarize_episode_distances(initial, minimum, final, prefix=prefix)


TRINET_SUCCESS_STAT = "TriNetCapture/success"


def _drain_trinet_terminal_outcomes(env) -> list[bool]:
    """Return Unity-reported success outcomes completed since the last step."""
    stats_channel = getattr(env, "stats_channel", None)
    if stats_channel is None:
        return []
    stats = stats_channel.get_and_reset_stats()
    return [
        float(value) >= 0.5
        for value, _aggregation in stats.get(TRINET_SUCCESS_STAT, ())
    ]


@dataclass
class EvalArgs:
    """CLI arguments for standalone evaluation."""

    config: str
    exp_name: str
    output_dir: Optional[str] = None
    resolved_config: Optional[str] = None
    checkpoint_path: Optional[str] = None
    checkpoint_dir: Optional[str] = None
    env_type: Optional[str] = None
    env_path: Optional[str] = None
    unity_env_binary_path: Optional[str] = None
    env_base_port: Optional[int] = None
    timeout_wait: Optional[int] = None
    num_envs: Optional[int] = None
    num_episodes: Optional[int] = None
    time_scale: float = 1.0
    parallel_mode: Optional[str] = None
    use_editor: bool = False
    visualize: bool = False
    show_graphics: bool = False
    device: Optional[str] = None
    seed: Optional[int] = None
    controller_backend: Optional[str] = None
    target_body_delta_limits: Optional[Tuple[float, float, float]] = None
    yaw_error_limit_deg: Optional[float] = None
    enable_yaw_control: bool = False
    critic_variant: Optional[str] = None
    critic_attention_heads: Optional[int] = None
    deterministic: bool = True
    debug: bool = False  # Pass --debug to enable debugpy for VS Code attach


def merge_cli_args(target, args: EvalArgs, fields: Tuple[str | Tuple[str, str], ...]):
    """Merge CLI overrides into a config-like object.

    Optional arguments only override when they are not None. Boolean flags only
    override when True, because False is also the tyro default for absent flags.
    Fields may be provided as ``("arg_name", "target_name")`` pairs when the
    CLI name differs from the config attribute name.
    """
    for field in fields:
        if isinstance(field, tuple):
            arg_name, target_name = field
        else:
            arg_name = field
            target_name = field

        if not hasattr(target, target_name):
            continue

        value = getattr(args, arg_name)
        if value is None:
            continue
        if isinstance(value, bool) and not value:
            continue

        setattr(target, target_name, value)
    return target


def build_run_name(config_name: str, exp_name: str) -> str:
    return f"{config_name}__{exp_name}" if exp_name else config_name


def create_eval_envs(config: TrainConfig, visualize: bool):
    """Create parallel Unity environments for standalone evaluation."""
    from finssim_marl.envs.unity.worker import EnvConfig, env_worker, get_unity_training_areas_env

    num_eval_envs = config.num_eval_envs
    if num_eval_envs <= 0:
        raise ValueError(f"num_eval_envs must be > 0, got {num_eval_envs}")
    if config.use_editor and num_eval_envs != 1:
        raise ValueError("--use-editor requires exactly one MARL evaluation area")
    if config.parallel_mode == "multi_area" or config.use_editor:
        if config.use_editor:
            print(
                "Editor mode enabled: Please open Unity Editor and press Play to start the evaluation environment.",
                flush=True,
            )
        env = get_unity_training_areas_env(
            EnvConfig(
                worker_id=0,
                env_base_port=config.env_base_port,
                unity_env_binary_path=config.unity_env_binary_path,
                timeout_wait=config.timeout_wait,
                time_scale=config.eval_time_scale,
                env_type=config.env_type,
                parallel_mode="multi_area",
                num_areas=num_eval_envs,
                use_editor=config.use_editor,
                no_graphics=False if config.use_editor else not visualize,
                herder_action_source=config.herder_action_source,
                netter_action_source=config.netter_action_source,
                prey_action_source=config.prey_action_source,
                environment_parameters=dict(config.environment_parameters),
            ),
            config.seed + 1000,
        )
        return env, None

    eval_base_port = config.env_base_port + config.num_envs + 100
    conns = [Pipe() for _ in range(num_eval_envs)]
    eval_conns, env_conns = zip(*conns)

    processes = [
        Process(
            target=env_worker,
            args=(
                env_conns[i],
                EnvConfig(
                    worker_id=100 + i,
                    env_base_port=eval_base_port + i,
                    unity_env_binary_path=config.unity_env_binary_path,
                    time_scale=config.eval_time_scale,
                    env_type=config.env_type,
                    no_graphics=not visualize,
                    herder_action_source=config.herder_action_source,
                    netter_action_source=config.netter_action_source,
                    prey_action_source=config.prey_action_source,
                ),
                config.seed + 1000 + i,
            ),
        )
        for i in range(num_eval_envs)
    ]

    for process in processes:
        process.daemon = False
        process.start()

    print("Creating evaluation environments...")
    for eval_conn in eval_conns:
        eval_conn.send(("create_env", None))

    pbar = tqdm(total=num_eval_envs, desc="Creating Eval Environments")
    for i, eval_conn in enumerate(eval_conns):
        response = eval_conn.recv()
        if "error" in response:
            raise RuntimeError(f"Failed to create eval environment {i}: {response['error']}")
        pbar.update(1)
    pbar.close()

    return eval_conns, processes


def _select_env_actions(algorithm, obs, role_ids, deterministic: bool = False):
    """Select environment actions according to the algorithm action interface."""
    action_interface = getattr(algorithm, "action_interface", "direct_thruster")
    result = algorithm.select_action(obs, role_ids, deterministic=deterministic)

    if action_interface == "direct_thruster":
        if len(result) == 4:
            env_actions, _aux_actions, _log_probs, _chosen_heads = result
            return env_actions
        env_actions, _log_probs, _chosen_heads = result
        return env_actions

    if action_interface == "position_controller":
        if not (len(result) == 4 and isinstance(result[3], dict) and "buffer_actions" in result[3]):
            raise RuntimeError(
                "position_controller algorithms must return "
                "(env_actions, log_probs, chosen_heads, {'buffer_actions': ...})."
            )
        env_actions, _log_probs, _chosen_heads, _extra = result
        return env_actions

    raise ValueError(f"Unknown action_interface: {action_interface}")


def _maybe_reset_controller_state(algorithm, env_mask=None):
    reset_fn = getattr(algorithm, "reset_controller_state", None)
    if callable(reset_fn):
        reset_fn(env_mask)


def _emit_trinet_subgoal_debug(env, algorithm, step_index: int) -> None:
    """Mirror the static baseline's pre-PID subgoals into the Unity editor."""
    channel = resolve_trinet_subgoal_debug_channel(env)
    diagnostics = getattr(algorithm, "last_diagnostics", None)
    if channel is None or not isinstance(diagnostics, dict):
        return

    required = (
        "formation_waypoint_error_m",
        "formation_correction_m",
        "body_position_error_m",
        "yaw_error_deg",
        "phase",
    )
    if any(key not in diagnostics for key in required):
        return

    # Cyan debug target: the full un-clipped formation vertex waypoint. The
    # yellow marker remains the short error passed to the PID after clipping.
    translation = np.asarray(diagnostics["formation_waypoint_error_m"], dtype=np.float32)
    formation = np.asarray(diagnostics["formation_correction_m"], dtype=np.float32)
    body = np.asarray(diagnostics["body_position_error_m"], dtype=np.float32)
    yaw = np.asarray(diagnostics["yaw_error_deg"], dtype=np.float32)
    phase = np.asarray(diagnostics["phase"], dtype=np.int64)
    if (
        translation.ndim != 3
        or translation.shape[1:] != (3, 3)
        or formation.shape != translation.shape
        or body.shape != translation.shape
        or yaw.shape != translation.shape[:2]
        or phase.shape != translation.shape[:2]
    ):
        return

    for area_index in range(translation.shape[0]):
        channel.send_subgoals(
            step_index=step_index,
            area_index=area_index,
            shared_translation=translation[area_index],
            formation_correction=formation[area_index],
            body_subgoal=body[area_index],
            yaw_error_deg=yaw[area_index],
            phase=phase[area_index],
        )


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
            "reward_all_agents_labels changed during evaluation: "
            f"{tuple(info_labels)} vs {tuple(reward_agent_labels)}"
        )
    return reward_all_agents


def _build_agent_reward_metrics(
    reward_all_agents: np.ndarray,
    reward_agent_labels: Sequence[str],
) -> dict[str, float]:
    return {
        f"ep_reward_by_agent/{_sanitize_metric_label(label)}": float(
            reward_all_agents[:, idx].mean()
        )
        for idx, label in enumerate(reward_agent_labels)
    }


def evaluate(
    eval_conns,
    algorithm,
    config: TrainConfig,
    deterministic: bool,
    reward_agent_labels: Sequence[str],
):
    """Run standalone evaluation."""
    if hasattr(eval_conns, "num_areas"):
        return _evaluate_training_areas(
            eval_conns, algorithm, config, deterministic, reward_agent_labels
        )
    num_eval_envs = len(eval_conns)
    target_total_eps = config.num_eval_ep
    if target_total_eps <= 0:
        raise ValueError(f"num_eval_ep must be > 0, got {target_total_eps}")

    for eval_conn in eval_conns:
        eval_conn.send(("reset", None))

    contents = [eval_conn.recv() for eval_conn in eval_conns]
    _maybe_reset_controller_state(algorithm)
    obs = np.stack([content["obs"] for content in contents], axis=0)

    ep_reward = [0.0] * num_eval_envs
    ep_length = [0] * num_eval_envs
    completed_ep_rewards = []
    completed_ep_lengths = []
    num_reward_agents = len(reward_agent_labels)
    ep_reward_all_agents = np.zeros((num_eval_envs, num_reward_agents), dtype=np.float32)
    completed_ep_reward_all_agents = []
    initial_distances = nearest_chaser_prey_distance_m(obs, algorithm.role_ids)
    min_distances = initial_distances.copy()
    final_distances = initial_distances.copy()
    completed_initial_distances = []
    completed_min_distances = []
    completed_final_distances = []
    active_envs = set(range(num_eval_envs))

    pbar = tqdm(total=target_total_eps, desc="Evaluating")

    role_ids = algorithm.role_ids

    while len(completed_ep_rewards) < target_total_eps:
        active_list = list(active_envs)
        num_active = len(active_list)
        current_role_ids = np.tile(role_ids, (num_active, 1))
        actions_np = _select_env_actions(
            algorithm,
            obs[active_list],
            current_role_ids,
            deterministic=deterministic,
        )
        current_distances = nearest_chaser_prey_distance_m(obs[active_list], current_role_ids)

        for idx, env_idx in enumerate(active_list):
            eval_conns[env_idx].send(("step", actions_np[idx]))

        contents = [eval_conns[env_idx].recv() for env_idx in active_list]
        next_obs_list = [content["next_obs"] for content in contents]
        rewards = [content["reward"] for content in contents]
        dones = [content["done"] for content in contents]
        truncateds = [content["truncated"] for content in contents]
        infos = [content.get("infos") for content in contents]

        newly_completed = []
        for idx, env_idx in enumerate(active_list):
            ep_reward[env_idx] += rewards[idx]
            ep_length[env_idx] += 1
            info = infos[idx]
            ep_reward_all_agents[env_idx] += _extract_reward_all_agents(info, reward_agent_labels)
            next_distance = float(nearest_chaser_prey_distance_m(next_obs_list[idx], algorithm.role_ids)[0])
            min_distances[env_idx] = min(
                min_distances[env_idx], float(current_distances[idx]), next_distance
            )
            final_distances[env_idx] = next_distance

            if dones[idx] or truncateds[idx]:
                newly_completed.append(env_idx)
                if len(completed_ep_rewards) < target_total_eps:
                    completed_ep_rewards.append(ep_reward[env_idx])
                    completed_ep_lengths.append(ep_length[env_idx])
                    completed_ep_reward_all_agents.append(ep_reward_all_agents[env_idx].copy())
                    completed_initial_distances.append(float(initial_distances[env_idx]))
                    completed_min_distances.append(float(min_distances[env_idx]))
                    completed_final_distances.append(float(final_distances[env_idx]))
                    pbar.update(1)

        for env_idx in newly_completed:
            active_envs.remove(env_idx)

        for idx, env_idx in enumerate(active_list):
            if env_idx not in newly_completed:
                obs[env_idx] = next_obs_list[idx]

        if len(completed_ep_rewards) < target_total_eps:
            for env_idx in newly_completed:
                eval_conns[env_idx].send(("reset", None))
                content = eval_conns[env_idx].recv()
                env_mask = np.zeros((num_active,), dtype=bool)
                env_mask[active_list.index(env_idx)] = True
                _maybe_reset_controller_state(algorithm, env_mask)
                obs[env_idx] = content["obs"]
                ep_reward[env_idx] = 0.0
                ep_length[env_idx] = 0
                ep_reward_all_agents[env_idx].fill(0.0)
                reset_distance = float(nearest_chaser_prey_distance_m(content["obs"], algorithm.role_ids)[0])
                initial_distances[env_idx] = reset_distance
                min_distances[env_idx] = reset_distance
                final_distances[env_idx] = reset_distance
                active_envs.add(env_idx)

    pbar.close()
    completed_ep_reward_all_agents_np = np.stack(completed_ep_reward_all_agents, axis=0)

    metrics = {
        "ep_reward_mean": float(np.mean(completed_ep_rewards)),
        "ep_reward_std": float(np.std(completed_ep_rewards)),
        "ep_reward_min": float(np.min(completed_ep_rewards)),
        "ep_reward_max": float(np.max(completed_ep_rewards)),
        "ep_length_mean": float(np.mean(completed_ep_lengths)),
        "num_episodes": len(completed_ep_rewards),
    }
    metrics.update(_build_agent_reward_metrics(completed_ep_reward_all_agents_np, reward_agent_labels))
    metrics.update(
        _summarize_task_distances(
            config,
            completed_initial_distances,
            completed_min_distances,
            completed_final_distances,
        )
    )
    return metrics


def _evaluate_training_areas(env, algorithm, config: TrainConfig, deterministic: bool, reward_agent_labels: Sequence[str]):
    target_total_eps = config.num_eval_ep
    completed_rewards: list[float] = []
    completed_lengths: list[int] = []
    completed_all_rewards: list[np.ndarray] = []
    completed_initial_distances: list[float] = []
    completed_min_distances: list[float] = []
    completed_final_distances: list[float] = []
    completed_successes: list[bool] = []
    subgoal_debug_step = 0
    pbar = tqdm(total=target_total_eps, desc="Evaluating")
    try:
        while len(completed_rewards) < target_total_eps:
            obs, _states = env.reset()
            _maybe_reset_controller_state(algorithm)
            active = np.ones(env.num_areas, dtype=bool)
            rewards = np.zeros(env.num_areas, dtype=np.float64)
            lengths = np.zeros(env.num_areas, dtype=np.int32)
            all_rewards = np.zeros((env.num_areas, len(reward_agent_labels)), dtype=np.float32)
            role_ids = np.tile(algorithm.role_ids, (env.num_areas, 1))
            initial_distances = nearest_chaser_prey_distance_m(obs, role_ids)
            min_distances = initial_distances.copy()
            final_distances = initial_distances.copy()
            while np.any(active) and len(completed_rewards) < target_total_eps:
                actions = _select_env_actions(algorithm, obs, role_ids, deterministic=deterministic)
                _emit_trinet_subgoal_debug(env, algorithm, subgoal_debug_step)
                subgoal_debug_step += 1
                response = env.step(actions)
                terminal_outcomes = (
                    _drain_trinet_terminal_outcomes(env)
                    if config.env_type == "trinet_capture"
                    else []
                )
                next_distances = nearest_chaser_prey_distance_m(response["next_obs"], role_ids)
                min_distances = np.minimum(min_distances, next_distances)
                final_distances = next_distances
                for area in np.flatnonzero(active):
                    rewards[area] += float(response["reward"][area])
                    lengths[area] += 1
                    all_rewards[area] += _extract_reward_all_agents(response["infos"][area], reward_agent_labels)
                    if bool(response["terminal"][area]):
                        if config.env_type == "trinet_capture":
                            # A Unity terminal reason (success / explicit
                            # failure) emits one stat on this same step.
                            # Reaching the wrapper step limit emits none and
                            # is labelled TIMEOUT below.
                            has_unity_outcome = bool(terminal_outcomes)
                            success = terminal_outcomes.pop(0) if has_unity_outcome else False
                            completed_successes.append(success)
                            status = "SUCCESS" if success else (
                                "FAIL" if has_unity_outcome else "TIMEOUT"
                            )
                        completed_rewards.append(float(rewards[area]))
                        completed_lengths.append(int(lengths[area]))
                        completed_all_rewards.append(all_rewards[area].copy())
                        completed_initial_distances.append(float(initial_distances[area]))
                        completed_min_distances.append(float(min_distances[area]))
                        completed_final_distances.append(float(final_distances[area]))
                        active[area] = False
                        if config.env_type == "trinet_capture":
                            pbar.set_postfix(
                                status=status,
                                reward=f"{rewards[area]:.3f}",
                                success=f"{sum(completed_successes)}/{len(completed_successes)}",
                            )
                        pbar.update(1)
                        if len(completed_rewards) >= target_total_eps:
                            break
                obs = response["next_obs"]
    finally:
        pbar.close()
    completed = np.stack(completed_all_rewards, axis=0)
    metrics = {
        "ep_reward_mean": float(np.mean(completed_rewards)),
        "ep_reward_std": float(np.std(completed_rewards)),
        "ep_reward_min": float(np.min(completed_rewards)),
        "ep_reward_max": float(np.max(completed_rewards)),
        "ep_length_mean": float(np.mean(completed_lengths)),
        "num_episodes": len(completed_rewards),
    }
    metrics.update(_build_agent_reward_metrics(completed, reward_agent_labels))
    if config.env_type == "trinet_capture":
        metrics.update(
            {
                "successes": int(sum(completed_successes)),
                "failures": int(len(completed_successes) - sum(completed_successes)),
                "success_rate": float(np.mean(completed_successes)) if completed_successes else 0.0,
            }
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


def cleanup_envs(conns, processes):
    """Close evaluation workers."""
    for conn in conns:
        try:
            conn.send(("close", None))
        except (BrokenPipeError, EOFError):
            pass
        except Exception:
            pass
    for process in processes:
        try:
            process.join(timeout=5)
        except Exception:
            pass
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except Exception:
                process.terminate()
            process.join(timeout=2)
        if process.is_alive():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except Exception:
                process.kill()
            process.join(timeout=1)


def resolve_checkpoint_path(
    checkpoint_manager: CheckpointManager,
    run_name: str,
    explicit_checkpoint_path: Optional[str],
) -> str:
    """Resolve the checkpoint path to load."""
    if explicit_checkpoint_path is not None:
        if not os.path.exists(explicit_checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {explicit_checkpoint_path}")
        return explicit_checkpoint_path

    checkpoint_path = checkpoint_manager.get_latest(run_name)
    if checkpoint_path is None:
        raise FileNotFoundError(
            f"No checkpoint found for run '{run_name}' under '{checkpoint_manager.checkpoint_dir}'."
        )
    return str(checkpoint_path)


def main(config_name: str, base_config: BaseConfig, config: TrainConfig, visualize: bool, deterministic: bool):
    """Main evaluation entrypoint."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)

    def _interrupt_to_keyboard_interrupt(_signum, _frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGINT, _interrupt_to_keyboard_interrupt)
    signal.signal(signal.SIGTERM, _interrupt_to_keyboard_interrupt)

    device = torch.device(config.device)
    print(f"Using device: {device}")

    eval_conns = []
    eval_processes = []

    try:
        eval_conns, eval_processes = create_eval_envs(config, visualize=visualize)

        if hasattr(eval_conns, "get_info"):
            info = eval_conns.get_info()
        else:
            eval_conns[0].send(("get_info", None))
            info = eval_conns[0].recv()
        obs_size = info["obs_size"]
        action_size = info["action_size"]
        n_agents = info["n_agents"]
        state_size = info["state_size"]
        role_ids = info["role_ids"]
        reward_agent_labels = _resolve_reward_agent_labels(info)

        print(
            f"obs_size: {obs_size}, action_size: {action_size}, "
            f"n_agents: {n_agents}, state_size: {state_size}"
        )
        print(f"Reward agent labels: {reward_agent_labels}")

        if getattr(base_config, "requires_checkpoint", True):
            mappo_config = base_config.create_mappo_config(config)
            algorithm = base_config.create_algorithm(
                obs_dim=obs_size,
                action_dim=action_size,
                state_dim=state_size,
                n_agents=n_agents,
                role_ids=role_ids,
                mappo_config=mappo_config,
            )

            run_name = build_run_name(config_name, config.exp_name or "")
            checkpoint_manager = CheckpointManager(
                CheckpointConfig(checkpoint_dir=config.checkpoint_dir)
            )
            checkpoint_path = resolve_checkpoint_path(
                checkpoint_manager=checkpoint_manager,
                run_name=run_name,
                explicit_checkpoint_path=config.resume_from,
            )

            training_state = checkpoint_manager.load(checkpoint_path, algorithm)
            algorithm.eval()

            print(f"[INFO] Loaded checkpoint: {checkpoint_path}")
            if training_state:
                print(
                    f"[INFO] Training state: step={training_state.get('step')}, "
                    f"training_step={training_state.get('training_step')}"
                )
        else:
            create_baseline = getattr(base_config, "create_baseline_algorithm", None)
            if not callable(create_baseline):
                raise TypeError(
                    f"No-checkpoint config {base_config.name!r} must define create_baseline_algorithm()."
                )
            algorithm = create_baseline(n_agents=n_agents, role_ids=role_ids, device=config.device)
            algorithm.eval()
            print(f"[INFO] Using static baseline: {base_config.name}")

        metrics = evaluate(
            eval_conns=eval_conns,
            algorithm=algorithm,
            config=config,
            deterministic=deterministic,
            reward_agent_labels=reward_agent_labels,
        )

        print("\nEvaluation complete!")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user! Cleaning up Unity workers...")
    finally:
        if hasattr(eval_conns, "close"):
            eval_conns.close()
        elif eval_conns and eval_processes:
            cleanup_envs(eval_conns, eval_processes)


if __name__ == "__main__":
    args = tyro.cli(EvalArgs)

    if args.debug:
        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Waiting for debugger attach on port 10092... Please attach your VS Code debugger.")
        debugpy.wait_for_client()
    if args.show_graphics:
        args.visualize = True

    base_config = get_config(args.config)
    config = base_config.create_train_config()

    if args.output_dir is not None:
        config.output_dir = args.output_dir
        if args.checkpoint_dir is None:
            config.checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    if args.resolved_config is not None:
        config.resolved_config = args.resolved_config

    merge_cli_args(
        config,
        args,
        (
            "exp_name",
            ("checkpoint_path", "resume_from"),
            "checkpoint_dir",
            "output_dir",
            "resolved_config",
            "env_type",
            ("env_path", "unity_env_binary_path"),
            "unity_env_binary_path",
            "env_base_port",
            "timeout_wait",
            ("num_envs", "num_eval_envs"),
            ("num_episodes", "num_eval_ep"),
            ("time_scale", "eval_time_scale"),
            "parallel_mode",
            "use_editor",
            "device",
            "seed",
        ),
    )
    if args.resolved_config is not None:
        with Path(args.resolved_config).open("r", encoding="utf-8") as stream:
            resolved = yaml.safe_load(stream) or {}
        unity = resolved.get("unity") or {}
        if "parallel_mode" in unity:
            config.parallel_mode = str(unity["parallel_mode"])
        if "use_editor" in unity:
            config.use_editor = bool(unity["use_editor"])
    if args.use_editor:
        config.use_editor = True
        config.num_eval_envs = 1
    merge_cli_args(
        base_config,
        args,
        (
            "controller_backend",
            "target_body_delta_limits",
            "yaw_error_limit_deg",
            "enable_yaw_control",
            "critic_variant",
            "critic_attention_heads",
        ),
    )
    if "--no-enable-yaw-control" in sys.argv:
        base_config.enable_yaw_control = False

    if config.exp_name is None:
        print("[ERROR] --exp-name is required for standalone evaluation.")
        sys.exit(1)

    main(
        args.config,
        base_config,
        config,
        visualize=args.visualize,
        deterministic=args.deterministic,
    )
