"""
统一的评估脚本

使用方式:
    python -m scripts.eval --config=sac_default
    python -m scripts.eval --config=sac_default --exp-name my_sac_test --num-episodes 10
    python -m scripts.eval --config=ppo_chase --checkpoint-path ./checkpoints/best_model.zip
    python -m scripts.eval --config=sac_default --save-video --video-path ./output.mp4
"""

import numpy as np
import signal
import sys
import tyro
import dataclasses
import json
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finssim_rl.training.eval_args import EvalArgs
from finssim_rl.training.config import get_config, list_configs
from finssim_rl.models import make_unity_env, make_unity_env_for_eval, make_unity_training_areas_env
from finssim_rl.training.reward_protocols import (
    compile_reward_config_from_resolved_config,
    load_resolved_config,
    resolved_unity_environment_parameters,
)
from finssim_rl.training.one_chase_one_eval_metrics import (
    OneChaseOneDistanceTracker,
    is_one_chase_one_config,
)


def _merge_unity_additional_args_json(args: EvalArgs) -> EvalArgs:
    if not args.unity_additional_args_json:
        return args
    try:
        parsed = json.loads(args.unity_additional_args_json)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid --unity-additional-args-json: {exc}") from exc
    if not isinstance(parsed, list) or not all(isinstance(item, (str, int, float, bool)) for item in parsed):
        raise SystemExit("--unity-additional-args-json must be a JSON array of scalar values")
    extra_args = [str(item) for item in parsed]
    if args.unity_additional_args:
        extra_args = [*args.unity_additional_args, *extra_args]
    args.unity_additional_args = extra_args
    return args


def _apply_resolved_reward_config(args: EvalArgs, env_config):
    resolved_config = load_resolved_config(args.resolved_config)
    compiled = compile_reward_config_from_resolved_config(resolved_config)
    parameters = dict(getattr(env_config, "environment_parameters", {}) or {})
    if compiled is not None:
        parameters.update(compiled.environment_parameters)
        print(
            "[Reward Config] "
            f"protocol={compiled.protocol} version={compiled.version} "
            f"mode={compiled.mode_name} id={compiled.mode_id} "
            f"parameters={len(compiled.environment_parameters)}"
        )

    unity_parameters = resolved_unity_environment_parameters(resolved_config)
    if unity_parameters:
        parameters.update(unity_parameters)
        print(
            "[Unity Environment Parameters] "
            f"parameters={len(unity_parameters)} keys={', '.join(sorted(unity_parameters))}"
        )

    if parameters == dict(getattr(env_config, "environment_parameters", {}) or {}):
        return env_config
    return dataclasses.replace(env_config, environment_parameters=parameters)


def evaluate_model(args: EvalArgs) -> None:
    """
    评估模型
    
    Args:
        args: 评估参数
    """
    args = _merge_unity_additional_args_json(args)

    # 获取配置
    try:
        config = get_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable configs: {', '.join(list_configs())}")
        return
    
    # 如果没有指定 exp_name，使用 config 的名称
    if args.exp_name == "":
        print(f"No exp_name specified, using config name ({args.config}) as exp_name.")
        args.exp_name = args.config

    use_editor = bool(args.use_editor or getattr(config.env_config, "use_editor", False))

    # 如果在config中指定了unity二进制env_path，则覆盖命令行参数
    if not use_editor and config.env_config.env_path and not args.env_path:
        args.env_path = config.env_config.env_path
    
    if not use_editor and not args.env_path: # 两处都没有指定env_path
        print("Error: No Unity environment path specified!")
        print("Please provide the path using --env-path or specify it in the config.")
        return

    env_config = config.env_config
    env_config = _apply_resolved_reward_config(args, env_config)
    if args.port_offset is not None:
        env_config = dataclasses.replace(env_config, port_offset=int(args.port_offset))
    if args.env_base_port is not None:
        env_config = dataclasses.replace(env_config, env_base_port=int(args.env_base_port))
    if args.num_envs is not None:
        env_config = dataclasses.replace(env_config, num_envs=int(args.num_envs))
    if args.parallel_mode is not None:
        env_config = dataclasses.replace(env_config, parallel_mode=str(args.parallel_mode))
    if args.time_scale is not None:
        env_config = dataclasses.replace(env_config, time_scale=float(args.time_scale))
    if args.timeout_wait is not None:
        env_config = dataclasses.replace(env_config, timeout_wait=int(args.timeout_wait))
    if args.seed is not None:
        env_config = dataclasses.replace(env_config, seed=int(args.seed))
    if args.unity_additional_args is not None:
        env_config = dataclasses.replace(env_config, unity_additional_args=list(args.unity_additional_args))
    if use_editor:
        env_config = dataclasses.replace(
            env_config,
            use_editor=True,
            env_path=None,
            port_offset=0,
            env_base_port=5004,
        )
    config = dataclasses.replace(config, env_config=env_config)
    if config.env_config.num_envs < 1:
        raise SystemExit("num_envs must be positive")
    
    # 加载模型。传统控制器没有可训练权重，不需要 checkpoint。
    requires_checkpoint = getattr(config, "requires_checkpoint", True)
    model_path = ""
    if requires_checkpoint:
        if args.checkpoint_path is None:
            print("Error: checkpoint_path is not set.")
            return

        checkpoint_path = Path(args.checkpoint_path).expanduser()
        if not checkpoint_path.is_file():
            print(f"Error: checkpoint_path does not exist or is not a file: {checkpoint_path}")
            return

        model_path = str(checkpoint_path.resolve())
        print(f"Using checkpoint: {model_path}\n")
    else:
        print("Static controller config: no checkpoint lookup needed.\n")
    
    # 创建环境（评估时使用实时速度）
    eval_port = config.env_config.env_base_port + config.env_config.port_offset
    print(
        f"Creating environment (no_graphics={False if use_editor else args.no_graphics}, "
        f"base_port={config.env_config.env_base_port}, worker_id={config.env_config.port_offset}, "
        f"mlagents_port={eval_port}, use_editor={use_editor})..."
    )
    if use_editor:
        print("Editor mode enabled: open Unity Editor, load the target scene, then press Play to connect.")
    if config.env_config.parallel_mode == "multi_area":
        env = make_unity_training_areas_env(
            None if use_editor else args.env_path,
            config.env_config,
            no_graphics=False if use_editor else args.no_graphics,
            time_scale=args.time_scale if args.time_scale is not None else 1.0,
            worker_id=config.env_config.port_offset,
            base_port=config.env_config.env_base_port,
            window_width=args.window_width,
            window_height=args.window_height,
        )
    else:
        from finssim_rl.training.managed_subproc_vec_env import ManagedSubprocVecEnv
        env = ManagedSubprocVecEnv([
            make_unity_env_for_eval(
                None if use_editor else args.env_path, env_config=config.env_config,
                no_graphics=False if use_editor else args.no_graphics,
                time_scale=args.time_scale if args.time_scale is not None else config.env_config.time_scale,
                worker_id=config.env_config.port_offset + i,
                base_port=config.env_config.env_base_port,
                window_width=args.window_width, window_height=args.window_height,
            ) for i in range(config.env_config.num_envs)
        ])
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Experiment: {args.exp_name}")
    print(f"Model type: {config.model_type}")
    print(f"Number of episodes: {args.num_episodes}\n")
    if args.output_dir:
        print(f"Output Directory: {args.output_dir}")
    if args.resolved_config:
        print(f"Resolved FinsSim Config: {args.resolved_config}")
    if args.force_zero_thrust:
        print("Force-zero-thrust mode: ON (all actions are set to 0.0 each step)\n")

    # 注册信号处理和清理函数
    def cleanup_and_exit(_signum, _frame):
        print("\n\nReceived interrupt signal! Cleaning up...")
        print("Closing Unity environment...")
        env.close()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    # 加载模型
    print(f"Loading model...")
    model = config.load_model_for_eval(model_path, device=args.device, env=env)
    
    # 运行评估
    print(f"Starting evaluation ({args.num_episodes} episodes)...\n")
    episode_rewards = []
    episode_steps = []
    distance_tracker = OneChaseOneDistanceTracker(env.num_envs) if (
        hasattr(env, "num_envs") and is_one_chase_one_config(config)
    ) else None
    tensorboard_writer = None
    if args.output_dir:
        tensorboard_writer = SummaryWriter(log_dir=str(Path(args.output_dir) / "tensorboard" / "manual_eval"))
    
    try:
        # SB3 VecEnv returns vector-valued rewards even when it contains a
        # single slot (notably a one-area UnityTrainingAreasVecEnv).  Handle
        # all VecEnv instances through the same per-slot accounting path.
        if hasattr(env, "num_envs"):
            obs = env.reset()
            if distance_tracker is not None:
                distance_tracker.reset(obs)
            rewards_by_slot = np.zeros(env.num_envs, dtype=np.float64)
            steps_by_slot = np.zeros(env.num_envs, dtype=np.int64)
            while len(episode_rewards) < args.num_episodes:
                action, _states = model.predict(obs, deterministic=True)
                if args.force_zero_thrust:
                    action = np.zeros_like(action, dtype=np.float32)
                obs, rewards, dones, infos = env.step(action)
                distance_summaries = {}
                if distance_tracker is not None:
                    for slot in range(env.num_envs):
                        distance_summaries[slot] = distance_tracker.record_transition(
                            slot, obs, bool(dones[slot]), infos[slot],
                        )
                rewards_by_slot += rewards
                steps_by_slot += 1
                for slot in np.flatnonzero(dones):
                    distance_summary = distance_summaries.get(int(slot))
                    episode_rewards.append(float(rewards_by_slot[slot]))
                    episode_steps.append(int(steps_by_slot[slot]))
                    episode_index = len(episode_rewards)
                    if distance_summary is not None:
                        print(
                            f"Episode {episode_index:3d} distance: "
                            f"min = {distance_summary.minimum_m:.3f} m, "
                            f"final = {distance_summary.final_m:.3f} m"
                        )
                        if tensorboard_writer is not None:
                            tensorboard_writer.add_scalar(
                                "eval_episode/min_distance_to_prey_m",
                                distance_summary.minimum_m,
                                episode_index,
                            )
                            tensorboard_writer.add_scalar(
                                "eval_episode/final_distance_to_prey_m",
                                distance_summary.final_m,
                                episode_index,
                            )
                    print(
                        f"Episode {episode_index:3d} (area {slot:4d}): "
                        f"Reward = {rewards_by_slot[slot]:8.2f}, Steps = {steps_by_slot[slot]:5d}"
                    )
                    rewards_by_slot[slot] = 0.0
                    steps_by_slot[slot] = 0
                    if len(episode_rewards) >= args.num_episodes:
                        break
        else:
            for episode in range(args.num_episodes):
                # 处理不同版本 gym 的返回值差异
                reset_result = env.reset()
                obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
                if hasattr(model, "reset"):
                    model.reset()

                episode_reward = 0
                done = False
                step = 0

                while not done:
                    # 使用确定性策略进行评估
                    action, _states = model.predict(obs, deterministic=True)

                    if args.control_debug_interval > 0 and step % args.control_debug_interval == 0:
                        format_diagnostics = getattr(model, "format_diagnostics", None)
                        if callable(format_diagnostics):
                            print(f"Episode {episode + 1:3d}, Step {step:5d}: {format_diagnostics()}")

                    # 基线测试：强制零推力动作
                    if args.force_zero_thrust:
                        action = np.zeros_like(action, dtype=np.float32)

                    step_result = env.step(action)

                    # 处理不同版本的返回值（Gym vs Gymnasium）
                    if len(step_result) == 5:
                        obs, reward, terminated, truncated, info = step_result
                        done = terminated or truncated
                    else:
                        obs, reward, done, info = step_result

                    episode_reward += reward
                    step += 1

                episode_rewards.append(episode_reward)
                episode_steps.append(step)
                print(f"Episode {episode + 1:3d}: Reward = {episode_reward:8.2f}, Steps = {step:5d}")

        # 输出统计结果
        print(f"\n{'='*60}")
        print(f"Evaluation Results ({args.num_episodes} episodes)")
        print(f"{'='*60}")
        print(f"Mean Reward:       {np.mean(episode_rewards):8.2f}")
        print(f"Std Reward:        {np.std(episode_rewards):8.2f}")
        print(f"Min Reward:        {np.min(episode_rewards):8.2f}")
        print(f"Max Reward:        {np.max(episode_rewards):8.2f}")
        print(f"Mean Steps:        {np.mean(episode_steps):8.2f}")
        print(f"Total Steps:       {np.sum(episode_steps):8.0f}")
        if distance_tracker is not None:
            distance_metrics = distance_tracker.summary()
            for key, value in distance_metrics.items():
                print(f"{key}: {value:.3f} m")
                if tensorboard_writer is not None:
                    tensorboard_writer.add_scalar(key, value, len(episode_rewards))
        print(f"{'='*60}\n")
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user!")
    finally:
        if tensorboard_writer is not None:
            tensorboard_writer.flush()
            tensorboard_writer.close()
        print("Closing Unity environment...")
        env.close()

    print("✓ Evaluation completed!")


def main():
    """主函数"""
    args = tyro.cli(EvalArgs)
    evaluate_model(args)


if __name__ == '__main__':
    main()
