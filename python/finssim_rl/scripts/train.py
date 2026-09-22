"""
统一的训练脚本

使用方式:
    python -m scripts.train --config=ppo_chase --exp-name ppo_test --overwrite
    python -m scripts.train --config=sac_default --exp-name sac_test --resume
    python -m scripts.train --config=sac_aggressive --num-envs 64
"""

import signal
import sys
import tyro
import dataclasses
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finssim_rl.training.args import TrainArgs
from finssim_rl.training.config import get_config, list_configs, merge_args_with_config
from finssim_rl.models import (
    make_unity_env,
    make_unity_env_for_eval,
    make_unity_training_areas_env,
)
from finssim_rl.training.trainer import (
    find_latest_checkpoint,
    setup_directories,
    make_callbacks,
    create_model,
)
from finssim_rl.training.artifacts import resolve_training_output_paths, write_resolved_run_config
from finssim_rl.training.managed_subproc_vec_env import ManagedSubprocVecEnv
from finssim_rl.training.reward_protocols import (
    compile_reward_config_from_resolved_config,
    load_resolved_config,
    resolved_unity_environment_parameters,
)


def resolve_output_paths(args: TrainArgs) -> Path:
    """Resolve standard artifact paths for standalone or FinsSim-managed runs."""
    return resolve_training_output_paths(args, backend="rl")


def _with_test_suffix(exp_name: str) -> str:
    return exp_name if exp_name.endswith("_test") else f"{exp_name}_test"


def _merge_unity_additional_args_json(args: TrainArgs) -> TrainArgs:
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


def _apply_resolved_reward_config(args: TrainArgs, env_config):
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
        # Explicit experiment YAML values take precedence over protocol defaults.
        parameters.update(unity_parameters)
        print(
            "[Unity Environment Parameters] "
            f"parameters={len(unity_parameters)} keys={', '.join(sorted(unity_parameters))}"
        )

    if parameters == dict(getattr(env_config, "environment_parameters", {}) or {}):
        return env_config
    return dataclasses.replace(env_config, environment_parameters=parameters)


def _resolve_load_checkpoint(args: TrainArgs) -> tuple[str | None, str | None]:
    """Resolve either same-run resume or an explicit warm-start checkpoint."""
    initialization_count = sum(
        bool(value)
        for value in (args.resume, args.init_checkpoint, args.init_actor_checkpoint)
    )
    if initialization_count > 1:
        raise SystemExit(
            "--resume, --init-checkpoint, and --init-actor-checkpoint are mutually exclusive"
        )

    if args.init_checkpoint:
        checkpoint = Path(args.init_checkpoint).expanduser().resolve()
        if not checkpoint.is_file():
            raise SystemExit(f"--init-checkpoint does not exist or is not a file: {checkpoint}")
        return str(checkpoint), "initializing new run from"

    if args.resume:
        checkpoint = find_latest_checkpoint(args.checkpoint_dir)
        if not checkpoint:
            raise SystemExit(
                f"--resume requested but no checkpoint was found in: {args.checkpoint_dir}"
            )
        return checkpoint, "resuming from"

    return None, None


def _resolve_actor_checkpoint(args: TrainArgs) -> str | None:
    """Resolve an actor-only initialization checkpoint for a fresh run."""
    if not args.init_actor_checkpoint:
        return None
    checkpoint = Path(args.init_actor_checkpoint).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(
            "--init-actor-checkpoint does not exist or is not a file: "
            f"{checkpoint}"
        )
    return str(checkpoint)


def _resolve_training_steps(args: TrainArgs, configured_total: int, model_num_timesteps: int) -> int:
    """Return the SB3 budget for this invocation.

    ``model.learn(..., reset_num_timesteps=False)`` interprets its
    ``total_timesteps`` argument as an additional budget.  Same-run resume is
    instead defined as reaching the configured run-wide target, so subtract
    the counter restored from the checkpoint.  An explicit init checkpoint is
    intentionally still a warm start with a fresh invocation budget.
    """
    if configured_total < 0:
        raise ValueError("training_config.total_timesteps must not be negative")
    if not args.resume:
        return configured_total

    remaining = max(configured_total - max(model_num_timesteps, 0), 0)
    print(
        "[Resume] target_total_timesteps="
        f"{configured_total:,} checkpoint_timesteps={model_num_timesteps:,} "
        f"remaining_timesteps={remaining:,}"
    )
    return remaining


def main(args: TrainArgs) -> None:
    """
    主训练函数
    
    Args:
        args: 命令行参数
    """
    args = _merge_unity_additional_args_json(args)

    # 获取配置
    try:
        base_config = get_config(args.config)
        config = base_config
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable configs: {', '.join(list_configs())}")
        return
    
    # 如果没有指定 exp_name，使用 config 的名称
    if args.exp_name == "":
        args.exp_name = args.config
    if args.test:
        args.exp_name = _with_test_suffix(args.exp_name)
        
    # 生成最终配置：命令行参数只用于覆盖 config，后续统一读取 config。
    env_config = merge_args_with_config(args, config.env_config)
    env_config = _apply_resolved_reward_config(args, env_config)
    training_config = merge_args_with_config(args, config.training_config)
    config = dataclasses.replace(config, env_config=env_config, training_config=training_config)
    if args.test:
        config = dataclasses.replace(
            config,
            env_config=dataclasses.replace(
                config.env_config,
                num_envs=1,
                eval_num_envs=1,
            ),
        )
        print("[Test Mode] Overriding training/evaluation env and area counts to 1.")

    if getattr(config.env_config, "use_editor", False):
        print("Error: Unity Editor mode is only supported for single-environment eval, not training.")
        print("Please use a Unity binary for training, or run finssim rl eval with use_editor enabled.")
        return

    if (
        config.env_config.num_envs < 1
    ):
        raise SystemExit("num_envs must be positive")
    if config.env_config.parallel_mode not in {"multi_area", "multi_binary"}:
        raise SystemExit("parallel_mode must be multi_area or multi_binary")

    if not config.env_config.env_path: # 两处都没有指定env_path
        print("Error: No Unity environment path specified!")
        print("Please provide the path using --env-path or specify it in the config.")
        return

    # Resolve logs/checkpoints/TensorBoard after exp_name and FinsSim output_dir are known.
    resolve_output_paths(args)
    checkpoint_path, checkpoint_action = _resolve_load_checkpoint(args)
    actor_checkpoint_path = _resolve_actor_checkpoint(args)
    
    # 设置目录
    setup_directories(args)
    resolved_snapshot = write_resolved_run_config(
        args.output_dir,
        config_name=args.config,
        base_config=base_config,
        final_config=config,
        cli_args=args,
    )
    
    # 创建并行环境
    if config.env_config.parallel_mode == "multi_area":
        print(f"Creating one Unity binary with {config.env_config.num_envs} replicated training areas...")
        env = make_unity_training_areas_env(config.env_config.env_path, config.env_config)
    else:
        print(f"Creating {config.env_config.num_envs} parallel Unity binaries...")
        env = ManagedSubprocVecEnv(
            [
                make_unity_env(
                    config.env_config.env_path,
                    i,
                    config.env_config,
                )
                for i in range(config.env_config.num_envs)
            ]
        )
    
    # Eval must use a second Unity Player.  A worker id after all training
    # workers gives it a distinct ML-Agents port while keeping the configured
    # base port visible in one place.
    train_player_count = 1 if config.env_config.parallel_mode == "multi_area" else config.env_config.num_envs
    eval_worker_id = config.env_config.port_offset + train_player_count
    eval_env_config = dataclasses.replace(
        config.env_config,
        num_envs=config.env_config.eval_num_envs,
        time_scale=config.env_config.eval_time_scale,
    )
    if eval_env_config.parallel_mode == "multi_area":
        print(f"Creating separate Unity evaluation binary with {eval_env_config.num_envs} replicated areas...")
        eval_env = make_unity_training_areas_env(
            eval_env_config.env_path,
            eval_env_config,
            worker_id=eval_worker_id,
        )
    else:
        print(f"Creating {eval_env_config.num_envs} separate Unity evaluation binaries...")
        eval_env = ManagedSubprocVecEnv([
            make_unity_env_for_eval(
                eval_env_config.env_path, eval_env_config,
                no_graphics=eval_env_config.no_graphics,
                time_scale=eval_env_config.time_scale,
                worker_id=eval_worker_id + i,
            ) for i in range(eval_env_config.num_envs)
        ])

    # 打印信息
    print(f"\n{'='*60}")
    print(f"Configuration: {config.name}")
    print(f"Model Type: {config.model_type}")
    print(f"Experiment Name: {args.exp_name}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Parallel Mode: {config.env_config.parallel_mode}")
    print(f"Training Effective Vector Slots: {config.env_config.num_envs}")
    print(f"Effective Vector Slots: {env.num_envs}")
    print(f"Evaluation Unity starts at worker_id={eval_worker_id}; effective slots={eval_env.num_envs}")
    print(f"Unity Graphics: {'shown' if not config.env_config.no_graphics else 'hidden'}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"TensorBoard Directory: {args.tensorboard_dir}")
    print(f"Resolved Run Config: {resolved_snapshot}")
    if args.resolved_config:
        print(f"Input FinsSim Config: {args.resolved_config}")
    print(f"Device: {args.device}")
    print(f"{'='*60}\n")
    
    # 创建回调函数
    callback_list = make_callbacks(env, args, config, eval_env=eval_env)

    # --resume continues this run to its configured global step target;
    # --init-checkpoint restores compatible policy/optimizer state; and
    # --init-actor-checkpoint transfers only the compatible actor into a fresh
    # critic/optimizer/timestep state for reward-shaping fine tuning.
    if checkpoint_path:
        print(f"{checkpoint_action.capitalize()} checkpoint: {checkpoint_path}")
    if actor_checkpoint_path:
        print(f"Initializing actor only from checkpoint: {actor_checkpoint_path}")
    
    model = create_model(
        env=env,
        config=config,
        args=args,
        load_checkpoint=checkpoint_path,
    )
    if actor_checkpoint_path:
        initialize_actor = getattr(model, "initialize_actor_from_checkpoint", None)
        if not callable(initialize_actor):
            raise SystemExit(
                "--init-actor-checkpoint is supported only by PPO virtual-control/wrench models."
            )
        copied_parameters = initialize_actor(actor_checkpoint_path)
        print(
            "Actor-only initialization complete: "
            f"copied {len(copied_parameters)} actor parameters; critic/optimizer/timesteps are fresh."
        )

    env_closed = False

    def close_env_once() -> None:
        nonlocal env_closed
        if env_closed:
            return
        env_closed = True
        print("Closing Unity environments...")
        try:
            env.close()
        except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
            print(f"[Cleanup WARNING] Unity environment was already closed: {exc}")
        try:
            eval_env.close()
        except (BrokenPipeError, EOFError, ConnectionResetError) as exc:
            print(f"[Cleanup WARNING] Evaluation Unity environment was already closed: {exc}")

    # 注册信号处理和清理函数
    def cleanup_and_exit(_signum, _frame):
        print("\n\nReceived interrupt signal! Cleaning up...")
        close_env_once()
        sys.exit(0)

    signal.signal(signal.SIGINT, cleanup_and_exit)
    signal.signal(signal.SIGTERM, cleanup_and_exit)

    training_steps = _resolve_training_steps(
        args,
        int(config.training_config.total_timesteps),
        int(getattr(model, "num_timesteps", 0)),
    )

    # 开始训练
    print(f"Starting training...")
    try:
        if training_steps > 0:
            model.learn(
                total_timesteps=training_steps,
                callback=callback_list,
                progress_bar=True,
                reset_num_timesteps=checkpoint_path is None,
                tb_log_name=args.exp_name,
            )
        else:
            print("[Resume] checkpoint already meets or exceeds total_timesteps; no rollout is needed.")
    except KeyboardInterrupt:
        print("\n\nTraining interrupted by user! Cleaning up...")
    finally:
        for callback in callback_list.callbacks:
            close = getattr(callback, "close", None)
            if close is not None:
                close()
        # 确保环境被关闭
        close_env_once()
    
    # 保存最终模型
    final_model_path = str(Path(args.checkpoint_dir) / "final_model.zip")
    print(f"\nTraining complete! Saving final model to {final_model_path}")
    model.save(final_model_path)
    
    print(f"Experiment '{args.exp_name}' finished successfully!")


if __name__ == '__main__':
    args = tyro.cli(TrainArgs)
    if args.debug:
        import debugpy

        debugpy.listen(("0.0.0.0", 10092))
        print("🔍 Waiting for debugger attach on port 10092... Please attach your VS Code debugger.")
        debugpy.wait_for_client()
    main(args)

"""
使用说明：
1. 注意开代理，因为unity ml-agents通信的时候不知道为什么就会从google下载一个proto相关的通信协议
2. 小心端口冲突。有时候运行训练脚本发现环境创建失败无法连接，则需要查看是否还有别人在运行的训练脚本，需要在命令行中指定一下`--port_offset`。因为`--num-cpu`比较多的话，对端口占用还是比较大的
3. 训练中断（Ctrl+C）后会自动清理启动的 Unity 进程，无需手动清理

"""
