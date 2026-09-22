"""
命令行参数和通用配置
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class TrainArgs:
    """通用训练参数"""
    
    # 实验配置
    config: str = field(
        metadata={"help": "Configuration name (e.g., ppo_chase, sac_default)"}
    )
    exp_name: str = field(
        default="",
        metadata={"help": "Experiment name for logging"}
    )
    
    # 恢复和覆盖选项
    resume: bool = field(
        default=False,
        metadata={"help": "Resume training from the latest checkpoint in this run's checkpoint directory"}
    )
    init_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Initialize a new run from an explicit checkpoint. This restores the "
                "model/optimizer/timestep state but resets the Unity environment. "
                "Mutually exclusive with --resume."
            )
        },
    )
    init_actor_checkpoint: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Initialize only a compatible PPO virtual-control actor from an explicit "
                "checkpoint. Critic, optimizer, and timestep state are freshly initialized. "
                "Mutually exclusive with --resume and --init-checkpoint."
            )
        },
    )
    overwrite: bool = field(
        default=False,
        metadata={"help": "Overwrite existing experiment directory"}
    )
    debug: bool = field(
        default=False,
        metadata={"help": "Wait for a debugpy debugger to attach before training"}
    )
    test: bool = field(
        default=False,
        metadata={"help": "Run in lightweight test mode: num_envs=1 and num_eval_envs=1"}
    )
    
    # 路径配置
    env_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Unity environment executable"}
    )
    log_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Log directory (auto-generated if not specified)"}
    )
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint directory (auto-generated if not specified)"}
    )
    tensorboard_dir: Optional[str] = field(
        default=None,
        metadata={"help": "TensorBoard directory (auto-generated if not specified)"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "FinsSim artifact run directory"}
    )
    resolved_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the resolved FinsSim config YAML"}
    )
    
    # 并行处理
    num_envs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of parallel environments"}
    )
    eval_num_envs: Optional[int] = field(
        default=None,
        metadata={"help": "Number of evaluation vector slots"}
    )
    parallel_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Unity parallel mode: multi_area (default) or multi_binary"},
    )
    time_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Unity simulation time scale"}
    )
    eval_time_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Unity evaluation time scale"},
    )
    eval_mode: Optional[str] = field(
        default=None,
        metadata={"help": "Evaluation mode: asynchronous or serial"},
    )
    eval_num_episodes: Optional[int] = field(
        default=None,
        metadata={"help": "Completed episodes aggregated across all evaluation slots"},
    )
    show_graphics: bool = field(
        default=False,
        metadata={"help": "Show the Unity graphics window during training"}
    )
    
    # 设备配置
    device: str = field(
        default="cuda",
        metadata={"help": "Device to use (cuda or cpu)"}
    )

    # PPO training overrides. Keep these optional so config defaults remain authoritative.
    n_steps: Optional[int] = field(
        default=None,
        metadata={"help": "PPO rollout steps collected per environment before each update"}
    )
    batch_size: Optional[int] = field(
        default=None,
        metadata={"help": "PPO mini-batch size; should divide n_steps * num_envs"}
    )
    n_epochs: Optional[int] = field(
        default=None,
        metadata={"help": "PPO optimization epochs per rollout"}
    )
    learning_rate_schedule: Optional[str] = field(
        default=None,
        metadata={
            "help": (
                "Optional continuation scheduler. Supported value: "
                "cosine_warm_restarts"
            )
        },
    )
    learning_rate_peak: Optional[float] = field(
        default=None,
        metadata={"help": "Peak learning rate for an explicit continuation scheduler"},
    )
    learning_rate_min: Optional[float] = field(
        default=None,
        metadata={"help": "Minimum learning rate for an explicit continuation scheduler"},
    )
    learning_rate_warm_restart_cycles: Optional[int] = field(
        default=None,
        metadata={"help": "Cosine cycles over this training invocation"},
    )
    total_timesteps: Optional[int] = field(
        default=None,
        metadata={
            "help": (
                "Training budget. With --resume, this is the run-wide target "
                "num_timesteps and only the remaining steps are trained. With "
                "--init-checkpoint or --init-actor-checkpoint, this is the new run budget."
            )
        },
    )
    checkpoint_freq: Optional[int] = field(
        default=None,
        metadata={"help": "Checkpoint interval measured in aggregate environment transitions"},
    )
    eval_freq: Optional[int] = field(
        default=None,
        metadata={"help": "Evaluation interval measured in aggregate environment transitions"},
    )
    keep_eval_snapshots: Optional[bool] = field(
        default=None,
        metadata={
            "help": (
                "Keep every completed asynchronous evaluation snapshot under "
                "checkpoints/eval_snapshots. By default only best_model.zip is retained."
            )
        },
    )

    # unity端口偏移
    port_offset: Optional[int] = field(
        default=None,
        metadata={"help": "Port offset for Unity environments"}
    )
    
    env_base_port: Optional[int] = field(
        default=None,
        metadata={"help": "Base port for Unity environments"}
    )

    timeout_wait: Optional[int] = field(
        default=None,
        metadata={"help": "Seconds to wait for Unity ML-Agents startup handshake"}
    )

    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Seed passed to Unity domain randomization"}
    )

    unity_additional_args: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Additional command-line arguments passed through to the Unity player"}
    )
    unity_additional_args_json: Optional[str] = field(
        default=None,
        metadata={"help": "JSON array of additional command-line arguments passed through to the Unity player"}
    )
    
    def __post_init__(self):
        """Paths are resolved in scripts.train after exp_name is finalized."""


Args = TrainArgs
