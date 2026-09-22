"""
配置管理系统

采用模块化设计：
- BaseEnvironmentConfig / BaseTrainingConfig / BaseConfig 定义基类和通用接口
- 具体的场景配置在各自文件中（如 chasing_3_chase_1_config.py）
- 通过注册表动态加载配置

TrainConfig 是供 scripts/train.py 使用的最终配置类。
"""
from dataclasses import dataclass, field, fields, asdict, is_dataclass
from typing import Any, Callable, Dict, Optional


# ============================================================================
# TrainConfig - 供 train.py 使用的最终配置
# ============================================================================
@dataclass
class TrainConfig:
    """Main training configuration for scripts/train.py."""

    # Experiment control
    exp_name: Optional[str] = None
    output_dir: Optional[str] = None
    resolved_config: Optional[str] = None
    overwrite: bool = False
    auto_resume: bool = False
    resume_from: Optional[str] = None
    checkpoint_dir: str = "checkpoints"
    tensorboard_dir: Optional[str] = None
    save_freq: int = 10

    # Environment
    env_type: str = "unity"
    env_base_port: int = 5005
    unity_env_binary_path: str = "/RLChase/build/RLChase.x86_64"
    timeout_wait: int = 360
    num_envs: int = 8              # 并行环境数
    batch_size: int = 512           # 触发训练的episode总数阈值
    training_time_scale: float = 10.0
    parallel_mode: str = "multi_area"
    use_editor: bool = False
    herder_action_source: str = "python_policy"
    netter_action_source: str = "python_policy"
    prey_action_source: str = "wrapper_escape"
    env_restart_attempts: int = 3
    no_graphics: bool = True
    environment_parameters: Dict[str, float] = field(default_factory=dict)
    curriculum: Dict[str, Any] = field(default_factory=dict)

    # Network architecture
    actor_hidden_dim: int = 64
    actor_num_layers: int = 3
    critic_hidden_dim: int = 128
    critic_num_layers: int = 3

    # Training
    total_timesteps: int = 10_000_000
    epochs: int = 8
    learning_rate_actor: float = 0.0008
    learning_rate_critic: float = 0.0008
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_return: bool = True
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = -1

    # 3chase1 specific
    chaser_team_obs_dim: int = 30
    chaser_team_action_dim: int = 8
    chaser_team_reward_dim: int = 3

    # Evaluation
    eval_steps: int = 5
    num_eval_ep: int = 10
    num_eval_envs: int = 4
    eval_time_scale: float = 1.0
    eval_mode: str = "asynchronous"
    keep_eval_snapshots: bool = False

    # Logging
    log_every: int = 5
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""

    # Misc
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "runs"


# ============================================================================
# 工具函数
# ============================================================================
def merge_args_to_dict(args: Any, prefix: str = "") -> dict:
    """Convert dataclass args to dict, including nested dataclasses.

    Args:
        args: Dataclass instance
        prefix: Optional prefix for keys

    Returns:
        Dictionary of configuration
    """
    result = {}
    for f in fields(args):
        value = getattr(args, f.name)
        key = prefix + f.name
        if is_dataclass(value):
            result.update(merge_args_to_dict(value, key + "_"))
        else:
            result[key] = value
    return result


def override_from_args(config: Any, args: dict) -> Any:
    """Override config fields from args dict.

    Only updates fields that exist in both.

    Args:
        config: Dataclass instance
        args: Dictionary with potential overrides

    Returns:
        Updated config (same object)
    """
    for f in fields(config):
        key = f.name
        if key in args and args[key] is not None:
            setattr(config, f.name, args[key])
    return config


# ============================================================================
# 基础配置类
# ============================================================================


@dataclass
class BaseEnvironmentConfig:
    """环境基础配置"""
    env_path: Optional[str] = None
    env_base_port: int = 5005
    timeout_wait: int = 60
    time_scale: float = 10.0
    env_type: str = "3chase1"
    no_graphics: bool = True
    parallel_mode: str = "multi_area"
    use_editor: bool = False
    flatten_branched: bool = True
    allow_multiple_obs: bool = False
    uint8_visual: bool = False
    environment_parameters: Dict[str, float] = field(default_factory=dict)


@dataclass
class BaseTrainingConfig:
    """训练基础配置

    所有具体场景的 training_config 应继承此类。
    eval_steps 和 num_eval_ep 控制评估频率。
    """
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_return: bool = False
    epochs: int = 10
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    clip_gradients: float = 0.5
    total_timesteps: int = 10_000_000
    log_every: int = 10
    save_freq: int = 50_000
    eval_steps: int = 5
    num_eval_ep: int = 10
    num_eval_envs: int = 4
    eval_time_scale: float = 1.0
    eval_mode: str = "asynchronous"
    # Background evaluation snapshots are transient by default; best.pt is
    # retained independently. Enable only for offline snapshot analysis.
    keep_eval_snapshots: bool = False
    batch_size: int = 512


@dataclass
class BaseConfig:
    """配置基类（所有模型配置的基类）

    提供模型工厂接口：子类应该实现工厂方法，
    以便把模型构造逻辑放回各自的配置文件中。
    """
    name: str
    model_type: str  # 'MAPPO' 等
    description: str = ""
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)

    # 场景通用参数
    num_envs: int = 8
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    wandb_project: str = "finssim_marl"
    wandb_entity: str = ""
    curriculum: Dict[str, Any] = field(default_factory=dict)

    def create_train_config(self) -> Any:
        """创建 TrainConfig 实例（供 train.py 使用）"""
        raise NotImplementedError()

    def create_mappo_config(self, train_config: Any) -> Any:
        """创建 MAPPO 配置"""
        raise NotImplementedError()

    def create_algorithm(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        role_ids: Any,
        mappo_config: Any,
    ) -> Any:
        """创建算法实例"""
        raise NotImplementedError()

    def create_rollout_buffer(
        self,
        buffer_size: int,
        obs_space: int,
        state_space: int,
        action_space: int,
        num_agents: int,
        role_ids: Any,
        chaser_team_obs_dim: int,
        chaser_team_reward_dim: int,
        device: str,
    ) -> Any:
        """创建 RolloutBuffer"""
        raise NotImplementedError()


# ============================================================================
# 导入所有配置
# ============================================================================
from finssim_marl.training.chasing_3_chase_1_config import get_chasing_configs
from finssim_marl.training.trinet_capture_config import get_trinet_capture_configs


# ============================================================================
# 定义所有配置列表
# ============================================================================
_CONFIGS = [
    #
    # Chasing Configs (3chase1, 2chase1, etc.)
    #
    *get_chasing_configs(),
    *get_trinet_capture_configs(),
]

# 构建配置查找字典
_CONFIGS_DICT: Dict[str, BaseConfig] = {config.name: config for config in _CONFIGS}


def get_config(config_name: str) -> BaseConfig:
    """
    获取指定名称的配置

    Args:
        config_name: 配置名称（例如 'chasing_3_chase_1'）

    Returns:
        配置对象

    Raises:
        ValueError: 如果配置不存在
    """
    if config_name not in _CONFIGS_DICT:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available configs: {', '.join(list_configs())}"
        )
    return _CONFIGS_DICT[config_name]


def list_configs() -> list:
    """
    列出所有可用的配置

    Returns:
        配置名称列表
    """
    return sorted([config.name for config in _CONFIGS])


def list_configs_by_model(model_type: str) -> list:
    """
    列出特定模型类型的所有配置

    Args:
        model_type: 模型类型（'MAPPO' 等）

    Returns:
        该模型类型的所有配置名称
    """
    return sorted([
        config.name for config in _CONFIGS
        if config.model_type == model_type
    ])


def register_config(config: BaseConfig) -> None:
    """
    手动注册配置（用于扩展或动态配置）

    Args:
        config: 要注册的配置对象
    """
    _CONFIGS.append(config)
    _CONFIGS_DICT[config.name] = config


def print_all_configs() -> None:
    """打印所有可用配置的详细信息"""
    print(f"\n{'='*70}")
    print(f"{'Available Configurations':<70}")
    print(f"{'='*70}")

    configs_by_model: Dict[str, list] = {}
    for config in sorted(_CONFIGS, key=lambda c: c.name):
        model_type = config.model_type
        if model_type not in configs_by_model:
            configs_by_model[model_type] = []
        configs_by_model[model_type].append(config)

    for model_type in sorted(configs_by_model.keys()):
        print(f"\n{model_type} Configurations:")
        print("-" * 70)

        for config in configs_by_model[model_type]:
            desc = config.description or "No description"
            tc = config.training_config
            print(f"  • {config.name}")
            print(f"    Description: {desc}")
            print(f"    Learning Rate (Actor): {tc.learning_rate_actor}")
            print(f"    Learning Rate (Critic): {tc.learning_rate_critic}")
            print(f"    Total Timesteps: {tc.total_timesteps:,}")
            print(f"    Num Envs: {config.num_envs}")
            print()

    print(f"{'='*70}\n")


__all__ = [
    "TrainConfig",
    "BaseEnvironmentConfig",
    "BaseTrainingConfig",
    "BaseConfig",
    "get_config",
    "list_configs",
    "list_configs_by_model",
    "register_config",
    "print_all_configs",
    "merge_args_to_dict",
    "override_from_args",
]
