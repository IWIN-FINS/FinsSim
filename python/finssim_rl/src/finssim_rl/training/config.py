"""
配置管理系统

采用模块化设计：
- 基类和通用接口定义在此
- 具体的模型配置在 models/ 下的单独文件（包含特定的 Config、TrainingConfig、EnvironmentConfig）
- 通过注册表动态加载配置

架构：
    BaseEnvironmentConfig / BaseTrainingConfig / BaseConfig (scripts/config.py)
        ↓
    PPOEnvironmentConfig / PPOTrainingConfig / PPOConfig (models/ppo_control.py)
    SACEnvironmentConfig / SACTrainingConfig / SACConfig (models/sac_control.py)
"""
import dataclasses
from dataclasses import dataclass, field, fields
from typing import Any, Callable, Dict, Optional, Union


@dataclass
class BaseEnvironmentConfig:
    """环境基础配置"""
    no_graphics: bool = True # 不要修改此处！可能会导致服务器卡死！可视化请手动命令行传参！
    use_editor: bool = False
    flatten_branched: bool = True
    allow_multiple_obs: bool = False
    uint8_visual: bool = False
    env_path: Optional[str] = None
    time_scale: float = 10.0
    num_envs: int = 32
    parallel_mode: str = "multi_area"
    eval_num_envs: int = 1
    eval_time_scale: float = 10.0
    eval_mode: str = "asynchronous"
    eval_num_episodes: int = 5
    port_offset: int = 0
    env_base_port: int = 5005
    timeout_wait: int = 60
    seed: Optional[int] = None
    environment_parameters: dict[str, float] = field(default_factory=dict)
    unity_additional_args: list[str] = field(default_factory=list)


@dataclass
class BaseTrainingConfig:
    """训练基础配置"""
    learning_rate: Union[float, Callable[[float], float]] = 3e-4
    # Leave unset to preserve the scheduler serialized in a loaded checkpoint.
    # Explicit values are intended for continuation branches that deliberately
    # restart annealing from a saved policy.
    learning_rate_schedule: Optional[str] = None
    learning_rate_peak: Optional[float] = None
    learning_rate_min: Optional[float] = None
    learning_rate_warm_restart_cycles: Optional[int] = None
    gamma: float = 0.99
    total_timesteps: int = 500_0000
    checkpoint_freq: int = 50000
    eval_freq: int = 5000
    # Async evaluation needs immutable on-disk snapshots while jobs wait in
    # its queue. Keep them only when explicitly requested; the best snapshot
    # is always promoted to checkpoints/best_model.zip.
    keep_eval_snapshots: bool = False


@dataclass
class BaseConfig:
    """配置基类（所有模型配置的基类）

    提供模型工厂接口：子类应该实现 `create_model(env, args, load_checkpoint)`
    和 `load_model_for_eval(model_path, device)`，以便把模型构造逻辑放回
    各自的模型配置文件中。
    """
    name: str
    model_type: str  # 'PPO' or 'SAC'
    description: str = ""
    env_config: BaseEnvironmentConfig = field(default_factory=BaseEnvironmentConfig)
    training_config: BaseTrainingConfig = field(default_factory=BaseTrainingConfig)

    def create_model(self, env, args, load_checkpoint: Optional[str] = None):
        """在子类中实现：根据配置创建并返回一个训练用的模型实例。"""
        raise NotImplementedError()

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        """在子类中实现：加载并返回用于评估的模型对象。"""
        raise NotImplementedError()


_CONFIGS: Optional[list[BaseConfig]] = None
_CONFIGS_DICT: Dict[str, BaseConfig] = {}


def _load_configs() -> list[BaseConfig]:
    """懒加载配置，避免模型模块导入时触发循环导入。"""
    global _CONFIGS, _CONFIGS_DICT
    if _CONFIGS is None:
        loaded_configs = []

        # Traditional controllers have no SB3/TensorBoard dependency, so keep them
        # available even if optional RL training stacks are not importable.
        from finssim_rl.models.traditional_velocity_pid import get_traditional_velocity_pid_configs
        loaded_configs.extend(get_traditional_velocity_pid_configs().values())
        from finssim_rl.models.traditional_position_pid import get_traditional_position_pid_configs
        loaded_configs.extend(get_traditional_position_pid_configs().values())
        from finssim_rl.models.traditional_hold_position_pid import get_traditional_hold_position_pid_configs
        loaded_configs.extend(get_traditional_hold_position_pid_configs().values())
        from finssim_rl.models.traditional_hold_position_wrench import get_traditional_hold_position_wrench_configs
        loaded_configs.extend(get_traditional_hold_position_wrench_configs().values())
        from finssim_rl.models.traditional_trajectory_tracking import get_traditional_trajectory_tracking_configs
        loaded_configs.extend(get_traditional_trajectory_tracking_configs().values())
        from finssim_rl.models.traditional_chase_baselines import get_traditional_chase_baseline_configs
        loaded_configs.extend(get_traditional_chase_baseline_configs().values())

        try:
            from finssim_rl.models.ppo_control import get_ppo_control_configs
            loaded_configs.extend(get_ppo_control_configs().values())
        except Exception as exc:
            print(f"[Config WARNING] PPO configs unavailable: {exc}")

        try:
            from finssim_rl.models.ppo_control_v2 import get_ppo_control_v2_configs
            loaded_configs.extend(get_ppo_control_v2_configs().values())
        except Exception as exc:
            print(f"[Config WARNING] PPO v2 configs unavailable: {exc}")

        try:
            from finssim_rl.models.ppo_wrench_control import get_ppo_wrench_control_configs
            loaded_configs.extend(get_ppo_wrench_control_configs().values())
        except Exception as exc:
            print(f"[Config WARNING] PPO wrench configs unavailable: {exc}")

        try:
            from finssim_rl.models.sac_control import get_sac_control_configs
            loaded_configs.extend(get_sac_control_configs().values())
        except Exception as exc:
            print(f"[Config WARNING] SAC configs unavailable: {exc}")

        try:
            from finssim_rl.training.hybrid_ppo_pid_config_v3 import get_hybrid_control_configs_v3
            loaded_configs.extend(get_hybrid_control_configs_v3().values())
        except Exception as exc:
            print(f"[Config WARNING] Hybrid PPO+PID v3 configs unavailable: {exc}")

        try:
            from finssim_rl.training.hierarchy_chase_config import get_hierarchy_chase_configs
            loaded_configs.extend(get_hierarchy_chase_configs().values())
        except Exception as exc:
            print(f"[Config WARNING] Hierarchy chase configs unavailable: {exc}")

        try:
            from finssim_rl.training.hybrid_ppo_pid_residual_config import get_hybrid_control_configs_residual
            loaded_configs.extend(get_hybrid_control_configs_residual().values())
        except Exception as exc:
            print(f"[Config WARNING] Hybrid PPO+PID residual configs unavailable: {exc}")

        _CONFIGS = loaded_configs
        _CONFIGS_DICT = {config.name: config for config in _CONFIGS}
    return _CONFIGS


def get_config(config_name: str) -> BaseConfig:
    """
    获取指定名称的配置
    
    Args:
        config_name: 配置名称（例如 'ppo_chase', 'sac_aggressive'）
    
    Returns:
        配置对象
    
    Raises:
        ValueError: 如果配置不存在
    """
    if config_name.startswith("traditional_"):
        from finssim_rl.models.traditional_velocity_pid import get_traditional_velocity_pid_configs
        traditional_configs = get_traditional_velocity_pid_configs()
        if config_name in traditional_configs:
            return traditional_configs[config_name]

    configs_dict = {config.name: config for config in _load_configs()}
    if config_name not in configs_dict:
        raise ValueError(
            f"Unknown config: {config_name}. "
            f"Available configs: {', '.join(list_configs())}"
        )
    return configs_dict[config_name]


def list_configs() -> list:
    """
    列出所有可用的配置
    
    Returns:
        配置名称列表
    """
    return sorted([config.name for config in _load_configs()])


def list_configs_by_model(model_type: str) -> list:
    """
    列出特定模型类型的所有配置
    
    Args:
        model_type: 模型类型（'PPO' 或 'SAC'）
    
    Returns:
        该模型类型的所有配置名称
    """
    return sorted([
        config.name for config in _load_configs()
        if config.model_type == model_type
    ])


def register_config(config: BaseConfig) -> None:
    """
    手动注册配置（用于扩展或动态配置）

    Args:
        config: 要注册的配置对象
    """
    configs = _load_configs()
    configs.append(config)
    _CONFIGS_DICT[config.name] = config


def merge_args_with_config(args: Any, config: Any) -> Any:
    """
    自动将 args 中非 None 的同名值覆盖到 config 对象。

    特殊约定：
    - args.show_graphics=True 会覆盖环境配置中的 no_graphics=False。

    Args:
        args: 命令行参数对象
        config: 配置对象

    Returns:
        合并后的配置对象副本
    """
    config_fields = {f.name for f in fields(config)}
    args_dict = {}

    for f in fields(args):
        if f.name in config_fields and getattr(args, f.name) is not None:
            args_dict[f.name] = getattr(args, f.name)

    if "no_graphics" in config_fields and getattr(args, "show_graphics", False):
        args_dict["no_graphics"] = False

    return dataclasses.replace(config, **args_dict)


def print_all_configs() -> None:
    """打印所有可用配置的详细信息"""
    print(f"\n{'='*70}")
    print(f"{'Available Configurations':<70}")
    print(f"{'='*70}")
    
    configs_by_model = {}
    for config in sorted(_load_configs(), key=lambda c: c.name):
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
            print(f"    Learning Rate: {tc.learning_rate}")
            print(f"    Total Timesteps: {tc.total_timesteps:,}")
            print(f"    Batch Size: {tc.batch_size}")
            if config.model_type == "SAC":
                print(f"    Buffer Size: {tc.buffer_size:,}")
            print()
    
    print(f"{'='*70}\n")


__all__ = [
    "BaseEnvironmentConfig",
    "BaseTrainingConfig",
    "BaseConfig",
    "get_config",
    "list_configs",
    "list_configs_by_model",
    "register_config",
    "print_all_configs",
    "merge_args_with_config",
]
