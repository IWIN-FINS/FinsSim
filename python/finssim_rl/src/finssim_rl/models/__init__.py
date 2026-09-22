"""
环境创建和管理
"""
from __future__ import annotations

import importlib.util
from numbers import Real
import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Callable, Mapping, Optional
from gymnasium import spaces

if TYPE_CHECKING:
    from finssim_rl.training.config import BaseEnvironmentConfig


EDITOR_PORT = 5004
ONE_CHASE_ONE_OBSERVATION_MODE_KEY = "finsim_1chase1_observation_mode"
LOCAL_UNITY_GYM_ENV_MODULE = "_finssim_local_mlagents_unity_gym_env"


def _configure_local_unity_grpc_resolver() -> None:
    """Avoid the Linux c-ares localhost stall in Unity's bundled gRPC client.

    ML-Agents' Unity player connects back to the Python RPC server through
    ``localhost``.  On some Linux hosts the bundled gRPC C-core's c-ares
    resolver never promotes that channel from CONNECTING to READY, despite the
    local server already listening.  The native resolver is deterministic for
    the local loopback endpoint and remains overrideable by callers that have
    already explicitly set ``GRPC_DNS_RESOLVER``.
    """
    if sys.platform.startswith("linux"):
        os.environ.setdefault("GRPC_DNS_RESOLVER", "native")


def _attach_unity_stats_channel(env, stats_channel) -> None:
    """Expose ML-Agents custom stats to training/evaluation callbacks."""
    setattr(env, "_finssim_stats_channel", stats_channel)


def _load_local_unity_to_gym_wrapper():
    """Load UnityToGymWrapper without importing mlagents_envs.envs.__init__.

    ML-Agents 1.1.0 registers remote example environments from
    mlagents_envs.envs.__init__. Importing mlagents_envs.envs.unity_gym_env
    therefore can trigger a network request before our local Unity build/editor
    is even contacted. We only need the local wrapper file, so load it directly
    from the installed package when present.
    """
    cached_module = sys.modules.get(LOCAL_UNITY_GYM_ENV_MODULE)
    if cached_module is not None:
        return cached_module.UnityToGymWrapper

    package_spec = importlib.util.find_spec("mlagents_envs")
    search_locations = getattr(package_spec, "submodule_search_locations", None)
    if package_spec is None or not search_locations:
        raise ImportError("Cannot find installed mlagents_envs package")

    wrapper_path = Path(next(iter(search_locations))) / "envs" / "unity_gym_env.py"
    if not wrapper_path.exists():
        from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper

        return UnityToGymWrapper

    spec = importlib.util.spec_from_file_location(LOCAL_UNITY_GYM_ENV_MODULE, wrapper_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load local UnityToGymWrapper from {wrapper_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[LOCAL_UNITY_GYM_ENV_MODULE] = module
    spec.loader.exec_module(module)
    return module.UnityToGymWrapper


def _unity_domain_randomization_args(env_config: "BaseEnvironmentConfig", rank: int = 0) -> list[str]:
    seed = getattr(env_config, "seed", None)
    if seed is None:
        return []

    worker_seed = int(seed) + int(rank) * 1_000_003
    return ["-fins-dr-seed", str(worker_seed)]


def _unity_additional_args(env_config: "BaseEnvironmentConfig", rank: int = 0) -> list[str]:
    configured_args = getattr(env_config, "unity_additional_args", None) or []
    normalized_configured_args = [str(arg) for arg in configured_args]
    if any(
        arg.lower() == "-fins-dr-seed" or arg.lower().startswith("-fins-dr-seed=")
        for arg in normalized_configured_args
    ):
        raise ValueError(
            "-fins-dr-seed is managed by the RL launcher and must not be set in "
            "unity_additional_args. Set unity.seed (or experiment.seed) instead "
            "so every Unity worker receives a distinct deterministic DR seed."
        )

    additional_args = _unity_domain_randomization_args(env_config, rank)
    additional_args.extend(normalized_configured_args)
    return additional_args


def _normalize_environment_parameters(
    parameters: Mapping[str, object] | None,
    *,
    context: str = "environment_parameters",
) -> dict[str, float]:
    if parameters is None:
        return {}
    if not isinstance(parameters, Mapping):
        raise TypeError(f"{context} must be a mapping")

    normalized: dict[str, float] = {}
    for key, value in parameters.items():
        if not isinstance(value, Real):
            raise TypeError(f"{context}.{key} must be numeric, got {type(value).__name__}")
        normalized[str(key)] = float(value)
    return normalized


def _observation_mode_parameter_from_args(additional_args: list[str]) -> dict[str, float]:
    for index, arg in enumerate(additional_args[:-1]):
        if str(arg).lower() != "-fins-1chase1-mode":
            continue

        raw_value = str(additional_args[index + 1]).strip().lower()
        if raw_value in {"direct14", "direct_local14", "hierarchy_direct14"}:
            return {ONE_CHASE_ONE_OBSERVATION_MODE_KEY: 14.0}
    return {}


def _effective_environment_parameters(
    env_config: "BaseEnvironmentConfig",
    *,
    additional_args: list[str],
) -> dict[str, float]:
    parameters = _normalize_environment_parameters(
        getattr(env_config, "environment_parameters", None),
        context="env_config.environment_parameters",
    )
    inferred = _observation_mode_parameter_from_args(additional_args)
    for key, value in inferred.items():
        parameters.setdefault(key, value)
    return parameters


def _editor_runtime_settings(
    env_config: "BaseEnvironmentConfig",
    *,
    requested_worker_id: Optional[int],
    requested_base_port: Optional[int],
    requested_no_graphics: bool,
) -> tuple[int, int, bool]:
    use_editor = bool(getattr(env_config, "use_editor", False))
    if not use_editor:
        worker = env_config.port_offset if requested_worker_id is None else requested_worker_id
        port = env_config.env_base_port if requested_base_port is None else requested_base_port
        return int(worker), int(port), requested_no_graphics

    worker = 0 if requested_worker_id is None else int(requested_worker_id)
    if worker != 0:
        raise ValueError("Unity Editor mode only supports worker_id=0.")

    port = EDITOR_PORT if requested_base_port is None else int(requested_base_port)
    if port != EDITOR_PORT:
        raise ValueError(
            f"Unity Editor mode currently requires base_port={EDITOR_PORT}, got {port}."
        )

    return worker, port, False


def _print_obs_space_structure(obs_space, prefix: str = "obs") -> None:
    """递归打印观测空间结构，便于核对 Unity 观测维度。"""
    if isinstance(obs_space, spaces.Box):
        print(f"[ObsSpace] {prefix}: Box shape={obs_space.shape}, dtype={obs_space.dtype}")
        return
    if isinstance(obs_space, spaces.Tuple):
        print(f"[ObsSpace] {prefix}: Tuple branches={len(obs_space.spaces)}")
        for i, sub in enumerate(obs_space.spaces):
            _print_obs_space_structure(sub, prefix=f"{prefix}[{i}]")
        return
    if isinstance(obs_space, spaces.Dict):
        keys = list(obs_space.spaces.keys())
        print(f"[ObsSpace] {prefix}: Dict keys={keys}")
        for key, sub in obs_space.spaces.items():
            _print_obs_space_structure(sub, prefix=f"{prefix}.{key}")
        return

    print(f"[ObsSpace] {prefix}: Unsupported space type={type(obs_space).__name__}")


def make_unity_env(
    env_path: str | None,
    rank: int,
    env_config: BaseEnvironmentConfig,
) -> Callable:
    """
    创建单个 Unity 环境的工厂函数
    
    Args:
        env_path: Unity 环境可执行文件路径
        rank: 环境索引（用于区分 worker_id）
        env_config: 环境配置
    
    Returns:
        返回一个初始化函数，用于 SubprocVecEnv
    """

    def _init():
        _configure_local_unity_grpc_resolver()
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
        from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
        from finssim_rl.training.high_level_target_debug_channel import (
            HighLevelTargetDebugChannel,
            attach_high_level_target_debug_channel,
        )

        UnityToGymWrapper = _load_local_unity_to_gym_wrapper()
        channel = EngineConfigurationChannel()
        environment_parameters_channel = EnvironmentParametersChannel()
        stats_channel = StatsSideChannel()
        high_level_target_debug_channel = HighLevelTargetDebugChannel()
        additional_args = _unity_additional_args(env_config, rank)
        environment_parameters = _effective_environment_parameters(
            env_config,
            additional_args=additional_args,
        )
        for key, value in environment_parameters.items():
            environment_parameters_channel.set_float_parameter(key, value)
        unity_env = None
        file_name = None if getattr(env_config, "use_editor", False) else env_path
        worker_id, base_port, no_graphics = _editor_runtime_settings(
            env_config,
            requested_worker_id=env_config.port_offset + rank,
            requested_base_port=env_config.env_base_port,
            requested_no_graphics=env_config.no_graphics,
        )
        
        try:
            # 创建 Unity 环境（每个进程的 worker_id 必须唯一）
            unity_env = UnityEnvironment(
                file_name=file_name,
                side_channels=[
                    channel,
                    environment_parameters_channel,
                    stats_channel,
                    high_level_target_debug_channel,
                ],
                no_graphics=no_graphics,
                worker_id=worker_id,
                base_port=base_port,
                timeout_wait=env_config.timeout_wait,
                # ``multi_binary`` owns one training area per Unity Player.
                # Keep this explicit rather than relying on ML-Agents'
                # constructor default: mesh backends must never inherit the
                # high area count used by the batched Fossen fixtures.
                num_areas=1,
                additional_args=additional_args or None,
            )
            
            # 转换为 Gym 环境
            env = UnityToGymWrapper(
                unity_env,
                uint8_visual=env_config.uint8_visual,
                flatten_branched=env_config.flatten_branched,
                allow_multiple_obs=env_config.allow_multiple_obs,
            )
            attach_high_level_target_debug_channel(env, high_level_target_debug_channel)
            _attach_unity_stats_channel(env, stats_channel)
        except Exception:
            if unity_env is not None:
                try:
                    unity_env.close()
                except Exception:
                    pass
            raise

        # 只在第一个并行环境打印，避免多进程日志刷屏。
        if rank == 0:
            worker_dr_args = _unity_domain_randomization_args(env_config, rank)
            worker_dr_seed = worker_dr_args[1] if worker_dr_args else None
            print(
                f"[UnityEnv] allow_multiple_obs={env_config.allow_multiple_obs}, "
                f"flatten_branched={env_config.flatten_branched}, uint8_visual={env_config.uint8_visual}, "
                f"domain_randomization_seed={getattr(env_config, 'seed', None)}, "
                f"worker_dr_seed={worker_dr_seed}, "
                f"use_editor={getattr(env_config, 'use_editor', False)}, "
                f"base_port={base_port}, worker_id={worker_id}"
            )
            _print_obs_space_structure(env.observation_space)
        
        # 设置时间缩放
        channel.set_configuration_parameters(time_scale=env_config.time_scale)
        
        return env

    return _init

def make_unity_env_for_eval(
    env_path: str | None,
    env_config: BaseEnvironmentConfig,
    no_graphics: bool = True,
    time_scale: float = 1.0,
    worker_id: int | None = None,
    base_port: int | None = None,
    window_width: int | None = None,
    window_height: int | None = None,
) -> Callable:
    """
    创建用于评估的 Unity 环境工厂函数（单环境，实时速度）
    
    Args:
        env_path: Unity 环境可执行文件路径
        env_config: 环境配置
    
    Returns:
        返回一个初始化函数，用于创建评估环境
    """

    def _init():
        _configure_local_unity_grpc_resolver()
        from mlagents_envs.environment import UnityEnvironment
        from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
        from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
        from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
        from finssim_rl.training.high_level_target_debug_channel import (
            HighLevelTargetDebugChannel,
            attach_high_level_target_debug_channel,
        )

        UnityToGymWrapper = _load_local_unity_to_gym_wrapper()
        channel = EngineConfigurationChannel()
        environment_parameters_channel = EnvironmentParametersChannel()
        stats_channel = StatsSideChannel()
        high_level_target_debug_channel = HighLevelTargetDebugChannel()
        additional_args = _unity_additional_args(env_config, 0)
        environment_parameters = _effective_environment_parameters(
            env_config,
            additional_args=additional_args,
        )
        for key, value in environment_parameters.items():
            environment_parameters_channel.set_float_parameter(key, value)
        if not no_graphics and window_width is not None and window_height is not None:
            additional_args.extend([
                "-screen-width",
                str(int(window_width)),
                "-screen-height",
                str(int(window_height)),
            ])
        unity_env = None
        file_name = None if getattr(env_config, "use_editor", False) else env_path
        resolved_worker_id, resolved_base_port, resolved_no_graphics = _editor_runtime_settings(
            env_config,
            requested_worker_id=worker_id,
            requested_base_port=base_port,
            requested_no_graphics=no_graphics,
        )
        
        try:
            # 创建 Unity 环境
            unity_env = UnityEnvironment(
                file_name=file_name,
                side_channels=[
                    channel,
                    environment_parameters_channel,
                    stats_channel,
                    high_level_target_debug_channel,
                ],
                no_graphics=resolved_no_graphics,
                worker_id=resolved_worker_id,
                base_port=resolved_base_port,
                timeout_wait=env_config.timeout_wait,
                # Evaluation follows the same one-Player/one-area topology
                # as multi-binary training.
                num_areas=1,
                additional_args=additional_args or None,
            )
            
            # 转换为 Gym 环境
            env = UnityToGymWrapper(
                unity_env,
                uint8_visual=env_config.uint8_visual,
                flatten_branched=env_config.flatten_branched,
                allow_multiple_obs=env_config.allow_multiple_obs,
            )
            attach_high_level_target_debug_channel(env, high_level_target_debug_channel)
            _attach_unity_stats_channel(env, stats_channel)
        except Exception:
            if unity_env is not None:
                try:
                    unity_env.close()
                except Exception:
                    pass
            raise

        print(
            f"[UnityEvalEnv] allow_multiple_obs={env_config.allow_multiple_obs}, "
            f"flatten_branched={env_config.flatten_branched}, uint8_visual={env_config.uint8_visual}, "
            f"base_port={resolved_base_port}, "
            f"worker_id={resolved_worker_id}, "
            f"domain_randomization_seed={getattr(env_config, 'seed', None)}, "
            f"use_editor={getattr(env_config, 'use_editor', False)}, "
            f"window={'editor' if getattr(env_config, 'use_editor', False) else (str(window_width) + 'x' + str(window_height) if not resolved_no_graphics else 'headless')}"
        )
        _print_obs_space_structure(env.observation_space)
        
        # 设置时间缩放为 1.0（评估时使用实时速度）
        channel.set_configuration_parameters(time_scale=time_scale)
        
        return env

    return _init


def make_unity_training_areas_env(
    env_path: str | None,
    env_config: BaseEnvironmentConfig,
    *,
    no_graphics: bool | None = None,
    time_scale: float | None = None,
    worker_id: int | None = None,
    base_port: int | None = None,
    window_width: int | None = None,
    window_height: int | None = None,
):
    """Create one Unity binary whose replicated areas form an SB3 ``VecEnv``."""
    _configure_local_unity_grpc_resolver()
    from mlagents_envs.environment import UnityEnvironment
    from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
    from mlagents_envs.side_channel.environment_parameters_channel import EnvironmentParametersChannel
    from mlagents_envs.side_channel.stats_side_channel import StatsSideChannel
    from finssim_rl.training.high_level_target_debug_channel import (
        HighLevelTargetDebugChannel,
        attach_high_level_target_debug_channel,
    )
    from finssim_rl.training.unity_training_areas_vec_env import UnityTrainingAreasVecEnv

    num_areas = int(getattr(env_config, "num_envs", 1))
    if getattr(env_config, "parallel_mode", "multi_area") != "multi_area":
        raise ValueError("make_unity_training_areas_env requires parallel_mode=multi_area")
    if num_areas < 1:
        raise ValueError("num_envs must be positive")

    channel = EngineConfigurationChannel()
    environment_parameters_channel = EnvironmentParametersChannel()
    stats_channel = StatsSideChannel()
    high_level_target_debug_channel = HighLevelTargetDebugChannel()
    additional_args = _unity_additional_args(env_config, 0)
    for key, value in _effective_environment_parameters(env_config, additional_args=additional_args).items():
        environment_parameters_channel.set_float_parameter(key, value)

    requested_no_graphics = env_config.no_graphics if no_graphics is None else bool(no_graphics)
    if not requested_no_graphics and window_width is not None and window_height is not None:
        additional_args.extend(["-screen-width", str(int(window_width)), "-screen-height", str(int(window_height))])
    resolved_worker_id, resolved_base_port, resolved_no_graphics = _editor_runtime_settings(
        env_config,
        requested_worker_id=env_config.port_offset if worker_id is None else worker_id,
        requested_base_port=env_config.env_base_port if base_port is None else base_port,
        requested_no_graphics=requested_no_graphics,
    )
    unity_env = None
    try:
        unity_env = UnityEnvironment(
            file_name=None if getattr(env_config, "use_editor", False) else env_path,
            side_channels=[channel, environment_parameters_channel, stats_channel, high_level_target_debug_channel],
            no_graphics=resolved_no_graphics,
            worker_id=resolved_worker_id,
            base_port=resolved_base_port,
            timeout_wait=env_config.timeout_wait,
            num_areas=num_areas,
            additional_args=additional_args or None,
        )
        env = UnityTrainingAreasVecEnv(
            unity_env,
            num_areas=num_areas,
            uint8_visual=env_config.uint8_visual,
            allow_multiple_obs=env_config.allow_multiple_obs,
            ready_timeout_seconds=env_config.timeout_wait,
        )
        attach_high_level_target_debug_channel(env, high_level_target_debug_channel)
        _attach_unity_stats_channel(env, stats_channel)
    except Exception:
        if unity_env is not None:
            unity_env.close()
        raise

    channel.set_configuration_parameters(time_scale=env_config.time_scale if time_scale is None else float(time_scale))
    print(
        f"[UnityTrainingAreasEnv] num_envs/areas={num_areas}, base_port={resolved_base_port}, "
        f"worker_id={resolved_worker_id}, no_graphics={resolved_no_graphics}"
    )
    _print_obs_space_structure(env.observation_space)
    return env
