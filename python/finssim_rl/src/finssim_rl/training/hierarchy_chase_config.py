"""
1Chase1 hierarchy chase config/model.

语义：
- 高层 PPO 不再学习 PID 参数，也不直接输出 8 路推进器。
- PPO 默认输出 body frame 下的局部目标点增量 [dx, dy, dz]；可选的第 4
  维是相对当前朝向的 yaw 误差命令（单位为 deg）。
- 子目标和 yaw 误差直接交给固定参数的 TraditionalPositionPID 低层控制器，
  生成 8 维推进器动作。
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from io import BufferedIOBase
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv

from finssim_rl.models.pid_controller import PIDController
from finssim_rl.models.ppo_control import PPOEnvironmentConfig, PPOTrainingConfig
from finssim_rl.models.ppo_rewrite import PPORewrite
from finssim_rl.models.traditional_position_pid import (
    TraditionalPositionPIDModel,
    TraditionalPositionPIDTrainingConfig,
)
from finssim_rl.training.config import BaseConfig
from finssim_rl.training.high_level_target_debug_channel import resolve_high_level_target_debug_channel
from finssim_rl.training.utils import linear_schedule


HIERARCHY_POLICY_ACTION_DIM = 3
HIERARCHY_POSE_YAW_POLICY_ACTION_DIM = 4
THRUSTER_ACTION_DIM = 8
HIERARCHY_ACTOR_OBS_DIM = 14


SIM_PHYSICAL_WRENCH_LIMITS_BODY: Tuple[float, float, float, float, float, float] = (
    19.528527,  # Fx surge, N
    22.501886,  # Fy heave, N
    18.415027,  # Fz sway, N
    3.863995,   # Mx roll, N*m
    7.080833,   # My yaw, N*m
    3.079985,   # Mz pitch, N*m
)


def _legacy_low_level_pid_config() -> TraditionalPositionPIDTrainingConfig:
    return replace(
        TraditionalPositionPIDTrainingConfig(),
        enable_yaw_control=False,
        enable_logging=False,
        debug_print_interval=0,
    )


def _physical_wrench_low_level_pid_config() -> TraditionalPositionPIDTrainingConfig:
    """PID outputs physical body wrench values before bounded B allocation."""
    fx_limit, fy_limit, fz_limit, _mx_limit, my_limit, _mz_limit = SIM_PHYSICAL_WRENCH_LIMITS_BODY
    return replace(
        TraditionalPositionPIDTrainingConfig(),
        enable_yaw_control=False,
        enable_logging=False,
        debug_print_interval=0,
        output_range=(-fy_limit, fy_limit),
        surge_output_limit=fx_limit,
        sway_output_limit=fz_limit,
        yaw_output_limit=my_limit,
        allocator_control_axis_ranges=SIM_PHYSICAL_WRENCH_LIMITS_BODY,
        allocator_allocation_mode="physical_wrench_allocator",
    )


def _physical_wrench_pose_yaw_low_level_pid_config() -> TraditionalPositionPIDTrainingConfig:
    """Physical wrench PID configuration for a local subgoal plus yaw error."""
    return replace(
        _physical_wrench_low_level_pid_config(),
        enable_yaw_control=True,
        # Same yaw PID tuning used by the heading-first traditional chase baseline.
        yaw_pid_params=(0.12, 0.0, 0.01),
        yaw_deadband_deg=1.0,
    )


def _sanitize_np_array(array: np.ndarray, clamp_abs: Optional[float] = None) -> np.ndarray:
    result = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None and clamp_abs > 0.0:
        result = np.clip(result, -float(clamp_abs), float(clamp_abs))
    return result.astype(np.float32, copy=False)


def _as_action_batch(actions: np.ndarray, action_dim: int) -> tuple[np.ndarray, bool]:
    actions_np = _sanitize_np_array(actions)
    was_1d = actions_np.ndim == 1
    if was_1d:
        actions_np = actions_np.reshape(1, -1)
    elif actions_np.ndim > 2:
        actions_np = actions_np.reshape(actions_np.shape[0], -1)

    if actions_np.shape[1] < action_dim:
        padded = np.zeros((actions_np.shape[0], action_dim), dtype=np.float32)
        padded[:, :actions_np.shape[1]] = actions_np
        actions_np = padded
    elif actions_np.shape[1] > action_dim:
        actions_np = actions_np[:, :action_dim]

    return actions_np.astype(np.float32, copy=False), was_1d


def _fit_env_action_dim(actions: np.ndarray, expected_dim: int) -> np.ndarray:
    actions_np = _sanitize_np_array(actions, clamp_abs=1.0)
    if actions_np.ndim == 1:
        if actions_np.shape[0] < expected_dim:
            actions_np = np.pad(actions_np, (0, expected_dim - actions_np.shape[0]), constant_values=0.0)
        elif actions_np.shape[0] > expected_dim:
            actions_np = actions_np[:expected_dim]
        return actions_np.astype(np.float32, copy=False)

    if actions_np.shape[1] < expected_dim:
        actions_np = np.pad(actions_np, ((0, 0), (0, expected_dim - actions_np.shape[1])), constant_values=0.0)
    elif actions_np.shape[1] > expected_dim:
        actions_np = actions_np[:, :expected_dim]
    return actions_np.astype(np.float32, copy=False)


def _normalize_obs_array(
    obs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
    default_num_envs: int = 1,
) -> np.ndarray:
    if obs is None:
        return np.zeros((default_num_envs, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)

    if isinstance(obs, dict):
        if "state" in obs:
            obs = obs["state"]
        elif "observation" in obs:
            obs = obs["observation"]
        elif "obs" in obs:
            obs = obs["obs"]
        else:
            obs = next(iter(obs.values()))

    if isinstance(obs, (list, tuple)):
        vector_candidates = []
        for item in obs:
            item_array = np.asarray(item, dtype=np.float32)
            if item_array.ndim >= 1 and item_array.shape[-1] >= 14:
                vector_candidates.append(item_array)
        obs = vector_candidates[0] if vector_candidates else obs[0]

    obs_array = np.asarray(obs, dtype=np.float32)
    if obs_array.ndim == 1:
        obs_array = obs_array.reshape(1, -1)
    elif obs_array.ndim > 2:
        obs_array = np.squeeze(obs_array)
        if obs_array.ndim == 1:
            obs_array = obs_array.reshape(1, -1)
        elif obs_array.ndim > 2 and obs_array.shape[-1] >= 14:
            obs_array = obs_array.reshape(-1, obs_array.shape[-1])

    if obs_array.ndim != 2:
        obs_array = np.zeros((default_num_envs, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)
    return obs_array.astype(np.float32, copy=False)


@dataclass
class HierarchyChaseTrainingConfig(PPOTrainingConfig):
    """PPO high-level local target policy + fixed low-level position PID."""

    learning_rate: Union[float, Callable[[float], float]] = 3e-4
    batch_size: int = 1024
    n_steps: int = 512
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.005
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    target_body_delta_limits: Tuple[float, float, float] = (1.5, 0.5, 1.5)
    policy_action_dim: int = HIERARCHY_POLICY_ACTION_DIM
    yaw_error_limit_deg: float = 90.0
    low_level_pid: TraditionalPositionPIDTrainingConfig = field(default_factory=_legacy_low_level_pid_config)
    policy_net_arch: Dict[str, list[int]] = field(default_factory=lambda: {"pi": [256, 256], "vf": [256, 256]})


class HierarchyChasePolicy(ActorCriticPolicy):
    """Policy action is a local subgoal (and optionally yaw error), not 8D thrust."""

    def forward(self, obs: torch.Tensor, deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            latent_pi, latent_vf = self.mlp_extractor(features)
        else:
            pi_features, vf_features = features
            latent_pi = self.mlp_extractor.forward_actor(pi_features)
            latent_vf = self.mlp_extractor.forward_critic(vf_features)

        values = self.value_net(latent_vf)
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        return actions, values, log_prob


class HierarchyChaseModel(PPORewrite):
    """High-level PPO that emits body-frame local target deltas."""

    def __init__(
        self,
        policy,
        env: VecEnv,
        learning_rate: Union[float, Callable] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Callable] = 0.2,
        clip_range_vf: Optional[Union[float, Callable]] = None,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        target_kl: Optional[float] = None,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[torch.device, str] = "auto",
        _init_setup_model: bool = True,
        target_body_delta_limits: Sequence[float] = (1.5, 0.5, 1.5),
        policy_action_dim: int = HIERARCHY_POLICY_ACTION_DIM,
        yaw_error_limit_deg: float = 90.0,
        low_level_pid_config: Optional[TraditionalPositionPIDTrainingConfig] = None,
        action_dim: Optional[int] = None,
    ) -> None:
        self.target_body_delta_limits = tuple(float(v) for v in target_body_delta_limits)
        if len(self.target_body_delta_limits) != 3:
            raise ValueError(
                "target_body_delta_limits must have 3 elements, "
                f"got {len(self.target_body_delta_limits)}"
            )
        self.policy_action_dim = int(policy_action_dim)
        if self.policy_action_dim not in (HIERARCHY_POLICY_ACTION_DIM, HIERARCHY_POSE_YAW_POLICY_ACTION_DIM):
            raise ValueError(
                "policy_action_dim must be 3 ([forward, up, left]) or "
                "4 ([forward, up, left, yaw]), "
                f"got {self.policy_action_dim}"
            )
        self.yaw_error_limit_deg = float(max(yaw_error_limit_deg, 0.0))
        self.low_level_pid_config = low_level_pid_config or _legacy_low_level_pid_config()
        self.actual_action_dim = action_dim or (
            int(env.action_space.shape[0])
            if env is not None and hasattr(env, "action_space") and getattr(env.action_space, "shape", None)
            else THRUSTER_ACTION_DIM
        )
        self._high_level_target_debug_channel = None
        self._high_level_target_debug_step = 0
        self.low_level_controller_model = self._build_low_level_model(str(device))
        policy_for_sb3 = HierarchyChasePolicy if policy == "MlpPolicy" else policy

        super().__init__(
            policy=policy_for_sb3,
            env=env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            target_kl=target_kl,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=False,
            policy_action_dim=self.policy_action_dim,
            action_postprocess_fn=self._policy_actions_to_env_actions,
        )

        if _init_setup_model:
            self._setup_model()
            self._sync_low_level_device()

    def _build_low_level_model(self, device: str) -> TraditionalPositionPIDModel:
        cfg = self.low_level_pid_config
        controller = PIDController(
            action_dim=THRUSTER_ACTION_DIM,
            device=device,
            dt=cfg.dt,
            integral_decay=cfg.integral_decay,
            output_range=cfg.output_range,
            yaw_deadband_deg=cfg.yaw_deadband_deg,
            yaw_output_limit=cfg.yaw_output_limit,
            yaw_integral_state_limit=cfg.yaw_integral_state_limit,
            surge_output_limit=cfg.surge_output_limit,
            sway_output_limit=cfg.sway_output_limit,
            integral_state_limit=cfg.integral_state_limit,
            derivative_filter_alpha=cfg.derivative_filter_alpha,
            output_slew_rate_limit=cfg.output_slew_rate_limit,
            allocator_control_linear_range=cfg.allocator_control_linear_range,
            allocator_control_axis_ranges=cfg.allocator_control_axis_ranges or None,
            allocator_allocation_mode=cfg.allocator_allocation_mode,
            debug_print_interval=cfg.debug_print_interval if cfg.enable_logging else 0,
        )
        return TraditionalPositionPIDModel(
            controller=controller,
            pid_params=cfg.pid_params,
            yaw_pid_params=cfg.yaw_pid_params,
            enable_yaw_control=cfg.enable_yaw_control,
            action_dim=THRUSTER_ACTION_DIM,
            enable_logging=cfg.enable_logging,
        )

    def _sync_low_level_device(self) -> None:
        if not hasattr(self, "low_level_controller_model") or self.low_level_controller_model is None:
            self.low_level_controller_model = self._build_low_level_model(str(self.device))
            return
        self.low_level_controller_model.controller.to(self.device)
        self.low_level_controller_model.controller.device = str(self.device)

    def _build_low_level_pid_inputs(
        self,
        policy_actions: np.ndarray,
        obs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]],
    ) -> np.ndarray:
        obs_array = _normalize_obs_array(
            obs if obs is not None else np.zeros((1, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)
        )
        batch_size = obs_array.shape[0]
        actions_np, _ = _as_action_batch(policy_actions, self.policy_action_dim)
        if actions_np.shape[0] != batch_size:
            if actions_np.shape[0] == 1 and batch_size > 1:
                actions_np = np.repeat(actions_np, batch_size, axis=0)
            else:
                raise ValueError(f"Policy action batch {actions_np.shape[0]} does not match obs batch {batch_size}")

        actions_np = np.clip(actions_np, -1.0, 1.0)
        limits = np.asarray(self.target_body_delta_limits, dtype=np.float32).reshape(1, 3)
        local_target_delta = actions_np[:, :3] * limits

        return local_target_delta

    def _build_low_level_yaw_error_deg(self, policy_actions: np.ndarray, batch_size: int) -> np.ndarray:
        if self.policy_action_dim < HIERARCHY_POSE_YAW_POLICY_ACTION_DIM:
            return np.zeros(batch_size, dtype=np.float32)

        actions_np, _ = _as_action_batch(policy_actions, self.policy_action_dim)
        if actions_np.shape[0] != batch_size:
            if actions_np.shape[0] == 1 and batch_size > 1:
                actions_np = np.repeat(actions_np, batch_size, axis=0)
            else:
                raise ValueError(f"Policy action batch {actions_np.shape[0]} does not match obs batch {batch_size}")
        return np.clip(actions_np[:, 3], -1.0, 1.0).astype(np.float32) * self.yaw_error_limit_deg

    def _policy_actions_to_env_actions(
        self,
        policy_actions: np.ndarray,
        obs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
        *,
        emit_debug_target: bool = True,
    ) -> np.ndarray:
        local_target_delta = self._build_low_level_pid_inputs(policy_actions, obs)
        yaw_error_deg = self._build_low_level_yaw_error_deg(policy_actions, local_target_delta.shape[0])
        env_actions, _ = self.low_level_controller_model.predict_from_body_position_error(
            local_target_delta,
            yaw_error_deg=yaw_error_deg,
            deterministic=True,
        )
        if emit_debug_target:
            zero_world_debug = np.zeros_like(local_target_delta, dtype=np.float32)
            self._maybe_send_high_level_target_debug(
                local_target_delta=local_target_delta,
                target_pos=zero_world_debug,
                current_pos=zero_world_debug,
            )
        return _fit_env_action_dim(env_actions, self.actual_action_dim)

    def _resolve_high_level_target_debug_channel(self):
        if self._high_level_target_debug_channel is not None:
            return self._high_level_target_debug_channel

        self._high_level_target_debug_channel = resolve_high_level_target_debug_channel(getattr(self, "env", None))
        return self._high_level_target_debug_channel

    def _maybe_send_high_level_target_debug(
        self,
        *,
        local_target_delta: np.ndarray,
        target_pos: np.ndarray,
        current_pos: np.ndarray,
    ) -> None:
        local_target_delta = np.asarray(local_target_delta, dtype=np.float32)
        target_pos = np.asarray(target_pos, dtype=np.float32)
        current_pos = np.asarray(current_pos, dtype=np.float32)
        if local_target_delta.ndim == 1:
            local_target_delta = local_target_delta.reshape(1, -1)
        if target_pos.ndim == 1:
            target_pos = target_pos.reshape(1, -1)
        if current_pos.ndim == 1:
            current_pos = current_pos.reshape(1, -1)
        if local_target_delta.shape[0] == 0 or target_pos.shape[0] == 0 or current_pos.shape[0] == 0:
            return

        env = getattr(self, "env", None)
        subproc_sender = getattr(env, "send_high_level_target_debug", None)
        if callable(subproc_sender):
            subproc_sender(
                step_index=self._high_level_target_debug_step,
                local_target_delta=local_target_delta,
                target_world=target_pos,
                current_world=current_pos,
            )
            self._high_level_target_debug_step += local_target_delta.shape[0]
            return

        channel = self._resolve_high_level_target_debug_channel()
        if channel is None:
            return

        channel.send_high_level_target(
            step_index=self._high_level_target_debug_step,
            local_target_delta=local_target_delta[0, :3].tolist(),
            target_world=target_pos[0, :3].tolist(),
            current_world=current_pos[0, :3].tolist(),
        )
        self._high_level_target_debug_step += 1

    def _on_rollout_env_step(
        self,
        new_obs: Any,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        del new_obs, rewards, infos
        if np.any(dones):
            self.low_level_controller_model.controller.reset(np.asarray(dones, dtype=bool))

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        if episode_start is not None and np.any(episode_start):
            self.low_level_controller_model.controller.reset(np.asarray(episode_start, dtype=bool))

        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)

        with torch.no_grad():
            policy_actions, _, _ = self.policy(obs_tensor, deterministic=deterministic)

        policy_actions_np = policy_actions.detach().cpu().numpy()
        policy_actions_np = self._clip_policy_actions(policy_actions_np)
        env_actions = self._policy_actions_to_env_actions(
            policy_actions_np,
            observation,
            emit_debug_target=True,
        )

        if not vectorized_env and env_actions.ndim == 2 and env_actions.shape[0] == 1:
            env_actions = env_actions[0]

        return env_actions, state

    def reset(self) -> None:
        self.low_level_controller_model.controller.reset()
        self._high_level_target_debug_step = 0

    def save(
        self,
        path: Union[str, Path, BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        exclude_set = set(exclude or [])
        exclude_set.add("action_postprocess_fn")
        exclude_set.add("_high_level_target_debug_channel")
        super().save(path, exclude=list(exclude_set), include=include)

    @classmethod
    def load(
        cls,
        path: Union[str, Path, BufferedIOBase],
        env=None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
        if not hasattr(model, "low_level_controller_model") or model.low_level_controller_model is None:
            model.low_level_controller_model = model._build_low_level_model(str(device))
        if not hasattr(model, "_high_level_target_debug_channel"):
            model._high_level_target_debug_channel = None
        if not hasattr(model, "_high_level_target_debug_step"):
            model._high_level_target_debug_step = 0
        model._sync_low_level_device()
        if hasattr(model, "set_action_postprocess_fn"):
            model.set_action_postprocess_fn(model._policy_actions_to_env_actions)
        model._high_level_target_debug_channel = resolve_high_level_target_debug_channel(getattr(model, "env", None))
        return model


@dataclass
class HierarchyChaseConfig(BaseConfig):
    """Dedicated config for hierarchy chase tasks."""

    model_type: str = "HIERARCHY_CHASE"
    env_config: PPOEnvironmentConfig = field(default_factory=PPOEnvironmentConfig)
    training_config: HierarchyChaseTrainingConfig = field(default_factory=HierarchyChaseTrainingConfig)

    def create_model(self, env: VecEnv, args: Any, load_checkpoint: Optional[str] = None):
        train_config: HierarchyChaseTrainingConfig = self.training_config  # type: ignore[assignment]

        model = HierarchyChaseModel(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=train_config.learning_rate,
            n_steps=train_config.n_steps,
            batch_size=train_config.batch_size,
            gamma=train_config.gamma,
            gae_lambda=train_config.gae_lambda,
            n_epochs=train_config.n_epochs,
            clip_range=train_config.clip_range,
            ent_coef=train_config.ent_coef,
            vf_coef=train_config.vf_coef,
            max_grad_norm=train_config.max_grad_norm,
            target_body_delta_limits=train_config.target_body_delta_limits,
            policy_action_dim=train_config.policy_action_dim,
            yaw_error_limit_deg=train_config.yaw_error_limit_deg,
            low_level_pid_config=train_config.low_level_pid,
            policy_kwargs={
                "net_arch": train_config.policy_net_arch,
                "activation_fn": nn.ReLU,
            },
            tensorboard_log=getattr(args, "tensorboard_dir", None) or "./logs",
            device=args.device,
        )

        if load_checkpoint:
            print(f"Loading model from checkpoint: {load_checkpoint}")
            model = HierarchyChaseModel.load(load_checkpoint, env=env, device=args.device)

        return model

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        return HierarchyChaseModel.load(model_path, env=env, device=device)


def get_hierarchy_chase_configs() -> Dict[str, HierarchyChaseConfig]:
    return {
        "hierarchy_chase_1chase1": HierarchyChaseConfig(
            name="hierarchy_chase_1chase1",
            model_type="HIERARCHY_CHASE",
            description=(
                "1Chase1 hierarchy chase: PPO high-level local target policy + "
                "traditional position PID low-level controller"
            ),
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/FinsROV/1Chase1_headless/1Chase1.x86_64",
                num_envs=32,
                eval_num_envs=4,
                env_base_port=15125,
                timeout_wait=360,
                no_graphics=True,
                environment_parameters={
                    "finsim_1chase1_observation_mode": float(HIERARCHY_ACTOR_OBS_DIM),
                },
                unity_additional_args=["-fins-1chase1-mode", "direct14"],
            ),
            training_config=HierarchyChaseTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                total_timesteps=5_000_000,
                checkpoint_freq=100_000,
                eval_freq=20_000,
                target_body_delta_limits=(1.5, 0.5, 1.5),
                low_level_pid=_legacy_low_level_pid_config(),
            ),
        ),
        "hierarchy_chase_physical_wrench_allocator_1chase1": HierarchyChaseConfig(
            name="hierarchy_chase_physical_wrench_allocator_1chase1",
            model_type="HIERARCHY_CHASE",
            description=(
                "1Chase1 hierarchy chase: PPO high-level local target policy + "
                "traditional position PID + bounded physical wrench allocator"
            ),
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/FinsROV/1Chase1_headless/1Chase1.x86_64",
                num_envs=32,
                eval_num_envs=4,
                env_base_port=15125,
                timeout_wait=360,
                no_graphics=True,
                environment_parameters={
                    "finsim_1chase1_observation_mode": float(HIERARCHY_ACTOR_OBS_DIM),
                },
                unity_additional_args=["-fins-1chase1-mode", "direct14"],
            ),
            training_config=HierarchyChaseTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                total_timesteps=5_000_000,
                checkpoint_freq=100_000,
                eval_freq=20_000,
                target_body_delta_limits=(1.5, 0.5, 1.5),
                low_level_pid=_physical_wrench_low_level_pid_config(),
            ),
        ),
        "hierarchy_chase_subgoal_yaw_physical_wrench_allocator_1chase1": HierarchyChaseConfig(
            name="hierarchy_chase_subgoal_yaw_physical_wrench_allocator_1chase1",
            model_type="HIERARCHY_CHASE",
            description=(
                "1Chase1 hierarchy chase: PPO local subgoal plus yaw error, fixed "
                "position/yaw PID, and bounded physical wrench allocator"
            ),
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/FinsROV/1Chase1_headless/1Chase1.x86_64",
                num_envs=32,
                eval_num_envs=4,
                env_base_port=15125,
                timeout_wait=360,
                no_graphics=True,
                environment_parameters={
                    "finsim_1chase1_observation_mode": float(HIERARCHY_ACTOR_OBS_DIM),
                },
                unity_additional_args=["-fins-1chase1-mode", "direct14"],
            ),
            training_config=HierarchyChaseTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                total_timesteps=5_000_000,
                checkpoint_freq=100_000,
                eval_freq=20_000,
                target_body_delta_limits=(1.5, 0.5, 1.5),
                policy_action_dim=HIERARCHY_POSE_YAW_POLICY_ACTION_DIM,
                yaw_error_limit_deg=90.0,
                low_level_pid=_physical_wrench_pose_yaw_low_level_pid_config(),
            ),
        ),
    }


__all__ = [
    "HierarchyChaseConfig",
    "HierarchyChaseModel",
    "HierarchyChaseTrainingConfig",
    "get_hierarchy_chase_configs",
]
