"""
PPO Control v2

与 ppo_control.py 的区别：
- PPO 策略只学习 6 维虚拟控制量：
  [surge, sway, heave, roll, pitch, yaw]
- 虚拟控制量会先映射为推力分配器使用的 6 维合力/合矩：
  [Fx, Fy, Fz, Mx, My, Mz]
- 再通过 ThrustAllocator 转换成 Unity 环境需要的 8 个推进器动作。

这样 RL 学的是“期望合力/合矩”，底层 8 推进器分配固定由推力分配矩阵完成。

训练语句：
export DISPLAY=:0
python -m scripts.train --config ppo_control_v2_for_pose --exp-name ppo_v2_pose --num-envs 8 --overwrite
export DISPLAY=:0
python -m scripts.train --config ppo_control_v2_for_velocity --exp-name ppo_v2_vel --num-envs 8 --overwrite

"""
from dataclasses import dataclass, field
from io import BufferedIOBase
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.vec_env import VecEnv

from finssim_rl.models.ppo_control import PPOEnvironmentConfig, PPOTrainingConfig
from finssim_rl.models.ppo_rewrite import PPORewrite
from finssim_rl.models.thrust_allocator import ThrustAllocator
from finssim_rl.training.config import BaseConfig
from finssim_rl.training.utils import (
    apply_continuation_learning_rate_schedule,
    linear_schedule,
)


VIRTUAL_CONTROL_DIM = 6
THRUSTER_ACTION_DIM = 8

# PPO 输出顺序，便于人理解和日志分析。
VIRTUAL_CONTROL_NAMES: Tuple[str, ...] = (
    "surge",
    "sway",
    "heave",
    "roll",
    "pitch",
    "yaw",
)

POLICY_AXIS_ORDER_LEGACY = "legacy_surge_sway_heave"
POLICY_AXIS_ORDER_ALLOCATOR_BODY = "allocator_body"

# PPO 输出是 [-1, 1]，这里把它缩放到 ThrustAllocator 的控制域。
# 顺序与 VIRTUAL_CONTROL_NAMES 一致。
DEFAULT_VIRTUAL_CONTROL_LIMITS: Tuple[float, ...] = (
    20.0,  # surge -> Fx
    20.0,  # sway  -> Fz
    20.0,  # heave -> Fy
    0.8,   # roll  -> Mx
    0.8,   # pitch -> Mz
    0.8,   # yaw   -> My
)


def _sanitize_np_array(array: np.ndarray, clamp_abs: Optional[float] = None) -> np.ndarray:
    """把 NaN/Inf 清成有限值，避免异常动作污染环境。"""
    result = np.nan_to_num(np.asarray(array, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if clamp_abs is not None and clamp_abs > 0.0:
        result = np.clip(result, -float(clamp_abs), float(clamp_abs))
    return result.astype(np.float32, copy=False)


def _as_action_batch(actions: np.ndarray, action_dim: int) -> Tuple[np.ndarray, bool]:
    """整理动作为 (batch, action_dim)，返回是否原本为单条动作。"""
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


def _normalize_control_limits(control_limits: Sequence[float]) -> np.ndarray:
    limits_np = _sanitize_np_array(np.asarray(tuple(control_limits), dtype=np.float32), clamp_abs=1e6)
    if limits_np.ndim != 1:
        limits_np = limits_np.reshape(-1)
    if limits_np.shape[0] < VIRTUAL_CONTROL_DIM:
        padded = np.asarray(DEFAULT_VIRTUAL_CONTROL_LIMITS, dtype=np.float32).copy()
        padded[:limits_np.shape[0]] = limits_np
        limits_np = padded
    elif limits_np.shape[0] > VIRTUAL_CONTROL_DIM:
        limits_np = limits_np[:VIRTUAL_CONTROL_DIM]
    return np.maximum(limits_np, 0.0).astype(np.float32, copy=False)


def normalize_policy_axis_order(policy_axis_order: str) -> str:
    normalized = str(policy_axis_order).strip().lower().replace("-", "_")
    if normalized in {"legacy", "surge_sway_heave", POLICY_AXIS_ORDER_LEGACY}:
        return POLICY_AXIS_ORDER_LEGACY
    if normalized in {"physical_allocator_body", POLICY_AXIS_ORDER_ALLOCATOR_BODY}:
        return POLICY_AXIS_ORDER_ALLOCATOR_BODY
    raise ValueError(
        "policy_axis_order must be `legacy_surge_sway_heave` or `allocator_body`"
    )


def policy_actions_to_allocator_tau(
    normalized_virtual_controls: np.ndarray,
    control_limits: Sequence[float] = DEFAULT_VIRTUAL_CONTROL_LIMITS,
    policy_axis_order: str = POLICY_AXIS_ORDER_LEGACY,
) -> np.ndarray:
    """
    将 PPO 的 6D 归一化虚拟控制量转换为 ThrustAllocator 的 tau。

    ``legacy_surge_sway_heave`` policy order is
    ``[surge, sway, heave, roll, pitch, yaw]``. ``allocator_body`` is the
    T2 contract and is already ``[Fx, Fy, Fz, Mx, My, Mz]``.
    """
    actions_np, _ = _as_action_batch(np.asarray(normalized_virtual_controls, dtype=np.float32), VIRTUAL_CONTROL_DIM)
    actions_np = np.clip(actions_np, -1.0, 1.0)
    limits_np = _normalize_control_limits(control_limits).reshape(1, VIRTUAL_CONTROL_DIM)
    virtual = actions_np * limits_np

    if normalize_policy_axis_order(policy_axis_order) == POLICY_AXIS_ORDER_ALLOCATOR_BODY:
        return virtual.astype(np.float32, copy=False)

    tau = np.zeros_like(virtual, dtype=np.float32)
    tau[:, 0] = virtual[:, 0]  # Fx: surge
    tau[:, 1] = virtual[:, 2]  # Fy: heave
    tau[:, 2] = virtual[:, 1]  # Fz: sway
    tau[:, 3] = virtual[:, 3]  # Mx: roll
    tau[:, 4] = virtual[:, 5]  # My: yaw
    tau[:, 5] = virtual[:, 4]  # Mz: pitch
    return tau.astype(np.float32, copy=False)


def virtual_controls_to_allocator_tau(
    normalized_virtual_controls: np.ndarray,
    control_limits: Sequence[float] = DEFAULT_VIRTUAL_CONTROL_LIMITS,
) -> np.ndarray:
    """Backward-compatible legacy virtual-control mapping."""
    return policy_actions_to_allocator_tau(
        normalized_virtual_controls,
        control_limits,
        POLICY_AXIS_ORDER_LEGACY,
    )


def _fit_env_action_dim(actions: np.ndarray, expected_dim: int) -> np.ndarray:
    """对齐环境动作维度，通常 expected_dim=8。"""
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


class PPOVirtualControlPolicy(ActorCriticPolicy):
    """
    6D 虚拟控制策略。

    SB3 默认 ActorCriticPolicy.forward() 会把 action reshape 成 env.action_space
    的 8D 形状；v2 的 policy action 是 6D，所以这里保留标准 PPO 计算流程，
    但不做 8D reshape。
    """

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


class PPOVirtualControlModel(PPORewrite):
    """
    PPO 策略输出 6D 虚拟合力/合矩指令，内部转换为 8D 推进器动作。

    rollout buffer 里保存的是 6D policy action，因此 PPO 优化的也是虚拟控制空间；
    env.step() 收到的始终是 ThrustAllocator 生成的 8D 推进器动作。
    """

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
        virtual_control_limits: Sequence[float] = DEFAULT_VIRTUAL_CONTROL_LIMITS,
        allocator_deadzone_comp: float = 0.0,
        allocator_control_linear_range: float = 200.0,
        allocator_control_axis_ranges: Optional[Sequence[float]] = None,
        allocator_allocation_mode: str = "empirical_thruster_mixer",
        allocator_thruster_force_limit_positive: Optional[Sequence[float]] = None,
        allocator_thruster_force_limit_negative: Optional[Sequence[float]] = None,
        allocator_debug_print_interval: int = 0,
        policy_axis_order: str = POLICY_AXIS_ORDER_LEGACY,
        action_dim: Optional[int] = None,
    ) -> None:
        self.virtual_control_limits = tuple(float(v) for v in _normalize_control_limits(virtual_control_limits))
        self.allocator_deadzone_comp = float(allocator_deadzone_comp)
        self.allocator_control_linear_range = float(allocator_control_linear_range)
        self.allocator_control_axis_ranges = (
            tuple(float(v) for v in allocator_control_axis_ranges)
            if allocator_control_axis_ranges is not None
            else None
        )
        self.allocator_allocation_mode = str(allocator_allocation_mode)
        self.allocator_thruster_force_limit_positive = (
            tuple(float(v) for v in allocator_thruster_force_limit_positive)
            if allocator_thruster_force_limit_positive is not None
            else None
        )
        self.allocator_thruster_force_limit_negative = (
            tuple(float(v) for v in allocator_thruster_force_limit_negative)
            if allocator_thruster_force_limit_negative is not None
            else None
        )
        self.allocator_debug_print_interval = int(allocator_debug_print_interval)
        self.policy_axis_order = normalize_policy_axis_order(policy_axis_order)

        actual_action_dim = action_dim or (
            int(env.action_space.shape[0])
            if env is not None
            and hasattr(env, "action_space")
            and hasattr(env.action_space, "shape")
            and env.action_space.shape
            else THRUSTER_ACTION_DIM
        )
        self.actual_action_dim = actual_action_dim

        self.thrust_allocator = ThrustAllocator(
            device=str(device),
            deadzone_comp=self.allocator_deadzone_comp,
            control_linear_range=self.allocator_control_linear_range,
            control_axis_ranges=self.allocator_control_axis_ranges,
            allocation_mode=self.allocator_allocation_mode,
            physical_wrench_limits=policy_actions_to_allocator_tau(
                np.ones((1, VIRTUAL_CONTROL_DIM), dtype=np.float32),
                self.virtual_control_limits,
                self.policy_axis_order,
            )[0],
            thruster_force_limit_positive=self.allocator_thruster_force_limit_positive,
            thruster_force_limit_negative=self.allocator_thruster_force_limit_negative,
            debug_print_interval=self.allocator_debug_print_interval,
        )

        if policy_kwargs is None:
            policy_kwargs = {}
        else:
            policy_kwargs = dict(policy_kwargs)

        policy_for_sb3 = PPOVirtualControlPolicy if policy == "MlpPolicy" else policy

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
            policy_action_dim=VIRTUAL_CONTROL_DIM,
            action_postprocess_fn=self._policy_actions_to_env_actions,
        )

        if _init_setup_model:
            self._setup_model()
            self._sync_allocator_device()

    def _sync_allocator_device(self) -> None:
        if hasattr(self, "thrust_allocator"):
            self.thrust_allocator = self.thrust_allocator.to(self.device)
            self.thrust_allocator.device = str(self.device)

    def _policy_actions_to_env_actions(
        self,
        policy_actions: np.ndarray,
        obs: Optional[Union[np.ndarray, Dict[str, np.ndarray]]] = None,
    ) -> np.ndarray:
        del obs  # v2 的分配只依赖虚拟控制量，不依赖观测。
        actions_np, was_1d = _as_action_batch(policy_actions, VIRTUAL_CONTROL_DIM)
        tau_np = policy_actions_to_allocator_tau(
            actions_np,
            self.virtual_control_limits,
            self.policy_axis_order,
        )
        tau_tensor = torch.as_tensor(tau_np, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            thrust_tensor = self.thrust_allocator(tau_tensor)
            thrust_np = thrust_tensor.detach().cpu().numpy()

        thrust_np = _fit_env_action_dim(thrust_np, self.actual_action_dim)
        if was_1d and thrust_np.ndim == 2 and thrust_np.shape[0] == 1:
            return thrust_np[0]
        return thrust_np

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        推理时返回环境可直接使用的 8D 推进器动作。

        这里不能直接调用父类 predict，因为父类会按原始 env action_space
        裁剪动作；而本模型的 policy action 是 6D。
        """
        del episode_start
        self.policy.set_training_mode(False)
        obs_tensor, vectorized_env = self.policy.obs_to_tensor(observation)

        with torch.no_grad():
            policy_actions, _, _ = self.policy(obs_tensor, deterministic=deterministic)

        policy_actions_np = policy_actions.detach().cpu().numpy()
        policy_actions_np = self._clip_policy_actions(policy_actions_np)
        env_actions = self._policy_actions_to_env_actions(policy_actions_np, observation)

        if not vectorized_env and env_actions.ndim == 2 and env_actions.shape[0] == 1:
            env_actions = env_actions[0]

        return env_actions, state

    def initialize_actor_from_checkpoint(
        self,
        checkpoint_path: Union[str, Path],
    ) -> tuple[str, ...]:
        """Copy only compatible actor weights from a virtual-control checkpoint.

        Reward reshaping changes the value target even when the observation and
        action contracts are unchanged.  This transfer mode retains the policy
        network, action head and exploration standard deviation, while leaving
        the critic and optimizer freshly initialized for the new reward.
        """
        source = type(self).load(checkpoint_path, device=self.device, force_reset=True)
        if not isinstance(source, PPOVirtualControlModel):
            raise ValueError(
                "Actor-only initialization requires a PPOVirtualControlModel checkpoint, "
                f"got {type(source).__name__}."
            )

        compatibility = (
            ("observation shape", getattr(source.observation_space, "shape", None), getattr(self.observation_space, "shape", None)),
            ("environment action shape", getattr(source.action_space, "shape", None), getattr(self.action_space, "shape", None)),
            ("virtual control limits", source.virtual_control_limits, self.virtual_control_limits),
            ("allocator mode", source.allocator_allocation_mode, self.allocator_allocation_mode),
            ("policy axis order", source.policy_axis_order, self.policy_axis_order),
        )
        mismatches = []
        for name, source_value, target_value in compatibility:
            # SB3 serializes Python tuples through PyTorch as float32 lists.
            # The resulting ~1e-7 representation difference is not a change
            # to the wrench contract, so compare these physical limits with a
            # small numerical tolerance.
            same_value = (
                np.allclose(source_value, target_value, rtol=1e-6, atol=1e-6)
                if name == "virtual control limits"
                else source_value == target_value
            )
            if not same_value:
                mismatches.append(
                    f"{name}: checkpoint={source_value!r}, target={target_value!r}"
                )
        if mismatches:
            raise ValueError(
                "Actor-only checkpoint is not semantically compatible with the target run:\n  - "
                + "\n  - ".join(mismatches)
            )

        target_state = self.policy.state_dict()
        source_state = source.policy.state_dict()
        actor_keys = tuple(
            name
            for name in target_state
            if name.startswith("mlp_extractor.policy_net.")
            or name.startswith("action_net.")
            or name == "log_std"
        )
        if not actor_keys:
            raise RuntimeError("Could not find actor parameters in PPOVirtualControlPolicy.")

        missing_or_mismatched = [
            name
            for name in actor_keys
            if name not in source_state or source_state[name].shape != target_state[name].shape
        ]
        if missing_or_mismatched:
            raise ValueError(
                "Actor-only checkpoint has incompatible policy parameters: "
                + ", ".join(missing_or_mismatched)
            )

        transferred_state = dict(target_state)
        transferred_state.update({name: source_state[name] for name in actor_keys})
        self.policy.load_state_dict(transferred_state, strict=True)
        # This model instance was freshly constructed, so its critic, optimizer,
        # rollout buffer, and timestep counter remain new by design.
        self.num_timesteps = 0
        return actor_keys

    def save(
        self,
        path: Union[str, Path, BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        exclude_set = set(exclude or [])
        # 绑定方法不能稳定序列化；load 后恢复即可。
        exclude_set.add("action_postprocess_fn")
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
        if not hasattr(model, "thrust_allocator"):
            model.thrust_allocator = ThrustAllocator(device=str(device))
        if not hasattr(model, "policy_axis_order"):
            model.policy_axis_order = POLICY_AXIS_ORDER_LEGACY
        else:
            model.policy_axis_order = normalize_policy_axis_order(model.policy_axis_order)
        model._sync_allocator_device()
        if hasattr(model, "set_action_postprocess_fn"):
            model.set_action_postprocess_fn(model._policy_actions_to_env_actions)
        return model


@dataclass
class PPOV2TrainingConfig(PPOTrainingConfig):
    """PPO v2 训练配置：策略动作空间为 6D 虚拟控制量。"""

    learning_rate: Union[float, Callable[[float], float]] = 3e-4
    batch_size: int = 1024
    n_steps: int = 512
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    virtual_control_limits: Tuple[float, float, float, float, float, float] = DEFAULT_VIRTUAL_CONTROL_LIMITS
    allocator_deadzone_comp: float = 0.0
    allocator_control_linear_range: float = 200.0
    allocator_control_axis_ranges: Optional[Tuple[float, float, float, float, float, float]] = None
    allocator_allocation_mode: str = "empirical_thruster_mixer"
    allocator_thruster_force_limit_positive: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
    allocator_thruster_force_limit_negative: Optional[Tuple[float, float, float, float, float, float, float, float]] = None
    allocator_debug_print_interval: int = 0
    policy_axis_order: str = POLICY_AXIS_ORDER_LEGACY
    policy_net_arch: Dict[str, list[int]] = field(default_factory=lambda: {"pi": [256, 256], "vf": [256, 256]})


@dataclass
class PPOV2Config(BaseConfig):
    """PPO v2 配置。"""

    model_type: str = "PPO_VIRTUAL_CONTROL"
    env_config: PPOEnvironmentConfig = field(default_factory=PPOEnvironmentConfig)
    training_config: PPOV2TrainingConfig = field(default_factory=PPOV2TrainingConfig)

    def create_model(self, env: VecEnv, args: Any, load_checkpoint: Optional[str] = None):
        train_config: PPOV2TrainingConfig = self.training_config  # type: ignore[assignment]

        model = PPOVirtualControlModel(
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
            virtual_control_limits=train_config.virtual_control_limits,
            allocator_deadzone_comp=train_config.allocator_deadzone_comp,
            allocator_control_linear_range=train_config.allocator_control_linear_range,
            allocator_control_axis_ranges=train_config.allocator_control_axis_ranges,
            allocator_allocation_mode=train_config.allocator_allocation_mode,
            allocator_thruster_force_limit_positive=train_config.allocator_thruster_force_limit_positive,
            allocator_thruster_force_limit_negative=train_config.allocator_thruster_force_limit_negative,
            allocator_debug_print_interval=train_config.allocator_debug_print_interval,
            policy_axis_order=train_config.policy_axis_order,
            policy_kwargs={
                "net_arch": train_config.policy_net_arch,
                "activation_fn": nn.ReLU,
            },
            seed=getattr(args, "seed", None),
            tensorboard_log=getattr(args, "tensorboard_dir", None) or "./logs",
            device=args.device,
        )

        if load_checkpoint:
            print(f"Loading model from checkpoint: {load_checkpoint}")
            model = PPOVirtualControlModel.load(
                load_checkpoint,
                env=env,
                device=args.device,
                force_reset=True,
                tensorboard_log=getattr(args, "tensorboard_dir", None) or "./logs",
            )

        schedule_steps = int(train_config.total_timesteps)
        if getattr(args, "resume", False):
            schedule_steps = max(schedule_steps - int(getattr(model, "num_timesteps", 0)), 0)
        if schedule_steps > 0:
            apply_continuation_learning_rate_schedule(
                model,
                train_config,
                additional_steps=schedule_steps,
            )

        return model

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        return PPOVirtualControlModel.load(model_path, env=env, device=device)


def get_ppo_control_v2_configs() -> Dict[str, PPOV2Config]:
    """返回 PPO v2 虚拟控制量相关配置。"""
    return {
        "ppo_control_v2_for_pose": PPOV2Config(
            name="ppo_control_v2_for_pose",
            model_type="PPO_VIRTUAL_CONTROL",
            description="PPO v2 for pose control: 6D virtual wrench -> thrust allocator -> 8 thrusters",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=PPOV2TrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.0,
                vf_coef=0.5,
                max_grad_norm=0.5,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
                virtual_control_limits=DEFAULT_VIRTUAL_CONTROL_LIMITS,
                allocator_control_axis_ranges=DEFAULT_VIRTUAL_CONTROL_LIMITS,
            ),
        ),
        "ppo_control_v2_for_velocity": PPOV2Config(
            name="ppo_control_v2_for_velocity",
            model_type="PPO_VIRTUAL_CONTROL",
            description="PPO v2 for velocity control: 6D virtual wrench -> thrust allocator -> 8 thrusters",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForVelocity.x86_64",
                num_envs=64,
            ),
            training_config=PPOV2TrainingConfig(
                learning_rate=1e-4,
                batch_size=1024,
                n_steps=300,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
            ),
        ),
    }


__all__ = [
    "DEFAULT_VIRTUAL_CONTROL_LIMITS",
    "PPOV2Config",
    "PPOV2TrainingConfig",
    "PPOVirtualControlPolicy",
    "POLICY_AXIS_ORDER_ALLOCATOR_BODY",
    "POLICY_AXIS_ORDER_LEGACY",
    "PPOVirtualControlModel",
    "VIRTUAL_CONTROL_DIM",
    "VIRTUAL_CONTROL_NAMES",
    "get_ppo_control_v2_configs",
    "normalize_policy_axis_order",
    "policy_actions_to_allocator_tau",
    "virtual_controls_to_allocator_tau",
]
