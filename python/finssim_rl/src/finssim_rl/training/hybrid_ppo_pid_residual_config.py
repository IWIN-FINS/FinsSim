"""
混合 PPO+PID residual 配置

当前支持两种方法：
- residual: PPO 输出 9 维归一化残差动作，通过 log-gain 映射到 baseline PID 上
- direct: PPO 输出 9 维归一化 PID 坐标，直接映射到 PID 参数范围
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv

from finssim_rl.models.hybrid_ppo_pid_residual import (
    DEFAULT_DIRECT_PID_MAX,
    DEFAULT_DIRECT_PID_MIN,
    DEFAULT_RESIDUAL_LOG_SCALE,
    HybridPPOWithPIDResidual,
)
from finssim_rl.models.hybrid_ppo_pid_v3 import DEFAULT_PID_PARAMS
from finssim_rl.models.ppo_control import PPOEnvironmentConfig
from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig
from finssim_rl.training.utils import linear_schedule


@dataclass
class HybridResidualTrainingConfig(BaseTrainingConfig):
    """residual/direct PID PPO 训练配置。"""

    learning_rate: Union[float, Callable[[float], float]] = 4e-4
    batch_size: int = 64
    n_steps: int = 512
    n_epochs: int = 10
    clip_range: float = 0.2
    ent_coef: float = 0.0
    gamma: float = 0.99
    gae_lambda: float = 0.95
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: Optional[float] = None
    total_timesteps: int = 300_000
    checkpoint_freq: int = 30_000
    eval_freq: int = 10_000

    freeze_pid_updates: bool = False
    stage1_ratio: float = 0.3
    stage2_ratio: float = 0.3

    baseline_pid_params: Tuple[float, float, float, float, float, float, float, float, float] = tuple(
        float(v) for v in DEFAULT_PID_PARAMS
    )
    residual_log_scale: Tuple[float, float, float, float, float, float, float, float, float] = tuple(
        float(v) for v in DEFAULT_RESIDUAL_LOG_SCALE
    )
    pid_mapping_mode: str = "residual"
    direct_pid_min_params: Tuple[float, float, float, float, float, float, float, float, float] = tuple(
        float(v) for v in DEFAULT_DIRECT_PID_MIN
    )
    direct_pid_max_params: Tuple[float, float, float, float, float, float, float, float, float] = tuple(
        float(v) for v in DEFAULT_DIRECT_PID_MAX
    )
    residual_log_l2_coef: float = 0.0
    residual_boundary_loss_coef: float = 0.0
    residual_boundary_threshold: float = 0.85
    random_initial_residual: bool = False
    random_initial_residual_seed: Optional[int] = None
    initial_residual_log_std: float = -3.0
    initial_residual_reset_weights: bool = True

    policy_net_arch: Dict = field(default_factory=lambda: {"pi": [256, 256], "vf": [256, 256]})


@dataclass
class HybridResidualConfig(BaseConfig):
    """residual/direct PID PPO 配置。"""

    model_type: str = "HYBRID_PPO_PID_RESIDUAL"
    env_config: BaseEnvironmentConfig = field(default_factory=PPOEnvironmentConfig)
    training_config: BaseTrainingConfig = field(default_factory=HybridResidualTrainingConfig)

    def create_model(self, env: VecEnv, args: Any, load_checkpoint: Optional[str] = None):
        train_config: HybridResidualTrainingConfig = self.training_config  # type: ignore

        model = HybridPPOWithPIDResidual(
            policy="MlpPolicy",
            env=env,
            verbose=2,
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
            target_kl=train_config.target_kl,
            stage1_ratio=train_config.stage1_ratio,
            stage2_ratio=train_config.stage2_ratio,
            freeze_pid_updates=train_config.freeze_pid_updates,
            baseline_pid_params=train_config.baseline_pid_params,
            residual_log_scale=train_config.residual_log_scale,
            pid_mapping_mode=train_config.pid_mapping_mode,
            direct_pid_min_params=train_config.direct_pid_min_params,
            direct_pid_max_params=train_config.direct_pid_max_params,
            residual_log_l2_coef=train_config.residual_log_l2_coef,
            residual_boundary_loss_coef=train_config.residual_boundary_loss_coef,
            residual_boundary_threshold=train_config.residual_boundary_threshold,
            random_initial_residual=train_config.random_initial_residual,
            random_initial_residual_seed=train_config.random_initial_residual_seed,
            initial_residual_log_std=train_config.initial_residual_log_std,
            initial_residual_reset_weights=train_config.initial_residual_reset_weights,
            policy_kwargs={
                "net_arch": train_config.policy_net_arch,
                "activation_fn": nn.ReLU,
            },
            tensorboard_log=getattr(args, "tensorboard_dir", None) or "logs",
            device=args.device,
        )

        if load_checkpoint:
            model = HybridPPOWithPIDResidual.load(load_checkpoint, env=env, device=args.device)

        return model

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        return HybridPPOWithPIDResidual.load(model_path, env=env, device=device)


def get_hybrid_control_configs_residual() -> Dict[str, HybridResidualConfig]:
    """获取所有 residual PPO+PID 配置。"""
    return {
        "hybrid_pid_pose_residual": HybridResidualConfig(
            name="hybrid_pid_pose_residual",
            model_type="HYBRID_PPO_PID_RESIDUAL",
            description="Hybrid PPO+PID residual - stabilized axis-aware log-gain box around baseline PID",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=HybridResidualTrainingConfig(
                learning_rate=linear_schedule(2e-4),
                batch_size=128,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.03,
                total_timesteps=3_000_000,
                checkpoint_freq=30_000,
                eval_freq=5_000,
                stage1_ratio=0.0,
                stage2_ratio=0.0,
                baseline_pid_params=tuple(float(v) for v in DEFAULT_PID_PARAMS),
                # 保留按轴调节，但收窄搜索盒子，减少推进器饱和时的边界吸附。
                # 顺序：depth[Kp,Ki,Kd], surge[Kp,Ki,Kd], sway[Kp,Ki,Kd]
                residual_log_scale=(0.8, 1.0, 0.8, 0.8, 1.0, 0.8, 0.8, 1.0, 0.8),
                residual_log_l2_coef=0.01,
                residual_boundary_loss_coef=0.05,
                residual_boundary_threshold=0.8,
            ),
        ),
        "hybrid_pid_pose_residual_random": HybridResidualConfig(
            name="hybrid_pid_pose_residual_random",
            model_type="HYBRID_PPO_PID_RESIDUAL",
            description="Hybrid PPO+PID direct - random initial PID inside configured bounds",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=HybridResidualTrainingConfig(
                learning_rate=linear_schedule(2e-4),
                batch_size=128,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                target_kl=0.03,
                total_timesteps=3_000_000,
                checkpoint_freq=30_000,
                eval_freq=5_000,
                stage1_ratio=0.0,
                stage2_ratio=0.0,
                pid_mapping_mode="direct",
                direct_pid_min_params=(0.2, 0.001, 0.001, 0.2, 0.001, 0.001, 0.2, 0.001, 0.001),
                direct_pid_max_params=(20.0, 2.0, 5.0, 20.0, 2.0, 5.0, 20.0, 2.0, 5.0),
                residual_log_l2_coef=0.0,
                residual_boundary_loss_coef=0.0,
                residual_boundary_threshold=0.8,
                random_initial_residual=True,
                random_initial_residual_seed=None,
                initial_residual_log_std=-3.0,
                initial_residual_reset_weights=True,
            ),
        ),
        "hybrid_pid_pose_residual_frozen": HybridResidualConfig(
            name="hybrid_pid_pose_residual_frozen",
            model_type="HYBRID_PPO_PID_RESIDUAL",
            description="Hybrid PPO+PID residual - baseline PID only, PPO head frozen",
            env_config=PPOEnvironmentConfig(
                time_scale=3.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=HybridResidualTrainingConfig(
                learning_rate=3e-4,
                batch_size=64,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,
                vf_coef=0.5,
                max_grad_norm=0.5,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=10_000,
                freeze_pid_updates=True,
                stage1_ratio=0.0,
                stage2_ratio=0.0,
            ),
        ),
    }


__all__ = ["get_hybrid_control_configs_residual"]
