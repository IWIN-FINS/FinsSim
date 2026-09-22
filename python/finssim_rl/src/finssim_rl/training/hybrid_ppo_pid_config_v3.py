"""
混合 PPO+PID v3 配置

当前 v3 方法：
- PPO 输出 9 维归一化极点参数，表示 depth/surge/sway 3组 [τ1, τ2, τ3]
- 极点范围由模型中的 TAU_MIN/TAU_MAX 约束
- PID 参数由极点配置公式生成，不再配置手动 PID 参数缩放范围
- 训练阶段只决定哪些极点参数接收梯度
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

import torch.nn as nn
from stable_baselines3.common.vec_env import VecEnv

from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig
from finssim_rl.models.hybrid_ppo_pid_v3 import HybridPPOWithPIDv3
from finssim_rl.models.ppo_control import PPOEnvironmentConfig
from finssim_rl.training.utils import linear_schedule


@dataclass
class HybridTrainingConfigV3(BaseTrainingConfig):
    """第三代混合PPO+PID训练配置"""

    # PPO参数
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
    total_timesteps: int = 500_000
    checkpoint_freq: int = 50_000
    eval_freq: int = 10_000
    stage1_ratio=0.0,
    stage2_ratio=0.0,

    # 是否冻结PID参数更新（保持 DEFAULT_PID_PARAMS）
    freeze_pid_updates: bool = False
    random_initial_tau: bool = False
    random_initial_tau_log_std: float = -0.5
    random_initial_tau_seed: Optional[int] = None
    random_initial_tau_reset_weights: bool = True

    # 分阶段训练占比：
    # stage1=depth_only, stage2=outer_only, 其余为all
    stage1_ratio: float = 0.3
    stage2_ratio: float = 0.3

    # 网络配置
    policy_net_arch: Dict = field(default_factory=lambda: {"pi": [256, 256], "vf": [256, 256]})


@dataclass
class HybridConfigV3(BaseConfig):
    """第三代混合PPO+PID配置"""
    
    model_type: str = "HYBRID_PPO_PID_V3"
    env_config: BaseEnvironmentConfig = field(default_factory=PPOEnvironmentConfig)
    training_config: BaseTrainingConfig = field(default_factory=HybridTrainingConfigV3)

    def create_model(self, env: VecEnv, args: Any, load_checkpoint: Optional[str] = None):
        """创建第三代混合模型"""
        train_config: HybridTrainingConfigV3 = self.training_config  # type: ignore

        model = HybridPPOWithPIDv3(
            policy="MlpPolicy",
            env=env,
            verbose=2,
            learning_rate=train_config.learning_rate,

            # PPO参数
            n_steps=train_config.n_steps,
            batch_size=train_config.batch_size,
            gamma=train_config.gamma,
            gae_lambda=train_config.gae_lambda,
            n_epochs=train_config.n_epochs,
            clip_range=train_config.clip_range,
            ent_coef=train_config.ent_coef,
            vf_coef=train_config.vf_coef,
            max_grad_norm=train_config.max_grad_norm,

            # 分阶段训练控制
            stage1_ratio=train_config.stage1_ratio,
            stage2_ratio=train_config.stage2_ratio,
            freeze_pid_updates=getattr(train_config, "freeze_pid_updates", False),
            random_initial_tau=getattr(train_config, "random_initial_tau", False),
            random_initial_tau_log_std=getattr(train_config, "random_initial_tau_log_std", -0.5),
            random_initial_tau_seed=getattr(train_config, "random_initial_tau_seed", None),
            random_initial_tau_reset_weights=getattr(train_config, "random_initial_tau_reset_weights", True),

            # 网络配置
            policy_kwargs={
                "net_arch": train_config.policy_net_arch,
                "activation_fn": nn.ReLU,
            },

            tensorboard_log=getattr(args, "tensorboard_dir", None) or "logs",
            device=args.device,
        )

        if load_checkpoint:
            model = HybridPPOWithPIDv3.load(load_checkpoint, env=env, device=args.device)

        return model

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        """加载第三代混合模型用于评估。"""
        return HybridPPOWithPIDv3.load(model_path, env=env, device=device)


def get_hybrid_control_configs_v3() -> Dict[str, HybridConfigV3]:
    """获取所有第三代混合PPO+PID配置"""
    return {
        "hybrid_pid_pose_v3": HybridConfigV3(
            name="hybrid_pid_pose_v3",
            model_type="HYBRID_PPO_PID_V3",
            description="Hybrid PPO+PID v3 - 9D pole output for x/y/z position control",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=HybridTrainingConfigV3(
                learning_rate=linear_schedule(2e-4),  # 降低初始学习率（从 3e-4）
                batch_size=128,  # 减小 batch_size（从 1024，确保 <= n_steps）
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,  # 增大 clip_range（从 0.1 到标准值）
                ent_coef=0.005,
                vf_coef=0.5,
                max_grad_norm=0.5,
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
                stage1_ratio=0.0,
                stage2_ratio=0.0,
            ),
        ),
        "hybrid_pid_pose_v3_random_tau": HybridConfigV3(
            name="hybrid_pid_pose_v3_random_tau",
            model_type="HYBRID_PPO_PID_V3",
            description="Hybrid PPO+PID v3 - random initial τ policy for x/y/z position control",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=HybridTrainingConfigV3(
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
                total_timesteps=5_000_000,
                checkpoint_freq=50_000,
                eval_freq=5_000,
                stage1_ratio=0.0,
                stage2_ratio=0.0,
                random_initial_tau=True,
                random_initial_tau_log_std=-0.5,
                random_initial_tau_seed=None,
                random_initial_tau_reset_weights=True,
            ),
        ),
        "hybrid_pid_aggressive_v3": HybridConfigV3(
            name="hybrid_pid_aggressive_v3",
            model_type="HYBRID_PPO_PID_V3",
            description="Hybrid PPO+PID v3 - aggressive PPO hyperparameters",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/RLControl.x86_64",
            ),
            training_config=HybridTrainingConfigV3(
                learning_rate=6e-4,  # 更高学习率
                batch_size=64,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                gae_lambda=0.95,
                clip_range=0.2,
                ent_coef=0.01,  # 更多探索
                vf_coef=0.5,
                max_grad_norm=0.5,
                total_timesteps=1_000_000,  # 更多训练
                checkpoint_freq=50_000,
                eval_freq=10_000,
            ),
        ),

        "hybrid_pid_frozen_v3": HybridConfigV3(
            name="hybrid_pid_frozen_v3",
            model_type="HYBRID_PPO_PID_V3",
            # 历史配置名保留；当前含义是跳过分阶段预热，直接进入 all 阶段。
            description="Hybrid PPO+PID v3 - start directly from all-stage training",
            env_config=PPOEnvironmentConfig(
                time_scale=3.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=HybridTrainingConfigV3(
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
                # 禁止PID参数更新，保留默认PID
                freeze_pid_updates=True,
                # 立即进入 all 阶段，同时训练 depth/yaw 极点参数。
                stage1_ratio=0.0,
                stage2_ratio=0.0,
            ),
        ),

    }

__all__ = ["get_hybrid_control_configs_v3"]
