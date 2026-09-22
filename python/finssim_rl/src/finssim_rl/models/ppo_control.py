"""
PPO Control 模型配置

定义了 PPO 特定的配置参数和实验配置
"""
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Union

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecEnv

from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig
from finssim_rl.training.utils import linear_schedule

@dataclass
class PPOEnvironmentConfig(BaseEnvironmentConfig):
    """PPO 特定的环境配置"""
    time_scale: float = 10.0  # PPO 推荐时间缩放


@dataclass
class PPOTrainingConfig(BaseTrainingConfig):
    """PPO 特定的训练配置"""
    learning_rate: Union[float, Callable[[float], float]] = 3e-4  # PPO 支持学习率调度
    batch_size: int = 64
    n_steps: int = 2048  # PPO 特定参数：每次更新前收集的步数
    n_epochs: int = 10  # PPO 特定参数：每次更新的 epoch 数
    clip_range: float = 0.2  # PPO 特定参数：策略剪裁范围。此处使用的是算法的默认值
    ent_coef: float = 0.0  # PPO 特定参数：熵系数。此处使用的是算法的默认值


@dataclass
class PPOConfig(BaseConfig):
    """PPO Control 配置"""
    model_type: str = "PPO"
    env_config: PPOEnvironmentConfig = field(default_factory=PPOEnvironmentConfig)
    training_config: PPOTrainingConfig = field(default_factory=PPOTrainingConfig)

    def create_model(self, env: VecEnv, args: Any, load_checkpoint: Optional[str] = None):
        train_config = self.training_config
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=train_config.learning_rate,
            
            n_steps=train_config.n_steps,
            batch_size=train_config.batch_size,
            gamma=train_config.gamma,
            n_epochs=train_config.n_epochs,
            
            clip_range=train_config.clip_range,
            ent_coef=train_config.ent_coef,
            seed=getattr(args, "seed", None),
            
            tensorboard_log=getattr(args, "tensorboard_dir", None) or "./logs",
            device=args.device,
        )

        if load_checkpoint:
            print(f"Loading model from checkpoint: {load_checkpoint}")
            model = type(model).load(
                load_checkpoint,
                env=env,
                device=args.device,
                # A resumed/warm-started run uses freshly spawned Unity scenes.
                # Keep checkpoint timesteps, but never reuse its final observation.
                force_reset=True,
                tensorboard_log=getattr(args, "tensorboard_dir", None) or "./logs",
            )

        return model

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        return PPO.load(model_path, env=env, device=device)


def get_ppo_control_configs() -> Dict[str, PPOConfig]:
    """
    返回所有 PPO Control 相关的配置
    
    Returns:
        配置名称到配置对象的映射 
    """
    return {
        "ppo_chase": PPOConfig(
            name="ppo_chase",
            model_type="PPO",
            description="PPO for chasing target",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/RLControl.x86_64"
            ),
            training_config=PPOTrainingConfig(
                learning_rate=3e-4,
                batch_size=64,
                n_steps=2048,
                n_epochs=10,
                clip_range=0.2,
                gamma=0.99,
                total_timesteps=500_0000,
                checkpoint_freq=50000,
                eval_freq=5000,
            ),
        ),
        "ppo_control_for_pose": PPOConfig(
            name="ppo_control_for_pose",
            model_type="PPO",
            description="PPO for controlling pose",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForPosition.x86_64",
            ),
            training_config=PPOTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=1024,
                n_steps=512,
                n_epochs=10,
                gamma=0.99,
                total_timesteps=500_0000,
                checkpoint_freq=50000,
                eval_freq=5000,
            ),
        ),
        "ppo_control_for_moving_target_dynamic_smoke": PPOConfig(
            name="ppo_control_for_moving_target_dynamic_smoke",
            model_type="PPO",
            description="PPO smoke test for ControlForPosition_Dynamic moving-target tracking",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/ControlForPosition_Dynamic_10Hz_Server/ControlForPosition.x86_64",
                num_envs=1,
                env_base_port=11605,
                timeout_wait=360,
                no_graphics=True,
            ),
            training_config=PPOTrainingConfig(
                learning_rate=linear_schedule(3e-4),
                batch_size=64,
                n_steps=64,
                n_epochs=1,
                gamma=0.99,
                total_timesteps=256,
                checkpoint_freq=128,
                eval_freq=10_000,
            ),
        ),
        "ppo_control_for_moving_target_dynamic": PPOConfig(
            name="ppo_control_for_moving_target_dynamic",
            model_type="PPO",
            description="PPO for ControlForPosition_Dynamic moving-target tracking",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="../../artifacts/unity_builds/rl/linux/ControlForPosition_Dynamic_10Hz_Server/ControlForPosition.x86_64",
                num_envs=32,
                env_base_port=5005,
                timeout_wait=180,
                no_graphics=True,
            ),
            training_config=PPOTrainingConfig(
                learning_rate=linear_schedule(2e-4),
                batch_size=2048,
                n_steps=1024,
                n_epochs=10,
                gamma=0.99,
                total_timesteps=10_000_000,
                checkpoint_freq=50_000,
                eval_freq=10_000,
            ),
        ),
        "ppo_control_for_velocity": PPOConfig(
            name="ppo_control_for_velocity",
            model_type="PPO",
            description="PPO for controlling velocity",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForVelocity.x86_64",
                num_envs=64
            ),
            training_config=PPOTrainingConfig(
                # env max step: 300
                learning_rate=1e-4,
                batch_size=1024,
                n_steps=300,
                n_epochs=10,
                gamma=0.99,
                total_timesteps=500_0000,
                checkpoint_freq=50000,
                eval_freq=5000,
            ),
        ),
        "ppo_control_for_velocity_no_rotation": PPOConfig(
            name="ppo_control_for_velocity_no_rotation",
            model_type="PPO",
            description="PPO for controlling velocity without rotation",
            env_config=PPOEnvironmentConfig(
                time_scale=10.0,
                env_path="/RLControl/build/ControlForVelocity_NoRotation.x86_64",
                num_envs=64
            ),
            training_config=PPOTrainingConfig(
                # env max step: 300
                learning_rate=1e-4,
                batch_size=1024,
                n_steps=300,
                n_epochs=10,
                gamma=0.99,
                total_timesteps=500_0000, # 1e6即收敛！
                checkpoint_freq=50000,
                eval_freq=5000,
            ),
        ),
    }


# 导出函数便于使用
__all__ = ["PPOEnvironmentConfig", "PPOTrainingConfig", "PPOConfig", "get_ppo_control_configs"]
