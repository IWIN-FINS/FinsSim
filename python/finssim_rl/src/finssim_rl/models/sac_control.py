"""
SAC Control 模型配置

定义了 SAC 特定的配置参数和实验配置
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import VecEnv

from finssim_rl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig


@dataclass
class SACEnvironmentConfig(BaseEnvironmentConfig):
    """SAC 特定的环境配置"""
    time_scale: float = 10.0  # SAC 推荐时间缩放


@dataclass
class SACTrainingConfig(BaseTrainingConfig):
    """SAC 特定的训练配置"""
    batch_size: int = 256  # SAC 特定参数
    buffer_size: int = 100_0000  # SAC 特定参数：回放缓冲区大小
    tau: float = 0.005  # SAC 特定参数：软更新系数
    train_freq: int = 32  # SAC 特定参数：多少个环境步后更新一次模型
    gradient_steps: int = 32  # SAC 特定参数：每次更新的梯度步数
    learning_starts: int = 1000  # SAC 特定参数：开始学习前的观察步数


@dataclass
class SACConfig(BaseConfig):
    """SAC Control 配置"""
    model_type: str = "SAC"
    env_config: SACEnvironmentConfig = field(default_factory=SACEnvironmentConfig)
    training_config: SACTrainingConfig = field(default_factory=SACTrainingConfig)

    def create_model(self, env: VecEnv, args: Any, load_checkpoint: Optional[str] = None):
        tc = self.training_config
        model = SAC(
            "MlpPolicy",
            env,
            verbose=1,
            learning_rate=tc.learning_rate,
            buffer_size=tc.buffer_size,
            batch_size=tc.batch_size,
            tau=tc.tau,
            gamma=tc.gamma,
            train_freq=tc.train_freq,
            gradient_steps=tc.gradient_steps,
            learning_starts=tc.learning_starts,
            tensorboard_log=getattr(args, "tensorboard_dir", None) or "./logs",
            device=args.device,
        )

        if load_checkpoint:
            print(f"Loading model from checkpoint: {load_checkpoint}")
            model = type(model).load(load_checkpoint, env=env, device=args.device)

        return model

    def load_model_for_eval(self, model_path: str, device: str = "auto", env=None):
        return SAC.load(model_path, env=env, device=device)


def get_sac_control_configs() -> Dict[str, SACConfig]:
    """
    返回所有 SAC Control 相关的配置
    
    Returns:
        配置名称到配置对象的映射
    """
    return {
        "sac_default": SACConfig(
            name="sac_default",
            model_type="SAC",
            description="SAC with default parameters",
            env_config=SACEnvironmentConfig(
                time_scale=10.0,
            ),
            training_config=SACTrainingConfig(
                learning_rate=3e-4,
                buffer_size=100_0000,
                batch_size=256,
                tau=0.005,
                gamma=0.99,
                train_freq=32,
                gradient_steps=32,
                learning_starts=1000,
                total_timesteps=500_0000,
                checkpoint_freq=50000,
                eval_freq=5000,
            ),
        ),
        "sac_aggressive": SACConfig(
            name="sac_aggressive",
            model_type="SAC",
            description="SAC with aggressive parameters",
            env_config=SACEnvironmentConfig(
                time_scale=15.0,
            ),
            training_config=SACTrainingConfig(
                learning_rate=1e-3,
                buffer_size=200_0000,
                batch_size=512,
                tau=0.01,
                gamma=0.99,
                train_freq=16,
                gradient_steps=64,
                learning_starts=500,
                total_timesteps=1000_0000,
                checkpoint_freq=100000,
                eval_freq=10000,
            ),
        ),
        "sac_gentle": SACConfig(
            name="sac_gentle",
            model_type="SAC",
            description="SAC with gentle learning rate",
            env_config=SACEnvironmentConfig(
                time_scale=5.0,
            ),
            training_config=SACTrainingConfig(
                learning_rate=1e-4,
                buffer_size=50_0000,
                batch_size=128,
                tau=0.001,
                gamma=0.995,
                train_freq=64,
                gradient_steps=16,
                learning_starts=2000,
                total_timesteps=500_0000,
                checkpoint_freq=50000,
                eval_freq=5000,
            ),
        ),
    }


# 导出函数便于使用
__all__ = ["SACEnvironmentConfig", "SACTrainingConfig", "SACConfig", "get_sac_control_configs"]
