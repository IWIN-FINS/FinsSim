
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel

def main():
    # 1. 创建侧信道 (Side Channel)
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        "/Water2/build/RLControl.x86_64",
        side_channels = [channel],
        no_graphics=True  # 不进行渲染，极大提升 CPU/GPU 效率
    )
    # 尝试不同的参数组合
    env = UnityToGymWrapper(
        unity_env,
        uint8_visual=False,
        flatten_branched=True,      # 设置为 True
        allow_multiple_obs=False,   # 设置为 False 来避免 Tuple 空间
    )

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    # 3. 设置时间缩放 (例如设置为 20 倍速)
    # 注意：这必须在环境 reset 之后或初始化后调用
    channel.set_configuration_parameters(time_scale=10.0)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=2048, # 每多少个样本就更新一遍脑子
        batch_size=64,
        gamma=0.99,
        tensorboard_log='./logs',
    )

    # 开始训练
    # total_timesteps 是总步数，建议先设一个小值（如 10000）看看能否跑通
    model.learn(total_timesteps=10000)

    # 训练完成后保存模型

    print("Saving model to unity_model.pkl")
    model.save("checkpoints/unity_model.pkl")



if __name__ == '__main__':
    main()