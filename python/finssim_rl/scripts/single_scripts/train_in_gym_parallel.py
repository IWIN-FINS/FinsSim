
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
def make_unity_env(env_path, rank, seed=0):
    """
    创建环境的工厂函数
    """
    def _init():
        channel = EngineConfigurationChannel()
        # 注意：每个进程的 worker_id 必须唯一，否则会端口冲突
        unity_env = UnityEnvironment(
            file_name=env_path,
            side_channels=[channel],
            no_graphics=True,
            worker_id=rank  # 关键：区分不同的 Unity 进程
        )
        env = UnityToGymWrapper(
            unity_env,
            uint8_visual=False,
            flatten_branched=True,
            allow_multiple_obs=False,
        )
        # env = Monitor(env) # 版本不兼容
        # 3. 设置时间缩放 (例如设置为 20 倍速)
        # 注意：这必须在环境 reset 之后或初始化后调用
        channel.set_configuration_parameters(time_scale=10.0)
        return env
    return _init


def make_callbacks(env, log_dir, model_save_path)->CallbackList:

    # 2. 定义 CheckpointCallback：定期保存模型（防止断电或崩溃）
    checkpoint_callback = CheckpointCallback(
        save_freq=1562,           # 每 50000/32 步保存一次
        save_path=model_save_path,
        name_prefix="ppo_underwater_model"
    )

    # 3. 定义 EvalCallback：自动保存“历史最佳”模型
    # 它会定期在一个独立的环境中测试模型，如果得分更高，就保存 best_model.zip
    eval_callback = EvalCallback(
        env,                             # 用于评估的环境
        best_model_save_path=model_save_path, # 最佳模型保存路径
        log_path=log_dir,                # 日志路径
        eval_freq=1562,                  # 每 5000 步评估一次
        deterministic=True,              # 评估时使用确定性动作
        render=False                     # 评估时不渲染
    )
    # 组合所有的 callback
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    return callback_list

def main():
    env_path = "/Water2/build/RLControl.x86_64"
    num_cpu = 32  # 想要并行的进程数
    # 1. 设置保存路径
    log_dir = "./logs/underwater_sim/"
    os.makedirs(log_dir, exist_ok=True)
    model_save_path = "./checkpoints/"
    os.makedirs(model_save_path, exist_ok=True)
    
    # 1. 创建并行环境
    env = SubprocVecEnv([make_unity_env(env_path, i) for i in range(num_cpu)])

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    callback_list = make_callbacks(env, log_dir, model_save_path)
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=2.5e-4,
        n_steps=512, # 每多少个样本就更新一遍脑子
        batch_size=1024,
        gamma=0.99,
        tensorboard_log='./logs',
        device="cpu",
    )

    # 开始训练
    # total_timesteps 是总步数，建议先设一个小值（如 10000）看看能否跑通
    model.learn(total_timesteps=1000000, callback=callback_list, progress_bar=True)

    # 训练完成后保存模型

    print("Saving model to unity_model.zip")
    model.save("checkpoints/unity_model.zip")



if __name__ == '__main__':
    main()