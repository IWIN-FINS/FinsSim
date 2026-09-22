import numpy as np
import argparse
from stable_baselines3 import PPO, SAC
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
import shutil
from pathlib import Path


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


def find_latest_checkpoint(model_save_path):
    """
    找到最新的检查点文件
    """
    model_save_path = Path(model_save_path)
    if not model_save_path.exists():
        return None

    # 查找所有 .zip 文件（稳定基线3保存为 .zip）
    checkpoint_files = sorted(model_save_path.glob("ppo_underwater_model_*.zip"))

    if checkpoint_files:
        # 返回最新的检查点
        return str(checkpoint_files[-1])

    return None


def make_callbacks(env, log_dir, model_save_path, num_cpu) -> CallbackList:
    # 2. 定义 CheckpointCallback：定期保存模型（防止断电或崩溃）
    checkpoint_callback = CheckpointCallback(
        save_freq=int(50000/num_cpu),  # 每 50000/32 步保存一次
        save_path=model_save_path,
        name_prefix="sac_underwater_model"
    )

    # 3. 定义 EvalCallback：自动保存“历史最佳”模型
    # 它会定期在一个独立的环境中测试模型，如果得分更高，就保存 best_model.zip
    eval_callback = EvalCallback(
        env,  # 用于评估的环境
        best_model_save_path=model_save_path,  # 最佳模型保存路径
        log_path=log_dir,  # 日志路径
        eval_freq=int(5000/num_cpu),  # 每 5000 步评估一次
        deterministic=True,  # 评估时使用确定性动作
        render=False  # 评估时不渲染
    )
    # 组合所有的 callback
    callback_list = CallbackList([checkpoint_callback, eval_callback])
    return callback_list


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Train SAC on underwater simulation")
    parser.add_argument("--exp-name", type=str, default="SAC_default",
                        help="Experiment name for logging")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from the last checkpoint")
    parser.add_argument("--overwrite", action="store_true",
                        help="Overwrite existing experiment directory")
    args = parser.parse_args()

    env_path = "/Water2/build/RLControl.x86_64"
    num_cpu = 32  # 想要并行的进程数

    # 1. 根据实验名称设置保存路径
    log_dir = f"./logs/{args.exp_name}/"
    model_save_path = f"./checkpoints/{args.exp_name}/"
    
    # 检查目录是否已存在且有文件
    model_dir_exists = os.path.exists(model_save_path)
    log_dir_exists = os.path.exists(log_dir)
    
    if (model_dir_exists or log_dir_exists) and not args.resume and not args.overwrite:
        print(f"Error: Experiment '{args.exp_name}' already exists!")
        print(f"Please use one of the following options:")
        print(f"  1. --resume: Resume training from checkpoint")
        print(f"  2. --overwrite: Overwrite existing files")
        print(f"  3. Use a different --exp-name")
        return
    
    if (model_dir_exists or log_dir_exists) and args.overwrite and not args.resume:
        print(f"Warning: Overwriting existing experiment '{args.exp_name}'")
        import shutil
        if model_dir_exists:
            shutil.rmtree(model_save_path)
        if log_dir_exists:
            shutil.rmtree(log_dir)
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_save_path, exist_ok=True)

    # 1. 创建并行环境
    env = SubprocVecEnv([make_unity_env(env_path, i) for i in range(num_cpu)])

    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Experiment name: {args.exp_name}")
    print(f"Log directory: {log_dir}")
    print(f"Model save path: {model_save_path}")

    callback_list = make_callbacks(env, log_dir, model_save_path, num_cpu)

    def train_from_scratch():
        return SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_0000,  # 经验池大小
        batch_size=256,       # 每次训练提取的样本数
        tau=0.005,            # 软更新系数
        gamma=0.99,
        train_freq=32,         # 采样32步就训练1次
        gradient_steps=32,
        learning_starts=1000, # 先收集点数据再开始练
        tensorboard_log='./logs',
        device="cuda",        # 强烈建议用 GPU
    )

    # 检查是否需要加载已有模型
    if args.resume:
        # 查找最新的检查点
        checkpoint_path = find_latest_checkpoint(model_save_path)
        if checkpoint_path:
            print(f"Resuming training from: {checkpoint_path}")
            model = SAC.load(checkpoint_path, env=env, device="cuda")
        else:
            print(f"No checkpoint found in {model_save_path}, training from scratch")
            model = train_from_scratch()
    else:
        # 从头开始训练
        model = train_from_scratch()

    # 开始训练
    model.learn(total_timesteps=500_0000, callback=callback_list, progress_bar=True, reset_num_timesteps=False)

    # 训练完成后保存模型
    final_model_path = f"{model_save_path}final_model.zip"
    print(f"Saving model to {final_model_path}")
    model.save(final_model_path)


if __name__ == '__main__':
    main()