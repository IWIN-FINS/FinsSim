import numpy as np
import argparse
from stable_baselines3 import SAC
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_gym_env import UnityToGymWrapper
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
import os
from pathlib import Path


def make_unity_env(env_path, rank=0, no_graphics=False):
    """
    创建单个环境的工厂函数
    """
    channel = EngineConfigurationChannel()
    unity_env = UnityEnvironment(
        file_name=env_path,
        side_channels=[channel],
        no_graphics=no_graphics,
        worker_id=rank
    )
    env = UnityToGymWrapper(
        unity_env,
        uint8_visual=False,
        flatten_branched=True,
        allow_multiple_obs=False,
    )
    channel.set_configuration_parameters(time_scale=1.0)  # 评估时实时速度
    return env


def find_best_model(model_save_path):
    """
    查找最好的模型文件
    """
    model_save_path = Path(model_save_path)
    if not model_save_path.exists():
        return None
    
    # 查找 best_model.zip
    best_model_path = model_save_path / "best_model.zip"
    if best_model_path.exists():
        return str(best_model_path)
    
    # 如果没有 best_model.zip，查找最新的检查点
    checkpoint_files = sorted(model_save_path.glob("sac_underwater_model_*.zip"))
    if checkpoint_files:
        return str(checkpoint_files[-1])
    
    return None


def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="Evaluate sac on underwater simulation")
    parser.add_argument("--exp-name", type=str, default="SAC_default",
                        help="Experiment name to evaluate")
    parser.add_argument("--num-episodes", type=int, default=5,
                        help="Number of episodes to run")
    parser.add_argument("--no-render", action="store_true",
                        help="Disable rendering")
    args = parser.parse_args()

    env_path = "/Water2/build/RLControl.x86_64"
    
    # 模型路径
    model_save_path = f"./models/{args.exp_name}/"
    
    # 创建环境，启用图形渲染（除非指定 --no-render）
    print("Creating environment...")
    env = make_unity_env(env_path, rank=0, no_graphics=args.no_render)
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Experiment name: {args.exp_name}")
    
    # 加载最好的模型
    model_path = find_best_model(model_save_path)
    if not model_path:
        print(f"Error: No model found in {model_save_path}")
        return
    
    print(f"Loading model from: {model_path}")
    model = SAC.load(model_path, env=env, device="auto")
    
    # 运行评估
    print(f"\nStarting evaluation ({args.num_episodes} episodes)...")
    episode_rewards = []
    
    for episode in range(args.num_episodes):
        # 处理不同版本 gym 的返回值差异
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result
        
        episode_reward = 0
        done = False
        step = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            
            # 处理不同版本的返回值
            if len(step_result) == 5:
                obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                obs, reward, done, info = step_result
            
            episode_reward += reward
            step += 1
        
        episode_rewards.append(episode_reward)
        print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    
    print(f"\n=== Evaluation Results ===")
    print(f"Mean Reward: {np.mean(episode_rewards):.2f}")
    print(f"Std Reward: {np.std(episode_rewards):.2f}")
    print(f"Min Reward: {np.min(episode_rewards):.2f}")
    print(f"Max Reward: {np.max(episode_rewards):.2f}")
    
    env.close()
    print("Evaluation completed!")


if __name__ == '__main__':
    main()
