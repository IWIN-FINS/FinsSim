import os
import torch
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.algorithms import MappoConfig
from torch import nn
from torchrl.envs import EnvBase, PettingZooEnv
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from benchmarl.environments import PettingZooTask
from benchmarl.algorithms import Mappo  # 例如使用 MAPPO 算法
from benchmarl.models.mlp import MlpConfig


# 建议设置：处理 Unity 启动时的端口偏移
seed = 114514


def main():
    # 1. 创建环境
    def pz_env_creator():
        """创建 PettingZoo 环境"""
        config_channel = EngineConfigurationChannel()
        config_channel.set_configuration_parameters(time_scale=10.0) 
        
        unity_env = UnityEnvironment(
            file_name="/RLChase/build/RLChase.x86_64",
            side_channels=[config_channel],
            no_graphics=True,
            seed=seed,
        )
        
        pz_env = UnityParallelEnv(unity_env)
        return pz_env
    
    env = pz_env_creator()

    # 2. 配置 MAPPO 算法
    mappo_config = MappoConfig.default()
    mappo_config.lr = 5e-4                   # 学习率
    mappo_config.num_epochs = 10             # SGD 迭代次数
    mappo_config.batch_size = 512            # 批大小

    # 定义任务
    # 注意：由于 Unity 是动态自定义的，建议使用 benchMARL 的通用 PettingZoo 加载器
    task = PettingZooTask.CUSTOM.get_from_env(env)

    # 配置实验
    experiment = Experiment(
        task=task,
        algorithm_config=mappo_config,
        model_config=MlpConfig(num_cells=[128, 128], layer_class=nn.Linear, activation_class=nn.ReLU),
        seed=42,
        config=ExperimentConfig(

        ),
        critic_model_config=MlpConfig(num_cells=[128, 128], layer_class=nn.Linear, activation_class=nn.ReLU),
        callbacks=None,
        # 或 "cuda"
    )

    print("开始使用 BenchMARL MAPPO 训练...")
    print(f"设备: {'GPU' if torch.cuda.is_available() else 'CPU'}")

    # 开始训练
    experiment.run()

    # 4. 开始训练


    # 5. 保存最终模型
    print("训练任务圆满完成。")

if __name__ == "__main__":
    main()