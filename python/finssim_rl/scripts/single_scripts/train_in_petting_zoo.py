import os
import ray
from marllib import marl
from ray.tune.registry import register_env
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

# 建议设置：处理 Unity 启动时的端口偏移
OS_SEED = 114514

def env_creator(env_config):
    """
    env_config 是 Ray 自动传入的字典，包含 worker_index 等信息
    """
    # 1. 侧信道配置：每个 Worker 独立配置
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=10.0) 
    
    # 2. 这里的 worker_index 非常重要，它决定了 Unity 实例的端口偏移
    # 避免多个 Unity 实例抢占同一个通讯端口
    worker_id = env_config.worker_index 
    
    unity_env = UnityEnvironment(
        file_name="/Water2/build/RLControl.x86_64",
        side_channels=[config_channel],
        no_graphics=True,
        seed=OS_SEED + worker_id,
        worker_id=worker_id  # 关键：确保端口不冲突
    )
    
    # 转换为 PettingZoo 接口再转为 RLlib 并行接口
    pz_env = UnityParallelEnv(unity_env)
    return ParallelPettingZooEnv(pz_env)

def main():
    # 初始化 Ray (可选：指定内存或临时目录)
    ray.init(ignore_reinit_error=True)

    # 1. 注册环境
    ENV_NAME = "my_unity_env"
    register_env(ENV_NAME, lambda config: env_creator(config))

    # 2. 准备环境配置
    # 注意：env_params 里的参数会传递给 env_creator 的 config
    env = marl.make_env(
        environment_name=ENV_NAME,
        map_name="underwater_task",
        force_coop=False,
        env_params={
            "worker_index": 0 # 默认值
        }
    )

    # 3. 配置算法超参数与设备
    # 这里的参数直接决定了训练的并行度和硬件使用
    params = {
        "algo_args": {
            "batch_episode": 20,    # 每次更新采集的回合数
            "lr": 5e-4,
            "num_sgd_iter": 10,
        },
        "ray_args": {
            "num_gpus": 1,          # 设备选择：1 表示使用 GPU，0 表示纯 CPU
            "num_workers": 32,       # 并行逻辑：开启多少个子进程（即多少个 Unity 窗口）
            "local_dir": "./underwater_results", # Tensorboard 日志和 Checkpoint 路径
        }
    }

    # 4. 选择算法
    mappo = marl.algos.mappo(hyperparam_source="common")

    # 5. 开始训练
    print(f"训练启动！设备使用情况: {'GPU' if params['ray_args']['num_gpus'] > 0 else 'CPU'}")
    print(f"并行实例数量: {params['ray_args']['num_workers']}")
    
    mappo.fit(
        env, 
        model_config={
            "core_arch": "mlp", 
            "encode_layer": "256-256" # 稍微加深网络处理复杂水下环境
        }, 
        stop={'timesteps_total': 1000000},
        checkpoint_freq=50,               # 每 50 次迭代存一次
        checkpoint_at_end=True,
        share_policy="all",               # 同质智能体建议开启，极大减少参数量
        **params["algo_args"],            # 展开算法参数
        **params["ray_args"]              # 展开资源参数
    )

    # 6. 显式保存最终模型
    mappo.save("ppo_underwater_marl_final")
    print("训练任务圆满完成。")

if __name__ == "__main__":
    main()