#!/usr/bin/env python3
"""
使用示例：展示新架构的各种用法

这个文件展示了如何使用重构后的训练框架
"""

import sys
from pathlib import Path

# 示例 1: 列出所有可用的配置
def example_list_configs():
    """示例：列出所有可用配置"""
    from finssim_rl.training.config import list_configs
    
    print("=" * 60)
    print("示例 1: 列出所有可用配置")
    print("=" * 60)
    
    configs = list_configs()
    print(f"可用的配置: {', '.join(configs)}\n")


# 示例 2: 获取特定配置的信息
def example_get_config():
    """示例：获取特定配置的信息"""
    from finssim_rl.training.config import get_config
    
    print("=" * 60)
    print("示例 2: 获取特定配置的信息")
    print("=" * 60)
    
    config = get_config("sac_default")
    print(f"配置名称: {config.name}")
    print(f"模型类型: {config.model_type}")
    print(f"学习率: {config.training_config.learning_rate}")
    print(f"Buffer 大小: {config.training_config.buffer_size}")
    print(f"总时间步: {config.training_config.total_timesteps}\n")


# 示例 3: 创建 Args 对象
def example_args():
    """示例：创建 Args 对象"""
    from finssim_rl.training.args import Args
    
    print("=" * 60)
    print("示例 3: 创建 Args 对象")
    print("=" * 60)
    
    # 方式 1: 最少参数
    args1 = Args(config="sac_default")
    print(f"方式 1 (最少参数):")
    print(f"  config: {args1.config}")
    print(f"  exp_name: {args1.exp_name}")
    print(f"  log_dir: {args1.log_dir}")
    print()
    
    # 方式 2: 自定义 exp_name
    args2 = Args(config="ppo_chase", exp_name="my_ppo_test")
    print(f"方式 2 (自定义 exp_name):")
    print(f"  config: {args2.config}")
    print(f"  exp_name: {args2.exp_name}")
    print(f"  log_dir: {args2.log_dir}")
    print()
    
    # 方式 3: 自定义所有参数
    args3 = Args(
        config="sac_aggressive",
        exp_name="my_sac_test",
        num_envs=64,
        device="cpu"
    )
    print(f"方式 3 (自定义所有参数):")
    print(f"  config: {args3.config}")
    print(f"  exp_name: {args3.exp_name}")
    print(f"  num_envs: {args3.num_envs}")
    print(f"  device: {args3.device}\n")


# 示例 4: 模拟命令行用法
def example_cli_usage():
    """示例：展示命令行用法"""
    print("=" * 60)
    print("示例 4: 命令行用法")
    print("=" * 60)
    
    cli_examples = [
        "finssim-rl train --config ppo_chase",
        "finssim-rl train --config ppo_chase --exp-name my_ppo_test",
        "finssim-rl train --config sac_default --exp-name my_sac_test --overwrite",
        "finssim-rl train --config sac_aggressive --exp-name my_sac_aggressive --num-envs 64",
        "finssim-rl train --config sac_default --exp-name my_sac_test --resume",
        "finssim-rl train --config ppo_chase --device cpu",
        "finssim-rl train --config sac_default --output-dir ./artifacts/runs/rl/my_sac_test",
    ]
    
    for i, example in enumerate(cli_examples, 1):
        print(f"{i}. {example}")
    print()


# 示例 5: 查看训练配置差异
def example_config_comparison():
    """示例：对比不同配置的差异"""
    from finssim_rl.training.config import get_config
    
    print("=" * 60)
    print("示例 5: 配置对比")
    print("=" * 60)
    
    configs_to_compare = ["ppo_chase", "sac_default", "sac_aggressive"]
    
    for config_name in configs_to_compare:
        config = get_config(config_name)
        tc = config.training_config
        
        print(f"\n{config_name}:")
        print(f"  模型: {config.model_type}")
        print(f"  学习率: {tc.learning_rate}")
        print(f"  总时间步: {tc.total_timesteps}")
        print(f"  Checkpoint 频率: {tc.checkpoint_freq}")
        if config.model_type == "SAC":
            print(f"  Buffer 大小: {tc.buffer_size}")
            print(f"  Batch 大小: {tc.batch_size}")
    print()


def main():
    """运行所有示例"""
    examples = [
        example_list_configs,
        example_get_config,
        example_args,
        example_cli_usage,
        example_config_comparison,
    ]
    
    for example_func in examples:
        example_func()


if __name__ == "__main__":
    main()
