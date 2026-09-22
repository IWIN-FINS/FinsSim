#!/usr/bin/env python3
"""
配置系统示例

演示如何使用新的模块化配置系统
"""

import sys
from pathlib import Path


def example_basic_usage():
    """示例 1: 基本用法"""
    print("\n" + "=" * 70)
    print("示例 1: 基本用法")
    print("=" * 70)
    
    from finssim_rl.training.config import get_config, list_configs
    
    # 列出所有配置
    configs = list_configs()
    print(f"可用的配置 ({len(configs)} 个):")
    for i, config_name in enumerate(configs, 1):
        print(f"  {i}. {config_name}")
    
    # 获取特定配置
    print("\n获取 'sac_default' 配置:")
    config = get_config("sac_default")
    print(f"  名称: {config.name}")
    print(f"  模型: {config.model_type}")
    print(f"  描述: {config.description}")


def example_by_model_type():
    """示例 2: 按模型类型过滤"""
    print("\n" + "=" * 70)
    print("示例 2: 按模型类型过滤")
    print("=" * 70)
    
    from finssim_rl.training.config import list_configs_by_model
    
    ppo_configs = list_configs_by_model("PPO")
    sac_configs = list_configs_by_model("SAC")
    
    print(f"PPO 配置 ({len(ppo_configs)} 个):")
    for config_name in ppo_configs:
        print(f"  - {config_name}")
    
    print(f"\nSAC 配置 ({len(sac_configs)} 个):")
    for config_name in sac_configs:
        print(f"  - {config_name}")


def example_config_details():
    """示例 3: 查看配置详情"""
    print("\n" + "=" * 70)
    print("示例 3: 查看配置详情")
    print("=" * 70)
    
    from finssim_rl.training.config import get_config
    
    config = get_config("sac_aggressive")
    tc = config.training_config
    ec = config.env_config
    
    print(f"\n配置名称: {config.name}")
    print(f"模型类型: {config.model_type}")
    print(f"描述: {config.description}")
    
    print(f"\n环境配置:")
    print(f"  - Graphics: {not ec.no_graphics}")
    print(f"  - Time Scale: {ec.time_scale}x")
    print(f"  - Flatten Branched: {ec.flatten_branched}")
    
    print(f"\n训练配置:")
    print(f"  - 学习率: {tc.learning_rate}")
    print(f"  - Buffer 大小: {tc.buffer_size:,}")
    print(f"  - Batch 大小: {tc.batch_size}")
    print(f"  - 总时间步: {tc.total_timesteps:,}")
    print(f"  - Checkpoint 频率: {tc.checkpoint_freq:,}")
    print(f"  - 评估频率: {tc.eval_freq:,}")
    
    if config.model_type == "SAC":
        print(f"  - Tau: {tc.tau}")
        print(f"  - Train Freq: {tc.train_freq}")
        print(f"  - Gradient Steps: {tc.gradient_steps}")


def example_print_all():
    """示例 4: 打印所有配置"""
    print("\n" + "=" * 70)
    print("示例 4: 打印所有配置详情")
    print("=" * 70)
    
    from finssim_rl.training.config import print_all_configs
    print_all_configs()


def example_compare_configs():
    """示例 5: 对比配置"""
    print("\n" + "=" * 70)
    print("示例 5: 对比不同的 SAC 配置")
    print("=" * 70)
    
    from finssim_rl.training.config import get_config
    
    sac_configs = ["sac_gentle", "sac_default", "sac_aggressive"]
    
    print(f"\n{'配置名称':<20} {'学习率':<15} {'Buffer':<15} {'Batch':<10} {'总步数':<15}")
    print("-" * 75)
    
    for config_name in sac_configs:
        config = get_config(config_name)
        tc = config.training_config
        
        print(f"{config_name:<20} {tc.learning_rate:<15.2e} {tc.buffer_size:<15,} "
              f"{tc.batch_size:<10} {tc.total_timesteps:<15,}")


def example_usage_in_code():
    """示例 6: 在代码中使用配置"""
    print("\n" + "=" * 70)
    print("示例 6: 在代码中使用配置")
    print("=" * 70)
    
    from finssim_rl.training.config import get_config
    
    # 这是训练脚本中的常见用法
    config_name = "sac_default"
    config = get_config(config_name)
    
    print(f"\n模拟训练脚本:")
    print(f"  1. 选择配置: {config_name}")
    print(f"  2. 获取模型类型: {config.model_type}")
    print(f"  3. 获取环境参数:")
    print(f"     - Time Scale: {config.env_config.time_scale}")
    print(f"  4. 获取训练参数:")
    print(f"     - Learning Rate: {config.training_config.learning_rate}")
    print(f"     - Batch Size: {config.training_config.batch_size}")
    print(f"  5. 创建模型和环境...")
    print(f"  6. 开始训练...")


def example_add_custom_config():
    """示例 7: 动态添加自定义配置"""
    print("\n" + "=" * 70)
    print("示例 7: 动态注册自定义配置")
    print("=" * 70)
    
    from finssim_rl.training.config import (
        BaseConfig,
        BaseEnvironmentConfig,
        BaseTrainingConfig,
        register_config,
        get_config,
    )
    
    # 创建自定义配置
    custom_config = BaseConfig(
        name="sac_custom_demo",
        model_type="SAC",
        description="动态创建的自定义配置演示",
        env_config=BaseEnvironmentConfig(
            time_scale=12.0,
        ),
        training_config=BaseTrainingConfig(
            learning_rate=2.5e-4,
            total_timesteps=750_0000,
        ),
    )
    
    # 注册配置
    register_config(custom_config)
    
    print(f"\n已注册新配置: {custom_config.name}")
    print(f"  描述: {custom_config.description}")
    print(f"  模型: {custom_config.model_type}")
    print(f"  学习率: {custom_config.training_config.learning_rate}")
    
    # 验证可以获取
    retrieved = get_config("sac_custom_demo")
    print(f"\n验证: 成功获取 '{retrieved.name}' 配置")


def main():
    """运行所有示例"""
    print("\n" + "=" * 70)
    print("配置系统示例 - 模块化配置架构演示")
    print("=" * 70)
    
    examples = [
        example_basic_usage,
        example_by_model_type,
        example_config_details,
        example_print_all,
        example_compare_configs,
        example_usage_in_code,
        example_add_custom_config,
    ]
    
    for example_func in examples:
        try:
            example_func()
        except Exception as e:
            print(f"\n❌ 错误: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 70)
    print("✓ 所有示例完成！")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
