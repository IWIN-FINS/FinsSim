"""
混合PPO+PID模型训练脚本

使用方式:
    python -m scripts.train_hybrid --config hybrid_pid_pose --exp-name hybrid_test --overwrite
    python -m scripts.train_hybrid --config hybrid_pid_chase --exp-name hybrid_chase_test --num-cpu 8
    python -m scripts.train_hybrid --config hybrid_pid_pose --exp-name hybrid_test --resume
"""

import tyro
from stable_baselines3.common.vec_env import SubprocVecEnv

from finssim_rl.training.args import Args
from finssim_rl.training.config import get_config, list_configs
from finssim_rl.models import make_unity_env
from finssim_rl.training.trainer import (
    find_latest_checkpoint,
    setup_directories,
    make_callbacks,
    create_model,
)


def main(args: Args) -> None:
    """
    训练混合PPO+PID模型
    
    Args:
        args: 命令行参数
    """
    # 获取配置
    try:
        config = get_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable configs: {', '.join(list_configs())}")
        return
    
    # 验证是混合模型配置
    if not config.model_type.startswith("HYBRID"):
        print(f"Error: Config '{args.config}' is not a hybrid model!")
        print(f"Please use one of the hybrid configs: {[c for c in list_configs() if 'hybrid' in c]}")
        return
    
    # 如果没有指定 exp_name，使用 config 的名称
    if args.exp_name == "":
        args.exp_name = args.config
        
    # 如果在config中指定了unity二进制env_path，则覆盖命令行参数
    if config.env_config.env_path and not args.env_path:
        args.env_path = config.env_config.env_path
    
    if not args.env_path:
        print("Error: No Unity environment path specified!")
        print("Please provide the path using --env-path or specify it in the config.")
        return
    
    # 重新生成日志和检查点路径（使用更新的 exp_name）
    args.log_dir = f"./logs/{args.exp_name}/"
    args.checkpoint_dir = f"./checkpoints/{args.exp_name}/"
    
    # 设置目录
    setup_directories(args)
    
    # 创建并行环境
    print(f"Creating {args.num_cpu} parallel environments...")
    env = SubprocVecEnv(
        [make_unity_env(args.env_path, i, config.env_config, args.port_offset) 
         for i in range(args.num_cpu)]
    )
    
    # 打印信息
    print(f"\n{'='*60}")
    print(f"Hybrid PPO+PID Model Training")
    print(f"{'='*60}")
    print(f"Configuration: {config.name}")
    print(f"Model Type: {config.model_type}")
    print(f"Experiment Name: {args.exp_name}")
    print(f"Observation Space: {env.observation_space}")
    print(f"Action Space: {env.action_space}")
    print(f"Number of Parallel Envs: {args.num_cpu}")
    print(f"Log Directory: {args.log_dir}")
    print(f"Checkpoint Directory: {args.checkpoint_dir}")
    print(f"Device: {args.device}")
    
    # 打印PID配置
    print(f"\n{'PPO Parameters':=^30}")
    print(f"Learning Rate: {config.training_config.learning_rate}")
    #print(f"N Steps: {config.training_config.n_steps}")
    #print(f"N Epochs: {config.training_config.n_epochs}")
    
    #print(f"\n{'PID Parameters':=^30}")
    #print(f"PID Learning Rate: {config.training_config.pid_learning_rate}")
    #print(f"PID Update Freq: {config.training_config.pid_update_freq}")
    #print(f"PID Loss Weight: {config.training_config.pid_loss_weight}")
    #print(f"Init Kp: {config.training_config.pid_init_Kp}")
    #print(f"Init Ki: {config.training_config.pid_init_Ki}")
    #print(f"Init Kd: {config.training_config.pid_init_Kd}")
    #print(f"{'='*60}\n")
    
    # 创建回调函数
    callback_list = make_callbacks(env, args, config)
    
    # 创建或加载模型
    checkpoint_path = None
    if args.resume:
        checkpoint_path = find_latest_checkpoint(args.checkpoint_dir)
        if checkpoint_path:
            print(f"Resuming from checkpoint: {checkpoint_path}")
    
    model = create_model(
        env=env,
        config=config,
        args=args,
        load_checkpoint=checkpoint_path,
    )
    
    # 开始训练
    print(f"Starting training...")
    model.learn(
        total_timesteps=config.training_config.total_timesteps,
        callback=callback_list,
        progress_bar=True,
        reset_num_timesteps=False if args.resume else True,
        tb_log_name=args.exp_name,
    )
    
    # 保存最终模型
    final_model_path = f"{args.checkpoint_dir}final_model.zip"
    print(f"\nTraining complete! Saving final model to {final_model_path}")
    model.save(final_model_path)
    
    print(f"Experiment '{args.exp_name}' finished successfully!")
    
    # 打印最终PID参数
    print(f"\n{'Final PID Parameters':=^30}")
    pid_params = model.get_pid_parameters()
    print(f"Kp: {pid_params['Kp']}")
    print(f"Ki: {pid_params['Ki']}")
    print(f"Kd: {pid_params['Kd']}")


if __name__ == '__main__':
    args = tyro.cli(Args)
    main(args)
