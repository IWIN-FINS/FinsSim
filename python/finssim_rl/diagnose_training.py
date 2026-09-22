#!/usr/bin/env python3
"""
PPO 训练诊断脚本

监控关键指标：
- clip_fraction：应该在 0.05-0.1 之间（越低越好）
- approx_kl：应该在 0.01-0.02 之间（越低越好）
- policy loss：应该稳定下降
- value loss：应该稳定下降
"""

import sys
from pathlib import Path
import numpy as np
from collections import deque

# 尝试导入 tensorboard
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
except ImportError:
    print("❌ 需要安装 tensorboard: pip install tensorboard")
    sys.exit(1)


def analyze_training_logs(log_dir: str, window_size: int = 100):
    """分析训练日志"""
    log_path = Path(log_dir)
    
    # 找到最新的事件文件
    event_files = list(log_path.glob("events.out.*"))
    if not event_files:
        print(f"❌ 没有找到事件文件在 {log_dir}")
        return
    
    latest_file = max(event_files, key=lambda p: p.stat().st_mtime)
    print(f"✓ 读取事件文件: {latest_file.name}")
    
    ea = EventAccumulator(str(latest_file.parent))
    ea.Reload()
    
    # 获取标量数据
    scalar_tags = ea.Tags()['scalars']
    print(f"\n✓ 找到 {len(scalar_tags)} 个标量")
    
    # 关键指标
    metrics = {
        'clip_fraction': None,
        'approx_kl': None,
        'loss/policy_loss': None,
        'loss/value_loss': None,
        'rollout/ep_rew_mean': None,
    }
    
    print("\n" + "=" * 70)
    print("关键指标诊断".center(70))
    print("=" * 70)
    
    for metric_name in metrics.keys():
        if metric_name not in scalar_tags:
            continue
        
        events = ea.Scalars(metric_name)
        values = np.array([e.value for e in events[-window_size:]])
        steps = np.array([e.step for e in events[-window_size:]])
        
        if len(values) == 0:
            continue
        
        latest_val = values[-1]
        mean_val = values.mean()
        std_val = values.std()
        
        # 判断指标是否健康
        status = "✓"
        advice = ""
        
        if metric_name == "clip_fraction":
            if latest_val > 0.25:
                status = "⚠️"
                advice = "→ 过高，学习率可能太大 or batch_size 太小"
            elif latest_val < 0.01:
                advice = "→ 可能太低，可以增加学习率"
            else:
                advice = "→ 正常范围"
        
        elif metric_name == "approx_kl":
            if latest_val > 0.05:
                status = "⚠️"
                advice = "→ 过高，策略变化太快，需要降低学习率"
            elif latest_val < 0.001:
                advice = "→ 可能太低，学习不充分"
            else:
                advice = "→ 正常范围"
        
        elif metric_name in ["loss/policy_loss", "loss/value_loss"]:
            trend = "↓" if values[-5:].mean() < values[-20:-15].mean() else "↑"
            status = "✓" if "↓" in trend else "⚠️"
            advice = f"→ {trend} 趋势 (最近5步均值 vs 之前15-20步)"
        
        elif metric_name == "rollout/ep_rew_mean":
            trend = "↑" if values[-5:].mean() > values[-20:-15].mean() else "↓"
            status = "✓" if "↑" in trend else "⚠️"
            advice = f"→ {trend} 趋势"
        
        print(f"\n{status} {metric_name:30s}")
        print(f"  最新值:     {latest_val:12.6f}")
        print(f"  窗口均值:   {mean_val:12.6f} ± {std_val:8.6f}")
        print(f"  步数范围:   {steps[0]:8.0f} ~ {steps[-1]:8.0f}")
        print(f"  {advice}")


def diagnose_training_config():
    """诊断训练配置"""
    print("\n" + "=" * 70)
    print("训练配置建议".center(70))
    print("=" * 70)
    
    suggestions = {
        "clip_fraction 过高 (> 0.25)": [
            "1. 检查 batch_size 是否 <= n_steps",
            "2. 降低学习率（当前 2e-4，可尝试 1e-4）",
            "3. 增加 clip_range（当前 0.2，可尝试 0.3）",
            "4. 增加 n_epochs（当前 10，可尝试 15）",
        ],
        "approx_kl 过高 (> 0.05)": [
            "1. 立即降低学习率",
            "2. 增加 target_kl 或设置较小的 clip_range",
            "3. 检查环境反馈是否正确（rewards 是否合理）",
            "4. 增加 gae_lambda（当前 0.95，可尝试 0.98）",
        ],
        "loss 持续上升": [
            "1. 学习率可能太高",
            "2. 检查数据质量（是否有 NaN）",
            "3. 检查梯度范数是否被 clip（max_grad_norm）",
            "4. 减少 ent_coef（当前 0.005）",
        ],
        "reward 没有增加": [
            "1. 检查极点范围 TAU_MIN/TAU_MAX 是否合理",
            "2. 验证环境是否正确运行",
            "3. 增加训练步数",
            "4. 检查 x/y/z/yaw 误差和 yaw 保持半径是否符合任务目标",
        ],
    }
    
    for issue, fixes in suggestions.items():
        print(f"\n📌 {issue}:")
        for fix in fixes:
            print(f"   {fix}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("用法: python diagnose_training.py <log_dir> [window_size]")
        print("\n示例:")
        print("  python diagnose_training.py logs/v5_pos_control")
        print("  python diagnose_training.py logs/v5_pos_control 200")
        sys.exit(1)
    
    log_dir = sys.argv[1]
    window_size = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    
    print(f"\n分析日志目录: {log_dir}")
    print(f"窗口大小: {window_size} 步")
    
    try:
        analyze_training_logs(log_dir, window_size)
        diagnose_training_config()
        
        print("\n" + "=" * 70)
        print("总结".center(70))
        print("=" * 70)
        print("""
✓ 如果 clip_fraction 和 approx_kl 仍然很高，最可能的原因是：

  1. batch_size 过大
     → 已修改配置: batch_size 256 (之前 1024)
  
  2. clip_range 过小
     → 已修改配置: clip_range 0.2 (之前 0.1)
  
  3. 学习率不匹配
     → 已修改配置: learning_rate 2e-4 (之前 3e-4)

✓ 重新训练命令:
  python -m scripts.train \\
    --config hybrid_pid_pose_v3 \\
    --exp-name v5_pos_control_fixed \\
    --num-envs 8 \\
    --overwrite

✓ 监控训练进度:
  tensorboard --logdir=./logs/ --port=6007
        """)
    
    except Exception as e:
        print(f"❌ 分析失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
