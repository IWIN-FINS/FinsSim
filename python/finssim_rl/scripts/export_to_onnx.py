"""
ONNX模型导出脚本

使用方式:
    python -m scripts.export_to_onnx --config=ppo_control_for_pose --exp-name ControlForPoseTest --opset_version=17
    python -m scripts.export_to_onnx sac_default --model-type best --output-dir ./onnx_models
    python -m scripts.export_to_onnx ppo_control_for_pose --exp-name ControlForPoseTest --opset-version 14
"""

import os
import sys
import torch
import numpy as np
import tyro
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from finssim_rl.training.export_args import ExportArgs
from finssim_rl.training.config import get_config, list_configs
from finssim_rl.training.trainer import find_model_for_eval


class PolicyInferenceWrapper(torch.nn.Module):
    def __init__(self, policy):
        super().__init__()
        self.policy = policy
        self.policy.eval()
    
    def forward(self, observations):
        # 核心逻辑：使用 SB3 官方推荐的确定性动作获取方式
        # deterministic=True 通常是推理和 Unity 部署时需要的
        # 这会自动处理 extract_features -> mlp_extractor -> action_net 的全过程
        return self.policy(observations, deterministic=True)

# see https://github.com/DLR-RM/stable-baselines3/issues/383 and https://stable-baselines3.readthedocs.io/en/master/guide/export.html
class OnnxablePolicy(torch.nn.Module):
    def __init__(self, features_extractor, mlp_extractor, action_net, value_net):
        super(OnnxablePolicy, self).__init__()
        self.features_extractor = features_extractor
        self.mlp_extractor = mlp_extractor
        self.action_net = action_net
        self.value_net = value_net

    def forward(self, observations):
        features = self.features_extractor(observations)
        action_hidden, value_hidden = self.mlp_extractor(features)
        return self.action_net(action_hidden), self.value_net(value_hidden)


def _write_success_export_info(model, output_path: str, obs_shape: int) -> None:
    info_path = output_path.replace(".onnx", "_info.txt")
    action_net = getattr(model.policy, "action_net", None)
    action_dim = getattr(action_net, "out_features", None)
    lines = [
        "ONNX Export Information",
        "=======================",
        "",
        f"Observation size: {obs_shape}",
        f"Exported action head size: {action_dim}",
        f"Training env action space: {model.action_space}",
        "",
        "The ONNX 'actions' output is the deterministic policy mean from action_net.",
        "Clip it to [-1, 1] before sending commands to runtime control code.",
    ]

    if hasattr(model, "virtual_control_limits"):
        limits = [float(v) for v in getattr(model, "virtual_control_limits")]
        lines.extend(
            [
                "",
                "6D wrench-policy metadata",
                "-------------------------",
                "Normalized policy order: [surge, sway, heave, roll, pitch, yaw]",
                "Physical wrench order: [Fx, Fy, Fz, Mx, My, Mz]",
                f"Policy action limits: {limits}",
                "",
                "Conversion:",
                "  a = clip(actions, -1, 1)",
                "  virtual = a * limits",
                "  wrench = [virtual[0], virtual[2], virtual[1], virtual[3], virtual[5], virtual[4]]",
                "",
                "Note: Python model.predict() returns simulator-ready 8D thruster actions for",
                "this model family. For real deployment with downstream allocation, use the",
                "ONNX/policy 6D output and the wrench conversion above.",
            ]
        )

    with open(info_path, "w", encoding="utf-8") as stream:
        stream.write("\n".join(lines))
        stream.write("\n")


def export_to_onnx(args: ExportArgs) -> None:
    """
    导出强化学习模型为ONNX格式
    
    Args:
        args: 导出参数
    """
    # 获取配置
    try:
        config = get_config(args.config)
    except ValueError as e:
        print(f"Error: {e}")
        print(f"\nAvailable configs: {', '.join(list_configs())}")
        return
    
    # 如果没有指定 exp_name，使用 config 的名称
    if args.exp_name == "":
        print(f"No exp_name specified, using config name ({args.config}) as exp_name.")
        args.exp_name = args.config
    
    # 重新生成输出目录（使用更新的 exp_name）
    if args.output_dir is None or args.output_dir == f"./models_onnx/{args.config}/":
        args.output_dir = f"./models_onnx/{args.exp_name}/"
    
    if args.checkpoint_dir is None or args.checkpoint_dir == f"./checkpoints/{args.config}/":
        args.checkpoint_dir = f"./checkpoints/{args.exp_name}/"
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 查找模型
    if args.verbose:
        print(f"Looking for {args.model_type} model in {args.checkpoint_dir}...")
    
    model_path = find_model_for_eval(args.checkpoint_dir, args.model_type)
    
    if not model_path:
        print(f"Error: No model found in {args.checkpoint_dir}")
        print(f"Available model types: best, latest, final")
        return
    
    if args.verbose:
        print(f"Found model: {model_path}\n")
    
    # 打印导出信息
    print(f"{'='*60}")
    print(f"ONNX Model Export")
    print(f"{'='*60}")
    print(f"Configuration: {config.name}")
    print(f"Model Type: {config.model_type}")
    print(f"Experiment: {args.exp_name}")
    print(f"Model Path: {model_path}")
    print(f"Output Directory: {args.output_dir}")
    print(f"Output File: {args.output_name}")
    print(f"Device: {args.device}")
    print(f"ONNX Opset Version: {args.opset_version}")
    print(f"Do Constant Folding: {args.do_constant_folding}")
    print(f"{'='*60}\n")
    
    # 加载模型
    if args.verbose:
        print(f"Loading model...")
    
    model = config.load_model_for_eval(model_path, device=args.device)
    
    # 获取模型的策略，导出到ONNX
    if args.verbose:
        print(f"Exporting to ONNX format...\n")
    
    try:
        # 导出模型
        output_path = os.path.join(args.output_dir, args.output_name)
        
        # 使用SB3的导出功能
        export_model_to_onnx(
            model=model,
            output_path=output_path,
            opset_version=args.opset_version,
            use_external_data_format=args.use_external_data_format,
            do_constant_folding=args.do_constant_folding,
            verbose=args.verbose,
        )
        
        print(f"\n{'='*60}")
        # 检查实际导出的文件
        pytorch_path = output_path.replace('.onnx', '.pt')
        full_model_path = output_path.replace('.onnx', '_full.pt')
        info_path = output_path.replace('.onnx', '_info.txt')
        
        exported_files = []
        if os.path.exists(pytorch_path):
            size_mb = os.path.getsize(pytorch_path) / 1024 / 1024
            exported_files.append(f"  - {pytorch_path} ({size_mb:.2f} MB)")
        
        if os.path.exists(full_model_path):
            size_mb = os.path.getsize(full_model_path) / 1024 / 1024
            exported_files.append(f"  - {full_model_path} ({size_mb:.2f} MB)")
        
        if os.path.exists(info_path):
            exported_files.append(f"  - {info_path} (info file)")
        
        if os.path.exists(output_path):
            size_mb = os.path.getsize(output_path) / 1024 / 1024
            exported_files.append(f"  - {output_path} ({size_mb:.2f} MB)")
        
        if exported_files:
            print(f"✓ Successfully exported files:")
            for f in exported_files:
                print(f)
        else:
            print(f"✗ No files were exported")
        
        print(f"{'='*60}")
        
    except Exception as e:
        print(f"\n✗ Error during export: {e}")
        import traceback
        traceback.print_exc()
        return


def export_model_to_onnx(
    model,
    output_path: str,
    opset_version: int = 14,
    use_external_data_format: bool = False,
    do_constant_folding: bool = True,
    verbose: bool = True,
) -> None:
    """
    将SB3模型导出为ONNX格式
    
    注意：由于ONNX导出工具链的依赖版本冲突，本脚本提供两种导出方式：
    1. 直接导出PolicyNetwork为PyTorch模型（推荐）
    2. 导出为ONNX格式（需要解决依赖问题）
    
    Args:
        model: SB3模型对象（PPO或SAC）
        output_path: 输出ONNX文件路径
        opset_version: ONNX opset版本
        use_external_data_format: 是否使用外部数据格式（用于大模型）
        do_constant_folding: 是否执行常数折叠优化
        verbose: 是否打印详细信息
    """
    # 获取政策网络
    policy = model.policy
    policy.eval()
    
    # 创建虚拟输入以获取模型结构信息
    obs_shape = model.observation_space.shape[0]
    dummy_input = torch.randn(1, obs_shape)
    
    if verbose:
        print(f"  Input shape: {dummy_input.shape}")
        print(f"  Observation space: {model.observation_space}")
        print(f"  Action space: {model.action_space}")
    
    try:
        if verbose:
            print(f"\n  Method 1: Trying ONNX export with OnnxablePolicy...")
        
        try:
            import onnx
            
            features_extractor = policy.features_extractor
            mlp_extractor = policy.mlp_extractor
            action_net = policy.action_net
            value_net = policy.value_net
            
            onnxable_model = OnnxablePolicy(features_extractor, mlp_extractor, action_net, value_net)
            onnxable_model.eval()
            
            with torch.no_grad():
                torch.onnx.export(
                    onnxable_model,
                    dummy_input,
                    output_path,
                    input_names=['observations'],
                    output_names=['actions', 'values'],
                    opset_version=opset_version,
                    do_constant_folding=do_constant_folding,
                    verbose=False,
                    export_params=True,
                )
            
            if verbose:
                print(f"  ✓ ONNX export succeeded")
            
            try:
                onnx_model = onnx.load(output_path)
                onnx.checker.check_model(onnx_model)
                if verbose:
                    print(f"  ✓ ONNX model is valid")
            except Exception as e:
                print(f"  ⚠ ONNX validation: {e}")
                
        except Exception as onnx_error:
            raise onnx_error  # 直接抛出异常
            # if verbose:
            #     print(f"  ⚠ ONNX export failed: {onnx_error}")
            #     print(f"\n  Method 2: Falling back to PyTorch model export...")
            
            # # 备选方案：导出为PyTorch模型
            # pytorch_path = output_path.replace('.onnx', '.pt')
            # torch.save(inference_wrapper.state_dict(), pytorch_path)
            
            # if verbose:
            #     print(f"  ✓ PyTorch model saved to: {pytorch_path}")
            #     print(f"\n  Note: To convert PyTorch to ONNX manually, use:")
            #     print(f"    torch.onnx.export(...)")
            
            # # 也保存完整模型
            # torch.save({
            #     'model': inference_wrapper,
            #     'observation_space': model.observation_space,
            #     'action_space': model.action_space,
            # }, pytorch_path.replace('.pt', '_full.pt'))
            
            # # 创建一个dummy ONNX以提示用户
            # create_dummy_onnx_info(output_path, pytorch_path)
        
        if verbose:
            print(f"  ✓ Model exported successfully")

        _write_success_export_info(model, output_path, obs_shape)
            
    except Exception as e:
        print(f"  ✗ Error exporting model: {e}")
        import traceback
        traceback.print_exc()
        raise


def create_dummy_onnx_info(onnx_path: str, pytorch_path: str) -> None:
    """
    创建一个信息文件，说明如何手动转换ONNX
    
    Args:
        onnx_path: ONNX文件路径
        pytorch_path: PyTorch模型文件路径
    """
    info_path = onnx_path.replace('.onnx', '_info.txt')
    with open(info_path, 'w', encoding='utf-8') as f:
        f.write(f"""ONNX Export Information
======================

Due to dependency version conflicts, ONNX export was not possible.
Instead, the PyTorch model has been saved to: {pytorch_path}

To convert this PyTorch model to ONNX format, you can use:

```python
import torch
import onnx

# Load the model
checkpoint = torch.load('{pytorch_path.replace('.pt', '_full.pt')}')
model = checkpoint['model']

# Create dummy input
dummy_input = torch.randn(1, 20)  # Adjust input size as needed

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    '{onnx_path}',
    input_names=['observations'],
    output_names=['actions'],
    opset_version=14,
    do_constant_folding=True,
    verbose=False,
    export_params=True,
)
```

PyTorch Model Usage:
====================
```python
import torch

# Load model
checkpoint = torch.load('{pytorch_path.replace('.pt', '_full.pt')}')
model = checkpoint['model']
obs_space = checkpoint['observation_space']
act_space = checkpoint['action_space']

# Inference
model.eval()
with torch.no_grad():
    observations = torch.randn(1, 20)  # Your observations
    actions = model(observations)
```
""")


def main(args: ExportArgs) -> None:
    """主导出函数"""
    export_to_onnx(args)


if __name__ == "__main__":
    args = tyro.cli(ExportArgs)
    main(args)
