"""
导出脚本的命令行参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ExportArgs:
    """导出模型为ONNX格式的命令行参数"""
    
    # 实验配置
    config: str = field(
        metadata={"help": "Configuration name (e.g., sac_default, ppo_chase)"}
    )
    
    # 导出参数
    exp_name: str = field(
        default="",
        metadata={"help": "Experiment name to export (if empty, uses config name)"}
    )
    
    # 模型选择
    model_type: str = field(
        default="best",
        metadata={"help": "Which model to load: 'best', 'latest', or 'final'"}
    )
    
    # 输出路径
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Output directory for ONNX file (auto-generated if not specified)"}
    )
    
    output_name: str = field(
        default="model.onnx",
        metadata={"help": "Output filename for ONNX model"}
    )
    
    # 检查点目录
    checkpoint_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Checkpoint directory (auto-generated if not specified)"}
    )
    
    # 设备配置
    device: str = field(
        default="cpu",
        metadata={"help": "Device to use for export (cuda or cpu)"}
    )
    
    # 导出选项
    use_external_data_format: bool = field(
        default=False,
        metadata={"help": "Use external data format for large models"}
    )
    
    opset_version: int = field(
        default=14,
        metadata={"help": "ONNX opset version"}
    )
    
    do_constant_folding: bool = field(
        default=True,
        metadata={"help": "Execute constant folding optimization"}
    )
    
    verbose: bool = field(
        default=True,
        metadata={"help": "Print detailed export information"}
    )
    
    def __post_init__(self):
        """初始化后处理：生成默认的输出路径"""
        if self.checkpoint_dir is None:
            self.checkpoint_dir = f"./checkpoints/{self.exp_name or self.config}/"
        if self.output_dir is None:
            self.output_dir = f"./models_onnx/{self.exp_name or self.config}/"
