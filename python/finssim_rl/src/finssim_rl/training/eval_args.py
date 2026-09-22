"""
评估脚本的命令行参数
"""
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class EvalArgs:
    """评估模型的命令行参数"""
    
    # 实验配置
    config: str = field(
        metadata={"help": "Configuration name (e.g., sac_default, ppo_chase)"}
    )
    
    # 评估参数
    exp_name: str = field(
        default="",
        metadata={"help": "Experiment name to evaluate (if empty, uses config name)"}
    )
    
    # 评估参数
    num_episodes: int = field(
        default=5,
        metadata={"help": "Number of episodes to run"}
    )

    force_zero_thrust: bool = field(
        default=False,
        metadata={"help": "Force action to all-zero at every step for baseline test"}
    )
    control_debug_interval: int = field(
        default=0,
        metadata={"help": "Print static-controller diagnostics every N evaluation steps (0 disables)"}
    )
    
    # 路径配置
    checkpoint_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the checkpoint file to evaluate"}
    )
    output_dir: Optional[str] = field(
        default=None,
        metadata={"help": "FinsSim artifact run directory"}
    )
    resolved_config: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the resolved FinsSim config YAML"}
    )
    env_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to Unity environment executable"}
    )
    num_envs: Optional[int] = field(default=None, metadata={"help": "Number of evaluation vector slots"})
    parallel_mode: Optional[str] = field(default=None, metadata={"help": "multi_area or multi_binary"})
    use_editor: bool = field(
        default=False,
        metadata={"help": "Connect to a running Unity Editor Play session instead of launching a binary"}
    )
    port_offset: Optional[int] = field(
        default=None,
        metadata={"help": "Port offset / worker id for Unity evaluation environment"}
    )
    env_base_port: Optional[int] = field(
        default=None,
        metadata={"help": "Base port for Unity evaluation environment"}
    )
    time_scale: Optional[float] = field(
        default=None,
        metadata={"help": "Unity evaluation time scale"}
    )
    timeout_wait: Optional[int] = field(
        default=None,
        metadata={"help": "Seconds to wait for Unity ML-Agents startup handshake"}
    )
    seed: Optional[int] = field(
        default=None,
        metadata={"help": "Seed passed to Unity domain randomization"}
    )
    unity_additional_args: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Additional command-line arguments passed through to the Unity player"}
    )
    unity_additional_args_json: Optional[str] = field(
        default=None,
        metadata={"help": "JSON array of additional command-line arguments passed through to the Unity player"}
    )
    
    # 渲染和输出
    no_graphics: bool = field(
        default=True,
        metadata={"help": "Disable graphics/rendering"}
    )
    show_graphics: bool = field(
        default=False,
        metadata={"help": "Show the Unity graphics window during evaluation"}
    )
    render: bool = field(
        default=True,
        metadata={"help": "Enable rendering (ignored if no_graphics=True)"}
    )
    window_width: int = field(
        default=1280,
        metadata={"help": "Unity player window width when graphics are shown"}
    )
    window_height: int = field(
        default=720,
        metadata={"help": "Unity player window height when graphics are shown"}
    )
    
    # 设备配置
    device: str = field(
        default="auto",
        metadata={"help": "Device to use (cuda, cpu, or auto)"}
    )
    
    # 输出选项
    save_video: bool = field(
        default=False,
        metadata={"help": "Save evaluation video"}
    )
    video_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to save video (auto-generated if not specified)"}
    )
    
    def __post_init__(self):
        """初始化后处理"""
        self.control_debug_interval = max(0, int(self.control_debug_interval))
        # 如果没有指定 exp_name，使用 config 名称
        if not self.exp_name:
            self.exp_name = self.config
        
        # 生成默认的视频保存路径
        if self.save_video and self.video_path is None:
            self.video_path = f"./videos/{self.exp_name}_eval.mp4"
        
        # 如果指定了 no_graphics，覆盖 render 参数
        if self.show_graphics:
            self.no_graphics = False
            self.render = True
        elif self.no_graphics:
            self.render = False
