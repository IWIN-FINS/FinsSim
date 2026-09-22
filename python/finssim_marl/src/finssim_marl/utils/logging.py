"""
Unified logging utilities for MARL training.
Supports both TensorBoard and Weights & Biases.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter


@dataclass
class LoggerConfig:
    """Configuration for logger."""
    log_dir: str = "runs"
    use_wandb: bool = False
    wandb_project: str = ""
    wandb_entity: str = ""
    run_name: str = ""
    tensorboard_dir: str | None = None


class Logger:
    """Unified logging interface for MARL training.

    Supports TensorBoard and optionally Weights & Biases.
    Metrics are logged via add_scalar() and add_histogram().
    """

    def __init__(self, config: LoggerConfig):
        self.config = config
        tensorboard_dir = Path(config.tensorboard_dir) if config.tensorboard_dir else Path(config.log_dir) / config.run_name
        self.writer = SummaryWriter(str(tensorboard_dir))

        if config.use_wandb:
            import wandb
            wandb.init(
                project=config.wandb_project,
                entity=config.wandb_entity,
                sync_tensorboard=True,
                name=config.run_name,
                id=config.run_name,
            )

    def log(self, metrics: dict, step: int, prefix: str = ""):
        """Log a dictionary of scalar metrics.

        Args:
            metrics: Dictionary of metric name -> value
            step: Global training step
            prefix: Optional prefix for all metric names (e.g., "train/", "eval/")
        """
        for key, value in metrics.items():
            self.writer.add_scalar(prefix + key, value, step)

    def log_histogram(self, key: str, values, step: int, prefix: str = ""):
        """Log a histogram of values.

        Args:
            key: Metric name
            values: Array of values
            step: Global training step
            prefix: Optional prefix
        """
        self.writer.add_histogram(prefix + key, values, step)

    def log_text(self, key: str, text: str, step: int):
        """Log text information.

        Args:
            key: Metric name
            text: Text to log
            step: Global training step
        """
        self.writer.add_text(key, text, step)

    def log_hyperparams(self, params: dict, step: int = 0):
        """Log hyperparameters as text table.

        Args:
            params: Dictionary of parameter name -> value
            step: Global training step
        """
        text = "|param|value|\n|-|-|\n" + "\n".join(
            [f"|{key}|{value}|" for key, value in params.items()]
        )
        self.writer.add_text("hyperparameters", text, step)

    def close(self):
        """Close the logger and any external integrations."""
        self.writer.close()
        if self.config.use_wandb:
            import wandb
            wandb.finish()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
