"""
Base algorithm interface for MARL.
All algorithms should implement this interface.
"""

from abc import ABC, abstractmethod
import torch


class BaseAlgorithm(ABC):
    """Abstract base class for MARL algorithms.

    Defines the standard interface that all algorithms must implement.
    The algorithm does NOT manage its own train loop - that is handled
    by the external Runner. This makes algorithms interchangeable and testable.
    """

    @abstractmethod
    def select_action(self, obs, **kwargs):
        """Select actions for given observations (for inference/rollout).

        Args:
            obs: Observations
            **kwargs: Additional algorithm-specific arguments

        Returns:
            actions: Selected actions
            log_probs: Log probabilities (if applicable)
            extra: Additional outputs (if applicable)
        """
        pass

    @abstractmethod
    def update(self, batch):
        """Update algorithm with a batch of rollout data.

        Args:
            batch: Dictionary containing rollout data

        Returns:
            metrics: Dictionary of training metrics
        """
        pass

    @abstractmethod
    def save(self, path: str):
        """Save model checkpoint.

        Args:
            path: Path to save checkpoint
        """
        pass

    @abstractmethod
    def load(self, path: str):
        """Load model checkpoint.

        Args:
            path: Path to load checkpoint from
        """
        pass

    def train(self):
        """Set algorithm to training mode."""
        pass

    def eval(self):
        """Set algorithm to evaluation mode."""
        pass

    def get_experiment_metadata(self) -> dict:
        """Return static metadata that should be logged once per experiment."""
        return {}
