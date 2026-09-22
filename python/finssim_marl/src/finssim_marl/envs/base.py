"""
Base environment interface for MARL.
All environment wrappers should implement this interface.
"""

from abc import ABC, abstractmethod
import numpy as np


class BaseEnvSpec(ABC):
    """Abstract base class for environment specifications.

    All MARL environments should implement this interface to ensure
    compatibility with the training runner.
    """

    @property
    @abstractmethod
    def n_agents(self) -> int:
        """Number of agents in the environment."""
        pass

    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Observation dimension per agent."""
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action dimension per agent."""
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        """Global state dimension."""
        pass

    @property
    @abstractmethod
    def role_ids(self) -> np.ndarray:
        """Role ID for each agent as numpy array."""
        pass

    @property
    def max_steps(self) -> int:
        """Maximum steps per episode. Override if needed."""
        return 900

    @abstractmethod
    def reset(self, seed=None):
        """Reset environment.

        Returns:
            obs: Observation array [n_agents, obs_dim]
        """
        pass

    @abstractmethod
    def step(self, actions):
        """Execute actions.

        Args:
            actions: Action array [n_agents, action_dim]

        Returns:
            obs: Next observation [n_agents, obs_dim]
            reward: Scalar reward
            done: Boolean, all agents done
            truncated: Boolean, episode truncated
            info: Dict with environment-specific info
        """
        pass

    @abstractmethod
    def close(self):
        """Clean up environment resources."""
        pass

    def get_obs_size(self) -> int:
        """Get observation dimension (alias for obs_dim for compatibility)."""
        return self.obs_dim

    def get_action_size(self) -> int:
        """Get action dimension (alias for action_dim for compatibility)."""
        return self.action_dim

    def get_state_size(self) -> int:
        """Get state dimension (alias for state_dim for compatibility)."""
        return self.state_dim

    def get_state(self):
        """Get global state. Override if needed."""
        raise NotImplementedError("get_state not implemented for this environment")
