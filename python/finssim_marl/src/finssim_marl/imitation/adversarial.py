"""Modern MAGAIL/MA-AIRL building blocks for FinsSim.

The original ermongroup implementations are TensorFlow 1.x research code. This
module is the PyTorch/TorchRL landing zone for migrated discriminator and reward
model pieces.
"""

from __future__ import annotations

import torch
from torch import nn


class JointStateActionDiscriminator(nn.Module):
    """Simple discriminator over global state and joint action."""

    def __init__(self, state_dim: int, n_agents: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = state_dim + n_agents * action_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        flat_action = action.flatten(start_dim=-2)
        return self.net(torch.cat([state, flat_action], dim=-1))
