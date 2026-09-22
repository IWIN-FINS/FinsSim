"""Permutation-aware actor and privileged critics for TriNetCapture."""

from __future__ import annotations

import torch
import torch.nn as nn

from .actors import (
    _estimate_squashed_normal_entropy,
    _sample_squashed_normal,
    _squashed_normal_log_prob,
)


TRINET_ACTOR_OBS_DIM = 34
TRINET_ROV_TOKEN_DIM = 15
TRINET_TASK_CONTEXT_DIM = 12
TRINET_PRIVILEGED_STATE_DIM = 57
TRINET_NUM_ROVS = 3

TRINET_CRITIC_VARIANT_IDS = {
    "trinet_deepsets": 10,
    "trinet_attention": 11,
    "trinet_flat_mlp": 12,
}


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int | None = None) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
        previous = hidden
    if output_dim is not None:
        layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class TriNetSharedPositionActor(nn.Module):
    """One shared 4D Gaussian policy with mean-pooled teammate features."""

    SELF_TASK_DIM = 22
    TEAMMATE_START = 14
    TEAMMATE_DIM = 6
    NUM_TEAMMATES = 2

    def __init__(self, action_dim: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.teammate_encoder = _mlp(self.TEAMMATE_DIM, (64, 64))
        self.policy_body = _mlp(self.SELF_TASK_DIM + 64, (128, 128))
        self.mean_layer = nn.Linear(128, action_dim)
        self.log_std = nn.Parameter(torch.zeros(action_dim))

    def _features(self, obs: torch.Tensor) -> torch.Tensor:
        if obs.ndim != 3 or obs.size(-1) != TRINET_ACTOR_OBS_DIM:
            raise ValueError(f"Expected [batch, 3, {TRINET_ACTOR_OBS_DIM}] observation, got {tuple(obs.shape)}")
        self_target = obs[..., : self.TEAMMATE_START]
        teammates = obs[..., self.TEAMMATE_START : self.TEAMMATE_START + 12].reshape(
            *obs.shape[:2], self.NUM_TEAMMATES, self.TEAMMATE_DIM
        )
        goal_and_net = obs[..., 26:]
        self_task = torch.cat((self_target, goal_and_net), dim=-1)
        teammate_features = self.teammate_encoder(teammates).mean(dim=-2)
        return self.policy_body(torch.cat((self_task, teammate_features), dim=-1))

    def distribution(self, obs: torch.Tensor) -> torch.distributions.Normal:
        mean = self.mean_layer(self._features(obs))
        return torch.distributions.Normal(mean, torch.exp(self.log_std).expand_as(mean))

    def act(self, obs: torch.Tensor, *, deterministic: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return _sample_squashed_normal(self.distribution(obs), deterministic=deterministic)

    def get_log_prob(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        return _squashed_normal_log_prob(self.distribution(obs), actions)

    def get_entropy(self, obs: torch.Tensor) -> torch.Tensor:
        return _estimate_squashed_normal_entropy(self.distribution(obs))


class _TriNetDualValueCritic(nn.Module):
    """Shared dual-head interface: one team value and three local values."""

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.ndim != 2 or state.size(-1) != TRINET_PRIVILEGED_STATE_DIM:
            raise ValueError(f"Expected [batch, {TRINET_PRIVILEGED_STATE_DIM}] critic state, got {tuple(state.shape)}")
        rov_tokens = state[:, :45].reshape(-1, TRINET_NUM_ROVS, TRINET_ROV_TOKEN_DIM)
        return rov_tokens, state[:, 45:]

    @staticmethod
    def _dual_values(global_feature: torch.Tensor, rov_features: torch.Tensor, team_head: nn.Module, individual_head: nn.Module):
        team_value = team_head(global_feature).squeeze(-1)
        repeated_global = global_feature.unsqueeze(1).expand(-1, TRINET_NUM_ROVS, -1)
        individual_value = individual_head(torch.cat((repeated_global, rov_features), dim=-1)).squeeze(-1)
        return team_value, individual_value


class TriNetDeepSetsCritic(_TriNetDualValueCritic):
    """Default permutation-invariant centralized critic."""

    def __init__(self) -> None:
        super().__init__()
        self.rov_encoder = _mlp(TRINET_ROV_TOKEN_DIM, (128, 128))
        self.task_encoder = _mlp(TRINET_TASK_CONTEXT_DIM, (64,))
        self.global_encoder = _mlp(192, (128,))
        self.team_head = _mlp(128, (64,), 1)
        self.individual_head = _mlp(256, (64,), 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rov_tokens, task_context = self._split_state(state)
        rov_features = self.rov_encoder(rov_tokens)
        global_feature = self.global_encoder(torch.cat((rov_features.mean(dim=1), self.task_encoder(task_context)), dim=-1))
        return self._dual_values(global_feature, rov_features, self.team_head, self.individual_head)


class TriNetAttentionCritic(_TriNetDualValueCritic):
    """Position-free attention ablation; mean pooling preserves permutation invariance."""

    def __init__(self, attention_heads: int = 4) -> None:
        super().__init__()
        if 128 % attention_heads != 0:
            raise ValueError("TriNet attention heads must divide the 128D token embedding.")
        self.rov_encoder = _mlp(TRINET_ROV_TOKEN_DIM, (128, 128))
        layer = nn.TransformerEncoderLayer(
            d_model=128,
            nhead=attention_heads,
            dim_feedforward=256,
            dropout=0.0,
            activation="relu",
            batch_first=True,
        )
        self.attention = nn.TransformerEncoder(layer, num_layers=2)
        self.task_encoder = _mlp(TRINET_TASK_CONTEXT_DIM, (64,))
        self.global_encoder = _mlp(192, (128,))
        self.team_head = _mlp(128, (64,), 1)
        self.individual_head = _mlp(256, (64,), 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rov_tokens, task_context = self._split_state(state)
        rov_features = self.attention(self.rov_encoder(rov_tokens))
        global_feature = self.global_encoder(torch.cat((rov_features.mean(dim=1), self.task_encoder(task_context)), dim=-1))
        return self._dual_values(global_feature, rov_features, self.team_head, self.individual_head)


class TriNetFlatMLPCritic(_TriNetDualValueCritic):
    """Order-sensitive flattened-state ablation with no role-specific parameters."""

    def __init__(self) -> None:
        super().__init__()
        self.global_encoder = _mlp(TRINET_PRIVILEGED_STATE_DIM, (256, 128))
        self.rov_encoder = _mlp(TRINET_ROV_TOKEN_DIM, (128,))
        self.team_head = _mlp(128, (64,), 1)
        self.individual_head = _mlp(256, (64,), 1)

    def forward(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        rov_tokens, _ = self._split_state(state)
        rov_features = self.rov_encoder(rov_tokens)
        global_feature = self.global_encoder(state)
        return self._dual_values(global_feature, rov_features, self.team_head, self.individual_head)


def normalize_trinet_critic_variant(variant: str) -> str:
    normalized = variant.lower().strip()
    aliases = {
        "default": "trinet_deepsets",
        "deepsets": "trinet_deepsets",
        "attention": "trinet_attention",
        "flat": "trinet_flat_mlp",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in TRINET_CRITIC_VARIANT_IDS:
        raise ValueError(f"Unknown TriNet critic variant {variant!r}. Expected one of {sorted(TRINET_CRITIC_VARIANT_IDS)}")
    return normalized


def build_trinet_critic(variant: str, *, attention_heads: int = 4) -> _TriNetDualValueCritic:
    normalized = normalize_trinet_critic_variant(variant)
    if normalized == "trinet_deepsets":
        return TriNetDeepSetsCritic()
    if normalized == "trinet_attention":
        return TriNetAttentionCritic(attention_heads=attention_heads)
    return TriNetFlatMLPCritic()


__all__ = [
    "TRINET_ACTOR_OBS_DIM",
    "TRINET_PRIVILEGED_STATE_DIM",
    "TRINET_CRITIC_VARIANT_IDS",
    "TriNetSharedPositionActor",
    "TriNetDeepSetsCritic",
    "TriNetAttentionCritic",
    "TriNetFlatMLPCritic",
    "build_trinet_critic",
    "normalize_trinet_critic_variant",
]
