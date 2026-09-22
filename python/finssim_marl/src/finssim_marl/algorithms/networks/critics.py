"""
Critic networks for MARL.
Contains standard Critic and MultiHead Critic for heterogeneous agents.
"""

import torch
import torch.nn as nn

CHASER_LOCAL_OBS_DIM = 30
PREY_OBS_DIM = 15
NUM_TOTAL_ROLES = 3
PREY_ROLE_ID = 2
EPS = 1e-6

CRITIC_VARIANT_IDS = {
    "critic_multihead": 0,
    "token_attention": 2,
    "geometric": 3,
    "flat_state": 4,
}


def _build_mlp(
    input_dim: int,
    hidden_dim: int,
    num_layers: int,
    output_dim: int | None = None,
) -> nn.Sequential:
    layers: list[nn.Module] = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
    for _ in range(max(0, num_layers - 1)):
        layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
    if output_dim is not None:
        layers.append(nn.Linear(hidden_dim, output_dim))
    return nn.Sequential(*layers)


def _build_role_onehot(role_ids: torch.Tensor, num_roles: int) -> torch.Tensor:
    role_ids = role_ids.long()
    onehot = torch.zeros(*role_ids.shape, num_roles, device=role_ids.device)
    onehot.scatter_(-1, role_ids.unsqueeze(-1), 1.0)
    return onehot


def _role_id_template(batch_size: int, role_ids: torch.Tensor) -> torch.Tensor:
    return role_ids.unsqueeze(0).expand(batch_size, -1)


def _vector_norm(x: torch.Tensor) -> torch.Tensor:
    return torch.linalg.vector_norm(x, dim=-1, keepdim=True)


def _closing_speed(relative_position: torch.Tensor, relative_velocity: torch.Tensor) -> torch.Tensor:
    distance = _vector_norm(relative_position).clamp_min(EPS)
    radial_rate = (relative_position * relative_velocity).sum(dim=-1, keepdim=True) / distance
    return -radial_rate


def normalize_critic_variant_name(critic_variant: str) -> str:
    variant = critic_variant.lower()
    alias_map = {
        "0": "critic_multihead",
        "critic_multihead": "critic_multihead",
        "multihead": "critic_multihead",
        "multi_head": "critic_multihead",
        "legacy": "critic_multihead",
        "legacy_local": "critic_multihead",
        "local": "critic_multihead",
        "default": "critic_multihead",
        "2": "token_attention",
        "scheme2": "token_attention",
        "attention": "token_attention",
        "3": "geometric",
        "scheme3": "geometric",
        "geometry": "geometric",
        "4": "flat_state",
        "scheme4": "flat_state",
        "flat": "flat_state",
    }
    return alias_map.get(variant, variant)


def critic_variant_to_id(critic_variant: str) -> int:
    return CRITIC_VARIANT_IDS.get(normalize_critic_variant_name(critic_variant), -1)


class Critic(nn.Module):
    """Standard Critic network for value function estimation."""

    def __init__(self, input_dim, hidden_dim, num_layer) -> None:
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.layers.append(nn.Sequential(nn.Linear(hidden_dim, 1)))

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class CriticMultiHead(nn.Module):
    """Centralized critic over ordered chaser-team observations.

    The critic first encodes the ordered chaser team `[Herder, Netter1, Netter2]`
    into a shared global context. Each agent observation is also encoded locally.
    The final value head for each role consumes `[global_context, local_feature]`.
    """

    def __init__(
        self,
        obs_dim,
        hidden_dim,
        num_layers,
        num_roles=2,
        agent_id_dim=None,
        num_trainable_agents: int = 3,
    ):
        super().__init__()
        del agent_id_dim  # Kept in the signature for backward compatibility.
        self.num_roles = num_roles
        self.obs_dim = obs_dim
        self.num_trainable_agents = num_trainable_agents

        team_input_dim = obs_dim * num_trainable_agents
        self.team_encoder = _build_mlp(team_input_dim, hidden_dim, num_layers)
        self.local_encoder = _build_mlp(obs_dim, hidden_dim, num_layers)

        self.role_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.role_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1)
                )
            )

    def forward(self, obs, role_ids, full_obs=None):
        """Get value estimates for all agents based on their roles.

        Args:
            obs: [batch_size, num_agents, obs_dim] ordered chaser observations
            role_ids: [batch_size, num_agents] Role index for each agent
        Returns:
            values: [batch_size, num_agents] Value estimates for each agent
        """
        del full_obs
        batch_size, num_agents, _ = obs.shape
        if num_agents != self.num_trainable_agents:
            raise ValueError(
                "CriticMultiHead expects ordered chaser-team observations with "
                f"num_agents={self.num_trainable_agents}, got num_agents={num_agents}."
            )
        if obs.size(-1) != self.obs_dim:
            raise ValueError(
                f"CriticMultiHead expects obs_dim={self.obs_dim}, got obs_dim={obs.size(-1)}."
            )

        team_features = self.team_encoder(obs.reshape(batch_size, -1))
        local_features = self.local_encoder(obs.reshape(batch_size * num_agents, self.obs_dim)).reshape(
            batch_size, num_agents, -1
        )
        global_features = team_features.unsqueeze(1).expand(-1, num_agents, -1)
        fused_features = torch.cat([global_features, local_features], dim=-1)

        values = torch.zeros(batch_size, num_agents, device=obs.device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = fused_features[mask]
                role_values = self.role_heads[role_idx](role_features).squeeze(-1)
                values[mask] = role_values

        return values


class _AttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.attn(x, x, x, need_weights=False)
        x = self.norm1(x + attn_out)
        ff_out = self.ff(x)
        return self.norm2(x + ff_out)


class AttentionCentralCritic(nn.Module):
    """Scheme 2: token-based self-attention critic over Herder/Netters/Prey.

    Chaser tokens consume only the 30D actor-visible prefix. The prey token
    consumes only its true 15D observation before being projected to the shared
    attention hidden size.
    """

    def __init__(
        self,
        full_obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_roles: int = 2,
        total_roles: int = NUM_TOTAL_ROLES,
        attention_heads: int = 4,
    ):
        super().__init__()
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                f"critic_hidden_dim={hidden_dim} must be divisible by critic_attention_heads={attention_heads}."
            )
        self.num_roles = num_roles
        self.total_roles = total_roles
        self.full_obs_dim = full_obs_dim
        self.chaser_token_encoder = _build_mlp(CHASER_LOCAL_OBS_DIM + total_roles, hidden_dim, 1)
        self.prey_token_encoder = _build_mlp(PREY_OBS_DIM + total_roles, hidden_dim, 1)
        self.blocks = nn.ModuleList(
            [_AttentionBlock(embed_dim=hidden_dim, num_heads=attention_heads) for _ in range(max(1, num_layers))]
        )
        self.role_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_roles)
            ]
        )

    def forward(self, obs, role_ids, full_obs=None):
        del obs # todo: Seek attention MARL architecture paper! 
        if full_obs is None:
            raise ValueError("AttentionCentralCritic requires full_obs in canonical [H, N1, N2, Prey] order.")
        if full_obs.size(1) != 4:
            raise ValueError(
                "AttentionCentralCritic expects canonical tokens [Herder, Netter1, Netter2, Prey], "
                f"got num_tokens={full_obs.size(1)}."
            )
        if full_obs.size(-1) < CHASER_LOCAL_OBS_DIM:
            raise ValueError(
                f"AttentionCentralCritic expected token dim >= {CHASER_LOCAL_OBS_DIM}, got {full_obs.size(-1)}."
            )

        batch_size = full_obs.size(0)
        prey_role = torch.full((batch_size, 1), PREY_ROLE_ID, dtype=role_ids.dtype, device=role_ids.device)
        full_role_ids = torch.cat([role_ids, prey_role], dim=1)
        role_onehot = _build_role_onehot(full_role_ids, self.total_roles)

        chaser_obs = full_obs[:, : role_ids.size(1), :CHASER_LOCAL_OBS_DIM]
        prey_obs = full_obs[:, role_ids.size(1) :, :PREY_OBS_DIM]
        chaser_role_onehot = role_onehot[:, : role_ids.size(1), :]
        prey_role_onehot = role_onehot[:, role_ids.size(1) :, :]

        chaser_tokens = self.chaser_token_encoder(torch.cat([chaser_obs, chaser_role_onehot], dim=-1))
        prey_tokens = self.prey_token_encoder(torch.cat([prey_obs, prey_role_onehot], dim=-1))
        tokens = torch.cat([chaser_tokens, prey_tokens], dim=1)
        for block in self.blocks:
            tokens = block(tokens)

        chaser_tokens = tokens[:, : role_ids.size(1), :]
        values = torch.zeros(batch_size, role_ids.size(1), device=full_obs.device)
        for role_idx in range(self.num_roles):
            mask = role_ids == role_idx
            if mask.any():
                values[mask] = self.role_heads[role_idx](chaser_tokens[mask]).squeeze(-1)
        return values


class GeometricCentralCritic(nn.Module):
    """Scheme 3: engineered geometry features with a standard neural critic head."""

    def __init__(
        self,
        local_obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_roles: int = 2,
        agent_id_dim: int = NUM_TOTAL_ROLES,
    ):
        super().__init__()
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim
        self.local_body = _build_mlp(local_obs_dim + agent_id_dim, hidden_dim, num_layers)
        self.global_body = _build_mlp(20, hidden_dim, num_layers)
        self.role_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                )
                for _ in range(num_roles)
            ]
        )

    def _build_geometry_features(self, local_obs: torch.Tensor, full_obs: torch.Tensor) -> torch.Tensor:
        herder_obs = local_obs[:, 0, :]
        netter1_obs = local_obs[:, 1, :]
        netter2_obs = local_obs[:, 2, :]
        prey_obs = full_obs[:, 3, :PREY_OBS_DIM]

        fish_pos_h = herder_obs[:, 6:9]
        fish_vel_h = herder_obs[:, 9:12]
        fish_pos_n1 = netter1_obs[:, 6:9]
        fish_vel_n1 = netter1_obs[:, 9:12]
        fish_pos_n2 = netter2_obs[:, 6:9]
        fish_vel_n2 = netter2_obs[:, 9:12]

        dist_hf = _vector_norm(fish_pos_h)
        dist_n1f = _vector_norm(fish_pos_n1)
        dist_n2f = _vector_norm(fish_pos_n2)

        # 30D chaser actor observation layout:
        # [0:3]   self_linear_velocity_body
        # [3:6]   self_angular_velocity_body
        # [6:9]   fish_relative_position_body
        # [9:12]  fish_relative_velocity_body
        # [12]    distance_to_fish
        # [13]    bearing_to_fish
        # [14:22] teammate_slot_a = onehot(2) + rel_pos(3) + rel_vel(3)
        # [22:30] teammate_slot_b = onehot(2) + rel_pos(3) + rel_vel(3)
        teammate_a_rel_pos = slice(16, 19)
        teammate_b_rel_pos = slice(24, 27)

        features = [
            dist_hf,
            dist_n1f,
            dist_n2f,
            _closing_speed(fish_pos_h, fish_vel_h),
            _closing_speed(fish_pos_n1, fish_vel_n1),
            _closing_speed(fish_pos_n2, fish_vel_n2),
            _vector_norm(herder_obs[:, 0:3]),
            _vector_norm(netter1_obs[:, 0:3]),
            _vector_norm(netter2_obs[:, 0:3]),
            _vector_norm(herder_obs[:, 3:6]),
            _vector_norm(netter1_obs[:, 3:6]),
            _vector_norm(netter2_obs[:, 3:6]),
            _vector_norm(herder_obs[:, teammate_a_rel_pos]),
            _vector_norm(herder_obs[:, teammate_b_rel_pos]),
            _vector_norm(netter1_obs[:, teammate_b_rel_pos]),
            _vector_norm(prey_obs[:, 3:6]),
            (dist_hf + dist_n1f + dist_n2f) / 3.0,
            torch.minimum(dist_n1f, dist_n2f),
            (_vector_norm(netter1_obs[:, 0:3]) + _vector_norm(netter2_obs[:, 0:3])) / 2.0,
            (dist_n1f - dist_n2f).abs(),
        ]
        return torch.cat(features, dim=-1)

    def forward(self, obs, role_ids, full_obs=None):
        if full_obs is None:
            raise ValueError("GeometricCentralCritic requires full_obs in canonical [H, N1, N2, Prey] order.")

        batch_size, num_agents, _ = obs.shape
        role_onehot = _build_role_onehot(role_ids, self.agent_id_dim)
        local_features = self.local_body(
            torch.cat([obs, role_onehot], dim=-1).reshape(batch_size * num_agents, -1)
        ).reshape(batch_size, num_agents, -1)
        global_features = self.global_body(self._build_geometry_features(obs, full_obs)).unsqueeze(1).expand(
            -1, num_agents, -1
        )
        features = torch.cat([local_features, global_features], dim=-1)
        values = torch.zeros(batch_size, num_agents, device=obs.device)
        for role_idx in range(self.num_roles):
            mask = role_ids == role_idx
            if mask.any():
                values[mask] = self.role_heads[role_idx](features[mask]).squeeze(-1)
        return values


class FlatStateCentralCritic(nn.Module):
    """Scheme 4: flattened semantic state MLP for all chaser values jointly.

    The flat state contains only the actor-visible chaser prefixes plus the
    prey's true observation:
    `[Herder30, Netter130, Netter230, Prey15]`.
    """

    def __init__(self, full_obs_dim: int, hidden_dim: int, num_layers: int, num_trainable_agents: int = 3):
        super().__init__()
        del full_obs_dim  # Kept for factory compatibility; semantic slices define the true input size.
        self.num_trainable_agents = num_trainable_agents
        semantic_input_dim = CHASER_LOCAL_OBS_DIM * num_trainable_agents + PREY_OBS_DIM
        self.body = _build_mlp(semantic_input_dim, hidden_dim, num_layers, output_dim=num_trainable_agents)

    def _build_flat_state(self, full_obs: torch.Tensor) -> torch.Tensor:
        expected_tokens = self.num_trainable_agents + 1
        if full_obs.size(1) != expected_tokens:
            raise ValueError(
                "FlatStateCentralCritic expects canonical tokens "
                f"[Herder, Netter1, Netter2, Prey], got num_tokens={full_obs.size(1)}."
            )
        chaser_obs = full_obs[:, : self.num_trainable_agents, :CHASER_LOCAL_OBS_DIM]
        prey_obs = full_obs[:, self.num_trainable_agents, :PREY_OBS_DIM]
        return torch.cat([chaser_obs.reshape(full_obs.size(0), -1), prey_obs], dim=-1)

    def forward(self, obs, role_ids, full_obs=None):
        del obs, role_ids
        if full_obs is None:
            raise ValueError("FlatStateCentralCritic requires full_obs in canonical [H, N1, N2, Prey] order.")
        return self.body(self._build_flat_state(full_obs))


def build_critic_network(
    critic_variant: str,
    local_obs_dim: int,
    full_obs_dim: int,
    hidden_dim: int,
    num_layers: int,
    num_roles: int = 2,
    agent_id_dim: int = NUM_TOTAL_ROLES,
    attention_heads: int = 4,
    num_trainable_agents: int = 3,
):
    variant = normalize_critic_variant_name(critic_variant)
    if variant in {"1", "scheme1", "dual_stream", "dual"}:
        raise ValueError(
            "critic_variant='dual_stream' has been removed. "
            "Use critic_multihead, token_attention, geometric, or flat_state instead."
        )
    if variant in {"critic_multihead", "multihead", "multi_head", "legacy", "legacy_local", "local", "default", "0"}:
        return CriticMultiHead(
            obs_dim=local_obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_roles=num_roles,
            agent_id_dim=agent_id_dim,
            num_trainable_agents=num_trainable_agents,
        )
    if variant in {"2", "scheme2", "token_attention", "attention"}:
        return AttentionCentralCritic(
            full_obs_dim=full_obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_roles=num_roles,
            total_roles=agent_id_dim,
            attention_heads=attention_heads,
        )
    if variant in {"3", "scheme3", "geometric", "geometry"}:
        return GeometricCentralCritic(
            local_obs_dim=local_obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_roles=num_roles,
            agent_id_dim=agent_id_dim,
        )
    if variant in {"4", "scheme4", "flat_state", "flat"}:
        return FlatStateCentralCritic(
            full_obs_dim=full_obs_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_trainable_agents=num_trainable_agents,
        )
    raise ValueError(
        f"Unknown critic_variant={critic_variant!r}. "
        "Supported values: critic_multihead, token_attention (2), geometric (3), flat_state (4)."
    )
