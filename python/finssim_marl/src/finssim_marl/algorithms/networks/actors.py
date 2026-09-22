"""
Actor networks for MARL.
Contains standard Actor and MultiHead Actor for heterogeneous agents.
"""

import torch
import torch.nn as nn

SQUASH_EPS = 1e-6


def _atanh_stable(x: torch.Tensor) -> torch.Tensor:
    clipped_x = x.clamp(-1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
    return 0.5 * (torch.log1p(clipped_x) - torch.log1p(-clipped_x))


def _squashed_normal_log_prob(
    dist: torch.distributions.Normal,
    actions: torch.Tensor,
) -> torch.Tensor:
    safe_actions = actions.clamp(-1.0 + SQUASH_EPS, 1.0 - SQUASH_EPS)
    pre_tanh = _atanh_stable(safe_actions)
    log_det_jacobian = torch.log(1.0 - safe_actions.pow(2) + SQUASH_EPS)
    return (dist.log_prob(pre_tanh) - log_det_jacobian).sum(dim=-1)


def _sample_squashed_normal(
    dist: torch.distributions.Normal,
    deterministic: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    pre_tanh = dist.mean if deterministic else dist.rsample()
    actions = torch.tanh(pre_tanh)
    log_det_jacobian = torch.log(1.0 - actions.pow(2) + SQUASH_EPS)
    log_prob = (dist.log_prob(pre_tanh) - log_det_jacobian).sum(dim=-1)
    return actions, log_prob


def _estimate_squashed_normal_entropy(dist: torch.distributions.Normal) -> torch.Tensor:
    sampled_actions, log_prob = _sample_squashed_normal(dist, deterministic=False)
    del sampled_actions
    return -log_prob


class Actor(nn.Module):
    """Standard Actor network with Gaussian policy."""

    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))

    def act(self, x):
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)

        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        action, log_prob = _sample_squashed_normal(distribution)

        return action, log_prob

    def get_log_prob(self, x, actions):
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)

        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        log_prob = _squashed_normal_log_prob(distribution, actions)

        return log_prob

    def get_entropy(self, x):
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)

        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        entropy = _estimate_squashed_normal_entropy(distribution)

        return entropy


# =============================================================================
# Hardcoded role mapping for Unity 3chase1 environment
# role_id -> one-hot index
# =============================================================================
ROLE_MAPPING = {
    'Herder': 0,   # Herder (fisherman)
    'Netter': 1,   # Netter (net puller) - two Netters share same role ID
    'Prey': 2,     # Prey (escapee)
}

ID_MAPPING = {v: k for k, v in ROLE_MAPPING.items()}

# Roles that need training (Prey is excluded)
TRAINABLE_ROLES = {'Herder', 'Netter'}


def get_role_from_agent_name(agent_name: str) -> int:
    """Parse role ID from Unity agent name.

    Args:
        agent_name: Unity agent name, format like 'Herder?team=0?agent_id=3'

    Returns:
        role_id: Role ID (0=Herder, 1=Netter, 2=Prey)

    Raises:
        ValueError: If role cannot be determined from agent name
    """
    for role_name, role_id in ROLE_MAPPING.items():
        if role_name in agent_name:
            return role_id
    raise ValueError(
        f"Cannot determine role for agent: '{agent_name}'. "
        f"Available roles: {list(ROLE_MAPPING.keys())}"
    )


class ActorMultiHead(nn.Module):
    """Multi-head Actor network for heterogeneous agents.

    Supports multiple roles (e.g., Herder, Netter) with:
    - Shared feature extraction body
    - Role-specific output heads
    - Agent ID embedding for role identification
    """

    def __init__(self, obs_dim, hidden_dim, num_layers, action_dim, num_roles=2, agent_id_dim=None):
        super().__init__()
        self.num_roles = num_roles
        self.action_dim = action_dim
        self.obs_dim = obs_dim
        self.agent_id_dim = agent_id_dim if agent_id_dim else num_roles

        input_dim = obs_dim + self.agent_id_dim
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.shared_body.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.role_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.role_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim)
                )
            )

        self.log_stds = nn.ParameterList([
            nn.Parameter(torch.zeros(action_dim)) for _ in range(num_roles)
        ])

    def _build_agent_id_onehot(self, batch_size, num_agents, role_ids, device):
        """Build one-hot agent ID tensor.

        Args:
            batch_size: Number of batches
            num_agents: Number of agents per batch
            role_ids: [batch_size, num_agents] role index for each agent
            device: Device to create tensor on
        Returns:
            agent_ids_onehot: [batch_size, num_agents, agent_id_dim]
        """
        agent_ids_onehot = torch.zeros(batch_size, num_agents, self.agent_id_dim, device=device)
        for b in range(batch_size):
            for a in range(num_agents):
                role = role_ids[b, a]
                if ID_MAPPING[role.item()] in TRAINABLE_ROLES:
                    agent_ids_onehot[b, a, role] = 1.0
        return agent_ids_onehot

    def act(self, obs, role_ids, deterministic: bool = False):
        """Sample actions for all agents based on their roles.

        Args:
            obs: [batch_size, num_agents, obs_dim] Observations
            role_ids: [batch_size, num_agents] Role index for each agent
            deterministic: If True, use the policy mean instead of sampling
        Returns:
            actions: [batch_size, num_agents, action_dim] Sampled actions
            log_probs: [batch_size, num_agents] Log probabilities
            chosen_heads: [batch_size, num_agents] Which head was used
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        actions = torch.zeros(batch_size, num_agents, self.action_dim, device=device)
        log_probs = torch.zeros(batch_size, num_agents, device=device)
        chosen_heads = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]
                mean = self.role_heads[role_idx](role_features)
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                sampled_actions, lp = _sample_squashed_normal(
                    dist, deterministic=deterministic
                )

                actions[mask] = sampled_actions
                log_probs[mask] = lp
                chosen_heads[mask] = role_idx

        return actions, log_probs, chosen_heads

    def get_log_prob(self, obs, role_ids, actions):
        """Calculate log probabilities for given actions.

        Args:
            obs: [batch_size, num_agents, obs_dim] Observations
            role_ids: [batch_size, num_agents] Role index for each agent
            actions: [batch_size, num_agents, action_dim] Actions to evaluate
        Returns:
            log_probs: [batch_size, num_agents] Log probabilities
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        log_probs = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]
                mean = self.role_heads[role_idx](role_features)
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                lp = _squashed_normal_log_prob(dist, actions[mask])
                log_probs[mask] = lp

        return log_probs

    def get_entropy(self, obs, role_ids):
        """Calculate entropy of the policy distribution.

        Args:
            obs: [batch_size, num_agents, obs_dim] Observations
            role_ids: [batch_size, num_agents] Role index for each agent
        Returns:
            entropy: [batch_size, num_agents] Entropy for each agent
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        entropy = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]
                mean = self.role_heads[role_idx](role_features)
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                ent = _estimate_squashed_normal_entropy(dist)
                entropy[mask] = ent

        return entropy
