"""
MAPPO (Multi-Agent Proximal Policy Optimization) algorithm for Unity 3chase1.

Reference: https://arxiv.org/abs/2103.01955

This implementation uses:
- ActorMultiHead: Heterogeneous agents (Herder, Netter) with role-specific heads
- CriticMultiHead: Centralized critic with role-specific heads
- TD(λ) for advantage estimation

The Unity environment may expose richer per-agent observations than the actor
actually consumes. For 3chase1, the trainable end-to-end MAPPO policy uses only
the `actor_obs` prefix of each Chaser observation, while any controller-specific
suffix is ignored.
"""

# Adapted from CleanMARL (MIT License).  Preserve the CleanMARL attribution and
# MIT terms for the portions derived from that project.

import numpy as np
import torch
import torch.optim as optim
from dataclasses import asdict, dataclass
from typing import Optional
from tqdm import tqdm

from .base import BaseAlgorithm
from .networks.actors import ActorMultiHead, ROLE_MAPPING
from .networks.critics import build_critic_network, critic_variant_to_id, normalize_critic_variant_name


@dataclass
class MAPPOMultiHeadConfig:
    """Configuration for MAPPO algorithm."""
    actor_hidden_dim: int = 64
    actor_num_layers: int = 3
    critic_hidden_dim: int = 128
    critic_num_layers: int = 3
    optimizer: str = "Adam"
    learning_rate_actor: float = 0.0008
    learning_rate_critic: float = 0.0008
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_return: bool = True
    epochs: int = 8
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = -1
    device: str = "cuda"

    # 3chase1 specific
    chaser_team_obs_dim: int = 30
    chaser_team_action_dim: int = 8
    num_trainable_roles: int = 2  # Herder and Netter (Prey is fixed policy)
    action_interface: str = "direct_thruster"
    rollout_action_dim: int = 8
    env_action_dim: int = 8
    critic_variant: str = "critic_multihead"
    critic_attention_heads: int = 4


def _norm_d(grads, d):
    """Compute norm of gradients."""
    valid_grads = [g for g in grads if g is not None]
    if not valid_grads:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in valid_grads]
    total_norm_d = torch.linalg.vector_norm(torch.stack(norms), d)
    return total_norm_d


class MAPPOMultiHeadAlgorithm(BaseAlgorithm):
    """MAPPO algorithm for heterogeneous multi-agent training.
    
    仅仅使用于3chase1场景，假设只有 Herder 和 Netter 需要训练，Prey 始终使用固定策略（不更新）。因此 Actor 和 Critic 都只为 Herder/Netter 提供 trainable heads，而 Prey 的部分则保持静态。

    Uses centralized training with decentralized execution (CTDE).
    The algorithm does NOT manage its own train loop - the external Runner
    handles rollout collection, logging, checkpointing, and evaluation.
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        role_ids: np.ndarray,
        config: MAPPOMultiHeadConfig,
    ):
        """
        Initialize MAPPO algorithm.

        Args:
            obs_dim: Full observation dimension per agent (after padding)
            action_dim: Action dimension per agent (after padding)
            state_dim: Global state dimension
            n_agents: Total number of agents
            role_ids: Role ID for each agent [n_agents]
            config: MAPPO configuration
        """
        self.config = config
        self.n_agents = n_agents
        self.role_ids = role_ids
        self.obs_dim = obs_dim
        self.device = torch.device(config.device)
        self.action_interface = config.action_interface
        self.rollout_action_dim = config.rollout_action_dim
        self.env_action_dim = config.env_action_dim

        # Determine number of trainable roles
        self.num_trainable_roles = config.num_trainable_roles
        num_role_types = len(ROLE_MAPPING)  # 3 (Herder, Netter, Prey)

        # Initialize ActorMultiHead for heterogeneous agents
        self.actor = ActorMultiHead(
            obs_dim=config.chaser_team_obs_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            action_dim=config.chaser_team_action_dim,
            num_roles=config.num_trainable_roles,
            agent_id_dim=num_role_types,
        ).to(self.device)

        # Precompute actor indices for Herder/Netter (excluding Prey)
        self.actor_indices = self._compute_actor_indices()
        self.chaser_team_order_tensor = torch.tensor(
            self._compute_chaser_team_order(), device=self.device
        ).long()
        prey_indices = np.where(self.role_ids == ROLE_MAPPING["Prey"])[0]
        if len(prey_indices) != 1:
            raise ValueError(f"Expected exactly one prey agent, got prey_indices={prey_indices}.")
        self.prey_index = int(prey_indices[0])
        self.chaser_team_role_ids = torch.tensor(
            [ROLE_MAPPING["Herder"], ROLE_MAPPING["Netter"], ROLE_MAPPING["Netter"]],
            dtype=torch.long,
            device=self.device,
        )

        self.critic = self._build_critic().to(self.device)

        # Optimizers
        Optimizer = getattr(optim, config.optimizer)
        self.actor_optimizer = Optimizer(
            self.actor.parameters(), lr=config.learning_rate_actor
        )
        self.critic_optimizer = Optimizer(
            self.critic.parameters(), lr=config.learning_rate_critic
        )

    def _build_critic(self):
        num_role_types = len(ROLE_MAPPING)
        return build_critic_network(
            critic_variant=self.config.critic_variant,
            local_obs_dim=self.config.chaser_team_obs_dim,
            full_obs_dim=self.obs_dim,
            hidden_dim=self.config.critic_hidden_dim,
            num_layers=self.config.critic_num_layers,
            num_roles=self.config.num_trainable_roles,
            agent_id_dim=num_role_types,
            attention_heads=self.config.critic_attention_heads,
            num_trainable_agents=len(self.actor_indices),
        )

    def _rebuild_critic_optimizer(self) -> None:
        Optimizer = getattr(optim, self.config.optimizer)
        self.critic_optimizer = Optimizer(
            self.critic.parameters(), lr=self.config.learning_rate_critic
        )

    def _infer_checkpoint_critic_variant(self, checkpoint: dict) -> Optional[str]:
        critic_state = checkpoint.get("critic", {})
        keys = set(critic_state.keys())
        if any(key.startswith("team_encoder.") for key in keys):
            return "critic_multihead"
        if any(key.startswith("chaser_token_encoder.") for key in keys):
            return "token_attention"
        if any(key.startswith("local_body.") or key.startswith("global_body.") for key in keys):
            return "geometric"
        if keys and all(key.startswith("body.") for key in keys):
            return "flat_state"
        return None

    def _restore_critic_architecture_from_checkpoint(self, checkpoint: dict) -> None:
        checkpoint_config = checkpoint.get("config")
        inferred_variant = self._infer_checkpoint_critic_variant(checkpoint)
        critic_fields = (
            "critic_variant",
            "critic_attention_heads",
            "critic_hidden_dim",
            "critic_num_layers",
        )
        changed = False

        if isinstance(checkpoint_config, dict):
            for field_name in critic_fields:
                if field_name in checkpoint_config and hasattr(self.config, field_name):
                    checkpoint_value = checkpoint_config[field_name]
                    if getattr(self.config, field_name) != checkpoint_value:
                        setattr(self.config, field_name, checkpoint_value)
                        changed = True

        if inferred_variant is not None and (
            normalize_critic_variant_name(self.config.critic_variant)
            != normalize_critic_variant_name(inferred_variant)
        ):
            self.config.critic_variant = inferred_variant
            changed = True

        if changed:
            self.critic = self._build_critic().to(self.device)
            self._rebuild_critic_optimizer()

    def _extract_actor_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        """Slice the actor-visible prefix from full environment observations."""
        if obs_tensor.size(-1) < self.config.chaser_team_obs_dim:
            raise ValueError(
                "MAPPO direct-thruster policy expected observations with at least "
                f"{self.config.chaser_team_obs_dim} actor dims, got obs_dim={obs_tensor.size(-1)}."
            )
        return obs_tensor[..., : self.config.chaser_team_obs_dim]

    def _compute_actor_indices(self):
        """Precompute indices of actors (non-Prey agents)."""
        role_ids_batch = np.tile(self.role_ids, (1, 1))
        actor_indices = np.where(
            np.any(role_ids_batch != ROLE_MAPPING['Prey'], axis=0)
        )[0]
        return torch.from_numpy(actor_indices).long().to(self.device)

    def _compute_chaser_team_order(self):
        """Return the Herder/Netter ordering used by obs_chaser_team."""
        env_role_ids = self.role_ids
        chaser_env_positions = [i for i, rid in enumerate(env_role_ids) if rid != ROLE_MAPPING["Prey"]]
        chaser_agent_ids = [env_role_ids[i] for i in chaser_env_positions]
        herder_pos = (
            chaser_env_positions[chaser_agent_ids.index(ROLE_MAPPING["Herder"])]
            if ROLE_MAPPING["Herder"] in chaser_agent_ids
            else None
        )
        netter_positions = [
            (chaser_env_positions[i], i)
            for i, rid in enumerate(chaser_agent_ids)
            if rid == ROLE_MAPPING["Netter"]
        ]
        netter_positions.sort(key=lambda x: env_role_ids[x[0]])
        if herder_pos is not None:
            return [herder_pos] + [pos for pos, _ in netter_positions]
        return [pos for pos, _ in netter_positions]

    def _ordered_chaser_role_ids(self, batch_size: int) -> torch.Tensor:
        return self.chaser_team_role_ids.unsqueeze(0).expand(batch_size, -1)

    def _extract_ordered_full_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        chaser_full_obs = obs_tensor[:, self.chaser_team_order_tensor, :]
        prey_full_obs = obs_tensor[:, self.prey_index : self.prey_index + 1, :]
        return torch.cat([chaser_full_obs, prey_full_obs], dim=1)

    def _critic_forward(self, ordered_full_obs: torch.Tensor) -> torch.Tensor:
        role_ids = self._ordered_chaser_role_ids(ordered_full_obs.size(0))
        chaser_local_obs = self._extract_actor_obs(ordered_full_obs[:, : len(self.actor_indices), :])
        return self.critic(obs=chaser_local_obs, role_ids=role_ids, full_obs=ordered_full_obs)

    def select_action(self, obs, role_ids, deterministic=False):
        """Select actions for given observations.

        This is used during rollout collection.

        Args:
            obs: Full observations [batch_size, n_agents, obs_dim]
            role_ids: Role IDs [batch_size, n_agents]
            deterministic: If True, use mean action (for evaluation)

        Returns:
            actions: [batch_size, n_agents, action_dim]
            log_probs: [batch_size, n_agents]
            chosen_heads: [batch_size, n_agents]
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        role_ids_tensor = torch.from_numpy(role_ids).long().to(self.device)

        with torch.no_grad():
            # Extract only the actor-visible prefix for Herder/Netter agents.
            obs_actor = self._extract_actor_obs(obs_tensor[:, self.actor_indices, :])
            role_ids_actor = role_ids_tensor[:, self.actor_indices]

            actions_actor, log_probs_actor, chosen_heads = self.actor.act(
                obs_actor, role_ids_actor, deterministic=deterministic
            )

        # Reconstruct full action array with Prey positions filled with zeros
        batch_size = obs_tensor.size(0)
        n_agents = self.n_agents
        actions = torch.zeros(batch_size, n_agents, self.config.chaser_team_action_dim, device=self.device)
        log_probs = torch.zeros(batch_size, n_agents, device=self.device)
        chosen_heads_full = torch.zeros(batch_size, n_agents, dtype=torch.long, device=self.device)

        actions[:, self.actor_indices] = actions_actor
        log_probs[:, self.actor_indices] = log_probs_actor
        chosen_heads_full[:, self.actor_indices] = chosen_heads

        return (
            actions.cpu().numpy(),
            log_probs.cpu().numpy(),
            chosen_heads_full.cpu().numpy(),
        )

    def update(self, batch):
        """Update policy with a batch of rollout data.

        Args:
            batch: Tuple from RolloutBuffer.get_batch():
                (obs, actions, log_probs, reward, states, done, mask,
                 role_ids, obs_chaser_team, reward_chaser_team)

        Returns:
            metrics: Dictionary of training metrics
        """
        config = self.config
        device = self.device

        (
            b_obs,
            b_actions,
            b_log_probs,
            b_reward,
            b_states,
            b_done,
            b_mask,
            b_role_ids,
            b_obs_chaser_team,
            b_reward_chaser_team,
        ) = batch
        del b_reward, b_states, b_done, b_role_ids, b_obs_chaser_team

        # Compute TD(λ) returns and advantages for chaser team
        n_agents = len(self.actor_indices)
        return_lambda = torch.zeros(
            (b_actions.size(0), b_actions.size(1), n_agents)
        ).float().to(device)
        advantages = torch.zeros(
            (b_actions.size(0), b_actions.size(1), n_agents)
        ).float().to(device)

        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                last_return_lambda = torch.zeros(n_agents, device=device)

                for t in reversed(range(int(ep_len))):
                    ordered_full_obs_t = self._extract_ordered_full_obs(
                        b_obs[ep_idx, t].unsqueeze(0)
                    )
                    reward_chaser_t = b_reward_chaser_team[ep_idx, t, :n_agents]

                    if t == int(ep_len - 1):
                        next_value = torch.zeros(n_agents, device=device)
                    else:
                        next_ordered_full_obs_t = self._extract_ordered_full_obs(
                            b_obs[ep_idx, t + 1].unsqueeze(0)
                        )
                        next_value = self._critic_forward(next_ordered_full_obs_t)

                    # TD(λ): G_λ(t) = r_t + γ[λ * G_λ(t+1) + (1-λ) * V(s_{t+1})]
                    return_lambda[ep_idx, t, :n_agents] = last_return_lambda = reward_chaser_t + config.gamma * (
                        config.td_lambda * last_return_lambda
                        + (1 - config.td_lambda) * next_value
                    )
                    current_value = self._critic_forward(ordered_full_obs_t)
                    advantages[ep_idx, t, :n_agents] = return_lambda[ep_idx, t, :n_agents] - current_value

        # Normalize advantages for chaser team
        advantages_chaser = advantages[:, :, :n_agents]
        return_lambda_chaser = return_lambda[:, :, :n_agents]

        if config.normalize_advantage:
            adv_mu = advantages_chaser[b_mask].mean()
            adv_std = advantages_chaser[b_mask].std()
            advantages[:, :, :n_agents] = (advantages_chaser - adv_mu) / (adv_std + 1e-8)

        if config.normalize_return:
            ret_mu = return_lambda_chaser[b_mask].mean()
            ret_std = return_lambda_chaser[b_mask].std()
            return_lambda[:, :, :n_agents] = (return_lambda_chaser - ret_mu) / (ret_std + 1e-8)

        # Training loop
        actor_losses = []
        critic_losses = []
        entropies = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []
        critic_predictions = []
        critic_targets = []

        for epoch_idx in range(config.epochs):
            epoch_pbar = tqdm(total=b_obs.size(1), desc=f"Epoch {epoch_idx+1}/{config.epochs}", leave=False)
            actor_loss = 0
            critic_loss = 0
            entropy_sum = 0
            kl_divergence = 0
            clipped_ratio = 0

            for t in range(b_obs.size(1)):
                epoch_pbar.update(1)
                ordered_full_obs_t = self._extract_ordered_full_obs(b_obs[:, t, :, :])
                obs_actor_t = self._extract_actor_obs(ordered_full_obs_t[:, :n_agents, :])
                actions_actor_t = b_actions[:, t, self.chaser_team_order_tensor, :]
                log_probs_actor_t = b_log_probs[:, t, self.chaser_team_order_tensor]
                role_ids_actor_t = self._ordered_chaser_role_ids(b_obs.size(0))

                valid_mask = b_mask[:, t]
                advantages_actor = advantages[:, t, :n_agents]

                # Policy gradient loss
                current_logprob = self.actor.get_log_prob(
                    obs_actor_t, role_ids_actor_t, actions_actor_t
                )
                log_ratio = current_logprob - log_probs_actor_t
                ratio = torch.exp(log_ratio)

                pg_loss1 = advantages_actor * ratio
                pg_loss2 = advantages_actor * torch.clamp(
                    ratio, 1 - config.ppo_clip, 1 + config.ppo_clip
                )
                pg_loss = (
                    torch.min(pg_loss1[valid_mask], pg_loss2[valid_mask])
                    .mean(dim=-1)
                    .sum()
                )

                # Entropy bonus
                entropy_loss = self.actor.get_entropy(obs_actor_t, role_ids_actor_t)[valid_mask].mean(dim=-1).sum()
                entropy_sum += entropy_loss
                actor_loss += -pg_loss - config.entropy_coef * entropy_loss

                # Value loss
                current_values = self.critic(
                    obs=obs_actor_t,
                    role_ids=role_ids_actor_t,
                    full_obs=ordered_full_obs_t,
                )
                return_lambda_actor = return_lambda[:, t, :n_agents]
                if valid_mask.any():
                    critic_predictions.append(current_values[valid_mask].detach().reshape(-1))
                    critic_targets.append(return_lambda_actor[valid_mask].detach().reshape(-1))
                weight = torch.tensor([0.4, 0.3, 0.3], device=device)[:n_agents]
                value_loss = (
                    ((current_values - return_lambda_actor) ** 2)
                    * weight
                    * valid_mask.unsqueeze(-1).float()
                ).sum()
                critic_loss += value_loss

                # Track KL divergence
                b_kl_divergence = (
                    ((ratio - 1) - log_ratio)[valid_mask].mean(dim=-1).sum()
                )
                kl_divergence += b_kl_divergence

                clipped_ratio += (
                    ((ratio - 1.0).abs() > config.ppo_clip)[valid_mask]
                    .float()
                    .mean(dim=-1)
                    .sum()
                )

            # Normalize by number of valid episodes
            actor_count = b_mask.sum()
            actor_loss /= actor_count
            critic_loss /= b_mask.sum()
            entropy_sum /= actor_count
            kl_divergence /= actor_count
            clipped_ratio /= actor_count

            epoch_pbar.close()

            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            actor_gradient = _norm_d([p.grad for p in self.actor.parameters()], 2)
            critic_gradient = _norm_d([p.grad for p in self.critic.parameters()], 2)

            if config.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.actor.parameters(), max_norm=config.clip_gradients
                )
                torch.nn.utils.clip_grad_norm_(
                    self.critic.parameters(), max_norm=config.clip_gradients
                )

            self.actor_optimizer.step()
            self.critic_optimizer.step()

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies.append(entropy_sum.item())
            kl_divergences.append(kl_divergence.item())
            actor_gradients.append(actor_gradient.item())
            critic_gradients.append(critic_gradient.item())
            clipped_ratios.append(clipped_ratio.cpu().item())

        valid_advantages = advantages[:, :, :n_agents][b_mask].detach().reshape(-1)
        if critic_predictions:
            critic_predictions_tensor = torch.cat(critic_predictions)
            critic_targets_tensor = torch.cat(critic_targets)
            target_var = torch.var(critic_targets_tensor, unbiased=False)
            critic_explained_variance = 1.0 - (
                torch.var(critic_targets_tensor - critic_predictions_tensor, unbiased=False)
                / (target_var + 1e-8)
            )
            critic_value_mean = critic_predictions_tensor.mean().item()
            critic_value_std = critic_predictions_tensor.std(unbiased=False).item()
            critic_target_mean = critic_targets_tensor.mean().item()
            critic_target_std = critic_targets_tensor.std(unbiased=False).item()
            critic_explained_variance_value = critic_explained_variance.item()
        else:
            critic_value_mean = 0.0
            critic_value_std = 0.0
            critic_target_mean = 0.0
            critic_target_std = 0.0
            critic_explained_variance_value = 0.0

        return {
            "actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies),
            "kl_divergence": np.mean(kl_divergences),
            "actor_gradient": np.mean(actor_gradients),
            "critic_gradient": np.mean(critic_gradients),
            "clipped_ratio": np.mean(clipped_ratios),
            "critic_value_mean": critic_value_mean,
            "critic_value_std": critic_value_std,
            "critic_target_mean": critic_target_mean,
            "critic_target_std": critic_target_std,
            "critic_explained_variance": critic_explained_variance_value,
            "advantage_mean": valid_advantages.mean().item(),
            "advantage_std": valid_advantages.std(unbiased=False).item(),
            "advantage_abs_mean": valid_advantages.abs().mean().item(),
        }

    def get_experiment_metadata(self) -> dict:
        normalized_variant = normalize_critic_variant_name(self.config.critic_variant)
        return {
            "critic_variant": normalized_variant,
            "critic_variant_id": critic_variant_to_id(normalized_variant),
            "critic_attention_heads": self.config.critic_attention_heads,
        }

    def save(self, path: str):
        """Save model checkpoint."""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": asdict(self.config),
        }, path)

    def load(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        self._restore_critic_architecture_from_checkpoint(checkpoint)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def train(self):
        """Set to training mode."""
        self.actor.train()
        self.critic.train()

    def eval(self):
        """Set to evaluation mode."""
        self.actor.eval()
        self.critic.eval()
