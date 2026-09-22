"""MAPPO with local subgoal actions and a fixed PID/wrench low-level path."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from .base import BaseAlgorithm
from .networks.actors import (
    ID_MAPPING,
    ROLE_MAPPING,
    TRAINABLE_ROLES,
    _estimate_squashed_normal_entropy,
    _sample_squashed_normal,
    _squashed_normal_log_prob,
)
from .networks.critics import build_critic_network
from .networks.critics import critic_variant_to_id, normalize_critic_variant_name
from .position_control_backends import (
    BodyFramePIDWrenchControllerConfig,
    make_body_frame_position_control_backend,
)


@dataclass
class MAPPOMultiHeadPositionControlConfig:
    """Configuration for MAPPO with external position-control backends."""

    actor_hidden_dim: int = 64
    actor_num_layers: int = 2
    critic_hidden_dim: int = 64
    critic_num_layers: int = 2
    action_interface: str = "position_controller"
    control_interface_version: str = "body_subgoal_pid_physical_wrench_v1"
    policy_action_dim: int = 4
    rollout_action_dim: int = 4
    env_action_dim: int = 8
    target_body_delta_limits: Tuple[float, float, float] = (1.5, 0.5, 1.5)
    enable_yaw_control: bool = True
    yaw_error_limit_deg: float = 90.0

    optimizer: str = "Adam"
    learning_rate_actor: float = 0.0003
    learning_rate_critic: float = 0.0003
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_return: bool = False
    epochs: int = 8
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = -1
    device: str = "cuda"

    chaser_team_obs_dim: int = 30
    chaser_team_action_dim: int = 8
    num_trainable_roles: int = 2
    critic_variant: str = "critic_multihead"
    critic_attention_heads: int = 4

    controller_backend: str = "body_pid_wrench"
    body_pid_wrench_controller: BodyFramePIDWrenchControllerConfig = field(
        default_factory=BodyFramePIDWrenchControllerConfig
    )


def _norm_d(grads, d):
    valid_grads = [g for g in grads if g is not None]
    if not valid_grads:
        return torch.tensor(0.0)
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in valid_grads]
    return torch.linalg.vector_norm(torch.stack(norms), d)


class PositionMetaActor(nn.Module):
    """Multi-head Gaussian actor for normalized position targets with optional yaw."""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        action_dim: int = 4,
        num_roles: int = 2,
        agent_id_dim: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim

        input_dim = obs_dim + agent_id_dim
        self.shared_body = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        for _ in range(num_layers - 1):
            self.shared_body.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.role_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.role_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim),
                )
            )

        self.log_stds = nn.ParameterList(
            [nn.Parameter(torch.zeros(action_dim)) for _ in range(num_roles)]
        )

    def _build_agent_id_onehot(self, batch_size, num_agents, role_ids, device):
        agent_ids_onehot = torch.zeros(batch_size, num_agents, self.agent_id_dim, device=device)
        for b in range(batch_size):
            for a in range(num_agents):
                role = role_ids[b, a]
                if ID_MAPPING[role.item()] in TRAINABLE_ROLES:
                    agent_ids_onehot[b, a, role] = 1.0
        return agent_ids_onehot

    def _features(self, obs, role_ids):
        batch_size, num_agents, _ = obs.shape
        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, obs.device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1).reshape(batch_size * num_agents, -1)
        features = self.shared_body(x)
        return features.reshape(batch_size, num_agents, -1)

    def act(self, obs, role_ids, deterministic: bool = False):
        batch_size, num_agents, _ = obs.shape
        device = obs.device
        features = self._features(obs, role_ids)

        actions = torch.zeros(batch_size, num_agents, self.action_dim, device=device)
        log_probs = torch.zeros(batch_size, num_agents, device=device)
        chosen_heads = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)

        for role_idx in range(self.num_roles):
            mask = role_ids == role_idx
            if mask.any():
                role_features = features[mask]
                mean = self.role_heads[role_idx](role_features)
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                sampled_actions, sampled_log_probs = _sample_squashed_normal(
                    dist, deterministic=deterministic
                )
                log_probs[mask] = sampled_log_probs
                actions[mask] = sampled_actions
                chosen_heads[mask] = role_idx

        return actions, log_probs, chosen_heads

    def get_log_prob(self, obs, role_ids, actions):
        batch_size, num_agents, _ = obs.shape
        device = obs.device
        features = self._features(obs, role_ids)
        log_probs = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = role_ids == role_idx
            if mask.any():
                role_features = features[mask]
                mean = self.role_heads[role_idx](role_features)
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                log_probs[mask] = _squashed_normal_log_prob(dist, actions[mask])
        return log_probs

    def get_entropy(self, obs, role_ids):
        batch_size, num_agents, _ = obs.shape
        device = obs.device
        features = self._features(obs, role_ids)
        entropy = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = role_ids == role_idx
            if mask.any():
                role_features = features[mask]
                mean = self.role_heads[role_idx](role_features)
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                entropy[mask] = _estimate_squashed_normal_entropy(dist)
        return entropy


class MAPPOMultiHeadPositionControlAlgorithm(BaseAlgorithm):
    """MAPPO high-level position policy with frozen low-level controller."""

    action_interface = "position_controller"

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        role_ids: np.ndarray,
        config: MAPPOMultiHeadPositionControlConfig,
    ):
        del action_dim, state_dim
        self.config = config
        self.n_agents = n_agents
        self.role_ids = role_ids
        self.obs_dim = obs_dim
        self.device = torch.device(config.device)
        self.action_interface = config.action_interface
        expected_policy_action_dim = 4 if config.enable_yaw_control else 3
        if int(config.policy_action_dim) != expected_policy_action_dim:
            raise ValueError(
                "policy_action_dim does not match enable_yaw_control: "
                f"enable_yaw_control={config.enable_yaw_control}, "
                f"policy_action_dim={config.policy_action_dim}, "
                f"expected={expected_policy_action_dim}."
            )
        if int(config.rollout_action_dim) != expected_policy_action_dim:
            raise ValueError(
                "rollout_action_dim does not match enable_yaw_control: "
                f"enable_yaw_control={config.enable_yaw_control}, "
                f"rollout_action_dim={config.rollout_action_dim}, "
                f"expected={expected_policy_action_dim}."
            )
        self.policy_action_dim = config.policy_action_dim
        self.rollout_action_dim = config.rollout_action_dim
        self.env_action_dim = config.env_action_dim

        num_role_types = len(ROLE_MAPPING)
        self.actor = PositionMetaActor(
            obs_dim=config.chaser_team_obs_dim,
            hidden_dim=config.actor_hidden_dim,
            num_layers=config.actor_num_layers,
            action_dim=config.policy_action_dim,
            num_roles=config.num_trainable_roles,
            agent_id_dim=num_role_types,
        ).to(self.device)

        self.actor_indices = self._compute_actor_indices()
        self.chaser_team_order_tensor = torch.tensor(
            self._compute_chaser_team_order(), device=self.device
        ).long()
        prey_indices = np.where(self.role_ids == ROLE_MAPPING["Prey"])[0]
        if len(prey_indices) > 1:
            raise ValueError(f"Expected at most one prey agent, got prey_indices={prey_indices}.")
        # TriNetCapture is a three-ROV cooperative task with a passive Target.
        # The legacy critics still receive a full observation tensor, but the
        # multihead critic uses only the ordered three-chaser prefix.
        self.prey_index = int(prey_indices[0]) if len(prey_indices) == 1 else None
        self.chaser_team_role_ids = torch.tensor(
            [ROLE_MAPPING["Herder"], ROLE_MAPPING["Netter"], ROLE_MAPPING["Netter"]],
            dtype=torch.long,
            device=self.device,
        )
        self.critic = self._build_critic().to(self.device)

        Optimizer = getattr(optim, config.optimizer)
        self.actor_optimizer = Optimizer(self.actor.parameters(), lr=config.learning_rate_actor)
        self.critic_optimizer = Optimizer(self.critic.parameters(), lr=config.learning_rate_critic)

        self.controller = make_body_frame_position_control_backend(
            config.controller_backend,
            body_pid_wrench_config=config.body_pid_wrench_controller,
        )
        self._last_target_body_delta_norm = 0.0
        self._last_target_yaw_abs_deg = 0.0
        self._last_thruster_norm = 0.0

    def _build_critic(self) -> nn.Module:
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

    def reset_controller_state(self, env_mask=None) -> None:
        if hasattr(self.controller, "reset"):
            self.controller.reset(env_mask)

    def _compute_actor_indices(self):
        role_ids_batch = np.tile(self.role_ids, (1, 1))
        actor_indices = np.where(np.any(role_ids_batch != ROLE_MAPPING["Prey"], axis=0))[0]
        return torch.from_numpy(actor_indices).long().to(self.device)

    def _compute_chaser_team_order(self):
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
        return [herder_pos] + [pos for pos, _ in netter_positions] if herder_pos is not None else [
            pos for pos, _ in netter_positions
        ]

    def _ordered_chaser_role_ids(self, batch_size: int) -> torch.Tensor:
        return self.chaser_team_role_ids.unsqueeze(0).expand(batch_size, -1)

    def _extract_ordered_full_obs(self, obs_tensor: torch.Tensor) -> torch.Tensor:
        chaser_full_obs = obs_tensor[:, self.chaser_team_order_tensor, :]
        if self.prey_index is None:
            return chaser_full_obs
        prey_full_obs = obs_tensor[:, self.prey_index : self.prey_index + 1, :]
        return torch.cat([chaser_full_obs, prey_full_obs], dim=1)

    def _critic_forward(self, ordered_full_obs: torch.Tensor) -> torch.Tensor:
        role_ids = self._ordered_chaser_role_ids(ordered_full_obs.size(0))
        chaser_local_obs = ordered_full_obs[:, : len(self.actor_indices), : self.config.chaser_team_obs_dim]
        return self.critic(obs=chaser_local_obs, role_ids=role_ids, full_obs=ordered_full_obs)

    def select_action(self, obs, role_ids, deterministic: bool = False):
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        role_ids_tensor = torch.from_numpy(role_ids).long().to(self.device)
        if obs_tensor.size(-1) < self.config.chaser_team_obs_dim:
            raise ValueError(
                "Body-subgoal position control expected the 30D chaser actor observation, "
                f"but got obs_dim={obs_tensor.size(-1)} and needs at least "
                f"{self.config.chaser_team_obs_dim}."
            )

        with torch.no_grad():
            obs_actor = obs_tensor[:, self.actor_indices, : self.config.chaser_team_obs_dim]
            role_ids_actor = role_ids_tensor[:, self.actor_indices]
            meta_action_actor, log_probs_actor, chosen_heads = self.actor.act(
                obs_actor, role_ids_actor, deterministic=deterministic
            )

        meta_action_actor_np = meta_action_actor.cpu().numpy()
        body_limits = np.asarray(
            self.config.target_body_delta_limits, dtype=np.float32
        ).reshape(1, 1, 3)
        body_position_error = np.clip(meta_action_actor_np[..., :3], -1.0, 1.0) * body_limits
        if self.config.enable_yaw_control:
            yaw_error_deg = (
                np.clip(meta_action_actor_np[..., 3], -1.0, 1.0)
                * float(self.config.yaw_error_limit_deg)
            )
        else:
            yaw_error_deg = np.zeros(meta_action_actor_np.shape[:2], dtype=np.float32)
        env_actions_actor_np = self.controller.act(
            body_position_error,
            yaw_error_deg,
            deterministic=deterministic,
        )

        batch_size = obs_tensor.size(0)
        env_actions = np.zeros((batch_size, self.n_agents, self.env_action_dim), dtype=np.float32)
        buffer_actions = np.zeros((batch_size, self.n_agents, self.policy_action_dim), dtype=np.float32)
        log_probs = np.zeros((batch_size, self.n_agents), dtype=np.float32)
        chosen_heads_full = np.zeros((batch_size, self.n_agents), dtype=np.int64)

        actor_indices_np = self.actor_indices.cpu().numpy()
        env_actions[:, actor_indices_np, :] = env_actions_actor_np
        buffer_actions[:, actor_indices_np, :] = meta_action_actor_np
        log_probs[:, actor_indices_np] = log_probs_actor.cpu().numpy()
        chosen_heads_full[:, actor_indices_np] = chosen_heads.cpu().numpy()

        self._last_target_body_delta_norm = float(
            np.linalg.norm(body_position_error, axis=-1).mean()
        )
        if self.config.enable_yaw_control:
            self._last_target_yaw_abs_deg = float(
                np.abs(yaw_error_deg).mean()
            )
        else:
            self._last_target_yaw_abs_deg = 0.0
        self._last_thruster_norm = float(np.linalg.norm(env_actions_actor_np, axis=-1).mean())

        return env_actions, log_probs, chosen_heads_full, {"buffer_actions": buffer_actions}

    def update(self, batch):
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

        n_agents = len(self.actor_indices)
        return_lambda = torch.zeros((b_actions.size(0), b_actions.size(1), n_agents), device=device)
        advantages = torch.zeros((b_actions.size(0), b_actions.size(1), n_agents), device=device)

        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = int(b_mask[ep_idx].sum())
                last_return_lambda = torch.zeros(n_agents, device=device)
                for t in reversed(range(ep_len)):
                    ordered_full_obs_t = self._extract_ordered_full_obs(
                        b_obs[ep_idx, t].unsqueeze(0)
                    )
                    reward_chaser_t = b_reward_chaser_team[ep_idx, t, :n_agents]

                    if t == ep_len - 1:
                        next_value = torch.zeros(n_agents, device=device)
                    else:
                        next_ordered_full_obs_t = self._extract_ordered_full_obs(
                            b_obs[ep_idx, t + 1].unsqueeze(0)
                        )
                        next_value = self._critic_forward(next_ordered_full_obs_t)

                    return_lambda[ep_idx, t] = last_return_lambda = reward_chaser_t + config.gamma * (
                        config.td_lambda * last_return_lambda + (1 - config.td_lambda) * next_value
                    )
                    current_value = self._critic_forward(ordered_full_obs_t)
                    advantages[ep_idx, t] = return_lambda[ep_idx, t] - current_value

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
            epoch_pbar = tqdm(total=b_obs.size(1), desc=f"Epoch {epoch_idx + 1}/{config.epochs}", leave=False)
            actor_loss = 0
            critic_loss = 0
            entropy_sum = 0
            kl_divergence = 0
            clipped_ratio = 0

            for t in range(b_obs.size(1)):
                epoch_pbar.update(1)
                ordered_full_obs_t = self._extract_ordered_full_obs(b_obs[:, t, :, :])
                obs_actor_t = ordered_full_obs_t[:, :n_agents, : config.chaser_team_obs_dim]
                actions_actor_t = b_actions[:, t, self.chaser_team_order_tensor, :]
                log_probs_actor_t = b_log_probs[:, t, self.chaser_team_order_tensor]
                role_ids_actor_t = self._ordered_chaser_role_ids(b_obs.size(0))
                valid_mask = b_mask[:, t]
                advantages_actor = advantages[:, t, :n_agents]

                current_logprob = self.actor.get_log_prob(obs_actor_t, role_ids_actor_t, actions_actor_t)
                log_ratio = current_logprob - log_probs_actor_t
                ratio = torch.exp(log_ratio)

                pg_loss1 = advantages_actor * ratio
                pg_loss2 = advantages_actor * torch.clamp(ratio, 1 - config.ppo_clip, 1 + config.ppo_clip)
                pg_loss = torch.min(pg_loss1[valid_mask], pg_loss2[valid_mask]).mean(dim=-1).sum()

                entropy_loss = self.actor.get_entropy(obs_actor_t, role_ids_actor_t)[valid_mask].mean(dim=-1).sum()
                entropy_sum += entropy_loss
                actor_loss += -pg_loss - config.entropy_coef * entropy_loss

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
                critic_loss += (
                    ((current_values - return_lambda_actor) ** 2)
                    * weight
                    * valid_mask.unsqueeze(-1).float()
                ).sum()

                kl_divergence += ((ratio - 1) - log_ratio)[valid_mask].mean(dim=-1).sum()
                clipped_ratio += ((ratio - 1.0).abs() > config.ppo_clip)[valid_mask].float().mean(dim=-1).sum()

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
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=config.clip_gradients)
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=config.clip_gradients)
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
        actor_log_stds = torch.cat([param.detach().reshape(-1) for param in self.actor.log_stds])
        actor_stds = torch.exp(actor_log_stds)

        return {
            "meta_actor_loss": np.mean(actor_losses),
            "critic_loss": np.mean(critic_losses),
            "entropy": np.mean(entropies),
            "kl_divergence": np.mean(kl_divergences),
            "meta_actor_gradient": np.mean(actor_gradients),
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
            "actor_log_std_mean": actor_log_stds.mean().item(),
            "actor_std_mean": actor_stds.mean().item(),
            "target_body_delta_norm_m": self._last_target_body_delta_norm,
            "target_yaw_abs_deg": self._last_target_yaw_abs_deg,
            "thruster_norm": self._last_thruster_norm,
        }

    def _backend_metric_id(self) -> int:
        names = {"zero": 0, "body_pid_wrench": 1}
        return names.get(getattr(self.controller, "name", ""), -1)

    def get_experiment_metadata(self) -> dict:
        normalized_variant = normalize_critic_variant_name(self.config.critic_variant)
        return {
            "controller_backend": getattr(self.controller, "name", self.config.controller_backend),
            "controller_backend_id": self._backend_metric_id(),
            "critic_variant": normalized_variant,
            "critic_variant_id": critic_variant_to_id(normalized_variant),
            "critic_attention_heads": self.config.critic_attention_heads,
            "enable_yaw_control": int(self.config.enable_yaw_control),
            "policy_action_dim": self.config.policy_action_dim,
            "control_interface_version": self.config.control_interface_version,
            "target_body_delta_limit_forward_m": self.config.target_body_delta_limits[0],
            "target_body_delta_limit_up_m": self.config.target_body_delta_limits[1],
            "target_body_delta_limit_left_m": self.config.target_body_delta_limits[2],
            "yaw_error_limit_deg": self.config.yaw_error_limit_deg,
            "physical_wrench_allocator": 1,
            "body_pid_yaw_kp": self.config.body_pid_wrench_controller.yaw_pid_params[0],
        }

    def save(self, path: str):
        torch.save(
            {
                "actor": self.actor.state_dict(),
                "critic": self.critic.state_dict(),
                "actor_optimizer": self.actor_optimizer.state_dict(),
                "critic_optimizer": self.critic_optimizer.state_dict(),
                "config": asdict(self.config),
                "controller_backend": self.config.controller_backend,
                "control_interface_version": self.config.control_interface_version,
            },
            path,
        )

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        checkpoint_interface = checkpoint.get("control_interface_version")
        if checkpoint_interface != self.config.control_interface_version:
            raise ValueError(
                "This checkpoint was created for the retired world-pose position-controller "
                "interface and cannot be loaded into body_subgoal_pid_physical_wrench_v1. "
                "Train a new 3Chase1 position-control policy with the 30D observation contract."
            )
        self._restore_critic_architecture_from_checkpoint(checkpoint)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        if "actor_optimizer" in checkpoint:
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
        if "critic_optimizer" in checkpoint:
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])

    def train(self):
        self.actor.train()
        self.critic.train()

    def eval(self):
        self.actor.eval()
        self.critic.eval()
