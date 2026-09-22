"""Homogeneous CTDE MAPPO for the three-ROV TriNetCapture task."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Tuple

import numpy as np
import torch
import torch.optim as optim

from .base import BaseAlgorithm
from .networks.trinet import (
    TRINET_ACTOR_OBS_DIM,
    TRINET_CRITIC_VARIANT_IDS,
    TRINET_PRIVILEGED_STATE_DIM,
    TriNetSharedPositionActor,
    build_trinet_critic,
    normalize_trinet_critic_variant,
)
from .position_control_backends import (
    BodyFramePIDWrenchControllerConfig,
    make_body_frame_position_control_backend,
)


TRINET_ARCHITECTURE_VERSION = "trinet_homogeneous_ctde_v1"
TRINET_NUM_ROVS = 3


@dataclass
class TriNetCaptureMAPPOConfig:
    action_interface: str = "position_controller"
    control_interface_version: str = "body_subgoal_pid_physical_wrench_v1"
    policy_action_dim: int = 4
    rollout_action_dim: int = 4
    env_action_dim: int = 8
    target_body_delta_limits: Tuple[float, float, float] = (1.0, 0.6, 1.0)
    enable_yaw_control: bool = True
    yaw_error_limit_deg: float = 90.0

    optimizer: str = "Adam"
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    epochs: int = 8
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = 0.5
    device: str = "cuda"

    actor_obs_dim: int = TRINET_ACTOR_OBS_DIM
    privileged_state_dim: int = TRINET_PRIVILEGED_STATE_DIM
    critic_variant: str = "trinet_deepsets"
    critic_attention_heads: int = 4
    individual_advantage_weight: float = 0.10
    desired_side_length_m: float = 0.60
    individual_shape_scale: float = 0.02
    individual_action_cost_scale: float = 0.005

    controller_backend: str = "body_pid_wrench"
    body_pid_wrench_controller: BodyFramePIDWrenchControllerConfig = field(
        default_factory=BodyFramePIDWrenchControllerConfig
    )


def _gradient_norm(parameters) -> torch.Tensor:
    gradients = [parameter.grad.detach() for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)
    return torch.linalg.vector_norm(torch.stack([torch.linalg.vector_norm(gradient) for gradient in gradients]))


class TriNetCaptureMAPPOAlgorithm(BaseAlgorithm):
    """One policy for all ROVs, with team and individual value decomposition."""

    action_interface = "position_controller"

    def __init__(self, *, obs_dim: int, action_dim: int, state_dim: int, n_agents: int, role_ids, config: TriNetCaptureMAPPOConfig):
        if obs_dim != config.actor_obs_dim:
            raise ValueError(f"TriNet actor expects {config.actor_obs_dim}D observations, got {obs_dim}.")
        if state_dim != config.privileged_state_dim:
            raise ValueError(f"TriNet critic expects {config.privileged_state_dim}D privileged state, got {state_dim}.")
        if n_agents != TRINET_NUM_ROVS:
            raise ValueError(f"TriNetCapture requires {TRINET_NUM_ROVS} homogeneous ROVs, got {n_agents}.")
        if action_dim != config.env_action_dim:
            raise ValueError(f"TriNetCapture expects {config.env_action_dim}D Unity thruster actions, got {action_dim}.")
        role_ids_np = np.asarray(role_ids, dtype=np.int64).reshape(-1)
        if role_ids_np.shape != (TRINET_NUM_ROVS,) or not np.all(role_ids_np == role_ids_np[0]):
            raise ValueError("TriNetCapture requires one shared role ID for all three ROVs.")

        self.config = config
        self.config.critic_variant = normalize_trinet_critic_variant(config.critic_variant)
        self.device = torch.device(config.device)
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.n_agents = n_agents
        self.role_ids = role_ids_np
        self.rollout_action_dim = config.rollout_action_dim
        self.env_action_dim = config.env_action_dim
        self.policy_action_dim = config.policy_action_dim

        self.actor = TriNetSharedPositionActor(action_dim=config.policy_action_dim).to(self.device)
        self.critic = build_trinet_critic(
            config.critic_variant,
            attention_heads=config.critic_attention_heads,
        ).to(self.device)
        optimizer_type = getattr(optim, config.optimizer)
        self.actor_optimizer = optimizer_type(self.actor.parameters(), lr=config.learning_rate_actor)
        self.critic_optimizer = optimizer_type(self.critic.parameters(), lr=config.learning_rate_critic)
        self.controller = make_body_frame_position_control_backend(
            config.controller_backend,
            body_pid_wrench_config=config.body_pid_wrench_controller,
        )
        self._last_target_body_delta_norm = 0.0
        self._last_target_yaw_abs_deg = 0.0
        self._last_thruster_norm = 0.0

    def reset_controller_state(self, env_mask=None) -> None:
        self.controller.reset(env_mask)

    def select_action(self, obs, role_ids=None, deterministic: bool = False):
        del role_ids
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_tensor.ndim != 3 or obs_tensor.shape[1:] != (TRINET_NUM_ROVS, self.config.actor_obs_dim):
            raise ValueError(f"Expected [batch, 3, {self.config.actor_obs_dim}] TriNet observations, got {tuple(obs_tensor.shape)}")
        with torch.no_grad():
            meta_actions, log_probs = self.actor.act(obs_tensor, deterministic=deterministic)

        meta_actions_np = meta_actions.cpu().numpy()
        body_limits = np.asarray(self.config.target_body_delta_limits, dtype=np.float32).reshape(1, 1, 3)
        body_errors = np.clip(meta_actions_np[..., :3], -1.0, 1.0) * body_limits
        yaw_errors = (
            np.clip(meta_actions_np[..., 3], -1.0, 1.0) * float(self.config.yaw_error_limit_deg)
            if self.config.enable_yaw_control
            else np.zeros(meta_actions_np.shape[:2], dtype=np.float32)
        )
        env_actions = self.controller.act(body_errors, yaw_errors, deterministic=deterministic)
        self._last_target_body_delta_norm = float(np.linalg.norm(body_errors, axis=-1).mean())
        self._last_target_yaw_abs_deg = float(np.abs(yaw_errors).mean())
        self._last_thruster_norm = float(np.linalg.norm(env_actions, axis=-1).mean())
        chosen_heads = np.zeros(meta_actions_np.shape[:2], dtype=np.int64)
        return env_actions, log_probs.cpu().numpy(), chosen_heads, {"buffer_actions": meta_actions_np}

    def _individual_rewards(self, obs: torch.Tensor, actions: torch.Tensor, unity_rewards: torch.Tensor) -> torch.Tensor:
        edge_a = torch.linalg.vector_norm(obs[..., 14:17], dim=-1)
        edge_b = torch.linalg.vector_norm(obs[..., 20:23], dim=-1)
        formation_error = 0.5 * (
            (edge_a - self.config.desired_side_length_m).abs() +
            (edge_b - self.config.desired_side_length_m).abs()
        )
        action_cost = actions.pow(2).mean(dim=-1)
        collision_residual = unity_rewards - unity_rewards.mean(dim=-1, keepdim=True)
        return (
            collision_residual
            - self.config.individual_shape_scale * formation_error
            - self.config.individual_action_cost_scale * action_cost
        )

    def _compute_advantages(self, states, team_rewards, individual_rewards, mask):
        episodes, steps = mask.shape
        with torch.no_grad():
            team_values, individual_values = self.critic(states.reshape(-1, self.state_dim))
            team_values = team_values.reshape(episodes, steps)
            individual_values = individual_values.reshape(episodes, steps, TRINET_NUM_ROVS)

        team_advantages = torch.zeros_like(team_rewards)
        individual_advantages = torch.zeros_like(individual_rewards)
        team_returns = torch.zeros_like(team_rewards)
        individual_returns = torch.zeros_like(individual_rewards)
        lengths = mask.sum(dim=1).detach().cpu().tolist()
        for episode, raw_length in enumerate(lengths):
            length = int(raw_length)
            next_team_advantage = torch.tensor(0.0, device=self.device)
            next_individual_advantage = torch.zeros(TRINET_NUM_ROVS, device=self.device)
            for step in range(length - 1, -1, -1):
                if step == length - 1:
                    next_team_value = torch.tensor(0.0, device=self.device)
                    next_individual_value = torch.zeros(TRINET_NUM_ROVS, device=self.device)
                else:
                    next_team_value = team_values[episode, step + 1]
                    next_individual_value = individual_values[episode, step + 1]
                team_delta = team_rewards[episode, step] + self.config.gamma * next_team_value - team_values[episode, step]
                individual_delta = individual_rewards[episode, step] + self.config.gamma * next_individual_value - individual_values[episode, step]
                next_team_advantage = team_delta + self.config.gamma * self.config.td_lambda * next_team_advantage
                next_individual_advantage = individual_delta + self.config.gamma * self.config.td_lambda * next_individual_advantage
                team_advantages[episode, step] = next_team_advantage
                individual_advantages[episode, step] = next_individual_advantage
            team_returns[episode, :length] = team_advantages[episode, :length] + team_values[episode, :length]
            individual_returns[episode, :length] = individual_advantages[episode, :length] + individual_values[episode, :length]
        return team_advantages, individual_advantages, team_returns, individual_returns

    def update(self, batch):
        (
            b_obs, b_actions, b_log_probs, b_reward, b_states, _b_done, b_mask,
            _b_role_ids, _b_obs_chaser_team, b_reward_chaser_team,
        ) = batch
        if b_obs.shape[-2:] != (TRINET_NUM_ROVS, self.config.actor_obs_dim):
            raise ValueError(f"Unexpected TriNet rollout observation shape {tuple(b_obs.shape)}")
        if b_states.shape[-1] != self.state_dim:
            raise ValueError(f"Unexpected TriNet privileged state shape {tuple(b_states.shape)}")
        if b_reward_chaser_team.shape[-1] != TRINET_NUM_ROVS:
            raise ValueError("TriNet rollout must retain three per-ROV Unity rewards.")

        individual_rewards = self._individual_rewards(b_obs, b_actions, b_reward_chaser_team)
        team_advantage, individual_advantage, team_returns, individual_returns = self._compute_advantages(
            b_states, b_reward, individual_rewards, b_mask
        )
        combined_advantage = team_advantage.unsqueeze(-1) + (
            self.config.individual_advantage_weight * individual_advantage
        )
        if self.config.normalize_advantage:
            valid_advantages = combined_advantage[b_mask]
            combined_advantage = (combined_advantage - valid_advantages.mean()) / (valid_advantages.std(unbiased=False) + 1e-8)

        flat_obs = b_obs.reshape(-1, TRINET_NUM_ROVS, self.config.actor_obs_dim)
        flat_actions = b_actions.reshape(-1, TRINET_NUM_ROVS, self.policy_action_dim)
        flat_old_log_probs = b_log_probs.reshape(-1, TRINET_NUM_ROVS)
        flat_states = b_states.reshape(-1, self.state_dim)
        flat_team_returns = team_returns.reshape(-1)
        flat_individual_returns = individual_returns.reshape(-1, TRINET_NUM_ROVS)
        flat_advantages = combined_advantage.reshape(-1, TRINET_NUM_ROVS)
        valid = b_mask.reshape(-1)

        metrics: dict[str, list[float]] = {key: [] for key in (
            "meta_actor_loss", "critic_loss", "team_critic_loss", "individual_critic_loss",
            "entropy", "kl_divergence", "clipped_ratio", "meta_actor_gradient", "critic_gradient",
        )}
        final_team_values = final_individual_values = None
        for _ in range(self.config.epochs):
            current_log_probs = self.actor.get_log_prob(flat_obs, flat_actions)
            ratio = torch.exp(current_log_probs - flat_old_log_probs)
            surrogate_a = flat_advantages * ratio
            surrogate_b = flat_advantages * torch.clamp(ratio, 1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip)
            actor_loss = -torch.minimum(surrogate_a[valid], surrogate_b[valid]).mean()
            entropy = self.actor.get_entropy(flat_obs)[valid].mean()
            actor_objective = actor_loss - self.config.entropy_coef * entropy

            team_values, individual_values = self.critic(flat_states)
            team_critic_loss = (team_values[valid] - flat_team_returns[valid]).pow(2).mean()
            individual_critic_loss = (individual_values[valid] - flat_individual_returns[valid]).pow(2).mean()
            critic_loss = team_critic_loss + individual_critic_loss

            self.actor_optimizer.zero_grad()
            actor_objective.backward()
            actor_gradient = _gradient_norm(self.actor.parameters())
            if self.config.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.clip_gradients)
            self.actor_optimizer.step()

            self.critic_optimizer.zero_grad()
            critic_loss.backward()
            critic_gradient = _gradient_norm(self.critic.parameters())
            if self.config.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.clip_gradients)
            self.critic_optimizer.step()

            with torch.no_grad():
                kl = ((ratio - 1.0) - (current_log_probs - flat_old_log_probs))[valid].mean()
                clipped = ((ratio - 1.0).abs() > self.config.ppo_clip)[valid].float().mean()
            metrics["meta_actor_loss"].append(float(actor_loss.detach()))
            metrics["critic_loss"].append(float(critic_loss.detach()))
            metrics["team_critic_loss"].append(float(team_critic_loss.detach()))
            metrics["individual_critic_loss"].append(float(individual_critic_loss.detach()))
            metrics["entropy"].append(float(entropy.detach()))
            metrics["kl_divergence"].append(float(kl))
            metrics["clipped_ratio"].append(float(clipped))
            metrics["meta_actor_gradient"].append(float(actor_gradient))
            metrics["critic_gradient"].append(float(critic_gradient))
            final_team_values, final_individual_values = team_values.detach(), individual_values.detach()

        assert final_team_values is not None and final_individual_values is not None
        valid_advantages = combined_advantage[b_mask]
        return {
            **{key: float(np.mean(value)) for key, value in metrics.items()},
            "team_value_mean": float(final_team_values[valid].mean()),
            "individual_value_mean": float(final_individual_values[valid].mean()),
            "team_return_mean": float(flat_team_returns[valid].mean()),
            "individual_return_mean": float(flat_individual_returns[valid].mean()),
            "advantage_mean": float(valid_advantages.mean()),
            "advantage_std": float(valid_advantages.std(unbiased=False)),
            "actor_log_std_mean": float(self.actor.log_std.detach().mean()),
            "actor_std_mean": float(torch.exp(self.actor.log_std.detach()).mean()),
            "target_body_delta_norm_m": self._last_target_body_delta_norm,
            "target_yaw_abs_deg": self._last_target_yaw_abs_deg,
            "thruster_norm": self._last_thruster_norm,
        }

    def get_experiment_metadata(self) -> dict:
        return {
            "trinet_architecture_version": TRINET_ARCHITECTURE_VERSION,
            "trinet_actor_obs_dim": self.config.actor_obs_dim,
            "trinet_privileged_state_dim": self.config.privileged_state_dim,
            "trinet_num_trainable_roles": 1,
            "critic_variant": self.config.critic_variant,
            "critic_variant_id": TRINET_CRITIC_VARIANT_IDS[self.config.critic_variant],
            "critic_attention_heads": self.config.critic_attention_heads,
            "individual_advantage_weight": self.config.individual_advantage_weight,
            "controller_backend": getattr(self.controller, "name", self.config.controller_backend),
            "policy_action_dim": self.config.policy_action_dim,
            "control_interface_version": self.config.control_interface_version,
            "physical_wrench_allocator": 1,
        }

    def save(self, path: str):
        torch.save({
            "trinet_architecture_version": TRINET_ARCHITECTURE_VERSION,
            "actor_observation_dim": self.config.actor_obs_dim,
            "privileged_state_dim": self.config.privileged_state_dim,
            "critic_variant": self.config.critic_variant,
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
            "config": asdict(self.config),
            "control_interface_version": self.config.control_interface_version,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        if checkpoint.get("trinet_architecture_version") != TRINET_ARCHITECTURE_VERSION:
            raise ValueError("Checkpoint is not compatible with the homogeneous TriNetCapture CTDE architecture.")
        if checkpoint.get("actor_observation_dim") != self.config.actor_obs_dim or checkpoint.get("privileged_state_dim") != self.config.privileged_state_dim:
            raise ValueError("TriNetCapture checkpoint observation/state contract does not match this configuration.")
        if normalize_trinet_critic_variant(checkpoint.get("critic_variant", "")) != self.config.critic_variant:
            raise ValueError("TriNetCapture checkpoint critic variant does not match this configuration.")
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


__all__ = ["TriNetCaptureMAPPOAlgorithm", "TriNetCaptureMAPPOConfig", "TRINET_ARCHITECTURE_VERSION"]
