"""
Rollout buffer for storing episodes from parallel environments.
"""

import numpy as np
import torch


class RolloutBuffer:
    def __init__(
        self,
        buffer_size,
        num_agents,
        obs_space,
        state_space,
        action_space,
        normalize_reward=False,
        device="cpu",
        role_ids=None,
        obs_chaser_team_dim=None,
        reward_chaser_team_dim=3,
    ):
        """Initialize rollout buffer.

        Args:
            buffer_size: Number of episodes to store
            num_agents: Number of agents
            obs_space: Observation dimension
            state_space: State dimension
            action_space: Action dimension
            normalize_reward: Whether to normalize rewards
            device: Device to store tensors on
            role_ids: Fixed role IDs for all agents [num_agents]
            obs_chaser_team_dim: Dimension of chaser team observations (for 3chase1)
            reward_chaser_team_dim: Dimension of chaser team rewards (for 3chase1)
        """
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0
        self.role_ids = torch.from_numpy(role_ids).long().to(device) if role_ids is not None else None
        self.obs_chaser_team_dim = obs_chaser_team_dim if obs_chaser_team_dim else 39
        self.reward_chaser_team_dim = reward_chaser_team_dim

    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

    def is_full(self):
        """Check if buffer has collected batch_size episodes."""
        return self.pos >= self.buffer_size

    def get_batch(self):
        self.pos = 0
        lengths = [len(episode["obs"]) for episode in self.episodes]
        max_length = max(lengths)
        obs = torch.zeros(
            (self.buffer_size, max_length, self.num_agents, self.obs_space)
        ).to(self.device)
        actions = torch.zeros((self.buffer_size, max_length, self.num_agents, self.action_space)).to(
            self.device
        )
        log_probs = torch.zeros((self.buffer_size, max_length, self.num_agents)).to(
            self.device
        )
        reward = torch.zeros((self.buffer_size, max_length)).to(self.device)
        states = torch.zeros((self.buffer_size, max_length, self.state_space)).to(
            self.device
        )
        done = torch.zeros((self.buffer_size, max_length)).to(self.device)
        mask = torch.zeros(self.buffer_size, max_length, dtype=torch.bool).to(
            self.device
        )
        obs_chaser_team = torch.zeros(
            (self.buffer_size, max_length, self.obs_chaser_team_dim)
        ).to(self.device)
        reward_chaser_team = torch.zeros(
            (self.buffer_size, max_length, self.reward_chaser_team_dim)
        ).to(self.device)

        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            done[i, :length] = self.episodes[i]["done"]
            obs_chaser_team[i, :length] = self.episodes[i]["obs_chaser_team"]
            reward_chaser_team[i, :length] = self.episodes[i]["reward_chaser_team"]
            mask[i, :length] = 1

        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)

        self.episodes = [None] * self.buffer_size
        role_ids_expanded = (
            self.role_ids.unsqueeze(0).expand(self.buffer_size, -1)
            if self.role_ids is not None
            else None
        )
        return (
            obs.float(),
            actions.float(),
            log_probs.float(),
            reward.float(),
            states.float(),
            done.float(),
            mask,
            role_ids_expanded,
            obs_chaser_team.float(),
            reward_chaser_team.float(),
        )
