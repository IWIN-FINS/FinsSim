"""
MAPPO (Multi-Agent Proximal Policy Optimization) algorithm with Simple Hierarchical Velocity Control.
简单分层架构：上层输出6D速度(vx,vy,vz,wx,wy,wz)，下层完全独立网络转换为8D推进器推力。

Reference: https://arxiv.org/abs/2103.01955

架构设计：
- Meta Actor/Critic: 输出6D速度目标 (vx, vy, vz, wx, wy, wz)
- Primitive Actor: 独立网络，接收速度目标 + 观测，输出8D推进器推力（可从预训练模型加载）
- 上下层完全分离，无共享编码器
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from .base import BaseAlgorithm
from .networks.actors import ROLE_MAPPING, ID_MAPPING, TRAINABLE_ROLES


@dataclass
class MAPPOMultiHeadVelocityConfig:
    """Configuration for Simple Hierarchical MAPPO with Velocity Control."""
    # Meta (上层) 策略配置 - 输出速度
    meta_hidden_dim: int = 64
    meta_num_layers: int = 2
    meta_action_dim: int = 6  # (vx, vy, vz, wx, wy, wz)

    # Primitive (下层) 策略配置 - 输出推力
    primitive_hidden_dim: int = 128
    primitive_num_layers: int = 3
    primitive_action_dim: int = 8  # 8个推进器推力

    # 优化器
    optimizer: str = "Adam"
    learning_rate_meta_actor: float = 0.0005
    learning_rate_meta_critic: float = 0.0005
    learning_rate_primitive: float = 0.0008

    # PPO参数
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_return: bool = True
    epochs: int = 8
    ppo_clip: float = 0.2
    entropy_coef: float = 0.01
    clip_gradients: float = -1
    device: str = "cuda"

    # 3chase1 specific
    chaser_team_obs_dim: int = 30
    chaser_team_action_dim: int = 8
    num_trainable_roles: int = 2


class SimpleMetaActor(nn.Module):
    """简单的Meta Actor网络 - 输出6D速度目标

    接收观测，输出期望的线速度(vx,vy,vz)和角速度(wx,wy,wz)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        action_dim: int = 6,
        num_roles: int = 2,
        agent_id_dim: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim

        # 编码器
        input_dim = obs_dim + agent_id_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.encoder.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        # 角色头
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
        """构建agent ID one-hot编码。"""
        agent_ids_onehot = torch.zeros(batch_size, num_agents, self.agent_id_dim, device=device)
        for b in range(batch_size):
            for a in range(num_agents):
                role = role_ids[b, a]
                if ID_MAPPING[role.item()] in TRAINABLE_ROLES:
                    agent_ids_onehot[b, a, role] = 1.0
        return agent_ids_onehot

    def act(self, obs, role_ids):
        """采样Meta动作(速度)。

        Args:
            obs: [batch_size, num_agents, obs_dim] 观测
            role_ids: [batch_size, num_agents] 角色ID
        Returns:
            meta_actions: [batch_size, num_agents, action_dim] 6D速度
            log_probs: [batch_size, num_agents] 对数概率
            chosen_heads: [batch_size, num_agents] 角色头索引
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        actions = torch.zeros(batch_size, num_agents, self.action_dim, device=device)
        log_probs = torch.zeros(batch_size, num_agents, device=device)
        chosen_heads = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                mean = torch.tanh(self.role_heads[role_idx](role_features))
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                sampled_actions = torch.clamp(dist.sample(), -1.0, 1.0)
                lp = dist.log_prob(sampled_actions).sum(dim=-1)

                actions[mask] = sampled_actions
                log_probs[mask] = lp
                chosen_heads[mask] = role_idx

        return actions, log_probs, chosen_heads

    def get_log_prob(self, obs, role_ids, actions):
        """计算对数概率。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        log_probs = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                mean = torch.tanh(self.role_heads[role_idx](role_features))
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                lp = dist.log_prob(actions[mask]).sum(dim=-1)
                log_probs[mask] = lp

        return log_probs

    def get_entropy(self, obs, role_ids):
        """计算策略熵。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        entropy = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                mean = torch.tanh(self.role_heads[role_idx](role_features))
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                ent = dist.entropy().sum(dim=-1)
                entropy[mask] = ent

        return entropy


class SimpleMetaCritic(nn.Module):
    """简单的Meta Critic网络 - 估计状态价值"""

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_roles: int = 2,
        agent_id_dim: int = 3,
    ):
        super().__init__()
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim

        input_dim = obs_dim + agent_id_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.encoder.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.role_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.role_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
            )

    def _build_agent_id_onehot(self, batch_size, num_agents, role_ids, device):
        """构建agent ID one-hot编码。"""
        agent_ids_onehot = torch.zeros(batch_size, num_agents, self.agent_id_dim, device=device)
        for b in range(batch_size):
            for a in range(num_agents):
                role = role_ids[b, a]
                if ID_MAPPING[role.item()] in TRAINABLE_ROLES:
                    agent_ids_onehot[b, a, role] = 1.0
        return agent_ids_onehot

    def forward(self, obs, role_ids):
        """计算价值估计。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        values = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                role_values = self.role_heads[role_idx](role_features).squeeze(-1)
                values[mask] = role_values

        return values


class PrimitiveActor(nn.Module):
    """独立的Primitive Actor网络 - 将速度目标转换为推力

    输入: 观测 + 速度目标(vx,vy,vz,wx,wy,wz)
    输出: 8维推进器推力
    """

    def __init__(
        self,
        obs_dim: int,
        meta_action_dim: int,
        hidden_dim: int,
        num_layers: int,
        action_dim: int = 8,
        num_roles: int = 2,
        agent_id_dim: int = 3,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim

        # 输入: obs + meta_action + agent_id
        input_dim = obs_dim + meta_action_dim + agent_id_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.encoder.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

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
        """构建agent ID one-hot编码。"""
        agent_ids_onehot = torch.zeros(batch_size, num_agents, self.agent_id_dim, device=device)
        for b in range(batch_size):
            for a in range(num_agents):
                role = role_ids[b, a]
                if ID_MAPPING[role.item()] in TRAINABLE_ROLES:
                    agent_ids_onehot[b, a, role] = 1.0
        return agent_ids_onehot

    def act(self, obs, meta_actions, role_ids):
        """采样Primitive动作(推力)。

        Args:
            obs: [batch_size, num_agents, obs_dim] 观测
            meta_actions: [batch_size, num_agents, meta_action_dim] 速度目标
            role_ids: [batch_size, num_agents] 角色ID
        Returns:
            primitive_actions: [batch_size, num_agents, action_dim] 8D推力
            log_probs: [batch_size, num_agents] 对数概率
            chosen_heads: [batch_size, num_agents] 角色头索引
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, meta_actions, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        actions = torch.zeros(batch_size, num_agents, self.action_dim, device=device)
        log_probs = torch.zeros(batch_size, num_agents, device=device)
        chosen_heads = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                mean = torch.tanh(self.role_heads[role_idx](role_features))
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                sampled_actions = torch.clamp(dist.sample(), -1.0, 1.0)
                lp = dist.log_prob(sampled_actions).sum(dim=-1)

                actions[mask] = sampled_actions
                log_probs[mask] = lp
                chosen_heads[mask] = role_idx

        return actions, log_probs, chosen_heads

    def get_log_prob(self, obs, meta_actions, role_ids, actions):
        """计算对数概率。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, meta_actions, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        log_probs = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                mean = torch.tanh(self.role_heads[role_idx](role_features))
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                lp = dist.log_prob(actions[mask]).sum(dim=-1)
                log_probs[mask] = lp

        return log_probs

    def get_entropy(self, obs, meta_actions, role_ids):
        """计算策略熵。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, meta_actions, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        entropy = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                mean = torch.tanh(self.role_heads[role_idx](role_features))
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                ent = dist.entropy().sum(dim=-1)
                entropy[mask] = ent

        return entropy


class PrimitiveCritic(nn.Module):
    """独立的Primitive Critic网络 - 估计状态+动作价值"""

    def __init__(
        self,
        obs_dim: int,
        meta_action_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_roles: int = 2,
        agent_id_dim: int = 3,
    ):
        super().__init__()
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim

        input_dim = obs_dim + meta_action_dim + agent_id_dim
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.encoder.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        self.role_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.role_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
            )

    def _build_agent_id_onehot(self, batch_size, num_agents, role_ids, device):
        """构建agent ID one-hot编码。"""
        agent_ids_onehot = torch.zeros(batch_size, num_agents, self.agent_id_dim, device=device)
        for b in range(batch_size):
            for a in range(num_agents):
                role = role_ids[b, a]
                if ID_MAPPING[role.item()] in TRAINABLE_ROLES:
                    agent_ids_onehot[b, a, role] = 1.0
        return agent_ids_onehot

    def forward(self, obs, meta_actions, role_ids):
        """计算价值估计。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, meta_actions, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        features = self.encoder(x)
        features = features.reshape(batch_size, num_agents, -1)

        values = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = features[mask]
                role_values = self.role_heads[role_idx](role_features).squeeze(-1)
                values[mask] = role_values

        return values


class MAPPOMultiHeadVelocityAlgorithm(BaseAlgorithm):
    """简单分层MAPPO算法 - 3chase1场景

    上层(Meta)策略输出6D速度目标(vx,vy,vz,wx,wy,wz)
    下层(Primitive)策略将速度目标转换为8D推进器推力
    上下层完全分离，Primitive网络可从预训练模型加载。

    Uses centralized training with decentralized execution (CTDE).
    """

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        role_ids: np.ndarray,
        config: MAPPOMultiHeadVelocityConfig,
    ):
        """
        Initialize Simple Hierarchical MAPPO algorithm.

        Args:
            obs_dim: Observation dimension per agent (after padding)
            action_dim: Action dimension per agent (for compatibility)
            state_dim: Global state dimension
            n_agents: Total number of agents
            role_ids: Role ID for each agent [n_agents]
            config: Hierarchical MAPPO configuration
        """
        self.config = config
        self.n_agents = n_agents
        self.role_ids = role_ids
        self.device = torch.device(config.device)

        # Meta (上层) 网络
        self.meta_actor = SimpleMetaActor(
            obs_dim=config.chaser_team_obs_dim,
            hidden_dim=config.meta_hidden_dim,
            num_layers=config.meta_num_layers,
            action_dim=config.meta_action_dim,
            num_roles=config.num_trainable_roles,
            agent_id_dim=len(ROLE_MAPPING),
        ).to(self.device)

        self.meta_critic = SimpleMetaCritic(
            obs_dim=config.chaser_team_obs_dim,
            hidden_dim=config.meta_hidden_dim,
            num_layers=config.meta_num_layers,
            num_roles=config.num_trainable_roles,
            agent_id_dim=len(ROLE_MAPPING),
        ).to(self.device)

        # Primitive (下层) 网络 - 独立的，可加载预训练权重
        self.primitive_actor = PrimitiveActor(
            obs_dim=config.chaser_team_obs_dim,
            meta_action_dim=config.meta_action_dim,
            hidden_dim=config.primitive_hidden_dim,
            num_layers=config.primitive_num_layers,
            action_dim=config.primitive_action_dim,
            num_roles=config.num_trainable_roles,
            agent_id_dim=len(ROLE_MAPPING),
        ).to(self.device)

        self.primitive_critic = PrimitiveCritic(
            obs_dim=config.chaser_team_obs_dim,
            meta_action_dim=config.meta_action_dim,
            hidden_dim=config.primitive_hidden_dim,
            num_layers=config.primitive_num_layers,
            num_roles=config.num_trainable_roles,
            agent_id_dim=len(ROLE_MAPPING),
        ).to(self.device)

        # 优化器
        Optimizer = getattr(optim, config.optimizer)
        self.meta_actor_optimizer = Optimizer(
            self.meta_actor.parameters(), lr=config.learning_rate_meta_actor
        )
        self.meta_critic_optimizer = Optimizer(
            self.meta_critic.parameters(), lr=config.learning_rate_meta_critic
        )
        self.primitive_optimizer = Optimizer(
            list(self.primitive_actor.parameters()) + list(self.primitive_critic.parameters()),
            lr=config.learning_rate_primitive
        )

        # 预计算actor indices (非Prey agents)
        self.actor_indices = self._compute_actor_indices()

    def _compute_actor_indices(self):
        """预计算actor索引(非Prey agents)。"""
        role_ids_batch = np.tile(self.role_ids, (1, 1))
        actor_indices = np.where(
            np.any(role_ids_batch != ROLE_MAPPING['Prey'], axis=0)
        )[0]
        return torch.from_numpy(actor_indices).long().to(self.device)

    def select_action(self, obs, role_ids, deterministic=False):
        """选择动作(用于rollout收集)。

        Args:
            obs: Observations [batch_size, n_agents, obs_dim]
            role_ids: Role IDs [batch_size, n_agents]
            deterministic: If True, use mean action (for evaluation)

        Returns:
            primitive_actions: [batch_size, n_agents, action_dim] 8D推力
            meta_actions: [batch_size, n_agents, 6D] 速度目标(额外返回)
            log_probs: [batch_size, n_agents]
            chosen_heads: [batch_size, n_agents]
        """
        obs_tensor = torch.from_numpy(obs).float().to(self.device)
        role_ids_tensor = torch.from_numpy(role_ids).long().to(self.device)

        with torch.no_grad():
            # 提取Herder/Netter观测
            obs_actor = obs_tensor[:, self.actor_indices, :self.config.chaser_team_obs_dim]
            role_ids_actor = role_ids_tensor[:, self.actor_indices]

            # 上层: 生成速度目标
            meta_actions_actor, meta_log_probs_actor, _ = self.meta_actor.act(
                obs_actor, role_ids_actor
            )

            # 下层: 将速度转换为推力
            primitive_actions_actor, primitive_log_probs_actor, chosen_heads = self.primitive_actor.act(
                obs_actor, meta_actions_actor, role_ids_actor
            )

        # 重构完整动作数组
        batch_size = obs_tensor.size(0)
        n_agents = self.n_agents
        primitive_actions = torch.zeros(batch_size, n_agents, self.config.primitive_action_dim, device=self.device)
        meta_actions_full = torch.zeros(batch_size, n_agents, self.config.meta_action_dim, device=self.device)
        log_probs = torch.zeros(batch_size, n_agents, device=self.device)
        chosen_heads_full = torch.zeros(batch_size, n_agents, dtype=torch.long, device=self.device)

        primitive_actions[:, self.actor_indices] = primitive_actions_actor
        meta_actions_full[:, self.actor_indices] = meta_actions_actor
        log_probs[:, self.actor_indices] = primitive_log_probs_actor
        chosen_heads_full[:, self.actor_indices] = chosen_heads

        return (
            primitive_actions.cpu().numpy(),
            meta_actions_full.cpu().numpy(),
            log_probs.cpu().numpy(),
            chosen_heads_full.cpu().numpy(),
        )

    def update(self, batch):
        """更新策略。

        Args:
            batch: Tuple from RolloutBuffer.get_batch()

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

        # TD(λ) returns and advantages
        n_agents = len(self.actor_indices)
        return_lambda = torch.zeros(
            (b_actions.size(0), b_actions.size(1), n_agents)
        ).float().to(device)
        advantages = torch.zeros(
            (b_actions.size(0), b_actions.size(1), n_agents)
        ).float().to(device)

        # 预计算chaser team顺序
        env_role_ids = self.role_ids
        chaser_env_positions = [i for i, rid in enumerate(env_role_ids) if rid != ROLE_MAPPING['Prey']]
        chaser_agent_ids = [env_role_ids[i] for i in chaser_env_positions]
        herder_pos = chaser_env_positions[chaser_agent_ids.index(ROLE_MAPPING['Herder'])] \
            if ROLE_MAPPING['Herder'] in chaser_agent_ids else None
        netter_positions = [
            (chaser_env_positions[i], i)
            for i, rid in enumerate(chaser_agent_ids)
            if rid == ROLE_MAPPING['Netter']
        ]
        netter_positions.sort(key=lambda x: env_role_ids[x[0]])
        chaser_team_order = (
            [herder_pos] + [pos for pos, _ in netter_positions]
            if herder_pos is not None
            else [pos for pos, _ in netter_positions]
        )
        chaser_team_order_tensor = torch.tensor(chaser_team_order, device=device).long()

        # 生成meta actions用于critic更新
        b_meta_actions = torch.zeros(
            b_actions.size(0), b_actions.size(1), n_agents, config.meta_action_dim
        ).float().to(device)

        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()

                for t in range(int(ep_len)):
                    obs_chaser_t = b_obs_chaser_team[ep_idx, t].reshape(3, config.chaser_team_obs_dim).unsqueeze(0)
                    role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)

                    meta_actions_t, _, _ = self.meta_actor.act(obs_chaser_t, role_ids_t)
                    b_meta_actions[ep_idx, t] = meta_actions_t.squeeze(0)

            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                last_return_lambda = torch.zeros(3, device=device)

                for t in reversed(range(int(ep_len))):
                    obs_chaser_t = b_obs_chaser_team[ep_idx, t].reshape(3, config.chaser_team_obs_dim).unsqueeze(0)
                    role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
                    meta_actions_t = b_meta_actions[ep_idx, t].unsqueeze(0)
                    reward_chaser_t = b_reward_chaser_team[ep_idx, t]

                    if t == int(ep_len - 1):
                        next_value = torch.zeros(3, device=device)
                    else:
                        next_obs_chaser_t = b_obs_chaser_team[ep_idx, t + 1].reshape(3, config.chaser_team_obs_dim).unsqueeze(0)
                        next_role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
                        next_meta_actions_t = b_meta_actions[ep_idx, t + 1].unsqueeze(0)
                        next_value = self.primitive_critic(
                            obs=next_obs_chaser_t,
                            meta_actions=next_meta_actions_t,
                            role_ids=next_role_ids_t
                        )

                    current_value = self.primitive_critic(
                        obs=obs_chaser_t,
                        meta_actions=meta_actions_t,
                        role_ids=role_ids_t
                    )

                    return_lambda[ep_idx, t, :3] = last_return_lambda = reward_chaser_t + config.gamma * (
                        config.td_lambda * last_return_lambda
                        + (1 - config.td_lambda) * next_value
                    )
                    advantages[ep_idx, t, :3] = return_lambda[ep_idx, t, :3] - current_value

        # 标准化advantages
        advantages_chaser = advantages[:, :, :3]
        return_lambda_chaser = return_lambda[:, :, :3]

        if config.normalize_advantage:
            adv_mu = advantages_chaser[b_mask].mean()
            adv_std = advantages_chaser[b_mask].std()
            advantages[:, :, :3] = (advantages_chaser - adv_mu) / (adv_std + 1e-8)

        if config.normalize_return:
            ret_mu = return_lambda_chaser[b_mask].mean()
            ret_std = return_lambda_chaser[b_mask].std()
            return_lambda[:, :, :3] = (return_lambda_chaser - ret_mu) / (ret_std + 1e-8)

        # 训练循环
        meta_actor_losses = []
        primitive_actor_losses = []
        meta_critic_losses = []
        primitive_critic_losses = []
        meta_entropies = []
        primitive_entropies = []
        meta_clipped_ratios = []
        primitive_clipped_ratios = []

        for epoch_idx in range(config.epochs):
            epoch_pbar = tqdm(total=b_obs.size(1), desc=f"Epoch {epoch_idx+1}/{config.epochs}", leave=False)
            meta_actor_loss = 0
            primitive_actor_loss = 0
            meta_critic_loss = 0
            primitive_critic_loss = 0
            meta_entropy_sum = 0
            primitive_entropy_sum = 0
            meta_clipped_ratio = 0
            primitive_clipped_ratio = 0

            for t in range(b_obs.size(1)):
                epoch_pbar.update(1)
                obs_actor_t = b_obs[:, t, self.actor_indices, :config.chaser_team_obs_dim]
                actions_actor_t = b_actions[:, t, self.actor_indices, :]
                log_probs_actor_t = b_log_probs[:, t, self.actor_indices]
                role_ids_actor_t = b_role_ids[:, chaser_team_order_tensor]
                meta_actions_t = b_meta_actions[:, t]

                valid_mask = b_mask[:, t]
                advantages_actor = advantages[:, t, :3]

                # 获取当前meta actions
                current_meta_actions, current_meta_lp, _ = self.meta_actor.act(obs_actor_t, role_ids_actor_t)

                # 获取当前primitive log probs
                current_primitive_lp = self.primitive_actor.get_log_prob(
                    obs_actor_t, current_meta_actions, role_ids_actor_t, actions_actor_t
                )

                # ==== Meta (上层) 策略损失 ====
                meta_ratio = torch.exp(current_meta_lp - log_probs_actor_t)
                meta_pg_loss1 = advantages_actor * meta_ratio
                meta_pg_loss2 = advantages_actor * torch.clamp(meta_ratio, 1 - config.ppo_clip, 1 + config.ppo_clip)
                meta_pg_loss = (
                    torch.min(meta_pg_loss1[valid_mask], meta_pg_loss2[valid_mask])
                    .mean(dim=-1)
                    .sum()
                )

                meta_entropy = self.meta_actor.get_entropy(obs_actor_t, role_ids_actor_t)
                meta_entropy_loss = meta_entropy[valid_mask].mean(dim=-1).sum()
                meta_entropy_sum += meta_entropy_loss

                meta_actor_loss += -meta_pg_loss - config.entropy_coef * meta_entropy_loss

                # ==== Primitive (下层) 策略损失 ====
                primitive_ratio = torch.exp(current_primitive_lp - log_probs_actor_t)
                primitive_pg_loss1 = advantages_actor * primitive_ratio
                primitive_pg_loss2 = advantages_actor * torch.clamp(
                    primitive_ratio, 1 - config.ppo_clip, 1 + config.ppo_clip
                )
                primitive_pg_loss = (
                    torch.min(primitive_pg_loss1[valid_mask], primitive_pg_loss2[valid_mask])
                    .mean(dim=-1)
                    .sum()
                )

                primitive_entropy = self.primitive_actor.get_entropy(
                    obs_actor_t, current_meta_actions, role_ids_actor_t
                )
                primitive_entropy_loss = primitive_entropy[valid_mask].mean(dim=-1).sum()
                primitive_entropy_sum += primitive_entropy_loss

                primitive_actor_loss += -primitive_pg_loss - config.entropy_coef * primitive_entropy_loss

                # ==== Critic损失 ====
                current_meta_values = self.meta_critic(obs=obs_actor_t, role_ids=role_ids_actor_t)
                current_primitive_values = self.primitive_critic(
                    obs=obs_actor_t, meta_actions=current_meta_actions, role_ids=role_ids_actor_t
                )

                return_lambda_actor = return_lambda[:, t, :3]
                weight = torch.tensor([0.4, 0.3, 0.3], device=device)

                meta_value_loss = (((current_meta_values - return_lambda_actor) ** 2) * weight).sum()
                primitive_value_loss = (((current_primitive_values - return_lambda_actor) ** 2) * weight).sum()

                meta_critic_loss += meta_value_loss
                primitive_critic_loss += primitive_value_loss

                # Clipped ratios
                meta_clipped_ratio += (
                    ((meta_ratio - 1.0).abs() > config.ppo_clip)[valid_mask]
                    .float()
                    .mean(dim=-1)
                    .sum()
                )
                primitive_clipped_ratio += (
                    ((primitive_ratio - 1.0).abs() > config.ppo_clip)[valid_mask]
                    .float()
                    .mean(dim=-1)
                    .sum()
                )

            # 归一化
            actor_count = b_mask.sum()
            meta_actor_loss /= actor_count
            primitive_actor_loss /= actor_count
            meta_critic_loss /= b_mask.sum()
            primitive_critic_loss /= b_mask.sum()
            meta_entropy_sum /= actor_count
            primitive_entropy_sum /= actor_count
            meta_clipped_ratio /= actor_count
            primitive_clipped_ratio /= actor_count

            epoch_pbar.close()

            # 更新网络
            self.meta_actor_optimizer.zero_grad()
            self.meta_critic_optimizer.zero_grad()
            self.primitive_optimizer.zero_grad()

            meta_actor_loss.backward()
            primitive_actor_loss.backward()
            meta_critic_loss.backward()
            primitive_critic_loss.backward()

            if config.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(self.meta_actor.parameters(), max_norm=config.clip_gradients)
                torch.nn.utils.clip_grad_norm_(self.primitive_actor.parameters(), max_norm=config.clip_gradients)

            self.meta_actor_optimizer.step()
            self.meta_critic_optimizer.step()
            self.primitive_optimizer.step()

            meta_actor_losses.append(meta_actor_loss.item())
            primitive_actor_losses.append(primitive_actor_loss.item())
            meta_critic_losses.append(meta_critic_loss.item())
            primitive_critic_losses.append(primitive_critic_loss.item())
            meta_entropies.append(meta_entropy_sum.item())
            primitive_entropies.append(primitive_entropy_sum.item())
            meta_clipped_ratios.append(meta_clipped_ratio.cpu().item())
            primitive_clipped_ratios.append(primitive_clipped_ratio.cpu().item())

        return {
            "meta_actor_loss": np.mean(meta_actor_losses),
            "primitive_actor_loss": np.mean(primitive_actor_losses),
            "meta_critic_loss": np.mean(meta_critic_losses),
            "primitive_critic_loss": np.mean(primitive_critic_losses),
            "meta_entropy": np.mean(meta_entropies),
            "primitive_entropy": np.mean(primitive_entropies),
            "meta_clipped_ratio": np.mean(meta_clipped_ratios),
            "primitive_clipped_ratio": np.mean(primitive_clipped_ratios),
        }

    def save(self, path: str):
        """保存模型checkpoint。"""
        torch.save({
            "meta_actor": self.meta_actor.state_dict(),
            "meta_critic": self.meta_critic.state_dict(),
            "primitive_actor": self.primitive_actor.state_dict(),
            "primitive_critic": self.primitive_critic.state_dict(),
            "meta_actor_optimizer": self.meta_actor_optimizer.state_dict(),
            "meta_critic_optimizer": self.meta_critic_optimizer.state_dict(),
            "primitive_optimizer": self.primitive_optimizer.state_dict(),
        }, path)

    def load(self, path: str, load_primitive: bool = True):
        """加载模型checkpoint。

        Args:
            path: Checkpoint路径
            load_primitive: 是否加载下层网络（用于从预训练primitive网络加载）
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_actor.load_state_dict(checkpoint["meta_actor"])
        self.meta_critic.load_state_dict(checkpoint["meta_critic"])

        if load_primitive and "primitive_actor" in checkpoint:
            self.primitive_actor.load_state_dict(checkpoint["primitive_actor"])
            self.primitive_critic.load_state_dict(checkpoint["primitive_critic"])

        if "meta_actor_optimizer" in checkpoint:
            self.meta_actor_optimizer.load_state_dict(checkpoint["meta_actor_optimizer"])
            self.meta_critic_optimizer.load_state_dict(checkpoint["meta_critic_optimizer"])
            self.primitive_optimizer.load_state_dict(checkpoint["primitive_optimizer"])

    def load_primitive_only(self, path: str):
        """仅加载下层(Primitive)网络权重。

        用于从已经训练好的单独网络加载（如mappo_multihead训练好的actor）。

        Args:
            path: 预训练的checkpoint路径
        """
        checkpoint = torch.load(path, map_location=self.device)

        # 尝试从checkpoint中提取primitive网络
        # 假设预训练checkpoint的格式与mappo_multihead兼容
        if "actor" in checkpoint:
            self.primitive_actor.load_state_dict(checkpoint["actor"])
        if "critic" in checkpoint:
            self.primitive_critic.load_state_dict(checkpoint["critic"])

    def train(self):
        """设置为训练模式。"""
        self.meta_actor.train()
        self.meta_critic.train()
        self.primitive_actor.train()
        self.primitive_critic.train()

    def eval(self):
        """设置为评估模式。"""
        self.meta_actor.eval()
        self.meta_critic.eval()
        self.primitive_actor.eval()
        self.primitive_critic.eval()
