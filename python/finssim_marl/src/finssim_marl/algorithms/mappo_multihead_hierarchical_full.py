"""
MAPPO (Multi-Agent Proximal Policy Optimization) algorithm with Hierarchical Velocity Control.
分层强化学习架构：上层输出6维速度(vx,vy,vz,wx,wy,wz)，下层转换为8维推进器推力。

Reference: https://arxiv.org/abs/2103.01955

架构设计：
- Meta Actor/Critic: 输出6D速度目标 (vx, vy, vz, wx, wy, wz)
- Primitive Actor/Critic: 接收速度目标 + 观测，输出8D推进器推力
- 上下层共享编码器，下层额外接收上层动作作为输入
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
from typing import Optional
from tqdm import tqdm

from .base import BaseAlgorithm
from .networks.actors import ActorMultiHead, ROLE_MAPPING, ID_MAPPING, TRAINABLE_ROLES, get_role_from_agent_name
from .networks.critics import CriticMultiHead
from .networks.rollout_buffer import RolloutBuffer


@dataclass
class MAPPOMultiHeadVelocityConfig:
    """Configuration for Hierarchical MAPPO with Velocity Control."""
    # 编码器配置
    encoder_hidden_dim: int = 128
    encoder_num_layers: int = 3

    # Meta (上层) 策略配置 - 输出速度
    meta_action_dim: int = 6  # (vx, vy, vz, wx, wy, wz)

    # Primitive (下层) 策略配置 - 输出推力
    primitive_action_dim: int = 8  # 8个推进器推力

    # 优化器
    optimizer: str = "Adam"
    learning_rate_meta_actor: float = 0.0005
    learning_rate_meta_critic: float = 0.0005
    learning_rate_primitive_actor: float = 0.0008
    learning_rate_primitive_critic: float = 0.0008

    # PPO参数
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_return: bool = True
    epochs: int = 8
    ppo_clip: float = 0.2
    entropy_coef_meta: float = 0.001
    entropy_coef_primitive: float = 0.01
    clip_gradients: float = -1
    device: str = "cuda"

    # 3chase1 specific
    chaser_team_obs_dim: int = 30
    chaser_team_action_dim: int = 8  # 原始8维推力动作
    num_trainable_roles: int = 2  # Herder and Netter


def _norm_d(grads, d):
    """Compute norm of gradients."""
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


class HierarchicalMultiHeadActor(nn.Module):
    """分层Actor网络：共享编码器 + Meta头(速度) + Primitive头(推力)

    架构：
        obs -> Shared Encoder -> shared_feature
                                      |
                    ┌─────────────────┴─────────────────┐
                    ▼                                   ▼
            Meta Head (6D)                      Primitive Body
            [vx,vy,vz,wx,wy,wz]           (shared_feat + meta_action)
                                            │
                                            ▼
                                    Primitive Head (8D)
                                    (8个推进器推力)
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        meta_action_dim: int,  # 6D velocity
        primitive_action_dim: int,  # 8D thrust
        num_roles: int = 2,
        agent_id_dim: int = 3,
    ):
        super().__init__()
        self.meta_action_dim = meta_action_dim
        self.primitive_action_dim = primitive_action_dim
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim

        # 共享编码器
        input_dim = obs_dim + agent_id_dim
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.shared_body.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        # Meta头 - 输出6D速度
        self.meta_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.meta_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, meta_action_dim)
                )
            )

        # Primitive身体 - 接收 shared_feat + meta_action(6D)
        primitive_input_dim = hidden_dim + meta_action_dim
        self.primitive_body = nn.Sequential(
            nn.Linear(primitive_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Primitive头 - 输出8D推力
        self.primitive_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.primitive_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, primitive_action_dim)
                )
            )

        # 独立的对数标准差参数
        self.log_stds_meta = nn.ParameterList([
            nn.Parameter(torch.zeros(meta_action_dim)) for _ in range(num_roles)
        ])
        self.log_stds_primitive = nn.ParameterList([
            nn.Parameter(torch.zeros(primitive_action_dim)) for _ in range(num_roles)
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
        """采样Meta动作(速度)并同时计算Primitive动作(推力)。

        Args:
            obs: [batch_size, num_agents, obs_dim] 观测
            role_ids: [batch_size, num_agents] 角色ID
        Returns:
            meta_actions: [batch_size, num_agents, meta_action_dim] 6D速度
            primitive_actions: [batch_size, num_agents, primitive_action_dim] 8D推力
            meta_log_probs: [batch_size, num_agents] Meta策略对数概率
            primitive_log_probs: [batch_size, num_agents] Primitive策略对数概率
            chosen_heads: [batch_size, num_agents] 使用的角色头索引
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        meta_actions = torch.zeros(batch_size, num_agents, self.meta_action_dim, device=device)
        primitive_actions = torch.zeros(batch_size, num_agents, self.primitive_action_dim, device=device)
        meta_log_probs = torch.zeros(batch_size, num_agents, device=device)
        primitive_log_probs = torch.zeros(batch_size, num_agents, device=device)
        chosen_heads = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]

                # Meta头 - 输出速度
                meta_mean = torch.tanh(self.meta_heads[role_idx](role_features))
                meta_std = torch.exp(self.log_stds_meta[role_idx]).expand_as(meta_mean)
                meta_dist = torch.distributions.Normal(meta_mean, meta_std)
                sampled_meta = torch.clamp(meta_dist.sample(), -1.0, 1.0)
                meta_lp = meta_dist.log_prob(sampled_meta).sum(dim=-1)

                # Primitive身体 - 接收速度目标
                primitive_input = torch.cat([role_features, sampled_meta], dim=-1)
                primitive_features = self.primitive_body(primitive_input)

                # Primitive头 - 输出推力
                primitive_mean = torch.tanh(self.primitive_heads[role_idx](primitive_features))
                primitive_std = torch.exp(self.log_stds_primitive[role_idx]).expand_as(primitive_mean)
                primitive_dist = torch.distributions.Normal(primitive_mean, primitive_std)
                sampled_primitive = torch.clamp(primitive_dist.sample(), -1.0, 1.0)
                primitive_lp = primitive_dist.log_prob(sampled_primitive).sum(dim=-1)

                meta_actions[mask] = sampled_meta
                primitive_actions[mask] = sampled_primitive
                meta_log_probs[mask] = meta_lp
                primitive_log_probs[mask] = primitive_lp
                chosen_heads[mask] = role_idx

        return meta_actions, primitive_actions, meta_log_probs, primitive_log_probs, chosen_heads

    def get_log_prob(self, obs, role_ids, meta_actions, primitive_actions):
        """计算Meta和Primitive动作的对数概率。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        meta_log_probs = torch.zeros(batch_size, num_agents, device=device)
        primitive_log_probs = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]

                # Meta对数概率
                meta_mean = torch.tanh(self.meta_heads[role_idx](role_features))
                meta_std = torch.exp(self.log_stds_meta[role_idx]).expand_as(meta_mean)
                meta_dist = torch.distributions.Normal(meta_mean, meta_std)
                meta_lp = meta_dist.log_prob(meta_actions[mask]).sum(dim=-1)

                # Primitive对数概率
                primitive_input = torch.cat([role_features, meta_actions[mask]], dim=-1)
                primitive_features = self.primitive_body(primitive_input)
                primitive_mean = torch.tanh(self.primitive_heads[role_idx](primitive_features))
                primitive_std = torch.exp(self.log_stds_primitive[role_idx]).expand_as(primitive_mean)
                primitive_dist = torch.distributions.Normal(primitive_mean, primitive_std)
                primitive_lp = primitive_dist.log_prob(primitive_actions[mask]).sum(dim=-1)

                meta_log_probs[mask] = meta_lp
                primitive_log_probs[mask] = primitive_lp

        return meta_log_probs, primitive_log_probs

    def get_entropy(self, obs, role_ids, meta_actions=None):
        """计算策略熵。"""
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        meta_entropy = torch.zeros(batch_size, num_agents, device=device)
        primitive_entropy = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]

                # Meta熵
                meta_mean = torch.tanh(self.meta_heads[role_idx](role_features))
                meta_std = torch.exp(self.log_stds_meta[role_idx]).expand_as(meta_mean)
                meta_dist = torch.distributions.Normal(meta_mean, meta_std)
                meta_ent = meta_dist.entropy().sum(dim=-1)

                # Primitive熵 (需要meta_actions来计算)
                if meta_actions is not None:
                    primitive_input = torch.cat([role_features, meta_actions[mask]], dim=-1)
                    primitive_features = self.primitive_body(primitive_input)
                    primitive_mean = torch.tanh(self.primitive_heads[role_idx](primitive_features))
                    primitive_std = torch.exp(self.log_stds_primitive[role_idx]).expand_as(primitive_mean)
                    primitive_dist = torch.distributions.Normal(primitive_mean, primitive_std)
                    primitive_ent = primitive_dist.entropy().sum(dim=-1)
                    primitive_entropy[mask] = primitive_ent

                meta_entropy[mask] = meta_ent

        return meta_entropy, primitive_entropy


class HierarchicalMultiHeadCritic(nn.Module):
    """分层Critic网络：共享编码器 + Meta Critic + Primitive Critic

    用于估计V(s)价值，为上下层提供advantage估计。
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_dim: int,
        num_layers: int,
        meta_action_dim: int = 6,
        num_roles: int = 2,
        agent_id_dim: int = 3,
    ):
        super().__init__()
        self.meta_action_dim = meta_action_dim
        self.num_roles = num_roles
        self.agent_id_dim = agent_id_dim

        # 共享编码器
        input_dim = obs_dim + agent_id_dim
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.shared_body.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        # Meta Critic头
        self.meta_critic_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.meta_critic_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, 1)
                )
            )

        # Primitive Critic身体 - 接收 shared_feat + meta_action
        primitive_input_dim = hidden_dim + meta_action_dim
        self.primitive_critic_body = nn.Sequential(
            nn.Linear(primitive_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Primitive Critic头
        self.primitive_critic_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.primitive_critic_heads.append(
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

    def forward(self, obs, role_ids, meta_actions=None):
        """前向传播计算价值。

        Args:
            obs: [batch_size, num_agents, obs_dim] 观测
            role_ids: [batch_size, num_agents] 角色ID
            meta_actions: [batch_size, num_agents, meta_action_dim] Meta动作(可选)
        Returns:
            values: [batch_size, num_agents] 价值估计
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        values = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]

                if meta_actions is not None:
                    # Primitive Critic
                    primitive_input = torch.cat([role_features, meta_actions[mask]], dim=-1)
                    primitive_features = self.primitive_critic_body(primitive_input)
                    role_values = self.primitive_critic_heads[role_idx](primitive_features).squeeze(-1)
                else:
                    # Meta Critic
                    role_values = self.meta_critic_heads[role_idx](role_features).squeeze(-1)

                values[mask] = role_values

        return values


class MAPPOMultiHeadVelocityAlgorithm(BaseAlgorithm):
    """分层MAPPO算法 - 3chase1场景

    上层(Meta)策略输出6D速度目标(vx,vy,vz,wx,wy,wz)
    下层(Primitive)策略将速度目标转换为8D推进器推力

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
        Initialize Hierarchical MAPPO algorithm.

        Args:
            obs_dim: Observation dimension per agent (after padding)
            action_dim: Action dimension per agent (for compatibility, not used directly)
            state_dim: Global state dimension
            n_agents: Total number of agents
            role_ids: Role ID for each agent [n_agents]
            config: Hierarchical MAPPO configuration
        """
        self.config = config
        self.n_agents = n_agents
        self.role_ids = role_ids
        self.device = torch.device(config.device)

        # 分层Actor网络
        self.actor = HierarchicalMultiHeadActor(
            obs_dim=config.chaser_team_obs_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            meta_action_dim=config.meta_action_dim,  # 6D velocity
            primitive_action_dim=config.primitive_action_dim,  # 8D thrust
            num_roles=config.num_trainable_roles,
            agent_id_dim=len(ROLE_MAPPING),  # 3 (Herder, Netter, Prey)
        ).to(self.device)

        # 分层Critic网络
        self.critic = HierarchicalMultiHeadCritic(
            obs_dim=config.chaser_team_obs_dim,
            hidden_dim=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            meta_action_dim=config.meta_action_dim,
            num_roles=config.num_trainable_roles,
            agent_id_dim=len(ROLE_MAPPING),
        ).to(self.device)

        # 独立优化器
        Optimizer = getattr(optim, config.optimizer)
        self.meta_actor_optimizer = Optimizer(
            list(self.actor.meta_heads.parameters()) +
            list(self.actor.shared_body.parameters()),
            lr=config.learning_rate_meta_actor
        )
        self.meta_critic_optimizer = Optimizer(
            list(self.critic.meta_critic_heads.parameters()) +
            list(self.critic.shared_body.parameters()),
            lr=config.learning_rate_meta_critic
        )
        self.primitive_actor_optimizer = Optimizer(
            list(self.actor.primitive_body.parameters()) +
            list(self.actor.primitive_heads.parameters()),
            lr=config.learning_rate_primitive_actor
        )
        self.primitive_critic_optimizer = Optimizer(
            list(self.critic.primitive_critic_body.parameters()) +
            list(self.critic.primitive_critic_heads.parameters()),
            lr=config.learning_rate_primitive_critic
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

            # 前向传播获取所有动作
            meta_actions_actor, primitive_actions_actor, meta_log_probs_actor, primitive_log_probs_actor, chosen_heads = self.actor.act(
                obs_actor, role_ids_actor
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
            meta_actions_full.cpu().numpy(),  # 返回meta动作供分析
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
            b_actions,  # 原始8D推力动作
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

        # 分离meta和primitive actions的占位符(实际使用时需要从buffer获取或推断)
        # 这里暂时用actor生成meta actions用于critic更新
        b_meta_actions = torch.zeros(
            b_actions.size(0), b_actions.size(1), n_agents, config.meta_action_dim
        ).float().to(device)

        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()

                for t in reversed(range(int(ep_len))):
                    obs_chaser_t = b_obs_chaser_team[ep_idx, t].reshape(3, config.chaser_team_obs_dim).unsqueeze(0)
                    role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
                    reward_chaser_t = b_reward_chaser_team[ep_idx, t]

                    if t == int(ep_len - 1):
                        next_value = torch.zeros(3, device=device)
                    else:
                        next_obs_chaser_t = b_obs_chaser_team[ep_idx, t + 1].reshape(3, config.chaser_team_obs_dim).unsqueeze(0)
                        next_role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
                        next_value = self.critic(obs=next_obs_chaser_t, role_ids=next_role_ids_t)

                    current_value = self.critic(obs=obs_chaser_t, role_ids=role_ids_t)

                    return_lambda[ep_idx, t, :3] = reward_chaser_t + config.gamma * (
                        config.td_lambda * return_lambda[ep_idx, t + 1, :3].detach()
                        + (1 - config.td_lambda) * next_value
                    ) if t < int(ep_len) - 1 else reward_chaser_t + config.gamma * config.td_lambda * torch.zeros(3, device=device)
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
        meta_kl_divergences = []
        primitive_kl_divergences = []
        meta_gradients = []
        primitive_gradients = []
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
            meta_kl_divergence = 0
            primitive_kl_divergence = 0
            meta_clipped_ratio = 0
            primitive_clipped_ratio = 0

            for t in range(b_obs.size(1)):
                epoch_pbar.update(1)
                # 提取Herder/Netter数据
                obs_actor_t = b_obs[:, t, self.actor_indices, :config.chaser_team_obs_dim]
                actions_actor_t = b_actions[:, t, self.actor_indices, :]  # 8D推力
                log_probs_actor_t = b_log_probs[:, t, self.actor_indices]
                role_ids_actor_t = b_role_ids[:, chaser_team_order_tensor]

                valid_mask = b_mask[:, t]
                advantages_actor = advantages[:, t, :3]

                # 使用当前actor生成meta actions用于计算log_prob
                with torch.no_grad():
                    current_meta_actions, _, _, _, _ = self.actor.act(obs_actor_t, role_ids_actor_t)

                # 获取log probabilities
                current_meta_lp, current_primitive_lp = self.actor.get_log_prob(
                    obs_actor_t, role_ids_actor_t, current_meta_actions, actions_actor_t
                )

                # ==== Meta (上层) 策略损失 ====
                # 由于我们没有存储的meta log probs，使用当前策略的近似
                meta_ratio = torch.exp(current_meta_lp - log_probs_actor_t)
                meta_adv = advantages_actor  # 使用相同的advantage

                meta_pg_loss1 = meta_adv * meta_ratio
                meta_pg_loss2 = meta_adv * torch.clamp(meta_ratio, 1 - config.ppo_clip, 1 + config.ppo_clip)
                meta_pg_loss = (
                    torch.min(meta_pg_loss1[valid_mask], meta_pg_loss2[valid_mask])
                    .mean(dim=-1)
                    .sum()
                )

                meta_entropy, _ = self.actor.get_entropy(obs_actor_t, role_ids_actor_t, current_meta_actions)
                meta_entropy_loss = meta_entropy[valid_mask].mean(dim=-1).sum()
                meta_entropy_sum += meta_entropy_loss

                meta_actor_loss += -meta_pg_loss - config.entropy_coef_meta * meta_entropy_loss

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

                _, primitive_entropy = self.actor.get_entropy(obs_actor_t, role_ids_actor_t, current_meta_actions)
                primitive_entropy_loss = primitive_entropy[valid_mask].mean(dim=-1).sum()
                primitive_entropy_sum += primitive_entropy_loss

                primitive_actor_loss += -primitive_pg_loss - config.entropy_coef_primitive * primitive_entropy_loss

                # ==== Critic损失 ====
                current_meta_values = self.critic(obs=obs_actor_t, role_ids=role_ids_actor_t)
                current_primitive_values = self.critic(
                    obs=obs_actor_t, role_ids=role_ids_actor_t, meta_actions=current_meta_actions
                )

                return_lambda_actor = return_lambda[:, t, :3]
                weight = torch.tensor([0.4, 0.3, 0.3], device=device)

                meta_value_loss = (((current_meta_values - return_lambda_actor) ** 2) * weight).sum()
                primitive_value_loss = (((current_primitive_values - return_lambda_actor) ** 2) * weight).sum()

                meta_critic_loss += meta_value_loss
                primitive_critic_loss += primitive_value_loss

                # KL散度
                meta_kl_divergence += (
                    ((meta_ratio - 1) - (current_meta_lp - log_probs_actor_t))[valid_mask]
                    .mean(dim=-1)
                    .sum()
                )
                primitive_kl_divergence += (
                    ((primitive_ratio - 1) - (current_primitive_lp - log_probs_actor_t))[valid_mask]
                    .mean(dim=-1)
                    .sum()
                )

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
            meta_kl_divergence /= actor_count
            primitive_kl_divergence /= actor_count
            meta_clipped_ratio /= actor_count
            primitive_clipped_ratio /= actor_count

            epoch_pbar.close()

            # 更新Meta策略
            self.meta_actor_optimizer.zero_grad()
            self.meta_critic_optimizer.zero_grad()
            meta_actor_loss.backward(retain_graph=True)

            meta_gradient = _norm_d([p.grad for p in list(self.actor.meta_heads.parameters()) +
                                     list(self.actor.shared_body.parameters())], 2)

            if config.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=config.clip_gradients)

            self.meta_actor_optimizer.step()

            # 更新Primitive策略
            self.primitive_actor_optimizer.zero_grad()
            self.primitive_critic_optimizer.zero_grad()
            primitive_actor_loss.backward(retain_graph=True)

            primitive_gradient = _norm_d([p.grad for p in list(self.actor.primitive_body.parameters()) +
                                         list(self.actor.primitive_heads.parameters())], 2)

            self.primitive_actor_optimizer.step()

            # 更新Critic
            meta_critic_loss.backward()
            primitive_critic_loss.backward()

            self.meta_critic_optimizer.step()
            self.primitive_critic_optimizer.step()

            meta_actor_losses.append(meta_actor_loss.item())
            primitive_actor_losses.append(primitive_actor_loss.item())
            meta_critic_losses.append(meta_critic_loss.item())
            primitive_critic_losses.append(primitive_critic_loss.item())
            meta_entropies.append(meta_entropy_sum.item())
            primitive_entropies.append(primitive_entropy_sum.item())
            meta_kl_divergences.append(meta_kl_divergence.item())
            primitive_kl_divergences.append(primitive_kl_divergence.item())
            meta_gradients.append(meta_gradient.item())
            primitive_gradients.append(primitive_gradient.item())
            meta_clipped_ratios.append(meta_clipped_ratio.cpu().item())
            primitive_clipped_ratios.append(primitive_clipped_ratio.cpu().item())

        return {
            "meta_actor_loss": np.mean(meta_actor_losses),
            "primitive_actor_loss": np.mean(primitive_actor_losses),
            "meta_critic_loss": np.mean(meta_critic_losses),
            "primitive_critic_loss": np.mean(primitive_critic_losses),
            "meta_entropy": np.mean(meta_entropies),
            "primitive_entropy": np.mean(primitive_entropies),
            "meta_kl_divergence": np.mean(meta_kl_divergences),
            "primitive_kl_divergence": np.mean(primitive_kl_divergences),
            "meta_gradient": np.mean(meta_gradients),
            "primitive_gradient": np.mean(primitive_gradients),
            "meta_clipped_ratio": np.mean(meta_clipped_ratios),
            "primitive_clipped_ratio": np.mean(primitive_clipped_ratios),
        }

    def save(self, path: str):
        """保存模型checkpoint。"""
        torch.save({
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "meta_actor_optimizer": self.meta_actor_optimizer.state_dict(),
            "meta_critic_optimizer": self.meta_critic_optimizer.state_dict(),
            "primitive_actor_optimizer": self.primitive_actor_optimizer.state_dict(),
            "primitive_critic_optimizer": self.primitive_critic_optimizer.state_dict(),
        }, path)

    def load(self, path: str):
        """加载模型checkpoint。"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint["actor"])
        self.critic.load_state_dict(checkpoint["critic"])
        self.meta_actor_optimizer.load_state_dict(checkpoint["meta_actor_optimizer"])
        self.meta_critic_optimizer.load_state_dict(checkpoint["meta_critic_optimizer"])
        self.primitive_actor_optimizer.load_state_dict(checkpoint["primitive_actor_optimizer"])
        self.primitive_critic_optimizer.load_state_dict(checkpoint["primitive_critic_optimizer"])

    def train(self):
        """设置为训练模式。"""
        self.actor.train()
        self.critic.train()

    def eval(self):
        """设置为评估模式。"""
        self.actor.eval()
        self.critic.eval()
