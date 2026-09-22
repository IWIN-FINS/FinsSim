from asyncio import base_events
from multiprocessing import Pipe, Process
from gitdb import base
import torch
import tyro
import datetime
import random
import numpy as np
import torch.nn as nn
import torch.optim as optim
from dataclasses import dataclass
import torch.nn.functional as F
from scripts.single_scripts.env.pettingzoo_wrapper import PettingZooWrapper
from scripts.single_scripts.env.smaclite_wrapper import SMACliteWrapper
from scripts.single_scripts.env.lbf import LBFWrapper
from torch.distributions.categorical import Categorical
from torch.utils.tensorboard import SummaryWriter
import os
import ray
from tqdm import tqdm
from ray.tune.registry import register_env
from mlagents_envs.environment import UnityEnvironment
from mlagents_envs.envs.unity_parallel_env import UnityParallelEnv
from mlagents_envs.side_channel.engine_configuration_channel import EngineConfigurationChannel
# from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

SQUASH_EPS = 1e-6

"""

# 模拟一个 1280x1024 的显示器并在其中运行 Python 脚本 如果你是ssh启动的，则需要添加这样的模拟环境参数。原理是unity环境启动的时候可能需要检测显示器并使用相应的库。
xvfb-run --auto-servernum --server-args='-screen 0 1280x1024x24' python cleanmarl/mappo_3chase1_unity.py --env-base-port=9969
或者使用环境变量：
export DISPLAY=:0


python cleanmarl/mappo_3chase1_unity.py --env-base-port=9969 --batch_size=4 --epochs=2
python cleanmarl/mappo_3chase1_unity.py --env-base-port=9969 --batch_size=32 --total-timesteps=5000000


"""


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


@dataclass
class Args:
    env_type: str = "Unity"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3chase1"
    """ Name of the environment"""
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 2
    """ Number of episodes to collect in each rollout 实际train的时候一般开32以上"""
    actor_hidden_dim: int = 64
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 3
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 128
    """ Hidden dimension of critic network 一般要比actor大，因为设计比较多的信息融合？"""
    critic_num_layers: int = 3
    """ Number of hidden layers of critic network"""
    optimizer: str = "Adam"
    """ The optimizer"""
    learning_rate_actor: float = 0.0008
    """ Learning rate for the actor"""
    learning_rate_critic: float = 0.0008
    """ Learning rate for the critic"""
    total_timesteps: int = 1000000
    """ Total steps in the environment during training"""
    gamma: float = 0.99
    """ Discount factor"""
    td_lambda: float = 0.95
    """ TD(λ) discount factor"""
    normalize_reward: bool = False
    """ Normalize the rewards if True"""
    normalize_advantage: bool = True # 一般必须
    """ Normalize the advantage if True"""
    normalize_return: bool = True # 一般必须
    """ Normalize the returns if True"""
    epochs: int = 8
    """ Number of training epochs 每次收集一遍数据之后更新多少步数 建议5-15"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    log_every: int = 5
    """ Logging steps """
    eval_steps: int = 5
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 42
    """ Random seed"""
    unity_env_binary_path: str = "/RLChase/build/RLChase.x86_64"
    """ Path to the Unity environment binary"""
    env_base_port: int = 5005
    """ Base port for Unity environment instances (each worker will offset this by its worker_id)"""


@dataclass
class EnvConfig:
    """Configuration passed to each environment worker process."""
    worker_id: int
    env_base_port: int
    unity_env_binary_path: str


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
        self.buffer_size = buffer_size
        self.num_agents = num_agents
        self.obs_space = obs_space
        self.state_space = state_space
        self.action_space = action_space
        self.normalize_reward = normalize_reward
        self.device = device
        self.episodes = [None] * buffer_size
        self.pos = 0
        # Store fixed role_ids for all agents: [num_agents]
        self.role_ids = torch.from_numpy(role_ids).long().to(device) if role_ids is not None else None
        # obs_chaser_team 的维度（3 * chaser_team_obs_dim = 39）
        self.obs_chaser_team_dim = obs_chaser_team_dim if obs_chaser_team_dim else 39
        self.reward_chaser_team_dim = reward_chaser_team_dim
    def add(self, episode):
        for key, values in episode.items():
            episode[key] = torch.from_numpy(np.stack(values)).float().to(self.device)
        self.episodes[self.pos] = episode
        self.pos += 1

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
        # obs_chaser_team: [buffer_size, max_length, obs_chaser_team_dim]
        obs_chaser_team = torch.zeros(
            (self.buffer_size, max_length, self.obs_chaser_team_dim)
        ).to(self.device)
        # reward_chaser_team: [buffer_size, max_length, reward_chaser_team_dim]
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
        # Expand role_ids to [buffer_size, num_agents] for all batches
        role_ids_expanded = self.role_ids.unsqueeze(0).expand(self.buffer_size, -1) if self.role_ids is not None else None
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


class Actor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layer, output_dim) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.layers = nn.ModuleList()
        self.layers.append(nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()))
        for i in range(num_layer):
            self.layers.append(
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            )
        # 输出均值和对数标准差
        self.mean_layer = nn.Linear(hidden_dim, output_dim)
        self.log_std = nn.Parameter(torch.zeros(output_dim))  # 可学习的log标准差

    def act(self, x):
        # 前向传播获取均值
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)
        
        # 构建正态分布
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        action, log_prob = _sample_squashed_normal(distribution)
        
        return action, log_prob
    
    def get_log_prob(self, x, actions):
        """计算给定动作的log概率"""
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)
        
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        log_prob = _squashed_normal_log_prob(distribution, actions)
        
        return log_prob
    
    def get_entropy(self, x):
        """计算分布的熵"""
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)

        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        entropy = _estimate_squashed_normal_entropy(distribution)

        return entropy


# =============================================================================
# 硬编码的角色映射：基于Unity环境中agent名称的关键词
# role_id -> one-hot索引
# =============================================================================
ROLE_MAPPING = {
    'Herder': 0,   # 赶网者（Herder）
    'Netter': 1,   # 拉网者（Netter） 两个Netter共享同一个角色ID，同样的模型，但是输出是根据自身的obs来的
    'Prey': 2,     # 猎物
}

ID_MAPPING = {v: k for k, v in ROLE_MAPPING.items()} # ROLE_MAPPINT的反向映射dict

# 需要训练的角色（不包括Prey）
TRAINABLE_ROLES = {'Herder', 'Netter'}


def get_role_from_agent_name(agent_name: str) -> int:
    """从agent名称中解析角色ID

    Args:
        agent_name: Unity环境的agent名称，格式如 'Herder?team=0?agent_id=3'

    Returns:
        role_id: 角色ID (0=Herder, 1=Netter, 2=Prey)

    Raises:
        ValueError: 如果无法从agent名称中解析出角色ID
    """
    for role_name, role_id in ROLE_MAPPING.items():
        if role_name in agent_name:
            return role_id
    raise ValueError(f"Cannot determine role for agent: '{agent_name}'. Available roles: {list(ROLE_MAPPING.keys())}")


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
        self.agent_id_dim = agent_id_dim if agent_id_dim else num_roles  # Default: one-hot for each role

        # Shared body for feature extraction
        input_dim = obs_dim + self.agent_id_dim
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.shared_body.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        # Role-specific heads
        self.role_heads = nn.ModuleList()
        for _ in range(num_roles):
            self.role_heads.append(
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.ReLU(),
                    nn.Linear(hidden_dim // 2, action_dim)
                )
            )

        # Separate log_std for each role (Netter and Herder)
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
        # Create one-hot encoding based on role_ids
        agent_ids_onehot = torch.zeros(batch_size, num_agents, self.agent_id_dim, device=device)
        for b in range(batch_size):
            for a in range(num_agents):
                role = role_ids[b, a]
                if ID_MAPPING[role.item()] in TRAINABLE_ROLES:
                    agent_ids_onehot[b, a, role] = 1.0
        return agent_ids_onehot

    def act(self, obs, role_ids):
        """Sample actions for all agents based on their roles.
        
        注意这边的observation在仿真训练的时候，出于方便，直接把我方chaser_team的一个批次收集到的obs一并传入。
        但是CTDE的时候(Centerized Training Decentralized Execution)，actor只能访问到自己的obs
        
        但是这个逻辑为什么正确呢？
        
        原因在于`x = x.reshape(batch_size * num_agents, -1)`，把num_agents得到的，变成batch_size的一部分，通过one-hot编码进行区分！
        实机执行的时候只保留Actor网络，此时当作认为每次输入的obs的batch_size只有1，并加上one-hot编码！

        Args:
            obs: [batch_size, num_agents, obs_dim] Observations
            role_ids: [batch_size, num_agents] Role index for each agent (0=Herder, 1=Netter)
        Returns:
            actions: [batch_size, num_agents, action_dim] Sampled actions
            log_probs: [batch_size, num_agents] Log probabilities of sampled actions
            chosen_heads: [batch_size, num_agents] Which head was used for each agent
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        # Build one-hot agent IDs 只有两个角色 [1,0] [0,1] onehot编码区分
        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)

        # Concatenate obs with agent IDs
        x = torch.cat([obs, agent_ids_onehot], dim=-1)  # [batch, num_agents, obs+id（也就是chaser_team_obs_dim+2=15个维度）]

        # Flatten for shared body processing
        x = x.reshape(batch_size * num_agents, -1) # [batch * num_agents, obs+id（也就是15+2=17)]
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        # Sample actions from each role's head
        actions = torch.zeros(batch_size, num_agents, self.action_dim, device=device)
        log_probs = torch.zeros(batch_size, num_agents, device=device)
        chosen_heads = torch.zeros(batch_size, num_agents, dtype=torch.long, device=device)

        for role_idx in range(self.num_roles): # 两个角色Herder, Netter
            mask = (role_ids == role_idx)
            if mask.any(): # 如果张量中至少有一个元素为 True（对于布尔类型）或非零值（对于数值类型），它就返回 True；
                # Get features for agents with this role
                role_features = shared_features[mask]  # [num_matching, hidden_dim]
                mean = self.role_heads[role_idx](role_features) # 仍然是把第一个维度当作batch处理
                std = torch.exp(self.log_stds[role_idx]).expand_as(mean)
                dist = torch.distributions.Normal(mean, std)
                sampled_actions, lp = _sample_squashed_normal(dist)

                # Place results back
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

        # Build one-hot agent IDs
        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)

        # Concatenate and process through shared body
        x = torch.cat([obs, agent_ids_onehot], dim=-1)
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        log_probs = torch.zeros(batch_size, num_agents, device=device)

        # 只处理Herder(0)和Netter(1)，跳过Prey(role >= num_roles)
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


class Critic(nn.Module):
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
    """Multi-head Critic network for heterogeneous agents.
    
    原理同ActorMultiHead，使用one-hot编码，把不同的Agent当作不同的batch来处理，最后根据不同的角色id对应输出不同的Values
    但毕竟Critic是train only的，实机的时候会丢弃掉。所以输出的values仍然是三个一组有序拼接排列
    
    Supports multiple roles (e.g., Herder, Netter) with:
    - Shared feature extraction body
    - Role-specific output heads
    - Agent ID embedding for role identification
    Returns value estimate for each agent individually.
    """

    def __init__(self, obs_dim, hidden_dim, num_layers, num_roles=2, agent_id_dim=None):
        super().__init__()
        self.num_roles = num_roles
        self.obs_dim = obs_dim
        self.agent_id_dim = agent_id_dim if agent_id_dim else num_roles

        # Shared body for feature extraction
        input_dim = obs_dim + self.agent_id_dim
        self.shared_body = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
        )
        for _ in range(num_layers - 1):
            self.shared_body.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()))

        # Role-specific heads
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

    def forward(self, obs, role_ids):
        """Get value estimates for all agents based on their roles.

        Args:
            obs: [batch_size, num_agents, obs_dim] Observations
            role_ids: [batch_size, num_agents] Role index for each agent (0=Herder, 1=Netter)
        Returns:
            values: [batch_size, num_agents] Value estimates for each agent
        """
        batch_size, num_agents, _ = obs.shape
        device = obs.device

        # Build one-hot agent IDs
        agent_ids_onehot = self._build_agent_id_onehot(batch_size, num_agents, role_ids, device)

        # Concatenate obs with agent IDs
        x = torch.cat([obs, agent_ids_onehot], dim=-1)

        # Flatten for shared body processing
        x = x.reshape(batch_size * num_agents, -1)
        shared_features = self.shared_body(x)
        shared_features = shared_features.reshape(batch_size, num_agents, -1)

        # Get value estimates from each role's head
        values = torch.zeros(batch_size, num_agents, device=device)

        for role_idx in range(self.num_roles):
            mask = (role_ids == role_idx)
            if mask.any():
                role_features = shared_features[mask]
                role_values = self.role_heads[role_idx](role_features).squeeze(-1)
                values[mask] = role_values

        return values


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


def environment(env_type, env_name, env_family, agent_ids, kwargs):
    if env_type == "pz":
        env = PettingZooWrapper(
            family=env_family, env_name=env_name, agent_ids=agent_ids, **kwargs
        )
    # elif env_type == "smaclite":
    #     env = SMACliteWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)
    elif env_type == "lbf":
        env = LBFWrapper(map_name=env_name, agent_ids=agent_ids, **kwargs)

    return env


def norm_d(grads, d):
    norms = [torch.linalg.vector_norm(g.detach(), d) for g in grads]
    total_norm_d = torch.linalg.vector_norm(torch.tensor(norms), d)
    return total_norm_d


class CloudpickleWrapper:
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, env):
        self.env = env

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.env)

    def __setstate__(self, env):
        import pickle

        self.env = pickle.loads(env)


def env_worker(conn, env_config: EnvConfig, seed):
    """环境工作进程 - 在子进程内部创建环境"""
    # 设置子进程的随机种子，确保可复现性
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    env = None
    try:
        while True:
            task, content = conn.recv()
            if task == "create_env":
                # 在子进程内部创建Unity环境
                env = get_unity_env(env_config, seed)
                conn.send({"status": "created"})
            elif task == "reset":
                if env is None:
                    conn.send({"error": "Environment not created"})
                    continue
                obs = env.reset(seed=random.randint(0, 100000))
                state = env.get_state()
                content = {"obs": obs, "state": state}
                conn.send(content)
            elif task == "get_env_info":
                if env is None:
                    conn.send({"error": "Environment not created"})
                    continue
                content = {
                    "obs_size": env.get_obs_size(),
                    "action_size": env.get_action_size(),
                    "n_agents": env.n_agents,
                    "state_size": env.get_state_size(),
                }
                conn.send(content)
            elif task == "sample":
                if env is None:
                    conn.send({"error": "Environment not created"})
                    continue
                actions = env.sample()
                content = {"actions": actions}
                conn.send(content)
            elif task == "step":
                if env is None:
                    conn.send({"error": "Environment not created"})
                    continue
                next_obs, reward, done, truncated, infos = env.step(content)
                state = env.get_state()
                content = {
                    "next_obs": next_obs,
                    "reward": reward,
                    "done": done,
                    "truncated": truncated,
                    "infos": infos,
                    "next_state": state,
                }
                conn.send(content)
            elif task == "close":
                if env is not None:
                    env.close()
                conn.close()
                break
    except Exception as e:
        print(f"Error in env_worker: {e}")
        import traceback
        traceback.print_exc()
        if env is not None:
            env.close()
        conn.close()

# 建议设置：处理 Unity 启动时的端口偏移
# OS_SEED = 11451

class UnityEnvWrapper:
    """包装 UnityParallelEnv 以提供统一的接口"""
    def __init__(self, unity_parallel_env: UnityParallelEnv, max_steps=900):
        self.env = unity_parallel_env
        self.n_agents = len(unity_parallel_env.possible_agents)
        self.agents = unity_parallel_env.possible_agents
        
        # 计算每个agent的观察空间大小，找到最大值（用于padding）
        self.agent_obs_sizes = []
        for agent in self.agents:
            obs_space = unity_parallel_env.observation_space(agent)
            if hasattr(obs_space, 'spaces'):  # Tuple space (多个观察)
                obs_size = sum(int(np.prod(space.shape)) for space in obs_space.spaces)
            elif hasattr(obs_space, 'shape'):
                obs_size = int(np.prod(obs_space.shape))
            else:
                obs_size = obs_space.n
            self.agent_obs_sizes.append(obs_size)
        
        # 使用最大观察维度作为统一的obs_size
        self.obs_size = max(self.agent_obs_sizes)
        print(f"Agent observation sizes: {self.agent_obs_sizes}, using max: {self.obs_size}")
        
        # 计算每个agent的动作空间大小，找到最大值（用于统一动作空间）
        self.agent_action_sizes = []
        for agent in self.agents:
            action_space = unity_parallel_env.action_space(agent)
            if hasattr(action_space, 'n'):  # Discrete
                action_size = action_space.n
            elif hasattr(action_space, 'spaces'):  # Tuple (组合空间)
                # 假设是 (continuous, discrete) 的组合
                action_size = 0
                for space in action_space.spaces:
                    if hasattr(space, 'n'):
                        action_size += space.n
                    elif hasattr(space, 'shape') and space.shape is not None:
                        action_size += int(np.prod(space.shape))
            elif hasattr(action_space, 'shape') and action_space.shape is not None:  # Box
                action_size = int(np.prod(action_space.shape))
            else:
                print(f"Warning: Unknown action space type for agent {agent}: {type(action_space)}")
                action_size = 1
            self.agent_action_sizes.append(action_size)
        
        # 使用最大动作空间大小作为统一的action_size
        self.action_size = max(self.agent_action_sizes)
        print(f"Agent action sizes: {self.agent_action_sizes}, using max: {self.action_size}")
        
        # 使用padding后的观察空间作为状态空间
        self.state_size = self.obs_size * self.n_agents

        # 角色映射：根据agent名称关键词硬编码分配角色
        # 支持任意顺序和随机agent_id
        self.role_ids = np.array([
            get_role_from_agent_name(agent) for agent in self.agents
        ], dtype=np.int32)
        print(f"Agent roles (by name): {dict(zip(self.agents, self.role_ids))}")

        # 获取最大步数限制（从Unity环境的behavior spec中获取）
        # 如果环境没有设置max_step，使用默认值
        self.max_steps = max_steps  # 默认值
        self._current_step = 0
        self._last_obs = None
        self._last_info = None
    
    def reset(self, seed=None):
        """重置环境"""
        if seed is not None:
            self.env.seed(seed)
        obs_dict = self.env.reset()
        self._last_obs = obs_dict
        self._current_step = 0  # 重置步数计数器
        # 转换为数组格式 [n_agents, obs_dim]
        # 处理可能的多个观察传感器或嵌套结构，并进行padding
        obs_list = []
        for i, agent in enumerate(self.agents):
            obs = obs_dict[agent]
            # 如果是元组（多个观察），展平并拼接
            if isinstance(obs, dict) and "observation" in obs:
                obs = np.concatenate([np.array(o).flatten() for o in obs["observation"]])
            else:
                obs = np.array(obs).flatten()
            
            # Padding到统一维度
            if len(obs) < self.obs_size:
                obs = np.pad(obs, (0, self.obs_size - len(obs)), constant_values=0)
            elif len(obs) > self.obs_size:
                obs = obs[:self.obs_size]  # 截断（理论上不应该发生）
            
            obs_list.append(obs)
        obs_array = np.array(obs_list)
        return obs_array

    def _prey_escape(self, obs):
        """Prey的简单逃避策略：朝距离最近的追捕者相反方向逃跑
        
        需要根据Unity具体返回的格式修改算法
        
        Args:
            obs: Prey的观察数据，格式因环境而异
        Returns:
            escape_action: 逃避动作向量
        """
        # 获取Prey在所有agent中的索引
        prey_idx = np.where(self.role_ids == ROLE_MAPPING["Prey"])[0]
        if len(prey_idx) == 0:
            return None
        prey_agent_idx = prey_idx[0]

        # 获取其他agent（追捕者）的观察
        # 假设obs包含所有agent的相对位置信息
        # 观察格式: [self_pos, self_vel, other1_pos, other1_vel, ...] 或类似结构
        # 这里需要根据Unity环境的实际观察格式进行调整

        # 简单策略1：完全随机动作
        agent_action_size = self.agent_action_sizes[prey_agent_idx]
        random_action = np.random.uniform(-1, 1, agent_action_size).astype(np.float32)

        # 简单策略2：如果观察中包含其他agent的位置信息，计算逃避方向
        # 这里假设obs的前几个维度是位置信息
        # 实际使用时需要根据你的Unity环境观察空间调整
        try:
            obs_data = np.array(obs).flatten() if not isinstance(obs, np.ndarray) else obs

            # 观察格式: [my_pos(3), my_vel(3), chaser1_pos(3), chaser2_pos(3), chaser3_pos(3)]
            my_pos = obs_data[:3]

            # 找到最近的追捕者，朝其相反方向逃跑
            n_chasers = 3
            min_dist = float('inf')
            escape_dir = np.zeros(3)

            for j in range(n_chasers):
                offset = 3 + 3 + j * 3  # my_pos(3) + my_vel(3) + chaser_j_pos(3)
                if len(obs_data) > offset + 3:
                    chaser_pos = obs_data[offset:offset + 3]
                    # 方向：prey位置 - chaser位置（远离chaser）
                    direction = my_pos - chaser_pos
                    dist = np.linalg.norm(direction)
                    if dist > 0.1 and dist < min_dist:
                        min_dist = dist
                        escape_dir = direction / dist  # 归一化逃离方向

            # 使用最近追捕者的相反方向作为逃离方向
            if min_dist < float('inf'):
                escape_action = np.zeros(agent_action_size, dtype=np.float32)
                # 当前Unity的动作格式暂定为[-1,1]的连续动作，代表[vx, vy, vz]的速度指令
                escape_action[:min(3, agent_action_size)] = escape_dir[:min(3, agent_action_size)]
                return escape_action

        except (IndexError, TypeError) as e:
            pass

        return random_action

    def step(self, actions):
        """执行动作"""
        # 将数组动作转换为字典（连续动作，已经是float值）
        # 对于Prey（role=2），使用内置的逃避策略替代网络输出的动作
        action_dict = {}
        for i, agent in enumerate(self.agents):
            # 如果是Prey，使用逃避策略
            if self.role_ids[i] == ROLE_MAPPING["Prey"]:
                # 从上一步的观察获取Prey的观察来计算逃避方向
                prey_obs = self._last_obs.get(agent, None) if self._last_obs else None
                action = self._prey_escape(prey_obs)
                if action is None:
                    action = np.zeros(self.agent_action_sizes[i], dtype=np.float32)
            else:
                action = np.array(actions[i]).flatten()  # 确保是1D数组
                # 对于连续动作，直接使用（已经在[-1, 1]范围内）
                # 如果动作维度不匹配，进行padding或截断
                agent_action_size = self.agent_action_sizes[i]
                if action.shape[0] < agent_action_size:
                    # Padding
                    action = np.pad(action, (0, agent_action_size - action.shape[0]), constant_values=0.0)
                elif action.shape[0] > agent_action_size:
                    # 截断
                    action = action[:agent_action_size]
            action_dict[agent] = action.astype(np.float32)
        obs_dict, reward_dict, done_dict, info_dict = self.env.step(action_dict)
        
        self._last_obs = obs_dict
        self._last_info = info_dict
        
        # 转换为数组格式
        # 处理可能的多个观察传感器或嵌套结构，并进行padding
        obs_list = []
        for i, agent in enumerate(self.agents):
            obs = obs_dict[agent]
            obs = np.array(obs).flatten()
            # Padding到统一维度
            if len(obs) < self.obs_size:
                obs = np.pad(obs, (0, self.obs_size - len(obs)), constant_values=0)
            elif len(obs) > self.obs_size:
                obs = obs[:self.obs_size]  # 截断（理论上不应该发生）
            
            obs_list.append(obs)
        obs_array = np.array(obs_list) # todo: 修改先把obs_array和reward作为输入向量一一对齐！和每个角色固定位置对齐！
        
        # 分离 Chaser方（Herder + Netter）和 Prey 的奖励
        chaser_rewards = []
        prey_reward = 0.0
        herder_reward = 0.0
        netter_rewards = []

        for agent_name, r in reward_dict.items():
            role_id = get_role_from_agent_name(agent_name)
            if role_id == ROLE_MAPPING["Prey"]:
                prey_reward = float(r)
            else:  # Herder(0) 或 Netter(1) 都是 Chaser 方
                chaser_rewards.append(float(r))
                if role_id == ROLE_MAPPING["Herder"]:
                    herder_reward = float(r)
                elif role_id == ROLE_MAPPING["Netter"]:
                    netter_rewards.append(float(r))

        # Chaser方的全局奖励 = Herder + 两个Netter的总和
        global_reward_chaser = sum(chaser_rewards) if chaser_rewards else 0.0 # 仅仅只是训练的时候的一个接口

        # 返回给外界的 reward 是 Chaser 方的全局奖励
        reward = global_reward_chaser

        done = all(done_dict.values())  # 所有agent都结束

        # 增加步数计数器
        self._current_step += 1

        # 检查是否达到最大步数
        truncated = self._current_step >= self.max_steps
        if truncated and not done:
            print(f"Episode truncated at step {self._current_step} (max_steps={self.max_steps})")

        # 生成 obs_chaser_team 和 reward_chaser_team: 按 [Herder, Netter1, Netter2] 顺序排列
        herder_obs = None
        netter1_obs = None
        netter2_obs = None
        netter1_id = None
        netter2_id = None

        herder_reward = None
        netter1_reward = None
        netter2_reward = None

        for agent_name, obs in obs_dict.items():
            role_id = get_role_from_agent_name(agent_name)
            obs = np.array(obs).flatten()
            reward = float(reward_dict.get(agent_name, 0.0))
            agent_id = int(agent_name.split("agent_id=")[1])

            if role_id == ROLE_MAPPING["Herder"]:
                herder_obs = obs
                herder_reward = reward
            elif role_id == ROLE_MAPPING["Netter"]:
                # 从 agent_name 中提取 agent_id 用于排序
                if netter1_obs is None or agent_id < netter1_id:
                    netter2_obs = netter1_obs
                    netter2_id = netter1_id
                    netter1_obs = obs
                    netter1_id = agent_id
                    netter2_reward = netter1_reward
                    netter1_reward = reward
                else:
                    netter2_obs = obs
                    netter2_id = agent_id
                    netter2_reward = reward

        # 验证所有必需的 agent 都被找到
        if herder_obs is None or netter1_obs is None or netter2_obs is None:
            raise ValueError(
                f"Missing agent observations in chaser team: "
                f"Herder={herder_obs is not None}, Netter1={netter1_obs is not None}, Netter2={netter2_obs is not None}. "
                f"Available agents: {list(obs_dict.keys())}"
            )

        # 组装 obs_chaser_team
        obs_chaser_team = np.concatenate([herder_obs, netter1_obs, netter2_obs])

        # 组装 reward_chaser_team: [r_Herder, r_Netter1, r_Netter2]
        reward_chaser_team = np.array([herder_reward, netter1_reward, netter2_reward], dtype=np.float32)

        info_dict["obs_chaser_team"] = obs_chaser_team
        info_dict["reward_chaser_team"] = reward_chaser_team
        
        
        

        return obs_array, reward, done, truncated, info_dict
    
    def get_obs_size(self):
        return self.obs_size
    
    def get_action_size(self):
        return self.action_size
    
    def get_state_size(self):
        return self.state_size
    
    def get_state(self):
        """获取全局状态（所有agent的观察拼接）"""
        if self._last_obs is None:
            return np.zeros(self.state_size)
        # 处理可能的多个观察传感器，并进行padding
        obs_list = []
        for i, agent in enumerate(self.agents):
            obs = self._last_obs[agent]
            obs = np.array(obs).flatten()
            # Padding到统一维度
            if len(obs) < self.obs_size:
                obs = np.pad(obs, (0, self.obs_size - len(obs)), constant_values=0)
            elif len(obs) > self.obs_size:
                obs = obs[:self.obs_size]  # 截断（理论上不应该发生）
            
            obs_list.append(obs)
        return np.concatenate(obs_list)
    
    def sample(self):
        """随机采样动作"""
        actions = []
        for i, agent in enumerate(self.agents):
            action = self.env.action_space(agent).sample()
            action = np.array(action).flatten()  # 确保是1D数组
            # Padding到统一维度
            if action.shape[0] < self.action_size:
                action = np.pad(action, (0, self.action_size - action.shape[0]), constant_values=0.0)
            elif action.shape[0] > self.action_size:
                action = action[:self.action_size]
            actions.append(action)
        return np.array(actions, dtype=np.float32)
    
    def close(self):
        self.env.close()

def get_unity_env(env_config: EnvConfig, seed, max_retries=1):
    """
    env_config: EnvConfig 对象，包含 worker_index 和 unity_env_binary_path
    max_retries: 最大重试次数，用于处理端口冲突
    """
    # 1. 侧信道配置：每个 Worker 独立配置
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=10.0)

    # 2. 这里的 worker_index 非常重要，它决定了 Unity 实例的端口偏移
    # 避免多个 Unity 实例抢占同一个通讯端口
    worker_id = env_config.worker_id
    env_base_port = env_config.env_base_port

    # 3. 端口冲突处理：如果端口被占用，尝试下一个端口
    last_exception = None
    for attempt in range(max_retries):
        try:
            unity_env = UnityEnvironment(
                file_name=env_config.unity_env_binary_path,
                side_channels=[config_channel],
                no_graphics=True,
                seed=seed + worker_id,
                base_port=env_base_port + attempt,
                worker_id=worker_id
            )
            pz_env = UnityParallelEnv(unity_env)
            wrapped_env = UnityEnvWrapper(pz_env)
            return wrapped_env
        except Exception as e:
            last_exception = e
            if attempt < max_retries - 1:
                print(f"Warning: Port {env_base_port + attempt} is in use, retrying with port {env_base_port + attempt + 1}...")
                # 重新创建 config_channel 因为旧的可能已经被部分使用
                config_channel = EngineConfigurationChannel()
                config_channel.set_configuration_parameters(time_scale=10.0)
            else:
                raise RuntimeError(
                    f"Failed to create Unity environment after {max_retries} attempts. "
                    f"Last error: {last_exception}. "
                    f"Consider increasing --env-base-port or reducing --batch-size."
                ) from last_exception

def get_unity_env_eval(seed, env_base_port=5004, worker_id=100):
    """创建评估用 Unity 环境

    Args:
        seed: 随机种子
        env_base_port: 基础端口（默认5004，与训练环境的基础端口5005错开）
        worker_id: Worker ID（默认100，确保与训练环境的worker不冲突）
    """
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=1.0)
    try:
        unity_env = UnityEnvironment(
            file_name="/RLChase/build/RLChase.x86_64",
            side_channels=[config_channel],
            no_graphics=True,
            seed=seed,
            base_port=env_base_port,
            worker_id=worker_id
        )
    except Exception as e:
        raise RuntimeError(
            f"Failed to create eval Unity environment. "
            f"base_port={env_base_port}, worker_id={worker_id}. Error: {e}"
        ) from e
    wrapped_env = UnityEnvWrapper(UnityParallelEnv(unity_env), max_steps=900)
    """注意这个max_steps一定要和Unity内部定义的一致！！！"""

    return wrapped_env

if __name__ == "__main__":
    args = tyro.cli(Args)
    # Set the random seed
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device)
    ## import the environment
    kwargs = {}  # {"render_mode":'human',"shared_reward":False}
    ## Create the pipes to communicate between the main process (MAPPO algorithm) and child processes (envs)
    conns = [Pipe() for _ in range(args.batch_size)]
    mappo_conns, env_conns = zip(*conns) # conns: connection是多进程相互通信的，使用conn.send()和conn.recv()来发送和接收消息。zip(*)是解包
    
    # 创建进程，传递环境配置而不是环境实例
    processes = [
        Process(target=env_worker, args=(env_conns[i], EnvConfig(worker_id=i, unity_env_binary_path=args.unity_env_binary_path, env_base_port=args.env_base_port), args.seed + i))
        for i in range(args.batch_size)
    ]
    for process in processes:
        process.daemon = True
        process.start()
    
    # 在子进程中创建环境
    print("Creating environments in worker processes...")
    for mappo_conn in mappo_conns:
        mappo_conn.send(("create_env", None))
    print("Waiting for environments to be created...")
    pbar_env_creation = tqdm(total=args.batch_size, desc="Creating Environments")
    for i, mappo_conn in enumerate(mappo_conns):
        response = mappo_conn.recv()
        if "error" in response:
            raise RuntimeError(f"Failed to create environment {i}: {response['error']}")
        pbar_env_creation.update(1)
    pbar_env_creation.close()
    print("All environments created successfully")
    
    eval_env = get_unity_env_eval(args.seed)

    ## Extract environment information from wrapped Unity environment
    obs_size = eval_env.get_obs_size()
    action_size = eval_env.get_action_size()
    n_agents = eval_env.n_agents
    state_size = eval_env.get_state_size()
    max_steps = eval_env.max_steps
    role_ids = eval_env.role_ids  # Get role IDs from environment

    print(f"obs_size: {obs_size}, action_size: {action_size}, n_agents: {n_agents}, state_size: {state_size}")
    print(f"Max steps per episode: {max_steps}")
    print(f"Role IDs: {role_ids}")

    # Determine number of roles that need training (Herder=0, Netter=1, Prey=2 excluded)
    # Prey uses fixed escape strategy, not trained
    num_roles = len([r for r in np.unique(role_ids) if r < 2])
    # Total number of role types (for one-hot encoding), including Prey
    num_role_types = len(ROLE_MAPPING)  # 3 (Herder, Netter, Prey)

    ## Initialize the actor, critic networks
    # Using ActorMultiHead to handle heterogeneous agents (Herder vs Netter)
    chaser_team_obs_dim = 13 # 追方团队的观察维度（根据Unity环境的实际观察空间调整），包含了队友的位置
    chaser_team_action_dim = 8 # 追方团队的动作维度（根据Unity环境的实际动作空间调整）
    actor = ActorMultiHead(
        obs_dim=chaser_team_obs_dim, # 人为根据Unity内部的角色设定处理的。缺少通用性。但是针对这个具体问题也够用了
        hidden_dim=args.actor_hidden_dim,
        num_layers=args.actor_num_layers,
        action_dim=chaser_team_action_dim,
        num_roles=2,  # 人为指定。2个需要训练的角色：Herder(0)和Netter(1)，Prey(2)不训练
        agent_id_dim=2,  # One-hot encoding for all role types
    ).to(device)
    # critic = Critic(
    #     input_dim=chaser_team_obs_dim * 3, # 13 * 3 = 39，按照$S = [o_{Herder}, o_{Netter1}, o_{Netter2}]$的方式拼接
    #     hidden_dim=args.critic_hidden_dim,
    #     num_layer=args.critic_num_layers,
    # ).to(device)
    critic = CriticMultiHead(
        obs_dim=chaser_team_obs_dim,  # 不进行顺序拼接，而是按照one-hot编码区别每个Agent
        hidden_dim=args.critic_hidden_dim,
        num_layers=args.critic_num_layers,
        num_roles=2,  # 2个需要参与网络里面训练的角色：Herder(0)和Netter(1)，Prey(2)不训练
        agent_id_dim=2,  # One-hot encoding for all role types
    ).to(device)

    Optimizer = getattr(optim, args.optimizer)
    actor_optimizer = Optimizer(actor.parameters(), lr=args.learning_rate_actor)
    critic_optimizer = Optimizer(critic.parameters(), lr=args.learning_rate_critic)

    time_token = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_name = f"{args.env_type}__{args.env_name}__{time_token}"
    if args.use_wnb:
        import wandb

        wandb.init(
            project=args.wnb_project,
            entity=args.wnb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=f"MAPPO-multienvs-{run_name}",
        )
    writer = SummaryWriter(f"runs/MAPPO-multienvs-{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s"
        % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    rb = RolloutBuffer(
        buffer_size=args.batch_size,
        obs_space=obs_size,
        state_space=state_size,
        action_space=action_size,
        num_agents=n_agents,
        normalize_reward=args.normalize_reward,
        device=device,
        role_ids=role_ids,
        obs_chaser_team_dim=chaser_team_obs_dim * 3,  # 39 = 3 agents * 13 dims
        reward_chaser_team_dim=3,  # [r_Herder, r_Netter1, r_Netter2]
    )
    ep_rewards = []
    ep_lengths = []
    ep_stats = []
    training_step = 0
    num_episodes = 0
    step = 0
    pbar = tqdm(total=args.total_timesteps, desc="Training Progress")
    while step < args.total_timesteps:
        episodes = [
            {
                "obs": [],
                "actions": [],
                "log_prob": [],
                "reward": [],
                "states": [],
                "done": [],
                "obs_chaser_team": [],
                "reward_chaser_team": [],
            }
            for _ in range(args.batch_size)
        ]

        for mappo_conn in mappo_conns:
            mappo_conn.send(("reset", None))

        contents = [mappo_conn.recv() for mappo_conn in mappo_conns]
        obs = np.stack([content["obs"] for content in contents], axis=0) # 总observation [num_evs, 4, 15]
        state = np.stack([content["state"] for content in contents], axis=0) # 总state [num_envs, 60]
        alive_envs = list(range(args.batch_size))
        ep_reward, ep_length, ep_stat = (
            [0] * args.batch_size,
            [0] * args.batch_size,
            [0] * args.batch_size,
        )

        # Create role_ids tensor for ActorMultiHead: [batch_size, n_agents]
        role_ids_batch = np.tile(role_ids, (args.batch_size, 1))  # [batch_size, n_agents] tile: 重复重构为第一位batch
        role_ids_tensor = torch.from_numpy(role_ids_batch).long().to(device)

        # 预计算 actor 索引位置（role_ids 固定，所以这些索引在所有 episode 中保持不变）
        actor_indices = np.where(np.any(role_ids_batch[:, :len(role_ids)] != ROLE_MAPPING['Prey'], axis=0))[0]
        num_actors = len(actor_indices)  # 3 (1 Herder + 2 Netter)
        actor_indices_tensor = torch.from_numpy(actor_indices).long().to(device)
        print(f"Actor indices: {actor_indices}, num_actors: {num_actors}")

        while len(alive_envs) > 0:
            with torch.no_grad():
                num_envs = len(alive_envs)
                # 当前活跃环境的 role_ids
                role_ids_current = role_ids_tensor[:num_envs]

                # 提取只包含 Herder/Netter 的 obs，保持 [batch_size, num_actors, obs_dim] 结构
                current_obs = torch.from_numpy(obs).float().to(device)  # [num_envs, n_agents, obs_dim]
                obs_actor = current_obs[:, actor_indices, :chaser_team_obs_dim]  # [num_envs, num_actors, chaser_team_obs_dim]
                role_ids_actor = role_ids_current[:, actor_indices]  # [num_envs, num_actors]

                # 只对 Herder/Netter 做决策
                actions_actor, log_probs_actor, _ = actor.act(obs_actor, role_ids_actor)

                # 重建完整的 actions 和 log_probs 数组（Prey 位置填 0）
                actions = torch.zeros(num_envs, n_agents, action_size, device=device)
                log_probs = torch.zeros(num_envs, n_agents, device=device)
                actions[:, actor_indices] = actions_actor
                log_probs[:, actor_indices] = log_probs_actor

                actions, log_probs = actions.cpu().numpy(), log_probs.cpu().numpy()
            for i, j in enumerate(alive_envs):
                mappo_conns[j].send(("step", actions[i]))
            contents = [mappo_conns[i].recv() for i in alive_envs]
            next_obs = [content["next_obs"] for content in contents]
            reward = [content["reward"] for content in contents]
            done = [content["done"] for content in contents]
            truncated = [content["truncated"] for content in contents]
            infos = [content.get("infos") for content in contents]
            next_state = [content["next_state"] for content in contents]
            # 从 infos 中提取 obs_chaser_team 和 reward_chaser_team
            obs_chaser_team_list = []
            reward_chaser_team_list = []
            for i, info in enumerate(infos):
                if info is not None and "obs_chaser_team" in info:
                    obs_chaser_team_list.append(info["obs_chaser_team"])
                else:
                    # Fallback: 使用零向量
                    obs_chaser_team_list.append(np.zeros(39))
                if info is not None and "reward_chaser_team" in info:
                    reward_chaser_team_list.append(info["reward_chaser_team"])
                else:
                    # Fallback: 使用零向量
                    reward_chaser_team_list.append(np.zeros(3))
            for i, j in enumerate(alive_envs):
                episodes[j]["obs"].append(obs[i])
                episodes[j]["actions"].append(actions[i])
                episodes[j]["log_prob"].append(log_probs[i])
                episodes[j]["reward"].append(reward[i])
                episodes[j]["states"].append(state[i])
                episodes[j]["done"].append(done[i])
                episodes[j]["obs_chaser_team"].append(obs_chaser_team_list[i])
                episodes[j]["reward_chaser_team"].append(reward_chaser_team_list[i])
                ep_reward[j] += reward[i]
                ep_length[j] += 1
            step += len(alive_envs)
            obs = []
            state = []
            # 先收集需要移除的环境，避免在循环中修改列表导致索引错乱
            completed_envs = []
            for i, j in enumerate(alive_envs):
                if done[i] or truncated[i]:
                    completed_envs.append(j)
            for j in completed_envs:
                alive_envs.remove(j)
                rb.add(episodes[j])
                episodes[j] = dict()
                if args.env_type == "smaclite":
                    ep_stat[j] = infos[i]
            # 收集未完成环境的数据
            obs = []
            state = []
            for i, j in enumerate(alive_envs):
                if not done[i] and not truncated[i]:
                    obs.append(next_obs[i])
                    state.append(next_state[i])
            if obs:
                obs = np.stack(obs, axis=0)
                state = np.stack(state, axis=0)
        
        # 更新进度条
        pbar.update(int(np.sum(ep_length)))
        pbar.set_postfix({
            'reward': f"{np.mean(ep_reward):.2f}",
            'episodes': num_episodes
        })
        
        ep_rewards.extend(ep_reward)
        ep_lengths.extend(ep_length)
        # if args.env_type == "smaclite":
        #     ep_stats.extend([info["battle_won"] for info in ep_stat])
        num_episodes += args.batch_size
        ## logging
        if len(ep_rewards) > args.log_every:
            writer.add_scalar("rollout/ep_reward", np.mean(ep_rewards), step)
            writer.add_scalar("rollout/ep_length", np.mean(ep_lengths), step)
            writer.add_scalar("rollout/num_episodes", num_episodes, step)
            if args.env_type == "smaclite":
                writer.add_scalar("rollout/battle_won", np.mean(ep_stats), step)
            ep_rewards = []
            ep_lengths = []
            ep_stats = []
        ## Collate episodes in buffer into single batch
        (
            b_obs, # 每个Agent自己的obs，用于输入到Actor网络
            b_actions,
            b_log_probs,
            b_reward,
            b_states, # 全局状态，所有Agents的obs的拼接，为env.get_states()返回值。目前不使用改成chaser_team的obs，输入到Critic网络
            b_done,
            b_mask,
            b_role_ids,
            b_obs_chaser_team,
            b_reward_chaser_team,
        ) = rb.get_batch()

        # Compute the advantage
        #####  Compute TD(λ) using "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        #####  Compute the advantage using A(s,a) = λ-Returns -V(s), see page 47 in David Silver's lecture n 4 (https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-4-model-free-prediction-.pdf)
        return_lambda = torch.zeros((b_actions.size(0), b_actions.size(1), n_agents)).float().to(device)
        advantages = torch.zeros((b_actions.size(0), b_actions.size(1), n_agents)).float().to(device) # 总共4个Agent！
        with torch.no_grad():
            # 预计算 chaser team 的 role_ids 排列顺序，使其与 obs_chaser_team 的 [Herder, Netter1, Netter2] 顺序一致
            # obs_chaser_team 按 agent_id 排序 (Herder=0, agent_id小的Netter在前, agent_id大的Netter在后)
            # 需要从环境 agent 顺序转换为 agent_id 顺序
            env_role_ids = role_ids_batch[0]  # 环境 agent 顺序的 role_ids
            chaser_env_positions = [i for i, rid in enumerate(env_role_ids) if rid != ROLE_MAPPING['Prey']]
            chaser_agent_ids = [env_role_ids[i] for i in chaser_env_positions]
            # 按 [Herder, Netter, Netter] 的 role_id 顺序 + agent_id 排序
            herder_pos = chaser_env_positions[chaser_agent_ids.index(ROLE_MAPPING['Herder'])] if ROLE_MAPPING['Herder'] in chaser_agent_ids else None
            netter_positions = [(chaser_env_positions[i], i) for i, rid in enumerate(chaser_agent_ids) if rid == ROLE_MAPPING['Netter']]
            netter_positions.sort(key=lambda x: env_role_ids[x[0]])  # 按 agent_id 排序
            chaser_team_order = [herder_pos] + [pos for pos, _ in netter_positions] if herder_pos is not None else [pos for pos, _ in netter_positions]
            chaser_team_order_tensor = torch.tensor(chaser_team_order, device=device).long()

            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                # last_return_lambda 初始化为 [3] 形状（chaser team 3个agent）
                last_return_lambda = torch.zeros(3, device=device)
                for t in reversed(range(ep_len)):
                    """
                    计算 return_lambda[ep_idx, t] 需要 return_lambda[ep_idx, t+1]，必须从后往前递归计算，无法完全并行化。
                    所以这边并没有用到batch_size的维度等于buffer_size
                    """
                    # 提取 chaser team 的 obs: [1, 3, chaser_team_obs_dim]
                    # obs_chaser_team 按 agent_id 顺序排列: [Herder, Netter1, Netter2]
                    obs_chaser_t = b_obs_chaser_team[ep_idx, t].reshape(3, chaser_team_obs_dim).unsqueeze(0)
                    # 提取 chaser team 的 role_ids，按与 obs_chaser_team 一致的顺序排列
                    role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
                    # 当前 chaser team 的 reward: [3]
                    reward_chaser_t = b_reward_chaser_team[ep_idx, t]
                    if t == (ep_len - 1):
                        next_value = torch.zeros(3, device=device)
                    else:
                        next_obs_chaser_t = b_obs_chaser_team[ep_idx, t + 1].reshape(3, chaser_team_obs_dim).unsqueeze(0)
                        next_role_ids_t = b_role_ids[ep_idx, chaser_team_order_tensor].unsqueeze(0)
                        next_value = critic(obs=next_obs_chaser_t, role_ids=next_role_ids_t)
                    # return_lambda 和 advantages 对每个 chaser agent 单独计算
                    return_lambda[ep_idx, t, :3] = last_return_lambda = reward_chaser_t + args.gamma * (
                        args.td_lambda * last_return_lambda
                        + (1 - args.td_lambda) * next_value
                    )
                    current_value = critic(obs=obs_chaser_t, role_ids=role_ids_t)  # [3]
                    advantages[ep_idx, t, :3] = return_lambda[ep_idx, t, :3] - current_value
        # 只对 chaser team 的 3 个 agent 进行 normalization
        advantages_chaser = advantages[:, :, :3]
        return_lambda_chaser = return_lambda[:, :, :3]
        if args.normalize_advantage:
            """
            b_mask怎么全为true？
            
            因为在计算优势函数时，我们只考虑那些有效的时间步（即那些没有被 done 标记为 True 的时间步）。
            b_mask 是一个布尔张量，指示哪些时间步是有效的（即还没有结束的）。当我们使用 b_mask 来索引 advantages_chaser
            时，我们实际上是在选择那些对应于有效时间步的优势值。
            
            由于这些优势值可能具有不同的分布（例如，可能有一些非常大的正
            值或负值），我们通常会对它们进行归一化处理，以提高训练的稳定性和效率。通过计算这些有效优势值的均值（adv_mu）
            和标准差（adv_std），我们可以将优势值标准化，使其具有零均值和单位方差，从而帮助模型更好地学习。
            """
            
            adv_mu = advantages_chaser[b_mask].mean()
            adv_std = advantages_chaser[b_mask].std()
            advantages[:, :, :3] = (advantages_chaser - adv_mu) / (adv_std + 1e-8)
        if args.normalize_return:
            ret_mu = return_lambda_chaser[b_mask].mean()
            ret_std = return_lambda_chaser[b_mask].std()
            return_lambda[:, :, :3] = (return_lambda_chaser - ret_mu) / (ret_std + 1e-8)
        # training loop
        actor_losses = []
        critic_losses = []
        entropies_bonuses = []
        kl_divergences = []
        actor_gradients = []
        critic_gradients = []
        clipped_ratios = []

        for _ in tqdm(range(args.epochs), desc="Training Epochs", leave=False):
            actor_loss = 0
            critic_loss = 0
            entropies = 0
            kl_divergence = 0
            clipped_ratio = 0
            for t in range(b_obs.size(1)):
                # 提取只包含 Herder/Netter 的数据（chaser team 共 3 个 agent）
                # b_obs 中存储的是所有agent的obs，使用 actor_indices_tensor 提取 chaser 的数据
                obs_actor_t = b_obs[:, t, actor_indices_tensor, :chaser_team_obs_dim]  # [batch, num_actors, chaser_team_obs_dim]
                # b_actions 和 b_log_probs 在存储时已经按 actor_indices 排列，可以使用 [:3] 提取
                actions_actor_t = b_actions[:, t, :]  # [batch, num_actors, action_dim]
                log_probs_actor_t = b_log_probs[:, t, :]  # [batch, num_actors]
                # role_ids 需要按 agent_id 顺序排列以匹配 obs_chaser_team 的顺序
                role_ids_actor_t = b_role_ids[:, chaser_team_order_tensor]  # [batch, num_actors] chaser team 的 role_ids

                # valid mask 只考虑 actor 部分
                valid_mask = b_mask[:, t]  # [batch]
                # 提取 actor 对应的 advantages（chaser team 3个agent）
                advantages_actor = advantages[:, t, :3]  # [batch, num_actors]

                # policy gradient (PG) loss
                ## PG: compute the ratio:
                current_logprob = actor.get_log_prob(obs_actor_t, role_ids_actor_t, actions_actor_t)
                log_ratio = current_logprob - log_probs_actor_t
                ratio = torch.exp(log_ratio)
                ## Compute PG the loss 这边的loss都是一个向量，对于不同的Head输出的不同的batch去处理一样
                pg_loss1 = advantages_actor * ratio
                pg_loss2 = advantages_actor * torch.clamp(ratio, 1 - args.ppo_clip, 1 + args.ppo_clip)
                pg_loss = (
                    torch.min(pg_loss1[valid_mask], pg_loss2[valid_mask])
                    .mean(dim=-1)
                    .sum()
                )

                # Compute entropy bonus
                entropy_loss = actor.get_entropy(obs_actor_t, role_ids_actor_t)[valid_mask].mean(dim=-1).sum()
                entropies += entropy_loss
                actor_loss += -pg_loss - args.entropy_coef * entropy_loss

                # Compute the value loss (加权平均)
                # critic 输出: [batch, 3]（3个chaser agent各自的value）
                current_values = critic(obs=obs_actor_t, role_ids=role_ids_actor_t)  # [batch, 3]
                return_lambda_actor = return_lambda[:, t, :3]  # [batch, 3]
                # 加权 MSE: Herder 和 Netter 等权重（各 1/3）
                weight = torch.tensor([0.4, 0.3, 0.3], device=device)
                value_loss = (((current_values - return_lambda_actor) ** 2) * weight).sum()
                critic_loss += value_loss # 梯度反传的时候，根据求导公式，可以做到不同Head不同的梯度，即使最后加权合在一起

                # track kl distance
                b_kl_divergence = (
                    ((ratio - 1) - log_ratio)[valid_mask].mean(dim=-1).sum()
                )
                kl_divergence += b_kl_divergence
                clipped_ratio += (
                    ((ratio - 1.0).abs() > args.ppo_clip)[valid_mask]
                    .float()
                    .mean(dim=-1)
                    .sum()
                )

            # 使用有效 episode 数量作为 normalization 基数
            actor_count = b_mask.sum()
            actor_loss /= actor_count
            critic_loss /= b_mask.sum()
            entropies /= actor_count
            kl_divergence /= actor_count
            clipped_ratio /= actor_count

            actor_optimizer.zero_grad()
            critic_optimizer.zero_grad()

            actor_loss.backward()
            critic_loss.backward()

            actor_gradient = norm_d([p.grad for p in actor.parameters()], 2)
            critic_gradient = norm_d([p.grad for p in critic.parameters()], 2)
            if args.clip_gradients > 0:
                torch.nn.utils.clip_grad_norm_(
                    actor.parameters(), max_norm=args.clip_gradients
                )
                torch.nn.utils.clip_grad_norm_(
                    critic.parameters(), max_norm=args.clip_gradients
                )
            actor_optimizer.step()
            critic_optimizer.step()
            training_step += 1

            actor_losses.append(actor_loss.item())
            critic_losses.append(critic_loss.item())
            entropies_bonuses.append(entropies.item())
            kl_divergences.append(kl_divergence.item())
            actor_gradients.append(actor_gradient)
            critic_gradients.append(critic_gradient)
            clipped_ratios.append(clipped_ratio.cpu())

        writer.add_scalar("train/critic_loss", np.mean(critic_losses), step)
        writer.add_scalar("train/actor_loss", np.mean(actor_losses), step)
        writer.add_scalar("train/entropy", np.mean(entropies_bonuses), step)
        writer.add_scalar("train/kl_divergence", np.mean(kl_divergences), step)
        writer.add_scalar("train/clipped_ratios", np.mean(clipped_ratios), step)
        writer.add_scalar("train/actor_gradients", np.mean(actor_gradients), step)
        writer.add_scalar("train/critic_gradients", np.mean(critic_gradients), step)
        writer.add_scalar("train/num_updates", training_step, step)

        if training_step > 0 and (training_step / args.epochs) % args.eval_steps == 0:
            eval_obs = eval_env.reset()
            eval_ep = 0
            eval_ep_reward = []
            eval_ep_length = []
            eval_ep_stats = []
            current_reward = 0
            current_ep_length = 0
            eval_pbar = tqdm(total=args.num_eval_ep, desc="Evaluation", leave=False)
            while eval_ep < args.num_eval_ep:
                with torch.no_grad():
                    # 构造评估时的输入，参照训练时的处理方式
                    eval_obs_tensor = torch.from_numpy(eval_obs).float().to(device).unsqueeze(0)  # [1, n_agents, obs_dim]
                    obs_actor_eval = eval_obs_tensor[:, actor_indices, :chaser_team_obs_dim]  # [1, num_actors, chaser_team_obs_dim]
                    role_ids_eval = role_ids_tensor[:1][:, actor_indices]  # [1, num_actors]
                    actions_actor_eval, _, _ = actor.act(obs_actor_eval, role_ids_eval)  # [1, num_actors, action_dim]
                    # 重建完整的 actions 数组（Prey 位置填 0）
                    actions = torch.zeros(1, n_agents, action_size, device=device)
                    actions[:, actor_indices] = actions_actor_eval
                next_obs_, reward, done, truncated, infos = eval_env.step(
                    actions.squeeze(0).cpu().numpy()
                )
                current_reward += reward
                current_ep_length += 1
                eval_obs = next_obs_
                if done or truncated:
                    eval_obs = eval_env.reset()
                    eval_ep_reward.append(current_reward)
                    eval_ep_length.append(current_ep_length)
                    eval_ep_stats.append(infos)
                    current_reward = 0
                    current_ep_length = 0
                    eval_ep += 1
                    eval_pbar.update(1)
                    eval_pbar.set_postfix({'reward': f"{current_reward:.2f}"})
            eval_pbar.close()
            writer.add_scalar("eval/ep_reward", np.mean(eval_ep_reward), step)
            writer.add_scalar("eval/std_ep_reward", np.std(eval_ep_reward), step)
            writer.add_scalar("eval/ep_length", np.mean(eval_ep_length), step)
            if args.env_type == "smaclite":
                writer.add_scalar(
                    "eval/battle_won",
                    np.mean(np.mean([info["battle_won"] for info in eval_ep_stats])),
                    step,
                )
    
    pbar.close()
    writer.close()
    if args.use_wnb:
        wandb.finish()
    eval_env.close()
    for conn in mappo_conns:
        conn.send(("close", None))
    for process in processes:
        process.join()
