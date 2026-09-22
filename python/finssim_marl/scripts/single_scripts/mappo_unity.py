from multiprocessing import Pipe, Process
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

@dataclass
class Args:
    env_type: str = "smaclite"
    """ Pettingzoo, SMAClite ... """
    env_name: str = "3m"
    """ Name of the environment"""
    env_family: str = "mpe"
    """ Env family when using pz"""
    agent_ids: bool = True
    """ Include id (one-hot vector) at the agent of the observations"""
    batch_size: int = 1
    """ Number of episodes to collect in each rollout"""
    actor_hidden_dim: int = 32
    """ Hidden dimension of actor network"""
    actor_num_layers: int = 1
    """ Number of hidden layers of actor network"""
    critic_hidden_dim: int = 64
    """ Hidden dimension of critic network"""
    critic_num_layers: int = 1
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
    normalize_advantage: bool = False
    """ Normalize the advantage if True"""
    normalize_return: bool = False
    """ Normalize the returns if True"""
    epochs: int = 3
    """ Number of training epochs"""
    ppo_clip: float = 0.2
    """ PPO clipping factor """
    entropy_coef: float = 0.001
    """ Entropy coefficient """
    clip_gradients: float = -1
    """ 0< for no clipping and 0> if clipping at clip_gradients"""
    log_every: int = 10
    """ Logging steps """
    eval_steps: int = 50
    """ Evaluate the policy each «eval_steps» training steps"""
    num_eval_ep: int = 10
    """ Number of evaluation episodes"""
    use_wnb: bool = False
    """ Logging to Weights & Biases if True"""
    wnb_project: str = ""
    """ Weights & Biases project name"""
    wnb_entity: str = ""
    """ Weights & Biases entity name"""
    device: str = "cpu"
    """ Device (cpu, cuda, mps)"""
    seed: int = 42
    """ Random seed"""


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
        for i in range(self.buffer_size):
            length = lengths[i]
            obs[i, :length] = self.episodes[i]["obs"]
            actions[i, :length] = self.episodes[i]["actions"]
            log_probs[i, :length] = self.episodes[i]["log_prob"]
            reward[i, :length] = self.episodes[i]["reward"]
            states[i, :length] = self.episodes[i]["states"]
            done[i, :length] = self.episodes[i]["done"]
            mask[i, :length] = 1
        if self.normalize_reward:
            mu = torch.mean(reward[mask])
            std = torch.std(reward[mask])
            reward[mask.bool()] = (reward[mask] - mu) / (std + 1e-6)
        self.episodes = [None] * self.buffer_size
        return (
            obs.float(),
            actions.float(),
            log_probs.float(),
            reward.float(),
            states.float(),
            done.float(),
            mask,
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
        mean = torch.tanh(mean)  # 将均值限制在[-1, 1]
        
        # 构建正态分布
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        action = distribution.sample()
        action = torch.clamp(action, -1.0, 1.0)  # 裁剪到[-1, 1]
        log_prob = distribution.log_prob(action).sum(dim=-1)  # 对所有动作维度求和
        
        return action, log_prob
    
    def get_log_prob(self, x, actions):
        """计算给定动作的log概率"""
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)
        mean = torch.tanh(mean)
        
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        
        return log_prob
    
    def get_entropy(self, x):
        """计算分布的熵"""
        for layer in self.layers:
            x = layer(x)
        mean = self.mean_layer(x)
        
        std = torch.exp(self.log_std).expand_as(mean)
        distribution = torch.distributions.Normal(mean, std)
        entropy = distribution.entropy().sum(dim=-1)
        
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


def env_worker(conn, env_config, seed):
    """环境工作进程 - 在子进程内部创建环境"""
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
    def __init__(self, unity_parallel_env, max_steps=900):
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
    
    def step(self, actions):
        """执行动作"""
        # 将数组动作转换为字典（连续动作，已经是float值）
        action_dict = {}
        for i, agent in enumerate(self.agents):
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
            # 如果是元组（多个观察），展平并拼接
            if isinstance(obs, tuple):
                obs = np.concatenate([np.array(o).flatten() for o in obs])
            elif isinstance(obs, dict) and "observation" in obs:
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
        reward = sum(reward_dict.values()) / len(reward_dict)  # 平均奖励
        done = all(done_dict.values())  # 所有agent都结束
        
        # 增加步数计数器
        self._current_step += 1
        
        # 检查是否达到最大步数
        truncated = self._current_step >= self.max_steps
        if truncated and not done:
            print(f"Episode truncated at step {self._current_step} (max_steps={self.max_steps})")
        
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
            if isinstance(obs, tuple):
                obs = np.concatenate([np.array(o).flatten() for o in obs])
            elif isinstance(obs, dict) and "observation" in obs:
                obs = np.concatenate([np.array(o).flatten() for o in obs["observation"]])
            else:
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

def get_unity_env(env_config, seed):
    """
    env_config 是 Ray 自动传入的字典，包含 worker_index 等信息
    """
    # 1. 侧信道配置：每个 Worker 独立配置
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=10.0) 
    
    # 2. 这里的 worker_index 非常重要，它决定了 Unity 实例的端口偏移
    # 避免多个 Unity 实例抢占同一个通讯端口
    worker_id = env_config["worker_index"] 
    
    unity_env = UnityEnvironment(
        file_name="/RLChase/build/RLChase.x86_64",
        side_channels=[config_channel],
        no_graphics=True,
        seed=seed + worker_id,
        worker_id=worker_id  # 关键：确保端口不冲突
    )

    pz_env = UnityParallelEnv(unity_env)
    wrapped_env = UnityEnvWrapper(pz_env)
    return wrapped_env

def get_unity_env_eval(seed):
    config_channel = EngineConfigurationChannel()
    config_channel.set_configuration_parameters(time_scale=1.0) 
    unity_env = UnityEnvironment(
        file_name="/RLChase/build/RLChase.x86_64",
        side_channels=[config_channel],
        no_graphics=True,
        seed=seed,
        worker_id=100  # 确保与训练环境的 worker_id 不冲突
    )
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
    mappo_conns, env_conns = zip(*conns)
    
    # 创建进程，传递环境配置而不是环境实例
    processes = [
        Process(target=env_worker, args=(env_conns[i], {"worker_index": i}, args.seed + i))
        for i in range(args.batch_size)
    ]
    for process in processes:
        process.daemon = True
        process.start()
    
    # 在子进程中创建环境
    print("Creating environments in worker processes...")
    for mappo_conn in mappo_conns:
        mappo_conn.send(("create_env", None))
    for i, mappo_conn in enumerate(mappo_conns):
        response = mappo_conn.recv()
        if "error" in response:
            raise RuntimeError(f"Failed to create environment {i}: {response['error']}")
        print(f"Environment {i} created successfully")
    
    eval_env = get_unity_env_eval(args.seed)

    ## Extract environment information from wrapped Unity environment
    obs_size = eval_env.get_obs_size()
    action_size = eval_env.get_action_size()
    n_agents = eval_env.n_agents
    state_size = eval_env.get_state_size()
    max_steps = eval_env.max_steps

    print(f"obs_size: {obs_size}, action_size: {action_size}, n_agents: {n_agents}, state_size: {state_size}")
    print(f"Max steps per episode: {max_steps}")
    
    ## Initialize the actor, critic and target-critic networks
    actor = Actor(
        input_dim=obs_size,
        hidden_dim=args.actor_hidden_dim,
        num_layer=args.actor_num_layers,
        output_dim=action_size,
    ).to(device)
    critic = Critic(
        input_dim=state_size,
        hidden_dim=args.critic_hidden_dim,
        num_layer=args.critic_num_layers,
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
            }
            for _ in range(args.batch_size)
        ]

        for mappo_conn in mappo_conns:
            mappo_conn.send(("reset", None))

        contents = [mappo_conn.recv() for mappo_conn in mappo_conns]
        obs = np.stack([content["obs"] for content in contents], axis=0)
        state = np.stack([content["state"] for content in contents])
        alive_envs = list(range(args.batch_size))
        ep_reward, ep_length, ep_stat = (
            [0] * args.batch_size,
            [0] * args.batch_size,
            [0] * args.batch_size,
        )
        while len(alive_envs) > 0:
            with torch.no_grad():
                actions, log_probs = actor.act(
                    torch.from_numpy(obs).float().to(device)
                )
                actions, log_probs = actions.cpu().numpy(), log_probs.cpu()
            for i, j in enumerate(alive_envs):
                mappo_conns[j].send(("step", actions[i]))
            contents = [mappo_conns[i].recv() for i in alive_envs]
            next_obs = [content["next_obs"] for content in contents]
            reward = [content["reward"] for content in contents]
            done = [content["done"] for content in contents]
            truncated = [content["truncated"] for content in contents]
            infos = [content.get("infos") for content in contents]
            next_state = [content["next_state"] for content in contents]
            for i, j in enumerate(alive_envs):
                episodes[j]["obs"].append(obs[i])
                episodes[j]["actions"].append(actions[i])
                episodes[j]["log_prob"].append(log_probs[i])
                episodes[j]["reward"].append(reward[i])
                episodes[j]["states"].append(state[i])
                episodes[j]["done"].append(done[i])
                ep_reward[j] += reward[i]
                ep_length[j] += 1
            step += len(alive_envs)
            obs = []
            state = []
            for i, j in enumerate(alive_envs[:]):
                if done[i] or truncated[i]:
                    alive_envs.remove(j)
                    rb.add(episodes[j])
                    episodes[j] = dict()
                    if args.env_type == "smaclite":
                        ep_stat[j] = infos[i]
                else:
                    obs.append(next_obs[i])
                    state.append(next_state[i])
            if obs:
                obs = np.stack(obs, axis=0)
                state = np.stack(state, axis=0)
        
        # 更新进度条
        pbar.update(args.batch_size * np.mean(ep_length))
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
            b_obs,
            b_actions,
            b_log_probs,
            b_reward,
            b_states,
            b_done,
            b_mask,
        ) = rb.get_batch()

        # Compute the advantage
        #####  Compute TD(λ) using "Reconciling λ-Returns with Experience Replay"(https://arxiv.org/pdf/1810.09967 Equation 3)
        #####  Compute the advantage using A(s,a) = λ-Returns -V(s), see page 47 in David Silver's lecture n 4 (https://davidstarsilver.wordpress.com/wp-content/uploads/2025/04/lecture-4-model-free-prediction-.pdf)
        return_lambda = torch.zeros((b_actions.size(0), b_actions.size(1), n_agents)).float().to(device)
        advantages = torch.zeros((b_actions.size(0), b_actions.size(1), n_agents)).float().to(device)
        with torch.no_grad():
            for ep_idx in range(return_lambda.size(0)):
                ep_len = b_mask[ep_idx].sum()
                last_return_lambda = 0
                last_advantage = 0
                for t in reversed(range(ep_len)):
                    if t == (ep_len - 1):
                        next_value = 0
                    else:
                        next_value = critic(x=b_states[ep_idx, t + 1])
                    return_lambda[ep_idx, t] = last_return_lambda = b_reward[
                        ep_idx, t
                    ] + args.gamma * (
                        args.td_lambda * last_return_lambda
                        + (1 - args.td_lambda) * next_value
                    )
                    advantages[ep_idx, t] = return_lambda[ep_idx, t] - critic(
                        x=b_states[ep_idx, t]
                    )
        if args.normalize_advantage:
            adv_mu = advantages.mean(dim=-1)[b_mask].mean()
            adv_std = advantages.mean(dim=-1)[b_mask].std()
            advantages = (advantages - adv_mu) / adv_std
        if args.normalize_return:
            ret_mu = return_lambda.mean(dim=-1)[b_mask].mean()
            ret_std = return_lambda.mean(dim=-1)[b_mask].std()
            return_lambda = (return_lambda - ret_mu) / ret_std
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
                # policy gradient (PG) loss
                ## PG: compute the ratio:
                current_logprob = actor.get_log_prob(b_obs[:, t], b_actions[:, t])
                log_ratio = current_logprob - b_log_probs[:, t]
                ratio = torch.exp(log_ratio)
                ## Compute PG the loss
                pg_loss1 = advantages[:, t] * ratio
                pg_loss2 = advantages[:, t] * torch.clamp(
                    ratio, 1 - args.ppo_clip, 1 + args.ppo_clip
                )
                pg_loss = (
                    torch.min(pg_loss1[b_mask[:, t]], pg_loss2[b_mask[:, t]])
                    .mean(dim=-1)
                    .sum()
                )

                # Compute entropy bonus
                entropy_loss = actor.get_entropy(b_obs[:, t])[b_mask[:, t]].mean(dim=-1).sum()
                entropies += entropy_loss
                actor_loss += -pg_loss - args.entropy_coef * entropy_loss

                # Compute the value loss
                current_values = critic(x=b_states[:, t]).expand(-1, n_agents)
                value_loss = F.mse_loss(
                    current_values[b_mask[:, t]], return_lambda[:, t][b_mask[:, t]]
                ) * (b_mask[:, t].sum())
                critic_loss += value_loss

                # track kl distance
                b_kl_divergence = (
                    ((ratio - 1) - log_ratio)[b_mask[:, t]].mean(dim=-1).sum()
                )
                kl_divergence += b_kl_divergence
                clipped_ratio += (
                    ((ratio - 1.0).abs() > args.ppo_clip)[b_mask[:, t]]
                    .float()
                    .mean(dim=-1)
                    .sum()
                )

            actor_loss /= b_mask.sum()
            critic_loss /= b_mask.sum()
            entropies /= b_mask.sum()
            kl_divergence /= b_mask.sum()
            clipped_ratio /= b_mask.sum()

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

        if (training_step / args.epochs) % args.eval_steps == 0:
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
                    actions, _ = actor.act(
                        torch.from_numpy(eval_obs).float().to(device)
                    )
                next_obs_, reward, done, truncated, infos = eval_env.step(
                    actions.cpu().numpy()
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
