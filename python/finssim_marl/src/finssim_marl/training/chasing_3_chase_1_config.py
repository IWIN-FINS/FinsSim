"""
3chase1 (herding) training configuration for underwater MARL.

Config for the 3chaser1 scenario: 1 Herder + 2 Netter chasing 1 Prey.

Usage:
    from finssim_marl.training.config import get_config

    cfg = get_config("chasing_3_chase_1")
    config = cfg.create_train_config()
"""
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from finssim_marl.algorithms.mappo_multihead import MAPPOMultiHeadAlgorithm, MAPPOMultiHeadConfig
from finssim_marl.algorithms.mappo_multihead_position_controller import (
    MAPPOMultiHeadPositionControlAlgorithm,
    MAPPOMultiHeadPositionControlConfig,
)
from finssim_marl.algorithms.traditional_formation_baseline import (
    TraditionalFormationBaselineConfig,
    TraditionalFormationPIDWrenchBaseline,
)
from finssim_marl.algorithms.networks.rollout_buffer import RolloutBuffer
from finssim_marl.training.config import BaseConfig, BaseEnvironmentConfig, BaseTrainingConfig, TrainConfig

THREE_CHASE_ONE_UNITY_BUILD_PATH = (
    "artifacts/unity_builds/marl/linux/FinsROV/3Chase1_headless/3Chase1.x86_64"
)


def get_mappo_config(
    train_config: TrainConfig,
    critic_variant: str = "critic_multihead",
    critic_attention_heads: int = 4,
) -> MAPPOMultiHeadConfig:
    """Build MAPPOMultiHeadConfig from TrainConfig for 3chase1.

    Args:
        train_config: Base training configuration

    Returns:
        MAPPOMultiHeadConfig instance
    """
    return MAPPOMultiHeadConfig(
        actor_hidden_dim=train_config.actor_hidden_dim,
        actor_num_layers=train_config.actor_num_layers,
        critic_hidden_dim=train_config.critic_hidden_dim,
        critic_num_layers=train_config.critic_num_layers,
        learning_rate_actor=train_config.learning_rate_actor,
        learning_rate_critic=train_config.learning_rate_critic,
        gamma=train_config.gamma,
        td_lambda=train_config.td_lambda,
        normalize_advantage=train_config.normalize_advantage,
        normalize_return=train_config.normalize_return,
        epochs=train_config.epochs,
        ppo_clip=train_config.ppo_clip,
        entropy_coef=train_config.entropy_coef,
        clip_gradients=train_config.clip_gradients,
        device=train_config.device,
        chaser_team_obs_dim=train_config.chaser_team_obs_dim,
        chaser_team_action_dim=train_config.chaser_team_action_dim,
        action_interface="direct_thruster",
        rollout_action_dim=train_config.chaser_team_action_dim,
        env_action_dim=train_config.chaser_team_action_dim,
        critic_variant=critic_variant,
        critic_attention_heads=critic_attention_heads,
    )


def create_algorithm(
    obs_dim: int,
    action_dim: int,
    state_dim: int,
    n_agents: int,
    role_ids: Any,
    mappo_config: MAPPOMultiHeadConfig,
) -> MAPPOMultiHeadAlgorithm:
    """Create MAPPO algorithm for 3chase1.

    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        state_dim: State dimension
        n_agents: Number of agents
        role_ids: Role IDs for each agent
        mappo_config: MAPPO configuration

    Returns:
        MAPPOMultiHeadAlgorithm instance
    """
    return MAPPOMultiHeadAlgorithm(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=state_dim,
        n_agents=n_agents,
        role_ids=role_ids,
        config=mappo_config,
    )


def create_rollout_buffer(
    buffer_size: int,
    obs_space: int,
    state_space: int,
    action_space: int,
    num_agents: int,
    role_ids: Any,
    chaser_team_obs_dim: int,
    chaser_team_reward_dim: int,
    device: str,
) -> RolloutBuffer:
    """Create RolloutBuffer for 3chase1.

    Args:
        buffer_size: Number of episodes to store
        obs_space: Observation dimension
        state_space: State dimension
        action_space: Action dimension
        num_agents: Number of agents
        role_ids: Role IDs
        chaser_team_obs_dim: Chaser team observation dimension
        chaser_team_reward_dim: Chaser team reward dimension
        device: Device for tensors

    Returns:
        RolloutBuffer instance
    """
    return RolloutBuffer(
        buffer_size=buffer_size,
        obs_space=obs_space,
        state_space=state_space,
        action_space=action_space,
        num_agents=num_agents,
        normalize_reward=False,
        device=device,
        role_ids=role_ids,
        obs_chaser_team_dim=chaser_team_obs_dim * 3,
        reward_chaser_team_dim=chaser_team_reward_dim,
    )


@dataclass
class Chasing3Chase1TrainingConfig(BaseTrainingConfig):
    """3chase1 训练配置"""
    learning_rate_actor: float = 3e-4
    learning_rate_critic: float = 3e-4
    gamma: float = 0.99
    td_lambda: float = 0.95
    normalize_advantage: bool = True
    normalize_return: bool = False
    epochs: int = 10
    ppo_clip: float = 0.2
    entropy_coef: float = 0.001
    clip_gradients: float = 0.5
    total_timesteps: int = 10_000_000
    log_every: int = 10
    save_freq: int = 50_000
    num_eval_envs: int = 4
    eval_time_scale: float = 1.0


@dataclass
class Chasing3Chase1EnvironmentConfig(BaseEnvironmentConfig):
    """3chase1 环境配置"""
    env_path: Optional[str] = THREE_CHASE_ONE_UNITY_BUILD_PATH
    env_base_port: int = 5005
    timeout_wait: int = 60
    time_scale: float = 10.0
    herder_action_source: str = "python_policy"
    netter_action_source: str = "python_policy"
    prey_action_source: str = "wrapper_escape"

@dataclass
class Chasing3Chase1Config(BaseConfig):
    """3chase1 配置"""

    model_type: str = "MAPPO"
    env_config: Chasing3Chase1EnvironmentConfig = field(default_factory=Chasing3Chase1EnvironmentConfig)
    training_config: Chasing3Chase1TrainingConfig = field(default_factory=Chasing3Chase1TrainingConfig)
    # MAPPO 特有参数
    # Chasing问题的MAPPO 特有参数（可选，用于 MAPPO 类算法）
    actor_hidden_dim: int = 64
    actor_num_layers: int = 2
    critic_hidden_dim: int = 64
    critic_num_layers: int = 2
    critic_variant: str = "critic_multihead"
    critic_attention_heads: int = 4

    # 场景特有参数
    chaser_team_obs_dim: int = 30
    chaser_team_action_dim: int = 8
    chaser_team_reward_dim: int = 3

    # 场景通用参数（覆盖 BaseConfig 默认值）
    num_envs: int = 8
    seed: int = 42
    device: str = "cuda"
    log_dir: str = "runs"
    checkpoint_dir: str = "checkpoints"
    use_wandb: bool = False
    wandb_project: str = "finssim_marl"
    wandb_entity: str = ""
    curriculum: Dict[str, Any] = field(default_factory=dict)

    def create_train_config(self) -> TrainConfig:
        """创建供 train.py 使用的 TrainConfig 实例"""
        env_cfg = self.env_config
        train_cfg = self.training_config

        # env_path 可能为 None，但 TrainConfig.unity_env_binary_path 期望 str
        env_path = env_cfg.env_path or ""

        return TrainConfig(
            # 通用参数
            num_envs=self.num_envs,
            batch_size=self.training_config.batch_size,  # 收集够512个episode后进行一次训练
            seed=self.seed,
            device=self.device,
            log_dir=self.log_dir,
            checkpoint_dir=self.checkpoint_dir,
            use_wandb=self.use_wandb,
            wandb_project=self.wandb_project,
            wandb_entity=self.wandb_entity,
            env_type=env_cfg.env_type,
            total_timesteps=train_cfg.total_timesteps,

            # 环境参数
            unity_env_binary_path=env_path,
            env_base_port=env_cfg.env_base_port,
            timeout_wait=env_cfg.timeout_wait,
            training_time_scale=env_cfg.time_scale,
            parallel_mode=env_cfg.parallel_mode,
            use_editor=env_cfg.use_editor,
            herder_action_source=env_cfg.herder_action_source,
            netter_action_source=env_cfg.netter_action_source,
            prey_action_source=env_cfg.prey_action_source,
            environment_parameters=dict(env_cfg.environment_parameters),
            curriculum=dict(self.curriculum),

            # MAPPO 参数
            actor_hidden_dim=self.actor_hidden_dim,
            actor_num_layers=self.actor_num_layers,
            critic_hidden_dim=self.critic_hidden_dim,
            critic_num_layers=self.critic_num_layers,
            learning_rate_actor=train_cfg.learning_rate_actor,
            learning_rate_critic=train_cfg.learning_rate_critic,
            gamma=train_cfg.gamma,
            td_lambda=train_cfg.td_lambda,
            normalize_advantage=train_cfg.normalize_advantage,
            normalize_return=train_cfg.normalize_return,
            epochs=train_cfg.epochs,
            ppo_clip=train_cfg.ppo_clip,
            entropy_coef=train_cfg.entropy_coef,
            clip_gradients=train_cfg.clip_gradients,

            # 场景特有参数
            chaser_team_obs_dim=self.chaser_team_obs_dim,
            chaser_team_action_dim=self.chaser_team_action_dim,
            chaser_team_reward_dim=self.chaser_team_reward_dim,

            # 评估参数
            eval_steps=self.training_config.eval_steps,
            num_eval_ep=self.training_config.num_eval_ep,
            num_eval_envs=self.training_config.num_eval_envs,
            eval_time_scale=self.training_config.eval_time_scale,
            eval_mode=self.training_config.eval_mode,
            keep_eval_snapshots=self.training_config.keep_eval_snapshots,
            save_freq=train_cfg.save_freq,
            log_every=train_cfg.log_every,
        )

    def create_mappo_config(self, train_config: TrainConfig) -> MAPPOMultiHeadConfig:
        return get_mappo_config(
            train_config,
            critic_variant=self.critic_variant,
            critic_attention_heads=self.critic_attention_heads,
        )

    def create_algorithm(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        role_ids: Any,
        mappo_config: MAPPOMultiHeadConfig,
    ) -> MAPPOMultiHeadAlgorithm:
        return create_algorithm(obs_dim, action_dim, state_dim, n_agents, role_ids, mappo_config)

    def create_rollout_buffer(
        self,
        buffer_size: int,
        obs_space: int,
        state_space: int,
        action_space: int,
        num_agents: int,
        role_ids: Any,
        chaser_team_obs_dim: int,
        chaser_team_reward_dim: int,
        device: str,
    ) -> RolloutBuffer:
        return create_rollout_buffer(
            buffer_size, obs_space, state_space, action_space,
            num_agents, role_ids, chaser_team_obs_dim, chaser_team_reward_dim, device
        )


@dataclass
class Chasing3Chase1PositionControlConfig(Chasing3Chase1Config):
    """3Chase1 local-subgoal MAPPO with fixed PID and physical allocation."""

    model_type: str = "MAPPO_POSITION_CONTROL"
    controller_backend: str = "body_pid_wrench"
    target_body_delta_limits: tuple[float, float, float] = (1.5, 0.5, 1.5)
    enable_yaw_control: bool = True
    yaw_error_limit_deg: float = 90.0

    def create_mappo_config(self, train_config: TrainConfig) -> MAPPOMultiHeadPositionControlConfig:
        meta_action_dim = 4 if self.enable_yaw_control else 3
        config = MAPPOMultiHeadPositionControlConfig(
            actor_hidden_dim=train_config.actor_hidden_dim,
            actor_num_layers=train_config.actor_num_layers,
            critic_hidden_dim=train_config.critic_hidden_dim,
            critic_num_layers=train_config.critic_num_layers,
            learning_rate_actor=train_config.learning_rate_actor,
            learning_rate_critic=train_config.learning_rate_critic,
            gamma=train_config.gamma,
            td_lambda=train_config.td_lambda,
            normalize_advantage=train_config.normalize_advantage,
            normalize_return=train_config.normalize_return,
            epochs=train_config.epochs,
            ppo_clip=train_config.ppo_clip,
            entropy_coef=train_config.entropy_coef,
            clip_gradients=train_config.clip_gradients,
            device=train_config.device,
            chaser_team_obs_dim=train_config.chaser_team_obs_dim,
            chaser_team_action_dim=train_config.chaser_team_action_dim,
            action_interface="position_controller",
            policy_action_dim=meta_action_dim,
            rollout_action_dim=meta_action_dim,
            env_action_dim=train_config.chaser_team_action_dim,
            target_body_delta_limits=self.target_body_delta_limits,
            enable_yaw_control=self.enable_yaw_control,
            yaw_error_limit_deg=self.yaw_error_limit_deg,
            controller_backend=self.controller_backend,
            critic_variant=self.critic_variant,
            critic_attention_heads=self.critic_attention_heads,
        )
        config.body_pid_wrench_controller.device = train_config.device
        config.body_pid_wrench_controller.enable_yaw_control = self.enable_yaw_control
        return config

    def create_algorithm(
        self,
        obs_dim: int,
        action_dim: int,
        state_dim: int,
        n_agents: int,
        role_ids: Any,
        mappo_config: MAPPOMultiHeadPositionControlConfig,
    ) -> MAPPOMultiHeadPositionControlAlgorithm:
        return MAPPOMultiHeadPositionControlAlgorithm(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            role_ids=role_ids,
            config=mappo_config,
        )


@dataclass
class Chasing3Chase1TraditionalFormationBaselineConfig(Chasing3Chase1PositionControlConfig):
    """No-checkpoint geometric 3Chase1 baseline with the PID/wrench backend."""

    model_type: str = "TRADITIONAL_FORMATION_PID_WRENCH_BASELINE"
    requires_checkpoint: bool = False
    desired_netter_spacing: float = 6.0
    netter_forward_offset: float = 2.0
    herder_behind_offset: float = 3.0

    def create_baseline_algorithm(self, n_agents: int, role_ids: Any, device: str):
        pid_config = self.create_mappo_config(self.create_train_config()).body_pid_wrench_controller
        pid_config.device = device
        return TraditionalFormationPIDWrenchBaseline(
            n_agents=n_agents,
            role_ids=role_ids,
            config=TraditionalFormationBaselineConfig(
                target_body_delta_limits=self.target_body_delta_limits,
                yaw_error_limit_deg=self.yaw_error_limit_deg,
                desired_netter_spacing=self.desired_netter_spacing,
                netter_forward_offset=self.netter_forward_offset,
                herder_behind_offset=self.herder_behind_offset,
                controller_backend=self.controller_backend,
                body_pid_wrench_controller=pid_config,
            ),
        )


def get_chasing_configs() -> list:
    """获取所有追逐场景配置"""
    return [
        Chasing3Chase1Config(
            name="chasing_3_chase_1_end_to_end",
            model_type="MAPPO",
            description="3chaser1 (herding): 1 Herder + 2 Netter chasing 1 Prey",
            env_config=Chasing3Chase1EnvironmentConfig(
                env_path=THREE_CHASE_ONE_UNITY_BUILD_PATH,
                env_base_port=6000,
                time_scale=10.0,
                env_type="3chase1",
                herder_action_source="python_policy",
                netter_action_source="python_policy",
                prey_action_source="unity_baseline",
            ),
            training_config=Chasing3Chase1TrainingConfig(
                learning_rate_actor=1e-4,
                learning_rate_critic=1e-4,
                gamma=0.99,
                td_lambda=0.95,
                normalize_advantage=True,
                normalize_return=False,
                epochs=10, # 每收集够batch_size*num_envs个episode进行一次训练，训练epochs轮（梯度下降epochs次），然后再和环境交互继续收集数据
                ppo_clip=0.2,
                entropy_coef=0.001,
                clip_gradients=0.5,
                total_timesteps=10_000_000,
                # eval params
                eval_steps=3, # 每train（梯度下降epochs结束算train一次）多少次才eval
                num_eval_ep=16, # 每次eval的时候进行多少轮 (episode)
                num_eval_envs=4, # 使用4个平行环境评估
                eval_time_scale=1.0,
                save_freq=50_000,
                log_every=10,
                batch_size=64,
            ),
            actor_hidden_dim=64,
            actor_num_layers=2,
            critic_hidden_dim=64,
            critic_num_layers=2,
            chaser_team_obs_dim=30,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=32,
            seed=42,
            device="cuda",
        ),
        Chasing3Chase1Config(
            name="chasing_3_chase_1_test", # 冒烟测试专用
            model_type="MAPPO",
            description="3chaser1 test: 1 Herder + 2 Netter chasing 1 Prey (test env)",
            env_config=Chasing3Chase1EnvironmentConfig(
                env_path=THREE_CHASE_ONE_UNITY_BUILD_PATH,
                env_base_port=5005,
                time_scale=30.0,
                env_type="3chase1",
            ),
            training_config=Chasing3Chase1TrainingConfig(
                learning_rate_actor=3e-4,
                learning_rate_critic=3e-4,
                gamma=0.99,
                td_lambda=0.95,
                normalize_advantage=True,
                normalize_return=False,
                epochs=5,
                ppo_clip=0.2,
                entropy_coef=0.001,
                clip_gradients=0.5,
                total_timesteps=900*2*2, # 也就是每个环境两轮
                batch_size=1, # 实际的batch_size为2
                log_every=10, # 每 10 timesteps (就是和环境交互的次数) 记录一次日志
                eval_steps=1,
                eval_time_scale=30.0,
                num_eval_ep=16,
                num_eval_envs=4,
                save_freq=900,
            ),
            actor_hidden_dim=64,
            actor_num_layers=2,
            critic_hidden_dim=64,
            critic_num_layers=2,
            chaser_team_obs_dim=30,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=2,
            seed=42,
            device="cuda",
        ),
        Chasing3Chase1PositionControlConfig(
            name="chasing_3_chase_1_position_control",
            model_type="MAPPO_POSITION_CONTROL",
            description="3chase1 30D MAPPO local subgoals with fixed body PID and physical wrench allocation",
            controller_backend="body_pid_wrench",
            env_config=Chasing3Chase1EnvironmentConfig(
                env_path=THREE_CHASE_ONE_UNITY_BUILD_PATH,
                env_base_port=1000,
                time_scale=10.0,
                env_type="3chase1",
                herder_action_source="python_policy",
                netter_action_source="python_policy",
                prey_action_source="unity_baseline",
            ),
            training_config=Chasing3Chase1TrainingConfig(
                learning_rate_actor=1e-4,
                learning_rate_critic=1e-4,
                gamma=0.99,
                td_lambda=0.95,
                normalize_advantage=True,
                normalize_return=False,
                epochs=8,
                ppo_clip=0.2,
                entropy_coef=0.001,
                clip_gradients=0.5,
                total_timesteps=10_000_000,
                eval_steps=3,
                num_eval_ep=16,
                num_eval_envs=4,
                eval_time_scale=1.0,
                save_freq=50_000,
                log_every=10,
                batch_size=64,
            ),
            actor_hidden_dim=64,
            actor_num_layers=2,
            critic_hidden_dim=64,
            critic_num_layers=2,
            chaser_team_obs_dim=30,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=32,
            seed=42,
            device="cuda",
            
            target_body_delta_limits=(1.5, 0.5, 1.5),
            yaw_error_limit_deg=90.0,
        ),
        Chasing3Chase1TraditionalFormationBaselineConfig(
            name="chasing_3_chase_1_traditional_formation_baseline",
            description=(
                "3Chase1 local geometric formation guidance -> body PID -> "
                "physical wrench allocator -> eight thrusters"
            ),
            controller_backend="body_pid_wrench",
            env_config=Chasing3Chase1EnvironmentConfig(
                env_path=THREE_CHASE_ONE_UNITY_BUILD_PATH,
                env_base_port=1800,
                time_scale=1.0,
                env_type="3chase1",
                herder_action_source="python_policy",
                netter_action_source="python_policy",
                prey_action_source="wrapper_escape",
            ),
            training_config=Chasing3Chase1TrainingConfig(
                num_eval_ep=10,
                num_eval_envs=1,
                eval_time_scale=1.0,
            ),
            chaser_team_obs_dim=30,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=1,
            seed=42,
            device="cuda",
            target_body_delta_limits=(1.5, 0.5, 1.5),
            yaw_error_limit_deg=90.0,
            desired_netter_spacing=6.0,
            netter_forward_offset=2.0,
            herder_behind_offset=3.0,
        ),
        Chasing3Chase1PositionControlConfig(
            name="chasing_3_chase_1_position_control_test", # 冒烟测试专用
            model_type="MAPPO_POSITION_CONTROL",
            description="3chase1 30D MAPPO local-subgoal PID/wrench smoke-test configuration",
            controller_backend="body_pid_wrench",
            env_config=Chasing3Chase1EnvironmentConfig(
                env_path=THREE_CHASE_ONE_UNITY_BUILD_PATH,
                env_base_port=1000,
                time_scale=10.0,
                env_type="3chase1",
                herder_action_source="python_policy",
                netter_action_source="python_policy",
                prey_action_source="unity_baseline",
            ),
            training_config=Chasing3Chase1TrainingConfig(
                learning_rate_actor=1e-4,
                learning_rate_critic=1e-4,
                gamma=0.99,
                td_lambda=0.95,
                normalize_advantage=True,
                normalize_return=False,
                epochs=8,
                ppo_clip=0.2,
                entropy_coef=0.001,
                clip_gradients=0.5,
                total_timesteps=10_000_000,
                eval_steps=3,
                num_eval_ep=16,
                num_eval_envs=1,
                eval_time_scale=1.0,
                save_freq=50_000,
                log_every=10,
                batch_size=64,
            ),
            actor_hidden_dim=64,
            actor_num_layers=2,
            critic_hidden_dim=64,
            critic_num_layers=2,
            chaser_team_obs_dim=30,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=1,
            seed=42,
            device="cuda",
            
            target_body_delta_limits=(1.5, 0.5, 1.5),
            yaw_error_limit_deg=90.0,
        ),
        Chasing3Chase1PositionControlConfig(
            # Historical config id retained for existing YAML files. It now has
            # the same local body-subgoal semantics as the main configuration.
            name="chasing_3_chase_1_traditional_pose_controller",
            model_type="MAPPO_POSITION_CONTROL",
            description="Compatibility alias for 3chase1 body PID plus physical wrench allocation",
            controller_backend="body_pid_wrench",
            env_config=Chasing3Chase1EnvironmentConfig(
                env_path=THREE_CHASE_ONE_UNITY_BUILD_PATH,
                env_base_port=1000,
                time_scale=10.0,
                env_type="3chase1",
                herder_action_source="python_policy",
                netter_action_source="python_policy",
                prey_action_source="unity_baseline",
            ),
            training_config=Chasing3Chase1TrainingConfig(
                learning_rate_actor=1e-4,
                learning_rate_critic=1e-4,
                gamma=0.99,
                td_lambda=0.95,
                normalize_advantage=True,
                normalize_return=False,
                epochs=8,
                ppo_clip=0.2,
                entropy_coef=0.001,
                clip_gradients=0.5,
                total_timesteps=10_000_000,
                eval_steps=3,
                num_eval_ep=16,
                num_eval_envs=4,
                eval_time_scale=1.0,
                save_freq=500_000,
                log_every=10,
                batch_size=64,
            ),
            actor_hidden_dim=64,
            actor_num_layers=2,
            critic_hidden_dim=64,
            critic_num_layers=2,
            chaser_team_obs_dim=30,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=32,
            seed=42,
            device="cuda",
            
            target_body_delta_limits=(1.5, 0.5, 1.5),
            yaw_error_limit_deg=90.0,
        ),
        
    ]


__all__ = [
    "get_chasing_configs",
    "Chasing3Chase1Config",
    "Chasing3Chase1PositionControlConfig",
    "Chasing3Chase1TrainingConfig",
    "Chasing3Chase1EnvironmentConfig",
    "get_mappo_config",
    "create_algorithm",
    "create_rollout_buffer",
]
