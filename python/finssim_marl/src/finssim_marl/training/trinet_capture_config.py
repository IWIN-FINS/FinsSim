"""Configurations for the three-ROV passive-target TriNetCapture task."""

from dataclasses import dataclass, field
from typing import Any

from finssim_marl.algorithms.trinet_transport_baseline import (
    TriNetTransportStateMachineBaseline,
    TriNetTransportStateMachineBaselineConfig,
)
from finssim_marl.algorithms.trinet_capture_mappo import (
    TriNetCaptureMAPPOAlgorithm,
    TriNetCaptureMAPPOConfig,
)
from finssim_marl.training.chasing_3_chase_1_config import (
    Chasing3Chase1EnvironmentConfig,
    Chasing3Chase1PositionControlConfig,
    Chasing3Chase1TrainingConfig,
)


TRINET_CAPTURE_UNITY_BUILD_PATH = (
    "artifacts/unity_builds/marl/linux/FinsROV/TriNetCapture_headless/TriNetCapture.x86_64"
)


@dataclass
class TriNetCaptureEnvironmentConfig(Chasing3Chase1EnvironmentConfig):
    env_path: str | None = TRINET_CAPTURE_UNITY_BUILD_PATH
    env_type: str = "trinet_capture"
    prey_action_source: str = "python_policy"  # compatibility-only; there is no Prey behaviour.


@dataclass
class TriNetCapturePositionControlConfig(Chasing3Chase1PositionControlConfig):
    """Homogeneous 34D TriNet task using 4D subgoal PID/wrench control."""

    env_config: TriNetCaptureEnvironmentConfig = field(default_factory=TriNetCaptureEnvironmentConfig)
    training_config: Chasing3Chase1TrainingConfig = field(default_factory=Chasing3Chase1TrainingConfig)
    chaser_team_obs_dim: int = 34
    chaser_team_reward_dim: int = 3
    target_body_delta_limits: tuple[float, float, float] = (1.0, 0.6, 1.0)
    critic_variant: str = "trinet_deepsets"
    individual_advantage_weight: float = 0.10

    def create_mappo_config(self, train_config) -> TriNetCaptureMAPPOConfig:
        config = TriNetCaptureMAPPOConfig(
            learning_rate_actor=train_config.learning_rate_actor,
            learning_rate_critic=train_config.learning_rate_critic,
            gamma=train_config.gamma,
            td_lambda=train_config.td_lambda,
            normalize_advantage=train_config.normalize_advantage,
            epochs=train_config.epochs,
            ppo_clip=train_config.ppo_clip,
            entropy_coef=train_config.entropy_coef,
            clip_gradients=train_config.clip_gradients,
            device=train_config.device,
            actor_obs_dim=self.chaser_team_obs_dim,
            target_body_delta_limits=self.target_body_delta_limits,
            enable_yaw_control=self.enable_yaw_control,
            yaw_error_limit_deg=self.yaw_error_limit_deg,
            controller_backend=self.controller_backend,
            critic_variant=self.critic_variant,
            critic_attention_heads=self.critic_attention_heads,
            individual_advantage_weight=self.individual_advantage_weight,
        )
        config.body_pid_wrench_controller.device = train_config.device
        config.body_pid_wrench_controller.enable_yaw_control = self.enable_yaw_control
        return config

    def create_algorithm(self, obs_dim, action_dim, state_dim, n_agents, role_ids, mappo_config):
        return TriNetCaptureMAPPOAlgorithm(
            obs_dim=obs_dim,
            action_dim=action_dim,
            state_dim=state_dim,
            n_agents=n_agents,
            role_ids=role_ids,
            config=mappo_config,
        )


@dataclass
class TriNetCaptureStateMachineBaselineConfig(TriNetCapturePositionControlConfig):
    model_type: str = "TRINET_TRANSPORT_STATE_MACHINE_BASELINE"
    requires_checkpoint: bool = False
    hold_goal_distance_m: float = 0.08
    # Keep this wrapper in sync with the state-machine default.  Evaluation
    # constructs the baseline through this config, so a duplicate literal here
    # would otherwise silently override the algorithm setting.
    final_lift_height_m: float = TriNetTransportStateMachineBaselineConfig().final_lift_height_m
    final_lift_tolerance_m: float = 0.05
    tow_goal_waypoint_tolerance_m: float = 0.20
    # Keep room for an additive Top lift. Phase 2 also raises this dynamically
    # if ``net_lift_height_m + top_extra_lift_height_m`` is configured larger.
    target_body_delta_limits: tuple[float, float, float] = (1.0, 1.2, 1.0)
    yaw_error_limit_deg: float = 35.0
    # All three vertices share the initial controller basis. The transport
    # baseline uses reversible surge/sway instead of independent yaw turns,
    # which would shear the triangular net.
    use_heading_control: bool = False
    desired_rov_spacing_m: float = 0.85
    net_surface_side_length_m: float = 0.7287565
    # Target's desired interior location relative to the ROV-triangle centroid
    # in the shared initial controller frame (body YZ).
    target_in_net_plane_body_yz: tuple[float, float] = (0.35, 0.0)
    # The net endpoints follow the ROVs one-way, so a bounded edge-length
    # feedback is still needed to prevent vehicle drift from collapsing the
    # triangle. Keep it far below the earlier unstable high-gain loop.
    formation_gain: float = 0.08
    formation_velocity_gain: float = 0.0
    max_formation_correction_m: float = 0.05

    def create_baseline_algorithm(self, n_agents: int, role_ids: Any, device: str):
        pid = self.create_mappo_config(self.create_train_config()).body_pid_wrench_controller
        pid.device = device
        return TriNetTransportStateMachineBaseline(
            n_agents=n_agents,
            role_ids=role_ids,
            config=TriNetTransportStateMachineBaselineConfig(
                target_body_delta_limits=self.target_body_delta_limits,
                yaw_error_limit_deg=self.yaw_error_limit_deg,
                use_heading_control=self.use_heading_control,
                hold_goal_distance_m=self.hold_goal_distance_m,
                final_lift_height_m=self.final_lift_height_m,
                final_lift_tolerance_m=self.final_lift_tolerance_m,
                tow_goal_waypoint_tolerance_m=self.tow_goal_waypoint_tolerance_m,
                desired_rov_spacing_m=self.desired_rov_spacing_m,
                net_surface_side_length_m=self.net_surface_side_length_m,
                target_in_net_plane_body_yz=self.target_in_net_plane_body_yz,
                formation_gain=self.formation_gain,
                formation_velocity_gain=self.formation_velocity_gain,
                max_formation_correction_m=self.max_formation_correction_m,
                controller_backend=self.controller_backend,
                body_pid_wrench_controller=pid,
            ),
        )


def get_trinet_capture_configs() -> list:
    common_env = dict(
        env_path=TRINET_CAPTURE_UNITY_BUILD_PATH,
        env_type="trinet_capture",
        herder_action_source="python_policy",
        netter_action_source="python_policy",
        prey_action_source="python_policy",
    )
    return [
        TriNetCapturePositionControlConfig(
            name="trinet_capture_position_control",
            model_type="MAPPO_POSITION_CONTROL",
            description="Three homogeneous FinsROVs transport a passive Target with a triangular net into a submerged recovery Goal.",
            env_config=TriNetCaptureEnvironmentConfig(env_base_port=4300, time_scale=10.0, **common_env),
            training_config=Chasing3Chase1TrainingConfig(
                total_timesteps=10_000_000, batch_size=64, num_eval_envs=4,
                eval_time_scale=1.0, eval_steps=3, save_freq=50_000,
            ),
            chaser_team_obs_dim=34,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=32,
            device="cuda",
        ),
        TriNetCapturePositionControlConfig(
            name="trinet_capture_position_control_smoke",
            model_type="MAPPO_POSITION_CONTROL",
            description="Small single-area TriNetCapture smoke configuration.",
            env_config=TriNetCaptureEnvironmentConfig(env_base_port=4300, time_scale=2.0, **common_env),
            training_config=Chasing3Chase1TrainingConfig(
                total_timesteps=1_200, batch_size=1, num_eval_envs=1, num_eval_ep=1,
                eval_mode="serial", eval_steps=120, eval_time_scale=1.0, save_freq=1_200,
            ),
            chaser_team_obs_dim=34,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=1,
            device="cpu",
        ),
        TriNetCaptureStateMachineBaselineConfig(
            name="trinet_capture_transport_baseline",
            model_type="TRINET_TRANSPORT_STATE_MACHINE_BASELINE",
            description="Local 34D homogeneous Deploy/Capture/Tow/Hold baseline through the same PID/wrench allocator.",
            env_config=TriNetCaptureEnvironmentConfig(env_base_port=4300, time_scale=1.0, **common_env),
            training_config=Chasing3Chase1TrainingConfig(num_eval_envs=1, num_eval_ep=10, eval_mode="serial"),
            chaser_team_obs_dim=34,
            chaser_team_action_dim=8,
            chaser_team_reward_dim=3,
            num_envs=1,
            device="cuda",
        ),
    ]
