import warnings
from typing import Any, Callable, ClassVar, Optional, TypeVar, Union

import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn
from torch.nn import functional as F

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import ActorCriticCnnPolicy, ActorCriticPolicy, BasePolicy, MultiInputActorCriticPolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

SelfPPORewrite = TypeVar("SelfPPORewrite", bound="PPORewrite")


class PPORewrite(OnPolicyAlgorithm):
    """
    PPO rewrite for PID-parameter policy action + environment-action postprocess.

    Key additions vs upstream PPO:
    - policy_action_dim: override policy output dim independently from env.action_space.
    - action_postprocess_fn: map sampled policy actions to env actions before env.step().
    """

    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpPolicy": ActorCriticPolicy,
        "CnnPolicy": ActorCriticCnnPolicy,
        "MultiInputPolicy": MultiInputActorCriticPolicy,
    }

    def __init__(
        self,
        policy: Union[str, type[ActorCriticPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        use_sde: bool = False,
        sde_sample_freq: int = -1,
        rollout_buffer_class: Optional[type[RolloutBuffer]] = None,
        rollout_buffer_kwargs: Optional[dict[str, Any]] = None,
        target_kl: Optional[float] = None,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        policy_action_dim: Optional[int] = None,
        action_postprocess_fn: Optional[Callable[[np.ndarray, Any], np.ndarray]] = None,
    ):
        self.policy_action_dim = policy_action_dim
        self.action_postprocess_fn = action_postprocess_fn
        self._policy_action_space: Optional[spaces.Box] = None

        super().__init__(
            policy,
            env,
            learning_rate=learning_rate,
            n_steps=n_steps,
            gamma=gamma,
            gae_lambda=gae_lambda,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            use_sde=use_sde,
            sde_sample_freq=sde_sample_freq,
            rollout_buffer_class=rollout_buffer_class,
            rollout_buffer_kwargs=rollout_buffer_kwargs,
            stats_window_size=stats_window_size,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            device=device,
            seed=seed,
            _init_setup_model=False,
            supported_action_spaces=(
                spaces.Box,
                spaces.Discrete,
                spaces.MultiDiscrete,
                spaces.MultiBinary,
            ),
        )

        if normalize_advantage:
            assert batch_size > 1, "`batch_size` must be greater than 1."

        if self.env is not None:
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (not normalize_advantage), (
                f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            )
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )

        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl

        if _init_setup_model:
            self._setup_model()

    def set_action_postprocess_fn(self, fn: Optional[Callable[[np.ndarray, Any], np.ndarray]]) -> None:
        self.action_postprocess_fn = fn

    def _on_rollout_env_step(
        self,
        new_obs: Any,
        rewards: np.ndarray,
        dones: np.ndarray,
        infos: list[dict[str, Any]],
    ) -> None:
        """子类可在每次 env.step 后同步内部状态，例如重置控制器缓存。"""
        del new_obs, rewards, dones, infos

    def _compute_auxiliary_loss(
        self,
        rollout_data: Any,
        values: th.Tensor,
        log_prob: th.Tensor,
        entropy: Optional[th.Tensor],
    ) -> tuple[th.Tensor, dict[str, float]]:
        """子类可附加额外正则项；默认不添加。"""
        del rollout_data, values, log_prob, entropy
        return th.zeros((), device=self.device), {}

    def _setup_model(self) -> None:
        super()._setup_model()

        # Optional: resize policy output independent from env action space
        if self.policy_action_dim is not None and isinstance(self.action_space, spaces.Box):
            action_net = getattr(self.policy, "action_net", None)
            if isinstance(action_net, nn.Linear):
                in_features = action_net.in_features
                new_action_net = nn.Linear(in_features, int(self.policy_action_dim), device=self.device)
                self.policy.action_net = new_action_net
                self.policy.log_std = nn.Parameter(th.zeros(int(self.policy_action_dim), device=self.device))

                # 关键：替换参数后必须重建优化器，否则新参数不会被更新
                optimizer_kwargs = dict(self.policy.optimizer_kwargs)
                optimizer_kwargs["lr"] = self.lr_schedule(1)
                self.policy.optimizer = self.policy.optimizer_class(
                    self.policy.parameters(),
                    **optimizer_kwargs,
                )

                self._policy_action_space = spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(int(self.policy_action_dim),),
                    dtype=np.float32,
                )

                self.rollout_buffer = RolloutBuffer(
                    self.n_steps,
                    self.observation_space,
                    self._policy_action_space,
                    device=self.device,
                    gamma=self.gamma,
                    gae_lambda=self.gae_lambda,
                    n_envs=self.n_envs,
                )

        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, "`clip_range_vf` must be positive, pass `None` to deactivate vf clipping"
            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def _clip_policy_actions(self, actions: np.ndarray) -> np.ndarray:
        if self._policy_action_space is None:
            if isinstance(self.action_space, spaces.Box):
                return np.clip(actions, self.action_space.low, self.action_space.high)
            return actions
        target_dim = int(self._policy_action_space.shape[0])
        if actions.ndim == 2 and actions.shape[1] != target_dim and actions.shape[0] == target_dim:
            actions = actions.T
        if actions.ndim == 1 and actions.shape[0] != target_dim:
            raise ValueError(f"Policy action dim mismatch: got {actions.shape[0]}, expected {target_dim}")
        if actions.ndim == 2 and actions.shape[1] != target_dim:
            raise ValueError(f"Policy action shape mismatch: got {actions.shape}, expected (*, {target_dim})")
        return np.clip(actions, self._policy_action_space.low, self._policy_action_space.high)

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
    ) -> bool:
        assert self._last_obs is not None, "No previous observation was provided"
        self.policy.set_training_mode(False)

        n_steps = 0
        rollout_buffer.reset()
        dones = np.zeros(env.num_envs, dtype=bool)
        new_obs = self._last_obs
        if self.use_sde:
            self.policy.reset_noise(env.num_envs)

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            if self.use_sde and self.sde_sample_freq > 0 and n_steps % self.sde_sample_freq == 0:
                self.policy.reset_noise(env.num_envs)

            with th.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)  # type: ignore[arg-type]
                policy_actions, values, log_probs = self.policy(obs_tensor)

            sampled_actions = policy_actions.cpu().numpy()
            clipped_policy_actions = self._clip_policy_actions(sampled_actions)

            env_actions = clipped_policy_actions
            if self.action_postprocess_fn is not None:
                env_actions = self.action_postprocess_fn(clipped_policy_actions, self._last_obs)

            new_obs, rewards, dones, infos = env.step(env_actions)

            self.num_timesteps += env.num_envs

            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                sampled_actions = sampled_actions.reshape(-1, 1)

            for idx, done in enumerate(dones):
                if done and infos[idx].get("terminal_observation") is not None and infos[idx].get("TimeLimit.truncated", False):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]  # type: ignore[arg-type]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,  # type: ignore[arg-type]
                sampled_actions,
                rewards,
                self._last_episode_starts,  # type: ignore[arg-type]
                values,
                log_probs,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones
            self._on_rollout_env_step(new_obs, rewards, dones, infos)

        with th.no_grad():
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()
        return True

    def train(self) -> None:
        self.policy.set_training_mode(True)
        self._update_learning_rate(self.policy.optimizer)
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        clip_range_vf: Optional[float] = None
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        head3_grad_norms: list[float] = []
        action_abs_means: list[float] = []
        action_sat_fracs: list[float] = []
        auxiliary_metric_values: dict[str, list[float]] = {}

        continue_training = True
        approx_kl_divs: list[float] = []
        loss = th.tensor(0.0, device=self.device)
        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(rollout_data.observations, actions)
                values = values.flatten()
                action_abs_means.append(float(actions.abs().mean().detach().cpu().item()))
                action_sat_fracs.append(float((actions.abs() > 0.95).float().mean().detach().cpu().item()))
                advantages = rollout_data.advantages
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    assert clip_range_vf is not None
                    clip_vf = float(clip_range_vf)
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_vf, clip_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())
                auxiliary_loss, auxiliary_metrics = self._compute_auxiliary_loss(
                    rollout_data,
                    values,
                    log_prob,
                    entropy,
                )
                auxiliary_loss = auxiliary_loss.to(device=self.device, dtype=policy_loss.dtype)
                for name, value in auxiliary_metrics.items():
                    auxiliary_metric_values.setdefault(name, []).append(float(value))

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss + auxiliary_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = float(th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().item())
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                if hasattr(self.policy, "log_std") and self.policy.log_std.grad is not None:
                    grad_head = self.policy.log_std.grad[: min(3, self.policy.log_std.grad.shape[0])]
                    head3_grad_norms.append(float(th.norm(grad_head, p=2).detach().cpu().item()))

                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break

        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        self.logger.record(
            "train/policy_action_abs_mean",
            float(np.mean(action_abs_means)) if len(action_abs_means) > 0 else 0.0,
        )
        self.logger.record(
            "train/policy_action_sat_frac",
            float(np.mean(action_sat_fracs)) if len(action_sat_fracs) > 0 else 0.0,
        )
        for name, values_list in auxiliary_metric_values.items():
            self.logger.record(
                f"train/{name}",
                float(np.mean(values_list)) if len(values_list) > 0 else 0.0,
            )
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())
            head3 = self.policy.log_std[: min(3, self.policy.log_std.shape[0])]
            self.logger.record("train/log_std_head3_mean", head3.mean().item())
            self.logger.record("train/std_head3", th.exp(head3).mean().item())
            self.logger.record("train/log_std_head3_grad_norm", float(np.mean(head3_grad_norms)) if len(head3_grad_norms) > 0 else 0.0)

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)

    def learn(
        self: SelfPPORewrite,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "PPORewrite",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfPPORewrite:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )
