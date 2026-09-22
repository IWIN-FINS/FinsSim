"""
混合 PPO+PID 控制器（residual 或 direct bounded PID）

核心思想：
- residual 模式：PPO 输出 9 维对数残差，围绕 baseline PID 微调
- direct 模式：PPO 输出 9 维归一化 PID 坐标，直接映射到 PID 参数范围
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional, Sequence, Tuple, Union

from io import BufferedIOBase
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from stable_baselines3.common.vec_env import VecEnv

from finssim_rl.models.hybrid_ppo_pid_v3 import (
    ACTION_MEAN_CLAMP,
    DEFAULT_PID_PARAMS,
    LOG_STD_MAX,
    LOG_STD_MIN,
    PID_PARAM_DIM,
    TRAIN_TENSOR_CLAMP,
    HybridPPOPolicy,
    HybridPPOWithPIDv3,
    _sanitize_np_array,
    _sanitize_torch_obs,
    _sanitize_torch_tensor,
    build_error4_from_obs,
    extract_raw_position_error,
    get_stage_mask,
)


RESIDUAL_EPS = 1e-6
DEFAULT_RESIDUAL_LOG_SCALE = np.ones(PID_PARAM_DIM, dtype=np.float32)
DEFAULT_DIRECT_PID_MIN = np.array(
    [
        0.2, 0.001, 0.001,  # depth
        0.2, 0.001, 0.001,  # surge
        0.2, 0.001, 0.001,  # sway
    ],
    dtype=np.float32,
)
DEFAULT_DIRECT_PID_MAX = np.array(
    [
        20.0, 2.0, 5.0,  # depth
        20.0, 2.0, 5.0,  # surge
        20.0, 2.0, 5.0,  # sway
    ],
    dtype=np.float32,
)

DEFAULT_POSITION_PID_PARAMS: Tuple[float, ...] = (
   5.0, 0.1, 0.5,     # depth
   5.0, 0.3, 0.5,     # surge
   5.0, 0.3, 0.5,     # sway
)

def _prepare_pid_param_vector(
    values: Union[Sequence[float], np.ndarray],
    *,
    default_value: float = 1.0,
) -> np.ndarray:
    """将输入整理为长度 9 的参数向量。"""
    array = np.asarray(values, dtype=np.float32).reshape(-1)
    if array.shape[0] < PID_PARAM_DIM:
        padded = np.full((PID_PARAM_DIM,), float(default_value), dtype=np.float32)
        padded[: array.shape[0]] = array
        array = padded
    elif array.shape[0] > PID_PARAM_DIM:
        array = array[:PID_PARAM_DIM]
    return array.astype(np.float32, copy=False)


def map_policy_output_to_pid_params_and_residuals(
    policy_actions: np.ndarray,
    stage: str,
    base_pid_params: np.ndarray,
    residual_log_scale: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    将 PPO 输出的归一化残差动作映射为最终 PID 参数。

    公式：
      log_residual = action * residual_log_scale
      multiplier = exp(log_residual)
      pid_final = baseline_pid * multiplier
    """
    actions_np = np.asarray(policy_actions, dtype=np.float32)
    if actions_np.ndim == 1:
        actions_np = actions_np.reshape(1, -1)

    if actions_np.shape[1] < PID_PARAM_DIM:
        padded = np.zeros((actions_np.shape[0], PID_PARAM_DIM), dtype=np.float32)
        padded[:, : actions_np.shape[1]] = actions_np
        actions_np = padded
    elif actions_np.shape[1] > PID_PARAM_DIM:
        actions_np = actions_np[:, :PID_PARAM_DIM]

    actions_squashed = np.clip(actions_np, -1.0, 1.0)
    base = _prepare_pid_param_vector(base_pid_params, default_value=1.0).reshape(1, -1)
    base = np.maximum(base, RESIDUAL_EPS)
    scale = _prepare_pid_param_vector(residual_log_scale, default_value=1.0).reshape(1, -1)
    scale = np.maximum(scale, 0.0)

    log_residual = actions_squashed * scale
    residual_multiplier = np.exp(log_residual).astype(np.float32)
    generated_params = (base * residual_multiplier).astype(np.float32)

    stage_mask = get_stage_mask(stage).reshape(1, -1)
    params_final = generated_params * stage_mask + base * (1.0 - stage_mask)
    return params_final.astype(np.float32), residual_multiplier.astype(np.float32), log_residual.astype(np.float32)


def map_policy_output_to_direct_pid_params(
    policy_actions: np.ndarray,
    stage: str,
    pid_min_params: np.ndarray,
    pid_max_params: np.ndarray,
    inactive_pid_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    将 PPO 输出的归一化动作直接映射为 PID 参数。

    action=-1 对应 pid_min，action=1 对应 pid_max，中间按 log 尺度插值。
    """
    actions_np = np.asarray(policy_actions, dtype=np.float32)
    if actions_np.ndim == 1:
        actions_np = actions_np.reshape(1, -1)

    if actions_np.shape[1] < PID_PARAM_DIM:
        padded = np.zeros((actions_np.shape[0], PID_PARAM_DIM), dtype=np.float32)
        padded[:, : actions_np.shape[1]] = actions_np
        actions_np = padded
    elif actions_np.shape[1] > PID_PARAM_DIM:
        actions_np = actions_np[:, :PID_PARAM_DIM]

    actions_squashed = np.clip(actions_np, -1.0, 1.0)
    pid_min = _prepare_pid_param_vector(pid_min_params, default_value=RESIDUAL_EPS).reshape(1, -1)
    pid_max = _prepare_pid_param_vector(pid_max_params, default_value=1.0).reshape(1, -1)
    pid_min = np.maximum(pid_min, RESIDUAL_EPS)
    pid_max = np.maximum(pid_max, pid_min + RESIDUAL_EPS)

    ratio = (actions_squashed + 1.0) * 0.5
    log_pid = np.log(pid_min) + ratio * (np.log(pid_max) - np.log(pid_min))
    generated_params = np.exp(log_pid).astype(np.float32)

    if inactive_pid_params is None:
        inactive = np.sqrt(pid_min * pid_max).astype(np.float32)
    else:
        inactive = _prepare_pid_param_vector(inactive_pid_params, default_value=1.0).reshape(1, -1)
        inactive = np.clip(inactive, pid_min, pid_max).astype(np.float32)

    stage_mask = get_stage_mask(stage).reshape(1, -1)
    params_final = generated_params * stage_mask + inactive * (1.0 - stage_mask)
    return params_final.astype(np.float32)


def map_policy_output_to_pid_params(
    policy_actions: np.ndarray,
    stage: str,
    base_pid_params: np.ndarray,
    residual_log_scale: np.ndarray,
    pid_mapping_mode: str = "residual",
    direct_pid_min_params: Optional[np.ndarray] = None,
    direct_pid_max_params: Optional[np.ndarray] = None,
    inactive_pid_params: Optional[np.ndarray] = None,
) -> np.ndarray:
    """只返回策略输出映射后的最终 PID 参数。"""
    if pid_mapping_mode == "direct":
        return map_policy_output_to_direct_pid_params(
            policy_actions=policy_actions,
            stage=stage,
            pid_min_params=DEFAULT_DIRECT_PID_MIN if direct_pid_min_params is None else direct_pid_min_params,
            pid_max_params=DEFAULT_DIRECT_PID_MAX if direct_pid_max_params is None else direct_pid_max_params,
            inactive_pid_params=inactive_pid_params,
        )

    params_final, _, _ = map_policy_output_to_pid_params_and_residuals(
        policy_actions=policy_actions,
        stage=stage,
        base_pid_params=base_pid_params,
        residual_log_scale=residual_log_scale,
    )
    return params_final


class HybridPPOWithPIDResidual(HybridPPOWithPIDv3):
    """
    基于 baseline PID 的 residual PPO 控制器。

    与 v3 的区别：
    - v3: PPO 输出极点/时间常数，再映射到 PID
    - residual: PPO 直接输出每个 PID 增益的对数残差
    """

    def __init__(
        self,
        *args,
        baseline_pid_params: Sequence[float] = DEFAULT_PID_PARAMS,
        residual_log_scale: Union[float, Sequence[float], np.ndarray] = 1.0,
        pid_mapping_mode: str = "residual",
        direct_pid_min_params: Sequence[float] = DEFAULT_DIRECT_PID_MIN,
        direct_pid_max_params: Sequence[float] = DEFAULT_DIRECT_PID_MAX,
        residual_log_l2_coef: float = 0.0,
        residual_boundary_loss_coef: float = 0.0,
        residual_boundary_threshold: float = 0.85,
        random_initial_residual: bool = False,
        random_initial_residual_seed: Optional[int] = None,
        initial_residual_log_std: float = -3.0,
        initial_residual_reset_weights: bool = True,
        **kwargs,
    ) -> None:
        self.pid_mapping_mode = str(pid_mapping_mode).lower().strip()
        if self.pid_mapping_mode not in {"residual", "direct"}:
            raise ValueError("pid_mapping_mode must be 'residual' or 'direct'")

        self.baseline_pid_params = _prepare_pid_param_vector(baseline_pid_params, default_value=1.0)
        if np.any(self.baseline_pid_params <= 0.0):
            raise ValueError("baseline_pid_params must be positive for residual log-gain mapping")

        self.direct_pid_min_params = _prepare_pid_param_vector(direct_pid_min_params, default_value=RESIDUAL_EPS)
        self.direct_pid_max_params = _prepare_pid_param_vector(direct_pid_max_params, default_value=1.0)
        self.direct_pid_min_params = np.maximum(self.direct_pid_min_params, RESIDUAL_EPS)
        self.direct_pid_max_params = np.maximum(
            self.direct_pid_max_params,
            self.direct_pid_min_params + RESIDUAL_EPS,
        ).astype(np.float32, copy=False)
        self.direct_inactive_pid_params = np.sqrt(self.direct_pid_min_params * self.direct_pid_max_params).astype(
            np.float32,
            copy=False,
        )

        if np.isscalar(residual_log_scale):
            residual_log_scale_array = np.full(
                (PID_PARAM_DIM,),
                float(residual_log_scale),
                dtype=np.float32,
            )
        else:
            residual_log_scale_array = _prepare_pid_param_vector(residual_log_scale, default_value=1.0)
        self.residual_log_scale = np.maximum(residual_log_scale_array, 0.0).astype(np.float32, copy=False)
        self.residual_log_l2_coef = max(float(residual_log_l2_coef), 0.0)
        self.residual_boundary_loss_coef = max(float(residual_boundary_loss_coef), 0.0)
        self.residual_boundary_threshold = float(np.clip(residual_boundary_threshold, 0.0, 0.999))
        self.random_initial_residual = bool(random_initial_residual)
        self.random_initial_residual_seed = random_initial_residual_seed
        self.initial_residual_log_std = float(initial_residual_log_std)
        self.initial_residual_reset_weights = bool(initial_residual_reset_weights)

        # residual 版不使用随机 tau 初始化。
        kwargs.pop("random_initial_tau", None)
        kwargs.pop("random_initial_tau_log_std", None)
        kwargs.pop("random_initial_tau_seed", None)
        kwargs.pop("random_initial_tau_reset_weights", None)

        super().__init__(
            *args,
            random_initial_tau=False,
            random_initial_tau_log_std=-3.0,
            random_initial_tau_seed=None,
            random_initial_tau_reset_weights=False,
            **kwargs,
        )

    def _set_initial_log_std(self) -> None:
        """把 residual 动作初始方差设置为指定值。"""
        log_std_value = float(np.clip(self.initial_residual_log_std, LOG_STD_MIN, LOG_STD_MAX))
        nn.init.constant_(self.policy.log_std, log_std_value)
        print(
            f"[residual] log_std initialized to {log_std_value:.3g} "
            f"(std={np.exp(log_std_value):.4g})"
        )

    def _initialize_baseline_residual_action_head(self, action_net: nn.Linear) -> None:
        """让初始 residual 动作严格为 0，对应 baseline PID。"""
        with torch.no_grad():
            if self.initial_residual_reset_weights:
                action_net.weight.zero_()
            if action_net.bias is not None:
                action_net.bias.zero_()
        if self.pid_mapping_mode == "direct":
            params_np = self.direct_inactive_pid_params
            print(
                "[direct PID] action head initialized to midpoint PID: "
                f"DEPTH pid=[{params_np[0]:.4g}, {params_np[1]:.4g}, {params_np[2]:.4g}] | "
                f"SURGE pid=[{params_np[3]:.4g}, {params_np[4]:.4g}, {params_np[5]:.4g}] | "
                f"SWAY pid=[{params_np[6]:.4g}, {params_np[7]:.4g}, {params_np[8]:.4g}]",
                flush=True,
            )
        else:
            print("[residual] action head initialized to zero residual (baseline PID start)")

    def _initialize_random_residual_action_head(self, action_net: nn.Linear) -> None:
        """随机初始化动作 bias，对应范围内的一组初始 PID。"""
        generator = torch.Generator(device=self.device)
        if self.random_initial_residual_seed is not None:
            generator.manual_seed(int(self.random_initial_residual_seed))
        else:
            generator.seed()

        random_action = torch.empty(PID_PARAM_DIM, device=self.device)
        random_action.uniform_(-1.0, 1.0, generator=generator)

        with torch.no_grad():
            if self.initial_residual_reset_weights:
                action_net.weight.zero_()
            if action_net.bias is not None:
                action_net.bias.copy_(random_action)

        action_np = random_action.detach().cpu().numpy().astype(np.float32)
        if self.pid_mapping_mode == "direct":
            params_np = map_policy_output_to_direct_pid_params(
                policy_actions=action_np.reshape(1, -1),
                stage="all",
                pid_min_params=self.direct_pid_min_params,
                pid_max_params=self.direct_pid_max_params,
            )[0]
            print(
                "[direct PID random-init] initialized action bias from random PID in configured bounds: "
                f"DEPTH pid=[{params_np[0]:.4g}, {params_np[1]:.4g}, {params_np[2]:.4g}] | "
                f"SURGE pid=[{params_np[3]:.4g}, {params_np[4]:.4g}, {params_np[5]:.4g}] | "
                f"SWAY pid=[{params_np[6]:.4g}, {params_np[7]:.4g}, {params_np[8]:.4g}]",
                flush=True,
            )
            return

        params, multiplier, _ = map_policy_output_to_pid_params_and_residuals(
            policy_actions=action_np.reshape(1, -1),
            stage="all",
            base_pid_params=self.baseline_pid_params,
            residual_log_scale=self.residual_log_scale,
        )
        params_np = params[0]
        multiplier_np = multiplier[0]
        print(
            "[residual random-init] initialized action bias from random residuals: "
            f"DEPTH pid=[{params_np[0]:.4g}, {params_np[1]:.4g}, {params_np[2]:.4g}] "
            f"mult=[{multiplier_np[0]:.3g}, {multiplier_np[1]:.3g}, {multiplier_np[2]:.3g}] | "
            f"SURGE pid=[{params_np[3]:.4g}, {params_np[4]:.4g}, {params_np[5]:.4g}] "
            f"mult=[{multiplier_np[3]:.3g}, {multiplier_np[4]:.3g}, {multiplier_np[5]:.3g}] | "
            f"SWAY pid=[{params_np[6]:.4g}, {params_np[7]:.4g}, {params_np[8]:.4g}] "
            f"mult=[{multiplier_np[6]:.3g}, {multiplier_np[7]:.3g}, {multiplier_np[8]:.3g}]",
            flush=True,
        )

    def _policy_actions_to_env_actions(
        self,
        policy_actions: np.ndarray,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> np.ndarray:
        """将策略输出的 9 维 PID 控制参数转换为环境需要的 8 维推进器推力。"""
        actions_np = _sanitize_np_array(policy_actions, clamp_abs=1.0)
        if actions_np.ndim == 1:
            actions_np = actions_np.reshape(1, -1)

        params_final = map_policy_output_to_pid_params(
            policy_actions=actions_np,
            stage=self.current_stage,
            base_pid_params=self.baseline_pid_params,
            residual_log_scale=self.residual_log_scale,
            pid_mapping_mode=self.pid_mapping_mode,
            direct_pid_min_params=self.direct_pid_min_params,
            direct_pid_max_params=self.direct_pid_max_params,
            inactive_pid_params=self.direct_inactive_pid_params,
        )
        params_final = _sanitize_np_array(params_final, clamp_abs=TRAIN_TENSOR_CLAMP)
        params_tensor = torch.from_numpy(params_final).float().to(self.pid_controller.device)

        error4 = _sanitize_np_array(
            build_error4_from_obs(obs, default_num_envs=actions_np.shape[0]),
            clamp_abs=TRAIN_TENSOR_CLAMP,
        )
        raw_error = _sanitize_np_array(
            extract_raw_position_error(obs, default_num_envs=actions_np.shape[0]),
            clamp_abs=TRAIN_TENSOR_CLAMP,
        )
        error_tensor = torch.from_numpy(error4).float().to(self.pid_controller.device)
        raw_error_tensor = torch.from_numpy(raw_error).float().to(self.pid_controller.device)
        if error_tensor.ndim == 1:
            error_tensor = error_tensor.unsqueeze(0)
        if raw_error_tensor.ndim == 1:
            raw_error_tensor = raw_error_tensor.unsqueeze(0)

        actions_8d_list = []
        self.pid_controller.latest_tau_params = None
        with torch.no_grad():
            for i in range(actions_np.shape[0]):
                action_8d_i = self.pid_controller(
                    error=error_tensor[i : i + 1],
                    pid_params=params_tensor[i],
                    return_thrust=True,
                    raw_error=raw_error_tensor[i : i + 1],
                )
                actions_8d_list.append(action_8d_i)
            actions_8d = torch.cat(actions_8d_list, dim=0)
            actions_8d = _sanitize_torch_tensor(actions_8d, clamp_abs=1.0).cpu().numpy()

        return actions_8d

    def _setup_model(self) -> None:
        """设置模型，将 action_net 输出维度固定为 9 维 residual 参数。"""
        super(HybridPPOWithPIDv3, self)._setup_model()

        pid_action_dim = PID_PARAM_DIM
        action_net = getattr(self.policy, "action_net", None)
        if not isinstance(action_net, nn.Linear):
            print("[residual] action_net is not a Linear layer, skipping resize")
            return

        old_action_dim = int(action_net.out_features)
        old_weight = action_net.weight.data.clone()
        old_bias = action_net.bias.data.clone() if action_net.bias is not None else None

        if old_action_dim != pid_action_dim:
            action_net.out_features = pid_action_dim
            action_net.weight = nn.Parameter(torch.zeros(pid_action_dim, old_weight.shape[1]))
            action_net.bias = nn.Parameter(torch.zeros(pid_action_dim)) if old_bias is not None else None

            copy_dim = min(old_action_dim, pid_action_dim)
            action_net.weight.data[:copy_dim] = old_weight[:copy_dim]
            if action_net.bias is not None and old_bias is not None:
                action_net.bias.data[:copy_dim] = old_bias[:copy_dim]
            print(f"[residual] action_net resized: {old_action_dim} -> {pid_action_dim}")
        else:
            copy_dim = pid_action_dim
            print(f"[residual] action_net already has {pid_action_dim} dims")

        old_log_std = self.policy.log_std.detach().clone()
        last_log_std_val = old_log_std[-1] if old_log_std.numel() > 0 else torch.tensor(0.0)
        new_log_std = torch.full(
            (pid_action_dim,),
            float(last_log_std_val.item()),
            device=self.device,
            dtype=old_log_std.dtype,
        )
        new_log_std[: min(copy_dim, old_log_std.shape[0])] = old_log_std[: min(copy_dim, old_log_std.shape[0])]
        self.policy.log_std = nn.Parameter(new_log_std)
        print(f"[residual] log_std resized to: {tuple(self.policy.log_std.shape)}")
        self._set_initial_log_std()

        if self.random_initial_residual:
            self._initialize_random_residual_action_head(action_net)
        else:
            self._initialize_baseline_residual_action_head(action_net)

        self._trainable_pid_mask = self._trainable_pid_mask.to(self.device)

        def freeze_weight_hook(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if grad is not None:
                mask = self._trainable_pid_mask[: grad.shape[0]].view(-1, 1)
                grad = grad * mask
            return grad

        def freeze_bias_hook(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if grad is not None:
                mask = self._trainable_pid_mask[: grad.shape[0]]
                grad = grad * mask
            return grad

        action_net.weight.register_hook(freeze_weight_hook)
        if action_net.bias is not None:
            action_net.bias.register_hook(freeze_bias_hook)

        def freeze_log_std_hook(grad: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
            if grad is not None:
                mask = self._trainable_pid_mask[: grad.shape[0]]
                grad = grad * mask
            return grad

        self.policy.log_std.register_hook(freeze_log_std_hook)
        print("[residual] staged gradient hooks enabled")

        if hasattr(self, "pid_controller"):
            self.pid_controller = self.pid_controller.to(self.device)

    def _compute_auxiliary_loss(
        self,
        rollout_data: Any,
        values: torch.Tensor,
        log_prob: torch.Tensor,
        entropy: Optional[torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """对策略动作增加轻量正则，抑制长期贴边。"""
        del values, log_prob, entropy

        actions = _sanitize_torch_tensor(rollout_data.actions, clamp_abs=1.0)
        if actions.ndim == 1:
            actions = actions.unsqueeze(0)
        actions = actions[:, :PID_PARAM_DIM]

        action_mask = self._trainable_pid_mask[: actions.shape[1]].to(device=actions.device, dtype=actions.dtype).view(1, -1)
        active_dims = torch.clamp(action_mask.sum(dim=1), min=1.0)

        if self.pid_mapping_mode == "direct":
            action_l2_per_sample = (actions.pow(2) * action_mask).sum(dim=1) / active_dims
            rms_metric = torch.sqrt(action_l2_per_sample.mean())
        else:
            scale = torch.as_tensor(
                self.residual_log_scale[: actions.shape[1]],
                dtype=actions.dtype,
                device=actions.device,
            ).view(1, -1)
            log_residual = actions * scale
            action_l2_per_sample = (log_residual.pow(2) * action_mask).sum(dim=1) / active_dims
            rms_metric = torch.sqrt(action_l2_per_sample.mean())

        boundary_over = torch.relu(torch.abs(actions) - self.residual_boundary_threshold)
        boundary_l2_per_sample = (boundary_over.pow(2) * action_mask).sum(dim=1) / active_dims
        boundary_frac = ((torch.abs(actions) > self.residual_boundary_threshold).float() * action_mask).sum(dim=1) / active_dims

        auxiliary_loss = (
            self.residual_log_l2_coef * action_l2_per_sample.mean()
            + self.residual_boundary_loss_coef * boundary_l2_per_sample.mean()
        )
        metrics = {
            "residual_aux_loss": float(auxiliary_loss.detach().cpu().item()),
            "residual_log_rms": float(rms_metric.detach().cpu().item()),
            "residual_boundary_frac": float(boundary_frac.mean().detach().cpu().item()),
        }
        return auxiliary_loss, metrics

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """使用 residual PPO + baseline PID 进行推理。"""
        try:
            if episode_start is not None and np.any(episode_start):
                self._reset_pid_controller(np.asarray(episode_start, dtype=bool))
            self.policy.set_training_mode(False)

            def _get_obs_from_dict(obs_dict: Dict[str, Any]) -> Any:
                if "state" in obs_dict:
                    return obs_dict["state"]
                if "observation" in obs_dict:
                    return obs_dict["observation"]
                if "obs" in obs_dict:
                    return obs_dict["obs"]
                return next(iter(obs_dict.values()))

            if isinstance(observation, dict):
                obs_ref = _get_obs_from_dict(observation)
                obs_is_batched = np.asarray(obs_ref).ndim > 1
            else:
                obs_is_batched = observation.ndim > 1

            with torch.no_grad():
                if isinstance(observation, dict):
                    obs_tensor = {
                        k: torch.from_numpy(v).to(self.device) if isinstance(v, np.ndarray) else v
                        for k, v in observation.items()
                    }
                else:
                    obs_array = np.asarray(observation)
                    obs_tensor = torch.from_numpy(obs_array).float().to(self.device)
                    if obs_tensor.ndim == 1:
                        obs_tensor = obs_tensor.unsqueeze(0)
                obs_tensor = _sanitize_torch_obs(obs_tensor)

                features = self.policy.features_extractor(obs_tensor)
                features = _sanitize_torch_tensor(features, clamp_abs=TRAIN_TENSOR_CLAMP)
                latent_pi, _ = self.policy.mlp_extractor(features)
                latent_pi = _sanitize_torch_tensor(latent_pi, clamp_abs=TRAIN_TENSOR_CLAMP)
                distribution = self.policy._get_action_dist_from_latent(latent_pi)
                if deterministic:
                    policy_output = distribution.mode()
                else:
                    policy_output = distribution.get_actions(deterministic=False)
                policy_output = _sanitize_torch_tensor(policy_output, clamp_abs=1.0)

                pid_params_np = map_policy_output_to_pid_params(
                    policy_actions=policy_output.detach().cpu().numpy(),
                    stage=self.current_stage,
                    base_pid_params=self.baseline_pid_params,
                    residual_log_scale=self.residual_log_scale,
                    pid_mapping_mode=self.pid_mapping_mode,
                    direct_pid_min_params=self.direct_pid_min_params,
                    direct_pid_max_params=self.direct_pid_max_params,
                    inactive_pid_params=self.direct_inactive_pid_params,
                )
                pid_params_np = _sanitize_np_array(pid_params_np, clamp_abs=TRAIN_TENSOR_CLAMP)
                pid_params_tensor = torch.from_numpy(pid_params_np).to(self.device)

                error4 = _sanitize_np_array(
                    self._build_predict_error4(observation),
                    clamp_abs=TRAIN_TENSOR_CLAMP,
                )
                raw_error = _sanitize_np_array(
                    extract_raw_position_error(observation, default_num_envs=pid_params_tensor.shape[0]),
                    clamp_abs=TRAIN_TENSOR_CLAMP,
                )
                error_tensor = torch.from_numpy(error4).float().to(self.device)
                raw_error_tensor = torch.from_numpy(raw_error).float().to(self.device)
                if error_tensor.ndim == 1:
                    error_tensor = error_tensor.unsqueeze(0)
                if raw_error_tensor.ndim == 1:
                    raw_error_tensor = raw_error_tensor.unsqueeze(0)
                if error_tensor.shape[-1] < 4:
                    error_tensor = torch.nn.functional.pad(error_tensor, (0, 4 - error_tensor.shape[-1]))

                self.pid_controller.latest_tau_params = None
                action_8d_list = []
                for i in range(pid_params_tensor.shape[0]):
                    action_8d_i = self.pid_controller(
                        error=error_tensor[i : i + 1],
                        pid_params=pid_params_tensor[i],
                        return_thrust=True,
                        raw_error=raw_error_tensor[i : i + 1],
                    )
                    action_8d_list.append(action_8d_i)

                action_tensor = torch.cat(action_8d_list, dim=0)
                action_tensor = _sanitize_torch_tensor(action_tensor, clamp_abs=1.0)

            action_np = _sanitize_np_array(action_tensor.cpu().numpy(), clamp_abs=1.0)
            expected_dim = self.actual_action_dim
            if not obs_is_batched and action_np.ndim == 2 and action_np.shape[0] == 1:
                action_np = action_np[0]

            if action_np.ndim == 1:
                if action_np.shape[0] < expected_dim:
                    action_np = np.pad(action_np, (0, expected_dim - action_np.shape[0]), constant_values=0.0)
                elif action_np.shape[0] > expected_dim:
                    action_np = action_np[:expected_dim]
            else:
                if action_np.shape[1] < expected_dim:
                    action_np = np.pad(
                        action_np,
                        ((0, 0), (0, expected_dim - action_np.shape[1])),
                        constant_values=0.0,
                    )
                elif action_np.shape[1] > expected_dim:
                    action_np = action_np[:, :expected_dim]
            return action_np, state
        except Exception as exc:
            print(f"ERROR in residual predict: {exc}")
            import traceback

            traceback.print_exc()
            if isinstance(observation, dict):
                obs_ref = next(iter(observation.values()))
                obs_ref_np = np.asarray(obs_ref)
                batch_size = obs_ref_np.shape[0] if obs_ref_np.ndim > 1 else 1
            else:
                batch_size = observation.shape[0] if observation.ndim > 1 else 1
            expected_dim = self.actual_action_dim
            zeros = np.zeros((batch_size, expected_dim) if batch_size > 1 else (expected_dim,), dtype=np.float32)
            return zeros, state

    @classmethod
    def load(
        cls,
        path: Union[str, Path, BufferedIOBase],
        env=None,
        device: Union[torch.device, str] = "auto",
        custom_objects: Optional[Dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """加载 residual 模型并恢复动作后处理函数。"""
        model = super().load(
            path,
            env=env,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
            force_reset=force_reset,
            **kwargs,
        )
        if hasattr(model, "set_action_postprocess_fn"):
            model.set_action_postprocess_fn(model._policy_actions_to_env_actions)
        return model


__all__ = [
    "DEFAULT_RESIDUAL_LOG_SCALE",
    "DEFAULT_DIRECT_PID_MIN",
    "DEFAULT_DIRECT_PID_MAX",
    "HybridPPOPolicy",
    "HybridPPOWithPIDResidual",
    "map_policy_output_to_direct_pid_params",
    "map_policy_output_to_pid_params",
    "map_policy_output_to_pid_params_and_residuals",
]
