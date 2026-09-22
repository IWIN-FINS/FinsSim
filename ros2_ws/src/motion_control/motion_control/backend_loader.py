from __future__ import annotations

from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from dataclasses import replace
import importlib
import os
from pathlib import Path
import sys
from typing import Any, Optional

import numpy as np


DEFAULT_LINEAR_VELOCITY_SCALE: tuple[float, float, float] = (0.5641, 0.4965, 0.8870)
DEFAULT_ANGULAR_VELOCITY_SCALE: tuple[float, float, float] = (0.6163, 0.6000, 0.6621)
DEFAULT_REFERENCE_VELOCITY_LIMIT: tuple[float, float, float] = (0.30, 0.30, 0.30)


@dataclass(frozen=True)
class BackendSpec:
    module_name: str
    factory_name: str
    observation_mode: str
    requires_checkpoint: bool
    supports_target_orientation: bool = False
    target_orientation_mode: str = "none"


BACKEND_SPECS: dict[str, BackendSpec] = {
    "traditional_pid_position": BackendSpec(
        module_name="finssim_rl.models.traditional_position_pid",
        factory_name="get_traditional_position_pid_configs",
        observation_mode="pose20",
        requires_checkpoint=False,
        supports_target_orientation=True,
        target_orientation_mode="yaw_only",
    ),
    "traditional_ff_velocity": BackendSpec(
        module_name="finssim_rl.models.traditional_velocity_pid",
        factory_name="get_traditional_velocity_pid_configs",
        observation_mode="velocity_normalized_body",
        requires_checkpoint=False,
    ),
    "traditional_pid_velocity": BackendSpec(
        module_name="finssim_rl.models.traditional_velocity_pid",
        factory_name="get_traditional_velocity_pid_configs",
        observation_mode="velocity_normalized_body",
        requires_checkpoint=False,
    ),
    "traditional_pid_ff_velocity": BackendSpec(
        module_name="finssim_rl.models.traditional_velocity_pid",
        factory_name="get_traditional_velocity_pid_configs",
        observation_mode="velocity_normalized_body",
        requires_checkpoint=False,
    ),
    "traditional_pid_ff_eso_velocity": BackendSpec(
        module_name="finssim_rl.models.traditional_velocity_pid",
        factory_name="get_traditional_velocity_pid_configs",
        observation_mode="velocity_normalized_body",
        requires_checkpoint=False,
    ),
    "ppo_control_for_pose": BackendSpec(
        module_name="finssim_rl.models.ppo_control",
        factory_name="get_ppo_control_configs",
        observation_mode="pose20",
        requires_checkpoint=True,
        supports_target_orientation=True,
        target_orientation_mode="full",
    ),
    "ppo_control_for_moving_target_dynamic": BackendSpec(
        module_name="finssim_rl.models.ppo_control",
        factory_name="get_ppo_control_configs",
        observation_mode="pose13",
        requires_checkpoint=True,
    ),
    "ppo_control_for_moving_target_dynamic_smoke": BackendSpec(
        module_name="finssim_rl.models.ppo_control",
        factory_name="get_ppo_control_configs",
        observation_mode="pose13",
        requires_checkpoint=True,
    ),
    "ppo_control_for_velocity": BackendSpec(
        module_name="finssim_rl.models.ppo_control",
        factory_name="get_ppo_control_configs",
        observation_mode="velocity_normalized_body",
        requires_checkpoint=True,
    ),
    "ppo_control_for_velocity_no_rotation": BackendSpec(
        module_name="finssim_rl.models.ppo_control",
        factory_name="get_ppo_control_configs",
        observation_mode="velocity_normalized_body",
        requires_checkpoint=True,
    ),
    "ppo_control_v2_for_pose": BackendSpec(
        module_name="finssim_rl.models.ppo_control_v2",
        factory_name="get_ppo_control_v2_configs",
        observation_mode="pose20",
        requires_checkpoint=True,
        supports_target_orientation=True,
        target_orientation_mode="full",
    ),
    "ppo_wrench_for_pose_empirical_thruster_mixer": BackendSpec(
        module_name="finssim_rl.models.ppo_wrench_control",
        factory_name="get_ppo_wrench_control_configs",
        observation_mode="pose20",
        requires_checkpoint=True,
        supports_target_orientation=True,
        target_orientation_mode="full",
    ),
    "ppo_wrench_for_pose_physical_wrench_allocator": BackendSpec(
        module_name="finssim_rl.models.ppo_wrench_control",
        factory_name="get_ppo_wrench_control_configs",
        observation_mode="pose20",
        requires_checkpoint=True,
        supports_target_orientation=True,
        target_orientation_mode="full",
    ),
    "ppo_trajectory_tracking_thruster8": BackendSpec(
        module_name="finssim_rl.models.ppo_control",
        factory_name="get_ppo_control_configs",
        observation_mode="trajectory30",
        requires_checkpoint=True,
    ),
    "ppo_trajectory_tracking_wrench6": BackendSpec(
        module_name="finssim_rl.models.ppo_wrench_control",
        factory_name="get_ppo_wrench_control_configs",
        observation_mode="trajectory30",
        requires_checkpoint=True,
    ),
    "ppo_control_v2_for_velocity": BackendSpec(
        module_name="finssim_rl.models.ppo_control_v2",
        factory_name="get_ppo_control_v2_configs",
        observation_mode="velocity_normalized_body",
        requires_checkpoint=True,
    ),
    "hybrid_pid_pose_v3": BackendSpec(
        module_name="finssim_rl.training.hybrid_ppo_pid_config_v3",
        factory_name="get_hybrid_control_configs_v3",
        observation_mode="pose20",
        requires_checkpoint=True,
    ),
    "hybrid_pid_pose_v3_random_tau": BackendSpec(
        module_name="finssim_rl.training.hybrid_ppo_pid_config_v3",
        factory_name="get_hybrid_control_configs_v3",
        observation_mode="pose20",
        requires_checkpoint=True,
    ),
    "hybrid_pid_frozen_v3": BackendSpec(
        module_name="finssim_rl.training.hybrid_ppo_pid_config_v3",
        factory_name="get_hybrid_control_configs_v3",
        observation_mode="pose20",
        requires_checkpoint=True,
    ),
    "hybrid_pid_pose_residual": BackendSpec(
        module_name="finssim_rl.training.hybrid_ppo_pid_residual_config",
        factory_name="get_hybrid_control_configs_residual",
        observation_mode="pose20",
        requires_checkpoint=True,
    ),
    "hybrid_pid_pose_residual_frozen": BackendSpec(
        module_name="finssim_rl.training.hybrid_ppo_pid_residual_config",
        factory_name="get_hybrid_control_configs_residual",
        observation_mode="pose20",
        requires_checkpoint=True,
    ),
    "isaaclab_warpauv_poshold": BackendSpec(
        module_name="finssim_isaaclab.policy",
        factory_name="load_isaaclab_backend",
        observation_mode="isaaclab_pose17",
        requires_checkpoint=True,
        supports_target_orientation=True,
        target_orientation_mode="full",
    ),
    "isaaclab_finsrov_hold_for_position": BackendSpec(
        module_name="finssim_isaaclab.policy",
        factory_name="load_finsrov_hold_for_position_backend",
        observation_mode="pose16_rot6d",
        requires_checkpoint=True,
        supports_target_orientation=True,
        target_orientation_mode="yaw_only",
    ),
    "isaaclab_finsrov_hold_for_position_wrench": BackendSpec(
        module_name="finssim_isaaclab.policy",
        factory_name="load_finsrov_hold_for_position_wrench_backend",
        observation_mode="pose16_rot6d",
        requires_checkpoint=True,
        supports_target_orientation=True,
        target_orientation_mode="yaw_only",
    ),
}


@dataclass
class LoadedBackend:
    name: str
    observation_mode: str
    model: Any
    supports_target_orientation: bool = False
    target_orientation_mode: str = "none"
    linear_velocity_scale: tuple[float, float, float] = DEFAULT_LINEAR_VELOCITY_SCALE
    angular_velocity_scale: tuple[float, float, float] = DEFAULT_ANGULAR_VELOCITY_SCALE
    reference_velocity_limit: tuple[float, float, float] = DEFAULT_REFERENCE_VELOCITY_LIMIT

    def predict_action(self, observation: np.ndarray) -> np.ndarray:
        action, _ = self.model.predict(observation, deterministic=True)
        action_np = np.asarray(action, dtype=np.float32)
        if action_np.ndim > 1:
            action_np = action_np.reshape(-1)
        return action_np.astype(np.float32, copy=False)

    def predict_policy_action(self, observation: np.ndarray) -> np.ndarray:
        """Return the raw deterministic policy output before model-specific postprocessing.

        PPOVirtualControlModel.predict() intentionally returns 8D simulator thruster
        commands.  Wrench deployment needs the learned 6D policy head output instead.
        """

        policy = getattr(self.model, "policy", None)
        if policy is None:
            return self.predict_action(observation)

        import torch

        if hasattr(policy, "set_training_mode"):
            policy.set_training_mode(False)
        obs_tensor, _ = policy.obs_to_tensor(observation)
        with torch.no_grad():
            action, _, _ = policy(obs_tensor, deterministic=True)
        action_np = action.detach().cpu().numpy()
        if action_np.ndim > 1:
            action_np = action_np.reshape(-1)
        return np.asarray(action_np, dtype=np.float32)

    def reset(self) -> None:
        if hasattr(self.model, "reset"):
            self.model.reset()


def list_supported_backends() -> list[str]:
    return sorted(BACKEND_SPECS.keys())


def _prefer_virtualenv_site_packages() -> None:
    virtual_env = os.environ.get("VIRTUAL_ENV", "").strip()
    if not virtual_env:
        return

    version_tag = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site_packages = Path(virtual_env).expanduser() / "lib" / version_tag / "site-packages"
    if not site_packages.is_dir():
        return

    site_packages_str = str(site_packages)
    if site_packages_str in sys.path:
        sys.path.remove(site_packages_str)
    sys.path.insert(0, site_packages_str)


def _resolve_repo_root() -> Optional[Path]:
    env_root = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if env_root:
        candidate = Path(env_root).expanduser().resolve()
        if (candidate / "python/finssim_rl/src/finssim_rl").is_dir():
            return candidate

    env_rl_src = os.environ.get("FINSSIM_RL_SRC", "").strip()
    if env_rl_src:
        candidate = Path(env_rl_src).expanduser().resolve()
        if (candidate / "finssim_rl").is_dir():
            return candidate.parent

    search_roots = [Path.cwd(), Path(__file__).resolve()]
    for root in search_roots:
        for parent in [root, *root.parents]:
            rl_dir = parent / "python/finssim_rl/src/finssim_rl"
            if rl_dir.is_dir():
                return parent
    return None


def _ensure_finssim_rl_importable() -> Path:
    repo_root = _resolve_repo_root()
    if repo_root is None:
        raise RuntimeError(
            "Could not locate the FinsSim repository root. "
            "Set FINSSIM_REPO_ROOT=/path/to/FinsSim before running the controller."
        )

    rl_src = repo_root / "python/finssim_rl/src"
    if str(rl_src) not in sys.path:
        sys.path.insert(0, str(rl_src))
    return repo_root


def _load_config(spec: BackendSpec, backend_name: str) -> Any:
    module = importlib.import_module(spec.module_name)
    factory = getattr(module, spec.factory_name)
    configs = factory()
    config_key = "ppo_control_for_pose" if backend_name == "ppo_trajectory_tracking_thruster8" else backend_name
    if config_key not in configs:
        raise RuntimeError(
            f"Backend `{backend_name}` was not returned by {spec.module_name}.{spec.factory_name}()."
        )
    return configs[config_key]


def _apply_training_overrides(config: Any, overrides: dict[str, Any]) -> None:
    if not overrides or not hasattr(config, "training_config"):
        return
    training_config = getattr(config, "training_config")
    if not is_dataclass(training_config):
        return
    valid_fields = {field.name for field in fields(training_config)}
    filtered = {key: value for key, value in overrides.items() if key in valid_fields}
    if not filtered:
        return
    setattr(config, "training_config", replace(training_config, **filtered))


def _build_runtime_hint(exc: Exception) -> str:
    message = str(exc)
    if "_ARRAY_API" in message or "numpy.core.multiarray failed to import" in message:
        return (
            "RL backend import failed because the current Python environment mixes "
            "stable-baselines3 / matplotlib binaries built for NumPy 1.x with NumPy 2.x. "
            "Please run the controller in the workspace virtualenv or downgrade to `numpy<2`."
        )
    return message


def _resolve_observation_mode(spec: BackendSpec, model: Any) -> str:
    observation_mode = spec.observation_mode
    observation_space = getattr(model, "observation_space", None)
    shape = getattr(observation_space, "shape", None)
    if observation_mode == "pose20" and shape == (14,):
        return "pose14"
    if observation_mode == "pose20" and shape == (16,):
        return "pose16_rot6d"
    return observation_mode


def load_backend(
    backend_name: str,
    *,
    checkpoint_path: str,
    device: str,
    training_overrides: Optional[dict[str, Any]] = None,
) -> LoadedBackend:
    if backend_name not in BACKEND_SPECS:
        raise RuntimeError(
            f"Unsupported backend `{backend_name}`. "
            f"Supported backends: {', '.join(list_supported_backends())}"
        )

    _prefer_virtualenv_site_packages()
    repo_root = _resolve_repo_root()
    if repo_root is None:
        raise RuntimeError(
            "Could not locate the FinsSim repository root. "
            "Set FINSSIM_REPO_ROOT=/path/to/FinsSim before running the controller."
        )
    if backend_name in {
        "isaaclab_warpauv_poshold",
        "isaaclab_finsrov_hold_for_position",
        "isaaclab_finsrov_hold_for_position_wrench",
    }:
        # The Isaac policy itself is pure PyTorch, but the output path reuses
        # FinsSim's existing ThrustAllocator.  The adapter is an explicit uv
        # dependency of ros2_ws; do not inject a source directory into sys.path.
        _ensure_finssim_rl_importable()
    else:
        _ensure_finssim_rl_importable()
    spec = BACKEND_SPECS[backend_name]
    checkpoint = checkpoint_path.strip()
    if checkpoint.lower().endswith(".zip"):
        checkpoint = checkpoint[:-4]
    if spec.requires_checkpoint and not checkpoint:
        raise RuntimeError(
            f"Backend `{backend_name}` requires a checkpoint_path, but none was provided."
        )

    try:
        if backend_name in {
            "isaaclab_warpauv_poshold",
            "isaaclab_finsrov_hold_for_position",
            "isaaclab_finsrov_hold_for_position_wrench",
        }:
            module = importlib.import_module(spec.module_name)
            factory = getattr(module, spec.factory_name)
            model = factory(checkpoint, device=device)
        else:
            config = _load_config(spec, backend_name)
            _apply_training_overrides(config, training_overrides or {})
            model = config.load_model_for_eval(checkpoint, device=device)
    except Exception as exc:  # pragma: no cover - runtime dependency issues are environment-specific.
        raise RuntimeError(_build_runtime_hint(exc)) from exc

    linear_velocity_scale = tuple(
        float(v) for v in getattr(model, "linear_velocity_scale", DEFAULT_LINEAR_VELOCITY_SCALE)
    )
    angular_velocity_scale = tuple(
        float(v) for v in getattr(model, "angular_velocity_scale", DEFAULT_ANGULAR_VELOCITY_SCALE)
    )
    reference_velocity_limit = tuple(
        float(v) for v in getattr(model, "reference_velocity_limit", DEFAULT_REFERENCE_VELOCITY_LIMIT)
    )

    loaded = LoadedBackend(
        name=backend_name,
        observation_mode=_resolve_observation_mode(spec, model),
        model=model,
        supports_target_orientation=spec.supports_target_orientation,
        target_orientation_mode=spec.target_orientation_mode,
        linear_velocity_scale=linear_velocity_scale,
        angular_velocity_scale=angular_velocity_scale,
        reference_velocity_limit=reference_velocity_limit,
    )
    loaded.reset()
    return loaded
