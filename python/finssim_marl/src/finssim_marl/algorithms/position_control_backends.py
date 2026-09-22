"""Fixed body-frame PID plus physical-wrench position-control backends.

The 3Chase1 high-level policy emits a local subgoal error directly in the
controller body basis ``[surge, heave, sway]``. In Unity this is local
``[+X/right, +Y/up, +Z/forward]`` and is shared by ControllerBodyFrame. The
fourth meta action is an independent yaw error in degrees. This module
deliberately does not accept world poses or quaternion targets: those belonged
to the retired 43D observation interface.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch


THRUSTER_ACTION_DIM = 8

# [Fx, Fy, Fz, Mx, My, Mz], in controller body order. These are the
# simulated FinsROV physical wrench capabilities documented in
# docs/finsrov_wrench_capability_zh-CN.md and used by the 1Chase1 hierarchy.
SIM_PHYSICAL_WRENCH_LIMITS_BODY: Tuple[float, float, float, float, float, float] = (
    19.528527,
    22.501886,
    18.415027,
    3.863995,
    7.080833,
    3.079985,
)

DEFAULT_POSITION_PID_PARAMS: Tuple[float, ...] = (
    5.0,
    0.1,
    0.5,
    5.0,
    0.3,
    0.5,
    5.0,
    0.3,
    0.5,
)
DEFAULT_YAW_PID_PARAMS: Tuple[float, float, float] = (0.12, 0.0, 0.01)


def _find_workspace_root(start: Optional[Path] = None) -> Optional[Path]:
    current = (start or Path(__file__).resolve()).resolve()
    for candidate in (current, *current.parents):
        if (candidate / "configs").exists() and (candidate / "python" / "finssim_rl").exists():
            return candidate
    return None


def _resolve_rl_project_src(path_value: str) -> str:
    raw_path = Path(path_value).expanduser()
    if raw_path.is_absolute():
        resolved = raw_path.resolve()
    else:
        workspace_root = _find_workspace_root()
        candidates = [Path.cwd() / raw_path]
        if workspace_root is not None:
            candidates.append(workspace_root / raw_path)
        resolved = next(
            (candidate.resolve() for candidate in candidates if candidate.is_dir()),
            candidates[0].resolve(),
        )
    if not resolved.is_dir():
        raise FileNotFoundError(f"Could not resolve finssim_rl source dir: {path_value!r}")
    return str(resolved)


def _fit_thruster_actions(actions: np.ndarray) -> np.ndarray:
    result = np.nan_to_num(
        np.asarray(actions, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    if result.shape[-1] < THRUSTER_ACTION_DIM:
        padding = [(0, 0)] * result.ndim
        padding[-1] = (0, THRUSTER_ACTION_DIM - result.shape[-1])
        result = np.pad(result, padding, constant_values=0.0)
    elif result.shape[-1] > THRUSTER_ACTION_DIM:
        result = result[..., :THRUSTER_ACTION_DIM]
    return np.clip(result, -1.0, 1.0).astype(np.float32, copy=False)


class BodyFramePositionControlBackend:
    """Backend interface for body-frame relative subgoal errors."""

    name = "base"

    def act(
        self,
        body_position_error: np.ndarray,
        yaw_error_deg: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        raise NotImplementedError

    def reset(self, env_mask: Optional[object] = None) -> None:
        del env_mask


class ZeroBodyFramePositionControlBackend(BodyFramePositionControlBackend):
    """Deterministic zero-output backend used by focused tests."""

    name = "zero"

    def act(self, body_position_error, yaw_error_deg, *, deterministic: bool = False) -> np.ndarray:
        del yaw_error_deg, deterministic
        error = np.asarray(body_position_error, dtype=np.float32)
        if error.ndim < 1 or error.shape[-1] != 3:
            raise ValueError(f"body_position_error must end in 3, got {error.shape}")
        return np.zeros(error.shape[:-1] + (THRUSTER_ACTION_DIM,), dtype=np.float32)


@dataclass
class BodyFramePIDWrenchControllerConfig:
    """Fixed PID configuration shared with the 1Chase1 pose-yaw hierarchy."""

    rl_project_src: str = "./python/finssim_rl/src"
    pid_params: Tuple[float, ...] = DEFAULT_POSITION_PID_PARAMS
    yaw_pid_params: Tuple[float, float, float] = DEFAULT_YAW_PID_PARAMS
    enable_yaw_control: bool = True
    dt: float = 0.02
    integral_decay: float = 0.95
    yaw_deadband_deg: float = 1.0
    allocator_allocation_mode: str = "physical_wrench_allocator"
    action_dim: int = THRUSTER_ACTION_DIM
    device: str = "cpu"

    @property
    def output_range(self) -> Tuple[float, float]:
        return (-SIM_PHYSICAL_WRENCH_LIMITS_BODY[1], SIM_PHYSICAL_WRENCH_LIMITS_BODY[1])

    @property
    def surge_output_limit(self) -> float:
        return SIM_PHYSICAL_WRENCH_LIMITS_BODY[0]

    @property
    def sway_output_limit(self) -> float:
        return SIM_PHYSICAL_WRENCH_LIMITS_BODY[2]

    @property
    def yaw_output_limit(self) -> float:
        return SIM_PHYSICAL_WRENCH_LIMITS_BODY[4]


class BodyFramePIDWrenchControllerBackend(BodyFramePositionControlBackend):
    """Map local pose errors to normalized thrusters through physical allocation."""

    name = "body_pid_wrench"

    def __init__(self, config: Optional[BodyFramePIDWrenchControllerConfig] = None):
        self.config = config or BodyFramePIDWrenchControllerConfig()
        if self.config.allocator_allocation_mode != "physical_wrench_allocator":
            raise ValueError(
                "body_pid_wrench only supports allocator_allocation_mode="
                "'physical_wrench_allocator'."
            )
        self._rl_project_src = _resolve_rl_project_src(self.config.rl_project_src)
        self.model = self._build_pid_model()

    def _load_pid_classes(self):
        if self._rl_project_src not in sys.path:
            sys.path.insert(0, self._rl_project_src)
        try:
            from finssim_rl.models.pid_controller import PIDController
            from finssim_rl.models.traditional_position_pid import TraditionalPositionPIDModel
        except Exception as exc:
            raise ImportError(
                "body_pid_wrench requires finssim_rl TraditionalPositionPIDModel and PIDController."
            ) from exc
        return PIDController, TraditionalPositionPIDModel

    def _build_pid_model(self):
        PIDController, TraditionalPositionPIDModel = self._load_pid_classes()
        controller = PIDController(
            action_dim=int(self.config.action_dim),
            device=str(self.config.device),
            dt=float(self.config.dt),
            integral_decay=float(self.config.integral_decay),
            output_range=self.config.output_range,
            yaw_deadband_deg=float(self.config.yaw_deadband_deg),
            yaw_output_limit=self.config.yaw_output_limit,
            surge_output_limit=self.config.surge_output_limit,
            sway_output_limit=self.config.sway_output_limit,
            allocator_control_axis_ranges=SIM_PHYSICAL_WRENCH_LIMITS_BODY,
            allocator_allocation_mode="physical_wrench_allocator",
            use_thrust_allocator=True,
            debug_print_interval=0,
        )
        return TraditionalPositionPIDModel(
            controller=controller,
            pid_params=tuple(float(value) for value in self.config.pid_params),
            yaw_pid_params=tuple(float(value) for value in self.config.yaw_pid_params),
            enable_yaw_control=bool(self.config.enable_yaw_control),
            action_dim=int(self.config.action_dim),
            enable_logging=False,
        )

    def reset(self, env_mask: Optional[object] = None) -> None:
        if not hasattr(self.model, "reset"):
            return
        if env_mask is None:
            self.model.reset()
            return

        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        yaw_integral = getattr(self.model, "_yaw_integral_np", None)
        if not isinstance(yaw_integral, np.ndarray):
            self.model.reset()
            return
        if yaw_integral.size != mask.size:
            if mask.size <= 0 or yaw_integral.size % mask.size != 0:
                self.model.reset()
                return
            # The PID batch is flattened from [environment, chaser]. A worker
            # reset mask is per environment, so expand it over all chasers.
            mask = np.repeat(mask, yaw_integral.size // mask.size)
        if yaw_integral.shape != mask.shape:
            self.model.reset()
            return
        controller = getattr(self.model, "controller", None)
        if controller is not None and hasattr(controller, "reset"):
            controller.reset(mask)
        for attr_name in ("_yaw_integral_np", "_yaw_prev_error_np", "_yaw_derivative_filter_np"):
            values = getattr(self.model, attr_name, None)
            if isinstance(values, np.ndarray) and values.shape == mask.shape:
                values[mask] = 0.0

    def act(
        self,
        body_position_error: np.ndarray,
        yaw_error_deg: np.ndarray,
        *,
        deterministic: bool = False,
    ) -> np.ndarray:
        error = np.nan_to_num(
            np.asarray(body_position_error, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        if error.ndim < 1 or error.shape[-1] != 3:
            raise ValueError(f"body_position_error must end in 3, got {error.shape}")
        leading_shape = error.shape[:-1]
        flat_error = error.reshape(-1, 3)

        yaw = np.nan_to_num(
            np.asarray(yaw_error_deg, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        if yaw.shape != leading_shape:
            try:
                yaw = np.broadcast_to(yaw, leading_shape)
            except ValueError as exc:
                raise ValueError(
                    "yaw_error_deg must match body_position_error leading shape; "
                    f"got yaw={yaw.shape}, body_error={error.shape}."
                ) from exc

        with torch.no_grad():
            actions, _ = self.model.predict_from_body_position_error(
                flat_error,
                yaw_error_deg=yaw.reshape(-1),
                deterministic=deterministic,
            )
        return _fit_thruster_actions(actions).reshape(leading_shape + (THRUSTER_ACTION_DIM,))


def make_body_frame_position_control_backend(
    backend_type: str,
    *,
    body_pid_wrench_config: Optional[BodyFramePIDWrenchControllerConfig] = None,
) -> BodyFramePositionControlBackend:
    normalized_type = backend_type.lower()
    if normalized_type in {"zero", "fake"}:
        return ZeroBodyFramePositionControlBackend()
    if normalized_type in {"body_pid_wrench", "traditional_body_pid_wrench"}:
        return BodyFramePIDWrenchControllerBackend(config=body_pid_wrench_config)
    raise ValueError(
        f"Unknown body-frame position-control backend: {backend_type!r}. "
        "Supported backends: body_pid_wrench, zero."
    )


__all__ = [
    "BodyFramePIDWrenchControllerBackend",
    "BodyFramePIDWrenchControllerConfig",
    "BodyFramePositionControlBackend",
    "SIM_PHYSICAL_WRENCH_LIMITS_BODY",
    "THRUSTER_ACTION_DIM",
    "ZeroBodyFramePositionControlBackend",
    "make_body_frame_position_control_backend",
]
