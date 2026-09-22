"""Deterministic 3Chase1 formation baseline using the PID/wrench backend.

The high-level guidance is deliberately local: it consumes the same 30D
controller-body observation as MAPPO and produces the same bounded body
subgoal interface.  It does not use Unity world poses or Unity's historical
direct-thruster mixer.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional, Tuple

import numpy as np

from .networks.actors import ROLE_MAPPING
from .position_control_backends import (
    BodyFramePIDWrenchControllerConfig,
    make_body_frame_position_control_backend,
)


CHASER_ACTOR_OBS_DIM = 30
THRUSTER_ACTION_DIM = 8
CHASE_PID_PARAMS: Tuple[float, ...] = (8.0, 0.05, 0.20, 8.0, 0.05, 0.20, 8.0, 0.05, 0.20)


@dataclass
class TraditionalFormationBaselineConfig:
    """Guidance and low-level settings for the static 3Chase1 baseline."""

    target_body_delta_limits: Tuple[float, float, float] = (1.5, 0.5, 1.5)
    yaw_error_limit_deg: float = 90.0
    desired_netter_spacing: float = 6.0
    netter_forward_offset: float = 2.0
    herder_behind_offset: float = 3.0
    heading_gate_deg: float = 15.0
    debug_print_interval: int = 20
    pid_params: Tuple[float, ...] = CHASE_PID_PARAMS
    controller_backend: str = "body_pid_wrench"
    body_pid_wrench_controller: BodyFramePIDWrenchControllerConfig = field(
        default_factory=BodyFramePIDWrenchControllerConfig
    )


class TraditionalFormationPIDWrenchBaseline:
    """Herder/netter geometric guidance plus shared PID+wrench allocation."""

    action_interface = "position_controller"

    def __init__(
        self,
        *,
        n_agents: int,
        role_ids: np.ndarray,
        config: Optional[TraditionalFormationBaselineConfig] = None,
    ) -> None:
        self.n_agents = int(n_agents)
        self.role_ids = np.asarray(role_ids, dtype=np.int64).reshape(-1)
        if self.role_ids.shape != (self.n_agents,):
            raise ValueError(
                f"role_ids must have shape ({self.n_agents},), got {self.role_ids.shape}."
            )
        self.config = config or TraditionalFormationBaselineConfig()
        self.chaser_indices = np.flatnonzero(self.role_ids != ROLE_MAPPING["Prey"])
        if self.chaser_indices.size != 3:
            raise ValueError(
                "TraditionalFormationPIDWrenchBaseline requires exactly three chasers, "
                f"got indices={self.chaser_indices.tolist()}."
            )
        pid_config = replace(
            self.config.body_pid_wrench_controller,
            pid_params=tuple(float(value) for value in self.config.pid_params),
        )
        self.controller = make_body_frame_position_control_backend(
            self.config.controller_backend,
            body_pid_wrench_config=pid_config,
        )
        self.last_diagnostics: dict[str, np.ndarray | float] = {}
        self._decision_count = 0

    @staticmethod
    def _normalized_horizontal(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
        horizontal = np.asarray(vector, dtype=np.float32).copy()
        horizontal[1] = 0.0
        norm = float(np.linalg.norm(horizontal))
        if norm <= 1e-5:
            return np.asarray(fallback, dtype=np.float32)
        return horizontal / norm

    @staticmethod
    def _teammates(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return obs[14:16], obs[16:19], obs[22:24], obs[24:27]

    def _drive_axis(
        self,
        prey_relative_position: np.ndarray,
        netter_midpoint_relative: np.ndarray,
        fallback: np.ndarray,
    ) -> np.ndarray:
        return self._normalized_horizontal(
            netter_midpoint_relative - prey_relative_position,
            fallback,
        )

    def _herder_target_error(self, obs: np.ndarray) -> np.ndarray:
        prey = obs[6:9]
        _role_a, pos_a, _role_b, pos_b = self._teammates(obs)
        netter_midpoint = 0.5 * (pos_a + pos_b)
        drive_axis = self._drive_axis(prey, netter_midpoint, fallback=np.array([1.0, 0.0, 0.0]))
        return prey - drive_axis * float(self.config.herder_behind_offset)

    def _netter_target_error(self, obs: np.ndarray, netter_ordinal: int) -> np.ndarray:
        prey = obs[6:9]
        role_a, pos_a, role_b, pos_b = self._teammates(obs)
        a_is_herder = bool(role_a[0] > role_a[1])
        herder_position = pos_a if a_is_herder else pos_b
        other_netter_position = pos_b if a_is_herder else pos_a
        netter_midpoint = 0.5 * other_netter_position
        drive_axis = self._drive_axis(prey, netter_midpoint, fallback=prey - herder_position)
        lateral_axis = self._normalized_horizontal(
            np.cross(np.array([0.0, 1.0, 0.0], dtype=np.float32), drive_axis),
            fallback=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        )

        current_side = float(np.dot(-netter_midpoint, lateral_axis))
        if abs(current_side) > 1e-3:
            lateral_sign = float(np.sign(current_side))
        else:
            lateral_sign = -1.0 if netter_ordinal == 0 else 1.0

        return (
            prey
            + drive_axis * float(self.config.netter_forward_offset)
            + lateral_axis * (lateral_sign * 0.5 * float(self.config.desired_netter_spacing))
        )

    def _build_subgoals(self, obs: np.ndarray, role_ids: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        batch_size = obs.shape[0]
        body_errors = np.zeros((batch_size, self.chaser_indices.size, 3), dtype=np.float32)
        yaw_errors_deg = np.zeros((batch_size, self.chaser_indices.size), dtype=np.float32)

        for batch_index in range(batch_size):
            netter_ordinal = 0
            for chaser_slot, agent_index in enumerate(self.chaser_indices):
                chaser_obs = obs[batch_index, agent_index, :CHASER_ACTOR_OBS_DIM]
                role_id = int(role_ids[batch_index, agent_index])
                if role_id == ROLE_MAPPING["Herder"]:
                    target_error = self._herder_target_error(chaser_obs)
                elif role_id == ROLE_MAPPING["Netter"]:
                    target_error = self._netter_target_error(chaser_obs, netter_ordinal)
                    netter_ordinal += 1
                else:
                    raise ValueError(f"Unexpected chaser role_id={role_id}.")

                bearing_deg = -np.degrees(np.arctan2(target_error[2], target_error[0]))
                body_errors[batch_index, chaser_slot] = target_error
                yaw_errors_deg[batch_index, chaser_slot] = bearing_deg

        body_limits = np.asarray(self.config.target_body_delta_limits, dtype=np.float32)
        body_errors = np.clip(body_errors, -body_limits, body_limits)
        yaw_errors_deg = np.clip(
            yaw_errors_deg,
            -float(self.config.yaw_error_limit_deg),
            float(self.config.yaw_error_limit_deg),
        )

        # Match the 1Chase1 wrench baseline's heading-first pursuit contract.
        # This controller must not combine a lateral Fz command with yaw while
        # the target formation point is side-on: that makes a FinsROV slide
        # toward the point instead of turning its nose toward it.  Once the
        # bearing is inside the gate, advance with surge only.  Vertical
        # control remains independent throughout the turn.
        heading_aligned = np.abs(yaw_errors_deg) <= float(self.config.heading_gate_deg)
        body_errors[..., 2] = 0.0
        body_errors[..., 0] = np.where(
            heading_aligned,
            np.maximum(body_errors[..., 0], 0.0),
            0.0,
        )
        return body_errors, yaw_errors_deg

    def select_action(self, obs, role_ids, deterministic: bool = True):
        del deterministic
        obs_np = np.asarray(obs, dtype=np.float32)
        current_role_ids = np.asarray(role_ids, dtype=np.int64)
        if obs_np.ndim != 3 or obs_np.shape[1] != self.n_agents or obs_np.shape[2] < CHASER_ACTOR_OBS_DIM:
            raise ValueError(
                "Expected observation shape [batch, n_agents, >=30], got "
                f"{obs_np.shape}."
            )
        if current_role_ids.shape != obs_np.shape[:2]:
            raise ValueError(
                f"role_ids must have shape {obs_np.shape[:2]}, got {current_role_ids.shape}."
            )

        body_errors, yaw_errors_deg = self._build_subgoals(obs_np, current_role_ids)
        chaser_actions = self.controller.act(body_errors, yaw_errors_deg, deterministic=True)
        env_actions = np.zeros((obs_np.shape[0], self.n_agents, THRUSTER_ACTION_DIM), dtype=np.float32)
        env_actions[:, self.chaser_indices, :] = chaser_actions
        self.last_diagnostics = {
            "body_position_error_m": body_errors.copy(),
            "yaw_error_deg": yaw_errors_deg.copy(),
            "heading_aligned": (
                np.abs(yaw_errors_deg) <= float(self.config.heading_gate_deg)
            ).copy(),
            "thruster_action": chaser_actions.copy(),
        }
        self._decision_count += 1
        if (
            self.config.debug_print_interval > 0
            and self._decision_count % int(self.config.debug_print_interval) == 0
        ):
            self._print_trace(obs_np, current_role_ids, body_errors, yaw_errors_deg, chaser_actions)
        return env_actions, None, None, {"buffer_actions": None}

    def _print_trace(
        self,
        obs: np.ndarray,
        role_ids: np.ndarray,
        body_errors: np.ndarray,
        yaw_errors_deg: np.ndarray,
        thruster_actions: np.ndarray,
    ) -> None:
        """Emit a compact, values-only trace for runtime coordinate audits."""
        labels = {ROLE_MAPPING["Herder"]: "herder", ROLE_MAPPING["Netter"]: "netter"}
        parts: list[str] = []
        for slot, agent_index in enumerate(self.chaser_indices):
            prey = obs[0, agent_index, 6:9]
            target = body_errors[0, slot]
            horizontal = thruster_actions[0, slot, 4:8]
            label = labels.get(int(role_ids[0, agent_index]), f"role{int(role_ids[0, agent_index])}")
            parts.append(
                f"agent={agent_index}:{label} prey={np.array2string(prey, precision=2)} "
                f"target={np.array2string(target, precision=2)} yaw_deg={yaw_errors_deg[0, slot]:+.1f} "
                f"H={np.array2string(horizontal, precision=2)}"
            )
        print(f"[TraditionalFormationBaseline step={self._decision_count}] " + " | ".join(parts), flush=True)

    def reset_controller_state(self, env_mask=None) -> None:
        self.controller.reset(env_mask)

    def eval(self) -> None:
        return None


__all__ = [
    "TraditionalFormationBaselineConfig",
    "TraditionalFormationPIDWrenchBaseline",
]
