"""Local-observation baseline for passive-target triangular-net transport."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Tuple

import numpy as np

from .position_control_backends import (
    BodyFramePIDWrenchControllerConfig,
    make_body_frame_position_control_backend,
)


TRINET_OBS_DIM = 34
THRUSTER_ACTION_DIM = 8

PHASE_FIXED_INITIAL_FORMATION = 0
PHASE_FORM_SYMMETRIC = 1
PHASE_RAISE_NET = 2
PHASE_TOW_TO_GOAL = 3
PHASE_FINAL_ASCENT = 4
PHASE_HOLD_STILL = 5


@dataclass
class TriNetTransportStateMachineBaselineConfig:
    target_body_delta_limits: Tuple[float, float, float] = (1.0, 1.2, 1.0)
    # Keep the triangular task's common-heading policy, but use the same
    # physical wrench magnitude as the 1Chase1 wrench baseline.
    yaw_error_limit_deg: float = 35.0
    # A triangular net is transported by a shared 3D translation. Keeping a
    # common heading avoids three independently rotating vehicle frames from
    # shearing that translation into opposing thrust commands.
    use_heading_control: bool = False
    shared_heading_control: bool = True
    hold_goal_distance_m: float = 0.08
    heading_gate_deg: float = 20.0
    desired_rov_spacing_m: float = 0.85
    formation_gain: float = 0.20
    formation_velocity_gain: float = 0.0
    max_formation_correction_m: float = 0.12
    net_surface_side_length_m: float = 0.7287565
    # Target must lie at an interior point of the YZ triangular footprint, not
    # at the ROV-triangle centroid. In TriNetCapture's fixed spawn this keeps
    # the upper ROV below y=0 while the Target is surrounded by the net.
    target_in_net_plane_body_yz: Tuple[float, float] = (0.35, 0.0)
    # Unity action/observation order is [FinsROV_Fossen_Left,
    # FinsROV_Fossen_Top, FinsROV_Fossen_Right]. With Target fixed at
    # (1.20, -0.40, 0.0), these are the currently configured first-stage
    # target-relative points. Edit this triplet to change Left/Top/Right.
    fixed_initial_target_offsets: Tuple[Tuple[float, float, float], ...] = (
        (0.4, -0.4, -0.50),
        (0.4, -0.4, 0.0),
        (0.4, -0.4, 0.50),
    )
    # Once the first-stage formation is settled, the whole team rises to
    # scoop Target with the net. Top rises slightly farther to tilt the net
    # into a carrying shape.
    net_lift_height_m: float = 0.2
    top_extra_lift_height_m: float = 0.5
    # Phase 3/4 uses its own target-relative formation at RecoveryGoal. This
    # is intentionally independent of fixed_initial_target_offsets so the
    # final towing pose can be tuned without moving the approach waypoints.
    # Order: [FinsROV_Fossen_Left, FinsROV_Fossen_Top, FinsROV_Fossen_Right].
    tow_goal_target_offsets: Tuple[Tuple[float, float, float], ...] = (
        (-0.15, -0.2, -0.50),
        (-0.13, 0.3, 0.0),
        (-0.15, -0.2, 0.50),
    )
    # Phase 4 starts once all three ROVs reach their Phase-3 Goal + offset
    # waypoints. They then rise together by this distance from their own
    # transition heights.
    final_lift_height_m: float = 0.5
    final_lift_tolerance_m: float = 0.05
    # Phase 3 -> 4 is a per-ROV waypoint check at Goal +
    # tow_goal_target_offsets, not a Target-to-Goal distance check.
    tow_goal_waypoint_tolerance_m: float = 0.20
    # Phase 1 must run for at least 3 s (30 decisions at 0.10 s). Afterwards
    # each ROV must be within this 3D distance of its own requested point.
    first_phase_waypoint_tolerance_m: float = 0.20
    first_phase_settle_steps: int = 30
    # Phase 2 ends only after every ROV reaches its own recorded-height lift
    # target. This is a physical pool-local Y check, not a timer.
    lift_height_tolerance_m: float = 0.05
    # Stop the intact net this far in front of Target before shaping it.  The
    # final raise-net phase then advances the net plane over Target.
    approach_standoff_m: float = 0.45
    approach_tolerance_m: float = 0.10
    formation_tolerance_m: float = 0.08
    formation_settle_steps: int = 5
    approach_gain: float = 0.70
    capture_alignment_gain: float = 0.40
    target_net_distance_m: float = 0.10
    target_edge_margin_m: float = 0.06
    # ML-Agents requests an action every five 0.02 s physics steps in this
    # scene. The controller state therefore advances at 0.10 s, not 0.02 s.
    control_dt_seconds: float = 0.10
    # [depth, surge, sway], [Kp, Ki, Kd].  Kp=8 produces the same 8 N
    # unit-position-error wrench as 1Chase1's physical wrench PD path.
    # TriNet does not use the 1Chase1 heading gate because the Left/Top/Right
    # vertices must translate holonomically to different lateral setpoints.
    pid_params: Tuple[float, ...] = (8.0, 0.0, 0.0, 8.0, 0.0, 0.0, 8.0, 0.0, 0.0)
    yaw_pid_params: Tuple[float, float, float] = (0.12, 0.0, 0.01)
    # Same normalized action domain as 1Chase1 after physical allocation.
    max_thruster_action: float = 1.0
    # Eval owns user-facing progress/reward reporting. Keep the per-decision
    # PID/state-machine trace disabled unless a developer explicitly enables
    # it on a manually constructed baseline config.
    debug_print_interval: int = 0
    controller_backend: str = "body_pid_wrench"
    body_pid_wrench_controller: BodyFramePIDWrenchControllerConfig = field(
        default_factory=BodyFramePIDWrenchControllerConfig
    )


class TriNetTransportStateMachineBaseline:
    """Three-stage triangular-net controller: form, lift, then tow."""

    action_interface = "position_controller"

    def __init__(self, *, n_agents: int, role_ids: np.ndarray, config: TriNetTransportStateMachineBaselineConfig):
        self.n_agents = int(n_agents)
        self.role_ids = np.asarray(role_ids, dtype=np.int64).reshape(-1)
        if self.n_agents != 3 or self.role_ids.shape != (3,) or not np.all(self.role_ids == self.role_ids[0]):
            raise ValueError("TriNetTransportStateMachineBaseline requires exactly three homogeneous ROVs.")
        self.config = config
        if len(config.fixed_initial_target_offsets) != self.n_agents:
            raise ValueError("fixed_initial_target_offsets must provide one point for each of the three ROVs.")
        if len(config.tow_goal_target_offsets) != self.n_agents:
            raise ValueError("tow_goal_target_offsets must provide one point for each of the three ROVs.")
        pid_config = replace(
            config.body_pid_wrench_controller,
            pid_params=tuple(float(value) for value in config.pid_params),
            yaw_pid_params=tuple(float(value) for value in config.yaw_pid_params),
            dt=float(config.control_dt_seconds),
        )
        self.controller = make_body_frame_position_control_backend(
            config.controller_backend,
            body_pid_wrench_config=pid_config,
        )
        self._decision_count = 0
        self._phase_by_env: np.ndarray | None = None
        self._formation_settle_counts: np.ndarray | None = None
        # Pool-local Y sampled at the Phase-1 -> Phase-2 transition.  Phase 2
        # uses this fixed origin to command an actual relative lift distance.
        self._lift_start_y_m: np.ndarray | None = None
        # Pool-local Y sampled at the Phase-3 -> Phase-4 transition. The final
        # ascent is therefore relative to each ROV's actual towing height.
        self._final_lift_start_y_m: np.ndarray | None = None
        self.last_diagnostics: dict[str, np.ndarray] = {}

    @staticmethod
    def _net_center_relative(obs: np.ndarray) -> np.ndarray:
        # self is the third vertex at the origin; homogeneous teammate blocks
        # expose their relative positions at 14:17 and 20:23.
        # ``obs`` can be either one [features] observation or a batch
        # [environments, features] while phase readiness is evaluated.
        return (obs[..., 14:17] + obs[..., 20:23]) / 3.0

    def _formation_correction(self, obs: np.ndarray) -> np.ndarray:
        """Return a symmetric local correction for the two incident edges."""
        correction = np.zeros(3, dtype=np.float32)
        for position, velocity in ((obs[14:17], obs[17:20]), (obs[20:23], obs[23:26])):
            distance = float(np.linalg.norm(position))
            if distance <= 1e-5:
                continue
            direction = position / distance
            # For a stretched edge follow the teammate; for a compressed edge
            # move away. Summing this same pairwise law at all vertices keeps
            # the net centroid unchanged in a common world frame.
            correction += self.config.formation_gain * (distance - self.config.desired_rov_spacing_m) * direction
            correction += self.config.formation_velocity_gain * velocity
        magnitude = float(np.linalg.norm(correction))
        if magnitude > self.config.max_formation_correction_m:
            correction *= self.config.max_formation_correction_m / magnitude
        return correction

    def _target_relative_to_net_center(self) -> np.ndarray:
        """Desired Target position relative to the ROV-triangle centroid."""
        return np.asarray(
            (0.0, *self.config.target_in_net_plane_body_yz),
            dtype=np.float32,
        )

    def _target_offsets_for_phase(self, phase: int) -> np.ndarray:
        """Return Left/Top/Right offsets in the shared task frame."""
        if phase >= PHASE_TOW_TO_GOAL:
            return np.asarray(self.config.tow_goal_target_offsets, dtype=np.float32).copy()

        offsets = np.asarray(self.config.fixed_initial_target_offsets, dtype=np.float32).copy()
        if phase >= PHASE_RAISE_NET:
            offsets[:, 1] += self.config.net_lift_height_m
            offsets[1, 1] += self.config.top_extra_lift_height_m
        return offsets

    @staticmethod
    def _task_offset_to_body(offset_task: np.ndarray, goal_from_target_body: np.ndarray) -> np.ndarray:
        """Express a task-frame formation offset in one ROV's body frame.

        The task +X axis is opposite Target->Goal (the initial approach side),
        +Y is vertical, and +Z completes the right-handed horizontal frame.
        Target and Goal are both observed in body coordinates, so this frame is
        common to all ROVs even when the net yaws one vehicle.
        """
        toward_goal = np.asarray(goal_from_target_body, dtype=np.float32).copy()
        toward_goal[1] = 0.0
        magnitude = float(np.linalg.norm(toward_goal))
        task_x_body = -toward_goal / magnitude if magnitude > 1e-5 else np.asarray((1.0, 0.0, 0.0), dtype=np.float32)
        task_y_body = np.asarray((0.0, 1.0, 0.0), dtype=np.float32)
        task_z_body = np.cross(task_x_body, task_y_body)
        return (
            offset_task[0] * task_x_body
            + offset_task[1] * task_y_body
            + offset_task[2] * task_z_body
        ).astype(np.float32)

    def _phase_offsets_in_body(self, values: np.ndarray, phase: int) -> np.ndarray:
        """Return [environment, agent, xyz] formation offsets in body axes."""
        offsets_task = self._target_offsets_for_phase(phase)
        if phase >= PHASE_TOW_TO_GOAL:
            # Phase 3 is a fixed recovery formation: each endpoint is the
            # RecoveryGoal guidance point plus its configured Left/Top/Right
            # offset in the common initial controller frame.  In particular,
            # do not rebuild this frame from Goal-Target: Target crossing the
            # Goal would flip the formation and prevent the Phase 3 -> 4
            # waypoint condition from ever becoming true.
            return np.broadcast_to(offsets_task, values[:, :, 6:9].shape).copy()
        offsets_body = np.zeros_like(values[:, :, 6:9])
        for agent in range(self.n_agents):
            for batch in range(values.shape[0]):
                offsets_body[batch, agent] = self._task_offset_to_body(
                    offsets_task[agent],
                    values[batch, agent, 26:29] - values[batch, agent, 6:9],
                )
        return offsets_body

    def _approach_target_relative_to_net_center(self, goal_from_target: np.ndarray) -> np.ndarray:
        """Return the target's desired location while the team stages nearby.

        ``goal_from_target`` points from Target to the recovery area.  The
        approach staging plane is therefore placed on the opposite (capture)
        side of Target, so the subsequent raise-net command travels from the
        team toward Target instead of beginning the tow prematurely.
        """
        anchor = self._target_relative_to_net_center()
        direction = -np.asarray(goal_from_target, dtype=np.float32)
        magnitude = float(np.linalg.norm(direction))
        if magnitude > 1e-5:
            anchor += direction * (self.config.approach_standoff_m / magnitude)
        return anchor

    def _subgoal(
        self, obs: np.ndarray, phase: int, agent_index: int, lift_error_m: float = 0.0
    ) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, np.ndarray]:
        target = obs[6:9]
        goal = obs[26:29]
        goal_from_target = goal - target
        offset_task = self._target_offsets_for_phase(phase)[agent_index]
        offset_body = (
            offset_task.copy()
            if phase >= PHASE_TOW_TO_GOAL
            else self._task_offset_to_body(offset_task, goal_from_target)
        )
        if phase == PHASE_FIXED_INITIAL_FORMATION:
            # Target-relative points map directly to the three requested
            # first-stage world positions.
            waypoint = target + offset_body
            translation = waypoint.copy()
        elif phase == PHASE_RAISE_NET or phase == PHASE_FINAL_ASCENT:
            # Pure own-frame lift: do not use Target or Goal here.  Both
            # contain horizontal motion from the deformable net and would
            # turn this phase into an unintended retreat.
            waypoint = np.asarray((0.0, lift_error_m, 0.0), dtype=np.float32)
            translation = waypoint.copy()
        elif phase == PHASE_HOLD_STILL:
            # Freeze at the completed recovery pose: retain the Phase-3
            # horizontal Goal + offset waypoint and the Phase-4 recorded
            # final height.  This is a position hold, not another ascent or
            # a new towing command.
            waypoint = goal + offset_body
            waypoint[1] = lift_error_m
            translation = waypoint.copy()
        else:
            # Place the raised Left/Top/Right formation at the recovery Goal,
            # not at an intermediate fraction of Goal-Target.  This gives the
            # net a full retreat vector to drag Target into the goal volume.
            waypoint = goal + offset_body
            translation = waypoint.copy()
        formation = np.zeros(3, dtype=np.float32)
        desired = translation + formation
        formation_waypoint = waypoint + formation
        return (
            desired.astype(np.float32),
            # This scripted transport task translates a net and never
            # commands a heading change. Keep yaw at zero for every phase.
            0.0,
            formation,
            translation,
            formation_waypoint.astype(np.float32),
        )

    def _ensure_state(self, batch_size: int) -> None:
        if self._phase_by_env is None or self._phase_by_env.shape != (batch_size,):
            self._phase_by_env = np.full(batch_size, PHASE_FIXED_INITIAL_FORMATION, dtype=np.int64)
            self._formation_settle_counts = np.zeros(batch_size, dtype=np.int32)
            self._lift_start_y_m = np.zeros((batch_size, self.n_agents), dtype=np.float32)
            self._final_lift_start_y_m = np.zeros((batch_size, self.n_agents), dtype=np.float32)

    def _team_phases(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Advance only after all three ROVs settle at each shared stage."""
        self._ensure_state(values.shape[0])
        assert self._phase_by_env is not None
        assert self._formation_settle_counts is not None
        assert self._lift_start_y_m is not None
        assert self._final_lift_start_y_m is not None
        initial_offsets_body = self._phase_offsets_in_body(values, PHASE_FIXED_INITIAL_FORMATION)
        initial_waypoint_error = values[:, :, 6:9] + initial_offsets_body
        initial_ready = np.max(np.linalg.norm(initial_waypoint_error, axis=2), axis=1) < (
            self.config.first_phase_waypoint_tolerance_m
        )
        lift_targets = np.full(self.n_agents, self.config.net_lift_height_m, dtype=np.float32)
        lift_targets[1] += self.config.top_extra_lift_height_m
        lift_ready = np.all(
            values[:, :, 12] >= self._lift_start_y_m + lift_targets[None, :] - self.config.lift_height_tolerance_m,
            axis=1,
        )
        final_lift_ready = np.all(
            values[:, :, 12] >= self._final_lift_start_y_m + self.config.final_lift_height_m - self.config.final_lift_tolerance_m,
            axis=1,
        )
        tow_offsets_body = self._phase_offsets_in_body(values, PHASE_TOW_TO_GOAL)
        # In each ROV's body observation, Goal is at 26:29 and the configured
        # final formation offset is already expressed in that same body frame.
        # This directly measures each ROV's distance to its own Goal + offset.
        tow_waypoint_error = values[:, :, 26:29] + tow_offsets_body
        tow_ready = np.max(np.linalg.norm(tow_waypoint_error, axis=2), axis=1) < (
            self.config.tow_goal_waypoint_tolerance_m
        )

        in_initial = self._phase_by_env == PHASE_FIXED_INITIAL_FORMATION
        in_lift = self._phase_by_env == PHASE_RAISE_NET
        in_tow = self._phase_by_env == PHASE_TOW_TO_GOAL
        in_final_ascent = self._phase_by_env == PHASE_FINAL_ASCENT
        # This is the elapsed Phase-1 duration, not a consecutive-centering
        # counter. Once the mandatory 3 s has elapsed, the current per-ROV
        # waypoint check decides whether it may enter the lift phase.
        self._formation_settle_counts[in_initial] += 1
        initial_advance = (
            in_initial
            & initial_ready
            & (self._formation_settle_counts >= self.config.first_phase_settle_steps)
        )
        lift_advance = in_lift & lift_ready
        tow_advance = in_tow & tow_ready
        final_ascent_advance = in_final_ascent & final_lift_ready
        self._phase_by_env[initial_advance] = PHASE_RAISE_NET
        self._phase_by_env[lift_advance] = PHASE_TOW_TO_GOAL
        self._phase_by_env[tow_advance] = PHASE_FINAL_ASCENT
        self._phase_by_env[final_ascent_advance] = PHASE_HOLD_STILL
        self._formation_settle_counts[initial_advance | lift_advance | tow_advance | final_ascent_advance] = 0
        # Observation feature 12 is each ROV's pool-local Y. Snapshot it once
        # at the phase transition, rather than estimating displacement by
        # integrating a velocity affected by net tension.
        self._lift_start_y_m[initial_advance] = values[initial_advance, :, 12]
        self._final_lift_start_y_m[tow_advance] = values[tow_advance, :, 12]
        return (
            self._phase_by_env.copy(),
            initial_ready,
            lift_ready,
            tow_ready,
        )

    def select_action(self, obs, role_ids, deterministic: bool = True):
        del deterministic, role_ids
        values = np.asarray(obs, dtype=np.float32)
        if values.ndim != 3 or values.shape[1:] != (3, values.shape[2]) or values.shape[2] < TRINET_OBS_DIM:
            raise ValueError(f"Expected [batch, 3, >=33] TriNet observations, got {values.shape}.")
        body = np.zeros((values.shape[0], 3, 3), dtype=np.float32)
        yaw = np.zeros((values.shape[0], 3), dtype=np.float32)
        phase_by_env, approach_ready, formation_symmetric, net_held = self._team_phases(values)
        assert self._lift_start_y_m is not None
        assert self._final_lift_start_y_m is not None
        lift_targets = np.full(self.n_agents, self.config.net_lift_height_m, dtype=np.float32)
        lift_targets[1] += self.config.top_extra_lift_height_m
        lift_error = self._lift_start_y_m + lift_targets[None, :] - values[:, :, 12]
        final_lift_error = (
            self._final_lift_start_y_m + self.config.final_lift_height_m - values[:, :, 12]
        )
        tow_goal_waypoint_error = values[:, :, 26:29] + self._phase_offsets_in_body(
            values, PHASE_TOW_TO_GOAL
        )
        phases = np.repeat(phase_by_env[:, None], 3, axis=1)
        formation = np.zeros_like(body)
        translation = np.zeros_like(body)
        formation_waypoint = np.zeros_like(body)
        for batch in range(values.shape[0]):
            for agent in range(3):
                (
                    body[batch, agent],
                    yaw[batch, agent],
                    formation[batch, agent],
                    translation[batch, agent],
                    formation_waypoint[batch, agent],
                ) = self._subgoal(
                    values[batch, agent],
                    int(phase_by_env[batch]),
                    agent,
                    float(
                        final_lift_error[batch, agent]
                        if phase_by_env[batch] in (PHASE_FINAL_ASCENT, PHASE_HOLD_STILL)
                        else lift_error[batch, agent]
                    ),
                )
        limits = np.asarray(self.config.target_body_delta_limits, dtype=np.float32)
        # Top's lift is additive. Do not silently clip its requested
        # ``net_lift_height_m + top_extra_lift_height_m`` to a smaller generic
        # position-error limit before it reaches the PID/wrench controller.
        if np.any(phase_by_env == PHASE_RAISE_NET):
            limits[1] = max(
                limits[1],
                float(self.config.net_lift_height_m + self.config.top_extra_lift_height_m),
            )
        if np.any((phase_by_env == PHASE_FINAL_ASCENT) | (phase_by_env == PHASE_HOLD_STILL)):
            limits[1] = max(limits[1], float(self.config.final_lift_height_m))
        body = np.clip(body, -limits, limits)
        # Observation slot 13 is the signed bearing of pool +X in the
        # controller frame. Controller-body +Z is left, while positive yaw
        # wrench turns the FinsROV right, so the feedback sign must be
        # inverted. This remains the same yaw=0 hold in every phase; Phase 3
        # backs up through reversible surge/sway and never commands a turn.
        yaw[:] = -values[:, :, 13]
        yaw = np.clip(yaw, -self.config.yaw_error_limit_deg, self.config.yaw_error_limit_deg)
        thrusters = self.controller.act(body, yaw, deterministic=True)
        # The ROVs are coupled by a deformable net. This baseline intentionally
        # trades speed for bounded formation transients; the RL task remains
        # free to use the full [-1, 1] action range.
        thrusters = np.clip(
            thrusters,
            -float(self.config.max_thruster_action),
            float(self.config.max_thruster_action),
        ).astype(np.float32, copy=False)
        self.last_diagnostics = {
            "body_position_error_m": body,
            "shared_translation_error_m": translation,
            "formation_waypoint_error_m": formation_waypoint,
            "formation_correction_m": formation,
            "yaw_error_deg": yaw,
            "phase": phases,
            "approach_ready": approach_ready,
            "formation_symmetric": formation_symmetric,
            "net_held": net_held,
            "lift_start_y_m": self._lift_start_y_m.copy(),
            "lift_error_m": lift_error.copy(),
            "final_lift_start_y_m": self._final_lift_start_y_m.copy(),
            "final_lift_error_m": final_lift_error.copy(),
            "tow_goal_waypoint_error_m": tow_goal_waypoint_error.copy(),
            "thruster_action": thrusters,
        }
        self._decision_count += 1
        if self.config.debug_print_interval > 0 and self._decision_count % self.config.debug_print_interval == 0:
            center_target = np.linalg.norm(
                values[:, :, 6:9] - np.stack(
                    [self._net_center_relative(values[batch, agent]) for batch in range(values.shape[0]) for agent in range(3)]
                ).reshape(values.shape[0], 3, 3),
                axis=-1,
            ).mean(axis=1)
            target_goal = np.linalg.norm(values[:, :, 26:29] - values[:, :, 6:9], axis=-1).mean(axis=1)
            print(
                "[TriNetTransportBaseline] "
                f"step={self._decision_count} phase={phase_by_env.tolist()} "
                f"approach_ready={approach_ready.tolist()} symmetric={formation_symmetric.tolist()} net_held={net_held.tolist()} "
                f"net_target_m={np.round(center_target, 3).tolist()} "
                f"net_target0_body={np.round(values[:, 0, 6:9] - np.stack([self._net_center_relative(row) for row in values[:, 0]]), 3).tolist()} "
                f"target_goal_m={np.round(target_goal, 3).tolist()} "
                f"formation_m={np.round(np.linalg.norm(formation, axis=-1).mean(axis=1), 3).tolist()} "
                f"edge_m={np.round(self._mean_edge_lengths(values), 3).tolist()} "
                f"surface_m={np.round(np.abs(values[:, :, 32]).mean(axis=1) * self.config.net_surface_side_length_m, 3).tolist()} "
                f"margin_m={np.round(values[:, :, 33].mean(axis=1) * self.config.net_surface_side_length_m, 3).tolist()} "
                f"formation_waypoint0={np.round(formation_waypoint[:, 0], 3).tolist()} "
                f"controller_delta0={np.round(translation[:, 0], 3).tolist()} "
                f"body0={np.round(body[:, 0], 3).tolist()} "
                f"yaw_deg={np.round(yaw, 1).tolist()} "
                f"thruster_abs={np.round(np.abs(thrusters).mean(axis=(1, 2)), 3).tolist()}",
                flush=True,
            )
        return thrusters, None, None, {"buffer_actions": None}

    def reset_controller_state(self, env_mask=None) -> None:
        self.controller.reset(env_mask)
        if (
            self._phase_by_env is None
            or self._formation_settle_counts is None
            or self._lift_start_y_m is None
            or self._final_lift_start_y_m is None
        ):
            return
        if env_mask is None:
            self._phase_by_env.fill(PHASE_FIXED_INITIAL_FORMATION)
            self._formation_settle_counts.fill(0)
            self._lift_start_y_m.fill(0.0)
            self._final_lift_start_y_m.fill(0.0)
            return
        mask = np.asarray(env_mask, dtype=bool).reshape(-1)
        if mask.shape != self._phase_by_env.shape:
            raise ValueError(f"Expected reset mask {self._phase_by_env.shape}, got {mask.shape}.")
        self._phase_by_env[mask] = PHASE_FIXED_INITIAL_FORMATION
        self._formation_settle_counts[mask] = 0
        self._lift_start_y_m[mask] = 0.0
        self._final_lift_start_y_m[mask] = 0.0

    @staticmethod
    def _mean_edge_lengths(values: np.ndarray) -> np.ndarray:
        """Return one symmetric mean ROV-edge length per environment."""
        teammate_a = np.linalg.norm(values[:, :, 14:17], axis=-1)
        teammate_b = np.linalg.norm(values[:, :, 20:23], axis=-1)
        return 0.5 * (teammate_a.mean(axis=1) + teammate_b.mean(axis=1))

    def eval(self) -> None:
        return None


__all__ = ["TriNetTransportStateMachineBaselineConfig", "TriNetTransportStateMachineBaseline"]
