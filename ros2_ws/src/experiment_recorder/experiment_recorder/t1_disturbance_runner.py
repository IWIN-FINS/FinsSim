"""Dedicated E8 hardware runner for manual T1 disturbance-recovery trials.

This command is intentionally separate from :mod:`runner`: E6/E7 static T1
uses eight target points and a fixed 40--60 s steady-state window, whereas E8
uses exactly one centre target and reports state-observed recovery after a
manual perturbation.  The two experiment families share only the safe ROS2
control and recording infrastructure.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import rclpy
import yaml

from .manifest import file_sha256, git_revision, now_iso, read_json, write_json
from .runner import (
    _ManagedProcess,
    _RosExperimentNode,
    _as_mapping,
    _config_files,
    _controller_checkpoint,
    _controller_storage_subdirectory,
    _infer_controller_name,
    _launch_commands,
    _new_run_id,
    _record_command,
    _repo_root,
    _require_process_running,
    _resolve_cli_path,
    _resolve_path,
    _session_strategy_slug,
    _strategy_name_from_config,
    _wait_for_topics,
    init_experiment_ros,
)
from .t1_disturbance import DisturbanceDetector, T1State, make_state, state_within_thresholds


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"E8 disturbance YAML not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid E8 disturbance YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"E8 disturbance YAML must contain a mapping: {path}")
    for key in ("runner", "target", "controllers", "disturbance_protocol"):
        if not isinstance(payload.get(key), dict):
            raise SystemExit(f"E8 disturbance YAML requires mapping `{key}`")
    return payload


def _target_from_config(config: dict[str, Any]) -> tuple[float, float, float, float]:
    target = _as_mapping(config["target"], "target")
    try:
        value = (
            float(target["x_m"]),
            float(target["y_depth_m"]),
            float(target["z_m"]),
            float(target["yaw_deg"]),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("target requires x_m, y_depth_m, z_m, and yaw_deg") from exc
    required = (0.0, -0.5, 0.0, 0.0)
    if any(abs(actual - expected) > 1e-9 for actual, expected in zip(value, required)):
        raise SystemExit(
            "E8 is intentionally a fixed-centre experiment; target must be "
            "[x=0.0, y_depth=-0.5, z=0.0, yaw=0.0] in controller_world"
        )
    return value


def _positive_float(mapping: dict[str, Any], key: str, label: str) -> float:
    try:
        value = float(mapping[key])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{label}.{key} must be a positive number") from exc
    if not math.isfinite(value) or value <= 0.0:
        raise SystemExit(f"{label}.{key} must be a positive finite number")
    return value


def _thresholds(mapping: dict[str, Any], label: str) -> dict[str, float]:
    result = {
        "x_m": _positive_float(mapping, "x_m", label),
        "depth_m": _positive_float(mapping, "depth_m", label),
        "z_m": _positive_float(mapping, "z_m", label),
        "yaw_deg": _positive_float(mapping, "yaw_deg", label),
    }
    return result


class _DisturbanceNode(_RosExperimentNode):
    """T1 controller-world monitor with online state-motion detection."""

    def __init__(self, *args: Any, target: tuple[float, float, float, float], **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._target = target
        self._latest_motion_state: T1State | None = None
        self._latest_motion_sequence = 0
        self._previous_pose: tuple[float, float, float, float, float] | None = None

    def _controller_pose_callback(self, message: Any) -> None:
        super()._controller_pose_callback(message)
        now = time.monotonic()
        point = message.pose.pose.position
        yaw_deg = self._controller_yaw_deg(message)
        pose = (float(point.x), float(point.y), float(point.z), float(yaw_deg))
        velocity: tuple[float | None, float | None, float | None, float | None] = (None, None, None, None)
        previous = self._previous_pose
        if previous is not None:
            previous_time, previous_x, previous_y, previous_z, previous_yaw = previous
            delta = now - previous_time
            # A large receipt gap represents a transport/vision gap, not a
            # physical low-frequency velocity sample. Keep its diagnostic
            # velocity undefined rather than fabricating a large value.
            if 0.005 <= delta <= 0.50:
                yaw_delta = math.degrees(
                    math.atan2(
                        math.sin(math.radians(yaw_deg - previous_yaw)),
                        math.cos(math.radians(yaw_deg - previous_yaw)),
                    )
                )
                velocity = (
                    (pose[0] - previous_x) / delta,
                    (pose[1] - previous_y) / delta,
                    (pose[2] - previous_z) / delta,
                    yaw_delta / delta,
                )
        self._previous_pose = (now, *pose)
        # Health is updated when the protocol loop takes a snapshot.  A pose
        # callback must never decide health from an arbitrarily old status.
        self._latest_motion_state = make_state(
            timestamp_sec=now,
            pose=pose,
            target=self._target,
            velocity=velocity,
            health_ok=True,
        )
        self._latest_motion_sequence += 1

    def latest_state(
        self,
        *,
        pose_stale_after_sec: float,
        allowed_vision_modes: set[str],
    ) -> tuple[T1State | None, str]:
        raw = self._latest_motion_state
        _, health = self._runtime_health(
            pose_stale_after_sec=pose_stale_after_sec,
            allowed_vision_modes=allowed_vision_modes,
        )
        if raw is None:
            return None, health
        state = T1State(
            timestamp_sec=raw.timestamp_sec,
            x_m=raw.x_m,
            depth_m=raw.depth_m,
            z_m=raw.z_m,
            yaw_deg=raw.yaw_deg,
            x_error_m=raw.x_error_m,
            depth_error_m=raw.depth_error_m,
            z_error_m=raw.z_error_m,
            yaw_error_deg=raw.yaw_error_deg,
            x_velocity_mps=raw.x_velocity_mps,
            depth_velocity_mps=raw.depth_velocity_mps,
            z_velocity_mps=raw.z_velocity_mps,
            yaw_rate_deg_s=raw.yaw_rate_deg_s,
            health_ok=health == "OK",
        )
        return state, health

    def is_tolerated_vision_only_health(
        self,
        health: str,
        tolerated_vision_modes: set[str],
    ) -> bool:
        """Return whether ``health`` is solely an explicitly tolerated vision mode.

        This does *not* make the state usable for arming, detection, or recovery:
        the returned :class:`T1State` remains unhealthy. It only prevents an
        explicitly tolerated visual dropout from immediately terminating the
        runner while the motion controller's own fusion gate publishes zero
        thrust.
        """

        status = self._latest_controller_state_status
        if not isinstance(status, dict):
            return False
        mode = str(status.get("vision_mode", "unknown"))
        if mode not in tolerated_vision_modes or not health.startswith("ABNORMAL(") or not health.endswith(")"):
            return False
        issues = {
            item.strip()
            for item in health.removeprefix("ABNORMAL(").removesuffix(")").split(",")
            if item.strip()
        }
        # Fusion reports ``ready=false`` together with ``vision_mode:lost``.
        # Accept that exact causal pair, but never conceal any independent
        # freshness, IMU, depth, or transport issue.
        tolerated_issues = {f"vision_mode:{mode}"}
        if mode == "lost":
            tolerated_issues.add("ready")
        return issues == tolerated_issues

    @staticmethod
    def _state_text(state: T1State | None, health: str) -> str:
        if state is None:
            return f"pose=unavailable status={health}"
        velocity = (
            f"u={state.x_velocity_mps:+.3f}, v_depth={state.depth_velocity_mps:+.3f}, "
            f"w={state.z_velocity_mps:+.3f} m/s, r={state.yaw_rate_deg_s:+.1f} deg/s"
            if None not in (
                state.x_velocity_mps,
                state.depth_velocity_mps,
                state.z_velocity_mps,
                state.yaw_rate_deg_s,
            )
            else "velocity=warming_up"
        )
        return (
            f"pose=[x={state.x_m:+.3f}, depth={state.depth_m:+.3f}, z={state.z_m:+.3f}]m "
            f"yaw={state.yaw_deg:+.1f}deg "
            f"error=[x={state.x_error_m:+.3f}, depth={state.depth_error_m:+.3f}, "
            f"z={state.z_error_m:+.3f}, yaw={state.yaw_error_deg:+.1f}deg] {velocity} status={health}"
        )

    def run_protocol(
        self,
        *,
        session_id: str,
        label: str,
        metadata: dict[str, Any],
        protocol: dict[str, Any],
        runtime_status: dict[str, Any],
        monitored_processes: Sequence[tuple[str, _ManagedProcess]],
    ) -> dict[str, Any]:
        """Run one centre-point manual disturbance experiment.

        The method does not command an external force.  It keeps the selected
        controller active, announces the armed phase, detects only a sustained
        observed departure, and performs a safe normal return after the
        post-detection recovery window.
        """

        pre_control_sec = _positive_float(protocol, "pre_control_sec", "disturbance_protocol")
        ready_hold_sec = _positive_float(protocol, "ready_hold_sec", "disturbance_protocol")
        max_ready_wait_sec = _positive_float(protocol, "max_ready_wait_sec", "disturbance_protocol")
        max_wait_sec = _positive_float(protocol, "max_wait_for_disturbance_sec", "disturbance_protocol")
        post_min_sec = _positive_float(protocol, "post_detection_min_control_sec", "disturbance_protocol")
        stable_hold_sec = _positive_float(protocol, "required_stable_hold_sec", "disturbance_protocol")
        post_timeout_sec = _positive_float(protocol, "post_detection_timeout_sec", "disturbance_protocol")
        if post_timeout_sec < post_min_sec:
            raise RuntimeError("post_detection_timeout_sec must be >= post_detection_min_control_sec")
        detection_config = _as_mapping(protocol.get("detection", {}), "disturbance_protocol.detection")
        detector = DisturbanceDetector(detection_config)
        arming = _thresholds(_as_mapping(protocol.get("arming_thresholds", {}), "disturbance_protocol.arming_thresholds"), "disturbance_protocol.arming_thresholds")
        recovery = _thresholds(_as_mapping(protocol.get("recovery_thresholds", {}), "disturbance_protocol.recovery_thresholds"), "disturbance_protocol.recovery_thresholds")
        log_interval_sec = _positive_float(runtime_status, "log_interval_sec", "runner.runtime_status")
        pose_stale_after_sec = _positive_float(runtime_status, "pose_stale_after_sec", "runner.runtime_status")
        modes = runtime_status.get("allowed_vision_modes", ["fresh", "coast"])
        if not isinstance(modes, list) or not all(isinstance(item, str) and item.strip() for item in modes):
            raise RuntimeError("runner.runtime_status.allowed_vision_modes must be a list of strings")
        allowed_vision_modes = {item.strip() for item in modes}
        nonfatal_modes = runtime_status.get("nonfatal_vision_modes", [])
        if not isinstance(nonfatal_modes, list) or not all(
            isinstance(item, str) and item.strip() for item in nonfatal_modes
        ):
            raise RuntimeError("runner.runtime_status.nonfatal_vision_modes must be a list of strings")
        nonfatal_vision_modes = {item.strip() for item in nonfatal_modes}
        unhealthy_abort_sec = _positive_float(protocol, "unhealthy_abort_after_sec", "disturbance_protocol")

        started = time.monotonic()
        phase = "PRE_STABILIZE"
        ready_since: float | None = None
        armed_at: float | None = None
        detected_at: float | None = None
        stable_since: float | None = None
        unhealthy_since: float | None = None
        seen_sequence = -1
        next_log = started
        result: dict[str, Any] = {
            "phase": phase,
            "detected": False,
            "target_controller_world": list(self._target),
            "pre_control_sec": pre_control_sec,
            "post_detection_min_control_sec": post_min_sec,
            "required_stable_hold_sec": stable_hold_sec,
        }
        self.publish_event(session_id, "phase_start", "pre_stabilize", metadata)
        self.get_logger().info(
            "[E8] centre target active. Maintaining for at least %.1f s before the perturbation readiness check."
            % pre_control_sec
        )

        while rclpy.ok():
            for process_name, process in monitored_processes:
                if process.returncode is not None:
                    raise RuntimeError(f"runtime process failure: {process_name} exited with code {process.returncode}")
            rclpy.spin_once(self, timeout_sec=0.05)
            now = time.monotonic()
            state, health = self.latest_state(
                pose_stale_after_sec=pose_stale_after_sec,
                allowed_vision_modes=allowed_vision_modes,
            )
            if state is None or not state.health_ok:
                if self.is_tolerated_vision_only_health(health, nonfatal_vision_modes):
                    # A tolerated visual dropout is not a valid experimental
                    # state: no arming/detection/recovery condition can pass
                    # with the health flag false. It is merely non-fatal while
                    # the controller fusion gate suppresses thrust and waits
                    # for fresh AprilTag state to return.
                    unhealthy_since = None
                else:
                    unhealthy_since = unhealthy_since or now
                    if now - unhealthy_since >= unhealthy_abort_sec:
                        raise RuntimeError(f"controller-world state unhealthy for {now - unhealthy_since:.1f}s: {health}")
            else:
                unhealthy_since = None

            if now >= next_log:
                elapsed = now - started
                self.get_logger().info(
                    f"[E8] phase={phase} elapsed={elapsed:.1f}s {self._state_text(state, health)}"
                )
                if self.is_tolerated_vision_only_health(health, nonfatal_vision_modes):
                    self.get_logger().warning(
                        "[E8] AprilTag fusion is in a tolerated visual dropout; the trial remains running, "
                        "but it cannot arm, detect, or confirm recovery until fresh/coast state returns."
                    )
                if phase == "WAIT_DISTURBANCE":
                    remaining = max(0.0, max_wait_sec - (now - float(armed_at))) if armed_at else max_wait_sec
                    self.get_logger().info(
                        f"[E8] waiting for one manual perturbation; no confirmed disturbance yet; timeout in {remaining:.1f}s"
                    )
                elif phase == "POST_DETECTION" and detected_at is not None:
                    self.get_logger().info(
                        f"[E8] recovery observation: elapsed_since_detection={now - detected_at:.1f}s "
                        f"timeout_in={max(0.0, post_timeout_sec - (now - detected_at)):.1f}s"
                    )
                next_log = now + log_interval_sec

            if phase == "PRE_STABILIZE":
                if now - started >= pre_control_sec:
                    phase = "WAIT_READY"
                    result["phase"] = phase
                    self.publish_event(session_id, "phase_end", "pre_stabilize", metadata)
                    self.publish_event(session_id, "phase_start", "wait_ready", metadata)
                    self.get_logger().info("[E8] minimum 10 s control complete; checking whether the vehicle is stably ready.")

            elif phase == "WAIT_READY":
                # Readiness establishes only that the ROV is close enough to
                # the centre target to apply a comparable manual perturbation.
                # Finite-difference speed estimates are intentionally not a
                # readiness, detection, or recovery gate: camera/pose sampling
                # jitter made those gates impractical on this hardware.
                if state is not None and state_within_thresholds(state, arming):
                    ready_since = ready_since or now
                    if now - ready_since >= ready_hold_sec:
                        phase = "WAIT_DISTURBANCE"
                        armed_at = now
                        result["phase"] = phase
                        result["armed_at_monotonic_sec"] = armed_at
                        self.publish_event(session_id, "phase_end", "wait_ready", metadata)
                        self.publish_event(session_id, "perturbation_armed", "centre_target", metadata)
                        self.get_logger().info(
                            "[E8] VEHICLE STABLE: you may apply ONE brief manual perturbation now. "
                            "The controller remains active until automatic detection."
                        )
                else:
                    if ready_since is not None:
                        self.get_logger().warning("[E8] readiness lost; continue controlling and do not perturb yet.")
                    ready_since = None
                    if now - started >= pre_control_sec + max_ready_wait_sec:
                        raise RuntimeError("vehicle did not meet the frozen pre-perturbation readiness criteria")

            elif phase == "WAIT_DISTURBANCE":
                if state is not None and self._latest_motion_sequence != seen_sequence:
                    seen_sequence = self._latest_motion_sequence
                    detection = detector.observe(state)
                    if detection is not None:
                        detected_at = now
                        phase = "POST_DETECTION"
                        result.update({"phase": phase, "detected": True, "detection": detection})
                        event_metadata = {**metadata, "disturbance_detection": detection}
                        self.publish_event(session_id, "disturbance_detected", str(detection["detected_axis"]), event_metadata)
                        self.publish_event(session_id, "phase_start", "post_detection", event_metadata)
                        self.get_logger().info(
                            "[E8] DISTURBANCE DETECTED: axis=%s; controller will continue through recovery."
                            % detection["detected_axis"]
                        )
                if armed_at is not None and now - armed_at >= max_wait_sec:
                    result.update({"phase": "NO_DETECTION_TIMEOUT", "reason": "max_wait_for_disturbance_elapsed"})
                    self.publish_event(session_id, "disturbance_not_detected", "timeout", metadata)
                    self.get_logger().warning("[E8] no observable disturbance was detected before timeout; ending this trial.")
                    break

            elif phase == "POST_DETECTION":
                assert detected_at is not None
                elapsed = now - detected_at
                recovered = state is not None and state_within_thresholds(state, recovery)
                if recovered:
                    stable_since = stable_since or now
                else:
                    stable_since = None
                stable_hold = now - stable_since if stable_since is not None else 0.0
                if elapsed >= post_min_sec and stable_hold >= stable_hold_sec:
                    result.update(
                        {
                            "phase": "RECOVERED",
                            "recovered": True,
                            "online_recovery_entry_after_detection_sec": stable_since - detected_at,
                            "online_stable_hold_sec": stable_hold,
                        }
                    )
                    self.publish_event(session_id, "recovery_confirmed", "all_axes", {**metadata, **result})
                    self.get_logger().info("[E8] recovery confirmed; stopping after the required stable hold.")
                    break
                if elapsed >= post_timeout_sec:
                    result.update({"phase": "RECOVERY_TIMEOUT", "recovered": False, "reason": "post_detection_timeout"})
                    self.publish_event(session_id, "recovery_timeout", "timeout", {**metadata, **result})
                    self.get_logger().warning("[E8] recovery was not confirmed before post-detection timeout.")
                    break

        result["finished_at_monotonic_sec"] = time.monotonic()
        return result


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run one or more E8 centre-point manual disturbance-recovery trials."
    )
    parser.add_argument("--config", required=True, type=Path, help="E8 YAML protocol.")
    parser.add_argument("--controller", choices=("PID", "PPO"), default=None)
    parser.add_argument("--controller-config", required=True, type=Path, help="Single selected motion-controller YAML.")
    parser.add_argument("--checkpoint", type=Path, default=None, help="Optional PPO checkpoint override.")
    parser.add_argument("--trials", type=int, default=None, help="Override disturbance_protocol.trials_per_run.")
    parser.add_argument("--arm", action="store_true", help="Explicitly enable the hardware bridge.")
    parser.add_argument("--dry-run", action="store_true", help="Print the frozen plan without launching ROS2 or hardware.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.trials is not None and args.trials < 1:
        raise SystemExit("--trials must be >= 1")
    repo_root = _repo_root()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    target = _target_from_config(config)
    selected_controller = args.controller
    controller_config = _resolve_cli_path(args.controller_config, repo_root)
    if not controller_config.is_file():
        raise SystemExit(f"--controller-config does not exist: {controller_config}")
    selected_controller = selected_controller or _infer_controller_name(controller_config)
    controllers = _as_mapping(config["controllers"], "controllers")
    selected = copy.deepcopy(_as_mapping(controllers.get(selected_controller), f"controllers.{selected_controller}"))
    selected["config_file"] = str(controller_config)
    checkpoint = _controller_checkpoint(controller_config, repo_root)
    if args.checkpoint is not None:
        checkpoint = _resolve_cli_path(args.checkpoint, repo_root)
    selected["checkpoint_path"] = str(checkpoint) if checkpoint else None
    controllers[selected_controller] = selected
    if selected_controller == "PPO" and not args.dry_run and (checkpoint is None or not checkpoint.is_file()):
        raise SystemExit("selected PPO profile must resolve to an existing checkpoint")

    runner = _as_mapping(config["runner"], "runner")
    auto = _as_mapping(runner.get("auto_launch", {}), "runner.auto_launch")
    safety = _as_mapping(runner.get("safety", {}), "runner.safety")
    logging_cfg = _as_mapping(runner.get("logging", {}), "runner.logging")
    if not bool(auto.get("enabled", True)):
        raise SystemExit("E8 runner.auto_launch.enabled must be true")
    if bool(runner.get("require_explicit_arm_flag", True)) and not args.arm and not args.dry_run:
        raise SystemExit("formal E8 execution requires --arm; use --dry-run to inspect the plan")
    protocol = _as_mapping(config["disturbance_protocol"], "disturbance_protocol")
    trials = int(protocol.get("trials_per_run", 1)) if args.trials is None else int(args.trials)
    if trials < 1:
        raise SystemExit("disturbance_protocol.trials_per_run must be >= 1")
    # Validate all protocol values before creating a run directory or arming hardware.
    DisturbanceDetector(_as_mapping(protocol.get("detection", {}), "disturbance_protocol.detection"))
    _thresholds(_as_mapping(protocol.get("arming_thresholds", {}), "disturbance_protocol.arming_thresholds"), "disturbance_protocol.arming_thresholds")
    _thresholds(_as_mapping(protocol.get("recovery_thresholds", {}), "disturbance_protocol.recovery_thresholds"), "disturbance_protocol.recovery_thresholds")

    output_root = _resolve_path(
        str(logging_cfg.get("root", "ros2_ws/data/experiments/hardware/T1_disturbance")),
        repo_root,
    )
    strategy = _strategy_name_from_config(selected["config_file"])
    session_slugs = _as_mapping(runner.get("session_strategy_slugs", {}), "runner.session_strategy_slugs")
    session_strategy = _session_strategy_slug(strategy, session_slugs)
    output_root = output_root / _controller_storage_subdirectory(selected_controller, selected)
    run_id = _new_run_id(output_root, f"E8_{strategy}")
    run_dir = output_root / run_id
    trials_dir = run_dir / "trials"
    logs_dir = run_dir / "logs"
    config_files = _config_files(config, config_path, repo_root)
    missing = [str(path) for path in config_files if not path.is_file()]
    if missing:
        raise SystemExit(f"configured files do not exist: {missing}")
    trial_plan = [
        {"trial_index": index, "repeat": index, "target_controller_world": list(target), "controller": selected_controller}
        for index in range(1, trials + 1)
    ]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "status": "planned" if args.dry_run else "running",
        "experiment_id": "E8",
        "experiment_kind": "manual_state_observed_disturbance_recovery",
        "run_id": run_id,
        "created_at": now_iso(),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git_revision": git_revision(repo_root),
        "armed": bool(args.arm),
        "target_controller_world": list(target),
        "controller_selection": {
            "controller": selected_controller,
            "strategy": strategy,
            "session_strategy": session_strategy,
            "config_file": str(controller_config),
            "checkpoint": str(checkpoint) if checkpoint else None,
        },
        "config_files": {str(path): file_sha256(path) for path in config_files},
        "trial_plan": trial_plan,
        "support_stack": {"mode": "launched_by_runner" if bool(auto.get("start_support_stack", False)) else "pre_started"},
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        return

    run_dir.mkdir(parents=True, exist_ok=False)
    trials_dir.mkdir()
    logs_dir.mkdir()
    write_json(run_dir / "run_manifest.json", manifest)
    event_topic = str(logging_cfg.get("event_topic", "/finsrov/experiment/event"))
    profile = str(logging_cfg.get("recorder_profile", "t1"))
    startup_timeout = _positive_float(auto, "startup_timeout_sec", "runner.auto_launch")
    startup_settle = float(auto.get("startup_settle_sec", 3.0))
    terminate_timeout = _positive_float(auto, "terminate_timeout_sec", "runner.auto_launch")
    required_support_topics = [str(item) for item in auto.get("required_support_topics", [])]
    required_controller_topics = [str(item) for item in auto.get("required_controller_topics", [])]
    support_processes: list[_ManagedProcess] = []
    node: _DisturbanceNode | None = None
    emergency_stop_done = False

    def emergency_shutdown(reason: str) -> None:
        nonlocal emergency_stop_done
        if emergency_stop_done:
            return
        emergency_stop_done = True
        record: dict[str, Any] = {"reason": reason, "requested_at": now_iso()}
        if node is None:
            record["status"] = "node_unavailable"
        else:
            emergency_cfg = _as_mapping(safety.get("emergency_stop", {}), "runner.safety.emergency_stop")
            try:
                record.update(node.emergency_stop(
                    zero_frames=int(emergency_cfg.get("zero_frames", 6)),
                    zero_frame_interval_sec=float(emergency_cfg.get("zero_frame_interval_sec", 0.05)),
                    parameter_timeout_sec=float(emergency_cfg.get("parameter_timeout_sec", 1.0)),
                    confirmation_timeout_sec=float(emergency_cfg.get("confirmation_timeout_sec", 2.0)),
                ))
                record["status"] = "complete" if record.get("bridge_disabled_confirmed") else "unconfirmed"
            except Exception as exc:  # pragma: no cover - hardware fallback.
                record.update({"status": "error", "error": repr(exc)})
        manifest["emergency_stop"] = record

    try:
        if bool(auto.get("start_support_stack", False)):
            for name, command in _launch_commands(config, selected_controller, args.arm, repo_root)[:3]:
                support_processes.append(_ManagedProcess(command, logs_dir / f"{name}.log"))
        _wait_for_topics(required_support_topics, startup_timeout)
        time.sleep(max(0.0, startup_settle))
        init_experiment_ros()
        emergency_cfg = _as_mapping(safety.get("emergency_stop", {}), "runner.safety.emergency_stop")
        node = _DisturbanceNode(
            event_topic,
            bridge_node_name=str(emergency_cfg.get("bridge_node", "/hardware_bridge")),
            bridge_status_topic=str(emergency_cfg.get("bridge_status_topic", "/finsrov/hardware/status")),
            node_name="finsrov_e8_disturbance_runner",
            target=target,
        )
        bridge_arming = {"requested": bool(args.arm), "status": "not_requested"}
        if args.arm:
            bridge_arming.update(node.set_bridge_enabled(
                enabled=True,
                parameter_timeout_sec=float(emergency_cfg.get("parameter_timeout_sec", 1.0)),
                confirmation_timeout_sec=float(emergency_cfg.get("confirmation_timeout_sec", 2.0)),
            ))
            bridge_arming["status"] = "confirmed" if bridge_arming.get("bridge_enabled_confirmed") else "unconfirmed"
        manifest["bridge_arming"] = bridge_arming
        write_json(run_dir / "run_manifest.json", manifest)
        if args.arm and not bridge_arming.get("bridge_enabled_confirmed"):
            raise RuntimeError("--arm requested but hardware bridge did not confirm enabled=true")

        for item in trial_plan:
            repeat = int(item["repeat"])
            session_id = f"{run_id}_E8_{session_strategy}_CENTER_R{repeat:02d}"
            metadata = {
                "analysis_kind": "t1_disturbance_recovery",
                "experiment_kind": "manual_state_observed_disturbance_recovery",
                "run_id": run_id,
                "controller": selected_controller,
                "strategy": session_strategy,
                "repeat": repeat,
                "setpoint_id": "CENTER_X0_DEPTH_NEG_0_5_Z0_YAW0",
                "target_controller_world": list(target),
                "protocol_config": str(config_path),
                "controller_config": str(controller_config),
                "disturbance_protocol": protocol,
            }
            controller_process: _ManagedProcess | None = None
            recorder_process: _ManagedProcess | None = None
            try:
                controller_command = _launch_commands(config, selected_controller, args.arm, repo_root)[3][1]
                controller_process = _ManagedProcess(controller_command, logs_dir / f"{session_id}.controller.log")
                _wait_for_topics(required_controller_topics, startup_timeout)
                time.sleep(max(0.0, startup_settle))
                _require_process_running(controller_process, "motion controller")
                recorder_process = _ManagedProcess(
                    _record_command(
                        experiment_id="E8",
                        strategy=session_strategy,
                        session_id=session_id,
                        output_root=trials_dir,
                        profile=profile,
                        metadata=metadata,
                        config_files=config_files,
                        checkpoint=checkpoint,
                    ),
                    logs_dir / f"{session_id}.recorder.log",
                )
                time.sleep(1.0)
                _require_process_running(recorder_process, "experiment recorder")
                node.publish_event(session_id, "trial_start", "centre_target", metadata)
                node.publish_goal(*target)
                node.publish_event(session_id, "hold_start", "centre_target", metadata)
                result = node.run_protocol(
                    session_id=session_id,
                    label="centre_target",
                    metadata=metadata,
                    protocol=protocol,
                    runtime_status=_as_mapping(runner.get("runtime_status", {}), "runner.runtime_status"),
                    monitored_processes=[("motion_controller", controller_process), ("experiment_recorder", recorder_process)],
                )
                metadata["runtime_result"] = result
                node.publish_event(session_id, "phase_end", str(result.get("phase", "unknown")), {**metadata, **result})
                node.cancel_goal()
                node.publish_event(session_id, "trial_end", "centre_target", {**metadata, **result})
                recorder_process.stop(terminate_timeout)
                recorder_process = None
                manifest.setdefault("completed_trials", []).append({"session_id": session_id, "result": result})
                trial_manifest_path = trials_dir / session_id / "manifest.json"
                if trial_manifest_path.is_file():
                    trial_manifest = read_json(trial_manifest_path)
                    manifest.setdefault("analysis_reports", []).append({
                        "session_id": session_id,
                        "status": trial_manifest.get("analysis", {}).get("status", "needs_review"),
                        "report": str(trials_dir / session_id / "derived" / "disturbance_analysis_report.json"),
                    })
            except KeyboardInterrupt:
                metadata["error"] = "KeyboardInterrupt"
                manifest.setdefault("failed_trials", []).append(metadata)
                try:
                    node.publish_event(session_id, "operator_abort", "centre_target", metadata)
                except Exception:
                    pass
                emergency_shutdown("keyboard_interrupt")
                raise
            except Exception as exc:
                metadata["error"] = repr(exc)
                manifest.setdefault("failed_trials", []).append(metadata)
                emergency_shutdown("runner_exception")
                raise
            finally:
                if recorder_process is not None:
                    recorder_process.stop(terminate_timeout, force=emergency_stop_done)
                if controller_process is not None:
                    controller_process.stop(terminate_timeout, force=emergency_stop_done)
                write_json(run_dir / "run_manifest.json", manifest)
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        emergency_shutdown("keyboard_interrupt_outer")
        raise
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        emergency_shutdown("runner_exception_outer")
        raise
    finally:
        if node is not None:
            try:
                if not emergency_stop_done:
                    normal_stop = node.stop_motion(
                        zero_frames=int(emergency_cfg.get("zero_frames", 6)),
                        zero_frame_interval_sec=float(emergency_cfg.get("zero_frame_interval_sec", 0.05)),
                    )
                    normal_stop["status"] = "complete"
                    manifest["normal_stop"] = normal_stop
            except Exception as exc:  # pragma: no cover - hardware fallback.
                manifest["normal_stop"] = {"status": "error", "error": repr(exc)}
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        for process in reversed(support_processes):
            process.stop(terminate_timeout)
        manifest["status"] = "complete" if manifest.get("status") == "running" else manifest.get("status", "complete")
        manifest["finished_at"] = now_iso()
        write_json(run_dir / "run_manifest.json", manifest)


if __name__ == "__main__":
    main()
