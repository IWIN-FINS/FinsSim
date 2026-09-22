"""Automatic T1 hardware experiment runner.

The runner accepts one versioned T1 protocol YAML and one motion-controller profile
supplied on the CLI. It generates the
run/test/session identifiers, verifies (or optionally launches) the fixed
perception/fusion/bridge support stack, starts one controller and rosbag session
per trial, publishes event markers and goals, and performs a safe shutdown. It
intentionally requires an explicit ``--arm`` flag for formal hardware execution;
the canonical protocol reuses a support stack that was enabled beforehand.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from typing import Any, Sequence

import rclpy
from rclpy.signals import SignalHandlerOptions
from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from rclpy.parameter import Parameter
from std_msgs.msg import Bool, Empty, Float32MultiArray, String
from msgs.msg import TrajectoryEvent
import yaml

from .analysis import analyze_session
from .layout import (
    canonical_run_dir,
    create_run_layout,
    prepare_trial_layout,
    register_trial_runtime_artifacts,
    shared_log_path,
)
from .manifest import file_sha256, git_revision, now_iso, read_json, write_json
from .progress import log_stage


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _resolve_cli_path(value: str | Path, repo_root: Path) -> Path:
    """Resolve CLI paths from either the caller's cwd or repository root."""

    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    cwd_path = (Path.cwd() / path).resolve()
    if cwd_path.exists():
        return cwd_path
    return _resolve_path(path, repo_root)


def _controller_storage_subdirectory(controller_key: str, controller: dict[str, Any]) -> str:
    """Return the immutable controller bucket used below each experiment group.

    Raw runs must never be mixed across PID, direct-eight-thruster PPO, and
    wrench-to-allocation PPO. The protocol YAML declares the bucket so an
    ambiguous user-facing controller key such as the T1 hardware ``PPO`` key
    cannot silently choose a different layout.
    """

    configured = str(controller.get("storage_subdirectory", "")).strip()
    allowed = {"PID", "PPO_THRUSTER8", "PPO_WRENCH6"}
    if configured in allowed:
        return configured

    identity = " ".join(
        str(value)
        for value in (
            controller_key,
            controller.get("label", ""),
            controller.get("experiment_label", ""),
            controller.get("action_interface", ""),
            controller.get("policy_action_interface", ""),
        )
    ).lower()
    if "thruster8" in identity or "eight_thruster" in identity:
        return "PPO_THRUSTER8"
    if "pid" in identity:
        return "PID"
    if "wrench6" in identity or "wrench" in identity:
        return "PPO_WRENCH6"
    raise SystemExit(
        f"controller `{controller_key}` has no valid storage_subdirectory; "
        f"set it to one of {sorted(allowed)} in the experiment YAML"
    )


def init_experiment_ros() -> None:
    """Initialize ROS without letting rclpy consume terminal Ctrl-C first.

    The experiment runner owns a hardware-safe interrupt sequence: cancel the
    controller, publish zero commands, disable the bridge, and finally stop
    the rosbag/controller process groups. The default rclpy SIGINT handler
    shuts down its context before Python raises ``KeyboardInterrupt``. That
    makes every publisher invalid exactly when this runner needs it most.
    Keeping the standard Python handler lets the surrounding ``try/finally``
    execute the shutdown sequence while ROS publishers and service clients
    are still usable.
    """

    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)


def _strategy_name_from_config(value: object) -> str:
    """Use the selected controller YAML stem as the stable strategy name."""

    stem = Path(str(value)).stem
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in stem)
    return safe.strip("-_") or "UNSPECIFIED"


def _session_strategy_slug(strategy_name: str, configured_slugs: dict[str, Any]) -> str:
    """Return a bounded filesystem slug while preserving the full profile elsewhere."""

    candidate = configured_slugs.get(strategy_name, strategy_name)
    if not isinstance(candidate, str) or not candidate.strip():
        raise SystemExit(f"runner.session_strategy_slugs[{strategy_name!r}] must be a non-empty string")
    safe = "".join(char if char.isalnum() or char in "-_" else "_" for char in candidate).strip("-_")
    if not safe:
        raise SystemExit(f"runner.session_strategy_slugs[{strategy_name!r}] resolves to an empty slug")
    # A configured alias should normally be used for long names.  The stable
    # digest fallback prevents a new long CLI profile from ever exceeding the
    # per-component filesystem limit before it is added to the protocol YAML.
    if len(safe) > 48:
        import hashlib

        safe = f"{safe[:36].rstrip('-_')}_{hashlib.sha256(strategy_name.encode()).hexdigest()[:8]}"
    return safe


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"T1 experiment YAML not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid T1 experiment YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"T1 experiment YAML must contain a mapping: {path}")
    for key in ("runner", "target_points", "controllers", "trial_protocol", "metric_protocol"):
        if not isinstance(payload.get(key), dict):
            raise SystemExit(f"T1 experiment YAML requires mapping `{key}`")
    return payload


def _controller_checkpoint(path: Path, repo_root: Path) -> Path | None:
    """Read an optional checkpoint from a controller profile YAML.

    Most profiles use the historic ``motion_controller`` node key.  Hybrid
    controllers intentionally use their executable name as the YAML node key,
    so treating that historic key as the only accepted shape would let the
    recorder's manifest disagree with the controller actually launched.
    """

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"controller YAML not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid controller YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"controller YAML must contain a mapping: {path}")
    parameter_sets: list[tuple[str, dict[str, Any]]] = []
    for node_name, node_config in payload.items():
        if not isinstance(node_config, dict):
            continue
        params = node_config.get("ros__parameters")
        if isinstance(params, dict):
            parameter_sets.append((str(node_name), params))
    if not parameter_sets:
        return None
    preferred = [item for item in parameter_sets if item[0] == "motion_controller"]
    candidates = preferred or parameter_sets
    values = {
        str(params.get("checkpoint_path", "")).strip()
        for _, params in candidates
        if str(params.get("checkpoint_path", "")).strip()
    }
    if not values:
        return None
    if len(values) != 1:
        raise SystemExit(
            f"controller YAML {path} declares multiple non-empty checkpoint_path values; "
            "the formal runner requires exactly one"
        )
    return _resolve_path(values.pop(), repo_root)


def _controller_profile_node_keys(path: Path) -> set[str]:
    """Return YAML node keys that carry a ROS parameter block."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read controller YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"controller YAML must contain a mapping: {path}")
    return {
        str(node_name)
        for node_name, node_config in payload.items()
        if isinstance(node_config, dict) and isinstance(node_config.get("ros__parameters"), dict)
    }


def _validate_controller_profile_runtime(selected: dict[str, Any], controller_config: Path) -> None:
    """Reject a profile whose YAML node key cannot configure its executable.

    ROS2 parameter files apply values by node name.  Launching the hybrid
    executable under ``hybrid_horizontal_ppo_depth_pid_controller`` with an
    ordinary ``motion_controller:`` YAML silently loads its defaults.  In a
    hardware experiment that must be a pre-arm error, not a zero-thrust trial.
    """

    expected_node_name = str(selected.get("controller_node_name", "motion_controller")).strip()
    if not expected_node_name:
        raise SystemExit("selected controller profile has an empty controller_node_name")
    available_keys = _controller_profile_node_keys(controller_config)
    if expected_node_name not in available_keys and "/**" not in available_keys:
        raise SystemExit(
            f"--controller-config {controller_config} configures ROS node key(s) {sorted(available_keys)}, "
            f"but this T1 controller launches `{expected_node_name}`. "
            "Use the matching hybrid_horizontal_ppo_depth_pid.yaml profile, or select a controller "
            "runtime whose node name matches the supplied YAML."
        )


def _infer_controller_name(path: Path) -> str:
    """Infer PID/PPO only when the profile name is unambiguous."""

    value = path.name.lower()
    if "pid" in value or "traditional" in value:
        return "PID"
    if "ppo" in value or "policy" in value:
        return "PPO"
    raise SystemExit(
        "cannot infer controller type from --controller-config; "
        "pass --controller PID or --controller PPO explicitly"
    )


def _new_run_id(root: Path, prefix: str) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{prefix}"
    candidate = base
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _as_mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a YAML mapping")
    return value


def _mcu_disarm_confirmed(
    status: dict[str, Any] | None,
    *,
    status_sequence: int,
    status_sequence_before_disarm: int,
) -> bool:
    """Return true only for a fresh firmware-confirmed disabled zero state."""

    return bool(
        isinstance(status, dict)
        and status_sequence > status_sequence_before_disarm
        and status.get("enabled") is False
        and status.get("mcu_diagnostics_supported") is True
        and status.get("mcu_has_received_command") is True
        and status.get("mcu_direct_thrusters_enabled") is False
        and status.get("mcu_target_rpm_nonzero") is False
        and status.get("mcu_target_throttle_nonzero") is False
    )


def _bridge_enabled_confirmed(
    status: dict[str, Any] | None,
    *,
    status_sequence: int,
    status_sequence_before_request: int,
) -> bool:
    """Return true only after a fresh bridge status confirms enabled=true."""

    return bool(
        isinstance(status, dict)
        and status_sequence > status_sequence_before_request
        and status.get("enabled") is True
    )


class _ManagedProcess:
    def __init__(self, command: list[str], log_path: Path) -> None:
        self.command = command
        self.log_path = log_path
        log_path.parent.mkdir(parents=True, exist_ok=True)
        self._log = log_path.open("w", encoding="utf-8")
        try:
            self.process = subprocess.Popen(
                command,
                stdout=self._log,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except Exception:
            self._log.close()
            raise

    @property
    def returncode(self) -> int | None:
        return self.process.poll()

    def _group_exists(self) -> bool:
        """Return whether the session/process group still contains a member.

        ``ros2 launch`` can exit after forwarding a signal while one of its
        Python node children survives.  In that state ``Popen.poll()`` is no
        longer sufficient: the group is still live and can retain a controller
        publisher.  ``killpg(..., 0)`` checks the group without modifying it.
        """

        try:
            os.killpg(self.process.pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        return True

    def _signal_group(self, signal_value: int) -> None:
        try:
            os.killpg(self.process.pid, signal_value)
        except ProcessLookupError:
            pass

    def _wait_for_group_exit(self, timeout_sec: float) -> bool:
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while self._group_exists() and time.monotonic() < deadline:
            time.sleep(0.05)
        return not self._group_exists()

    def _wait_for_leader(self, timeout_sec: float) -> None:
        if self.process.poll() is not None:
            return
        try:
            self.process.wait(timeout=max(float(timeout_sec), 0.05))
        except subprocess.TimeoutExpired:
            pass

    def stop(self, timeout_sec: float, *, force: bool = False) -> int | None:
        """Stop the complete child process group.

        Normal trial transitions use SIGINT so rosbag and launch can close
        cleanly. An armed hardware emergency first commands a disabled zero
        frame, then calls ``force=True`` to kill this group without waiting.
        """

        graceful_timeout = max(float(timeout_sec), 1.0)
        try:
            if force:
                self._signal_group(signal.SIGKILL)
                self._wait_for_leader(2.0)
                self._wait_for_group_exit(2.0)
                return self.process.returncode

            # Always signal the process *group*, even if the launch leader
            # already exited.  This is the critical orphan-controller case.
            self._signal_group(signal.SIGINT)
            self._wait_for_leader(graceful_timeout)
            if self._wait_for_group_exit(graceful_timeout):
                return self.process.returncode

            self._signal_group(signal.SIGTERM)
            self._wait_for_leader(graceful_timeout)
            if self._wait_for_group_exit(graceful_timeout):
                return self.process.returncode

            self._signal_group(signal.SIGKILL)
            self._wait_for_leader(5.0)
            self._wait_for_group_exit(5.0)
            return self.process.returncode
        finally:
            if not self._log.closed:
                self._log.close()


def _require_process_running(process: _ManagedProcess, label: str) -> None:
    """Fail closed when a just-launched critical child exits early."""

    returncode = process.returncode
    if returncode is not None:
        raise RuntimeError(f"{label} exited during startup with code {returncode}")


class _RosExperimentNode(Node):
    def __init__(
        self,
        event_topic: str,
        *,
        bridge_node_name: str,
        bridge_status_topic: str,
        node_name: str = "finsrov_t1_experiment_runner",
    ) -> None:
        super().__init__(node_name)
        self._event_pub = self.create_publisher(TrajectoryEvent, event_topic, 20)
        self._goal_pub = self.create_publisher(PoseStamped, "/motion_controller/command/position_controller_world", 10)
        self._cancel_pub = self.create_publisher(Empty, "/motion_controller/command/cancel", 10)
        self._zero_thruster_pub = self.create_publisher(Float32MultiArray, "/finsrov/thrusters_out", 20)
        self._bridge_parameter_service = f"{bridge_node_name.rstrip('/')}/set_parameters"
        self._bridge_set_parameters_client = self.create_client(
            SetParameters, self._bridge_parameter_service
        )
        self._bridge_status: dict[str, Any] | None = None
        self._bridge_status_sequence = 0
        # Runtime reporting deliberately subscribes to the same converted
        # controller-world state that the PID/PPO process consumes.  The raw
        # /finsrov/pose stream is pool_world and would make a printed T1
        # position disagree with the commanded controller-world target.
        self._latest_controller_pose: PoseWithCovarianceStamped | None = None
        self._latest_controller_pose_received_monotonic: float | None = None
        self._latest_controller_state_status: dict[str, Any] | None = None
        self._latest_controller_state_status_received_monotonic: float | None = None
        self._goal_reached = False
        self._goal_reached_sequence = 0
        self._last_runtime_health: str | None = None
        self.create_subscription(String, bridge_status_topic, self._bridge_status_callback, 20)
        self.create_subscription(
            PoseWithCovarianceStamped,
            "/finsrov/controller/pose",
            self._controller_pose_callback,
            20,
        )
        self.create_subscription(
            String,
            "/finsrov/controller/state/status",
            self._controller_state_status_callback,
            20,
        )
        self.create_subscription(Bool, "/motion_controller/status/reached", self._goal_reached_callback, 20)

    def _bridge_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._bridge_status = payload
            self._bridge_status_sequence += 1

    def _controller_pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        self._latest_controller_pose = message
        self._latest_controller_pose_received_monotonic = time.monotonic()

    def _controller_state_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._latest_controller_state_status = payload
            self._latest_controller_state_status_received_monotonic = time.monotonic()

    def _goal_reached_callback(self, message: Bool) -> None:
        self._goal_reached = bool(message.data)
        self._goal_reached_sequence += 1

    def clear_controller_preflight(self) -> None:
        """Forget prior-controller samples before accepting a new controller.

        A ROS graph can retain topic names briefly after a controller process
        exits.  The next phase must observe messages from its newly launched
        controller rather than treating those names as a readiness proof.
        """

        self._latest_controller_pose = None
        self._latest_controller_pose_received_monotonic = None
        self._latest_controller_state_status = None
        self._latest_controller_state_status_received_monotonic = None
        self._goal_reached = False

    @staticmethod
    def _controller_yaw_deg(message: PoseWithCovarianceStamped) -> float:
        """Extract the controller-world yaw (rotation about +Y) in degrees."""

        orientation = message.pose.pose.orientation
        # This is the Y-axis analogue of the usual Z-up yaw extraction.  The
        # controller command wire format also uses a pure Y-axis quaternion.
        sine = 2.0 * (orientation.w * orientation.y + orientation.x * orientation.z)
        cosine = 1.0 - 2.0 * (orientation.y * orientation.y + orientation.z * orientation.z)
        return math.degrees(math.atan2(sine, cosine))

    @staticmethod
    def _controller_tilt_deg(message: PoseWithCovarianceStamped) -> float:
        """Return the yaw-invariant angle between body +Y and world +Y.

        The T1 controller frame is Y-up.  Decomposing its quaternion into an
        Euler ``pitch`` is not a safe attitude guard: for example, a level
        vehicle at yaw=-127 deg can legitimately select the 180 deg pitch
        branch.  Rotate the body-up vector directly instead, which measures
        the physical roll/pitch tilt independently of yaw.
        """

        orientation = message.pose.pose.orientation
        x, y, z, w = float(orientation.x), float(orientation.y), float(orientation.z), float(orientation.w)
        norm = math.sqrt(x * x + y * y + z * z + w * w)
        if norm <= 1e-9:
            return math.inf
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        # The second column of R(q) is body +Y expressed in world axes.
        body_up_world_y = 1.0 - 2.0 * (x * x + z * z)
        return math.degrees(math.acos(max(-1.0, min(1.0, body_up_world_y))))

    def wait_for_motion_controller_ready(
        self,
        *,
        reached_sequence_before_start: int,
        process: _ManagedProcess,
        label: str,
        timeout_sec: float,
    ) -> None:
        """Require the controller child, not merely its launch parent, to run.

        ``ros2 launch`` remains alive when its ``controller_state_adapter``
        child survives a crashed motion-controller child.  Topic discovery is
        similarly insufficient because that adapter publishes the controller
        state topics by itself.  A live `/motion_controller` graph node plus a
        post-launch `reached=false` heartbeat proves the actual controller has
        reached its control loop before a goal is published.
        """

        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        missing_node_since: float | None = None
        while rclpy.ok() and time.monotonic() < deadline:
            if process.returncode is not None:
                raise RuntimeError(f"{label} launch process exited with code {process.returncode}")
            rclpy.spin_once(self, timeout_sec=0.1)
            node_present = any(
                name == "motion_controller" and namespace in {"", "/"}
                for name, namespace in self.get_node_names_and_namespaces()
            )
            now = time.monotonic()
            if not node_present:
                if missing_node_since is None:
                    missing_node_since = now
                elif now - missing_node_since >= 3.0:
                    raise RuntimeError(
                        f"{label} motion_controller node did not appear; inspect its controller log"
                    )
                continue
            missing_node_since = None
            if self._goal_reached_sequence > reached_sequence_before_start and not self._goal_reached:
                return
        raise RuntimeError(
            f"{label} did not publish a post-launch /motion_controller/status/reached=false heartbeat"
        )

    def wait_for_autoreset_success(
        self,
        *,
        target_controller_world: Sequence[float],
        reached_sequence_before_goal: int,
        timeout_sec: float,
        position_tolerance_m: float,
        yaw_tolerance_deg: float,
        enforce_state_health: bool,
        tilt_guard_enabled: bool,
        tilt_limit_deg: float,
        tilt_max_continuous_violation_sec: float,
        pose_stale_after_sec: float,
        allowed_vision_modes: set[str],
        monitored_processes: Sequence[tuple[str, _ManagedProcess]],
    ) -> dict[str, float]:
        """Wait for a fresh, validated return-to-start completion.

        ``/motion_controller/status/reached`` is emitted only after the
        controller's own continuous-hold timer.  This runner-side gate also
        requires a post-command false->true transition and independently
        checks the current pose/yaw and, when enabled, the tilt guard, so a
        retained status from the preceding target cannot release the next
        formal trial.
        """

        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        next_log = time.monotonic()
        abnormal_since: float | None = None
        tilt_violation_since: float | None = None
        saw_fresh_false = False
        target_x, target_y, target_z, target_yaw = (float(value) for value in target_controller_world)
        while rclpy.ok() and time.monotonic() < deadline:
            for label, process in monitored_processes:
                if process.returncode is not None:
                    raise RuntimeError(f"autoreset process failure: {label} exited with code {process.returncode}")
            rclpy.spin_once(self, timeout_sec=0.1)
            now = time.monotonic()
            health_position, health = self._runtime_health(
                pose_stale_after_sec=pose_stale_after_sec,
                allowed_vision_modes=allowed_vision_modes,
            )
            if enforce_state_health:
                if health == "OK":
                    abnormal_since = None
                elif abnormal_since is None:
                    abnormal_since = now
                elif now - abnormal_since >= max(2.0, pose_stale_after_sec * 2.0):
                    raise RuntimeError(f"autoreset state became unhealthy: {health}")

            if self._goal_reached_sequence > reached_sequence_before_goal and not self._goal_reached:
                saw_fresh_false = True

            pose = self._latest_controller_pose
            if pose is not None:
                point = pose.pose.pose.position
                position_error = math.sqrt(
                    (float(point.x) - target_x) ** 2
                    + (float(point.y) - target_y) ** 2
                    + (float(point.z) - target_z) ** 2
                )
                yaw_error = abs(((self._controller_yaw_deg(pose) - target_yaw + 180.0) % 360.0) - 180.0)
                tilt_deg = self._controller_tilt_deg(pose)
                tilt_safe = tilt_deg <= float(tilt_limit_deg)
                if tilt_guard_enabled and not tilt_safe:
                    if tilt_violation_since is None:
                        tilt_violation_since = now
                    elif now - tilt_violation_since >= float(tilt_max_continuous_violation_sec):
                        raise RuntimeError(
                            "autoreset tilt guard violated continuously: "
                            f"tilt={tilt_deg:.2f} deg limit={float(tilt_limit_deg):.2f} deg "
                            f"duration={now - tilt_violation_since:.2f}s"
                        )
                else:
                    tilt_violation_since = None
                if (
                    saw_fresh_false
                    and self._goal_reached_sequence > reached_sequence_before_goal
                    and self._goal_reached
                    and (not enforce_state_health or health == "OK")
                    and position_error <= float(position_tolerance_m)
                    and yaw_error <= float(yaw_tolerance_deg)
                ):
                    return {
                        "position_error_m": position_error,
                        "yaw_error_deg": yaw_error,
                        "tilt_deg": tilt_deg,
                        "reached_sequence": float(self._goal_reached_sequence),
                    }
            if now >= next_log:
                self.get_logger().info(
                    "T1 autoreset waiting: "
                    f"target=[{target_x:+.3f}, {target_y:+.3f}, {target_z:+.3f}] yaw={target_yaw:+.1f} "
                    f"fresh_false={saw_fresh_false} reached={self._goal_reached} "
                    f"position={health_position} status={health}"
                )
                next_log = now + 1.0
        raise RuntimeError(f"autoreset timed out after {float(timeout_sec):.1f}s without a validated success")

    def _runtime_health(
        self,
        *,
        pose_stale_after_sec: float,
        allowed_vision_modes: set[str],
    ) -> tuple[str, str]:
        """Return a printable controller-world pose and conservative health."""

        now = time.monotonic()
        issues: list[str] = []
        position = "unavailable"
        pose = self._latest_controller_pose
        pose_time = self._latest_controller_pose_received_monotonic
        if pose is None or pose_time is None:
            issues.append("controller_pose_missing")
        else:
            point = pose.pose.pose.position
            position = (
                f"[x={point.x:+.3f}, y(depth)={point.y:+.3f}, "
                f"z={point.z:+.3f}] m yaw={self._controller_yaw_deg(pose):+.1f} deg"
            )
            age = now - pose_time
            if age > pose_stale_after_sec:
                issues.append(f"controller_pose_stale:{age:.2f}s")

        status = self._latest_controller_state_status
        status_time = self._latest_controller_state_status_received_monotonic
        if status is None or status_time is None:
            issues.append("controller_state_status_missing")
        else:
            age = now - status_time
            if age > pose_stale_after_sec:
                issues.append(f"controller_state_status_stale:{age:.2f}s")
            for key in ("ready", "initialized", "imu_fresh", "depth_fresh"):
                if not bool(status.get(key, False)):
                    issues.append(key)
            vision_mode = str(status.get("vision_mode", "unknown"))
            if allowed_vision_modes and vision_mode not in allowed_vision_modes:
                issues.append(f"vision_mode:{vision_mode}")
        return position, "OK" if not issues else f"ABNORMAL({', '.join(issues)})"

    def spin_hold_with_runtime_status(
        self,
        duration_sec: float,
        *,
        setpoint_id: str,
        repeat: int,
        target_controller_world: Sequence[float],
        log_interval_sec: float,
        pose_stale_after_sec: float,
        allowed_vision_modes: set[str],
        monitored_processes: Sequence[tuple[str, _ManagedProcess]],
    ) -> None:
        """Run a T1 hold while printing progress, state and health at runtime."""

        interval = max(float(log_interval_sec), 0.1)
        duration = max(float(duration_sec), 0.0)
        started = time.monotonic()
        deadline = started + duration
        next_log = started
        self._last_runtime_health = None
        target = ", ".join(f"{float(value):+.3f}" for value in target_controller_world[:3])
        target_yaw = float(target_controller_world[3]) if len(target_controller_world) > 3 else 0.0
        while rclpy.ok() and time.monotonic() < deadline:
            for label, process in monitored_processes:
                if process.returncode is not None:
                    message = f"T1 runtime process failure: {label} exited with code {process.returncode}"
                    self.get_logger().error(message)
                    raise RuntimeError(message)
            now = time.monotonic()
            if now >= next_log:
                elapsed = min(max(now - started, 0.0), duration)
                position, health = self._runtime_health(
                    pose_stale_after_sec=pose_stale_after_sec,
                    allowed_vision_modes=allowed_vision_modes,
                )
                message = (
                    f"T1 target={setpoint_id} R{repeat:02d} "
                    f"progress={100.0 * elapsed / max(duration, 1e-9):5.1f}% "
                    f"t={elapsed:.1f}/{duration:.1f}s target=[{target}] m yaw={target_yaw:+.1f} deg "
                    f"position={position} status={health}"
                )
                self.get_logger().info(message)
                if health != "OK" and health != self._last_runtime_health:
                    self.get_logger().warning(f"T1 runtime state anomaly: {message}")
                elif health == "OK" and self._last_runtime_health not in {None, "OK"}:
                    self.get_logger().info("T1 runtime state recovered")
                self._last_runtime_health = health
                next_log = now + interval
            rclpy.spin_once(self, timeout_sec=min(0.1, max(deadline - time.monotonic(), 0.01)))

        position, health = self._runtime_health(
            pose_stale_after_sec=pose_stale_after_sec,
            allowed_vision_modes=allowed_vision_modes,
        )
        self.get_logger().info(
            f"T1 target={setpoint_id} R{repeat:02d} progress=100.0% "
            f"t={duration:.1f}/{duration:.1f}s position={position} status={health}"
        )

    def _spin_sleep(self, duration_sec: float) -> None:
        deadline = time.monotonic() + max(float(duration_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=min(0.1, max(deadline - time.monotonic(), 0.01)))

    def publish_event(self, session_id: str, event: str, label: str, metadata: dict[str, Any]) -> None:
        message = TrajectoryEvent()
        message.session_id = session_id
        message.event = event
        message.label = label
        message.metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
        for _ in range(3):
            message.header.stamp = self.get_clock().now().to_msg()
            self._event_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)

    def publish_goal(self, x_m: float, y_m: float, z_m: float, yaw_deg: float) -> None:
        """Publish a controller-world target using the controller Y-up yaw.

        The real-state adapter also presents ``controller_world`` to the
        motion controller, where yaw is defined about +Y.  Keep this wire
        format aligned with ``send_position_goal`` rather than using a ROS
        pool/world Z-yaw quaternion on a controller-world command topic.
        """
        message = PoseStamped()
        message.header.frame_id = "controller_world"
        message.pose.position.x = float(x_m)
        message.pose.position.y = float(y_m)
        message.pose.position.z = float(z_m)
        half_yaw = 0.5 * float(yaw_deg) * 3.141592653589793 / 180.0
        message.pose.orientation.y = float(math.sin(half_yaw))
        message.pose.orientation.w = float(math.cos(half_yaw))
        for _ in range(5):
            message.header.stamp = self.get_clock().now().to_msg()
            self._goal_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)

    def cancel_goal(self) -> None:
        message = Empty()
        for _ in range(5):
            self._cancel_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.1)

    def set_bridge_enabled(
        self,
        *,
        enabled: bool,
        parameter_timeout_sec: float,
        confirmation_timeout_sec: float,
    ) -> dict[str, Any]:
        """Set bridge arming state and wait for a fresh status confirmation."""

        requested_enabled = bool(enabled)
        result: dict[str, Any] = {
            "requested_enabled": requested_enabled,
            "bridge_parameter_acknowledged": False,
            "bridge_enabled_confirmed": False,
        }
        status_sequence_before_request = self._bridge_status_sequence
        try:
            if not self._bridge_set_parameters_client.wait_for_service(
                timeout_sec=max(float(parameter_timeout_sec), 0.0)
            ):
                result["bridge_parameter_error"] = (
                    f"service unavailable: {self._bridge_parameter_service}"
                )
                return result
            request = SetParameters.Request()
            request.parameters = [
                Parameter("enabled", value=requested_enabled).to_parameter_msg()
            ]
            future = self._bridge_set_parameters_client.call_async(request)
            deadline = time.monotonic() + max(float(parameter_timeout_sec), 0.0)
            while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.05, max(deadline - time.monotonic(), 0.0)))
            if future.done():
                response = future.result()
                result["bridge_parameter_acknowledged"] = bool(
                    response
                    and response.results
                    and all(item.successful for item in response.results)
                )
            else:
                result["bridge_parameter_error"] = "set_parameters timed out"
        except Exception as exc:  # pragma: no cover - hardware service failure is environment-specific.
            result["bridge_parameter_error"] = repr(exc)

        deadline = time.monotonic() + max(float(confirmation_timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if requested_enabled and _bridge_enabled_confirmed(
                self._bridge_status,
                status_sequence=self._bridge_status_sequence,
                status_sequence_before_request=status_sequence_before_request,
            ):
                result["bridge_enabled_confirmed"] = True
                break
        result["bridge_status"] = self._bridge_status
        result["bridge_status_sequence_after_request"] = self._bridge_status_sequence
        return result

    def stop_motion(
        self,
        *,
        zero_frames: int,
        zero_frame_interval_sec: float,
    ) -> dict[str, Any]:
        """Cancel controller output and send zero frames without changing arming."""

        result: dict[str, Any] = {
            "zero_frames_requested": max(int(zero_frames), 1),
            "bridge_enabled_preserved": True,
        }
        cancel = Empty()
        zero = Float32MultiArray()
        zero.data = [0.0] * 8
        for _ in range(result["zero_frames_requested"]):
            self._cancel_pub.publish(cancel)
            self._zero_thruster_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.0)
            time.sleep(max(float(zero_frame_interval_sec), 0.0))
        result["bridge_status"] = self._bridge_status
        result["bridge_status_sequence_after_stop"] = self._bridge_status_sequence
        return result

    def emergency_stop(
        self,
        *,
        zero_frames: int,
        zero_frame_interval_sec: float,
        parameter_timeout_sec: float,
        confirmation_timeout_sec: float,
    ) -> dict[str, Any]:
        """Immediately command and verify a disabled zero-thrust bridge state."""

        result: dict[str, Any] = {
            "zero_frames_requested": max(int(zero_frames), 1),
            "bridge_parameter_acknowledged": False,
            "bridge_disabled_confirmed": False,
        }
        # Do this before killing the controller: otherwise its final non-zero
        # frame could remain in the bridge or MCU until a watchdog expires.
        result.update(self.stop_motion(
            zero_frames=result["zero_frames_requested"],
            zero_frame_interval_sec=zero_frame_interval_sec,
        ))
        result["bridge_enabled_preserved"] = False
        zero = Float32MultiArray()
        zero.data = [0.0] * 8

        # Only accept an MCU confirmation delivered after this stop was
        # requested.  The bridge status contains firmware diagnostics, rather
        # than merely the ROS-side `enabled` parameter.
        status_sequence_before_disarm = self._bridge_status_sequence
        try:
            if self._bridge_set_parameters_client.wait_for_service(
                timeout_sec=max(float(parameter_timeout_sec), 0.0)
            ):
                request = SetParameters.Request()
                request.parameters = [Parameter("enabled", value=False).to_parameter_msg()]
                future = self._bridge_set_parameters_client.call_async(request)
                deadline = time.monotonic() + max(float(parameter_timeout_sec), 0.0)
                while rclpy.ok() and not future.done() and time.monotonic() < deadline:
                    rclpy.spin_once(self, timeout_sec=min(0.05, max(deadline - time.monotonic(), 0.0)))
                if future.done():
                    response = future.result()
                    result["bridge_parameter_acknowledged"] = bool(
                        response
                        and response.results
                        and all(item.successful for item in response.results)
                    )
            else:
                result["bridge_parameter_error"] = (
                    f"service unavailable: {self._bridge_parameter_service}"
                )
        except Exception as exc:  # pragma: no cover - hardware service failure is environment-specific.
            result["bridge_parameter_error"] = repr(exc)

        deadline = time.monotonic() + max(float(confirmation_timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            self._zero_thruster_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.05)
            status = self._bridge_status
            if _mcu_disarm_confirmed(
                status,
                status_sequence=self._bridge_status_sequence,
                status_sequence_before_disarm=status_sequence_before_disarm,
            ):
                result["bridge_disabled_confirmed"] = True
                break

        result["bridge_status"] = self._bridge_status
        result["bridge_status_sequence_after_stop"] = self._bridge_status_sequence
        return result

def _topic_list() -> set[str]:
    completed = subprocess.run(["ros2", "topic", "list"], check=False, capture_output=True, text=True)
    if completed.returncode != 0:
        return set()
    return {line.strip() for line in completed.stdout.splitlines() if line.strip()}


def _wait_for_topics(required: Sequence[str], timeout_sec: float) -> None:
    required_set = {str(item) for item in required if str(item).strip()}
    if not required_set:
        return
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        available = _topic_list()
        if required_set.issubset(available):
            return
        time.sleep(0.5)
    missing = sorted(required_set - _topic_list())
    raise RuntimeError(f"required ROS topics did not appear before timeout: {missing}")


def _launch_commands(
    config: dict[str, Any],
    controller_name: str,
    arm: bool,
    repo_root: Path,
    *,
    state_adapter_input_pose_topic: str = "",
    state_adapter_input_imu_topic: str = "",
    state_adapter_input_depth_topic: str = "",
    state_adapter_input_dvl_topic: str = "",
    state_adapter_input_status_topic: str = "",
    evaluation_clean_pose_topic: str = "",
) -> list[tuple[str, list[str]]]:
    runner = _as_mapping(config["runner"], "runner")
    auto = _as_mapping(runner.get("auto_launch", {}), "runner.auto_launch")
    bridge = _as_mapping(auto.get("hardware_bridge", {}), "runner.auto_launch.hardware_bridge")
    perception = _as_mapping(auto.get("perception", {}), "runner.auto_launch.perception")
    fusion = _as_mapping(auto.get("state_fusion", {}), "runner.auto_launch.state_fusion")
    motion = _as_mapping(auto.get("motion_controller", {}), "runner.auto_launch.motion_controller")
    controllers = _as_mapping(config["controllers"], "controllers")
    selected = _as_mapping(controllers.get(controller_name), f"controllers.{controller_name}")
    commands: list[tuple[str, list[str]]] = []

    bridge_enabled = bool(bridge.get("enabled_with_arm", True)) if arm else bool(bridge.get("enabled_without_arm", False))
    bridge_cmd = [
        "ros2", "launch", str(bridge.get("package", "hardware_bridge")),
        str(bridge.get("launch_file", "hardware_bridge.launch.py")),
        f"params_file:={_resolve_path(str(bridge['params_file']), repo_root)}",
        f"enabled:={'true' if bridge_enabled else 'false'}",
    ]
    commands.append(("hardware_bridge", bridge_cmd))

    perception_cmd = [
        "ros2", "launch", str(perception.get("package", "perception")),
        str(perception.get("launch_file", "refractive_apriltag.launch.py")),
        f"detector_params:={_resolve_path(str(perception['detector_params']), repo_root)}",
        f"refractive_params:={_resolve_path(str(perception['refractive_params']), repo_root)}",
        f"pool_world_params_file:={_resolve_path(str(auto['motion_controller']['pool_world_params_file']), repo_root)}",
    ]
    commands.append(("perception", perception_cmd))

    fusion_cmd = [
        "ros2", "launch", str(fusion.get("package", "state_estimation")),
        str(fusion.get("launch_file", "state_fusion.launch.py")),
        f"params_file:={_resolve_path(str(fusion['params_file']), repo_root)}",
        f"pool_world_params_file:={_resolve_path(str(auto['motion_controller']['pool_world_params_file']), repo_root)}",
    ]
    commands.append(("state_fusion", fusion_cmd))

    motion_cmd = [
        "ros2", "launch", str(motion.get("package", "motion_control")),
        str(motion.get("launch_file", "motion_controller.launch.py")),
        f"params_file:={_resolve_path(str(selected['config_file']), repo_root)}",
        f"pool_world_params_file:={_resolve_path(str(motion['pool_world_params_file']), repo_root)}",
        f"state_input_mode:={str(motion.get('state_input_mode', 'raw_fusion'))}",
    ]
    # A profile can reuse the common launch file while replacing the final
    # controller executable.  This keeps controller_state_adapter startup and
    # the runner's process lifecycle intact for the hybrid PPO+depth-PID node.
    controller_executable = str(selected.get("controller_executable", "motion_controller")).strip()
    controller_node_name = str(selected.get("controller_node_name", "motion_controller")).strip()
    if not controller_executable or not controller_node_name:
        raise RuntimeError(
            f"controllers.{controller_name} controller_executable and controller_node_name must be non-empty"
        )
    motion_cmd.extend(
        [
            f"controller_executable:={controller_executable}",
            f"motion_controller_node_name:={controller_node_name}",
        ]
    )
    checkpoint_value = selected.get("checkpoint_path")
    if checkpoint_value:
        motion_cmd.append(f"checkpoint_path:={_resolve_path(str(checkpoint_value), repo_root)}")
    # These overrides are used only by the localization-noise ablation.  The
    # controller consumes the perturbed state while a second, independent
    # adapter publishes a clean controller-world pose for offline metrics.
    if state_adapter_input_pose_topic:
        motion_cmd.append(f"state_adapter_input_pose_topic:={state_adapter_input_pose_topic}")
    if state_adapter_input_imu_topic:
        motion_cmd.append(f"state_adapter_input_imu_topic:={state_adapter_input_imu_topic}")
    if state_adapter_input_depth_topic:
        motion_cmd.append(f"state_adapter_input_depth_topic:={state_adapter_input_depth_topic}")
    if state_adapter_input_dvl_topic:
        motion_cmd.append(f"state_adapter_input_dvl_topic:={state_adapter_input_dvl_topic}")
    if state_adapter_input_status_topic:
        motion_cmd.append(f"state_adapter_input_status_topic:={state_adapter_input_status_topic}")
    if evaluation_clean_pose_topic:
        motion_cmd.append(f"evaluation_clean_pose_topic:={evaluation_clean_pose_topic}")
    commands.append((f"motion_controller_{controller_name}", motion_cmd))
    return commands


def _config_files(config: dict[str, Any], config_path: Path, repo_root: Path) -> list[Path]:
    files: list[Path] = [config_path]
    recording = _as_mapping(config.get("recording", {}), "recording")
    values = list(recording.get("required_config_files", []))
    controllers = _as_mapping(config["controllers"], "controllers")
    for item in controllers.values():
        if isinstance(item, dict) and item.get("config_file"):
            values.append(item["config_file"])
    runner = _as_mapping(config["runner"], "runner")
    auto = _as_mapping(runner.get("auto_launch", {}), "runner.auto_launch")
    for section_name in ("hardware_bridge", "perception", "state_fusion", "motion_controller"):
        section = auto.get(section_name, {})
        if isinstance(section, dict):
            for key in ("params_file", "detector_params", "refractive_params", "pool_world_params_file"):
                if section.get(key):
                    values.append(section[key])
    seen: set[Path] = set()
    result: list[Path] = []
    for value in values:
        path = _resolve_path(str(value), repo_root)
        if path not in seen:
            seen.add(path)
            result.append(path)
    return result


def _record_command(
    *,
    experiment_id: str,
    strategy: str,
    session_id: str,
    output_root: Path,
    profile: str,
    metadata: dict[str, Any],
    config_files: Sequence[Path],
    checkpoint: Path | None,
    prepared_session: bool = False,
    defer_analysis: bool = False,
) -> list[str]:
    command = [
        "ros2", "run", "experiment_recorder", "record_experiment",
        "--experiment-id", experiment_id,
        "--strategy", strategy,
        "--profile", profile,
        "--session-id", session_id,
        "--output-root", str(output_root),
        "--duration", "0",
        "--metadata-json", json.dumps(metadata, ensure_ascii=True, sort_keys=True),
    ]
    for path in config_files:
        command.extend(["--config", str(path)])
    if checkpoint is not None:
        command.extend(["--checkpoint", str(checkpoint)])
    if prepared_session:
        command.append("--prepared-session")
    if defer_analysis:
        command.append("--no-auto-analysis")
    return command


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the complete T1 hardware experiment from one recorder YAML."
    )
    parser.add_argument("--config", required=True, type=Path, help="T1 YAML under experiment_recorder/config.")
    parser.add_argument(
        "--controller",
        choices=("PID", "PPO", "PPO_HYBRID"),
        default=None,
        help="Run one registered controller: PID, standalone PPO, or PPO_HYBRID.",
    )
    parser.add_argument(
        "--controller-config",
        type=Path,
        required=True,
        help="The single motion-control YAML profile used for this run.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Optional PPO checkpoint override; otherwise read checkpoint_path from the controller YAML.",
    )
    parser.add_argument("--arm", action="store_true", help="Explicitly allow hardware_bridge enabled=true.")
    parser.add_argument(
        "--state-adapter-input-pose-topic",
        default="",
        help="Optional pose topic used only by the controller state adapter (for localization-noise ablations).",
    )
    parser.add_argument(
        "--state-adapter-input-imu-topic",
        default="",
        help="Optional IMU topic used only by the controller state adapter (for localization-noise ablations).",
    )
    parser.add_argument(
        "--state-adapter-input-depth-topic",
        default="",
        help="Optional depth topic used only by the controller state adapter (for full state-estimation ablations).",
    )
    parser.add_argument(
        "--state-adapter-input-dvl-topic",
        default="",
        help="Optional DVL/velocity topic used only by the controller state adapter (for full state-estimation ablations).",
    )
    parser.add_argument(
        "--state-adapter-input-status-topic",
        default="",
        help="Optional state-health topic used only by the controller state adapter (for full state-estimation ablations).",
    )
    parser.add_argument(
        "--localization-noise-config",
        type=Path,
        default=None,
        help="Noise-injector YAML to hash and register; requires at least one noisy adapter input topic.",
    )
    parser.add_argument(
        "--no-ekf-state-config",
        type=Path,
        default=None,
        help=(
            "Direct-measurement state YAML to hash and register. Requires all five "
            "--state-adapter-input-*-topic overrides and keeps normal EKF state for evaluation."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the generated run plan without launching ROS2 or hardware.")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = _repo_root()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    state_adapter_input_pose_topic = str(args.state_adapter_input_pose_topic).strip()
    state_adapter_input_imu_topic = str(args.state_adapter_input_imu_topic).strip()
    state_adapter_input_depth_topic = str(args.state_adapter_input_depth_topic).strip()
    state_adapter_input_dvl_topic = str(args.state_adapter_input_dvl_topic).strip()
    state_adapter_input_status_topic = str(args.state_adapter_input_status_topic).strip()
    state_adapter_input_topics = {
        "pose": state_adapter_input_pose_topic,
        "imu": state_adapter_input_imu_topic,
        "depth": state_adapter_input_depth_topic,
        "dvl": state_adapter_input_dvl_topic,
        "status": state_adapter_input_status_topic,
    }
    has_noisy_adapter_input = bool(
        state_adapter_input_pose_topic or state_adapter_input_imu_topic
    )
    if (
        args.no_ekf_state_config is None
        and has_noisy_adapter_input != (args.localization_noise_config is not None)
    ):
        raise SystemExit(
            "a localization-noise ablation requires both --localization-noise-config "
            "and at least one --state-adapter-input-*-topic override"
        )
    localization_noise_config: Path | None = None
    if args.localization_noise_config is not None:
        localization_noise_config = _resolve_cli_path(
            args.localization_noise_config, repo_root
        )
        if not localization_noise_config.is_file():
            raise SystemExit(
                f"--localization-noise-config does not exist: {localization_noise_config}"
            )
    no_ekf_state_config: Path | None = None
    if args.no_ekf_state_config is not None:
        if args.localization_noise_config is not None:
            raise SystemExit("--no-ekf-state-config and --localization-noise-config are mutually exclusive")
        if not all(state_adapter_input_topics.values()):
            missing = [name for name, topic in state_adapter_input_topics.items() if not topic]
            raise SystemExit(
                "--no-ekf-state-config requires all five controller state-input overrides; "
                f"missing: {', '.join(missing)}"
            )
        no_ekf_state_config = _resolve_cli_path(args.no_ekf_state_config, repo_root)
        if not no_ekf_state_config.is_file():
            raise SystemExit(f"--no-ekf-state-config does not exist: {no_ekf_state_config}")
    evaluation_clean_pose_topic = (
        "/finsrov/evaluation/pose"
        if localization_noise_config is not None or no_ekf_state_config is not None
        else ""
    )
    selected_controller = args.controller
    if args.controller_config is not None:
        controller_config = _resolve_cli_path(args.controller_config, repo_root)
        if not controller_config.is_file():
            raise SystemExit(f"--controller-config does not exist: {controller_config}")
        selected_controller = selected_controller or _infer_controller_name(controller_config)
        selected = _as_mapping(
            config["controllers"].get(selected_controller),
            f"controllers.{selected_controller}",
        )
        selected = copy.deepcopy(selected)
        selected["config_file"] = str(controller_config)
        _validate_controller_profile_runtime(selected, controller_config)
        selected_checkpoint = _controller_checkpoint(controller_config, repo_root)
        if args.checkpoint is not None:
            selected_checkpoint = _resolve_cli_path(args.checkpoint, repo_root)
        selected["checkpoint_path"] = str(selected_checkpoint) if selected_checkpoint is not None else None
        config["controllers"][selected_controller] = selected
    elif args.checkpoint is not None:
        if selected_controller is None:
            raise SystemExit("--checkpoint requires --controller or --controller-config")
        selected = copy.deepcopy(
            _as_mapping(config["controllers"].get(selected_controller), f"controllers.{selected_controller}")
        )
        selected["checkpoint_path"] = str(_resolve_cli_path(args.checkpoint, repo_root))
        config["controllers"][selected_controller] = selected
    elif selected_controller is not None:
        _as_mapping(config["controllers"].get(selected_controller), f"controllers.{selected_controller}")
    if selected_controller is not None:
        config["runner"]["selected_controller"] = selected_controller
    runner = _as_mapping(config["runner"], "runner")
    raw_session_strategy_slugs = runner.get("session_strategy_slugs", {})
    session_strategy_slugs = _as_mapping(raw_session_strategy_slugs, "runner.session_strategy_slugs")
    safety = _as_mapping(runner.get("safety", {}), "runner.safety")
    auto = _as_mapping(runner.get("auto_launch", {}), "runner.auto_launch")
    if not bool(auto.get("enabled", True)):
        raise SystemExit("automatic runner requires runner.auto_launch.enabled=true")
    # The lab normally keeps hardware_bridge, AprilTag perception and state
    # fusion alive across controller runs.  In that mode the runner verifies
    # their topics but does not spawn or stop the support processes.
    start_support_stack = bool(auto.get("start_support_stack", True))
    if bool(runner.get("require_explicit_arm_flag", True)) and not args.arm and not args.dry_run:
        raise SystemExit("formal T1 hardware execution requires explicit --arm; use --dry-run to inspect the plan")

    logging_cfg = _as_mapping(runner.get("logging", {}), "runner.logging")
    data_root = _resolve_path(str(logging_cfg.get("root", "ros2_ws/data/experiments")), repo_root)
    controllers = _as_mapping(config["controllers"], "controllers")
    selected_strategy = "UNSPECIFIED"
    selected_session_strategy = "UNSPECIFIED"
    if selected_controller is not None:
        selected_profile = _as_mapping(controllers.get(selected_controller), f"controllers.{selected_controller}")
        selected_strategy = _strategy_name_from_config(selected_profile.get("config_file", selected_controller))
        selected_session_strategy = _session_strategy_slug(selected_strategy, session_strategy_slugs)
        method_bucket = _controller_storage_subdirectory(selected_controller, selected_profile)
    else:  # guarded below, retained for type-checkable dry-run construction.
        method_bucket = "UNSPECIFIED"
    prefix = str(config.get("experiment_id", "T1")).replace(" ", "_")
    if selected_controller is not None:
        prefix = f"{prefix}_{selected_strategy}"
    run_parent = canonical_run_dir(
        data_root, domain="hardware", task="T1", method=method_bucket, run_id="placeholder"
    ).parent
    run_id = _new_run_id(run_parent, prefix)
    run_dir = run_parent / run_id
    trials_dir = run_dir / "trials"
    protocol = _as_mapping(config["trial_protocol"], "trial_protocol")
    autoreset_cfg = _as_mapping(config.get("autoreset", {}), "autoreset")
    autoreset_enabled = bool(autoreset_cfg.get("enabled", False))
    autoreset_continuous_handoff = bool(autoreset_cfg.get("continuous_handoff", False))
    autoreset_persist_across_trials = bool(autoreset_cfg.get("persist_across_trials", False))
    defer_trial_analysis = bool(autoreset_cfg.get("defer_analysis_until_run_complete", False))
    if autoreset_continuous_handoff and not autoreset_enabled:
        raise SystemExit("autoreset.continuous_handoff requires autoreset.enabled=true")
    if autoreset_persist_across_trials and not autoreset_continuous_handoff:
        raise SystemExit("autoreset.persist_across_trials requires autoreset.continuous_handoff=true")
    if defer_trial_analysis and not autoreset_persist_across_trials:
        raise SystemExit(
            "autoreset.defer_analysis_until_run_complete requires autoreset.persist_across_trials=true"
        )
    autoreset_controller_name = str(autoreset_cfg.get("controller", "")).strip()
    autoreset_profile: dict[str, Any] | None = None
    autoreset_checkpoint: Path | None = None
    if autoreset_enabled:
        if not autoreset_controller_name:
            raise SystemExit("autoreset.enabled requires autoreset.controller")
        autoreset_profile = _as_mapping(
            config["controllers"].get(autoreset_controller_name),
            f"controllers.{autoreset_controller_name}",
        )
        autoreset_config_path = _resolve_path(str(autoreset_profile.get("config_file", "")), repo_root)
        if not autoreset_config_path.is_file():
            raise SystemExit(f"autoreset controller YAML does not exist: {autoreset_config_path}")
        _validate_controller_profile_runtime(autoreset_profile, autoreset_config_path)
        autoreset_checkpoint = _controller_checkpoint(autoreset_config_path, repo_root)
        if autoreset_checkpoint is None or not autoreset_checkpoint.is_file():
            raise SystemExit(
                "autoreset controller must declare an existing PPO checkpoint_path; "
                f"got {autoreset_checkpoint}"
            )
        target_value = autoreset_cfg.get("target_controller_world")
        if not isinstance(target_value, list) or len(target_value) != 4:
            raise SystemExit("autoreset.target_controller_world must be [x, y, z, yaw_deg]")
        if float(autoreset_cfg.get("timeout_sec", 0.0)) <= 0.0:
            raise SystemExit("autoreset.timeout_sec must be positive")
    target_cfg = _as_mapping(config["target_points"], "target_points")
    points = _as_mapping(target_cfg.get("points"), "target_points.points")
    setpoints = [str(item) for item in protocol.get("setpoints", points.keys())]
    for setpoint in setpoints:
        if setpoint not in points:
            raise SystemExit(f"trial_protocol references unknown setpoint `{setpoint}`")
    repeats = int(protocol.get("repeats_per_controller_and_setpoint", 1))
    if selected_controller is None:
        raise SystemExit("one controller must be selected with --controller-config")
    if autoreset_continuous_handoff:
        selected_profile = _as_mapping(
            controllers.get(selected_controller), f"controllers.{selected_controller}"
        )
        selected_checkpoint_value = selected_profile.get("checkpoint_path")
        if not selected_checkpoint_value:
            raise SystemExit(
                "autoreset.continuous_handoff requires the selected controller to declare checkpoint_path"
            )
        selected_checkpoint_for_handoff = _resolve_path(str(selected_checkpoint_value), repo_root)
        if autoreset_checkpoint is None or selected_checkpoint_for_handoff != autoreset_checkpoint:
            raise SystemExit(
                "autoreset.continuous_handoff requires the selected controller and AUTO_RESET_PPO "
                "to use the identical checkpoint_path"
            )
    config_files = _config_files(config, config_path, repo_root)
    if localization_noise_config is not None and localization_noise_config not in config_files:
        config_files.append(localization_noise_config)
    if no_ekf_state_config is not None and no_ekf_state_config not in config_files:
        config_files.append(no_ekf_state_config)
    missing = [str(path) for path in config_files if not path.is_file()]
    if missing:
        raise SystemExit(f"configured files do not exist: {missing}")

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "planned" if args.dry_run else "running",
        "test_id": run_id,
        "run_id": run_id,
        "created_at": now_iso(),
        "config": str(config_path),
        "config_sha256": file_sha256(config_path),
        "git_revision": git_revision(repo_root),
        "armed": bool(args.arm),
        "run_directory": str(run_dir.relative_to(repo_root)),
        "experiment_layout": {"domain": "hardware", "task": "T1", "method": method_bucket},
        "config_files": {str(path): file_sha256(path) for path in config_files},
        "trial_plan": [],
        "autoreset": {
            "enabled": autoreset_enabled,
            "controller": autoreset_controller_name if autoreset_enabled else None,
            "controller_config": str(autoreset_profile.get("config_file")) if autoreset_profile else None,
            "checkpoint": str(autoreset_checkpoint) if autoreset_checkpoint else None,
            "checkpoint_sha256": file_sha256(autoreset_checkpoint) if autoreset_checkpoint else None,
            "target_controller_world": list(autoreset_cfg.get("target_controller_world", [])),
            "action_mask": {"roll": 0.0, "pitch": 0.0},
            "continuous_handoff": autoreset_continuous_handoff,
            "persist_across_trials": autoreset_persist_across_trials,
            "analysis_mode": "deferred_until_run_complete" if defer_trial_analysis else "per_trial",
            "runtime_controller": (
                selected_controller if autoreset_continuous_handoff else autoreset_controller_name
            ),
            "runtime_controller_config": (
                str(
                    _as_mapping(
                        controllers[selected_controller], f"controllers.{selected_controller}"
                    ).get("config_file")
                )
                if autoreset_continuous_handoff
                else str(autoreset_profile.get("config_file")) if autoreset_profile else None
            ),
            "tilt_guard_enabled": bool(autoreset_cfg.get("tilt_guard_enabled", True)),
            "enforce_state_health": bool(autoreset_cfg.get("enforce_state_health", True)),
        },
        "support_stack": {
            "mode": "launched_by_runner" if start_support_stack else "pre_started",
            "started_by_runner": start_support_stack,
            "required_topics": [
                str(item) for item in auto.get(
                    "required_support_topics", auto.get("required_topics", [])
                )
            ],
        },
    }
    if localization_noise_config is not None:
        manifest["localization_noise_ablation"] = {
            "enabled": True,
            "config_file": str(localization_noise_config),
            "config_sha256": file_sha256(localization_noise_config),
            "state_adapter_input_pose_topic": state_adapter_input_pose_topic or None,
            "state_adapter_input_imu_topic": state_adapter_input_imu_topic or None,
            "evaluation_pose_topic": evaluation_clean_pose_topic,
            "metric_source": "clean_raw_fusion_controller_world",
        }
        required_topics = manifest["support_stack"]["required_topics"]
        for topic in (state_adapter_input_pose_topic, state_adapter_input_imu_topic):
            if topic and topic not in required_topics:
                required_topics.append(topic)
    if no_ekf_state_config is not None:
        manifest["state_estimation_ablation"] = {
            "kind": "direct_measurement_no_ekf",
            "enabled": True,
            "config_file": str(no_ekf_state_config),
            "config_sha256": file_sha256(no_ekf_state_config),
            "state_adapter_input_topics": state_adapter_input_topics,
            "evaluation_pose_topic": evaluation_clean_pose_topic,
            "metric_source": "normal_ekf_controller_world",
            "controller_input_source": "direct_measurement_no_ekf",
        }
        required_topics = manifest["support_stack"]["required_topics"]
        for topic in state_adapter_input_topics.values():
            if topic not in required_topics:
                required_topics.append(topic)
    if selected_controller is not None:
        selected = _as_mapping(controllers[selected_controller], f"controllers.{selected_controller}")
        manifest["controller_selection"] = {
            "controller": selected_controller,
            "strategy": selected_strategy,
            "session_strategy": selected_session_strategy,
            "config_file": str(_resolve_path(str(selected["config_file"]), repo_root)),
            "checkpoint": selected.get("checkpoint_path"),
            "source": "cli_override" if args.controller_config is not None else "t1_yaml_profile",
        }

    planned: list[dict[str, Any]] = []
    for repeat_index in range(1, repeats + 1):
        pair = [selected_controller]
        for setpoint_id in setpoints:
            target = _as_mapping(points[setpoint_id], f"target_points.points.{setpoint_id}")
            for controller_name in pair:
                selected = _as_mapping(controllers.get(controller_name), f"controllers.{controller_name}")
                checkpoint_value = selected.get("checkpoint_path")
                checkpoint = _resolve_path(str(checkpoint_value), repo_root) if checkpoint_value else None
                if checkpoint is not None and not checkpoint.is_file():
                    raise SystemExit(f"controllers.{controller_name}.checkpoint_path does not exist: {checkpoint}")
                if bool(selected.get("checkpoint_required", False)) and not checkpoint and not args.dry_run:
                    raise SystemExit(
                        f"controllers.{controller_name}.checkpoint_path must be set before formal execution"
                    )
                planned.append({
                    "controller": controller_name,
                    "strategy": _strategy_name_from_config(selected.get("config_file", controller_name)),
                    "session_strategy": _session_strategy_slug(
                        _strategy_name_from_config(selected.get("config_file", controller_name)),
                        session_strategy_slugs,
                    ),
                    "setpoint_id": setpoint_id,
                    "repeat": repeat_index,
                    "target_controller_world": [
                        float(target["x_m"]), float(target["y_depth_m"]),
                        float(target["z_m"]), float(target["yaw_deg"]),
                    ],
                    "checkpoint": str(checkpoint) if checkpoint else None,
                })
    manifest["trial_plan"] = planned
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        for item in planned:
            print(
                f"{item['controller']} {item['setpoint_id']} repeat={item['repeat']} "
                f"session_strategy={item['session_strategy']} target={item['target_controller_world']}"
            )
        return

    run_layout = create_run_layout(
        data_root, domain="hardware", task="T1", method=method_bucket, run_id=run_id
    )
    run_dir, trials_dir = run_layout.run_dir, run_layout.trials_dir
    run_derived_dir = run_layout.derived_dir
    run_target_plot_dir = run_derived_dir / "target_plots"
    run_target_plot_dir.mkdir(parents=True)
    write_json(run_dir / "run_manifest.json", manifest)
    log_stage(
        f"T1 run initialized: controller={selected_controller} trials={len(planned)} run={run_dir}"
    )

    startup_timeout = float(auto.get("startup_timeout_sec", 30.0))
    startup_settle = float(auto.get("startup_settle_sec", 3.0))
    terminate_timeout = float(auto.get("terminate_timeout_sec", 10.0))
    runtime_status = _as_mapping(runner.get("runtime_status", {}), "runner.runtime_status")
    runtime_status_log_interval = float(runtime_status.get("log_interval_sec", 1.0))
    pose_stale_after = float(runtime_status.get("pose_stale_after_sec", 0.5))
    allowed_vision_modes_raw = runtime_status.get("allowed_vision_modes", ["fresh", "coast"])
    if not isinstance(allowed_vision_modes_raw, list) or not all(
        isinstance(item, str) and item.strip() for item in allowed_vision_modes_raw
    ):
        raise SystemExit("runner.runtime_status.allowed_vision_modes must be a list of non-empty strings")
    allowed_vision_modes = {item.strip() for item in allowed_vision_modes_raw}
    if runtime_status_log_interval <= 0.0 or pose_stale_after <= 0.0:
        raise SystemExit("runner.runtime_status log_interval_sec and pose_stale_after_sec must be positive")
    required_support_topics = [str(item) for item in auto.get("required_support_topics", auto.get("required_topics", []))]
    required_controller_topics = [str(item) for item in auto.get("required_controller_topics", [])]
    for topic in state_adapter_input_topics.values():
        if topic and topic not in required_support_topics:
            required_support_topics.append(topic)
    if evaluation_clean_pose_topic and evaluation_clean_pose_topic not in required_controller_topics:
        required_controller_topics.append(evaluation_clean_pose_topic)
    profile = str(logging_cfg.get("recorder_profile", "t1"))
    event_topic = str(logging_cfg.get("event_topic", "/finsrov/experiment/event"))
    trial_protocol = _as_mapping(config["trial_protocol"], "trial_protocol")
    metric_protocol = _as_mapping(config["metric_protocol"], "metric_protocol")
    primary_window = _as_mapping(metric_protocol.get("evaluation_window", {}), "metric_protocol.evaluation_window")
    response_window = _as_mapping(metric_protocol.get("response_window", {}), "metric_protocol.response_window")
    pre_goal = float(trial_protocol.get("pre_goal_record_sec", 5.0))
    control_duration = float(trial_protocol.get("control_duration_sec", 60.0))
    post_goal = float(trial_protocol.get("post_goal_record_sec", 5.0))
    support_processes: list[_ManagedProcess] = []
    shared_log_files: list[Path] = []
    node: _RosExperimentNode | None = None
    persistent_controller_process: _ManagedProcess | None = None
    deferred_trials: list[tuple[str, Any]] = []
    emergency_stop_done = False

    def emergency_shutdown(reason: str) -> None:
        """Perform the hardware stop sequence once, before forced child cleanup."""

        nonlocal emergency_stop_done
        if emergency_stop_done:
            return
        emergency_stop_done = True
        record: dict[str, Any] = {"reason": reason, "requested_at": now_iso()}
        if node is None:
            record["status"] = "node_unavailable"
        else:
            safety_cfg = _as_mapping(safety.get("emergency_stop", {}), "runner.safety.emergency_stop")
            try:
                record.update(
                    node.emergency_stop(
                        zero_frames=int(safety_cfg.get("zero_frames", 6)),
                        zero_frame_interval_sec=float(safety_cfg.get("zero_frame_interval_sec", 0.05)),
                        parameter_timeout_sec=float(safety_cfg.get("parameter_timeout_sec", 1.0)),
                        confirmation_timeout_sec=float(safety_cfg.get("confirmation_timeout_sec", 2.0)),
                    )
                )
                record["status"] = "complete" if record.get("bridge_disabled_confirmed") else "unconfirmed"
            except Exception as exc:  # pragma: no cover - best-effort hardware emergency path.
                record["status"] = "error"
                record["error"] = repr(exc)
        manifest["emergency_stop"] = record

    def finalize_deferred_analysis() -> None:
        """Run the expensive per-bag analysis only after propulsion has stopped."""

        for session_id, trial_layout in deferred_trials:
            trial_manifest_path = trial_layout.trial_dir / "manifest.json"
            analysis_status = "needs_review"
            target_plot_files: list[str] = []
            try:
                report = analyze_session(trial_layout.trial_dir)
                analysis_status = str(report.get("status", "needs_review"))
                if trial_manifest_path.is_file():
                    trial_manifest = read_json(trial_manifest_path)
                    trial_manifest["analysis"] = {
                        "enabled": True,
                        "status": analysis_status,
                        "report": "derived/analysis_report.json",
                        "plots": report.get("plots", []),
                        "warnings": report.get("warnings", []),
                        "execution": "deferred_until_run_complete",
                    }
                    write_json(trial_manifest_path, trial_manifest)
                source_target_plot_dir = trial_layout.derived_dir / "plots" / "targets"
                if source_target_plot_dir.is_dir():
                    for source_plot in sorted(source_target_plot_dir.glob("*_xyzyaw.png")):
                        destination = run_target_plot_dir / f"{session_id}_{source_plot.name}"
                        shutil.copy2(source_plot, destination)
                        target_plot_files.append(str(destination.relative_to(run_dir)))
            except Exception as exc:  # pragma: no cover - analysis failure is best-effort.
                analysis_status = "needs_review"
                if trial_manifest_path.is_file():
                    trial_manifest = read_json(trial_manifest_path)
                    trial_manifest["analysis"] = {
                        "enabled": True,
                        "status": analysis_status,
                        "report": "derived/analysis_report.json",
                        "error": repr(exc),
                        "execution": "deferred_until_run_complete",
                    }
                    write_json(trial_manifest_path, trial_manifest)
            analysis_records = manifest.setdefault("analysis_reports", [])
            analysis_record = next(
                (item for item in analysis_records if item.get("session_id") == session_id), None
            )
            if analysis_record is None:
                analysis_record = {"session_id": session_id}
                analysis_records.append(analysis_record)
            analysis_record.update({
                "status": analysis_status,
                "report": str(trial_layout.derived_dir / "analysis_report.json"),
                "target_plots": target_plot_files,
                "execution": "deferred_until_run_complete",
            })
            log_stage(
                f"T1 deferred analysis complete: session={session_id} status={analysis_status} "
                f"plots={len(target_plot_files)}"
            )

    try:
        # Start fixed support services only when explicitly enabled.  The
        # canonical hardware protocol reuses a pre-started support stack.
        if start_support_stack:
            base_commands = _launch_commands(config, "PID", args.arm, repo_root)[:3]
            for name, command in base_commands:
                log_path = shared_log_path(run_layout, f"{name}.log")
                shared_log_files.append(log_path)
                support_processes.append(_ManagedProcess(command, log_path))
        _wait_for_topics(required_support_topics, startup_timeout)
        time.sleep(max(startup_settle, 0.0))
        init_experiment_ros()
        emergency_cfg = _as_mapping(safety.get("emergency_stop", {}), "runner.safety.emergency_stop")
        node = _RosExperimentNode(
            event_topic,
            bridge_node_name=str(emergency_cfg.get("bridge_node", "/hardware_bridge")),
            bridge_status_topic=str(emergency_cfg.get("bridge_status_topic", "/finsrov/hardware/status")),
        )
        log_stage(
            f"T1 support stack ready; beginning {len(planned)} target acquisitions",
            logger=node.get_logger(),
        )
        # ``--arm`` has one unambiguous meaning even when the bridge belongs
        # to an operator-managed support stack: explicitly set enabled=true
        # and require a fresh status confirmation before any controller is
        # launched.
        bridge_arming = {
            "requested": bool(args.arm),
            "status": "not_requested",
        }
        if args.arm:
            bridge_arming.update(node.set_bridge_enabled(
                enabled=True,
                parameter_timeout_sec=float(emergency_cfg.get("parameter_timeout_sec", 1.0)),
                confirmation_timeout_sec=float(emergency_cfg.get("confirmation_timeout_sec", 2.0)),
            ))
            bridge_arming["status"] = (
                "confirmed" if bridge_arming.get("bridge_enabled_confirmed") else "unconfirmed"
            )
        manifest["bridge_arming"] = bridge_arming
        write_json(run_dir / "run_manifest.json", manifest)
        if args.arm and not bridge_arming.get("bridge_enabled_confirmed"):
            raise RuntimeError("--arm requested but hardware bridge did not confirm enabled=true")

        def start_selected_controller(
            *, controller_name: str, log_path: Path, label: str
        ) -> _ManagedProcess:
            """Start the formal controller and wait for its actual control loop.

            The controller-state adapter comes up before the controller model,
            so topic discovery alone is insufficient here.  The following
            heartbeat gate proves that this process owns a running controller
            before it receives either the common-start or trial target.
            """

            node.clear_controller_preflight()
            heartbeat_sequence = node._goal_reached_sequence
            motion_command = _launch_commands(
                config,
                controller_name,
                args.arm,
                repo_root,
                state_adapter_input_pose_topic=state_adapter_input_pose_topic,
                state_adapter_input_imu_topic=state_adapter_input_imu_topic,
                state_adapter_input_depth_topic=state_adapter_input_depth_topic,
                state_adapter_input_dvl_topic=state_adapter_input_dvl_topic,
                state_adapter_input_status_topic=state_adapter_input_status_topic,
                evaluation_clean_pose_topic=evaluation_clean_pose_topic,
            )[3][1]
            process = _ManagedProcess(motion_command, log_path)
            _wait_for_topics(required_controller_topics, startup_timeout)
            time.sleep(max(startup_settle, 0.0))
            _require_process_running(process, f"{label} motion controller")
            node.wait_for_motion_controller_ready(
                reached_sequence_before_start=heartbeat_sequence,
                process=process,
                label=label,
                timeout_sec=startup_timeout,
            )
            return process

        if autoreset_persist_across_trials:
            persistent_controller_log = shared_log_path(run_layout, "continuous_handoff_controller.log")
            shared_log_files.append(persistent_controller_log)
            persistent_controller_process = start_selected_controller(
                controller_name=selected_controller,
                log_path=persistent_controller_log,
                label="persistent continuous-handoff",
            )
            manifest["autoreset"]["persistent_controller_log"] = str(
                persistent_controller_log.relative_to(run_dir)
            )
            write_json(run_dir / "run_manifest.json", manifest)

        for index, item in enumerate(planned, start=1):
            controller_name = str(item["controller"])
            strategy_name = str(item["strategy"])
            session_strategy_name = str(item["session_strategy"])
            setpoint_id = str(item["setpoint_id"])
            repeat_index = int(item["repeat"])
            selected = _as_mapping(controllers[controller_name], f"controllers.{controller_name}")
            checkpoint = Path(item["checkpoint"]) if item["checkpoint"] else None
            experiment_id = "E6" if controller_name == "PID" else "E7"
            session_id = f"{run_id}_{experiment_id}_{session_strategy_name}_{setpoint_id}_R{repeat_index:02d}"
            trial_layout = prepare_trial_layout(trials_dir, session_id)
            log_stage(
                f"T1 acquisition {index}/{len(planned)}: target={setpoint_id} R{repeat_index:02d}; "
                "preparing controller and recorder",
                logger=node.get_logger(),
            )
            metadata = {
                "test_id": run_id,
                "run_id": run_id,
                "controller": controller_name,
                "strategy": session_strategy_name,
                "controller_profile_stem": strategy_name,
                "session_strategy": session_strategy_name,
                "setpoint_id": setpoint_id,
                "repeat": repeat_index,
                "target_controller_world": item["target_controller_world"],
                "protocol_config": str(config_path),
                "controller_config": str(selected.get("config_file")),
                "trial_index": index,
                "control_duration_sec": control_duration,
                # Store the resolved metric windows in every session event.
                # The offline analyzer must not infer them from a mutable
                # protocol YAML after data collection has begun.
                "primary_evaluation_window": {
                    "start_event": str(primary_window.get("start_after_event", "hold_start")),
                    "start_offset_sec": float(primary_window.get("start_offset_sec", 0.0)),
                    "duration_sec": float(primary_window.get("duration_sec", control_duration)),
                    "resample_hz": float(primary_window.get("resample_hz", 0.0)),
                    "max_interpolation_gap_sec": float(primary_window.get("max_interpolation_gap_sec", 0.0)),
                    "minimum_valid_fraction": float(primary_window.get("minimum_valid_fraction", 0.0)),
                },
                "response_window": {
                    "start_event": str(response_window.get("start_after_event", "hold_start")),
                    "duration_sec": float(response_window.get("duration_sec", control_duration)),
                },
                "t1_success_definition": copy.deepcopy(
                    metric_protocol.get("success_definition", {})
                ),
                "localization_noise_ablation": manifest.get("localization_noise_ablation"),
                "state_estimation_ablation": manifest.get("state_estimation_ablation"),
                "autoreset": copy.deepcopy(manifest.get("autoreset", {})),
            }
            controller_process: _ManagedProcess | None = None
            autoreset_process: _ManagedProcess | None = None
            recorder_process: _ManagedProcess | None = None
            autoreset_phase_active = False
            try:
                recorder_command = _record_command(
                    experiment_id=experiment_id,
                    strategy=session_strategy_name,
                    session_id=session_id,
                    output_root=trials_dir,
                    profile=profile,
                    metadata=metadata,
                    config_files=config_files,
                    checkpoint=checkpoint,
                    prepared_session=True,
                    defer_analysis=defer_trial_analysis,
                )
                recorder_process = _ManagedProcess(recorder_command, trial_layout.logs_dir / "recorder.log")
                time.sleep(1.0)
                _require_process_running(recorder_process, "experiment recorder")
                log_stage(
                    f"T1 acquisition {index}/{len(planned)}: recorder active; "
                    "starting autonomous return-to-start phase",
                    logger=node.get_logger(),
                )
                node.publish_event(session_id, "trial_start", setpoint_id, metadata)
                if autoreset_enabled and bool(autoreset_cfg.get("start_before_every_trial", True)):
                    assert autoreset_profile is not None
                    reset_target = [float(value) for value in autoreset_cfg["target_controller_world"]]
                    if autoreset_continuous_handoff:
                        # The selected PPO profile is action-compatible with
                        # AUTO_RESET_PPO (validated checkpoint above).  Keep
                        # this one process alive through the next target goal.
                        if persistent_controller_process is not None:
                            _require_process_running(
                                persistent_controller_process,
                                "persistent continuous-handoff controller",
                            )
                            controller_process = persistent_controller_process
                        else:
                            controller_process = start_selected_controller(
                                controller_name=controller_name,
                                log_path=trial_layout.logs_dir / "controller.log",
                                label="continuous-handoff",
                            )
                        reset_process = controller_process
                    else:
                        node.clear_controller_preflight()
                        autoreset_heartbeat_sequence = node._goal_reached_sequence
                        autoreset_command = _launch_commands(
                            config,
                            autoreset_controller_name,
                            args.arm,
                            repo_root,
                        )[3][1]
                        autoreset_process = _ManagedProcess(
                            autoreset_command, trial_layout.logs_dir / "autoreset_controller.log"
                        )
                        _wait_for_topics(required_controller_topics, startup_timeout)
                        time.sleep(max(startup_settle, 0.0))
                        _require_process_running(autoreset_process, "autoreset motion controller")
                        node.wait_for_motion_controller_ready(
                            reached_sequence_before_start=autoreset_heartbeat_sequence,
                            process=autoreset_process,
                            label="autoreset",
                            timeout_sec=startup_timeout,
                        )
                        reset_process = autoreset_process
                    reset_metadata = copy.deepcopy(metadata)
                    reset_metadata["autoreset"]["phase"] = "active"
                    node.publish_event(session_id, "phase_start", "autoreset", reset_metadata)
                    node.publish_event(session_id, "autoreset_start", "common_start", reset_metadata)
                    autoreset_phase_active = True
                    reached_sequence_before_goal = node._goal_reached_sequence
                    node.publish_goal(*reset_target)
                    result = node.wait_for_autoreset_success(
                        target_controller_world=reset_target,
                        reached_sequence_before_goal=reached_sequence_before_goal,
                        timeout_sec=float(autoreset_cfg["timeout_sec"]),
                        position_tolerance_m=float(autoreset_cfg["position_tolerance_m"]),
                        yaw_tolerance_deg=float(autoreset_cfg["yaw_tolerance_deg"]),
                        enforce_state_health=bool(autoreset_cfg.get("enforce_state_health", True)),
                        tilt_guard_enabled=bool(autoreset_cfg.get("tilt_guard_enabled", True)),
                        tilt_limit_deg=float(autoreset_cfg["tilt_limit_deg"]),
                        tilt_max_continuous_violation_sec=float(
                            autoreset_cfg.get("tilt_max_continuous_violation_sec", 1.0)
                        ),
                        pose_stale_after_sec=pose_stale_after,
                        allowed_vision_modes=allowed_vision_modes,
                        monitored_processes=[
                            ("autoreset_motion_controller", reset_process),
                            ("experiment_recorder", recorder_process),
                        ],
                    )
                    reset_metadata["autoreset"].update({"phase": "success", **result})
                    node.publish_event(session_id, "autoreset_success", "common_start", reset_metadata)
                    node.publish_event(session_id, "phase_end", "autoreset", reset_metadata)
                    autoreset_phase_active = False
                    manifest.setdefault("autoreset_results", []).append({
                        "session_id": session_id,
                        "target": reset_target,
                        **result,
                    })
                    if not autoreset_continuous_handoff:
                        # The reset controller must relinquish the sole
                        # thruster publisher before the evaluated controller
                        # is launched.
                        node.cancel_goal()
                        node._spin_sleep(float(autoreset_cfg.get("cancel_settle_sec", 1.0)))
                        autoreset_process.stop(terminate_timeout)
                        autoreset_process = None

                if controller_process is None:
                    controller_process = start_selected_controller(
                        controller_name=controller_name,
                        log_path=trial_layout.logs_dir / "controller.log",
                        label="trial",
                    )
                node.publish_event(session_id, "phase_start", "acquisition", metadata)
                node.get_logger().info(
                    f"T1 acquisition {index}/{len(planned)}: recording pre-goal interval={pre_goal:.1f}s "
                    f"with {'continuous common-start hold' if autoreset_continuous_handoff else 'no active goal'}",
                )
                node._spin_sleep(pre_goal)
                x_m, y_m, z_m, yaw_deg = item["target_controller_world"]
                node.publish_goal(x_m, y_m, z_m, yaw_deg)
                node.publish_event(session_id, "hold_start", setpoint_id, metadata)
                node.get_logger().info(
                    f"T1 hold started: target={setpoint_id} R{repeat_index:02d} "
                    f"target=[{x_m:+.3f}, {y_m:+.3f}, {z_m:+.3f}] m yaw={yaw_deg:+.1f} deg"
                )
                node.spin_hold_with_runtime_status(
                    control_duration,
                    setpoint_id=setpoint_id,
                    repeat=repeat_index,
                    target_controller_world=item["target_controller_world"],
                    log_interval_sec=runtime_status_log_interval,
                    pose_stale_after_sec=pose_stale_after,
                    allowed_vision_modes=allowed_vision_modes,
                    monitored_processes=[
                        ("motion_controller", controller_process),
                        ("experiment_recorder", recorder_process),
                    ],
                )
                node.publish_event(session_id, "phase_end", "hold", metadata)
                # Keep the target hold active for the recording tail.  A
                # persistent controller then receives the next common-start
                # target immediately after ``trial_end`` instead of entering
                # a zero-thrust interval while this bag is finalized.
                node._spin_sleep(post_goal)
                node.publish_event(session_id, "trial_end", setpoint_id, metadata)
                if autoreset_persist_across_trials and index < len(planned):
                    reset_target = [float(value) for value in autoreset_cfg["target_controller_world"]]
                    node.publish_goal(*reset_target)
                    node.get_logger().info(
                        "T1 continuous cross-trial handoff: next common-start goal published "
                        f"target=[{reset_target[0]:+.3f}, {reset_target[1]:+.3f}, "
                        f"{reset_target[2]:+.3f}] yaw={reset_target[3]:+.1f}"
                    )
                elif not autoreset_persist_across_trials:
                    node.cancel_goal()
                finalize_message = (
                    "finalizing rosbag; analysis is deferred until the run completes"
                    if defer_trial_analysis
                    else "finalizing rosbag and running automatic analysis"
                )
                log_stage(
                    f"T1 acquisition {index}/{len(planned)}: control finished; {finalize_message}",
                    logger=node.get_logger(),
                )
                recorder_process.stop(terminate_timeout)
                recorder_process = None
                manifest.setdefault("completed_trials", []).append(session_id)
                trial_manifest_path = trials_dir / session_id / "manifest.json"
                if trial_manifest_path.is_file():
                    register_trial_runtime_artifacts(
                        run_manifest=manifest,
                        run_dir=run_dir,
                        trial=trial_layout,
                        local_logs=("autoreset_controller.log", "controller.log", "recorder.log"),
                        shared_logs=shared_log_files,
                    )
                    trial_manifest = read_json(trial_manifest_path)
                    target_plot_files: list[str] = []
                    source_target_plot_dir = trials_dir / session_id / "derived" / "plots" / "targets"
                    if source_target_plot_dir.is_dir():
                        for source_plot in sorted(source_target_plot_dir.glob("*_xyzyaw.png")):
                            destination = run_target_plot_dir / f"{session_id}_{source_plot.name}"
                            shutil.copy2(source_plot, destination)
                            target_plot_files.append(str(destination.relative_to(run_dir)))
                    manifest.setdefault("analysis_reports", []).append({
                        "session_id": session_id,
                        "status": (
                            "deferred"
                            if defer_trial_analysis
                            else trial_manifest.get("analysis", {}).get("status", "needs_review")
                        ),
                        "report": str(trials_dir / session_id / "derived" / "analysis_report.json"),
                        "target_plots": target_plot_files,
                    })
                    if defer_trial_analysis:
                        deferred_trials.append((session_id, trial_layout))
                    log_stage(
                        f"T1 acquisition {index}/{len(planned)} complete: target={setpoint_id} "
                        f"analysis={'deferred' if defer_trial_analysis else trial_manifest.get('analysis', {}).get('status', 'needs_review')} "
                        f"plots={len(target_plot_files)} report={trials_dir / session_id / 'derived' / 'analysis_report.json'}",
                        logger=node.get_logger(),
                    )
            except KeyboardInterrupt:
                metadata["error"] = "KeyboardInterrupt"
                manifest.setdefault("failed_trials", []).append(metadata)
                try:
                    node.publish_event(session_id, "operator_abort", setpoint_id, metadata)
                except Exception as exc:
                    # A second signal or an external ROS shutdown must not
                    # prevent the following hardware/child-process cleanup.
                    metadata["operator_abort_event_error"] = repr(exc)
                emergency_shutdown("keyboard_interrupt")
                raise
            except Exception as exc:
                metadata["error"] = repr(exc)
                manifest.setdefault("failed_trials", []).append(metadata)
                if autoreset_phase_active:
                    try:
                        failure_metadata = copy.deepcopy(metadata)
                        failure_metadata.setdefault("autoreset", {})["phase"] = "failure"
                        node.publish_event(session_id, "autoreset_failure", "common_start", failure_metadata)
                    except Exception as event_exc:
                        metadata["autoreset_failure_event_error"] = repr(event_exc)
                emergency_shutdown("runner_exception")
                raise
            finally:
                if recorder_process is not None:
                    recorder_process.stop(terminate_timeout, force=emergency_stop_done)
                if controller_process is not None and controller_process is not persistent_controller_process:
                    controller_process.stop(terminate_timeout, force=emergency_stop_done)
                if autoreset_process is not None:
                    autoreset_process.stop(terminate_timeout, force=emergency_stop_done)
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
        if persistent_controller_process is not None:
            try:
                if node is not None and not emergency_stop_done:
                    node.cancel_goal()
                persistent_controller_process.stop(
                    terminate_timeout, force=emergency_stop_done
                )
            except Exception as exc:  # pragma: no cover - best-effort process cleanup.
                manifest["persistent_controller_stop_error"] = repr(exc)
        if node is not None:
            try:
                if not emergency_stop_done:
                    if safety.get("disarm_on_exit", False):
                        emergency_shutdown("normal_run_completion")
                    else:
                        normal_stop = node.stop_motion(
                            zero_frames=int(emergency_cfg.get("zero_frames", 6)),
                            zero_frame_interval_sec=float(emergency_cfg.get("zero_frame_interval_sec", 0.05)),
                        )
                        normal_stop["status"] = "complete"
                        manifest["normal_stop"] = normal_stop
            except Exception as exc:  # pragma: no cover - hardware shutdown failure is environment-specific.
                manifest["normal_stop"] = {"status": "error", "error": repr(exc)}
            node.destroy_node()
            if rclpy.ok():
                rclpy.shutdown()
        if defer_trial_analysis:
            finalize_deferred_analysis()
        for process in reversed(support_processes):
            process.stop(terminate_timeout)
        manifest["bridge_exit_policy"] = "disarm" if safety.get("disarm_on_exit", False) else "preserve_enabled"
        manifest["bridge_disarmed_on_exit"] = bool(
            manifest.get("emergency_stop", {}).get("bridge_disabled_confirmed", False)
        )
        if run_target_plot_dir.is_dir():
            manifest["target_plot_summary"] = {
                "expected_trials": len(planned),
                "generated_plots": len(list(run_target_plot_dir.glob("*_xyzyaw.png"))),
                "directory": str(run_target_plot_dir),
            }
        manifest["status"] = manifest.get("status", "complete") if manifest.get("status") != "running" else "complete"
        manifest["finished_at"] = now_iso()
        write_json(run_dir / "run_manifest.json", manifest)
        log_stage(
            f"T1 run finished: status={manifest['status']} completed={len(manifest.get('completed_trials', []))} "
            f"failed={len(manifest.get('failed_trials', []))} run={run_dir}"
        )


if __name__ == "__main__":
    main()
