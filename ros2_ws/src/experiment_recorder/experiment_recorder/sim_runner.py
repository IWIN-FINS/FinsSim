"""Unity/MARUS simulation experiment runners with training-native actions.

These runners deliberately do not use the reward-only ML-Agents evaluator.
Instead a graphical Unity Fossen scene is prepared with its ML-Agent disabled,
then ROS2 hosts the selected checkpoint and connects it through
``grpc_ros_adapter`` under the isolated ``/sim`` prefix.  The simulation
profiles retain the action interface used during training and deliberately do
not import deployment-only hardware gains, reprojection, or axis masking.
Each trial is recorded by :mod:`recorder`, so raw rosbag2 data, manifests,
derived CSVs and automatic plots use the same *data layout* as hardware trials.
"""

from __future__ import annotations

import argparse
import copy
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time
from typing import Any, Iterable, Sequence

from geometry_msgs.msg import PoseStamped, PoseWithCovarianceStamped, Transform, Twist
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter
from rclpy.signals import SignalHandlerOptions
from std_msgs.msg import Empty, Float32MultiArray, UInt64
from std_srvs.srv import Trigger
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
from msgs.msg import TrajectoryEvent
import yaml

from .layout import (
    canonical_run_dir,
    create_run_layout,
    prepare_trial_layout,
    register_trial_runtime_artifacts,
)
from .manifest import file_sha256, git_revision, now_iso, read_json, write_json
from .progress import log_stage
from .runner import (
    _ManagedProcess,
    _controller_storage_subdirectory,
    _record_command,
    _require_process_running,
    _resolve_cli_path,
    _resolve_path,
    _wait_for_topics,
    init_experiment_ros,
)
from .t2_runner import _build_catalog, _copy_trial_plots
from .t2_trajectory import ReferencePoint, write_reference_csv


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _mapping(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise SystemExit(f"{label} must be a YAML mapping")
    return value


def _load_yaml(path: Path, *, required: Iterable[str], label: str) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"{label} YAML not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid {label} YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} YAML must contain a mapping: {path}")
    for key in required:
        if not isinstance(payload.get(key), dict):
            raise SystemExit(f"{label} YAML requires mapping `{key}`")
    return payload


def _new_run_id(root: Path, prefix: str) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{prefix}"
    candidate, suffix = base, 2
    while (root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _safe_slug(value: object) -> str:
    result = "".join(ch if str(ch).isalnum() or ch in "-_" else "_" for ch in str(value))
    return result.strip("-_") or "UNSPECIFIED"


def _ros_parameters(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"could not read controller YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"controller YAML must contain a mapping: {path}")
    for node_name in ("motion_controller", "/motion_controller", "/**", "**"):
        node = payload.get(node_name)
        if not isinstance(node, dict):
            continue
        params = node.get("ros__parameters")
        if isinstance(params, dict):
            return params
    raise SystemExit(f"controller YAML has no motion-controller parameters: {path}")


def _controller_checkpoint(params: dict[str, Any], repo_root: Path) -> Path | None:
    raw = str(params.get("checkpoint_path", "")).strip()
    return _resolve_path(raw, repo_root) if raw else None


def _validate_checkpoint_contract(params: dict[str, Any], checkpoint: Path) -> dict[str, Any]:
    """Load a PPO checkpoint before starting Unity and verify its action contract.

    The ROS2 controller has two intentionally distinct PPO interfaces.  A
    ``wrench6d_force_n`` profile needs the raw six-output wrench policy head;
    a Unity ``eight_thruster`` checkpoint instead owns an eight-output direct
    thruster head.  Loading it only after Unity/ROS2 are running used to leave
    an operator with a failed first trial and orphan support processes.
    """

    backend_name = str(params.get("backend_name", "")).strip()
    if not backend_name.startswith("ppo_"):
        return {}
    try:
        from motion_control.backend_loader import load_backend

        backend = load_backend(backend_name, checkpoint_path=str(checkpoint), device="cpu")
    except Exception as exc:
        raise SystemExit(
            "checkpoint is incompatible with the selected ROS2 controller profile: "
            f"backend={backend_name!r}, checkpoint={checkpoint}. "
            "The selected profile must use the checkpoint's matching network and action interface. "
            "For T1, choose `--ppo-action-interface wrench6` for a 6D physical-wrench "
            "checkpoint or `--ppo-action-interface thruster8` for a direct eight-thruster checkpoint. "
            "Original loader error: "
            f"{exc}"
        ) from exc

    output_mode = str(params.get("thruster_output_mode", "force_n")).strip()
    observation_shape = tuple(getattr(getattr(backend.model, "observation_space", None), "shape", ()) or ())
    if not observation_shape or any(int(value) <= 0 for value in observation_shape):
        raise SystemExit(
            f"could not determine observation shape while validating checkpoint {checkpoint}: {observation_shape}"
        )
    zero_observation = np.zeros(observation_shape, dtype=np.float32)
    action = (
        backend.predict_policy_action(zero_observation)
        if output_mode == "wrench6d_force_n"
        else backend.predict_action(zero_observation)
    )
    action_dim = int(np.asarray(action).reshape(-1).shape[0])
    expected_action_dim = 6 if output_mode == "wrench6d_force_n" else 8
    if action_dim != expected_action_dim:
        expected_label = "6D raw wrench" if output_mode == "wrench6d_force_n" else "8D canonical thruster"
        raise SystemExit(
            "checkpoint action interface does not match the selected ROS2 controller profile: "
            f"profile backend={backend_name!r}, output_mode={output_mode!r} expects {expected_label}, "
            f"but {checkpoint} produces {action_dim} values."
        )
    return {
        "backend_name": backend_name,
        "observation_dim": int(np.prod(observation_shape)),
        "policy_action_dim": action_dim,
        "thruster_output_mode": output_mode,
        "preflight_device": "cpu",
    }


def _validate_sim_truth_status_contract(params: dict[str, Any], config_path: Path) -> None:
    """Reject a simulation profile whose readiness publisher and gate differ.

    ``motion_controller.launch.py`` starts ``sim_truth_state_status`` with the
    same parameter YAML.  The helper reads ``status_topic`` while the
    controller reads ``state_status_topic``.  If only the latter is remapped
    to ``/sim``, the gate cannot receive a ready message and the safe output
    is (correctly) eight zeros.  Detect that wiring error before opening
    Unity, including when an operator supplies ``--controller-config``.
    """

    if str(params.get("state_input_mode", "")).strip() != "sim_truth":
        return
    gate_topic = str(params.get("state_status_topic", "/finsrov/controller/state/status")).strip()
    publisher_topic = str(params.get("status_topic", "/finsrov/controller/state/status")).strip()
    if not gate_topic or not publisher_topic or gate_topic != publisher_topic:
        raise SystemExit(
            "invalid sim_truth controller profile: `status_topic` (published by "
            "sim_truth_state_status) must exactly equal `state_status_topic` "
            f"(consumed by motion_controller); profile={config_path}, "
            f"status_topic={publisher_topic!r}, state_status_topic={gate_topic!r}"
        )


def _simulation_clock_config(config: dict[str, Any]) -> dict[str, Any]:
    """Resolve the simulation-only clock contract from an experiment YAML.

    Unity/MARUS can run physics faster than wall time only when its stepped
    clock is also the ROS clock.  ``ros2_control_lockstep`` additionally makes
    Unity wait for a ROS2 controller acknowledgement on every physics step.
    That protects 10 Hz policy inference from a high Unity time scale without
    changing any training-side ``eval_time_scale`` default.
    """

    runner = _mapping(config["runner"], "runner")
    raw = _mapping(runner.get("simulation_clock", {}), "runner.simulation_clock")
    if raw.get("use_sim_time") is not True:
        raise SystemExit(
            "simulation experiments require runner.simulation_clock.use_sim_time: true; "
            "do not use a wall-clock timeout as the evaluation-duration source"
        )
    try:
        time_scale = float(raw.get("time_scale", 1.0))
        start_timeout = float(raw.get("clock_start_timeout_wall_sec", 30.0))
        stall_timeout = float(raw.get("clock_stall_timeout_wall_sec", 15.0))
        lockstep_ack_timeout = float(raw.get("lockstep_ack_timeout_wall_sec", 15.0))
    except (TypeError, ValueError) as exc:
        raise SystemExit("runner.simulation_clock time values must be numeric") from exc
    if not math.isfinite(time_scale) or not 0.0 < time_scale <= 20.0:
        raise SystemExit("runner.simulation_clock.time_scale must be finite and in (0, 20]")
    if start_timeout <= 0.0 or stall_timeout <= 0.0:
        raise SystemExit("runner.simulation_clock wall-clock timeouts must be positive")
    mode = str(raw.get("mode", "free_running")).strip().lower()
    if mode not in {"free_running", "ros2_control_lockstep"}:
        raise SystemExit(
            "runner.simulation_clock.mode must be `free_running` or `ros2_control_lockstep`; "
            f"got {mode!r}"
        )
    if lockstep_ack_timeout <= 0.0:
        raise SystemExit("runner.simulation_clock.lockstep_ack_timeout_wall_sec must be positive")
    clock_topic = str(raw.get("clock_topic", "/clock")).strip()
    if clock_topic != "/clock":
        raise SystemExit(
            "the Unity/MARUS grpc adapter publishes the simulation clock only on /clock; "
            f"got runner.simulation_clock.clock_topic={clock_topic!r}"
        )
    lockstep_ack_topic = str(
        raw.get("lockstep_ack_topic", "/sim/motion_controller/debug/control_tick_complete")
    ).strip()
    if mode == "ros2_control_lockstep" and not lockstep_ack_topic.startswith("/"):
        raise SystemExit("runner.simulation_clock.lockstep_ack_topic must be an absolute ROS topic")
    return {
        "mode": mode,
        "use_sim_time": True,
        "clock_topic": clock_topic,
        "time_scale": time_scale,
        "clock_start_timeout_wall_sec": start_timeout,
        "clock_stall_timeout_wall_sec": stall_timeout,
        "lockstep_ack_topic": lockstep_ack_topic,
        "lockstep_ack_timeout_wall_sec": lockstep_ack_timeout,
        "protocol_time_basis": "ros_simulation_time",
    }


def _validate_simulation_time_controller_profile(
    params: dict[str, Any], config_path: Path, simulation_clock: dict[str, Any]
) -> None:
    """Ensure controller timers and state freshness use the same simulated clock."""

    if simulation_clock["use_sim_time"] and params.get("use_sim_time") is not True:
        raise SystemExit(
            "simulation controller profile must declare `use_sim_time: true` so controller timers, "
            "state freshness, and experiment windows share Unity /clock; "
            f"profile={config_path}"
        )
    lockstep_enabled = bool(params.get("lockstep_enabled", False))
    if simulation_clock["mode"] == "ros2_control_lockstep":
        if not lockstep_enabled:
            raise SystemExit(
                "ros2_control_lockstep requires the selected controller profile to declare "
                "`lockstep_enabled: true`; "
                f"profile={config_path}"
            )
        profile_clock_topic = str(params.get("lockstep_clock_topic", "/clock")).strip()
        profile_ack_topic = str(params.get("lockstep_ack_topic", "")).strip()
        if profile_clock_topic != simulation_clock["clock_topic"]:
            raise SystemExit(
                "controller lockstep_clock_topic must match runner.simulation_clock.clock_topic; "
                f"profile={config_path}"
            )
        if profile_ack_topic != simulation_clock["lockstep_ack_topic"]:
            raise SystemExit(
                "controller lockstep_ack_topic must match runner.simulation_clock.lockstep_ack_topic; "
                f"profile={config_path}"
            )
    elif lockstep_enabled:
        raise SystemExit(
            "controller profile enables lockstep but runner.simulation_clock.mode is free_running; "
            f"profile={config_path}"
        )


def _t2_required_controller_topics(config: dict[str, Any], controller_name: str) -> list[str]:
    """Return the controller-specific T2 readiness contract.

    PPO backends expose a policy-observation debug topic, whereas the
    traditional PID backend deliberately does not create one.  Requiring the
    PPO-only topic from every controller stalls a valid PID trial before its
    trajectory is published.  A controller may therefore override the common
    readiness list in the experiment YAML; all other controllers retain the
    common list unchanged.
    """

    runner = _mapping(config["runner"], "runner")
    auto = _mapping(runner["auto_launch"], "runner.auto_launch")
    controllers = _mapping(config["controllers"], "controllers")
    selected = _mapping(controllers.get(controller_name), f"controllers.{controller_name}")
    raw_topics = selected.get("required_controller_topics", auto.get("required_controller_topics", []))
    if not isinstance(raw_topics, list) or not all(isinstance(topic, str) and topic.strip() for topic in raw_topics):
        raise SystemExit(
            f"controllers.{controller_name}.required_controller_topics must be a non-empty list of topic strings"
        )
    return [topic.strip() for topic in raw_topics]


def _resolve_controller(
    config: dict[str, Any],
    controller_name: str,
    *,
    repo_root: Path,
    controller_config_override: Path | None,
    checkpoint_override: Path | None,
) -> tuple[dict[str, Any], Path, Path | None, dict[str, Any]]:
    controllers = _mapping(config["controllers"], "controllers")
    selected = copy.deepcopy(_mapping(controllers.get(controller_name), f"controllers.{controller_name}"))
    if not bool(selected.get("enabled", True)):
        raise SystemExit(f"controller `{controller_name}` is disabled in the simulation protocol")
    config_path = controller_config_override or _resolve_path(str(selected.get("config_file", "")), repo_root)
    if not config_path.is_file():
        raise SystemExit(f"controller config does not exist: {config_path}")
    params = _ros_parameters(config_path)
    _validate_sim_truth_status_contract(params, config_path)
    expected_backend = str(selected.get("expected_backend_name", "")).strip()
    if expected_backend and str(params.get("backend_name", "")).strip() != expected_backend:
        raise SystemExit(
            f"controller backend mismatch: expected `{expected_backend}`, "
            f"profile declares `{params.get('backend_name')}`"
        )
    checkpoint = checkpoint_override or _controller_checkpoint(params, repo_root)
    if bool(selected.get("checkpoint_required", False)):
        if checkpoint is None:
            raise SystemExit(f"controller `{controller_name}` requires --checkpoint or checkpoint_path in its profile")
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    elif checkpoint_override is not None:
        raise SystemExit(f"controller `{controller_name}` does not accept --checkpoint")
    selected["config_file"] = str(config_path)
    selected["checkpoint_path"] = str(checkpoint) if checkpoint else None
    if checkpoint is not None:
        selected["checkpoint_preflight"] = _validate_checkpoint_contract(params, checkpoint)
    return selected, config_path, checkpoint, params


def _config_files(
    config_path: Path,
    controller_config: Path,
    config: dict[str, Any],
    repo_root: Path,
    *,
    extra: Sequence[Path] = (),
) -> list[Path]:
    files = [config_path, controller_config, *extra]
    auto = _mapping(_mapping(config["runner"], "runner").get("auto_launch", {}), "runner.auto_launch")
    motion = _mapping(auto.get("motion_controller", {}), "runner.auto_launch.motion_controller")
    pool = motion.get("pool_world_params_file")
    if pool:
        files.append(_resolve_path(str(pool), repo_root))
    unique: list[Path] = []
    for path in files:
        resolved = Path(path).resolve()
        if resolved not in unique:
            unique.append(resolved)
    missing = [str(path) for path in unique if not path.is_file()]
    if missing:
        raise SystemExit(f"required protocol/config files do not exist: {missing}")
    return unique


class _SimulationExperimentNode(Node):
    """Event, goal, reset and zero-thrust endpoint for the isolated /sim graph."""

    def __init__(
        self,
        *,
        event_topic: str,
        position_goal_topic: str,
        trajectory_topic: str,
        cancel_topic: str,
        thruster_topic: str,
        reset_service_name: str,
        pose_topic: str,
        use_sim_time: bool,
        clock_stall_timeout_wall_sec: float,
        trajectory_reference_stamped_topic: str = "",
        active_pose_topic: str = "",
        lockstep_ack_topic: str = "",
    ) -> None:
        super().__init__(
            "finsrov_simulation_experiment_runner",
            parameter_overrides=[Parameter("use_sim_time", Parameter.Type.BOOL, bool(use_sim_time))],
        )
        self._uses_sim_time = bool(use_sim_time)
        self._clock_stall_timeout_wall_sec = max(float(clock_stall_timeout_wall_sec), 0.1)
        self._event_pub = self.create_publisher(TrajectoryEvent, event_topic, 20)
        self._goal_pub = self.create_publisher(PoseStamped, position_goal_topic, 10)
        self._trajectory_pub = self.create_publisher(MultiDOFJointTrajectory, trajectory_topic, 10)
        self._cancel_pub = self.create_publisher(Empty, cancel_topic, 10)
        self._zero_pub = self.create_publisher(Float32MultiArray, thruster_topic, 10)
        self._reset_client = self.create_client(Trigger, reset_service_name)
        self._latest_pose: PoseWithCovarianceStamped | None = None
        self._latest_pose_monotonic: float | None = None
        self.create_subscription(PoseWithCovarianceStamped, pose_topic, self._pose_callback, 20)
        self._active_goal_history: list[tuple[int, PoseStamped]] = []
        self._active_goal_probe_pub = None
        active_topic = str(active_pose_topic).strip()
        if active_topic:
            # A non-publishing probe gives the runner a per-trial rosbag
            # readiness check for the controller's header-stamped goal state.
            # It does not alter the controller's established status topic.
            self._active_goal_probe_pub = self.create_publisher(PoseStamped, active_topic, 10)
            self.create_subscription(PoseStamped, active_topic, self._active_goal_callback, 20)
        self._latest_lockstep_ack_ns: int | None = None
        self._lockstep_ack_probe_pub = None
        ack_topic = str(lockstep_ack_topic).strip()
        if ack_topic:
            # The UInt64 payload is the exact Unity tick acknowledged by the
            # controller, unlike wall-clock rosbag receive timestamps.
            self._lockstep_ack_probe_pub = self.create_publisher(UInt64, ack_topic, 10)
            self.create_subscription(UInt64, ack_topic, self._lockstep_ack_callback, 50)
        self._trajectory_reference_probe_pub = None
        self._latest_trajectory_reference_ns: int | None = None
        topic = str(trajectory_reference_stamped_topic).strip()
        if topic:
            # A local, non-publishing probe lets the runner verify that the
            # per-trial rosbag has subscribed to the stamped reference before
            # the controller can emit its t=0 sample.
            self._trajectory_reference_probe_pub = self.create_publisher(PoseStamped, topic, 10)
            self.create_subscription(PoseStamped, topic, self._trajectory_reference_callback, 20)

    def _pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        self._latest_pose = message
        self._latest_pose_monotonic = time.monotonic()

    def _trajectory_reference_callback(self, message: PoseStamped) -> None:
        stamp = message.header.stamp
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if value > 0:
            self._latest_trajectory_reference_ns = value

    def _active_goal_callback(self, message: PoseStamped) -> None:
        stamp = message.header.stamp
        value = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if value <= 0:
            return
        self._active_goal_history.append((value, message))
        # A goal is a low-rate status stream. Keep enough history to retain
        # the first controller tick after a repeated runner publication while
        # bounding memory in an all-target protocol run.
        if len(self._active_goal_history) > 512:
            del self._active_goal_history[:-256]

    def _lockstep_ack_callback(self, message: UInt64) -> None:
        value = int(message.data)
        if value > 0:
            self._latest_lockstep_ack_ns = value

    def _clock_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def wait_for_clock(self, timeout_sec: float, *, clock_topic: str = "/clock") -> None:
        """Wait for the first non-zero Unity/MARUS simulated timestamp."""

        if not self._uses_sim_time:
            return
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
            now = self._clock_sec()
            if now > 0.0:
                self.get_logger().info(f"[SIM] received {clock_topic}; protocol time starts at t={now:.3f} s")
                return
        raise RuntimeError(
            f"Unity/MARUS did not publish {clock_topic} before the wall-clock startup timeout; "
            "the evaluation must not continue with wall-clock trial durations"
        )

    def _delivery_wait(self, *, simulation_sec: float, wall_sec: float) -> None:
        if self._uses_sim_time:
            self.spin_sleep(simulation_sec)
        else:
            time.sleep(wall_sec)

    def spin_sleep(self, duration_sec: float, *, log_interval_sec: float | None = None, label: str = "") -> None:
        duration = max(float(duration_sec), 0.0)
        if not self._uses_sim_time:
            deadline = time.monotonic() + duration
            next_log = time.monotonic()
            while rclpy.ok() and time.monotonic() < deadline:
                rclpy.spin_once(self, timeout_sec=min(0.1, max(0.01, deadline - time.monotonic())))
                now = time.monotonic()
                if log_interval_sec and now >= next_log:
                    next_log = now + max(float(log_interval_sec), 0.1)
                    self.get_logger().info(f"[SIM] {label} {self.position_text()}")
            return

        start = self._clock_sec()
        if start <= 0.0:
            raise RuntimeError("simulation-time wait requested before /clock was initialized")
        deadline = start + duration
        next_log = start
        last_clock = start
        last_progress_wall = time.monotonic()
        while rclpy.ok() and self._clock_sec() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            now = self._clock_sec()
            if now > last_clock:
                last_clock = now
                last_progress_wall = time.monotonic()
            elif time.monotonic() - last_progress_wall > self._clock_stall_timeout_wall_sec:
                raise RuntimeError(
                    "/clock stopped advancing during a simulation-time protocol wait; "
                    "Unity Play mode or grpc_ros_adapter may have stopped"
                )
            if log_interval_sec and now >= next_log:
                next_log = now + max(float(log_interval_sec), 0.1)
                self.get_logger().info(f"[SIM t={now:.2f}s] {label} {self.position_text()}")

    def position_text(self) -> str:
        if self._latest_pose is None or self._latest_pose_monotonic is None:
            return "position=unavailable"
        position = self._latest_pose.pose.pose.position
        age = time.monotonic() - self._latest_pose_monotonic
        return f"position=[{position.x:+.3f}, {position.y:+.3f}, {position.z:+.3f}] m age={age:.2f}s"

    def wait_for_pose(self, timeout_sec: float) -> None:
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            if self._latest_pose is not None:
                return
            rclpy.spin_once(self, timeout_sec=0.1)
        raise RuntimeError("Unity /sim controller pose did not publish before the trial timeout")

    def event_subscription_count(self) -> int:
        """Return the currently discovered experiment-event subscribers."""

        return int(self._event_pub.get_subscription_count())

    def wait_for_new_event_subscriber(self, baseline: int, timeout_sec: float) -> None:
        """Wait until this trial's rosbag recorder has joined the event graph.

        A fixed wall-time sleep is not a readiness condition: with repeated,
        accelerated trials the new rosbag subscription can still be absent
        when the first ``trajectory_start`` marker is emitted.  Snapshotting
        the count before launching the recorder also tolerates an unrelated
        experiment recorder already present in the ROS domain.
        """

        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.event_subscription_count() > int(baseline):
                return
        raise RuntimeError(
            "trial rosbag recorder did not subscribe to /finsrov/experiment/event before timeout; "
            "trajectory markers would be unauditable"
        )

    def active_goal_subscription_count(self) -> int:
        if self._active_goal_probe_pub is None:
            return 0
        return int(self._active_goal_probe_pub.get_subscription_count())

    def wait_for_active_goal_subscriber(self, baseline: int, timeout_sec: float) -> None:
        """Require the trial rosbag to record the stamped controller goal."""

        if self._active_goal_probe_pub is None:
            raise RuntimeError("T1 lockstep requires an active-pose recorder probe topic")
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.active_goal_subscription_count() > int(baseline):
                return
        raise RuntimeError(
            "trial rosbag recorder did not subscribe to /sim/motion_controller/status/active_pose before timeout; "
            "the T1 controller-tick timing boundary would be unauditable"
        )

    def lockstep_ack_subscription_count(self) -> int:
        if self._lockstep_ack_probe_pub is None:
            return 0
        return int(self._lockstep_ack_probe_pub.get_subscription_count())

    def wait_for_lockstep_ack_subscriber(self, baseline: int, timeout_sec: float) -> None:
        """Require the recorder to capture the exact controller tick handoff."""

        if self._lockstep_ack_probe_pub is None:
            raise RuntimeError("lockstep recorder readiness requires an acknowledgement probe topic")
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.lockstep_ack_subscription_count() > int(baseline):
                return
        raise RuntimeError(
            "trial rosbag recorder did not subscribe to the lockstep acknowledgement topic before timeout; "
            "the accelerated clock handoff would be unauditable"
        )

    @staticmethod
    def _active_goal_matches(
        message: PoseStamped,
        x_m: float,
        y_m: float,
        z_m: float,
        yaw_deg: float,
    ) -> bool:
        if str(message.header.frame_id).strip() != "controller_world":
            return False
        position = message.pose.position
        if max(abs(float(position.x) - x_m), abs(float(position.y) - y_m), abs(float(position.z) - z_m)) > 1e-4:
            return False
        expected_half = math.radians(float(yaw_deg)) * 0.5
        expected_y, expected_w = math.sin(expected_half), math.cos(expected_half)
        orientation = message.pose.orientation
        # Quaternion q and -q denote the same target orientation.
        dot = float(orientation.y) * expected_y + float(orientation.w) * expected_w
        return abs(abs(dot) - 1.0) <= 1e-4

    def active_goal_stamp_ns(self) -> int | None:
        return self._active_goal_history[-1][0] if self._active_goal_history else None

    def wait_for_applied_position_goal(
        self,
        *,
        baseline_stamp_ns: int | None,
        x_m: float,
        y_m: float,
        z_m: float,
        yaw_deg: float,
        timeout_sec: float,
    ) -> int:
        """Return the first controller tick that applied a T1 position goal.

        The runner command is repeated for transport robustness.  In an
        accelerated run its publication time is not the control boundary: the
        header-stamped ``active_pose`` emitted by the controller is.  This
        method retains the earliest matching tick after the pre-goal baseline.
        """

        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            for stamp_ns, message in self._active_goal_history:
                if baseline_stamp_ns is not None and stamp_ns <= baseline_stamp_ns:
                    continue
                if self._active_goal_matches(message, x_m, y_m, z_m, yaw_deg):
                    return stamp_ns
            rclpy.spin_once(self, timeout_sec=0.02)
        raise RuntimeError(
            "controller did not publish a matching header-stamped active position goal after the T1 command; "
            "the lockstep trial cannot establish an auditable t=0"
        )

    def trajectory_reference_subscription_count(self) -> int:
        if self._trajectory_reference_probe_pub is None:
            return 0
        return int(self._trajectory_reference_probe_pub.get_subscription_count())

    def wait_for_trajectory_reference_subscriber(self, baseline: int, timeout_sec: float) -> None:
        """Require the trial rosbag to subscribe to the stamped T2 reference."""

        if self._trajectory_reference_probe_pub is None:
            raise RuntimeError("T2 lockstep recorder readiness requires a stamped-reference probe topic")
        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.05)
            if self.trajectory_reference_subscription_count() > int(baseline):
                return
        raise RuntimeError(
            "trial rosbag recorder did not subscribe to the stamped T2 runtime-reference topic before timeout; "
            "the lockstep trajectory cannot start without a t=0 audit stream"
        )

    def trajectory_reference_stamp_ns(self) -> int | None:
        return self._latest_trajectory_reference_ns

    def wait_for_lockstep_trajectory_progress(
        self,
        *,
        baseline_stamp_ns: int | None,
        duration_sec: float,
        timeout_sec: float,
        log_interval_sec: float | None = None,
        label: str = "",
    ) -> tuple[int, int]:
        """Wait for an applied T2 reference to reach its declared duration.

        This deliberately does not use the runner node's ``get_clock()``.
        At high simulation speed that clock can be delivered later than the
        controller's explicit lockstep callback.  The controller's stamped
        reference is the sole source for both start and completion here.
        """

        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        start_ns: int | None = None
        target_ns: int | None = None
        next_log = 0.0
        required_ns = max(0, int(round(float(duration_sec) * 1_000_000_000.0)))
        while rclpy.ok() and time.monotonic() < deadline:
            rclpy.spin_once(self, timeout_sec=0.02)
            current_ns = self._latest_trajectory_reference_ns
            if current_ns is None or (baseline_stamp_ns is not None and current_ns <= baseline_stamp_ns):
                continue
            if start_ns is None:
                start_ns = current_ns
                target_ns = start_ns + required_ns
                next_log = 0.0
                self.get_logger().info(
                    f"[SIM] lockstep T2 reference activated at t={start_ns * 1e-9:.3f}s; "
                    f"waiting {duration_sec:.3f}s of controller-applied trajectory"
                )
            assert target_ns is not None
            elapsed_sec = max(0.0, (current_ns - start_ns) * 1e-9)
            if log_interval_sec is not None and elapsed_sec >= next_log:
                next_log = elapsed_sec + max(float(log_interval_sec), 0.1)
                self.get_logger().info(
                    f"[SIM lockstep t={elapsed_sec:.2f}s] {label} {self.position_text()}"
                )
            if current_ns >= target_ns:
                return start_ns, current_ns
        if start_ns is None:
            raise RuntimeError("controller did not publish a stamped T2 reference after trajectory publication")
        raise RuntimeError(
            "controller-stamped T2 reference did not reach its declared duration before the wall-clock timeout; "
            "the trial is incomplete and must not be cancelled as successful"
        )

    def publish_event(self, session_id: str, event: str, label: str, metadata: dict[str, Any]) -> None:
        message = TrajectoryEvent()
        message.session_id = session_id
        message.event = event
        message.label = label
        message.metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
        for _ in range(3):
            message.header.stamp = self.get_clock().now().to_msg()
            self._event_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.03)
            self._delivery_wait(simulation_sec=0.02, wall_sec=0.05)

    def publish_goal(self, x_m: float, y_m: float, z_m: float, yaw_deg: float) -> None:
        """Publish a controller-world pose target.

        ``controller_world`` is the Unity-native controller basis: its yaw is
        a rotation about +Y, not the conventional ROS ENU +Z yaw.  This must
        match ``quat_from_controller_ypr`` in the motion controller.  In
        particular, a Z-axis quaternion is interpreted there as pitch and
        silently reduces a yaw-only PID target to zero.
        """
        message = PoseStamped()
        message.header.frame_id = "controller_world"
        message.pose.position.x = float(x_m)
        message.pose.position.y = float(y_m)
        message.pose.position.z = float(z_m)
        half_yaw = math.radians(float(yaw_deg)) * 0.5
        message.pose.orientation.y = math.sin(half_yaw)
        message.pose.orientation.w = math.cos(half_yaw)
        for _ in range(5):
            message.header.stamp = self.get_clock().now().to_msg()
            self._goal_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.03)
            self._delivery_wait(simulation_sec=0.02, wall_sec=0.05)

    def publish_trajectory(self, points: Sequence[ReferencePoint]) -> None:
        message = MultiDOFJointTrajectory()
        message.header.frame_id = "controller_world"
        for reference in points:
            point = MultiDOFJointTrajectoryPoint()
            transform = Transform()
            transform.translation.x = float(reference.x_m)
            transform.translation.y = float(reference.y_m)
            transform.translation.z = float(reference.z_m)
            transform.rotation.w = 1.0
            point.transforms.append(transform)
            velocity = Twist()
            velocity.linear.x = float(reference.vx_mps)
            velocity.linear.y = float(reference.vy_mps)
            velocity.linear.z = float(reference.vz_mps)
            point.velocities.append(velocity)
            point.time_from_start.sec = int(reference.time_sec)
            point.time_from_start.nanosec = int(round((reference.time_sec - int(reference.time_sec)) * 1e9))
            message.points.append(point)
        for _ in range(3):
            message.header.stamp = self.get_clock().now().to_msg()
            self._trajectory_pub.publish(message)
            rclpy.spin_once(self, timeout_sec=0.03)
            self._delivery_wait(simulation_sec=0.02, wall_sec=0.05)

    def stop_motion(self, *, zero_frames: int = 6) -> None:
        cancel = Empty()
        zero = Float32MultiArray()
        zero.data = [0.0] * 8
        for _ in range(max(int(zero_frames), 1)):
            self._cancel_pub.publish(cancel)
            self._zero_pub.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.02)
            self._delivery_wait(simulation_sec=0.02, wall_sec=0.03)

    def reset_vehicle(self, timeout_sec: float) -> None:
        if not self._reset_client.wait_for_service(timeout_sec=max(float(timeout_sec), 0.0)):
            raise RuntimeError("Unity reset service is unavailable; grpc adapter/Unity GUI may not be ready")
        future = self._reset_client.call_async(Trigger.Request())
        rclpy.spin_until_future_complete(self, future, timeout_sec=max(float(timeout_sec), 0.0))
        if not future.done() or future.exception() is not None:
            raise RuntimeError(f"Unity reset service failed: {future.exception() if future.done() else 'timeout'}")
        response = future.result()
        if response is None or not response.success:
            raise RuntimeError(f"Unity reset service rejected the request: {getattr(response, 'message', '')}")


def _editor_status(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _status_mtime_ns(path: Path) -> int | None:
    try:
        return path.stat().st_mtime_ns
    except OSError:
        return None


def _write_editor_command(path: Path, command: str) -> None:
    temporary = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    temporary.write_text(f"{command}\n", encoding="utf-8")
    temporary.replace(path)


def _probe_editor_status(command_file: Path, status_file: Path, timeout_sec: float) -> dict[str, Any] | None:
    """Ask an already-open ExternalPlayController for one fresh status record.

    The command/status files are deliberately treated as a request--response
    pair.  A status JSON left by a previous Unity process is not evidence that
    an editor is currently reusable.
    """

    before_mtime = _status_mtime_ns(status_file)
    _write_editor_command(command_file, "status")
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        payload = _editor_status(status_file)
        after_mtime = _status_mtime_ns(status_file)
        if (
            payload is not None
            and after_mtime is not None
            and after_mtime != before_mtime
            and (
                payload.get("event") == "status_requested"
                or str(payload.get("lastCommand", "")) == "status"
            )
        ):
            return payload
        time.sleep(0.1)
    return None


def _same_resolved_path(value: object, expected: Path) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    try:
        return Path(value).expanduser().resolve() == expected.resolve()
    except OSError:
        return False


def _project_relative_scene_path(scene: Path, project: Path) -> str:
    try:
        relative = scene.resolve().relative_to(project.resolve())
    except ValueError as exc:
        raise RuntimeError(f"Unity scene {scene} is not inside project {project}") from exc
    asset_path = relative.as_posix()
    if not asset_path.startswith("Assets/") or not asset_path.endswith(".unity"):
        raise RuntimeError(f"Unity scene must be an Assets/*.unity path, got {asset_path!r}")
    return asset_path


def _unity_sim_thruster_command_mode(controller: dict[str, Any]) -> str:
    """Return the Unity bridge action ABI registered for one simulation policy.

    This setting is intentionally owned by the simulation protocol rather
    than by a hardware controller YAML.  A direct Unity-trained 8D policy
    publishes its native normalized action, whereas policies trained through
    the physical wrench allocator publish the allocator's force command.
    """

    requested = str(controller.get("unity_sim_thruster_command_mode", "force_n")).strip().lower()
    aliases = {
        "force": "force_n",
        "force_n": "force_n",
        "normalized": "normalized_direct",
        "normalized_direct": "normalized_direct",
    }
    if requested not in aliases:
        raise SystemExit(
            "simulation controller has invalid unity_sim_thruster_command_mode "
            f"{requested!r}; expected `force_n` or `normalized_direct`"
        )
    return aliases[requested]


def _same_unity_scene_path(value: object, scene: Path, project: Path) -> bool:
    """Accept either Unity's ``Assets/...`` status path or an absolute path.

    ``EditorSceneManager.GetActiveScene().path`` intentionally reports a
    project-relative AssetDatabase path.  Command-line and YAML configuration,
    on the other hand, resolve scenes to filesystem paths.  Treating the two
    representations as different makes a correctly opened Editor wait until
    timeout.
    """

    if not isinstance(value, str) or not value.strip():
        return False
    normalized = value.strip().replace("\\", "/")
    if normalized == _project_relative_scene_path(scene, project):
        return True
    return _same_resolved_path(value, scene)


def _unity_project_editor_pids(project: Path) -> list[int]:
    """Return Linux Unity Editor processes that own exactly ``project``.

    Unity protects a project with a lock, so spawning a second editor for an
    already open project is both slow and unreliable.  `/proc` gives us the
    command-line project argument without relying on localized `ps` output.
    """

    expected = project.resolve()
    pids: list[int] = []
    try:
        entries = list(Path("/proc").iterdir())
    except OSError:
        return pids
    for entry in entries:
        if not entry.name.isdigit():
            continue
        try:
            parts = [item.decode("utf-8", errors="replace") for item in (entry / "cmdline").read_bytes().split(b"\0") if item]
        except OSError:
            continue
        if not parts or Path(parts[0]).name != "Unity":
            continue
        # Hub launches the interactive editor with ``-projectpath`` on this
        # platform, whereas asset-import workers use ``-projectPath`` plus
        # ``-batchMode``.  Treat the flag case-insensitively and exclude the
        # workers: their PIDs cannot own ExternalPlayController's heartbeat.
        if any(item.lower() == "-batchmode" for item in parts):
            continue
        try:
            index = next(index for index, item in enumerate(parts) if item.lower() == "-projectpath")
            candidate = Path(parts[index + 1]).expanduser().resolve()
        except (StopIteration, IndexError, OSError):
            continue
        if candidate == expected:
            pids.append(int(entry.name))
    return sorted(pids)


def _require_matching_editor_status(
    payload: dict[str, Any],
    *,
    project: Path,
    scene: Path,
    expected_version: str,
    expected_pids: Sequence[int] | None = None,
) -> None:
    """Ensure a fresh status report belongs to the requested Unity scene."""

    if not _same_resolved_path(payload.get("projectPath"), project):
        raise RuntimeError(
            "a Unity ExternalPlayController responded, but it is not the requested project: "
            f"reported={payload.get('projectPath')!r}, requested={project}"
        )
    if not _same_unity_scene_path(payload.get("activeScenePath"), scene, project):
        raise RuntimeError(
            "the requested Unity project is already open with a different active scene; "
            f"reported={payload.get('activeScenePath')!r}, requested={scene}. "
            "Open the requested scene in that Editor, then run the experiment again; "
            "the runner will not start a second Editor for the same project."
        )
    reported_version = str(payload.get("unityVersion", "")).strip()
    if reported_version and reported_version != expected_version:
        raise RuntimeError(
            "the open Unity Editor version does not match the project: "
            f"reported={reported_version}, expected={expected_version}"
        )
    if expected_pids:
        try:
            reported_pid = int(payload.get("processId"))
        except (TypeError, ValueError):
            reported_pid = -1
        if reported_pid not in expected_pids:
            raise RuntimeError(
                "the open Unity project did not identify itself through its own ExternalPlayController; "
                f"reported processId={payload.get('processId')!r}, expected one of {list(expected_pids)}"
            )


def _wait_editor_status(
    path: Path,
    timeout_sec: float,
    predicate,
    description: str,
    *,
    process: _ManagedProcess | None = None,
    log_path: Path | None = None,
) -> None:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        payload = _editor_status(path)
        if payload is not None and predicate(payload):
            return
        if process is not None and process.returncode is not None:
            suffix = f"; see {log_path}" if log_path is not None else ""
            raise RuntimeError(
                f"Unity Editor exited with code {process.returncode} before `{description}`{suffix}"
            )
        time.sleep(0.2)
    raise RuntimeError(f"Unity Editor did not reach `{description}` before timeout; see Unity log")


def _project_editor_version(project: Path) -> str:
    version_file = project / "ProjectSettings" / "ProjectVersion.txt"
    try:
        for line in version_file.read_text(encoding="utf-8").splitlines():
            if line.startswith("m_EditorVersion:"):
                version = line.split(":", 1)[1].strip()
                if version:
                    return version
    except OSError as exc:
        raise RuntimeError(f"could not read Unity project version from {version_file}: {exc}") from exc
    raise RuntimeError(f"Unity project version is missing from {version_file}")


def _editor_version(editor: Path) -> str:
    try:
        completed = subprocess.run(
            [str(editor), "-version"],
            check=False,
            capture_output=True,
            text=True,
            timeout=30.0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not query Unity editor version for {editor}: {exc}") from exc
    version = (completed.stdout or completed.stderr).strip().splitlines()
    if completed.returncode != 0 or not version:
        detail = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(f"Unity editor version query failed for {editor}: {detail}")
    return version[0].strip()


def _launch_unity_gui(
    unity_cfg: dict[str, Any],
    *,
    repo_root: Path,
    logs_dir: Path,
    initial_pose: Sequence[float],
    unity_thruster_command_mode: str,
    simulation_time_scale: float,
    ros2_control_lockstep: bool,
    editor_override: Path | None,
) -> tuple[_ManagedProcess | None, bool]:
    editor = editor_override or _resolve_path(str(unity_cfg["editor_executable"]), repo_root)
    project = _resolve_path(str(unity_cfg["project_path"]), repo_root)
    scene = _resolve_path(str(unity_cfg["scene_path"]), repo_root)
    if not editor.is_file() or not project.is_dir() or not scene.is_file():
        raise RuntimeError(f"Unity GUI launch inputs invalid: editor={editor}, project={project}, scene={scene}")
    expected_version = _project_editor_version(project)
    actual_version = _editor_version(editor)
    if actual_version != expected_version:
        raise RuntimeError(
            "Unity editor/project version mismatch: "
            f"project {project} requires {expected_version}, but configured editor {editor} reports {actual_version}. "
            "Set runner.auto_launch.unity_gui.editor_executable to the matching Unity installation."
        )
    command_file = Path(str(unity_cfg["command_file"])).expanduser()
    status_file = Path(str(unity_cfg["status_file"])).expanduser()
    timeout = float(unity_cfg.get("status_timeout_sec", 300.0))
    process: _ManagedProcess | None = None
    project_pids = _unity_project_editor_pids(project)
    if project_pids:
        # A currently importing editor may not process the status command for
        # several minutes; wait for that *same* editor rather than launching a
        # second process against its project lock.
        existing = _probe_editor_status(command_file, status_file, timeout)
        if existing is None:
            raise RuntimeError(
                f"Unity Editor process(es) {project_pids} already own {project}, but their "
                "ExternalPlayController did not respond. Wait for import/compilation to finish, "
                "then retry; the runner intentionally will not launch another Editor."
            )
        _require_matching_editor_status(
            existing, project=project, scene=scene, expected_version=expected_version, expected_pids=project_pids
        )
        reused = True
    else:
        # Do not overwrite another Unity project's singleton command endpoint.
        # A non-responsive status file is stale and may safely be replaced by
        # the Editor this runner starts below.
        existing = _probe_editor_status(command_file, status_file, min(timeout, 2.0)) if status_file.exists() else None
        if existing is not None:
            _require_matching_editor_status(existing, project=project, scene=scene, expected_version=expected_version)
            raise RuntimeError(
                "a responsive Unity ExternalPlayController already owns the requested endpoint, "
                "but no matching Unity project process was found; refuse to overwrite it."
            )
        for path in (command_file, status_file):
            if path.exists():
                path.unlink()
        process = _ManagedProcess(
            [str(editor), "-projectPath", str(project), "-openFile", str(scene)],
            logs_dir / "unity_editor.log",
        )
        _wait_editor_status(
            status_file,
            timeout,
            lambda payload: "isPlaying" in payload,
            "ExternalPlayController initialization",
            process=process,
            log_path=logs_dir / "unity_editor.log",
        )
        reused = False

    if not reused:
        current = _editor_status(status_file) or {}
        if not _same_unity_scene_path(current.get("activeScenePath"), scene, project):
            asset_path = _project_relative_scene_path(scene, project)
            _write_editor_command(command_file, f"open_scene:{asset_path}")
            _wait_editor_status(
                status_file,
                timeout,
                lambda payload: (
                    payload.get("event") == "scene_opened"
                    or _same_resolved_path(payload.get("activeScenePath"), scene)
                ),
                "opening configured Unity scene",
                process=process,
                log_path=logs_dir / "unity_editor.log",
            )
    _wait_editor_status(
        status_file,
        timeout,
        lambda payload: (
            not bool(payload.get("isUpdating", False))
            and _same_resolved_path(payload.get("projectPath"), project)
            and _same_unity_scene_path(payload.get("activeScenePath"), scene, project)
        ),
        "requested Unity project and scene readiness",
        process=process,
        log_path=logs_dir / "unity_editor.log",
    )
    current = _editor_status(status_file) or {}
    if bool(current.get("isPlayingOrWillChangePlaymode", False)):
        _write_editor_command(command_file, str(unity_cfg.get("stop_command", "stop")))
        _wait_editor_status(
            status_file,
            timeout,
            lambda payload: not bool(payload.get("isPlayingOrWillChangePlaymode", False)),
            "stopping existing Play mode",
            process=process,
            log_path=logs_dir / "unity_editor.log",
        )
    values = [float(value) for value in initial_pose]
    if len(values) != 4:
        raise RuntimeError("Unity initial pose must be [x, y, z, yaw_deg]")
    prepare = str(unity_cfg.get("prepare_command", "prepare_ros_evaluation_scene"))
    prepare_values = [
        *(f"{value:.9g}" for value in values),
        unity_thruster_command_mode,
        f"{float(simulation_time_scale):.9g}",
    ]
    if ros2_control_lockstep:
        prepare_values.append("ros2_control_lockstep")
    _write_editor_command(
        command_file,
        prepare
        + ":"
        + ",".join(prepare_values),
    )
    _wait_editor_status(
        status_file,
        timeout,
        lambda payload: payload.get("event") == "ros_evaluation_scene_prepared"
        or str(payload.get("lastCommand", "")).startswith(prepare),
        "ROS evaluation scene preparation",
        process=process,
        log_path=logs_dir / "unity_editor.log",
    )
    _write_editor_command(command_file, str(unity_cfg.get("start_command", "play")))
    _wait_editor_status(
        status_file,
        timeout,
        lambda payload: bool(payload.get("isPlaying", False)),
        "Play mode",
        process=process,
        log_path=logs_dir / "unity_editor.log",
    )
    return process, reused


def _launch_support_stack(
    config: dict[str, Any],
    *,
    repo_root: Path,
    logs_dir: Path,
    initial_pose: Sequence[float],
    unity_thruster_command_mode: str,
    simulation_clock: dict[str, Any],
    gui: bool,
    editor_override: Path | None,
    launch_unity: bool = True,
) -> tuple[list[_ManagedProcess], _ManagedProcess | None, bool]:
    auto = _mapping(_mapping(config["runner"], "runner").get("auto_launch", {}), "runner.auto_launch")
    processes: list[_ManagedProcess] = []
    adapter = _mapping(auto["grpc_adapter"], "runner.auto_launch.grpc_adapter")
    adapter_script = _resolve_path(str(adapter["script"]), repo_root)
    adapter_args = [str(item) for item in adapter.get("args", [])]
    if simulation_clock["mode"] == "ros2_control_lockstep":
        adapter_args.extend(
            [
                "lockstep_enabled:=true",
                f"lockstep_ack_topic:={simulation_clock['lockstep_ack_topic']}",
                f"lockstep_ack_timeout_wall_sec:={float(simulation_clock['lockstep_ack_timeout_wall_sec']):.9g}",
            ]
        )
    processes.append(_ManagedProcess([str(adapter_script), *adapter_args], logs_dir / "grpc_ros_adapter.log"))
    reset = _mapping(auto["reset_service"], "runner.auto_launch.reset_service")
    reset_command = [
        "ros2", "run", str(reset["package"]), str(reset["executable"]), "--ros-args",
        "-p", f"reset_topic:={reset['reset_topic']}",
        "-p", f"thruster_topic:={reset['thruster_topic']}",
        "-p", f"command_cancel_topic:={reset['cancel_topic']}",
        "-p", f"reset_service_name:={reset['service_name']}",
    ]
    processes.append(_ManagedProcess(reset_command, logs_dir / "reset_service.log"))
    unity_process: _ManagedProcess | None = None
    unity_reused = False
    try:
        if gui and launch_unity:
            unity_process, unity_reused = _launch_unity_gui(
                _mapping(auto["unity_gui"], "runner.auto_launch.unity_gui"),
                repo_root=repo_root,
                logs_dir=logs_dir,
                initial_pose=initial_pose,
                unity_thruster_command_mode=unity_thruster_command_mode,
                simulation_time_scale=float(simulation_clock["time_scale"]),
                ros2_control_lockstep=simulation_clock["mode"] == "ros2_control_lockstep",
                editor_override=editor_override,
            )
    except BaseException:
        _stop_processes(processes, float(auto.get("terminate_timeout_sec", 15.0)))
        raise
    return processes, unity_process, unity_reused


def _controller_command(config: dict[str, Any], controller_config: Path, checkpoint: Path | None, repo_root: Path) -> list[str]:
    auto = _mapping(_mapping(config["runner"], "runner").get("auto_launch", {}), "runner.auto_launch")
    motion = _mapping(auto["motion_controller"], "runner.auto_launch.motion_controller")
    command = [
        "ros2", "launch", str(motion.get("package", "motion_control")),
        str(motion.get("launch_file", "motion_controller.launch.py")),
        f"params_file:={controller_config}",
        f"pool_world_params_file:={_resolve_path(str(motion['pool_world_params_file']), repo_root)}",
        f"state_input_mode:={motion.get('state_input_mode', 'sim_truth')}",
    ]
    if checkpoint is not None:
        command.append(f"checkpoint_path:={checkpoint}")
    return command


def _run_manifest_base(
    *,
    run_id: str,
    config_path: Path,
    config_files: Sequence[Path],
    controller_name: str,
    controller: dict[str, Any],
    controller_config: Path,
    checkpoint: Path | None,
    repo_root: Path,
    gui: bool,
    run_dir: Path,
    task: str,
    method: str,
) -> dict[str, Any]:
    return {
        "schema_version": 2,
        "status": "running",
        "run_id": run_id,
        "run_directory": str(run_dir.relative_to(repo_root)),
        "experiment_layout": {"domain": "simulation", "task": task, "method": method},
        "created_at": now_iso(),
        "protocol_config": str(config_path),
        "protocol_config_sha256": file_sha256(config_path),
        "git_revision": git_revision(repo_root),
        "simulation": True,
        "unity_gui_started_by_runner": gui,
        "controller": controller_name,
        "controller_profile": str(controller_config),
        "controller_profile_sha256": file_sha256(controller_config),
        "controller_contract": controller,
        "checkpoint": str(checkpoint) if checkpoint else None,
        "checkpoint_sha256": file_sha256(checkpoint) if checkpoint else None,
        "config_files": {str(path): file_sha256(path) for path in config_files},
        "trial_plan": [],
        "completed_trials": [],
        "analysis_reports": [],
    }


def _stop_processes(processes: Sequence[_ManagedProcess], timeout_sec: float) -> None:
    for process in reversed(list(processes)):
        try:
            process.stop(timeout_sec)
        except Exception:
            pass


def _parse_t1_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checkpoint-controlled Unity/MARUS T1 simulation experiment.")
    parser.add_argument("--config", type=Path, required=True, help="t1_simulation_experiment.yaml")
    parser.add_argument("--controller", choices=("PID", "PPO"), required=True)
    parser.add_argument(
        "--ppo-action-interface",
        choices=("wrench6", "thruster8"),
        help=(
            "Required with --controller PPO. `wrench6` loads the native 6D physical-wrench policy; "
            "`thruster8` sends the native 8D canonical action directly to Unity."
        ),
    )
    parser.add_argument("--controller-config", type=Path, help="Optional /sim motion-controller YAML override.")
    parser.add_argument("--checkpoint", type=Path, help="Required for PPO unless the selected profile declares one.")
    parser.add_argument("--setpoint", action="append", default=[], help="Repeatable T1 target ID; default runs all protocol targets.")
    parser.add_argument("--repeats", type=int, help="Override repeats per selected setpoint.")
    parser.add_argument("--no-gui", action="store_true", help="Use an already-running prepared Unity /sim scene.")
    parser.add_argument("--unity-editor", type=Path, help="Override Unity editor executable.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the plan without launching Unity/ROS2.")
    return parser.parse_args(argv)


def _resolve_t1_controller_key(args: argparse.Namespace) -> str:
    """Map the user-facing controller/interface pair to a frozen YAML contract."""

    if args.controller == "PID":
        if args.ppo_action_interface is not None:
            raise SystemExit("--ppo-action-interface is valid only with --controller PPO")
        return "PID"
    if args.ppo_action_interface is None:
        raise SystemExit(
            "--controller PPO requires --ppo-action-interface wrench6 or --ppo-action-interface thruster8"
        )
    return {
        "wrench6": "PPO_WRENCH6",
        "thruster8": "PPO_THRUSTER8",
    }[args.ppo_action_interface]


def run_t1_main(argv: Sequence[str] | None = None) -> None:
    args = _parse_t1_args(argv)
    repo_root = _repo_root()
    config_path = _resolve_cli_path(args.config, repo_root)
    config = _load_yaml(config_path, required=("runner", "controllers", "target_points", "trial_protocol", "metric_protocol"), label="T1 simulation")
    controller_key = _resolve_t1_controller_key(args)
    controller, controller_config, checkpoint, controller_params = _resolve_controller(
        config, controller_key, repo_root=repo_root,
        controller_config_override=_resolve_cli_path(args.controller_config, repo_root) if args.controller_config else None,
        checkpoint_override=_resolve_cli_path(args.checkpoint, repo_root) if args.checkpoint else None,
    )
    simulation_clock = _simulation_clock_config(config)
    _validate_simulation_time_controller_profile(controller_params, controller_config, simulation_clock)
    if simulation_clock["mode"] == "ros2_control_lockstep" and args.no_gui and not args.dry_run:
        raise SystemExit(
            "ros2_control_lockstep requires the runner-managed Unity Editor; "
            "do not pass --no-gui because the active scene must install the lockstep driver"
        )
    targets = _mapping(_mapping(config["target_points"], "target_points").get("points"), "target_points.points")
    protocol = _mapping(config["trial_protocol"], "trial_protocol")
    requested = [str(value) for value in args.setpoint] or [str(value) for value in protocol.get("setpoints", targets.keys())]
    unknown = [name for name in requested if name not in targets]
    if unknown:
        raise SystemExit(f"unknown T1 setpoint(s): {unknown}; available={sorted(targets)}")
    repeats = int(args.repeats if args.repeats is not None else protocol.get("repeats_per_controller_and_setpoint", 1))
    if repeats < 1:
        raise SystemExit("--repeats must be positive")
    initial = [float(value) for value in _mapping(config.get("initial_condition", {}), "initial_condition").get("reset_pose_controller_world", [0, 0, 0, 0])]
    if len(initial) != 4:
        raise SystemExit("initial_condition.reset_pose_controller_world must be [x,y,z,yaw_deg]")
    config_files = _config_files(config_path, controller_config, config, repo_root)
    data_root = _resolve_path(
        str(_mapping(_mapping(config["runner"], "runner").get("logging", {}), "runner.logging").get("root", "ros2_ws/data/experiments")),
        repo_root,
    )
    method_bucket = _controller_storage_subdirectory(controller_key, controller)
    run_parent = canonical_run_dir(
        data_root, domain="simulation", task="T1", method=method_bucket, run_id="placeholder"
    ).parent
    run_id = _new_run_id(run_parent, f"{_safe_slug(config.get('experiment_id', 'E5'))}_{_safe_slug(controller.get('label', args.controller))}")
    run_dir = canonical_run_dir(data_root, domain="simulation", task="T1", method=method_bucket, run_id=run_id)
    plan: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for setpoint_id in requested:
            point = _mapping(targets[setpoint_id], f"target_points.points.{setpoint_id}")
            plan.append({
                "setpoint_id": setpoint_id,
                "repeat": repeat,
                "target_controller_world": [float(point["x_m"]), float(point["y_depth_m"]), float(point["z_m"]), float(point["yaw_deg"])],
            })
    manifest = _run_manifest_base(
        run_id=run_id, config_path=config_path, config_files=config_files, controller_name=controller_key,
        controller=controller, controller_config=controller_config, checkpoint=checkpoint,
        repo_root=repo_root, gui=not args.no_gui, run_dir=run_dir, task="T1", method=method_bucket,
    )
    manifest["simulation_clock"] = simulation_clock
    manifest["trial_plan"] = plan
    if args.dry_run:
        manifest["status"] = "planned"
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        return
    _execute_t1(
        config,
        config_path,
        controller_config,
        checkpoint,
        run_id,
        data_root,
        method_bucket,
        manifest,
        plan,
        initial,
        unity_thruster_command_mode=_unity_sim_thruster_command_mode(controller),
        simulation_clock=simulation_clock,
        gui=not args.no_gui,
        editor_override=args.unity_editor,
    )


def _execute_t1(
    config: dict[str, Any], config_path: Path, controller_config: Path, checkpoint: Path | None,
    run_id: str, data_root: Path, method_bucket: str, manifest: dict[str, Any], plan: Sequence[dict[str, Any]],
    initial: Sequence[float], *, unity_thruster_command_mode: str, simulation_clock: dict[str, Any], gui: bool,
    editor_override: Path | None,
) -> None:
    repo_root = _repo_root()
    run_layout = create_run_layout(
        data_root, domain="simulation", task="T1", method=method_bucket, run_id=run_id
    )
    run_dir, trials_dir = run_layout.run_dir, run_layout.trials_dir
    logs_dir = run_layout.shared_logs_dir
    write_json(run_dir / "run_manifest.json", manifest)
    log_stage(
        f"simulation T1 run initialized: controller={manifest['controller']} trials={len(plan)} run={run_dir}"
    )
    runner = _mapping(config["runner"], "runner")
    auto = _mapping(runner["auto_launch"], "runner.auto_launch")
    logging_cfg = _mapping(runner["logging"], "runner.logging")
    reset_cfg = _mapping(auto["reset_service"], "runner.auto_launch.reset_service")
    startup_timeout = float(auto.get("startup_timeout_sec", 120.0))
    stop_timeout = float(auto.get("terminate_timeout_sec", 15.0))
    finalizer_timeout = max(stop_timeout, float(auto.get("recorder_finalize_timeout_sec", 90.0)))
    runtime = _mapping(runner.get("runtime_status", {}), "runner.runtime_status")
    metric = _mapping(config["metric_protocol"], "metric_protocol")
    primary = _mapping(metric.get("evaluation_window", {}), "metric_protocol.evaluation_window")
    response = _mapping(metric.get("response_window", {}), "metric_protocol.response_window")
    protocol = _mapping(config["trial_protocol"], "trial_protocol")
    initial_cfg = _mapping(config["initial_condition"], "initial_condition")
    support_processes: list[_ManagedProcess] = []
    unity_process: _ManagedProcess | None = None
    persistent_controller_process: _ManagedProcess | None = None
    node: _SimulationExperimentNode | None = None
    interrupted = False
    try:
        if simulation_clock["mode"] == "ros2_control_lockstep":
            # The first Unity fixed step blocks until a controller acknowledges
            # its /clock tick.  Launch one controller before entering Play
            # mode and retain it across trials; starting it after a reset (the
            # former free-running T1 sequence) races the 10x simulation clock.
            support_processes, _, _ = _launch_support_stack(
                config,
                repo_root=repo_root,
                logs_dir=logs_dir,
                initial_pose=initial,
                unity_thruster_command_mode=unity_thruster_command_mode,
                simulation_clock=simulation_clock,
                gui=gui,
                editor_override=_resolve_cli_path(editor_override, repo_root) if editor_override else None,
                launch_unity=False,
            )
            persistent_controller_process = _ManagedProcess(
                _controller_command(config, controller_config, checkpoint, repo_root),
                logs_dir / "lockstep.controller.log",
            )
            support_processes.append(persistent_controller_process)
            _require_process_running(persistent_controller_process, "lockstep simulation motion controller")
            unity_process, unity_reused = _launch_unity_gui(
                _mapping(auto["unity_gui"], "runner.auto_launch.unity_gui"),
                repo_root=repo_root,
                logs_dir=logs_dir,
                initial_pose=initial,
                unity_thruster_command_mode=unity_thruster_command_mode,
                simulation_time_scale=float(simulation_clock["time_scale"]),
                ros2_control_lockstep=True,
                editor_override=_resolve_cli_path(editor_override, repo_root) if editor_override else None,
            )
        else:
            support_processes, unity_process, unity_reused = _launch_support_stack(
                config, repo_root=repo_root, logs_dir=logs_dir, initial_pose=initial, gui=gui,
                unity_thruster_command_mode=unity_thruster_command_mode,
                simulation_clock=simulation_clock,
                editor_override=_resolve_cli_path(editor_override, repo_root) if editor_override else None,
            )
        manifest["unity_gui_started_by_runner"] = bool(gui and unity_process is not None)
        manifest["unity_editor_reused"] = bool(gui and unity_reused)
        manifest["unity_editor_retained_after_run"] = bool(gui)
        required_support = [str(value) for value in auto.get("required_support_topics", [])]
        _wait_for_topics(required_support, startup_timeout)
        init_experiment_ros()
        node = _SimulationExperimentNode(
            event_topic=str(logging_cfg["event_topic"]),
            position_goal_topic="/sim/motion_controller/command/position_controller_world",
            trajectory_topic="/sim/motion_controller/command/trajectory",
            cancel_topic=str(reset_cfg["cancel_topic"]),
            thruster_topic=str(reset_cfg["thruster_topic"]),
            reset_service_name=str(reset_cfg["service_name"]),
            pose_topic="/sim/finsrov/controller/pose",
            use_sim_time=bool(simulation_clock["use_sim_time"]),
            clock_stall_timeout_wall_sec=float(simulation_clock["clock_stall_timeout_wall_sec"]),
            active_pose_topic="/sim/motion_controller/status/active_pose",
            lockstep_ack_topic=(
                str(simulation_clock["lockstep_ack_topic"])
                if simulation_clock["mode"] == "ros2_control_lockstep"
                else ""
            ),
        )
        node.wait_for_clock(
            float(simulation_clock["clock_start_timeout_wall_sec"]),
            clock_topic=str(simulation_clock["clock_topic"]),
        )
        node.wait_for_pose(startup_timeout)
        if persistent_controller_process is not None:
            _wait_for_topics([str(value) for value in auto.get("required_controller_topics", [])], startup_timeout)
            _require_process_running(persistent_controller_process, "lockstep simulation motion controller")
        log_stage(
            f"simulation T1 support stack ready; beginning {len(plan)} target acquisitions",
            logger=node.get_logger(),
        )
        for index, item in enumerate(plan, start=1):
            setpoint_id, repeat = str(item["setpoint_id"]), int(item["repeat"])
            session_id = f"{run_id}_E5_{_safe_slug(manifest['controller'])}_{setpoint_id}_R{repeat:02d}"
            trial_layout = prepare_trial_layout(trials_dir, session_id)
            log_stage(
                f"simulation T1 acquisition {index}/{len(plan)}: target={setpoint_id} R{repeat:02d}; "
                "resetting vehicle and preparing recorder",
                logger=node.get_logger(),
            )
            metadata = {
                "test_id": run_id, "run_id": run_id, "simulation": True,
                "controller": manifest["controller"], "strategy": _safe_slug(manifest["controller"]),
                "setpoint_id": setpoint_id, "repeat": repeat, "trial_index": index,
                "target_controller_world": item["target_controller_world"],
                "initial_reset_pose_controller_world": list(initial),
                "protocol_config": str(config_path), "controller_config": str(controller_config),
                "environment": config.get("environment", {}),
                "simulation_clock": simulation_clock,
                "primary_evaluation_window": {
                    "start_event": str(primary.get("start_after_event", "hold_start")),
                    "start_offset_sec": float(primary.get("start_offset_sec", 0.0)),
                    "duration_sec": float(primary.get("duration_sec", protocol["control_duration_sec"])),
                    "resample_hz": float(primary.get("resample_hz", 10.0)),
                    "max_interpolation_gap_sec": float(primary.get("max_interpolation_gap_sec", 0.25)),
                    "minimum_valid_fraction": float(primary.get("minimum_valid_fraction", 0.90)),
                },
                "response_window": {"start_event": str(response.get("start_after_event", "hold_start")), "duration_sec": float(response.get("duration_sec", protocol["control_duration_sec"]))},
                # In lockstep, the first matching controller `active_pose`
                # header is the response t=0.  The repeated runner event is
                # retained as a transport audit marker, but is not allowed to
                # silently substitute for the controller tick in analysis.
                "timing_contract": {
                    "hold_start_mode": (
                        "first_controller_active_goal_tick_v1"
                        if simulation_clock["mode"] == "ros2_control_lockstep"
                        else "experiment_event_v1"
                    ),
                    "event_audit_required": True,
                    "lockstep_ack_audit_required": simulation_clock["mode"] == "ros2_control_lockstep",
                },
            }
            controller_process: _ManagedProcess | None = None
            recorder_process: _ManagedProcess | None = None
            try:
                node.reset_vehicle(startup_timeout)
                node.spin_sleep(float(initial_cfg.get("settle_before_goal_sec", 0.0)), log_interval_sec=float(runtime.get("log_interval_sec", 1.0)), label=f"T1 reset {setpoint_id}")
                if persistent_controller_process is not None:
                    controller_process = persistent_controller_process
                    _require_process_running(controller_process, "lockstep simulation motion controller")
                else:
                    controller_process = _ManagedProcess(
                        _controller_command(config, controller_config, checkpoint, repo_root),
                        trial_layout.logs_dir / "controller.log",
                    )
                    _wait_for_topics([str(value) for value in auto.get("required_controller_topics", [])], startup_timeout)
                    _require_process_running(controller_process, "simulation motion controller")
                node.wait_for_pose(startup_timeout)
                event_subscribers_before_recorder = node.event_subscription_count()
                active_goal_subscribers_before_recorder = node.active_goal_subscription_count()
                ack_subscribers_before_recorder = node.lockstep_ack_subscription_count()
                recorder_process = _ManagedProcess(
                    _record_command(experiment_id="E5", strategy=_safe_slug(manifest["controller"]), session_id=session_id,
                                    output_root=trials_dir, profile=str(logging_cfg["recorder_profile"]), metadata=metadata,
                                    config_files=[Path(path) for path in manifest["config_files"].keys()], checkpoint=checkpoint,
                                    prepared_session=True),
                    trial_layout.logs_dir / "recorder.log",
                )
                _require_process_running(recorder_process, "simulation experiment recorder")
                node.wait_for_new_event_subscriber(event_subscribers_before_recorder, startup_timeout)
                if simulation_clock["mode"] == "ros2_control_lockstep":
                    node.wait_for_active_goal_subscriber(active_goal_subscribers_before_recorder, startup_timeout)
                    node.wait_for_lockstep_ack_subscriber(ack_subscribers_before_recorder, startup_timeout)
                # Discovery does not prove that rosbag2 has opened its writer.
                # Do not consume protocol time while allowing that writer to
                # settle; Unity remains at the safe no-goal state.
                time.sleep(1.0)
                _require_process_running(recorder_process, "simulation experiment recorder")
                log_stage(
                    f"simulation T1 acquisition {index}/{len(plan)}: recorder active; publishing target",
                    logger=node.get_logger(),
                )
                node.publish_event(session_id, "trial_start", setpoint_id, metadata)
                node.publish_event(session_id, "phase_start", "acquisition", metadata)
                node.spin_sleep(float(protocol.get("pre_goal_record_sec", 5.0)), log_interval_sec=float(runtime.get("log_interval_sec", 1.0)), label=f"T1 pre-goal {setpoint_id}")
                active_goal_before = node.active_goal_stamp_ns()
                node.publish_goal(*item["target_controller_world"])
                if simulation_clock["mode"] == "ros2_control_lockstep":
                    activation_ns = node.wait_for_applied_position_goal(
                        baseline_stamp_ns=active_goal_before,
                        x_m=float(item["target_controller_world"][0]),
                        y_m=float(item["target_controller_world"][1]),
                        z_m=float(item["target_controller_world"][2]),
                        yaw_deg=float(item["target_controller_world"][3]),
                        timeout_sec=startup_timeout,
                    )
                    self_time = activation_ns * 1e-9
                    node.get_logger().info(
                        f"simulation T1 target {setpoint_id} activated at controller tick t={self_time:.3f}s"
                    )
                node.publish_event(session_id, "hold_start", setpoint_id, metadata)
                node.spin_sleep(float(protocol.get("control_duration_sec", 60.0)), log_interval_sec=float(runtime.get("log_interval_sec", 1.0)), label=f"T1 holding {setpoint_id}")
                node.publish_event(session_id, "phase_end", "hold", metadata)
                node.stop_motion()
                node.spin_sleep(float(protocol.get("post_goal_record_sec", 5.0)), label=f"T1 post-goal {setpoint_id}")
                node.publish_event(session_id, "trial_end", setpoint_id, metadata)
                log_stage(
                    f"simulation T1 acquisition {index}/{len(plan)}: control finished; "
                    "finalizing rosbag and running automatic analysis",
                    logger=node.get_logger(),
                )
                recorder_process.stop(finalizer_timeout)
                recorder_process = None
                manifest["completed_trials"].append(session_id)
                analysis_record = _trial_analysis_record(trials_dir, run_dir, session_id, "completed")
                manifest["analysis_reports"].append(analysis_record)
                log_stage(
                    f"simulation T1 acquisition {index}/{len(plan)} complete: target={setpoint_id} "
                    f"analysis={analysis_record['status']} report={analysis_record['report']}",
                    logger=node.get_logger(),
                )
                register_trial_runtime_artifacts(
                    run_manifest=manifest,
                    run_dir=run_dir,
                    trial=trial_layout,
                    local_logs=("recorder.log",) if persistent_controller_process is not None else ("controller.log", "recorder.log"),
                    shared_logs=tuple(sorted(logs_dir.glob("*.log"))),
                )
            except KeyboardInterrupt:
                interrupted = True
                metadata["termination_reason"] = "operator_interrupt"
                if node is not None:
                    node.publish_event(session_id, "operator_abort", setpoint_id, metadata)
                    node.stop_motion()
                if recorder_process is not None:
                    recorder_process.stop(finalizer_timeout)
                    recorder_process = None
                manifest.setdefault("interrupted_trials", []).append(session_id)
                manifest["analysis_reports"].append(_trial_analysis_record(trials_dir, run_dir, session_id, "operator_interrupted"))
                register_trial_runtime_artifacts(
                    run_manifest=manifest,
                    run_dir=run_dir,
                    trial=trial_layout,
                    local_logs=("recorder.log",) if persistent_controller_process is not None else ("controller.log", "recorder.log"),
                    shared_logs=tuple(sorted(logs_dir.glob("*.log"))),
                )
                break
            finally:
                if recorder_process is not None:
                    recorder_process.stop(finalizer_timeout)
                if controller_process is not None and controller_process is not persistent_controller_process:
                    controller_process.stop(stop_timeout)
                write_json(run_dir / "run_manifest.json", manifest)
            if index < len(plan):
                interval = float(protocol.get("inter_trial_interval_sec", 0.0))
                log_stage(
                    f"simulation T1 acquisition {index}/{len(plan)} complete; waiting {interval:.1f}s "
                    f"before acquisition {index + 1}/{len(plan)}",
                    logger=node.get_logger(),
                )
                node.spin_sleep(interval)
        manifest["status"] = "interrupted" if interrupted else "complete"
    except KeyboardInterrupt:
        # This includes Ctrl-C during Unity Editor startup, before an inner
        # trial-level interrupt handler exists.  The finally block still owns
        # the support-process teardown, so do not surface a traceback or leave
        # /sim bridge/reset nodes behind.
        manifest["status"] = "interrupted"
        manifest["termination_reason"] = "operator_interrupt_during_startup_or_between_trials"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        if node is not None:
            try:
                node.stop_motion()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if gui:
            try:
                unity_cfg = _mapping(auto["unity_gui"], "runner.auto_launch.unity_gui")
                if bool(unity_cfg.get("stop_play_mode_on_exit", True)):
                    _write_editor_command(Path(str(unity_cfg["command_file"])), str(unity_cfg.get("stop_command", "stop")))
                    time.sleep(0.5)
            except Exception:
                pass
        _stop_processes(support_processes, stop_timeout)
        write_json(run_dir / "run_manifest.json", manifest)
        log_stage(
            f"simulation T1 run finished: status={manifest['status']} "
            f"completed={len(manifest.get('completed_trials', []))} run={run_dir}"
        )


def _trial_analysis_record(trials_dir: Path, run_dir: Path, session_id: str, outcome: str) -> dict[str, Any]:
    trial_manifest = read_json(trials_dir / session_id / "manifest.json") if (trials_dir / session_id / "manifest.json").is_file() else {}
    return {
        "session_id": session_id,
        "outcome": outcome,
        "status": trial_manifest.get("analysis", {}).get("status", "needs_review"),
        "report": str(trials_dir / session_id / "derived" / "analysis_report.json"),
        "plots": _copy_trial_plots(run_dir, session_id),
    }


def _parse_t2_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a checkpoint-controlled Unity/MARUS T2 simulation experiment.")
    parser.add_argument("--config", type=Path, required=True, help="t2_simulation_experiment.yaml")
    parser.add_argument(
        "--controller",
        choices=("PID_POSITION", "PPO_WRENCH6", "PPO_THRUSTER8"),
        default="PPO_WRENCH6",
        help="T2 controller contract declared in t2_simulation_experiment.yaml.",
    )
    parser.add_argument("--controller-config", type=Path, help="Optional /sim motion-controller YAML override.")
    parser.add_argument("--checkpoint", type=Path, help="Checkpoint override.")
    parser.add_argument("--trajectory", required=True, help="Exactly one trajectory ID from the frozen T2 catalog.")
    parser.add_argument("--repeats", type=int, help="Override repeats for the selected trajectory.")
    parser.add_argument("--no-gui", action="store_true", help="Use an already-running prepared Unity /sim scene.")
    parser.add_argument("--unity-editor", type=Path, help="Override Unity editor executable.")
    parser.add_argument("--dry-run", action="store_true", help="Resolve and print the plan without launching Unity/ROS2.")
    return parser.parse_args(argv)


def run_t2_main(argv: Sequence[str] | None = None) -> None:
    args = _parse_t2_args(argv)
    repo_root = _repo_root()
    config_path = _resolve_cli_path(args.config, repo_root)
    config = _load_yaml(config_path, required=("runner", "controllers", "trial_protocol", "metric_protocol"), label="T2 simulation")
    controller, controller_config, checkpoint, controller_params = _resolve_controller(
        config, args.controller, repo_root=repo_root,
        controller_config_override=_resolve_cli_path(args.controller_config, repo_root) if args.controller_config else None,
        checkpoint_override=_resolve_cli_path(args.checkpoint, repo_root) if args.checkpoint else None,
    )
    simulation_clock = _simulation_clock_config(config)
    _validate_simulation_time_controller_profile(controller_params, controller_config, simulation_clock)
    if simulation_clock["mode"] == "ros2_control_lockstep" and args.no_gui and not args.dry_run:
        raise SystemExit(
            "ros2_control_lockstep requires the runner-managed Unity Editor; "
            "do not pass --no-gui because the active scene must install the lockstep driver"
        )
    source_path = _resolve_path(str(config.get("trajectory_source_config", "")), repo_root)
    source = _load_yaml(source_path, required=("trajectory_contract", "coordinate_contract", "surveyed_workspace", "trajectories"), label="T2 trajectory source")
    catalog = _build_catalog(source)
    if args.trajectory not in catalog:
        raise SystemExit(f"unknown/disabled T2 trajectory `{args.trajectory}`; available={sorted(catalog)}")
    points, diagnostics = catalog[args.trajectory]
    if not points:
        raise SystemExit(f"trajectory `{args.trajectory}` generated no reference points")
    protocol = _mapping(config["trial_protocol"], "trial_protocol")
    repeats = int(args.repeats if args.repeats is not None else protocol.get("repeats_per_controller_and_trajectory", 1))
    if repeats < 1:
        raise SystemExit("--repeats must be positive")
    # T2 follows current hardware semantics: a shallower declared placement
    # precedes a -0.5 m reference.  In simulation it becomes the Unity reset
    # pose, making the initial transient explicit rather than silently reset
    # at the reference point.
    declared_y = float(_mapping(source["coordinate_contract"], "coordinate_contract").get("declared_initial_placement", {}).get("y_m", -0.20))
    initial = [float(points[0].x_m), declared_y, float(points[0].z_m), 0.0]
    config_files = _config_files(config_path, controller_config, config, repo_root, extra=(source_path,))
    data_root = _resolve_path(
        str(_mapping(_mapping(config["runner"], "runner")["logging"], "runner.logging").get("root", "ros2_ws/data/experiments")),
        repo_root,
    )
    method_bucket = _controller_storage_subdirectory(args.controller, controller)
    run_parent = canonical_run_dir(
        data_root, domain="simulation", task="T2", method=method_bucket, run_id="placeholder"
    ).parent
    run_id = _new_run_id(run_parent, f"{_safe_slug(config.get('experiment_id', 'E9'))}_{_safe_slug(controller.get('label', args.controller))}_{_safe_slug(args.trajectory)}")
    run_dir = canonical_run_dir(data_root, domain="simulation", task="T2", method=method_bucket, run_id=run_id)
    manifest = _run_manifest_base(
        run_id=run_id, config_path=config_path, config_files=config_files, controller_name=args.controller,
        controller=controller, controller_config=controller_config, checkpoint=checkpoint,
        repo_root=repo_root, gui=not args.no_gui, run_dir=run_dir, task="T2", method=method_bucket,
    )
    manifest.update({
        "simulation_clock": simulation_clock,
        "trajectory_source_config": str(source_path),
        "trajectory_source_config_sha256": file_sha256(source_path),
        "trajectory_selection": {"id": args.trajectory, "diagnostics": diagnostics},
        "trial_plan": [{"trajectory_id": args.trajectory, "repeat": repeat, "duration_sec": points[-1].time_sec} for repeat in range(1, repeats + 1)],
    })
    if args.dry_run:
        manifest["status"] = "planned"
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        return
    _execute_t2(
        config,
        source,
        config_path,
        source_path,
        controller_config,
        checkpoint,
        run_id,
        data_root,
        method_bucket,
        manifest,
        args.trajectory,
        points,
        diagnostics,
        initial,
        repeats,
        unity_thruster_command_mode=_unity_sim_thruster_command_mode(controller),
        simulation_clock=simulation_clock,
        gui=not args.no_gui,
        editor_override=args.unity_editor,
    )


def _execute_t2(
    config: dict[str, Any], source: dict[str, Any], config_path: Path, source_path: Path, controller_config: Path,
    checkpoint: Path | None, run_id: str, data_root: Path, method_bucket: str, manifest: dict[str, Any], trajectory_id: str,
    points: Sequence[ReferencePoint], diagnostics: dict[str, float], initial: Sequence[float], repeats: int,
    *, unity_thruster_command_mode: str, simulation_clock: dict[str, Any], gui: bool, editor_override: Path | None,
) -> None:
    repo_root = _repo_root()
    run_layout = create_run_layout(
        data_root, domain="simulation", task="T2", method=method_bucket, run_id=run_id
    )
    run_dir, trials_dir, logs_dir = run_layout.run_dir, run_layout.trials_dir, run_layout.shared_logs_dir
    refs_dir = run_dir / "resolved_trajectories"
    refs_dir.mkdir()
    reference_path = refs_dir / f"{trajectory_id}.csv"
    write_reference_csv(reference_path, points)
    manifest["resolved_trajectory"] = {"path": str(reference_path), "sha256": file_sha256(reference_path)}
    write_json(run_dir / "run_manifest.json", manifest)
    log_stage(
        f"simulation T2 run initialized: controller={manifest['controller']} trajectory={trajectory_id} "
        f"trials={repeats} run={run_dir}"
    )
    runner = _mapping(config["runner"], "runner")
    auto = _mapping(runner["auto_launch"], "runner.auto_launch")
    logging_cfg = _mapping(runner["logging"], "runner.logging")
    reset_cfg = _mapping(auto["reset_service"], "runner.auto_launch.reset_service")
    startup_timeout = float(auto.get("startup_timeout_sec", 120.0))
    stop_timeout = float(auto.get("terminate_timeout_sec", 15.0))
    finalizer_timeout = max(stop_timeout, float(auto.get("recorder_finalize_timeout_sec", 90.0)))
    runtime = _mapping(runner.get("runtime_status", {}), "runner.runtime_status")
    protocol = _mapping(config["trial_protocol"], "trial_protocol")
    metric = _mapping(config["metric_protocol"], "metric_protocol")
    evaluation = _mapping(metric["evaluation_window"], "metric_protocol.evaluation_window")
    trajectory_spec = _mapping(
        _mapping(source["trajectories"], "trajectory-source trajectories").get(trajectory_id),
        f"trajectory-source trajectories.{trajectory_id}",
    )
    coordinate_transform = {
        "schema": "unity_controller_world_identity_v1",
        "pool_world_frame": "controller_world",
        "controller_world_frame": "controller_world",
        "basis_indices": [0, 1, 2], "basis_signs": [1.0, 1.0, 1.0], "post_basis_offset_m": [0.0, 0.0, 0.0],
        "provenance": "Unity VehicleRosBridge simulation truth",
    }
    support_processes: list[_ManagedProcess] = []
    unity_process: _ManagedProcess | None = None
    persistent_controller_process: _ManagedProcess | None = None
    node: _SimulationExperimentNode | None = None
    interrupted = False
    try:
        if simulation_clock["mode"] == "ros2_control_lockstep":
            # Unity's first fixed step waits for a controller acknowledgement.
            # Therefore the controller has to exist before entering Play mode;
            # launching it per trial (the free-running implementation) would
            # deadlock the first handshake.  A new trajectory callback still
            # resets the selected PPO/PID backend before every trial.
            support_processes, _, _ = _launch_support_stack(
                config,
                repo_root=repo_root,
                logs_dir=logs_dir,
                initial_pose=initial,
                unity_thruster_command_mode=unity_thruster_command_mode,
                simulation_clock=simulation_clock,
                gui=gui,
                editor_override=_resolve_cli_path(editor_override, repo_root) if editor_override else None,
                launch_unity=False,
            )
            persistent_controller_process = _ManagedProcess(
                _controller_command(config, controller_config, checkpoint, repo_root),
                logs_dir / "lockstep.controller.log",
            )
            support_processes.append(persistent_controller_process)
            _require_process_running(persistent_controller_process, "lockstep simulation motion controller")
            unity_process, unity_reused = _launch_unity_gui(
                _mapping(auto["unity_gui"], "runner.auto_launch.unity_gui"),
                repo_root=repo_root,
                logs_dir=logs_dir,
                initial_pose=initial,
                unity_thruster_command_mode=unity_thruster_command_mode,
                simulation_time_scale=float(simulation_clock["time_scale"]),
                ros2_control_lockstep=True,
                editor_override=_resolve_cli_path(editor_override, repo_root) if editor_override else None,
            )
        else:
            support_processes, unity_process, unity_reused = _launch_support_stack(
                config,
                repo_root=repo_root,
                logs_dir=logs_dir,
                initial_pose=initial,
                unity_thruster_command_mode=unity_thruster_command_mode,
                simulation_clock=simulation_clock,
                gui=gui,
                editor_override=_resolve_cli_path(editor_override, repo_root) if editor_override else None,
            )
        manifest["unity_gui_started_by_runner"] = bool(gui and unity_process is not None)
        manifest["unity_editor_reused"] = bool(gui and unity_reused)
        manifest["unity_editor_retained_after_run"] = bool(gui)
        _wait_for_topics([str(value) for value in auto.get("required_support_topics", [])], startup_timeout)
        init_experiment_ros()
        node = _SimulationExperimentNode(
            event_topic=str(logging_cfg["event_topic"]),
            position_goal_topic="/sim/motion_controller/command/position_controller_world",
            trajectory_topic="/sim/motion_controller/command/trajectory",
            cancel_topic=str(reset_cfg["cancel_topic"]), thruster_topic=str(reset_cfg["thruster_topic"]),
            reset_service_name=str(reset_cfg["service_name"]), pose_topic="/sim/finsrov/controller/pose",
            use_sim_time=bool(simulation_clock["use_sim_time"]),
            clock_stall_timeout_wall_sec=float(simulation_clock["clock_stall_timeout_wall_sec"]),
            trajectory_reference_stamped_topic="/sim/motion_controller/debug/trajectory_reference_stamped",
        )
        node.wait_for_clock(
            float(simulation_clock["clock_start_timeout_wall_sec"]),
            clock_topic=str(simulation_clock["clock_topic"]),
        )
        node.wait_for_pose(startup_timeout)
        if persistent_controller_process is not None:
            _wait_for_topics(_t2_required_controller_topics(config, str(manifest["controller"])), startup_timeout)
            _require_process_running(persistent_controller_process, "lockstep simulation motion controller")
        log_stage(
            f"simulation T2 support stack ready; beginning {repeats} {trajectory_id} trajectory acquisitions",
            logger=node.get_logger(),
        )
        for repeat in range(1, repeats + 1):
            session_id = f"{run_id}_E9_{_safe_slug(manifest['controller'])}_{_safe_slug(trajectory_id)}_R{repeat:02d}"
            trial_layout = prepare_trial_layout(trials_dir, session_id)
            log_stage(
                f"simulation T2 acquisition {repeat}/{repeats}: trajectory={trajectory_id} R{repeat:02d}; "
                "resetting vehicle and preparing recorder",
                logger=node.get_logger(),
            )
            metadata = {
                "test_id": run_id, "run_id": run_id, "simulation": True,
                "controller": manifest["controller"], "trajectory_id": trajectory_id, "repeat": repeat,
                "protocol_config": str(config_path), "trajectory_source_config": str(source_path),
                "controller_config": str(controller_config), "environment": config.get("environment", {}),
                "simulation_clock": simulation_clock,
                "trajectory_geometry": {
                    "generator": str(trajectory_spec.get("generator", "")),
                    "turns": trajectory_spec.get("turns"),
                    **diagnostics,
                    "source_config": str(source_path),
                },
                "declared_initial_controller_world": list(initial),
                "reference_start_controller_world": [points[0].x_m, points[0].y_m, points[0].z_m],
                "coordinate_transform": coordinate_transform,
                "evaluation_window": {
                    "start_event": str(evaluation.get("start_after_event", "trajectory_start")),
                    "duration_sec": float(points[-1].time_sec), "resample_hz": float(evaluation.get("resample_hz", 10.0)),
                    "max_interpolation_gap_sec": float(evaluation.get("max_interpolation_gap_sec", 0.25)),
                    "minimum_valid_fraction": float(evaluation.get("minimum_valid_fraction", 0.90)),
                },
                # The event marker remains the immutable recorder audit
                # boundary.  Under ROS2 lockstep, however, the controller
                # activates a received trajectory on its next exact /clock
                # tick.  The stamped controller reference is consequently
                # the authoritative t=0 for time alignment; this declaration
                # prevents the analyser from mixing it with a lagging
                # external runner clock.
                "timing_contract": {
                    "trajectory_start_mode": (
                        "first_controller_tick_v1"
                        if simulation_clock["mode"] == "ros2_control_lockstep"
                        else "experiment_event_v1"
                    ),
                    "event_audit_required": True,
                },
                "completion_contract": {
                    "required_progress_fraction": 0.98, "required_state_fresh_fraction": 0.90,
                    "status_max_age_sec": 0.25, "allowed_vision_modes": ["fresh", "coast"],
                    "forbidden_events": ["operator_abort", "state_timeout", "controller_fault", "unity_process_exit"],
                },
                "controlled_dofs": ["x", "y_depth", "z"], "orientation_control": "disabled",
            }
            controller_process: _ManagedProcess | None = None
            recorder_process: _ManagedProcess | None = None
            try:
                node.reset_vehicle(startup_timeout)
                node.spin_sleep(float(auto.get("startup_settle_sec", 3.0)), log_interval_sec=float(runtime.get("log_interval_sec", 1.0)), label=f"T2 reset {trajectory_id}")
                if persistent_controller_process is not None:
                    controller_process = persistent_controller_process
                    _require_process_running(controller_process, "lockstep simulation motion controller")
                else:
                    controller_process = _ManagedProcess(
                        _controller_command(config, controller_config, checkpoint, repo_root),
                        trial_layout.logs_dir / "controller.log",
                    )
                    _wait_for_topics(_t2_required_controller_topics(config, str(manifest["controller"])), startup_timeout)
                    _require_process_running(controller_process, "simulation motion controller")
                node.wait_for_pose(startup_timeout)
                event_subscribers_before_recorder = node.event_subscription_count()
                reference_subscribers_before_recorder = node.trajectory_reference_subscription_count()
                recorder_process = _ManagedProcess(
                    _record_command(experiment_id="E9", strategy=_safe_slug(manifest["controller"]), session_id=session_id,
                                    output_root=trials_dir, profile=str(logging_cfg["recorder_profile"]), metadata=metadata,
                                    config_files=[Path(path) for path in manifest["config_files"].keys()] + [reference_path], checkpoint=checkpoint,
                                    prepared_session=True),
                    trial_layout.logs_dir / "recorder.log",
                )
                _require_process_running(recorder_process, "simulation experiment recorder")
                node.wait_for_new_event_subscriber(event_subscribers_before_recorder, startup_timeout)
                node.wait_for_trajectory_reference_subscriber(reference_subscribers_before_recorder, startup_timeout)
                # A DDS match proves discovery, but rosbag2 can still be
                # creating its SQLite writer immediately afterwards.  Keep
                # Unity in its no-trajectory state for this short wall-clock
                # settling interval; no protocol time or reference sample is
                # consumed, and the first applied T2 tick remains auditable.
                time.sleep(1.0)
                _require_process_running(recorder_process, "simulation experiment recorder")
                log_stage(
                    f"simulation T2 acquisition {repeat}/{repeats}: recorder active; "
                    f"trajectory duration={points[-1].time_sec:.1f}s",
                    logger=node.get_logger(),
                )
                node.publish_event(session_id, "trial_start", trajectory_id, metadata)
                node.publish_event(session_id, "phase_start", "pre_trajectory", metadata)
                node.publish_event(session_id, "trajectory_start", trajectory_id, metadata)
                reference_before_trajectory = node.trajectory_reference_stamp_ns()
                node.publish_trajectory(points)
                if simulation_clock["mode"] == "ros2_control_lockstep":
                    node.wait_for_lockstep_trajectory_progress(
                        baseline_stamp_ns=reference_before_trajectory,
                        duration_sec=float(points[-1].time_sec),
                        timeout_sec=startup_timeout,
                        log_interval_sec=float(runtime.get("log_interval_sec", 1.0)),
                        label=f"T2 {trajectory_id} R{repeat}",
                    )
                else:
                    node.spin_sleep(
                        float(points[-1].time_sec),
                        log_interval_sec=float(runtime.get("log_interval_sec", 1.0)),
                        label=f"T2 {trajectory_id} R{repeat}",
                    )
                node.publish_event(session_id, "trajectory_end", trajectory_id, metadata)
                node.stop_motion()
                node.publish_event(session_id, "phase_end", "trajectory", metadata)
                node.spin_sleep(float(protocol.get("post_trajectory_record_sec", 7.0)), label=f"T2 post {trajectory_id}")
                node.publish_event(session_id, "trial_end", trajectory_id, metadata)
                log_stage(
                    f"simulation T2 acquisition {repeat}/{repeats}: trajectory finished; "
                    "finalizing rosbag and running automatic analysis",
                    logger=node.get_logger(),
                )
                recorder_process.stop(finalizer_timeout)
                recorder_process = None
                manifest["completed_trials"].append(session_id)
                analysis_record = _trial_analysis_record(trials_dir, run_dir, session_id, "completed")
                manifest["analysis_reports"].append(analysis_record)
                log_stage(
                    f"simulation T2 acquisition {repeat}/{repeats} complete: trajectory={trajectory_id} "
                    f"analysis={analysis_record['status']} report={analysis_record['report']}",
                    logger=node.get_logger(),
                )
                register_trial_runtime_artifacts(
                    run_manifest=manifest,
                    run_dir=run_dir,
                    trial=trial_layout,
                    local_logs=("recorder.log",) if persistent_controller_process is not None else ("controller.log", "recorder.log"),
                    shared_logs=tuple(sorted(logs_dir.glob("*.log"))),
                )
            except KeyboardInterrupt:
                interrupted = True
                metadata["termination_reason"] = "operator_interrupt"
                if node is not None:
                    node.publish_event(session_id, "operator_abort", trajectory_id, metadata)
                    node.stop_motion()
                if recorder_process is not None:
                    recorder_process.stop(finalizer_timeout)
                    recorder_process = None
                manifest.setdefault("interrupted_trials", []).append(session_id)
                manifest["analysis_reports"].append(_trial_analysis_record(trials_dir, run_dir, session_id, "operator_interrupted"))
                register_trial_runtime_artifacts(
                    run_manifest=manifest,
                    run_dir=run_dir,
                    trial=trial_layout,
                    local_logs=("recorder.log",) if persistent_controller_process is not None else ("controller.log", "recorder.log"),
                    shared_logs=tuple(sorted(logs_dir.glob("*.log"))),
                )
                break
            finally:
                if recorder_process is not None:
                    recorder_process.stop(finalizer_timeout)
                if controller_process is not None and controller_process is not persistent_controller_process:
                    controller_process.stop(stop_timeout)
                write_json(run_dir / "run_manifest.json", manifest)
            if repeat < repeats:
                interval = float(protocol.get("inter_trial_interval_sec", 0.0))
                log_stage(
                    f"simulation T2 acquisition {repeat}/{repeats} complete; waiting {interval:.1f}s "
                    f"before acquisition {repeat + 1}/{repeats}",
                    logger=node.get_logger(),
                )
                node.spin_sleep(interval)
        manifest["status"] = "interrupted" if interrupted else "complete"
    except KeyboardInterrupt:
        manifest["status"] = "interrupted"
        manifest["termination_reason"] = "operator_interrupt_during_startup_or_between_trials"
    except Exception as exc:
        manifest["status"] = "failed"
        manifest["error"] = repr(exc)
        raise
    finally:
        if node is not None:
            try:
                node.stop_motion()
            except Exception:
                pass
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
        if gui:
            try:
                unity_cfg = _mapping(auto["unity_gui"], "runner.auto_launch.unity_gui")
                if bool(unity_cfg.get("stop_play_mode_on_exit", True)):
                    _write_editor_command(Path(str(unity_cfg["command_file"])), str(unity_cfg.get("stop_command", "stop")))
                    time.sleep(0.5)
            except Exception:
                pass
        _stop_processes(support_processes, stop_timeout)
        write_json(run_dir / "run_manifest.json", manifest)
        log_stage(
            f"simulation T2 run finished: status={manifest['status']} "
            f"completed={len(manifest.get('completed_trials', []))} run={run_dir}"
        )


__all__ = ["run_t1_main", "run_t2_main"]
