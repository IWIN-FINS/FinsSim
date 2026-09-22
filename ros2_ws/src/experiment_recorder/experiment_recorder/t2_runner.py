"""Automatic T2 hardware trajectory experiment runner.

Unlike the T1 runner, this runner sends one complete
``MultiDOFJointTrajectory`` per trial.  The active experiment configuration,
geometry, controller profile, metric window, and safety contract are all
resolved from ``t2_hardware_experiment.yaml``.  A PPO checkpoint is always
read from the selected controller profile, rather than duplicated in the T2
protocol YAML.  The runner snapshots that profile and its training metadata,
then records SHA-256 values for both artifacts and the checkpoint.  T2
supports both the trajectory30 PPO policies and an explicitly translation-only
traditional position-PID baseline.  The PID baseline samples the moving
position reference directly; it is not a velocity-feedforward controller.
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
import time
from typing import Any, Sequence

from geometry_msgs.msg import PoseWithCovarianceStamped, Transform, Twist
import rclpy
from std_msgs.msg import String
from trajectory_msgs.msg import MultiDOFJointTrajectory, MultiDOFJointTrajectoryPoint
import yaml

from .layout import (
    canonical_run_dir,
    create_run_layout,
    prepare_trial_layout,
    register_trial_runtime_artifacts,
    shared_log_path,
)
from .manifest import file_sha256, git_revision, now_iso, read_json, write_json
from .progress import log_stage
from .runner import (
    _ManagedProcess,
    _RosExperimentNode,
    _as_mapping,
    _controller_storage_subdirectory,
    _config_files,
    init_experiment_ros,
    _launch_commands,
    _record_command,
    _require_process_running,
    _resolve_path,
    _wait_for_topics,
)
from .t2_trajectory import ReferencePoint, generate_trajectory, validate_reference, write_reference_csv


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"T2 experiment YAML not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid T2 experiment YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"T2 experiment YAML must contain a mapping: {path}")
    for key in (
        "runner", "coordinate_contract", "controllers",
        "trajectory_contract", "surveyed_workspace", "trajectories",
        "trial_protocol", "metric_protocol", "recording",
    ):
        if not isinstance(payload.get(key), dict):
            raise SystemExit(f"T2 experiment YAML requires mapping `{key}`")
    return payload


def _new_run_id(root: Path, prefix: str) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{prefix}"
    candidate = base
    suffix = 2
    while (root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _strategy_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "-_" else "_" for char in value).strip("_") or "UNSPECIFIED"


def _optional_file_provenance(key: str, path: Path | None) -> dict[str, str | None]:
    """Serialize a policy artifact when a controller has one.

    Traditional PID is deterministic from its resolved YAML and deliberately
    has no checkpoint.  Recording null here makes that distinction explicit
    instead of hashing the workspace directory by accident.
    """

    return {
        key: str(path) if path is not None else None,
        f"{key}_sha256": file_sha256(path) if path is not None else None,
    }


def _snapshot_controller_provenance(
    run_dir: Path,
    *,
    controller_config: Path,
    checkpoint_manifest: Path | None,
) -> dict[str, dict[str, str]]:
    """Copy small, mutable provenance inputs into an immutable run artifact.

    Checkpoints can be hundreds of megabytes and are therefore identified by
    their SHA-256 rather than copied into every rosbag run.  Controller YAML
    and training metadata are small and are copied, so a later edit or move
    cannot obscure exactly which deployment profile and training record were
    used for a hardware session.
    """

    provenance_dir = run_dir / "provenance"
    provenance_dir.mkdir()
    sources: list[tuple[str, Path, str]] = [
        ("controller_profile", controller_config, "controller_profile.yaml"),
    ]
    if checkpoint_manifest is not None:
        sources.append(("checkpoint_manifest", checkpoint_manifest, "checkpoint_metadata.json"))
    snapshots: dict[str, dict[str, str]] = {}
    for key, source, name in sources:
        destination = provenance_dir / name
        shutil.copy2(source, destination)
        snapshots[key] = {
            "source": str(source),
            "source_sha256": file_sha256(source),
            "snapshot": str(destination.relative_to(run_dir)),
            "snapshot_sha256": file_sha256(destination),
        }
    return snapshots


def _global_motion_controller_nodes(entries: Sequence[tuple[str, str]]) -> list[str]:
    """Return real-hardware controller nodes that would contend for commands."""

    return ["/motion_controller" for name, namespace in entries if name == "motion_controller" and namespace in ("", "/")]


class _T2ExperimentNode(_RosExperimentNode):
    def __init__(
        self,
        event_topic: str,
        *,
        trajectory_topic: str,
        bridge_node_name: str,
        bridge_status_topic: str,
        pose_topic: str,
        state_status_topic: str,
    ) -> None:
        super().__init__(
            event_topic,
            bridge_node_name=bridge_node_name,
            bridge_status_topic=bridge_status_topic,
            node_name="finsrov_t2_experiment_runner",
        )
        self._trajectory_pub = self.create_publisher(MultiDOFJointTrajectory, trajectory_topic, 10)
        self._latest_pose: PoseWithCovarianceStamped | None = None
        self._latest_pose_received_monotonic: float | None = None
        self._latest_state_status: dict[str, Any] | None = None
        self._latest_state_status_received_monotonic: float | None = None
        self._last_runtime_health: str | None = None
        # ``controller_state_adapter`` publishes PoseWithCovarianceStamped.
        # Subscribing as PoseStamped silently creates an incompatible ROS2
        # endpoint, so the old runner never received the position it printed
        # and could not prove that the recorder had a controller-world pose.
        self.create_subscription(PoseWithCovarianceStamped, pose_topic, self._pose_callback, 20)
        self.create_subscription(String, state_status_topic, self._state_status_callback, 20)

    def _pose_callback(self, message: PoseWithCovarianceStamped) -> None:
        self._latest_pose = message
        self._latest_pose_received_monotonic = time.monotonic()

    def wait_for_controller_pose(self, timeout_sec: float) -> dict[str, Any]:
        """Require one controller-world pose before a T2 bag is started.

        The trajectory reference is expressed in ``controller_world``.  A
        topic name appearing in the ROS graph is not enough: a stale
        subscriber can create the name even when the adapter has published no
        messages.  Starting only after a received pose protects both runtime
        logging and the reference-versus-measurement analysis contract.
        """

        deadline = time.monotonic() + max(float(timeout_sec), 0.0)
        while rclpy.ok() and time.monotonic() < deadline:
            if self._latest_pose is not None and self._latest_pose_received_monotonic is not None:
                position = self._latest_pose.pose.pose.position
                return {
                    "frame_id": str(self._latest_pose.header.frame_id),
                    "position_controller_world_m": [
                        float(position.x),
                        float(position.y),
                        float(position.z),
                    ],
                    "received_before_recording": True,
                }
            rclpy.spin_once(self, timeout_sec=min(0.1, max(deadline - time.monotonic(), 0.01)))
        raise RuntimeError(
            "no PoseWithCovarianceStamped message was received on "
            "/finsrov/controller/pose before the T2 trial; refusing to record "
            "a controller_world reference without the matching state stream"
        )

    def clear_controller_pose_preflight(self) -> None:
        """Discard a pre-existing pose before launching this trial's controller."""

        self._latest_pose = None
        self._latest_pose_received_monotonic = None

    def _state_status_callback(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except (TypeError, json.JSONDecodeError):
            return
        if isinstance(payload, dict):
            self._latest_state_status = payload
            self._latest_state_status_received_monotonic = time.monotonic()

    def warn_existing_motion_controller(self) -> list[str]:
        """Warn without blocking if a previous real-controller node survived."""

        existing = _global_motion_controller_nodes(self.get_node_names_and_namespaces())
        if existing:
            self.get_logger().warning(
                "[WARN] existing /motion_controller detected before T2 startup; "
                "it may contend for /finsrov/thrusters_out. Stop the stale controller before arming."
            )
        return existing

    def publish_trajectory(self, points: Sequence[ReferencePoint]) -> None:
        message = MultiDOFJointTrajectory()
        message.header.frame_id = "controller_world"
        for reference in points:
            point = MultiDOFJointTrajectoryPoint()
            transform = Transform()
            transform.translation.x = float(reference.x_m)
            transform.translation.y = float(reference.y_m)
            transform.translation.z = float(reference.z_m)
            # The trajectory30 transport requires a Transform, but T2 does not
            # command attitude.  Identity makes that non-control boundary
            # explicit; all rotational wrench components are zeroed by the
            # selected motion-controller profile.
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
        message.header.stamp = self.get_clock().now().to_msg()
        self._trajectory_pub.publish(message)
        rclpy.spin_once(self, timeout_sec=0.05)

    def _runtime_health(
        self,
        *,
        pose_stale_after_sec: float,
        allowed_vision_modes: set[str],
    ) -> tuple[str, str]:
        """Return printable position and a conservative state-health label."""

        now = time.monotonic()
        issues: list[str] = []
        position = "unavailable"
        if self._latest_pose is None or self._latest_pose_received_monotonic is None:
            issues.append("pose_missing")
        else:
            age = now - self._latest_pose_received_monotonic
            pose = self._latest_pose.pose.pose.position
            position = (
                f"[{pose.x:+.3f}, {pose.y:+.3f}, {pose.z:+.3f}] m"
            )
            if pose_stale_after_sec > 0.0 and age > pose_stale_after_sec:
                issues.append(f"pose_stale:{age:.2f}s")

        status = self._latest_state_status
        if status is None or self._latest_state_status_received_monotonic is None:
            issues.append("state_status_missing")
        else:
            age = now - self._latest_state_status_received_monotonic
            if pose_stale_after_sec > 0.0 and age > pose_stale_after_sec:
                issues.append(f"state_status_stale:{age:.2f}s")
            for key in ("ready", "initialized", "imu_fresh", "depth_fresh"):
                if not bool(status.get(key, False)):
                    issues.append(key)
            vision_mode = str(status.get("vision_mode", "unknown"))
            if allowed_vision_modes and vision_mode not in allowed_vision_modes:
                issues.append(f"vision_mode:{vision_mode}")
        return position, "OK" if not issues else f"ABNORMAL({', '.join(issues)})"

    def spin_trajectory_with_runtime_status(
        self,
        duration_sec: float,
        *,
        trajectory_id: str,
        repeat: int,
        log_interval_sec: float,
        pose_stale_after_sec: float,
        allowed_vision_modes: set[str],
        monitored_processes: Sequence[tuple[str, _ManagedProcess]],
    ) -> None:
        """Spin the T2 trajectory while continuously exposing runtime health."""

        interval = max(float(log_interval_sec), 0.1)
        duration = max(float(duration_sec), 0.0)
        started = time.monotonic()
        deadline = started + duration
        next_log = started
        while rclpy.ok() and time.monotonic() < deadline:
            for label, process in monitored_processes:
                if process.returncode is not None:
                    message = f"T2 runtime process failure: {label} exited with code {process.returncode}"
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
                    f"T2 trajectory={trajectory_id} R{repeat:02d} "
                    f"progress={100.0 * elapsed / max(duration, 1e-9):5.1f}% "
                    f"t={elapsed:.1f}/{duration:.1f}s position={position} status={health}"
                )
                self.get_logger().info(message)
                if health != "OK" and health != self._last_runtime_health:
                    self.get_logger().warning(f"runtime state anomaly: {message}")
                elif health == "OK" and self._last_runtime_health not in {None, "OK"}:
                    self.get_logger().info("T2 runtime state recovered")
                self._last_runtime_health = health
                next_log = now + interval
            rclpy.spin_once(self, timeout_sec=min(0.1, max(deadline - time.monotonic(), 0.01)))
        position, health = self._runtime_health(
            pose_stale_after_sec=pose_stale_after_sec,
            allowed_vision_modes=allowed_vision_modes,
        )
        self.get_logger().info(
            f"T2 trajectory={trajectory_id} R{repeat:02d} progress=100.0% "
            f"t={duration:.1f}/{duration:.1f}s position={position} status={health}"
        )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a T2 hardware trajectory experiment from one YAML.")
    parser.add_argument("--config", required=True, type=Path, help="T2 YAML under experiment_recorder/config.")
    parser.add_argument("--controller", default="PPO_WRENCH6", help="Enabled controller key declared in the T2 YAML.")
    parser.add_argument(
        "--controller-config",
        type=Path,
        default=None,
        help="Use this motion-controller YAML for the run. PPO checkpoint_path is always resolved from the selected profile; PID profiles are checkpoint-free.",
    )
    parser.add_argument(
        "--trajectory",
        action="append",
        required=True,
        metavar="ID",
        help="Required YAML trajectory ID. Exactly one ID is permitted per experiment run.",
    )
    parser.add_argument("--arm", action="store_true", help="Allow formal hardware execution after all preflight gates pass.")
    video_choice = parser.add_mutually_exclusive_group()
    video_choice.add_argument(
        "--record-video",
        dest="record_video",
        action="store_true",
        help="Record the unannotated overhead raw-camera MP4 for every T2 trial.",
    )
    video_choice.add_argument(
        "--no-record-video",
        dest="record_video",
        action="store_false",
        help="Disable per-trial overhead video even when the protocol YAML enables it.",
    )
    parser.set_defaults(record_video=None)
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the full trial plan without starting ROS2 or hardware.")
    return parser.parse_args(argv)


def _resolve_top_camera_video_config(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    """Resolve the optional, read-only T2 video artifact contract.

    The recorder only consumes the detector's explicitly unannotated raw-image
    topic.  Keeping this in the versioned experiment YAML makes a video's
    presence (or deliberate absence) auditable in the run and trial manifests.
    """

    recording = _as_mapping(config["recording"], "recording")
    configured = recording.get("top_camera_video", {})
    configured = _as_mapping(configured, "recording.top_camera_video")
    enabled = bool(configured.get("enabled", False))
    if args.record_video is not None:
        enabled = bool(args.record_video)
    topic = str(configured.get("image_topic", "/finsrov/camera/raw/compressed")).strip()
    filename = str(configured.get("filename", "top_camera_raw.mp4")).strip()
    codec = str(configured.get("codec", "mp4v")).strip()
    fps = float(configured.get("fps", 10.0))
    minimum_source_fps = float(configured.get("minimum_source_fps", 0.8 * fps))
    minimum_ready_frames = int(configured.get("minimum_ready_frames", 5))
    startup_timeout_sec = float(configured.get("startup_timeout_sec", 12.0))
    finalize_timeout_sec = float(configured.get("finalize_timeout_sec", 20.0))
    if not topic.startswith("/"):
        raise SystemExit("recording.top_camera_video.image_topic must be an absolute ROS topic")
    candidate = Path(filename)
    if not filename or candidate.is_absolute() or candidate.name != filename or candidate.suffix.lower() != ".mp4":
        raise SystemExit("recording.top_camera_video.filename must be one relative .mp4 filename")
    if not math.isfinite(fps) or fps <= 0.0:
        raise SystemExit("recording.top_camera_video.fps must be positive and finite")
    if not math.isfinite(minimum_source_fps) or minimum_source_fps <= 0.0:
        raise SystemExit("recording.top_camera_video.minimum_source_fps must be positive and finite")
    if minimum_source_fps > fps:
        raise SystemExit("recording.top_camera_video.minimum_source_fps cannot exceed fps")
    if minimum_ready_frames < 2:
        raise SystemExit("recording.top_camera_video.minimum_ready_frames must be at least 2")
    if len(codec) != 4:
        raise SystemExit("recording.top_camera_video.codec must be a four-character OpenCV FourCC")
    if not math.isfinite(startup_timeout_sec) or startup_timeout_sec <= 0.0:
        raise SystemExit("recording.top_camera_video.startup_timeout_sec must be positive and finite")
    if not math.isfinite(finalize_timeout_sec) or finalize_timeout_sec <= 0.0:
        raise SystemExit("recording.top_camera_video.finalize_timeout_sec must be positive and finite")
    return {
        "enabled": enabled,
        "image_topic": topic,
        "filename": filename,
        "fps": fps,
        "minimum_source_fps": minimum_source_fps,
        "minimum_ready_frames": minimum_ready_frames,
        "codec": codec,
        "startup_timeout_sec": startup_timeout_sec,
        "finalize_timeout_sec": finalize_timeout_sec,
        "source_annotation": "none; native camera frame before AprilTag debug rendering",
    }


def _top_camera_video_command(video: dict[str, Any], output_path: Path) -> list[str]:
    """Return the standalone, read-only raw-camera recorder command."""

    return [
        "ros2", "run", "experiment_recorder", "record_top_camera_video",
        "--output", str(output_path),
        "--topic", str(video["image_topic"]),
        "--fps", str(video["fps"]),
        "--codec", str(video["codec"]),
    ]


def _wait_for_trial_manifest(path: Path, process: _ManagedProcess, timeout_sec: float) -> None:
    """Wait until record_experiment owns its prepared trial directory."""

    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while time.monotonic() < deadline:
        _require_process_running(process, "experiment recorder")
        if path.is_file():
            return
        time.sleep(0.05)
    raise RuntimeError(f"experiment recorder did not create trial manifest: {path}")


def _wait_for_top_camera_video(
    path: Path,
    process: _ManagedProcess,
    timeout_sec: float,
    *,
    minimum_source_fps: float,
    minimum_ready_frames: int,
) -> None:
    """Require an actually writable video stream before sending a trajectory."""

    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    last_status = "missing"
    while time.monotonic() < deadline:
        _require_process_running(process, "top-camera video recorder")
        if path.is_file():
            try:
                payload = read_json(path)
            except (OSError, ValueError, json.JSONDecodeError):
                payload = {}
            last_status = str(payload.get("status", "unknown"))
            if last_status == "recording":
                try:
                    frames = int(payload.get("frames_written", 0))
                except (TypeError, ValueError):
                    frames = 0
                measured_fps = payload.get("source_fps_estimate")
                if (
                    frames >= minimum_ready_frames
                    and isinstance(measured_fps, (int, float))
                    and float(measured_fps) >= minimum_source_fps
                ):
                    return
            if last_status in {"failed", "no_frames"}:
                raise RuntimeError(f"top-camera video recorder reported {last_status}: {path}")
        time.sleep(0.05)
    raise RuntimeError(
        "no unannotated raw-camera frame was available before the T2 trajectory; "
        f"last video recorder status={last_status}, metadata={path}; "
        f"requires >= {minimum_ready_frames} frames at >= {minimum_source_fps:.2f} Hz"
    )


def _attach_top_camera_video_artifact(
    *,
    run_manifest: dict[str, Any],
    trials_dir: Path,
    session_id: str,
    video: dict[str, Any],
) -> None:
    """Record final video status in the trial and run manifests without hashing a large MP4."""

    trial_dir = trials_dir / session_id
    output_path = trial_dir / "video" / str(video["filename"])
    metadata_path = output_path.with_suffix(".metadata.json")
    details: dict[str, Any] = {
        "enabled": True,
        "file": str(output_path.relative_to(trial_dir)),
        "metadata": str(metadata_path.relative_to(trial_dir)),
        "source_topic": str(video["image_topic"]),
        "source_annotation": str(video["source_annotation"]),
        "exists": output_path.is_file(),
        "size_bytes": output_path.stat().st_size if output_path.is_file() else 0,
    }
    if metadata_path.is_file():
        try:
            details["recording"] = read_json(metadata_path)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            details["metadata_error"] = repr(exc)
    trial_manifest_path = trial_dir / "manifest.json"
    if trial_manifest_path.is_file():
        trial_manifest = read_json(trial_manifest_path)
        trial_manifest.setdefault("artifacts", {})["top_camera_video"] = details
        write_json(trial_manifest_path, trial_manifest)
    run_manifest.setdefault("top_camera_videos", {})[session_id] = details


def _resolve_cli_path(value: Path, repo_root: Path) -> Path:
    """Accept a CLI path relative to either the caller or repository root."""

    path = value.expanduser()
    if path.is_absolute():
        return path.resolve()
    from_cwd = (Path.cwd() / path).resolve()
    return from_cwd if from_cwd.exists() else _resolve_path(path, repo_root)


def _checkpoint_from_profile(config_file: Path, repo_root: Path) -> Path:
    try:
        profile = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        value = profile["motion_controller"]["ros__parameters"]["checkpoint_path"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read checkpoint_path from --controller-config {config_file}") from exc
    if not isinstance(value, str) or not value.strip():
        raise SystemExit(f"--controller-config {config_file} has no nonempty motion_controller.checkpoint_path")
    return _resolve_path(value, repo_root)


def _checkpoint_manifest_from_checkpoint(checkpoint: Path) -> Path:
    """Return the training metadata colocated with a standard run checkpoint.

    Training artifacts are laid out as ``<run>/checkpoints/<model>.zip`` with
    ``<run>/metadata.json``.  Keeping this association structural means the
    controller profile remains the single source of truth for the deployed
    policy, while formal T2 data still retains the policy's training record.
    """

    manifest = checkpoint.parent.parent / "metadata.json"
    if not manifest.is_file():
        raise SystemExit(
            "PPO checkpoint provenance is required beside the selected profile checkpoint; "
            f"expected {manifest} for {checkpoint}"
        )
    return manifest.resolve()


def _apply_cli_controller_overrides(
    config: dict[str, Any],
    args: argparse.Namespace,
    *,
    repo_root: Path,
) -> str:
    """Apply an auditable controller/profile override without editing the YAML."""

    controller_name = str(args.controller)
    controllers = _as_mapping(config["controllers"], "controllers")
    selected = copy.deepcopy(_as_mapping(controllers.get(controller_name), f"controllers.{controller_name}"))
    source = "t2_yaml_controller_profile"
    if args.controller_config is not None:
        controller_config = _resolve_cli_path(args.controller_config, repo_root)
        if not controller_config.is_file():
            raise SystemExit(f"--controller-config does not exist: {controller_config}")
        selected["config_file"] = str(controller_config)
        source = "cli_controller_config"
    controllers[controller_name] = selected
    return source


def _select_trajectory_ids(
    cli_values: Sequence[str],
    protocol: dict[str, Any],
    catalog: dict[str, tuple[list[ReferencePoint], dict[str, float]]],
) -> tuple[list[str], str]:
    """Resolve a nonempty, ordered trajectory subset from YAML or CLI."""

    del protocol  # Selection is intentionally CLI-only to keep one run = one path.
    requested = [str(value).strip() for value in cli_values if str(value).strip()]
    if len(requested) != 1:
        raise SystemExit("exactly one --trajectory ID is required for each T2 experiment run")
    unknown = [name for name in requested if name not in catalog]
    if unknown:
        raise SystemExit(f"selected trajectories are disabled or unknown: {unknown}; available={sorted(catalog)}")
    return requested, "cli_exactly_one"


def _controller(
    config: dict[str, Any],
    controller_name: str,
    repo_root: Path,
    *,
    dry_run: bool,
) -> tuple[dict[str, Any], Path, Path | None, Path | None]:
    selected = _as_mapping(_as_mapping(config["controllers"], "controllers").get(controller_name), f"controllers.{controller_name}")
    if not bool(selected.get("enabled", False)):
        raise SystemExit(f"controller `{controller_name}` is disabled in the T2 YAML")
    config_file = _resolve_path(str(selected.get("config_file", "")), repo_root)
    if not config_file.is_file():
        raise SystemExit(f"controllers.{controller_name}.config_file does not exist: {config_file}")
    checkpoint_required = bool(selected.get("checkpoint_required", True))
    checkpoint = _checkpoint_from_profile(config_file, repo_root) if checkpoint_required else None
    if checkpoint_required and (checkpoint is None or not checkpoint.is_file()):
        raise SystemExit(f"controller profile checkpoint_path does not exist: {checkpoint}")
    try:
        profile = yaml.safe_load(config_file.read_text(encoding="utf-8"))
        params = profile["motion_controller"]["ros__parameters"]
    except (KeyError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read motion_controller parameters from {config_file}") from exc
    expected_backend = str(selected.get("expected_backend_name", ""))
    if str(params.get("backend_name", "")) != expected_backend:
        raise SystemExit(f"controller profile backend_name does not match YAML expectation `{expected_backend}`")
    expected_observation = str(selected.get("expected_observation_mode", ""))
    is_trajectory30_policy = "trajectory" in expected_backend and expected_observation == "trajectory30"
    is_translation_only_pid = (
        expected_backend == "traditional_pid_position"
        and expected_observation == "pose20"
    )
    if not (is_trajectory30_policy or is_translation_only_pid):
        raise SystemExit("selected controller is not declared as a supported T2 trajectory backend")
    if is_translation_only_pid:
        if bool(params.get("use_target_orientation", False)) or bool(params.get("require_target_orientation", False)):
            raise SystemExit("T2 traditional PID profile must disable target-orientation/yaw tracking")
        if bool(params.get("traditional_enable_yaw_control", True)):
            raise SystemExit("T2 traditional PID profile must set traditional_enable_yaw_control=false")
        if str(params.get("thruster_output_mode", "")) != "force_n":
            raise SystemExit("T2 traditional PID profile must publish bounded canonical thruster force commands")
    if str(selected.get("action_interface", "")).startswith("wrench6"):
        wrench = _as_mapping(params.get("wrench6d"), "motion_controller.wrench6d")
        if str(wrench.get("policy_axis_order", "")).strip().lower().replace("-", "_") != "allocator_body":
            raise SystemExit("T2 wrench6 profile must declare wrench6d.policy_axis_order=allocator_body")
        raw_scale = wrench.get("wrench_scale")
        if not isinstance(raw_scale, (list, tuple)) or len(raw_scale) != 6:
            raise SystemExit("T2 wrench6 profile must declare exactly six wrench6d.wrench_scale entries")
        component_order = ("Fx", "Fy", "Fz", "Mx", "My", "Mz")
        try:
            profile_active = {component for component, value in zip(component_order, raw_scale) if abs(float(value)) > 1e-9}
        except (TypeError, ValueError) as exc:
            raise SystemExit("T2 wrench6d.wrench_scale must be numeric") from exc
        declared_active = {str(value) for value in selected.get("active_wrench_components", [])}
        if profile_active != declared_active:
            raise SystemExit(
                "T2 action contract mismatch: profile nonzero wrench axes "
                f"{sorted(profile_active)} != YAML active_wrench_components {sorted(declared_active)}"
            )
        expected_translation_only = {"Fx", "Fy", "Fz"}
        if profile_active != expected_translation_only:
            raise SystemExit(
                "T2 hardware scope is translation-only: the active wrench axes "
                f"must be {sorted(expected_translation_only)}, got {sorted(profile_active)}"
            )
    checkpoint_manifest: Path | None = None
    if checkpoint_required:
        checkpoint_manifest = _checkpoint_manifest_from_checkpoint(checkpoint)
    return selected, config_file, checkpoint, checkpoint_manifest


def _build_catalog(config: dict[str, Any]) -> dict[str, tuple[list[ReferencePoint], dict[str, float]]]:
    trajectory_contract = _as_mapping(config["trajectory_contract"], "trajectory_contract")
    fixed_depth = _as_mapping(config["coordinate_contract"], "coordinate_contract").get("fixed_depth", {})
    fixed_depth = _as_mapping(fixed_depth, "coordinate_contract.fixed_depth")
    depth = float(fixed_depth["y_m"])
    rate = float(trajectory_contract["sample_rate_hz"])
    workspace = _as_mapping(config["surveyed_workspace"], "surveyed_workspace")
    catalog: dict[str, tuple[list[ReferencePoint], dict[str, float]]] = {}
    for name, raw_spec in _as_mapping(config["trajectories"], "trajectories").items():
        spec = _as_mapping(raw_spec, f"trajectories.{name}")
        if not bool(spec.get("enabled", False)):
            continue
        points = generate_trajectory(spec, fixed_depth_m=depth, sample_rate_hz=rate)
        catalog[str(name)] = (points, validate_reference(points, workspace))
    if not catalog:
        raise SystemExit("no enabled T2 trajectories are declared")
    return catalog


def _ros_parameters_from_yaml(path: Path, preferred_node_name: str) -> dict[str, Any]:
    """Read one ROS2 YAML parameter block without accepting implicit defaults."""

    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"could not read ROS2 parameter YAML {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"ROS2 parameter YAML must be a mapping: {path}")
    for node_name in (preferred_node_name, f"/{preferred_node_name}", "/**", "**"):
        node = payload.get(node_name)
        if not isinstance(node, dict):
            continue
        parameters = node.get("ros__parameters")
        if isinstance(parameters, dict):
            return parameters
    raise SystemExit(f"ROS2 parameter YAML has no ros__parameters block: {path}")


def _coordinate_transform_metadata(
    config: dict[str, Any],
    *,
    controller_config: Path,
    repo_root: Path,
) -> dict[str, Any]:
    """Freeze the exact pool-world <-> controller-world mapping in a trial.

    T2 commands live in controller coordinates, while the physical fusion
    output remains in pool_world.  The recorder must therefore archive this
    mapping with each bag instead of relying on whichever YAML happens to be
    present when an old experiment is re-analysed.
    """

    runner = _as_mapping(config["runner"], "runner")
    auto = _as_mapping(runner.get("auto_launch", {}), "runner.auto_launch")
    motion = _as_mapping(auto.get("motion_controller", {}), "runner.auto_launch.motion_controller")
    pool_world_path = _resolve_path(str(motion["pool_world_params_file"]), repo_root)
    controller_parameters = _ros_parameters_from_yaml(controller_config, "motion_controller")
    pool_parameters = _ros_parameters_from_yaml(pool_world_path, "/**")
    raw_indices = controller_parameters.get("ros_to_controller_basis_indices")
    raw_signs = controller_parameters.get("ros_to_controller_basis_signs")
    if not isinstance(raw_indices, (list, tuple)) or not isinstance(raw_signs, (list, tuple)):
        raise SystemExit(
            "T2 controller profile must explicitly declare "
            "ros_to_controller_basis_indices and ros_to_controller_basis_signs"
        )
    try:
        indices = [int(value) for value in raw_indices]
        signs = [float(value) for value in raw_signs]
        water_surface_z_m = float(pool_parameters["water_surface_z_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("invalid T2 pool/controller coordinate parameters") from exc
    if sorted(indices) != [0, 1, 2] or len(signs) != 3:
        raise SystemExit("T2 coordinate basis must be a three-axis permutation with three signs")
    if not math.isfinite(water_surface_z_m) or any(not math.isfinite(value) or abs(value) < 1e-9 for value in signs):
        raise SystemExit("T2 coordinate transform contains non-finite water-surface or basis-sign values")
    coordinate_contract = _as_mapping(config["coordinate_contract"], "coordinate_contract")
    controller_frame = str(coordinate_contract.get("state_frame", "controller_world"))
    pool_frame = str(coordinate_contract.get("pool_world_frame", "pool_world"))
    return {
        "schema": "pool_world_to_controller_world_v1",
        "pool_world_frame": pool_frame,
        "controller_world_frame": controller_frame,
        # controller[j] = signs[j] * pool[indices[j]] + offset[j]
        "basis_indices": indices,
        "basis_signs": signs,
        "post_basis_offset_m": [0.0, -water_surface_z_m, 0.0],
        "water_surface_z_m": water_surface_z_m,
        "controller_config": str(controller_config),
        "pool_world_config": str(pool_world_path),
    }


def _validate_arm_gates(config: dict[str, Any], *, arm: bool, dry_run: bool) -> None:
    """Apply only runtime-enforceable T2 execution gates.

    Survey/pilot metadata remains in the YAML for auditability, but it is not
    an execution-blocking condition.  Physical pool review is an operator
    responsibility; the runner still requires an explicit ``--arm`` before
    it can enable hardware output.
    """

    if dry_run:
        return
    if bool(_as_mapping(config["runner"], "runner").get("require_explicit_arm_flag", True)) and not arm:
        raise SystemExit("formal T2 execution requires --arm; use --dry-run to inspect the plan")


def _copy_trial_plots(run_dir: Path, session_id: str) -> list[str]:
    source_dir = run_dir / "trials" / session_id / "derived" / "plots"
    destination_dir = run_dir / "derived" / "trial_plots"
    destination_dir.mkdir(parents=True, exist_ok=True)
    # Re-analysis must not leave an older, coordinate-invalid copy next to the
    # corrected plot.  Limit cleanup to this exact session prefix.
    prefix = f"{session_id}_"
    for destination in destination_dir.iterdir():
        if destination.is_file() and destination.name.startswith(prefix) and destination.suffix.lower() == ".png":
            destination.unlink()
    copied: list[str] = []
    if not source_dir.is_dir():
        return copied
    for source in sorted(source_dir.rglob("*.png")):
        relative = source.relative_to(source_dir)
        destination = destination_dir / f"{session_id}_{'_'.join(relative.parts)}"
        shutil.copy2(source, destination)
        copied.append(str(destination.relative_to(run_dir)))
    return copied


def _append_trial_analysis_record(
    manifest: dict[str, Any],
    *,
    run_dir: Path,
    trials_dir: Path,
    session_id: str,
    outcome: str,
) -> None:
    """Attach the recorder's post-stop analysis result to the run manifest."""

    trial_manifest_path = trials_dir / session_id / "manifest.json"
    trial_manifest = read_json(trial_manifest_path) if trial_manifest_path.is_file() else {}
    record = {
        "session_id": session_id,
        "outcome": outcome,
        "status": trial_manifest.get("analysis", {}).get("status", "needs_review"),
        "report": str(trials_dir / session_id / "derived" / "analysis_report.json"),
        "plots": _copy_trial_plots(run_dir, session_id),
    }
    reports = manifest.setdefault("analysis_reports", [])
    for index, previous in enumerate(reports):
        if isinstance(previous, dict) and previous.get("session_id") == session_id:
            reports[index] = record
            return
    reports.append(record)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = _repo_root()
    config_path = args.config.expanduser().resolve()
    config = _load_config(config_path)
    top_camera_video = _resolve_top_camera_video_config(config, args)
    controller_source = _apply_cli_controller_overrides(config, args, repo_root=repo_root)
    _validate_arm_gates(config, arm=args.arm, dry_run=args.dry_run)
    controller_name = str(args.controller)
    selected, controller_config, checkpoint, checkpoint_manifest = _controller(config, controller_name, repo_root, dry_run=args.dry_run)
    coordinate_transform = _coordinate_transform_metadata(
        config,
        controller_config=controller_config,
        repo_root=repo_root,
    )
    catalog = _build_catalog(config)
    runner_cfg = _as_mapping(config["runner"], "runner")
    auto = _as_mapping(runner_cfg.get("auto_launch", {}), "runner.auto_launch")
    if not bool(auto.get("enabled", True)):
        raise SystemExit("automatic T2 runner requires runner.auto_launch.enabled=true")
    protocol = _as_mapping(config["trial_protocol"], "trial_protocol")
    requested, trajectory_source = _select_trajectory_ids(args.trajectory, protocol, catalog)
    coordinate_contract = _as_mapping(config["coordinate_contract"], "coordinate_contract")
    initial_placement = _as_mapping(
        coordinate_contract.get("declared_initial_placement", {}),
        "coordinate_contract.declared_initial_placement",
    )
    try:
        initial_placement_y_m = float(initial_placement["y_m"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit("coordinate_contract.declared_initial_placement.y_m must be a finite number") from exc
    if not math.isfinite(initial_placement_y_m):
        raise SystemExit("coordinate_contract.declared_initial_placement.y_m must be a finite number")
    repeats = int(protocol.get("repeats_per_controller_and_trajectory", 1))
    if repeats < 1:
        raise SystemExit("repeats_per_controller_and_trajectory must be >= 1")
    recording = _as_mapping(runner_cfg.get("logging", {}), "runner.logging")
    data_root = _resolve_path(str(recording.get("root", "ros2_ws/data/experiments")), repo_root)
    method_bucket = _controller_storage_subdirectory(controller_name, selected)
    strategy = _strategy_name(str(selected.get("experiment_label", controller_name)))
    # One invocation selects exactly one trajectory. Put that selection in the
    # run-directory name itself, so raw bags, manifests, derived plots, and
    # copied artifacts remain distinguishable even outside their metadata.
    trajectory_label = _strategy_name(requested[0])
    run_id = _new_run_id(
        canonical_run_dir(
            data_root, domain="hardware", task="T2", method=method_bucket, run_id="placeholder"
        ).parent,
        f"{str(config.get('experiment_id', 'E10'))}_{strategy}_{trajectory_label}",
    )
    run_dir = canonical_run_dir(data_root, domain="hardware", task="T2", method=method_bucket, run_id=run_id)
    planned: list[dict[str, Any]] = []
    for repeat in range(1, repeats + 1):
        for trajectory_id in requested:
            points, diagnostics = catalog[trajectory_id]
            planned.append({
                "trajectory_id": trajectory_id,
                "repeat": repeat,
                "duration_sec": points[-1].time_sec,
                "declared_initial_controller_world": [
                    points[0].x_m,
                    initial_placement_y_m,
                    points[0].z_m,
                ],
                "reference_start_controller_world": [
                    points[0].x_m,
                    points[0].y_m,
                    points[0].z_m,
                ],
                # Compatibility key: this always denotes the first sent
                # reference, not the operator-placement point.
                "start_controller_world": [points[0].x_m, points[0].y_m, points[0].z_m],
                "orientation_control": "disabled",
                "peak_diagnostics": diagnostics,
            })
    config_files = _config_files(config, config_path, repo_root)
    if controller_config not in config_files:
        config_files.append(controller_config)
    missing = [str(path) for path in config_files if not path.is_file()]
    if missing:
        raise SystemExit(f"configured files do not exist: {missing}")
    manifest: dict[str, Any] = {
        "schema_version": 2,
        "status": "planned" if args.dry_run else "running",
        "run_id": run_id,
        "run_directory": str(run_dir.relative_to(repo_root)),
        "experiment_layout": {"domain": "hardware", "task": "T2", "method": method_bucket},
        "created_at": now_iso(),
        "protocol_config": str(config_path),
        "protocol_config_sha256": file_sha256(config_path),
        "git_revision": git_revision(repo_root),
        "armed": bool(args.arm),
        "controller": controller_name,
        "controller_source": controller_source,
        "controller_config": str(controller_config),
        "controller_config_sha256": file_sha256(controller_config),
        **_optional_file_provenance("checkpoint", checkpoint),
        **_optional_file_provenance("checkpoint_manifest", checkpoint_manifest),
        "config_files": {str(path): file_sha256(path) for path in config_files},
        "coordinate_transform": coordinate_transform,
        "trajectory_selection": {"source": trajectory_source, "ids": requested},
        "trial_plan": planned,
        "survey_signoff_metadata": config.get("survey_signoff", {}),
        "support_stack": {"mode": "launched_by_runner" if bool(auto.get("start_support_stack", False)) else "pre_started"},
        "top_camera_video": top_camera_video,
    }
    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        return

    run_layout = create_run_layout(
        data_root, domain="hardware", task="T2", method=method_bucket, run_id=run_id
    )
    run_dir, trials_dir = run_layout.run_dir, run_layout.trials_dir
    manifest["provenance_snapshots"] = _snapshot_controller_provenance(
        run_dir,
        controller_config=controller_config,
        checkpoint_manifest=checkpoint_manifest,
    )
    references_dir = run_dir / "resolved_trajectories"
    references_dir.mkdir()
    resolved_reference: dict[str, Path] = {}
    for trajectory_id in requested:
        points, _ = catalog[trajectory_id]
        path = references_dir / f"{trajectory_id}.csv"
        write_reference_csv(path, points)
        resolved_reference[trajectory_id] = path
    manifest["resolved_trajectories"] = {name: {"path": str(path), "sha256": file_sha256(path)} for name, path in resolved_reference.items()}
    write_json(run_dir / "run_manifest.json", manifest)
    log_stage(
        f"T2 run initialized: controller={controller_name} trajectory={requested[0]} "
        f"trials={len(planned)} run={run_dir}"
    )

    startup_timeout = float(auto.get("startup_timeout_sec", 30.0))
    startup_settle = float(auto.get("startup_settle_sec", 3.0))
    terminate_timeout = float(auto.get("terminate_timeout_sec", 10.0))
    # The recorder runs rosbag finalization and matplotlib analysis after its
    # SIGINT.  Give that child a separate bounded grace period; using the
    # short controller-stop timeout here can kill the analysis midway.
    recorder_finalize_timeout = max(
        terminate_timeout,
        float(auto.get("recorder_finalize_timeout_sec", 90.0)),
    )
    video_finalize_timeout = max(terminate_timeout, float(top_camera_video["finalize_timeout_sec"]))
    required_support_topics = [str(value) for value in auto.get("required_support_topics", [])]
    required_controller_topics = [str(value) for value in auto.get("required_controller_topics", [])]
    trajectory_contract = _as_mapping(config["trajectory_contract"], "trajectory_contract")
    profile = str(recording.get("recorder_profile", "t2"))
    event_topic = str(recording.get("event_topic", "/finsrov/experiment/event"))
    post_record = float(protocol.get("post_trajectory_record_sec", 7.0))
    inter_trial_interval = float(protocol.get("inter_trial_interval_sec", 15.0))
    if inter_trial_interval < 0.0:
        raise SystemExit("trial_protocol.inter_trial_interval_sec must be nonnegative")
    metric_protocol = _as_mapping(config["metric_protocol"], "metric_protocol")
    evaluation_window = _as_mapping(metric_protocol.get("evaluation_window", {}), "metric_protocol.evaluation_window")
    completion_contract = _as_mapping(metric_protocol.get("completion", {}), "metric_protocol.completion")
    runtime_status = _as_mapping(runner_cfg.get("runtime_status", {}), "runner.runtime_status")
    runtime_status_log_interval = float(runtime_status.get("log_interval_sec", 1.0))
    pose_stale_after = float(runtime_status.get("pose_stale_after_sec", 0.5))
    if runtime_status_log_interval <= 0.0 or pose_stale_after <= 0.0:
        raise SystemExit("runner.runtime_status log_interval_sec and pose_stale_after_sec must be positive")
    allowed_vision_modes = {str(value) for value in completion_contract.get("allowed_vision_modes", ("fresh", "coast"))}
    safety = _as_mapping(runner_cfg.get("safety", {}), "runner.safety")
    support_processes: list[_ManagedProcess] = []
    shared_log_files: list[Path] = []
    node: _T2ExperimentNode | None = None
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
            except Exception as exc:  # pragma: no cover - hardware failure is environment-specific.
                # Process-group termination below remains mandatory even when
                # the bridge service or ROS context is no longer available.
                record["status"] = "error"
                record["error"] = repr(exc)
        manifest["emergency_stop"] = record

    try:
        if bool(auto.get("start_support_stack", False)):
            for name, command in _launch_commands(config, controller_name, args.arm, repo_root)[:3]:
                log_path = shared_log_path(run_layout, f"{name}.log")
                shared_log_files.append(log_path)
                support_processes.append(_ManagedProcess(command, log_path))
        _wait_for_topics(required_support_topics, startup_timeout)
        time.sleep(max(startup_settle, 0.0))
        init_experiment_ros()
        emergency_cfg = _as_mapping(safety.get("emergency_stop", {}), "runner.safety.emergency_stop")
        node = _T2ExperimentNode(
            event_topic,
            trajectory_topic=str(trajectory_contract["command_topic"]),
            bridge_node_name=str(emergency_cfg.get("bridge_node", "/hardware_bridge")),
            bridge_status_topic=str(emergency_cfg.get("bridge_status_topic", "/finsrov/hardware/status")),
            pose_topic="/finsrov/controller/pose",
            state_status_topic="/finsrov/controller/state/status",
        )
        log_stage(
            f"T2 support stack ready; beginning {len(planned)} {requested[0]} trajectory acquisitions",
            logger=node.get_logger(),
        )
        # Give ROS graph discovery a brief opportunity to report an orphaned
        # controller from a previous interrupted run before launching ours.
        node._spin_sleep(0.2)
        existing_controllers = node.warn_existing_motion_controller()
        if existing_controllers:
            manifest.setdefault("startup_warnings", []).append({
                "kind": "pre_existing_motion_controller",
                "nodes": existing_controllers,
                "detected_at": now_iso(),
            })
            write_json(run_dir / "run_manifest.json", manifest)
        # The bridge is normally pre-started in the lab.  Treat --arm as an
        # explicit runtime enable request in that case too, rather than only
        # passing enabled:=true when this runner happened to launch the bridge.
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
        for index, trial in enumerate(planned, start=1):
            trajectory_id = str(trial["trajectory_id"])
            repeat = int(trial["repeat"])
            points, _ = catalog[trajectory_id]
            trajectory_spec = _as_mapping(
                _as_mapping(config["trajectories"], "trajectories").get(trajectory_id),
                f"trajectories.{trajectory_id}",
            )
            reference_path = resolved_reference[trajectory_id]
            # ``run_id`` already contains both the strategy and trajectory.
            # Keep the trial name readable while preserving those identifiers.
            session_id = f"{run_id}_R{repeat:02d}"
            trial_layout = prepare_trial_layout(trials_dir, session_id)
            log_stage(
                f"T2 acquisition {index}/{len(planned)}: trajectory={trajectory_id} R{repeat:02d}; "
                "preparing controller and recorder",
                logger=node.get_logger(),
            )
            metadata = {
                "run_id": run_id,
                "trial_index": index,
                "controller": controller_name,
                "strategy": strategy,
                "trajectory_id": trajectory_id,
                "repeat": repeat,
                "trajectory_frame": str(trajectory_contract["frame_id"]),
                "coordinate_transform": coordinate_transform,
                "trajectory_duration_sec": points[-1].time_sec,
                "trajectory_geometry": {
                    "generator": str(trajectory_spec.get("generator", "")),
                    "turns": trajectory_spec.get("turns"),
                },
                "trajectory_reference_file": str(reference_path),
                "trajectory_reference_sha256": file_sha256(reference_path),
                "start_controller_world": trial["start_controller_world"],
                "declared_initial_controller_world": trial["declared_initial_controller_world"],
                "reference_start_controller_world": trial["reference_start_controller_world"],
                "start_pose_verification": "disabled_by_trial_protocol",
                "controller_config": str(controller_config),
                **_optional_file_provenance("checkpoint", checkpoint),
                **_optional_file_provenance("checkpoint_manifest", checkpoint_manifest),
                "protocol_config": str(config_path),
                "evaluation_window": {
                    "start_event": str(evaluation_window.get("start_after_event", "trajectory_start")),
                    "duration_sec": (
                        points[-1].time_sec
                        if str(evaluation_window.get("duration_sec", "trajectory_duration")).strip().lower()
                        == "trajectory_duration"
                        else float(evaluation_window["duration_sec"])
                    ),
                    "resample_hz": float(evaluation_window.get("resample_hz", 10.0)),
                    "max_interpolation_gap_sec": float(evaluation_window.get("max_interpolation_gap_sec", 0.25)),
                },
                # Freeze analysis criteria with the raw bag.  Editing the
                # source YAML after a pool session must not silently redefine
                # completion metrics for already-recorded data.
                "completion_contract": completion_contract,
                "controlled_dofs": ["x", "y_depth", "z"],
                "orientation_control": "disabled",
                "top_camera_video": {
                    **top_camera_video,
                    "output": (
                        f"video/{top_camera_video['filename']}"
                        if top_camera_video["enabled"] else None
                    ),
                },
            }
            controller_process: _ManagedProcess | None = None
            recorder_process: _ManagedProcess | None = None
            video_process: _ManagedProcess | None = None
            video_metadata_path: Path | None = None
            trial_log_names = ["controller.log", "recorder.log"]
            try:
                motion_command = _launch_commands(config, controller_name, args.arm, repo_root)[3][1]
                node.clear_controller_pose_preflight()
                controller_process = _ManagedProcess(motion_command, trial_layout.logs_dir / "controller.log")
                _wait_for_topics(required_controller_topics, startup_timeout)
                time.sleep(max(startup_settle, 0.0))
                _require_process_running(controller_process, "motion controller")
                metadata["controller_pose_preflight"] = node.wait_for_controller_pose(
                    float(protocol.get("readiness_timeout_sec", startup_timeout))
                )
                recorder_process = _ManagedProcess(
                    _record_command(
                        experiment_id="E10", strategy=strategy, session_id=session_id,
                        output_root=trials_dir, profile=profile, metadata=metadata,
                        config_files=[
                            *config_files,
                            *([checkpoint_manifest] if checkpoint_manifest is not None else []),
                            reference_path,
                        ],
                        checkpoint=checkpoint,
                        prepared_session=True,
                    ),
                    trial_layout.logs_dir / "recorder.log",
                )
                trial_manifest_path = trial_layout.trial_dir / "manifest.json"
                _wait_for_trial_manifest(trial_manifest_path, recorder_process, startup_timeout)
                if top_camera_video["enabled"]:
                    video_path = trial_layout.trial_dir / "video" / str(top_camera_video["filename"])
                    video_metadata_path = video_path.with_suffix(".metadata.json")
                    video_process = _ManagedProcess(
                        _top_camera_video_command(top_camera_video, video_path),
                        trial_layout.logs_dir / "top_camera_video.log",
                    )
                    trial_log_names.append("top_camera_video.log")
                    _wait_for_top_camera_video(
                        video_metadata_path,
                        video_process,
                        float(top_camera_video["startup_timeout_sec"]),
                        minimum_source_fps=float(top_camera_video["minimum_source_fps"]),
                        minimum_ready_frames=int(top_camera_video["minimum_ready_frames"]),
                    )
                # record_experiment itself needs time to launch rosbag2 and
                # discover the runner's event publisher.  Reuse the declared
                # support-stack settle period rather than publishing the
                # critical trajectory_start marker after a fixed one second.
                time.sleep(max(startup_settle, 0.0))
                _require_process_running(recorder_process, "experiment recorder")
                if video_process is not None:
                    _require_process_running(video_process, "top-camera video recorder")
                log_stage(
                    f"T2 acquisition {index}/{len(planned)}: recorder active"
                    f"{'; raw overhead video active' if video_process is not None else ''}; "
                    f"trajectory duration={points[-1].time_sec:.1f}s",
                    logger=node.get_logger(),
                )
                node.publish_event(session_id, "trial_start", trajectory_id, metadata)
                node.publish_event(session_id, "phase_start", "pre_trajectory", metadata)
                node.publish_event(session_id, "trajectory_start", trajectory_id, metadata)
                node.publish_trajectory(points)
                monitored_processes: list[tuple[str, _ManagedProcess]] = [
                    ("motion_controller", controller_process),
                    ("experiment_recorder", recorder_process),
                ]
                if video_process is not None:
                    monitored_processes.append(("top_camera_video_recorder", video_process))
                node.spin_trajectory_with_runtime_status(
                    points[-1].time_sec,
                    trajectory_id=trajectory_id,
                    repeat=repeat,
                    log_interval_sec=runtime_status_log_interval,
                    pose_stale_after_sec=pose_stale_after,
                    allowed_vision_modes=allowed_vision_modes,
                    monitored_processes=tuple(monitored_processes),
                )
                node.publish_event(session_id, "trajectory_end", trajectory_id, metadata)
                node.cancel_goal()
                node.publish_event(session_id, "phase_end", "trajectory", metadata)
                node._spin_sleep(post_record)
                node.publish_event(session_id, "trial_end", trajectory_id, metadata)
                log_stage(
                    f"T2 acquisition {index}/{len(planned)}: trajectory finished; "
                    "finalizing rosbag and running automatic analysis",
                    logger=node.get_logger(),
                )
                if video_process is not None:
                    video_process.stop(video_finalize_timeout)
                    video_process = None
                recorder_process.stop(recorder_finalize_timeout)
                recorder_process = None
                if top_camera_video["enabled"]:
                    _attach_top_camera_video_artifact(
                        run_manifest=manifest,
                        trials_dir=trials_dir,
                        session_id=session_id,
                        video=top_camera_video,
                    )
                manifest.setdefault("completed_trials", []).append(session_id)
                _append_trial_analysis_record(
                    manifest,
                    run_dir=run_dir,
                    trials_dir=trials_dir,
                    session_id=session_id,
                    outcome="completed",
                )
                trial_manifest_path = trials_dir / session_id / "manifest.json"
                trial_manifest = read_json(trial_manifest_path) if trial_manifest_path.is_file() else {}
                log_stage(
                    f"T2 acquisition {index}/{len(planned)} complete: trajectory={trajectory_id} "
                    f"analysis={trial_manifest.get('analysis', {}).get('status', 'needs_review')} "
                    f"report={trials_dir / session_id / 'derived' / 'analysis_report.json'}",
                    logger=node.get_logger(),
                )
                register_trial_runtime_artifacts(
                    run_manifest=manifest,
                    run_dir=run_dir,
                    trial=trial_layout,
                    local_logs=tuple(trial_log_names),
                    shared_logs=shared_log_files,
                )
            except KeyboardInterrupt:
                # Ctrl-C is an operator-requested early terminal event, not a
                # reason to discard the already-recorded portion of a trial.
                # First mark/cancel/zero motion, then allow record_experiment
                # to close rosbag2 and run its automatic analysis before the
                # emergency path terminates controller processes.
                metadata["termination_reason"] = "operator_interrupt"
                metadata["trajectory_completed"] = False
                try:
                    node.publish_event(session_id, "operator_abort", trajectory_id, metadata)
                except Exception as exc:
                    # Preserve the interrupt path even if an external shutdown
                    # has invalidated publishers before this handler runs.
                    metadata["operator_abort_event_error"] = repr(exc)
                try:
                    emergency_cfg = _as_mapping(safety.get("emergency_stop", {}), "runner.safety.emergency_stop")
                    metadata["operator_stop_motion"] = node.stop_motion(
                        zero_frames=int(emergency_cfg.get("zero_frames", 6)),
                        zero_frame_interval_sec=float(emergency_cfg.get("zero_frame_interval_sec", 0.05)),
                    )
                except Exception as exc:
                    metadata["operator_stop_motion_error"] = repr(exc)
                if recorder_process is not None:
                    try:
                        node.get_logger().info(
                            "[T2] Ctrl-C received: finalizing rosbag and generating partial-trial plots"
                        )
                        recorder_process.stop(recorder_finalize_timeout)
                        recorder_process = None
                    except Exception as exc:
                        metadata["recorder_finalize_error"] = repr(exc)
                manifest.setdefault("interrupted_trials", []).append(metadata)
                _append_trial_analysis_record(
                    manifest,
                    run_dir=run_dir,
                    trials_dir=trials_dir,
                    session_id=session_id,
                    outcome="operator_interrupted",
                )
                register_trial_runtime_artifacts(
                    run_manifest=manifest,
                    run_dir=run_dir,
                    trial=trial_layout,
                    local_logs=tuple(trial_log_names),
                    shared_logs=shared_log_files,
                )
                emergency_shutdown("keyboard_interrupt")
                manifest["status"] = "interrupted"
                # A Ctrl-C terminates the requested run after the current
                # partial-trial report has been materialized.  Returning keeps
                # the shell outcome clean instead of discarding plots through
                # an outer KeyboardInterrupt cleanup path.
                return
            except Exception as exc:
                metadata["error"] = repr(exc)
                manifest.setdefault("failed_trials", []).append(metadata)
                try:
                    node.publish_event(session_id, "safety_abort", "runner_exception", metadata)
                except Exception as event_exc:
                    metadata["safety_abort_event_error"] = repr(event_exc)
                emergency_shutdown("runner_exception")
                raise
            finally:
                video_finalized = False
                if video_process is not None:
                    video_process.stop(
                        video_finalize_timeout,
                        force=emergency_stop_done,
                    )
                    video_finalized = True
                if recorder_process is not None:
                    recorder_process.stop(
                        recorder_finalize_timeout,
                        force=emergency_stop_done,
                    )
                if video_finalized:
                    _attach_top_camera_video_artifact(
                        run_manifest=manifest,
                        trials_dir=trials_dir,
                        session_id=session_id,
                        video=top_camera_video,
                    )
                if controller_process is not None:
                    controller_process.stop(terminate_timeout, force=emergency_stop_done)
                write_json(run_dir / "run_manifest.json", manifest)
            if index < len(planned):
                interval_record = {
                    "after_session_id": session_id,
                    "requested_minimum_sec": inter_trial_interval,
                    "started_at": now_iso(),
                }
                manifest.setdefault("inter_trial_intervals", []).append(interval_record)
                write_json(run_dir / "run_manifest.json", manifest)
                # The controller and recorder have already stopped.  This is a
                # deliberately minimum wall-clock separation; process startup
                # before the next trial can only make the separation longer.
                log_stage(
                    f"T2 acquisition {index}/{len(planned)} complete; waiting "
                    f"{inter_trial_interval:.1f}s before acquisition {index + 1}/{len(planned)}",
                    logger=node.get_logger(),
                )
                node._spin_sleep(inter_trial_interval)
                interval_record["finished_at"] = now_iso()
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
        for process in reversed(support_processes):
            process.stop(terminate_timeout, force=emergency_stop_done)
        manifest["bridge_exit_policy"] = "disarm" if safety.get("disarm_on_exit", False) else "preserve_enabled"
        manifest["bridge_disarmed_on_exit"] = bool(
            manifest.get("emergency_stop", {}).get("bridge_disabled_confirmed", False)
        )
        manifest["status"] = manifest.get("status", "complete") if manifest.get("status") != "running" else "complete"
        manifest["finished_at"] = now_iso()
        write_json(run_dir / "run_manifest.json", manifest)
        log_stage(
            f"T2 run finished: status={manifest['status']} completed={len(manifest.get('completed_trials', []))} "
            f"failed={len(manifest.get('failed_trials', []))} run={run_dir}"
        )


if __name__ == "__main__":
    main()
