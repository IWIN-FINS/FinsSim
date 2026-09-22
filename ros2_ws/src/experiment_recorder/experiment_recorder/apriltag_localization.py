"""Native ROS2 replay for the E2 AprilTag horizontal-localization ablation.

The archived images are intentionally *not* re-detected in Python.  This
command materializes a read-only replay list, starts the production C++
``direct_apriltag_node`` in replay mode, and forwards its 2-D detections to
the production ``refractive_apriltag_pose_node``.  The latter publishes both
its pinhole-PnP baseline and its Snell/depth/IMU-constrained estimate.

The archive has no synchronized pressure or IMU log.  A level attitude and
the known one-metre bottom-tag depth are therefore published as explicit
oracle constraints.  This is a horizontal geometry evaluation of the runtime
chain, not an end-to-end state-estimation result or a camera-stream detection
rate measurement.
"""

from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
from datetime import datetime
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
from typing import Any, Iterable, Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String
import yaml

from .manifest import file_sha256, git_revision, now_iso, safe_session_dir, write_json


@dataclass(frozen=True)
class DatasetSample:
    replay_index: int
    sample_id: str
    annotation_truth_x_m: float
    annotation_truth_y_m: float
    truth_x_m: float
    truth_y_m: float
    image_path: Path


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _default_config() -> Path:
    source = Path(__file__).resolve().parents[1] / "config" / "e2_apriltag_localization_dataset.yaml"
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("experiment_recorder")) / "config" / source.name
        if installed.is_file():
            return installed
    except Exception:
        pass
    return source


def _load_yaml(path: Path) -> dict[str, Any]:
    try:
        result = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"AprilTag localization configuration not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid AprilTag localization YAML {path}: {exc}") from exc
    if not isinstance(result, dict):
        raise SystemExit(f"AprilTag localization configuration must be a mapping: {path}")
    return result


def _mapping(payload: dict[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise SystemExit(f"configuration field '{key}' must be a mapping")
    return value


def _resolve_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(os.path.expandvars(str(value))).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _new_session_id(output_root: Path, experiment_id: str) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{experiment_id}_NativeReplay"
    candidate = base
    suffix = 2
    while (output_root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _number(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"source samples.csv has invalid {field!r}: {row.get(field)!r}") from exc


def _load_samples(
    source_session: Path,
    samples_file: str,
    truth_to_pool_world: dict[str, Any],
    truth_annotation_to_tag_center: dict[str, Any],
    limit: int | None,
) -> list[DatasetSample]:
    csv_path = source_session / samples_file
    if not csv_path.is_file():
        raise SystemExit(f"source samples file not found: {csv_path}")
    result: list[DatasetSample] = []
    with csv_path.open("r", encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            sample_id = str(row.get("sample_id", "")).strip()
            if not sample_id:
                raise SystemExit(f"source samples file has a row without sample_id: {csv_path}")
            image_path = source_session / "images" / "raw" / f"{sample_id}.png"
            if not image_path.is_file():
                raise SystemExit(f"archived raw image is missing for {sample_id}: {image_path}")
            annotation_x = _number(row, "truth_x_m")
            annotation_y = _number(row, "truth_y_m")
            try:
                grid_x = float(truth_to_pool_world.get("x_scale", 1.0)) * annotation_x + float(
                    truth_to_pool_world["x_offset_m"]
                )
                grid_y = float(truth_to_pool_world.get("y_scale", 1.0)) * annotation_y + float(
                    truth_to_pool_world["y_offset_m"]
                )
                # The archived annotation is the tag's outer-border lower-left
                # corner.  Native PnP/Snell output is the tag-centre pose, so
                # convert the truth to that same physical point before any
                # error is computed.
                truth_x = grid_x + float(truth_annotation_to_tag_center["x_offset_m"])
                truth_y = grid_y + float(truth_annotation_to_tag_center["y_offset_m"])
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    "source.truth_to_pool_world and source.truth_annotation_to_tag_center "
                    "must define numeric x_offset_m and y_offset_m"
                ) from exc
            result.append(
                DatasetSample(
                    replay_index=len(result),
                    sample_id=sample_id,
                    annotation_truth_x_m=annotation_x,
                    annotation_truth_y_m=annotation_y,
                    truth_x_m=truth_x,
                    truth_y_m=truth_y,
                    image_path=image_path,
                )
            )
            if limit is not None and len(result) >= limit:
                break
    if not result:
        raise SystemExit(f"source samples file is empty: {csv_path}")
    return result


def _copy_calibrated_extrinsics(source_path: Path, destination: Path, tag_id: int) -> None:
    """Copy T_world_camera and create only an evaluation-local virtual tag.

    The bottom marker is not a vehicle-mounted tag.  Identity T_body_tag lets
    the runtime pose node represent that marker centre as a virtual body while
    preserving the deployed camera-to-pool transform exactly.
    """

    payload = _load_yaml(source_path)
    world_camera = payload.get("T_world_camera")
    if not isinstance(world_camera, dict):
        raise SystemExit(f"calibrated extrinsics lacks T_world_camera: {source_path}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fixture = {
        "T_world_camera": world_camera,
        "T_body_tag": {
            int(tag_id): {
                "translation_xyz": [0.0, 0.0, 0.0],
                "rotation_rpy_deg": [0.0, 0.0, 0.0],
            }
        },
        "evaluation_note": (
            "T_world_camera copied from calibrated runtime artifact; T_body_tag is an "
            "experiment-local identity transform for the archived bottom marker."
        ),
    }
    destination.write_text(yaml.safe_dump(fixture, sort_keys=False), encoding="utf-8")


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")


def _prepare_native_params(
    output_dir: Path,
    samples: Sequence[DatasetSample],
    config: dict[str, Any],
    repo_root: Path,
) -> dict[str, Path]:
    source = _mapping(config, "source")
    tag = _mapping(config, "tag")
    calibration = _mapping(config, "runtime_calibration")
    geometry = _mapping(config, "geometry")
    replay = _mapping(config, "replay")
    topic_root = "/apriltag_localization_eval"
    input_dir = output_dir / "input"
    config_dir = output_dir / "config"
    input_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    replay_list = input_dir / "replay_image_list.txt"
    replay_list.write_text("\n".join(str(sample.image_path) for sample in samples) + "\n", encoding="utf-8")
    source_session = _resolve_path(source["dataset_root"], repo_root) / "sessions" / str(source["session_id"])
    shutil.copy2(source_session / str(source.get("samples_file", "samples.csv")), input_dir / "source_samples.csv")

    calibrated_extrinsics = _resolve_path(calibration["calibrated_extrinsics_file"], repo_root)
    virtual_extrinsics = config_dir / "virtual_bottom_tag_extrinsics.yaml"
    _copy_calibrated_extrinsics(calibrated_extrinsics, virtual_extrinsics, int(tag["id"]))

    camera_calibration = _resolve_path(calibration["camera_calibration_file"], repo_root)
    homography = _resolve_path(calibration["pool_homography_file"], repo_root)
    if not homography.is_file():
        raise SystemExit(f"native detector homography fixture not found: {homography}")
    direct_source_path = _resolve_path(calibration["direct_detector_config"], repo_root)
    direct_source = _load_yaml(direct_source_path)
    direct_node = direct_source.get("direct_apriltag_node")
    if not isinstance(direct_node, dict) or not isinstance(direct_node.get("ros__parameters"), dict):
        raise SystemExit(f"native direct detector config has no direct_apriltag_node.ros__parameters: {direct_source_path}")
    detector_runtime = copy.deepcopy(direct_node["ros__parameters"])
    # Apply only dataset-specific inputs, isolated topics, and nonessential
    # debug disablement. Detection tuning remains inherited from the runtime
    # configuration file recorded in the manifest.
    detector_runtime.update(
        {
            "device": "/dev/null",
            "enabled": True,
            "family": str(tag["family"]),
            "target_tag_ids": [int(tag["id"])],
            "homography_file": str(homography),
            "camera_calibration_file": str(camera_calibration),
            "marker_length_m": float(tag["marker_length_m"]),
            "publish_pnp_pose": False,
            "pnp_pose_topic": f"{topic_root}/pnp_camera",
            "pnp_array_topic": f"{topic_root}/tag_poses_3d_camera",
            "detection2d_topic": f"{topic_root}/tag_detections_2d",
            "pnp_frame_id": "finsrov_overhead_camera",
            "status_topic": f"{topic_root}/native_detector_status",
            "camera_status_topic": f"{topic_root}/camera_status",
            "debug_image_topic": f"{topic_root}/debug/compressed",
            "debug_image_enabled": False,
            "status_rate_hz": 1.0,
            "replay_image_list_file": str(replay_list),
            "replay_rate_hz": float(replay["rate_hz"]),
            "replay_event_topic": f"{topic_root}/replay_event",
        }
    )
    # The replay process receives an E2-specific node-name remap so that it
    # can coexist with a live detector.  A wildcard scope keeps this generated
    # parameter file valid after that intentional remap.
    detector_params = {"/**": {"ros__parameters": detector_runtime}}
    detector_params_path = config_dir / "native_detector_replay.yaml"
    _write_yaml(detector_params_path, detector_params)

    oracle_depth_topic = f"{topic_root}/oracle/depth"
    oracle_imu_topic = f"{topic_root}/oracle/imu"
    refractive_source_path = _resolve_path(calibration["refractive_pose_config"], repo_root)
    refractive_source = _load_yaml(refractive_source_path)
    refractive_node = refractive_source.get("refractive_apriltag_pose")
    if not isinstance(refractive_node, dict) or not isinstance(refractive_node.get("ros__parameters"), dict):
        raise SystemExit(
            f"native refractive pose config has no refractive_apriltag_pose.ros__parameters: {refractive_source_path}"
        )
    refractive_runtime = copy.deepcopy(refractive_node["ros__parameters"])
    refractive_runtime.update(
        {
            "detection_topic": f"{topic_root}/tag_detections_2d",
            "depth_topic": oracle_depth_topic,
            "imu_topic": oracle_imu_topic,
            "constrained_pose_topic": f"{topic_root}/snell_pose",
            "pure_pose_topic": f"{topic_root}/pinhole_pose",
            "status_topic": f"{topic_root}/refractive_status",
            "world_frame_id": "pool_world",
            "body_frame_id": "apriltag_bottom_virtual_body",
            "camera_calibration_file": str(camera_calibration),
            "extrinsics_file": str(virtual_extrinsics),
            "marker_length_m": float(tag["marker_length_m"]),
            "water_surface_z_m": float(geometry["water_surface_z_m"]),
            "water_plane_normal": [0.0, 0.0, 1.0],
            "water_plane_d": -float(geometry["water_surface_z_m"]),
            "depth_sign": -1.0,
            "pressure_sensor_offset_z_body": 0.0,
            "n_air": float(geometry["n_air"]),
            "n_water": float(geometry["n_water"]),
            "min_visible_tags": 1,
            "mode_selection_source": "depth",
            "publish_air_pose_on_constrained_topic": False,
            "publish_surface_transition_pose_on_constrained_topic": False,
            # Never substitute a pinhole fallback into the Snell result.
            "publish_pinhole_fallback_on_constrained_topic": False,
            "pinhole_fallback_use_pressure_depth_z": False,
        }
    )
    refractive_params = {"/**": {"ros__parameters": refractive_runtime}}
    refractive_params_path = config_dir / "refractive_pose_replay.yaml"
    _write_yaml(refractive_params_path, refractive_params)
    return {
        "replay_list": replay_list,
        "source_samples": input_dir / "source_samples.csv",
        "virtual_extrinsics": virtual_extrinsics,
        "detector_params": detector_params_path,
        "refractive_params": refractive_params_path,
        "homography": homography,
        "oracle_depth_topic": Path(oracle_depth_topic),
        "oracle_imu_topic": Path(oracle_imu_topic),
        "event_topic": Path(f"{topic_root}/replay_event"),
        "detector_status_topic": Path(f"{topic_root}/native_detector_status"),
        "refractive_status_topic": Path(f"{topic_root}/refractive_status"),
        "pinhole_topic": Path(f"{topic_root}/pinhole_pose"),
        "snell_topic": Path(f"{topic_root}/snell_pose"),
    }


def _stamp_ns(stamp: Any) -> int:
    return int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)


def _pose_value(message: PoseWithCovarianceStamped) -> dict[str, float]:
    pose = message.pose.pose
    return {
        "x_m": float(pose.position.x),
        "y_m": float(pose.position.y),
        "z_m": float(pose.position.z),
        "qx": float(pose.orientation.x),
        "qy": float(pose.orientation.y),
        "qz": float(pose.orientation.z),
        "qw": float(pose.orientation.w),
    }


class _ReplayCollector(Node):
    """Publishes declared oracle constraints and joins native messages by stamp."""

    def __init__(self, topics: dict[str, Path], geometry: dict[str, Any], event_log: Path) -> None:
        super().__init__("apriltag_localization_replay_collector")
        self._depth_pub = self.create_publisher(PoseWithCovarianceStamped, str(topics["oracle_depth_topic"]), 20)
        self._imu_pub = self.create_publisher(Imu, str(topics["oracle_imu_topic"]), 20)
        self.create_subscription(String, str(topics["event_topic"]), self._on_event, 50)
        self.create_subscription(String, str(topics["detector_status_topic"]), self._on_detector_status, 50)
        self.create_subscription(String, str(topics["refractive_status_topic"]), self._on_refractive_status, 50)
        self.create_subscription(PoseWithCovarianceStamped, str(topics["pinhole_topic"]), self._on_pinhole, 50)
        self.create_subscription(PoseWithCovarianceStamped, str(topics["snell_topic"]), self._on_snell, 50)
        self._timer = self.create_timer(0.05, self._publish_oracle_constraints)
        self._water_surface_z = float(geometry["water_surface_z_m"])
        self._pressure_depth_m = float(geometry["oracle_pressure_depth_m"])
        rpy = geometry.get("oracle_roll_pitch_yaw_rad", [0.0, 0.0, 0.0])
        if not isinstance(rpy, list) or len(rpy) != 3:
            raise SystemExit("geometry.oracle_roll_pitch_yaw_rad must have three values")
        self._rpy = tuple(float(value) for value in rpy)
        event_log.parent.mkdir(parents=True, exist_ok=True)
        self._event_log = event_log.open("w", encoding="utf-8")
        self.detector_status: dict[int, dict[str, Any]] = {}
        self.refractive_status: dict[int, dict[str, Any]] = {}
        self.pinhole: dict[int, dict[str, float]] = {}
        self.snell: dict[int, dict[str, float]] = {}
        self.replay_errors: dict[int, str] = {}
        self.complete_at: float | None = None

    def close(self) -> None:
        self._event_log.close()

    def _write_event(self, stream: str, payload: dict[str, Any]) -> None:
        self._event_log.write(json.dumps({"received_at": now_iso(), "stream": stream, **payload}, ensure_ascii=True) + "\n")
        self._event_log.flush()

    def _publish_oracle_constraints(self) -> None:
        stamp = self.get_clock().now().to_msg()
        depth = PoseWithCovarianceStamped()
        depth.header.stamp = stamp
        depth.header.frame_id = "pool_world"
        # The refractive node interprets depth_raw.position.z as positive
        # pressure depth when depth_sign=-1.  With surface z=1, depth=1 gives
        # pressure sensor/body z=0 for the identity virtual tag transform.
        depth.pose.pose.position.z = self._pressure_depth_m
        self._depth_pub.publish(depth)

        roll, pitch, yaw = self._rpy
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = "apriltag_localization_oracle_imu"
        cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
        cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
        cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
        imu.orientation.x = sr * cp * cy - cr * sp * sy
        imu.orientation.y = cr * sp * cy + sr * cp * sy
        imu.orientation.z = cr * cp * sy - sr * sp * cy
        imu.orientation.w = cr * cp * cy + sr * sp * sy
        self._imu_pub.publish(imu)

    def _on_event(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
        except json.JSONDecodeError:
            self._write_event("replay_event_invalid", {"raw": message.data})
            return
        self._write_event("replay_event", payload)
        index = payload.get("replay_index")
        if isinstance(index, int) and bool(payload.get("error", False)):
            self.replay_errors[index] = str(payload.get("phase", "replay_error"))
        if payload.get("phase") == "complete":
            self.complete_at = time.monotonic()

    def _on_detector_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            index = payload.get("replay_index")
            if not isinstance(index, int):
                return
            self.detector_status[index] = payload
            self._write_event("native_detector_status", payload)
        except json.JSONDecodeError:
            self._write_event("native_detector_status_invalid", {"raw": message.data})

    def _on_refractive_status(self, message: String) -> None:
        try:
            payload = json.loads(message.data)
            stamp = payload.get("input_stamp_ns")
            if not isinstance(stamp, int):
                return
            self.refractive_status[stamp] = payload
            self._write_event("refractive_status", payload)
        except json.JSONDecodeError:
            self._write_event("refractive_status_invalid", {"raw": message.data})

    def _on_pinhole(self, message: PoseWithCovarianceStamped) -> None:
        stamp = _stamp_ns(message.header.stamp)
        payload = _pose_value(message)
        self.pinhole[stamp] = payload
        self._write_event("pinhole_pose", {"input_stamp_ns": stamp, **payload})

    def _on_snell(self, message: PoseWithCovarianceStamped) -> None:
        stamp = _stamp_ns(message.header.stamp)
        payload = _pose_value(message)
        self.snell[stamp] = payload
        self._write_event("snell_pose", {"input_stamp_ns": stamp, **payload})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _horizontal_metrics(values: Sequence[tuple[np.ndarray, np.ndarray]]) -> dict[str, Any]:
    if not values:
        return {
            "n": 0,
            "horizontal_rmse_m": None,
            "mean_horizontal_error_m": None,
            "horizontal_p95_m": None,
            "bias_x_m": None,
            "bias_y_m": None,
        }
    delta = np.asarray([estimate - truth for estimate, truth in values], dtype=np.float64)
    horizontal = np.linalg.norm(delta, axis=1)
    return {
        "n": int(len(values)),
        "horizontal_rmse_m": float(np.sqrt(np.mean(horizontal**2))),
        "mean_horizontal_error_m": float(np.mean(horizontal)),
        "horizontal_p95_m": float(np.percentile(horizontal, 95)),
        "bias_x_m": float(np.mean(delta[:, 0])),
        "bias_y_m": float(np.mean(delta[:, 1])),
    }


def _build_frame_rows(samples: Sequence[DatasetSample], collector: _ReplayCollector) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sample in samples:
        native = collector.detector_status.get(sample.replay_index, {})
        stamp = native.get("frame_stamp_ns")
        if not isinstance(stamp, int):
            stamp = None
        refractive = collector.refractive_status.get(stamp, {}) if stamp is not None else {}
        pinhole = collector.pinhole.get(stamp) if stamp is not None else None
        snell = collector.snell.get(stamp) if stamp is not None else None
        row: dict[str, Any] = {
            "replay_index": sample.replay_index,
            "sample_id": sample.sample_id,
            "image_path": str(sample.image_path),
            "annotation_truth_x_m": sample.annotation_truth_x_m,
            "annotation_truth_y_m": sample.annotation_truth_y_m,
            "truth_x_m": sample.truth_x_m,
            "truth_y_m": sample.truth_y_m,
            "native_detector_status_received": bool(native),
            "native_tag_detected": native.get("detected"),
            "native_tag_id": native.get("tag_id"),
            "native_decision_margin": native.get("decision_margin"),
            "native_pnp_reprojection_error_px": native.get("pnp_reprojection_error_px"),
            "refractive_status_received": bool(refractive),
            "refractive_mode": refractive.get("mode"),
            "snell_output_model": refractive.get("constrained_output_model"),
            "snell_reject_reason": refractive.get("snell_reject_reason"),
            "replay_error": collector.replay_errors.get(sample.replay_index, ""),
            "pinhole_x_m": None,
            "pinhole_y_m": None,
            "pinhole_error_x_m": None,
            "pinhole_error_y_m": None,
            "pinhole_horizontal_error_m": None,
            "snell_x_m": None,
            "snell_y_m": None,
            "snell_error_x_m": None,
            "snell_error_y_m": None,
            "snell_horizontal_error_m": None,
        }
        truth = np.array([sample.truth_x_m, sample.truth_y_m], dtype=np.float64)
        for method, pose in (("pinhole", pinhole), ("snell", snell)):
            if pose is None:
                continue
            estimate = np.array([pose["x_m"], pose["y_m"]], dtype=np.float64)
            delta = estimate - truth
            row[f"{method}_x_m"] = float(estimate[0])
            row[f"{method}_y_m"] = float(estimate[1])
            row[f"{method}_error_x_m"] = float(delta[0])
            row[f"{method}_error_y_m"] = float(delta[1])
            row[f"{method}_horizontal_error_m"] = float(np.linalg.norm(delta))
        rows.append(row)
    return rows


def _point_rows(frame_rows: Sequence[dict[str, Any]], method: str) -> list[dict[str, Any]]:
    grouped: dict[tuple[float, float], list[dict[str, Any]]] = {}
    for row in frame_rows:
        grouped.setdefault((float(row["truth_x_m"]), float(row["truth_y_m"])), []).append(row)
    result: list[dict[str, Any]] = []
    for (truth_x, truth_y), items in sorted(grouped.items()):
        valid = [
            np.array([float(row[f"{method}_x_m"]), float(row[f"{method}_y_m"])])
            for row in items
            if row[f"{method}_x_m"] is not None and row[f"{method}_y_m"] is not None
        ]
        row: dict[str, Any] = {
            "method": method,
            "truth_x_m": truth_x,
            "truth_y_m": truth_y,
            "saved_image_count": len(items),
            "valid_estimate_count": len(valid),
            "availability_on_saved_images": len(valid) / len(items) if items else 0.0,
            "estimate_x_m": None,
            "estimate_y_m": None,
            "error_x_m": None,
            "error_y_m": None,
            "horizontal_error_m": None,
        }
        if valid:
            estimate = np.mean(np.asarray(valid), axis=0)
            delta = estimate - np.array([truth_x, truth_y])
            row.update(
                {
                    "estimate_x_m": float(estimate[0]),
                    "estimate_y_m": float(estimate[1]),
                    "error_x_m": float(delta[0]),
                    "error_y_m": float(delta[1]),
                    "horizontal_error_m": float(np.linalg.norm(delta)),
                }
            )
        result.append(row)
    return result


def _method_summary(frame_rows: Sequence[dict[str, Any]], point_rows: Sequence[dict[str, Any]], method: str) -> dict[str, Any]:
    frame_values = [
        (
            np.array([float(row[f"{method}_x_m"]), float(row[f"{method}_y_m"])]),
            np.array([float(row["truth_x_m"]), float(row["truth_y_m"])]),
        )
        for row in frame_rows
        if row[f"{method}_x_m"] is not None and row[f"{method}_y_m"] is not None
    ]
    points = [row for row in point_rows if row["method"] == method]
    point_values = [
        (
            np.array([float(row["estimate_x_m"]), float(row["estimate_y_m"])]),
            np.array([float(row["truth_x_m"]), float(row["truth_y_m"])]),
        )
        for row in points
        if row["estimate_x_m"] is not None and row["estimate_y_m"] is not None
    ]
    return {
        "saved_images": len(frame_rows),
        "valid_frames": len(frame_values),
        "frame_availability_on_saved_images": len(frame_values) / len(frame_rows) if frame_rows else 0.0,
        "frame_metrics": _horizontal_metrics(frame_values),
        "truth_position_count": len(points),
        "valid_truth_position_count": len(point_values),
        "truth_position_mean_metrics": _horizontal_metrics(point_values),
    }


def _write_plots(output_dir: Path, point_rows: Sequence[dict[str, Any]]) -> list[str]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return [f"matplotlib unavailable: {exc}"]
    plot_dir = output_dir / "derived" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    fig, axis = plt.subplots(figsize=(6.2, 3.8), constrained_layout=True)
    for method, color, label in (
        ("pinhole", "tab:orange", "Pinhole PnP baseline"),
        ("snell", "tab:blue", "Snell + oracle depth/level attitude"),
    ):
        errors = sorted(
            float(row["horizontal_error_m"])
            for row in point_rows
            if row["method"] == method and row["horizontal_error_m"] is not None
        )
        if errors:
            axis.plot(errors, np.linspace(0.0, 1.0, len(errors)), color=color, label=label)
    axis.set_xlabel("horizontal error [m]")
    axis.set_ylabel("empirical CDF")
    axis.set_title("Archived submerged-tag localization: native ROS2 replay")
    axis.grid(True, alpha=0.3)
    axis.legend(loc="lower right")
    cdf = plot_dir / "horizontal_error_cdf.png"
    fig.savefig(cdf, dpi=180)
    plt.close(fig)
    paths.append(str(cdf.relative_to(output_dir)))

    for method, cmap in (("pinhole", "Oranges"), ("snell", "Blues")):
        rows = [
            row for row in point_rows
            if row["method"] == method and row["horizontal_error_m"] is not None
        ]
        if not rows:
            continue
        x = np.asarray([float(row["truth_x_m"]) for row in rows])
        y = np.asarray([float(row["truth_y_m"]) for row in rows])
        estimate_x = np.asarray([float(row["estimate_x_m"]) for row in rows])
        estimate_y = np.asarray([float(row["estimate_y_m"]) for row in rows])
        error = np.asarray([float(row["horizontal_error_m"]) for row in rows])
        fig, axis = plt.subplots(figsize=(6.0, 4.3), constrained_layout=True)
        scatter = axis.scatter(x, y, c=error, cmap=cmap, s=62, edgecolors="black", linewidths=0.25)
        axis.set_xlabel("pool_world x [m]")
        axis.set_ylabel("pool_world y [m]")
        axis.set_title(f"{method}: truth-position mean horizontal error")
        axis.set_aspect("equal", adjustable="box")
        # Put the scale below the spatial plot so that the map has enough
        # horizontal room when it is reduced to one IEEE column.
        fig.colorbar(
            scatter,
            ax=axis,
            orientation="horizontal",
            pad=0.16,
            label="horizontal error [m]",
        )
        heatmap = plot_dir / f"{method}_error_heatmap.png"
        fig.savefig(heatmap, dpi=180)
        plt.close(fig)
        paths.append(str(heatmap.relative_to(output_dir)))

        fig, axis = plt.subplots(figsize=(6.0, 4.3), constrained_layout=True)
        axis.scatter(x, y, color="black", s=18, label="survey truth")
        axis.quiver(x, y, estimate_x - x, estimate_y - y, angles="xy", scale_units="xy", scale=1.0,
                    color="tab:red", width=0.004, label="estimate - truth")
        axis.set_xlabel("pool_world x [m]")
        axis.set_ylabel("pool_world y [m]")
        axis.set_title(f"{method}: truth-position mean error vectors")
        axis.set_aspect("equal", adjustable="box")
        axis.legend(loc="best")
        vectors = plot_dir / f"{method}_error_vectors.png"
        fig.savefig(vectors, dpi=180)
        plt.close(fig)
        paths.append(str(vectors.relative_to(output_dir)))

    coverage = [row for row in point_rows if row["method"] == "snell"]
    if coverage:
        fig, axis = plt.subplots(figsize=(6.0, 4.3), constrained_layout=True)
        x = [float(row["truth_x_m"]) for row in coverage]
        y = [float(row["truth_y_m"]) for row in coverage]
        availability = [float(row["availability_on_saved_images"]) for row in coverage]
        scatter = axis.scatter(x, y, c=availability, vmin=0.0, vmax=1.0, cmap="viridis", s=62,
                              edgecolors="black", linewidths=0.25)
        axis.set_xlabel("pool_world x [m]")
        axis.set_ylabel("pool_world y [m]")
        axis.set_title("Snell estimate availability on archived saved images")
        axis.set_aspect("equal", adjustable="box")
        fig.colorbar(scatter, ax=axis, label="availability")
        availability_path = plot_dir / "snell_saved_image_availability.png"
        fig.savefig(availability_path, dpi=180)
        plt.close(fig)
        paths.append(str(availability_path.relative_to(output_dir)))
    return paths


def _write_readme(output_dir: Path, report: dict[str, Any]) -> None:
    metrics = report["metrics"]
    lines = [
        "# E2 AprilTag 定位消融：原生 ROS2 链路回放评估",
        "",
        "此目录是 E2 AprilTag 定位消融结果；原始图像按顺序进入 C++ `direct_apriltag_node`，",
        "再进入 C++ `refractive_apriltag_pose_node`。`pinhole` 是该姿态节点发布的纯视觉 PnP 基线，",
        "`snell` 是同一节点发布的 Snell 折射、已知深度和水平姿态约束解。",
        "",
        "## 主统计（物理真值点重复图像均值）",
        "",
        "| 方法 | RMSE [m] | Mean [m] | P95 [m] | 有效真值点 |",
        "|---|---:|---:|---:|---:|",
    ]
    for method, label in (("pinhole", "Pinhole PnP"), ("snell", "Snell + oracle constraints")):
        value = metrics[method]["truth_position_mean_metrics"]
        number = lambda key: "n/a" if value[key] is None else f"{float(value[key]):.4f}"
        lines.append(
            f"| {label} | {number('horizontal_rmse_m')} | {number('mean_horizontal_error_m')} | "
            f"{number('horizontal_p95_m')} | {value['n']} |"
        )
    lines.extend(
        [
            "",
            "## 结果文件",
            "",
            "- `raw/ros_events.ndjson`：原生 detector、refractive pose 和回放事件的审计记录；",
            "- `derived/frame_predictions.csv`：逐图像的两条链路输出与水平误差；",
            "- `derived/point_predictions.csv`：每个物理真值位置上重复图像的均值；",
            "- `derived/summary_metrics.csv` / `derived/analysis_report.json`：可引用指标与配置哈希；",
            "- `derived/plots/`：CDF、空间误差热图、误差向量和可用性图。",
            "",
            "## 边界",
            "",
            "历史数据没有同步 IMU/深度日志，因而本评估向 runtime 节点发布了已声明的 1 m 深度和水平姿态 oracle。",
            "它只支持 `pool_world` 水平坐标精度结论；不报告 depth、roll/pitch/yaw、EKF 融合或完整视频流检测率。",
            "`T_world_camera` 来自已标定的运行时外参文件，数据集真值不参与外参拟合。",
            "",
        ]
    )
    (output_dir / "README.md").write_text("\n".join(lines), encoding="utf-8")


def _start_native_node(
    workspace: Path,
    package: str,
    executable: str,
    params_file: Path,
    log_path: Path,
    node_name: str,
) -> subprocess.Popen[str]:
    command = [
        str(workspace / "scripts" / "run_ros2_uv.sh"),
        "ros2",
        "run",
        package,
        executable,
        "--ros-args",
        "-r",
        f"__node:={node_name}",
        "--params-file",
        str(params_file),
    ]
    stream = log_path.open("w", encoding="utf-8")
    process = subprocess.Popen(
        command,
        cwd=workspace,
        stdout=stream,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )
    # Popen does not own the file handle after this function returns.  The
    # child inherited its descriptor, so close the parent copy immediately.
    stream.close()
    return process


def _stop_process(process: subprocess.Popen[str] | None) -> None:
    if process is None or process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGINT)
        process.wait(timeout=8.0)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=4.0)
        except (ProcessLookupError, subprocess.TimeoutExpired):
            os.killpg(process.pid, signal.SIGKILL)


def run_apriltag_localization(
    config_path: Path,
    *,
    session_id: str | None = None,
    output_root_override: Path | None = None,
    overwrite: bool = False,
    limit: int | None = None,
    dry_run: bool = False,
) -> Path:
    repo_root = _repo_root()
    config_path = config_path.expanduser().resolve()
    config = _load_yaml(config_path)
    experiment = _mapping(config, "experiment")
    source = _mapping(config, "source")
    calibration = _mapping(config, "runtime_calibration")
    geometry = _mapping(config, "geometry")
    replay = _mapping(config, "replay")
    tag = _mapping(config, "tag")
    source_session = _resolve_path(source["dataset_root"], repo_root) / "sessions" / str(source["session_id"])
    truth_to_pool_world = source.get("truth_to_pool_world")
    if not isinstance(truth_to_pool_world, dict):
        raise SystemExit("source.truth_to_pool_world must be a mapping")
    truth_annotation_to_tag_center = source.get("truth_annotation_to_tag_center")
    if not isinstance(truth_annotation_to_tag_center, dict):
        raise SystemExit("source.truth_annotation_to_tag_center must be a mapping")
    samples = _load_samples(
        source_session,
        str(source.get("samples_file", "samples.csv")),
        truth_to_pool_world,
        truth_annotation_to_tag_center,
        limit,
    )
    if float(geometry["water_surface_z_m"]) <= float(geometry["tag_plane_z_m"]):
        raise SystemExit("geometry.water_surface_z_m must be above geometry.tag_plane_z_m")
    if abs(float(geometry["oracle_pressure_depth_m"]) - (float(geometry["water_surface_z_m"]) - float(geometry["tag_plane_z_m"]))) > 1e-6:
        raise SystemExit("oracle_pressure_depth_m must place the virtual bottom tag on geometry.tag_plane_z_m")

    output_root = output_root_override or _resolve_path(experiment["output_root"], repo_root)
    output_root.mkdir(parents=True, exist_ok=True)
    resolved_session_id = session_id or _new_session_id(output_root, str(experiment["id"]))
    output_dir = safe_session_dir(output_root, resolved_session_id)
    if output_dir.exists() and not overwrite:
        raise SystemExit(f"output already exists: {output_dir}; use --overwrite for this exact session only")
    if dry_run:
        print(json.dumps({"source_session": str(source_session), "images": len(samples), "output_dir": str(output_dir)}, indent=2))
        return output_dir
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True)
    shutil.copy2(config_path, output_dir / "resolved_config.yaml")
    paths = _prepare_native_params(output_dir, samples, config, repo_root)
    logs = output_dir / "logs"
    logs.mkdir(exist_ok=True)

    calibrated_extrinsics = _resolve_path(calibration["calibrated_extrinsics_file"], repo_root)
    source_samples_csv = source_session / str(source.get("samples_file", "samples.csv"))
    collector: _ReplayCollector | None = None
    refractive_process: subprocess.Popen[str] | None = None
    detector_process: subprocess.Popen[str] | None = None
    rclpy.init(args=None)
    try:
        collector = _ReplayCollector(paths, geometry, output_dir / "raw" / "ros_events.ndjson")
        refractive_process = _start_native_node(
            repo_root / "ros2_ws", "perception", "refractive_apriltag_pose_node",
            paths["refractive_params"], logs / "refractive_apriltag_pose_node.log",
            "e2_refractive_apriltag_pose_node",
        )
        warmup_until = time.monotonic() + float(replay["startup_wait_sec"])
        while time.monotonic() < warmup_until:
            rclpy.spin_once(collector, timeout_sec=0.05)
        if refractive_process.poll() is not None:
            raise RuntimeError("refractive_apriltag_pose_node exited during startup; see logs")
        detector_process = _start_native_node(
            repo_root / "ros2_ws", "perception", "direct_apriltag_node",
            paths["detector_params"], logs / "direct_apriltag_node.log",
            "e2_direct_apriltag_node",
        )
        started = time.monotonic()
        grace = float(replay["completion_grace_sec"])
        deadline = started + float(replay["max_runtime_sec"])
        last_info = 0.0
        while True:
            rclpy.spin_once(collector, timeout_sec=0.1)
            now = time.monotonic()
            if now - last_info >= 5.0:
                print(
                    f"[INFO] native replay: detector statuses={len(collector.detector_status)}/{len(samples)}, "
                    f"pinhole={len(collector.pinhole)}, snell={len(collector.snell)}"
                )
                last_info = now
            if detector_process.poll() is not None and collector.complete_at is None:
                raise RuntimeError("direct_apriltag_node exited before replay completion; see logs")
            if refractive_process.poll() is not None:
                raise RuntimeError("refractive_apriltag_pose_node exited during replay; see logs")
            if collector.complete_at is not None and now - collector.complete_at >= grace:
                break
            if now >= deadline:
                raise RuntimeError(f"native replay timed out after {float(replay['max_runtime_sec']):.1f}s")
    finally:
        _stop_process(detector_process)
        _stop_process(refractive_process)
        if collector is not None:
            collector.close()
            collector.destroy_node()
        rclpy.shutdown()

    # Build all metrics from the raw messages only after both nodes have been
    # stopped, avoiding a partial output directory being mistaken for a result.
    # ``collector`` is guaranteed non-null here because a successful run got
    # through construction above.
    if collector is None:
        raise RuntimeError("collector did not start")
    frame_rows = _build_frame_rows(samples, collector)
    points = _point_rows(frame_rows, "pinhole") + _point_rows(frame_rows, "snell")
    metrics = {
        "pinhole": _method_summary(frame_rows, points, "pinhole"),
        "snell": _method_summary(frame_rows, points, "snell"),
    }
    derived = output_dir / "derived"
    _write_csv(
        derived / "frame_predictions.csv", frame_rows,
        [
            "replay_index", "sample_id", "image_path", "annotation_truth_x_m", "annotation_truth_y_m", "truth_x_m", "truth_y_m",
            "native_detector_status_received", "native_tag_detected", "native_tag_id", "native_decision_margin",
            "native_pnp_reprojection_error_px", "refractive_status_received", "refractive_mode",
            "snell_output_model", "snell_reject_reason", "replay_error",
            "pinhole_x_m", "pinhole_y_m", "pinhole_error_x_m", "pinhole_error_y_m", "pinhole_horizontal_error_m",
            "snell_x_m", "snell_y_m", "snell_error_x_m", "snell_error_y_m", "snell_horizontal_error_m",
        ],
    )
    _write_csv(
        derived / "point_predictions.csv", points,
        [
            "method", "truth_x_m", "truth_y_m", "saved_image_count", "valid_estimate_count",
            "availability_on_saved_images", "estimate_x_m", "estimate_y_m", "error_x_m", "error_y_m",
            "horizontal_error_m",
        ],
    )
    summary_rows: list[dict[str, Any]] = []
    for method, values in metrics.items():
        for unit, key in (("saved_image_frame", "frame_metrics"), ("truth_position_mean", "truth_position_mean_metrics")):
            summary_rows.append({"method": method, "statistical_unit": unit, **values[key]})
    _write_csv(
        derived / "summary_metrics.csv", summary_rows,
        ["method", "statistical_unit", "n", "horizontal_rmse_m", "mean_horizontal_error_m", "horizontal_p95_m", "bias_x_m", "bias_y_m"],
    )
    plots = _write_plots(output_dir, points)
    extrinsic_provenance = calibration.get("t_world_camera_provenance")
    if extrinsic_provenance is not None and not isinstance(extrinsic_provenance, dict):
        raise SystemExit("runtime_calibration.t_world_camera_provenance must be a YAML mapping when present")
    if extrinsic_provenance:
        extrinsic_caveat = (
            "T_world_camera is a declared full-E2-dataset planar-Snell refit; this replay is "
            "an in-sample calibration-consistency check, not an independent localization-accuracy estimate."
        )
    else:
        extrinsic_caveat = "No archived truth sample is used to fit or refine T_world_camera."
    report = {
        "analyzer": "experiment_recorder.apriltag_localization",
        "analysis_time": now_iso(),
        "experiment_id": str(experiment["id"]),
        "session_id": resolved_session_id,
        "scope": {
            "evaluated_dimensions": ["pool_world_x", "pool_world_y"],
            "statistical_unit": "truth_position_mean",
            "not_evaluated": ["depth", "roll", "pitch", "yaw", "ekf_fusion", "full_camera_stream_detection_rate"],
            "constraint_label": "Snell + declared oracle pressure depth and level IMU",
        },
        "runtime_chain": {
            "detector": "perception/direct_apriltag_node (replay_image_list_file)",
            "pose_estimator": "perception/refractive_apriltag_pose_node",
            "pinhole_topic": str(paths["pinhole_topic"]),
            "snell_topic": str(paths["snell_topic"]),
            "fallback_on_snell_topic": False,
            "replay_rate_hz": float(replay["rate_hz"]),
        },
        "input": {
            "source_session": str(source_session),
            "truth_annotation_frame": source.get("truth_annotation_frame"),
            "truth_to_pool_world": truth_to_pool_world,
            "truth_annotation_to_tag_center": truth_annotation_to_tag_center,
            "samples_csv": str(source_samples_csv),
            "samples_csv_sha256": file_sha256(source_samples_csv),
            "archived_image_count": len(samples),
            "camera_calibration_file": str(_resolve_path(calibration["camera_calibration_file"], repo_root)),
            "camera_calibration_sha256": file_sha256(_resolve_path(calibration["camera_calibration_file"], repo_root)),
            "direct_detector_config": str(_resolve_path(calibration["direct_detector_config"], repo_root)),
            "direct_detector_config_sha256": file_sha256(_resolve_path(calibration["direct_detector_config"], repo_root)),
            "refractive_pose_config": str(_resolve_path(calibration["refractive_pose_config"], repo_root)),
            "refractive_pose_config_sha256": file_sha256(_resolve_path(calibration["refractive_pose_config"], repo_root)),
            "pool_homography_file": str(paths["homography"]),
            "pool_homography_sha256": file_sha256(paths["homography"]),
            "calibrated_extrinsics_file": str(calibrated_extrinsics),
            "calibrated_extrinsics_sha256": file_sha256(calibrated_extrinsics),
            "t_world_camera_provenance": extrinsic_provenance,
            "virtual_tag_extrinsics_file": "config/virtual_bottom_tag_extrinsics.yaml",
            "virtual_tag_extrinsics_sha256": file_sha256(paths["virtual_extrinsics"]),
        },
        "geometry": {
            "water_surface_z_m": float(geometry["water_surface_z_m"]),
            "tag_plane_z_m": float(geometry["tag_plane_z_m"]),
            "n_air": float(geometry["n_air"]),
            "n_water": float(geometry["n_water"]),
            "oracle_pressure_depth_m": float(geometry["oracle_pressure_depth_m"]),
            "oracle_roll_pitch_yaw_rad": geometry["oracle_roll_pitch_yaw_rad"],
            "tag_id": int(tag["id"]),
            "marker_length_m": float(tag["marker_length_m"]),
        },
        "metrics": metrics,
        "plots": plots,
        "caveats": [
            extrinsic_caveat,
            "The virtual identity T_body_tag only maps the fixed bottom marker centre into the runtime pose-node body contract.",
            "Saved images are an archived sampling set and must not be interpreted as a continuous camera-stream detection rate.",
            "Oracle pressure depth and level attitude are injected because the archive has no synchronized IMU/depth telemetry.",
        ],
        "code_revision": git_revision(repo_root),
    }
    write_json(derived / "analysis_report.json", report)
    write_json(
        output_dir / "manifest.json",
        {
            "schema_version": 1,
            "created_at": now_iso(),
            "experiment_id": str(experiment["id"]),
            "session_id": resolved_session_id,
            "profile": str(experiment["profile"]),
            "config": str(config_path),
            "config_sha256": file_sha256(config_path),
            "analysis_report": "derived/analysis_report.json",
            "source_session": str(source_session),
            "code_revision": git_revision(repo_root),
        },
    )
    _write_readme(output_dir, report)
    return output_dir


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run E2 by replaying the archived AprilTag dataset through the native ROS2 detector and refractive pose chain."
    )
    parser.add_argument("--config", type=Path, default=None, help="Experiment YAML; defaults to config/e2_apriltag_localization_dataset.yaml")
    parser.add_argument("--session-id", default=None, help="Optional output session directory name")
    parser.add_argument("--output-root", type=Path, default=None, help="Optional output root override")
    parser.add_argument("--limit", type=int, default=None, help="Replay only the first N archived images (smoke test only)")
    parser.add_argument("--overwrite", action="store_true", help="Replace only the explicit --session-id output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate source/output paths without starting ROS2 nodes")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.limit is not None and args.limit <= 0:
        raise SystemExit("--limit must be positive")
    output_dir = run_apriltag_localization(
        args.config or _default_config(),
        session_id=args.session_id,
        output_root_override=args.output_root,
        overwrite=args.overwrite,
        limit=args.limit,
        dry_run=args.dry_run,
    )
    print(f"E2 AprilTag ablation output: {output_dir}")


if __name__ == "__main__":
    main()
