from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from msgs.msg import AprilTagDetection3DArray
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from .filters import PositionVelocityEKF, YawEKF
from .transforms import (
    inverse_rotate_vector,
    matrix_to_pose,
    normalize_quat_xyzw,
    pose_matrix,
    quat_to_rpy_xyzw,
    rpy_to_quat_xyzw,
    wrap_angle,
)


class StateFusionNode(Node):
    def __init__(self) -> None:
        super().__init__("state_fusion")

        self._vision_pnp_topic = str(
            self.declare_parameter("vision_pnp_topic", "/finsrov/vision/tag_poses_3d_camera").value
        )
        self._vision_6d_topic = str(
            self.declare_parameter("vision_6d_topic", "/finsrov/vision/refracted_pose_6d").value
        )
        self._use_vision_pnp = bool(self.declare_parameter("use_vision_pnp", True).value)
        self._use_vision_6d = bool(self.declare_parameter("use_vision_6d", False).value)
        self._use_vision_6d_z = bool(self.declare_parameter("use_vision_6d_z", False).value)
        self._imu_raw_topic = str(self.declare_parameter("imu_raw_topic", "/finsrov/hardware/imu_raw").value)
        self._depth_raw_topic = str(self.declare_parameter("depth_raw_topic", "/finsrov/hardware/depth_raw").value)
        self._pose_topic = str(self.declare_parameter("pose_topic", "/finsrov/pose").value)
        self._imu_topic = str(self.declare_parameter("imu_topic", "/finsrov/imu_link").value)
        self._depth_topic = str(self.declare_parameter("depth_topic", "/finsrov/depth_link").value)
        self._dvl_topic = str(self.declare_parameter("dvl_topic", "/finsrov/dvl_link").value)
        self._status_topic = str(self.declare_parameter("status_topic", "/finsrov/state/status").value)
        self._world_frame_id = str(self.declare_parameter("world_frame_id", "pool_world").value)
        self._body_frame_id = str(self.declare_parameter("body_frame_id", "finsrov_base_link").value)
        self._depth_frame_id = str(self.declare_parameter("depth_frame_id", "finsrov_depth_link").value)
        self._extrinsics_file = str(
            self.declare_parameter("extrinsics_file", "config/state_fusion_extrinsics.yaml").value
        )

        self._publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 60.0).value)
        self._vision_timeout_sec = float(self.declare_parameter("vision_timeout_sec", 0.25).value)
        self._imu_timeout_sec = float(self.declare_parameter("imu_timeout_sec", 0.10).value)
        self._depth_timeout_sec = float(self.declare_parameter("depth_timeout_sec", 0.30).value)
        # ROS/pool_world is z-up with z=0 at the pool bottom. Hardware depth is
        # positive downward from the water surface.
        self._water_surface_z_m = float(self.declare_parameter("water_surface_z_m", 1.0).value)
        self._depth_sign = float(self.declare_parameter("depth_sign", -1.0).value)
        self._pressure_sensor_offset_z_body = float(
            self.declare_parameter("pressure_sensor_offset_z_body", -0.0678).value
        )
        self._stale_velocity_decay = float(self.declare_parameter("stale_velocity_decay", 0.95).value)
        self._publish_pose_before_ready = bool(self.declare_parameter("publish_pose_before_ready", False).value)
        self._vision_coast_timeout_sec = float(self.declare_parameter("vision_coast_timeout_sec", 0.75).value)
        self._vision_hold_timeout_sec = float(self.declare_parameter("vision_hold_timeout_sec", 3.0).value)
        self._vision_hold_velocity_decay = float(self.declare_parameter("vision_hold_velocity_decay", 0.50).value)
        self._max_position_covariance_xy = float(self.declare_parameter("max_position_covariance_xy", 4.0).value)
        self._max_velocity_covariance_xy = float(self.declare_parameter("max_velocity_covariance_xy", 1.0).value)

        self._use_vision_yaw = bool(self.declare_parameter("use_vision_yaw", True).value)
        self._use_imu_yaw_measurement = bool(self.declare_parameter("use_imu_yaw_measurement", False).value)
        self._use_vision_z = bool(self.declare_parameter("use_vision_z", False).value)
        self._vision_yaw_covariance = float(self.declare_parameter("vision_yaw_covariance", 0.05).value)
        self._imu_yaw_covariance = float(self.declare_parameter("imu_yaw_covariance", 0.02).value)
        self._depth_covariance = float(self.declare_parameter("depth_covariance", 0.0025).value)
        self._vision_position_covariance_xy = float(
            self.declare_parameter("vision_position_covariance_xy", 0.0025).value
        )
        self._vision_position_covariance_z = float(self.declare_parameter("vision_position_covariance_z", 0.05).value)
        self._dvl_covariance = float(self.declare_parameter("dvl_covariance", 0.01).value)
        self._ekf_process_noise_position = float(self.declare_parameter("ekf_process_noise_position", 0.02).value)
        self._ekf_process_noise_velocity = float(self.declare_parameter("ekf_process_noise_velocity", 0.10).value)
        self._ekf_yaw_process_noise = float(self.declare_parameter("ekf_yaw_process_noise", 0.02).value)
        self._measurement_gate = float(self.declare_parameter("measurement_gate_mahalanobis", 9.0).value)
        self._max_reprojection_error_px = float(self.declare_parameter("max_reprojection_error_px", 4.0).value)
        self._multi_tag_conflict_distance_m = float(
            self.declare_parameter("multi_tag_conflict_distance_m", 0.25).value
        )

        self._t_world_camera, self._t_body_tag_inverse = self._load_extrinsics(self._extrinsics_file)
        self._position_ekf = PositionVelocityEKF(
            self._ekf_process_noise_position,
            self._ekf_process_noise_velocity,
        )
        self._yaw_ekf = YawEKF(self._ekf_yaw_process_noise)

        self._imu: Imu | None = None
        self._roll_pitch = (0.0, 0.0)
        self._latest_yaw_rate = 0.0
        self._depth: float | None = None
        self._last_vision_time: float | None = None
        self._last_vision_position_xy: np.ndarray | None = None
        self._last_depth_time: float | None = None
        self._last_imu_time: float | None = None
        self._last_used_tag_ids: list[int] = []
        self._last_reject_reason = "not_ready"
        self._last_vision_mahalanobis: float | None = None
        self._last_depth_mahalanobis: float | None = None
        self._last_yaw_mahalanobis: float | None = None
        self._last_vision_source = "none"
        self._vision_update_count = 0
        self._vision_reject_count = 0
        self._last_logged_state: tuple | None = None

        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, self._pose_topic, 10)
        self._imu_pub = self.create_publisher(Imu, self._imu_topic, 10)
        self._depth_pub = self.create_publisher(PoseWithCovarianceStamped, self._depth_topic, 10)
        self._dvl_pub = self.create_publisher(TwistWithCovarianceStamped, self._dvl_topic, 10)
        self._status_pub = self.create_publisher(String, self._status_topic, 10)

        if self._use_vision_pnp:
            self.create_subscription(AprilTagDetection3DArray, self._vision_pnp_topic, self._vision_pnp_callback, 10)
        if self._use_vision_6d:
            self.create_subscription(PoseWithCovarianceStamped, self._vision_6d_topic, self._vision_6d_callback, 10)
        self.create_subscription(Imu, self._imu_raw_topic, self._imu_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, self._depth_raw_topic, self._depth_callback, 10)
        self._timer = self.create_timer(1.0 / max(self._publish_rate_hz, 1.0), self._publish)

        self.get_logger().info(
            "state_fusion EKF started: "
            f"vision_pnp={self._vision_pnp_topic}, use_vision_pnp={self._use_vision_pnp}, "
            f"vision_6d={self._vision_6d_topic}, use_vision_6d={self._use_vision_6d}, "
            f"imu_raw={self._imu_raw_topic}, depth_raw={self._depth_raw_topic}, "
            f"water_surface_z_m={self._water_surface_z_m}, depth_sign={self._depth_sign}, "
            f"pressure_sensor_offset_z_body={self._pressure_sensor_offset_z_body}, "
            f"extrinsics={self._resolve_path(self._extrinsics_file)}, pose={self._pose_topic}, dvl={self._dvl_topic}"
        )

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    @staticmethod
    def _msg_stamp_sec(msg) -> float:
        stamp = msg.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return value

    def _stamp_or_now(self, msg) -> float:
        stamp = self._msg_stamp_sec(msg)
        return stamp if stamp > 0.0 else self._now_sec()

    def _resolve_path(self, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if root := os.environ.get("FINSSIM_REPO_ROOT"):
            candidate = Path(root) / "ros2_ws" / "src" / "state_estimation" / path
            if candidate.exists():
                return candidate
        source_candidate = Path.cwd() / "src" / "state_estimation" / path
        if source_candidate.exists():
            return source_candidate
        try:
            share = Path(get_package_share_directory("state_estimation"))
            candidate = share / path
            if candidate.exists():
                return candidate
        except PackageNotFoundError:
            pass
        return path

    def _load_extrinsics(self, path_value: str) -> tuple[np.ndarray, dict[int, np.ndarray]]:
        path = self._resolve_path(path_value)
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        t_world_camera = self._transform_from_node(data.get("T_world_camera", {}), "T_world_camera")
        tag_map = self._load_tag_body_map(data)
        if not tag_map:
            self.get_logger().warn("state fusion extrinsics has no T_tag_body/T_body_tag entries")
        return t_world_camera, tag_map

    @classmethod
    def _load_tag_body_map(cls, data: dict) -> dict[int, np.ndarray]:
        tag_map: dict[int, np.ndarray] = {}
        for key, node in (data.get("T_tag_body", {}) or {}).items():
            tag_map[int(key)] = cls._transform_from_node(node, f"T_tag_body.{key}")

        for key, node in (data.get("T_body_tag", {}) or {}).items():
            tag_id = int(key)
            if tag_id in tag_map:
                raise ValueError(
                    f"tag {tag_id} is defined in both T_tag_body and T_body_tag; "
                    "define only one direction to avoid ambiguous extrinsics"
            )
            t_body_tag = cls._transform_from_node(node, f"T_body_tag.{key}")
            tag_map[tag_id] = np.linalg.inv(t_body_tag)
        return tag_map

    @staticmethod
    def _transform_from_node(node, name: str) -> np.ndarray:
        if node is None:
            node = {}
        if "matrix" in node:
            matrix = np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4)
            return matrix
        translation = node.get("translation_xyz", [0.0, 0.0, 0.0])
        if "rotation_xyzw" in node:
            rotation = node["rotation_xyzw"]
        elif "rotation_rpy_deg" in node:
            roll, pitch, yaw = (math.radians(float(value)) for value in node["rotation_rpy_deg"])
            rotation = rpy_to_quat_xyzw(roll, pitch, yaw)
        else:
            rotation = [0.0, 0.0, 0.0, 1.0]
        try:
            return pose_matrix(translation, rotation)
        except Exception as exc:  # noqa: BLE001 - include transform name in startup error.
            raise ValueError(f"invalid transform {name}") from exc

    def _fresh(self, stamp_sec: float | None, timeout_sec: float, now_sec: float) -> bool:
        if stamp_sec is None:
            return False
        if timeout_sec <= 0.0:
            return True
        return now_sec - stamp_sec <= timeout_sec

    def _vision_6d_callback(self, msg: PoseWithCovarianceStamped) -> None:
        stamp_sec = self._stamp_or_now(msg)
        self._position_ekf.predict(stamp_sec)
        self._yaw_ekf.predict(stamp_sec, self._latest_yaw_rate)

        pose = msg.pose.pose
        measurement_indices = [0, 1, 2] if self._use_vision_6d_z else [0, 1]
        measurement_values = [pose.position.x, pose.position.y]
        covariance_values = [
            self._covariance_value_or_default(msg.pose.covariance[0], self._vision_position_covariance_xy),
            self._covariance_value_or_default(msg.pose.covariance[7], self._vision_position_covariance_xy),
        ]
        if self._use_vision_6d_z:
            measurement_values.append(pose.position.z)
            covariance_values.append(
                self._covariance_value_or_default(msg.pose.covariance[14], self._vision_position_covariance_z)
            )

        accepted, distance = self._position_ekf.update_indices(
            measurement_indices,
            measurement_values,
            np.diag(covariance_values),
            gate_mahalanobis=self._measurement_gate,
        )
        self._last_vision_mahalanobis = distance
        if not accepted:
            self._vision_reject_count += 1
            self._last_reject_reason = "vision_6d_mahalanobis_gate"
            self._last_vision_source = "vision_6d_rejected"
            return

        self._last_vision_time = stamp_sec
        self._last_vision_position_xy = self._position_ekf.x[:2].copy()
        self._last_used_tag_ids = []
        self._last_reject_reason = "none"
        self._last_vision_source = "vision_6d"
        self._vision_update_count += 1

        if self._use_vision_yaw:
            quat = normalize_quat_xyzw(
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w]
            )
            _, _, yaw = quat_to_rpy_xyzw(quat)
            yaw_covariance = self._covariance_value_or_default(
                msg.pose.covariance[35], self._vision_yaw_covariance
            )
            yaw_accepted, yaw_distance = self._yaw_ekf.update(
                yaw,
                yaw_covariance,
                gate_mahalanobis=self._measurement_gate,
            )
            self._last_yaw_mahalanobis = yaw_distance
            if not yaw_accepted:
                self._last_reject_reason = "vision_6d_yaw_mahalanobis_gate"

    def _vision_pnp_callback(self, msg: AprilTagDetection3DArray) -> None:
        stamp_sec = self._stamp_or_now(msg)
        self._position_ekf.predict(stamp_sec)
        self._yaw_ekf.predict(stamp_sec, self._latest_yaw_rate)

        candidates = []
        reject_reasons: list[str] = []
        for detection in msg.detections:
            tag_id = int(detection.tag_id)
            if tag_id not in self._t_body_tag_inverse:
                reject_reasons.append(f"unknown_tag_{tag_id}")
                continue
            reproj = float(detection.reprojection_error_px)
            if self._max_reprojection_error_px > 0.0 and reproj > self._max_reprojection_error_px:
                reject_reasons.append(f"tag_{tag_id}_reprojection")
                continue
            pose = detection.pose.pose
            t_camera_tag = pose_matrix(
                [pose.position.x, pose.position.y, pose.position.z],
                [pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w],
            )
            t_world_body = self._t_world_camera @ t_camera_tag @ self._t_body_tag_inverse[tag_id]
            position, quat = matrix_to_pose(t_world_body)
            _, _, yaw = quat_to_rpy_xyzw(quat)
            covariance_xy = self._covariance_or_default(
                detection.pose.covariance[0],
                detection.pose.covariance[7],
                self._vision_position_covariance_xy,
            )
            weight = 1.0 / max(covariance_xy * (1.0 + max(reproj, 0.0)), 1e-9)
            candidates.append(
                {
                    "tag_id": tag_id,
                    "position": position,
                    "yaw": yaw,
                    "weight": weight,
                    "reprojection": reproj,
                }
            )

        if not candidates:
            self._vision_reject_count += 1
            self._last_reject_reason = ",".join(reject_reasons) if reject_reasons else "no_pnp_detections"
            return

        selected = self._select_or_fuse_candidates(candidates)
        position = selected["position"]
        measurement_indices = [0, 1, 2] if self._use_vision_z else [0, 1]
        measurement = position[measurement_indices]
        covariance_values = [self._vision_position_covariance_xy, self._vision_position_covariance_xy]
        if self._use_vision_z:
            covariance_values.append(self._vision_position_covariance_z)
        accepted, distance = self._position_ekf.update_indices(
            measurement_indices,
            measurement,
            np.diag(covariance_values),
            gate_mahalanobis=self._measurement_gate,
        )
        self._last_vision_mahalanobis = distance
        if not accepted:
            self._vision_reject_count += 1
            self._last_reject_reason = "vision_mahalanobis_gate"
            return

        self._last_vision_time = stamp_sec
        self._last_vision_position_xy = self._position_ekf.x[:2].copy()
        self._last_used_tag_ids = selected["tag_ids"]
        self._last_reject_reason = selected["reason"]
        self._last_vision_source = "vision_pnp"
        self._vision_update_count += 1

        if self._use_vision_yaw:
            yaw_accepted, yaw_distance = self._yaw_ekf.update(
                selected["yaw"],
                self._vision_yaw_covariance,
                gate_mahalanobis=self._measurement_gate,
            )
            self._last_yaw_mahalanobis = yaw_distance
            if not yaw_accepted:
                self._last_reject_reason = "vision_yaw_mahalanobis_gate"

    @staticmethod
    def _covariance_or_default(x_cov: float, y_cov: float, default: float) -> float:
        values = [float(v) for v in (x_cov, y_cov) if float(v) > 0.0 and float(v) < 1e5]
        return float(np.mean(values)) if values else float(default)

    @staticmethod
    def _covariance_value_or_default(value: float, default: float) -> float:
        value = float(value)
        return value if value > 0.0 and value < 1e5 else float(default)

    def _select_or_fuse_candidates(self, candidates: list[dict]) -> dict:
        if len(candidates) == 1:
            candidate = candidates[0]
            return {
                "position": candidate["position"],
                "yaw": candidate["yaw"],
                "tag_ids": [candidate["tag_id"]],
                "reason": "none",
            }

        positions = [candidate["position"] for candidate in candidates]
        max_distance = 0.0
        for i, lhs in enumerate(positions):
            for rhs in positions[i + 1 :]:
                max_distance = max(max_distance, float(np.linalg.norm(lhs - rhs)))
        if max_distance > self._multi_tag_conflict_distance_m:
            best = min(candidates, key=lambda item: item["reprojection"])
            return {
                "position": best["position"],
                "yaw": best["yaw"],
                "tag_ids": [best["tag_id"]],
                "reason": "multi_tag_conflict_choose_lowest_reprojection",
            }

        weights = np.asarray([candidate["weight"] for candidate in candidates], dtype=np.float64)
        weights /= max(float(np.sum(weights)), 1e-12)
        position = np.zeros(3, dtype=np.float64)
        sin_sum = 0.0
        cos_sum = 0.0
        for weight, candidate in zip(weights, candidates, strict=False):
            position += weight * candidate["position"]
            sin_sum += weight * math.sin(candidate["yaw"])
            cos_sum += weight * math.cos(candidate["yaw"])
        yaw = math.atan2(sin_sum, cos_sum)
        return {
            "position": position,
            "yaw": yaw,
            "tag_ids": [candidate["tag_id"] for candidate in candidates],
            "reason": "multi_tag_weighted_fusion",
        }

    def _imu_callback(self, msg: Imu) -> None:
        stamp_sec = self._stamp_or_now(msg)
        self._imu = msg
        quat = normalize_quat_xyzw([msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w])
        roll, pitch, yaw = quat_to_rpy_xyzw(quat)
        self._roll_pitch = (roll, pitch)
        self._latest_yaw_rate = float(msg.angular_velocity.z)
        self._yaw_ekf.predict(stamp_sec, self._latest_yaw_rate)
        self._last_imu_time = stamp_sec
        if self._use_imu_yaw_measurement:
            accepted, distance = self._yaw_ekf.update(yaw, self._imu_yaw_covariance)
            if not accepted:
                self._last_reject_reason = "imu_yaw_mahalanobis_gate"
            self._last_yaw_mahalanobis = distance

    def _depth_callback(self, msg: PoseWithCovarianceStamped) -> None:
        stamp_sec = self._stamp_or_now(msg)
        pressure_sensor_z = self._water_surface_z_m + float(msg.pose.pose.position.z) * self._depth_sign
        depth = self._body_z_from_pressure_sensor_z(pressure_sensor_z)
        self._position_ekf.predict(stamp_sec)
        accepted, distance = self._position_ekf.update_indices(
            [2],
            [depth],
            np.diag([self._depth_covariance]),
            gate_mahalanobis=self._measurement_gate,
        )
        self._last_depth_mahalanobis = distance
        if accepted:
            self._depth = depth
            self._last_depth_time = stamp_sec
        else:
            self._last_reject_reason = "depth_mahalanobis_gate"

    def _body_z_from_pressure_sensor_z(self, sensor_z: float) -> float:
        roll, pitch = self._roll_pitch
        offset_world_z = self._pressure_sensor_offset_z_body * math.cos(roll) * math.cos(pitch)
        return float(sensor_z - offset_world_z)

    def _publish(self) -> None:
        now_sec = self._now_sec()
        self._position_ekf.predict(now_sec)
        self._yaw_ekf.predict(now_sec, self._latest_yaw_rate)

        vision_fresh = self._fresh(self._last_vision_time, self._vision_timeout_sec, now_sec)
        imu_fresh = self._fresh(self._last_imu_time, self._imu_timeout_sec, now_sec)
        depth_fresh = self._fresh(self._last_depth_time, self._depth_timeout_sec, now_sec)
        initialized = (
            self._last_vision_time is not None
            and self._last_depth_time is not None
            and self._last_imu_time is not None
        )
        vision_mode = self._apply_vision_dropout_policy(now_sec, vision_fresh)
        ready = initialized and vision_mode in {"fresh", "coast", "hold"}
        should_publish_navigation = initialized or self._publish_pose_before_ready
        self._log_state_transition(
            now_sec,
            ready=ready,
            initialized=initialized,
            vision_mode=vision_mode,
            vision_fresh=vision_fresh,
            imu_fresh=imu_fresh,
            depth_fresh=depth_fresh,
            publishing_navigation=should_publish_navigation,
        )

        roll, pitch = self._roll_pitch
        yaw = self._yaw_ekf.yaw if self._yaw_ekf.initialized else 0.0
        quat = rpy_to_quat_xyzw(roll, pitch, yaw)
        position = self._position_ekf.x[:3]
        velocity_world = self._position_ekf.x[3:6].copy()
        if not (vision_fresh or depth_fresh):
            velocity_world *= self._stale_velocity_decay
        if vision_mode == "hold":
            velocity_world[0:2] = 0.0
        velocity_body = inverse_rotate_vector(quat, velocity_world)

        stamp = self.get_clock().now().to_msg()
        self._publish_imu(stamp, quat, imu_fresh)
        self._publish_depth(stamp, depth_fresh)
        if should_publish_navigation:
            self._publish_pose(stamp, position, quat, vision_fresh, depth_fresh, imu_fresh, vision_mode)
            self._publish_dvl(stamp, velocity_body, vision_mode)
        self._publish_status(
            now_sec,
            ready=ready,
            initialized=initialized,
            vision_mode=vision_mode,
            vision_fresh=vision_fresh,
            imu_fresh=imu_fresh,
            depth_fresh=depth_fresh,
            velocity_world=velocity_world,
        )

    def _apply_vision_dropout_policy(self, now_sec: float, vision_fresh: bool) -> str:
        if self._last_vision_time is None:
            self._position_ekf.decay_velocity([3, 4], self._vision_hold_velocity_decay)
            self._position_ekf.cap_covariance([0, 1], self._max_position_covariance_xy)
            self._position_ekf.cap_covariance([3, 4], self._max_velocity_covariance_xy)
            return "uninitialized"

        age = now_sec - self._last_vision_time
        coast_timeout = max(self._vision_coast_timeout_sec, self._vision_timeout_sec)
        hold_timeout = max(self._vision_hold_timeout_sec, coast_timeout)
        if vision_fresh:
            self._position_ekf.cap_covariance([0, 1], self._max_position_covariance_xy)
            self._position_ekf.cap_covariance([3, 4], self._max_velocity_covariance_xy)
            return "fresh"
        if age <= coast_timeout:
            self._position_ekf.cap_covariance([0, 1], self._max_position_covariance_xy)
            self._position_ekf.cap_covariance([3, 4], self._max_velocity_covariance_xy)
            return "coast"

        if self._last_vision_position_xy is not None:
            self._position_ekf.x[0:2] = self._last_vision_position_xy
        self._position_ekf.decay_velocity([3, 4], self._vision_hold_velocity_decay)
        self._position_ekf.cap_covariance([0, 1], self._max_position_covariance_xy)
        self._position_ekf.cap_covariance([3, 4], self._max_velocity_covariance_xy)
        if age <= hold_timeout:
            return "hold"
        return "lost"

    def _log_state_transition(
        self,
        now_sec: float,
        *,
        ready: bool,
        initialized: bool,
        vision_mode: str,
        vision_fresh: bool,
        imu_fresh: bool,
        depth_fresh: bool,
        publishing_navigation: bool,
    ) -> None:
        state = (
            bool(ready),
            bool(initialized),
            str(vision_mode),
            bool(vision_fresh),
            bool(imu_fresh),
            bool(depth_fresh),
            bool(publishing_navigation),
        )
        if state == self._last_logged_state:
            return

        previous = self._last_logged_state
        self._last_logged_state = state

        def age_text(stamp_sec: float | None) -> str:
            if stamp_sec is None:
                return "none"
            return f"{max(0.0, now_sec - stamp_sec):.3f}s"

        transition = "initial" if previous is None else f"{previous[2]} -> {vision_mode}"
        self.get_logger().info(
            "[LOG] fusion state change: "
            f"transition={transition}, ready={ready}, initialized={initialized}, "
            f"vision_mode={vision_mode}, fresh=(vision={vision_fresh}, imu={imu_fresh}, depth={depth_fresh}), "
            f"ages=(vision={age_text(self._last_vision_time)}, imu={age_text(self._last_imu_time)}, "
            f"depth={age_text(self._last_depth_time)}), "
            f"publish_navigation={publishing_navigation}, used_tag_ids={self._last_used_tag_ids}, "
            f"vision_source={self._last_vision_source}, reject_reason={self._last_reject_reason}, "
            f"cov_pos_diag={[round(float(self._position_ekf.p[i, i]), 6) for i in range(3)]}, "
            f"cov_vel_diag={[round(float(self._position_ekf.p[i, i]), 6) for i in range(3, 6)]}"
        )

    def _publish_pose(
        self,
        stamp,
        position,
        quat,
        vision_fresh: bool,
        depth_fresh: bool,
        imu_fresh: bool,
        vision_mode: str,
    ) -> None:
        pose = PoseWithCovarianceStamped()
        pose.header.stamp = stamp
        pose.header.frame_id = self._world_frame_id
        pose.pose.pose.position.x = float(position[0])
        pose.pose.pose.position.y = float(position[1])
        pose.pose.pose.position.z = float(position[2])
        pose.pose.pose.orientation.x = float(quat[0])
        pose.pose.pose.orientation.y = float(quat[1])
        pose.pose.pose.orientation.z = float(quat[2])
        pose.pose.pose.orientation.w = float(quat[3])
        xy_multiplier = 1.0 if vision_mode == "fresh" else 10.0
        pose.pose.covariance[0] = float(min(self._position_ekf.p[0, 0] * xy_multiplier, self._max_position_covariance_xy))
        pose.pose.covariance[7] = float(min(self._position_ekf.p[1, 1] * xy_multiplier, self._max_position_covariance_xy))
        pose.pose.covariance[14] = float(self._position_ekf.p[2, 2]) if depth_fresh else float(self._position_ekf.p[2, 2] * 10.0)
        pose.pose.covariance[21] = 0.02 if imu_fresh else 2.0
        pose.pose.covariance[28] = 0.02 if imu_fresh else 2.0
        pose.pose.covariance[35] = float(self._yaw_ekf.p) if imu_fresh else float(self._yaw_ekf.p * 10.0)
        self._pose_pub.publish(pose)

    def _publish_imu(self, stamp, quat, imu_fresh: bool) -> None:
        imu = Imu()
        imu.header.stamp = stamp
        imu.header.frame_id = self._body_frame_id
        imu.orientation.x = float(quat[0])
        imu.orientation.y = float(quat[1])
        imu.orientation.z = float(quat[2])
        imu.orientation.w = float(quat[3])
        if self._imu is not None:
            imu.angular_velocity = self._imu.angular_velocity
            imu.linear_acceleration = self._imu.linear_acceleration
            imu.angular_velocity_covariance = self._imu.angular_velocity_covariance
            imu.linear_acceleration_covariance = self._imu.linear_acceleration_covariance
        imu.orientation_covariance[0] = 0.02 if imu_fresh else 2.0
        imu.orientation_covariance[4] = 0.02 if imu_fresh else 2.0
        imu.orientation_covariance[8] = float(self._yaw_ekf.p)
        self._imu_pub.publish(imu)

    def _publish_depth(self, stamp, depth_fresh: bool) -> None:
        depth = PoseWithCovarianceStamped()
        depth.header.stamp = stamp
        depth.header.frame_id = self._depth_frame_id
        depth.pose.pose.position.z = float(self._position_ekf.x[2])
        depth.pose.covariance[14] = self._depth_covariance if depth_fresh else self._depth_covariance * 100.0
        self._depth_pub.publish(depth)

    def _publish_dvl(self, stamp, velocity_body: np.ndarray, vision_mode: str) -> None:
        dvl = TwistWithCovarianceStamped()
        dvl.header.stamp = stamp
        dvl.header.frame_id = self._body_frame_id
        dvl.twist.twist.linear.x = float(velocity_body[0])
        dvl.twist.twist.linear.y = float(velocity_body[1])
        dvl.twist.twist.linear.z = float(velocity_body[2])
        xy_covariance = self._dvl_covariance if vision_mode in {"fresh", "coast"} else self._dvl_covariance * 100.0
        dvl.twist.covariance[0] = xy_covariance
        dvl.twist.covariance[7] = xy_covariance
        dvl.twist.covariance[14] = self._dvl_covariance
        self._dvl_pub.publish(dvl)

    def _publish_status(
        self,
        now_sec: float,
        *,
        ready: bool,
        initialized: bool = False,
        vision_mode: str = "uninitialized",
        vision_fresh: bool = False,
        imu_fresh: bool = False,
        depth_fresh: bool = False,
        velocity_world=None,
    ) -> None:
        payload = {
            "ready": ready,
            "initialized": initialized,
            "vision_mode": vision_mode,
            "coordinate_convention": "pool_world_z_up_bottom_origin_depth_corrected_to_body_origin",
            "water_surface_z_m": self._water_surface_z_m,
            "depth_sign": self._depth_sign,
            "pressure_sensor_offset_z_body": self._pressure_sensor_offset_z_body,
            "vision_fresh": vision_fresh,
            "vision_timeout_sec": self._vision_timeout_sec,
            "vision_coast_timeout_sec": self._vision_coast_timeout_sec,
            "vision_hold_timeout_sec": self._vision_hold_timeout_sec,
            "imu_fresh": imu_fresh,
            "depth_fresh": depth_fresh,
            "age_vision": None if self._last_vision_time is None else now_sec - self._last_vision_time,
            "age_imu": None if self._last_imu_time is None else now_sec - self._last_imu_time,
            "age_depth": None if self._last_depth_time is None else now_sec - self._last_depth_time,
            "used_tag_ids": self._last_used_tag_ids,
            "vision_source": self._last_vision_source,
            "reject_reason": self._last_reject_reason,
            "vision_updates": self._vision_update_count,
            "vision_rejects": self._vision_reject_count,
            "vision_mahalanobis": self._last_vision_mahalanobis,
            "depth_mahalanobis": self._last_depth_mahalanobis,
            "yaw_mahalanobis": self._last_yaw_mahalanobis,
            "position_covariance_diag": [float(self._position_ekf.p[i, i]) for i in range(3)],
            "velocity_covariance_diag": [float(self._position_ekf.p[i, i]) for i in range(3, 6)],
            "yaw_covariance": float(self._yaw_ekf.p),
            "velocity_world": None if velocity_world is None else [float(v) for v in velocity_world],
        }
        msg = String()
        msg.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = StateFusionNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
