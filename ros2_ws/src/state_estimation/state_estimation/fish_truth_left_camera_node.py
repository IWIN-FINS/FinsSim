from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import rclpy
import yaml
from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from msgs.msg import AprilTagDetection3DArray
from geometry_msgs.msg import PoseWithCovarianceStamped
from rclpy._rclpy_pybind11 import RCLError
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node

from .transforms import matrix_to_quat_xyzw, pose_matrix, quat_xyzw_to_rotation_matrix, rpy_to_quat_xyzw


def _as_vec3(value) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(3)
    return arr.astype(np.float64, copy=False)


def _homogeneous(point_xyz) -> np.ndarray:
    point = _as_vec3(point_xyz)
    return np.array([point[0], point[1], point[2], 1.0], dtype=np.float64)


def _matrix_from_rpy_deg(rpy_deg) -> np.ndarray:
    roll, pitch, yaw = [math.radians(float(v)) for v in rpy_deg]
    quat = rpy_to_quat_xyzw(roll, pitch, yaw)
    return quat_xyzw_to_rotation_matrix(quat)


def _standard_optical_rotation_body_to_left() -> np.ndarray:
    # body frame:   x forward, y left,  z up   (ROS FLU)
    # optical frame x right,   y down,  z forward
    return np.array(
        [
            [0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )


@dataclass
class _StampedTransform:
    stamp_sec: float
    transform: np.ndarray


@dataclass
class _StampedTag:
    stamp_sec: float
    tag_id: int
    world_tag: np.ndarray
    reprojection_error_px: float
    decision_margin: float


class FishTruthLeftCameraNode(Node):
    def __init__(self) -> None:
        super().__init__("fish_truth_left_camera")

        self._pnp_topic = str(
            self.declare_parameter("pnp_topic", "/finsrov/vision/tag_poses_3d_camera").value
        )
        self._body_pose_topic = str(
            self.declare_parameter("body_pose_topic", "/finsrov/pose").value
        )
        self._output_topic = str(
            self.declare_parameter("output_topic", "/finsrov/vision/fish_truth_left_camera").value
        )
        self._extrinsics_file = str(
            self.declare_parameter(
                "extrinsics_file", "config/state_fusion_extrinsics.yaml"
            ).value
        )
        self._fish_tag_id = int(self.declare_parameter("fish_tag_id", 17).value)
        self._body_tag_ids = [
            int(v) for v in self.declare_parameter("body_tag_ids", [15, 16]).value
        ]
        self._left_camera_frame_id = str(
            self.declare_parameter("left_camera_frame_id", "finsrov_stereo_left_optical").value
        )
        self._world_frame_id = str(self.declare_parameter("world_frame_id", "pool_world").value)
        self._publish_rate_hz = float(self.declare_parameter("publish_rate_hz", 20.0).value)
        self._max_pose_match_dt_sec = float(
            self.declare_parameter("max_pose_match_dt_sec", 0.20).value
        )
        self._max_tag_age_sec = float(self.declare_parameter("max_tag_age_sec", 0.50).value)
        self._max_body_pose_age_sec = float(
            self.declare_parameter("max_body_pose_age_sec", 0.50).value
        )
        self._log_period_sec = float(self.declare_parameter("log_period_sec", 0.25).value)
        self._fallback_use_pnp_body_pose = bool(
            self.declare_parameter("fallback_use_pnp_body_pose", True).value
        )
        self._target_point_mode = str(
            self.declare_parameter("target_point_mode", "world_offset").value
        )
        self._fish_offset_tag_xyz = _as_vec3(
            self.declare_parameter("fish_offset_tag_xyz", [0.0, 0.0, 0.0]).value
        )
        self._fish_offset_world_xyz = _as_vec3(
            self.declare_parameter("fish_offset_world_xyz", [0.0, 0.0, -0.15]).value
        )
        self._left_camera_translation_xyz = _as_vec3(
            self.declare_parameter("left_camera_translation_xyz", [0.30, 0.0, 0.0]).value
        )
        self._left_camera_rotation_mode = str(
            self.declare_parameter(
                "left_camera_rotation_mode", "standard_optical_from_body"
            ).value
        )
        self._left_camera_rotation_rpy_deg = _as_vec3(
            self.declare_parameter("left_camera_rotation_rpy_deg", [0.0, 0.0, 0.0]).value
        )
        self._left_camera_rotation_xyzw = np.asarray(
            self.declare_parameter("left_camera_rotation_xyzw", [0.0, 0.0, 0.0, 1.0]).value,
            dtype=np.float64,
        ).reshape(4)

        self._t_world_camera = self._load_t_world_camera(self._extrinsics_file)
        self._t_body_tag_inverse = self._load_t_body_tag_inverse(self._extrinsics_file)
        self._t_body_left = self._build_t_body_left()

        self._body_pose_buffer: deque[_StampedTransform] = deque(maxlen=512)
        self._latest_tag: _StampedTag | None = None
        self._latest_pnp_body_pose: _StampedTransform | None = None
        self._last_publish_stamp_sec: float | None = None
        self._last_log_sec = 0.0

        self._pub = self.create_publisher(PoseWithCovarianceStamped, self._output_topic, 10)
        self.create_subscription(AprilTagDetection3DArray, self._pnp_topic, self._pnp_callback, 10)
        self.create_subscription(
            PoseWithCovarianceStamped, self._body_pose_topic, self._body_pose_callback, 10
        )

        period = 1.0 / max(self._publish_rate_hz, 1e-3)
        self.create_timer(period, self._timer_publish)

        self.get_logger().info(
            "fish_truth_left_camera started: "
            f"pnp_topic={self._pnp_topic}, body_pose_topic={self._body_pose_topic}, "
            f"output_topic={self._output_topic}, fish_tag_id={self._fish_tag_id}, "
            f"body_tag_ids={self._body_tag_ids}, "
            f"target_point_mode={self._target_point_mode}, "
            f"left_camera_frame_id={self._left_camera_frame_id}"
        )

    @staticmethod
    def _msg_stamp_sec(msg) -> float:
        stamp = msg.header.stamp
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _now_sec(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

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

    def _load_t_world_camera(self, path_value: str) -> np.ndarray:
        path = self._resolve_path(path_value)
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        node = data.get("T_world_camera", {})
        translation = node.get("translation_xyz", [0.0, 0.0, 0.0])
        if "rotation_xyzw" in node:
            rotation = node["rotation_xyzw"]
        else:
            rotation = rpy_to_quat_xyzw(
                *[math.radians(float(v)) for v in node.get("rotation_rpy_deg", [0.0, 0.0, 0.0])]
            )
        return pose_matrix(translation, rotation)

    def _load_t_body_tag_inverse(self, path_value: str) -> dict[int, np.ndarray]:
        path = self._resolve_path(path_value)
        with path.open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream) or {}
        body_tag_map = data.get("T_body_tag", {}) or {}
        output: dict[int, np.ndarray] = {}
        for tag_id_text, tag_node in body_tag_map.items():
            tag_id = int(tag_id_text)
            translation = tag_node.get("translation_xyz", [0.0, 0.0, 0.0])
            if "rotation_xyzw" in tag_node:
                quat = tag_node["rotation_xyzw"]
            else:
                quat = rpy_to_quat_xyzw(
                    *[
                        math.radians(float(v))
                        for v in tag_node.get("rotation_rpy_deg", [0.0, 0.0, 0.0])
                    ]
                )
            t_body_tag = pose_matrix(translation, quat)
            output[tag_id] = np.linalg.inv(t_body_tag)
        return output

    def _build_t_body_left(self) -> np.ndarray:
        mode = self._left_camera_rotation_mode.strip().lower()
        if mode == "standard_optical_from_body":
            rotation_matrix = _standard_optical_rotation_body_to_left()
            quat = matrix_to_quat_xyzw(rotation_matrix)
        elif mode == "identity":
            quat = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
        elif mode == "explicit_rpy_deg":
            quat = rpy_to_quat_xyzw(
                *[math.radians(float(v)) for v in self._left_camera_rotation_rpy_deg]
            )
        elif mode == "explicit_xyzw":
            quat = self._left_camera_rotation_xyzw
        else:
            raise ValueError(
                "unsupported left_camera_rotation_mode="
                f"{self._left_camera_rotation_mode!r}; expected one of "
                "'standard_optical_from_body', 'identity', "
                "'explicit_rpy_deg', 'explicit_xyzw'"
            )
        return pose_matrix(self._left_camera_translation_xyz, quat)

    def _body_pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        translation = [
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z,
        ]
        quat = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        stamp_sec = self._msg_stamp_sec(msg)
        if stamp_sec <= 0.0:
            stamp_sec = self._now_sec()
        self._body_pose_buffer.append(
            _StampedTransform(stamp_sec=stamp_sec, transform=pose_matrix(translation, quat))
        )

    def _pnp_callback(self, msg: AprilTagDetection3DArray) -> None:
        stamp_sec = self._msg_stamp_sec(msg)
        if stamp_sec <= 0.0:
            stamp_sec = self._now_sec()

        best = None
        body_candidates: list[tuple[float, np.ndarray, int]] = []
        for det in msg.detections:
            tag_id = int(det.tag_id)
            pose = det.pose.pose
            t_camera_tag = pose_matrix(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                ],
                [
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ],
            )
            if tag_id in self._t_body_tag_inverse and tag_id in self._body_tag_ids:
                t_world_body = self._t_world_camera @ t_camera_tag @ self._t_body_tag_inverse[tag_id]
                body_candidates.append((float(det.reprojection_error_px), t_world_body, tag_id))
            if int(det.tag_id) != self._fish_tag_id:
                continue
            best = det

        if body_candidates:
            body_candidates.sort(key=lambda item: item[0])
            self._latest_pnp_body_pose = _StampedTransform(
                stamp_sec=stamp_sec,
                transform=body_candidates[0][1],
            )

        if best is None:
            return

        t_camera_tag = pose_matrix(
            [best.pose.pose.position.x, best.pose.pose.position.y, best.pose.pose.position.z],
            [
                best.pose.pose.orientation.x,
                best.pose.pose.orientation.y,
                best.pose.pose.orientation.z,
                best.pose.pose.orientation.w,
            ],
        )
        t_world_tag = self._t_world_camera @ t_camera_tag
        self._latest_tag = _StampedTag(
            stamp_sec=stamp_sec,
            tag_id=int(best.tag_id),
            world_tag=t_world_tag,
            reprojection_error_px=float(best.reprojection_error_px),
            decision_margin=float(best.decision_margin),
        )

    def _nearest_body_pose(self, stamp_sec: float) -> _StampedTransform | None:
        best = None
        best_dt = None
        for item in self._body_pose_buffer:
            dt = abs(item.stamp_sec - stamp_sec)
            if best is None or dt < best_dt:
                best = item
                best_dt = dt
        if best is None:
            return None
        if best_dt is None or best_dt > self._max_pose_match_dt_sec:
            return None
        if (self._now_sec() - best.stamp_sec) > self._max_body_pose_age_sec:
            return None
        return best

    def _fallback_body_pose_from_pnp(self, tag_stamp_sec: float) -> _StampedTransform | None:
        if not self._fallback_use_pnp_body_pose:
            return None
        item = self._latest_pnp_body_pose
        if item is None:
            return None
        if abs(item.stamp_sec - tag_stamp_sec) > self._max_pose_match_dt_sec:
            return None
        if (self._now_sec() - item.stamp_sec) > self._max_body_pose_age_sec:
            return None
        return item

    def _fish_world_point(self, tagged: _StampedTag) -> np.ndarray:
        mode = self._target_point_mode.strip().lower()
        tag_origin = tagged.world_tag[:3, 3]
        if mode == "tag_origin":
            return tag_origin.copy()
        if mode == "tag_frame_offset":
            return (tagged.world_tag @ _homogeneous(self._fish_offset_tag_xyz))[:3]
        if mode == "world_offset":
            return tag_origin + self._fish_offset_world_xyz
        raise ValueError(
            "unsupported target_point_mode="
            f"{self._target_point_mode!r}; expected one of "
            "'tag_origin', 'tag_frame_offset', 'world_offset'"
        )

    def _timer_publish(self) -> None:
        tagged = self._latest_tag
        if tagged is None:
            return
        now_sec = self._now_sec()
        if (now_sec - tagged.stamp_sec) > self._max_tag_age_sec:
            return

        body_pose = self._nearest_body_pose(tagged.stamp_sec)
        body_pose_source = "fused_pose"
        if body_pose is None:
            body_pose = self._fallback_body_pose_from_pnp(tagged.stamp_sec)
            body_pose_source = "pnp_body_fallback"
        if body_pose is None:
            if (now_sec - self._last_log_sec) >= self._log_period_sec:
                self.get_logger().warn(
                    "fish_truth_left_camera waiting for body pose: "
                    "no fresh /finsrov/pose match and no usable PnP body tag fallback"
                )
                self._last_log_sec = now_sec
            return

        fish_world = self._fish_world_point(tagged)
        t_world_left = body_pose.transform @ self._t_body_left
        fish_left = (np.linalg.inv(t_world_left) @ _homogeneous(fish_world))[:3]

        msg = PoseWithCovarianceStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._left_camera_frame_id
        msg.pose.pose.position.x = float(fish_left[0])
        msg.pose.pose.position.y = float(fish_left[1])
        msg.pose.pose.position.z = float(fish_left[2])
        msg.pose.pose.orientation.w = 1.0
        msg.pose.covariance[0] = 0.0025
        msg.pose.covariance[7] = 0.0025
        msg.pose.covariance[14] = 0.0025
        msg.pose.covariance[21] = 1e6
        msg.pose.covariance[28] = 1e6
        msg.pose.covariance[35] = 1e6
        self._pub.publish(msg)

        self._last_publish_stamp_sec = tagged.stamp_sec
        if (now_sec - self._last_log_sec) >= self._log_period_sec:
            pose_dt_ms = (tagged.stamp_sec - body_pose.stamp_sec) * 1000.0
            self.get_logger().info(
                "fish_truth_left_camera | "
                f"tag={tagged.tag_id} | "
                f"X={fish_left[0]:+.3f} m  Y={fish_left[1]:+.3f} m  Z={fish_left[2]:+.3f} m | "
                f"body_src={body_pose_source} | "
                f"tag_age={1000.0 * (now_sec - tagged.stamp_sec):.1f} ms | "
                f"pose_dt={pose_dt_ms:+.1f} ms | "
                f"reproj={tagged.reprojection_error_px:.2f} px"
            )
            self._last_log_sec = now_sec


def main(args=None) -> None:
    rclpy.init(args=args)
    node = FishTruthLeftCameraNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException, RCLError):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
