from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu

from .math_utils import build_basis_matrix, inverse_rotate_vector, reorder_vector, rotate_vector


@dataclass
class VehicleState:
    position_world: np.ndarray
    position_ros: np.ndarray
    orientation_world_body: np.ndarray
    linear_velocity_body: np.ndarray
    linear_velocity_world: np.ndarray
    linear_acceleration_body: np.ndarray
    linear_acceleration_world: np.ndarray
    angular_velocity_body_xyz: np.ndarray
    angular_velocity_body_ypr: np.ndarray
    angular_velocity_world: np.ndarray
    stamp_sec: float


class VehicleStateEstimator:
    """Track controller-native vehicle state for controller inference."""

    def __init__(
        self,
        node: Node,
        *,
        odom_topic: str,
        pose_topic: str,
        imu_topic: str,
        depth_topic: str,
        dvl_topic: str,
        basis_indices: Sequence[int],
        basis_signs: Sequence[float],
        position_offset_controller: Sequence[float] | None = None,
    ) -> None:
        self._node = node
        self._basis_matrix = build_basis_matrix(indices=basis_indices, signs=basis_signs)
        self._position_offset_controller = (
            np.zeros(3, dtype=np.float32)
            if position_offset_controller is None
            else np.asarray(position_offset_controller, dtype=np.float32).reshape(3)
        )
        self._odom_enabled = bool(odom_topic)
        self._pose_enabled = bool(pose_topic)

        self._position_world = np.zeros(3, dtype=np.float32)
        self._position_ros = np.zeros(3, dtype=np.float32)
        self._orientation_world_body = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        self._linear_velocity_body = np.zeros(3, dtype=np.float32)
        self._linear_acceleration_body = np.zeros(3, dtype=np.float32)
        self._angular_velocity_body_xyz = np.zeros(3, dtype=np.float32)

        self._has_pose = False
        self._has_imu = False
        self._has_depth = False
        self._has_dvl = False
        self._has_odom = False
        self._last_update_sec: Optional[float] = None
        self._last_state_sec: float = 0.0
        self._last_pose_sec: Optional[float] = None
        self._last_imu_sec: Optional[float] = None
        self._last_depth_sec: Optional[float] = None
        self._last_dvl_sec: Optional[float] = None
        self._last_odom_sec: Optional[float] = None

        if odom_topic:
            node.create_subscription(Odometry, odom_topic, self._odom_callback, 10)
        if pose_topic:
            node.create_subscription(PoseWithCovarianceStamped, pose_topic, self._pose_callback, 10)
        if imu_topic:
            node.create_subscription(Imu, imu_topic, self._imu_callback, 10)
        if depth_topic:
            node.create_subscription(PoseWithCovarianceStamped, depth_topic, self._depth_callback, 10)
        if dvl_topic:
            node.create_subscription(TwistWithCovarianceStamped, dvl_topic, self._dvl_callback, 10)

    @property
    def basis_matrix(self) -> np.ndarray:
        return self._basis_matrix

    @property
    def position_offset_controller(self) -> np.ndarray:
        return self._position_offset_controller

    def ros_position_to_controller(self, position_ros: Sequence[float]) -> np.ndarray:
        return reorder_vector(position_ros, self._basis_matrix) + self._position_offset_controller

    @staticmethod
    def _msg_time_sec(stamp) -> float:
        return float(stamp.sec) + float(stamp.nanosec) * 1e-9

    def _now_sec(self) -> float:
        return float(self._node.get_clock().now().nanoseconds) * 1e-9

    def _odom_callback(self, msg: Odometry) -> None:
        received_sec = self._now_sec()
        position_ros = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        quat_ros = np.array(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        linear_body_ros = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        angular_body_ros = np.array(
            [
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            ],
            dtype=np.float32,
        )

        self._position_world = position_ros.astype(np.float32, copy=True)
        self._position_ros = position_ros.astype(np.float32, copy=True)
        self._orientation_world_body = quat_ros.astype(np.float32, copy=True)
        self._linear_velocity_body = linear_body_ros.astype(np.float32, copy=True)
        self._linear_acceleration_body = np.zeros(3, dtype=np.float32)
        self._angular_velocity_body_xyz = angular_body_ros.astype(np.float32, copy=True)
        self._has_odom = True
        self._last_state_sec = self._msg_time_sec(msg.header.stamp)
        self._last_odom_sec = received_sec
        self._last_update_sec = received_sec

    def _pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        received_sec = self._now_sec()
        position_ros = np.array(
            [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg.pose.pose.position.z,
            ],
            dtype=np.float32,
        )
        quat_ros = np.array(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ],
            dtype=np.float32,
        )
        self._position_world = position_ros.astype(np.float32, copy=True)
        self._position_ros = position_ros.astype(np.float32, copy=True)
        self._orientation_world_body = quat_ros.astype(np.float32, copy=True)
        self._has_pose = True
        self._last_state_sec = self._msg_time_sec(msg.header.stamp)
        self._last_pose_sec = received_sec
        self._last_update_sec = received_sec

    def _imu_callback(self, msg: Imu) -> None:
        received_sec = self._now_sec()
        quat_ros = np.array(
            [
                msg.orientation.x,
                msg.orientation.y,
                msg.orientation.z,
                msg.orientation.w,
            ],
            dtype=np.float32,
        )
        angular_body_ros = np.array(
            [
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            ],
            dtype=np.float32,
        )
        acceleration_body_ros = np.array(
            [
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z,
            ],
            dtype=np.float32,
        )
        self._orientation_world_body = quat_ros.astype(np.float32, copy=True)
        self._angular_velocity_body_xyz = angular_body_ros.astype(np.float32, copy=True)
        self._linear_acceleration_body = acceleration_body_ros.astype(np.float32, copy=True)
        self._has_imu = True
        self._last_state_sec = self._msg_time_sec(msg.header.stamp)
        self._last_imu_sec = received_sec

    def _depth_callback(self, msg: PoseWithCovarianceStamped) -> None:
        received_sec = self._now_sec()
        if not self._pose_enabled:
            self._position_world[1] = float(msg.pose.pose.position.y)
        self._has_depth = True
        self._last_state_sec = self._msg_time_sec(msg.header.stamp)
        self._last_depth_sec = received_sec

    def _dvl_callback(self, msg: TwistWithCovarianceStamped) -> None:
        received_sec = self._now_sec()
        linear_body_ros = np.array(
            [
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
            ],
            dtype=np.float32,
        )
        self._linear_velocity_body = linear_body_ros.astype(np.float32, copy=True)
        self._has_dvl = True
        self._last_state_sec = self._msg_time_sec(msg.header.stamp)
        self._last_dvl_sec = received_sec

    @staticmethod
    def _fresh(stamp_sec: Optional[float], now_sec: float, timeout_sec: float) -> bool:
        if stamp_sec is None:
            return False
        if timeout_sec <= 0.0:
            return True
        return now_sec - stamp_sec <= timeout_sec

    def is_fresh(self, now_sec: float, timeout_sec: float) -> bool:
        if self._odom_enabled:
            return self._has_odom and self._fresh(self._last_odom_sec, now_sec, timeout_sec)
        if self._pose_enabled:
            return (
                self._has_pose
                and self._has_imu
                and self._has_dvl
                and self._fresh(self._last_pose_sec, now_sec, timeout_sec)
                and self._fresh(self._last_imu_sec, now_sec, timeout_sec)
                and self._fresh(self._last_dvl_sec, now_sec, timeout_sec)
            )
        return (
            self._has_imu
            and self._has_dvl
            and self._fresh(self._last_imu_sec, now_sec, timeout_sec)
            and self._fresh(self._last_dvl_sec, now_sec, timeout_sec)
            and (not self._has_depth or self._fresh(self._last_depth_sec, now_sec, timeout_sec))
        )

    def advance(self, now_sec: float) -> None:
        if self._odom_enabled and self._has_odom:
            return
        if self._pose_enabled and self._has_pose:
            return
        if not self._has_imu or not self._has_dvl:
            return
        if self._last_update_sec is None:
            self._last_update_sec = now_sec
            return

        dt = max(0.0, float(now_sec) - float(self._last_update_sec))
        self._last_update_sec = float(now_sec)
        if dt <= 0.0:
            return

        linear_world = rotate_vector(self._orientation_world_body, self._linear_velocity_body)
        if self._has_depth:
            linear_world[1] = 0.0
        self._position_world = self._position_world + linear_world * np.float32(dt)

    def snapshot(self, now_sec: float) -> Optional[VehicleState]:
        if self._odom_enabled and not self._has_odom:
            return None
        if self._pose_enabled and not self._has_pose:
            return None
        if not self._has_odom and not self._has_pose and not self._has_imu:
            return None

        linear_body = self._linear_velocity_body.astype(np.float32, copy=True)
        linear_world = rotate_vector(self._orientation_world_body, linear_body)
        linear_acceleration_body = self._linear_acceleration_body.astype(np.float32, copy=True)
        linear_acceleration_world = rotate_vector(self._orientation_world_body, linear_acceleration_body)
        angular_body_xyz = self._angular_velocity_body_xyz.astype(np.float32, copy=True)
        angular_body_ypr = np.array(
            [
                angular_body_xyz[1],
                angular_body_xyz[2],
                angular_body_xyz[0],
            ],
            dtype=np.float32,
        )
        angular_world = rotate_vector(self._orientation_world_body, angular_body_xyz)
        stamp_sec = self._last_state_sec if self._last_state_sec > 0.0 else float(now_sec)

        return VehicleState(
            position_world=self._position_world.astype(np.float32, copy=True),
            position_ros=self._position_ros.astype(np.float32, copy=True),
            orientation_world_body=self._orientation_world_body.astype(np.float32, copy=True),
            linear_velocity_body=linear_body,
            linear_velocity_world=linear_world.astype(np.float32, copy=True),
            linear_acceleration_body=linear_acceleration_body,
            linear_acceleration_world=linear_acceleration_world.astype(np.float32, copy=True),
            angular_velocity_body_xyz=angular_body_xyz,
            angular_velocity_body_ypr=angular_body_ypr,
            angular_velocity_world=angular_world.astype(np.float32, copy=True),
            stamp_sec=stamp_sec,
        )

    def world_error_to_body(self, error_world_xyz: Sequence[float]) -> np.ndarray:
        return inverse_rotate_vector(self._orientation_world_body, error_world_xyz)
