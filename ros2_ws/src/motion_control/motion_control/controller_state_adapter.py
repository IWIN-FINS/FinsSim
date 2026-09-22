from __future__ import annotations

from copy import deepcopy
import json
from typing import Sequence

import numpy as np
import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import String

from .math_utils import (
    axial_basis_matrix,
    build_basis_matrix,
    reorder_vector,
    transform_axial_covariance,
    transform_axial_vector,
    transform_quat_basis,
)


def _vec3_from_position(position) -> np.ndarray:
    return np.array([position.x, position.y, position.z], dtype=np.float32)


def _vec3_from_vector(vector) -> np.ndarray:
    return np.array([vector.x, vector.y, vector.z], dtype=np.float32)


def _write_position(position, value: Sequence[float]) -> None:
    array = np.asarray(value, dtype=np.float32).reshape(3)
    position.x = float(array[0])
    position.y = float(array[1])
    position.z = float(array[2])


def _write_vector(vector, value: Sequence[float]) -> None:
    array = np.asarray(value, dtype=np.float32).reshape(3)
    vector.x = float(array[0])
    vector.y = float(array[1])
    vector.z = float(array[2])


def _transform_covariance(covariance: Sequence[float], basis_matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(covariance, dtype=np.float64).reshape(6, 6)
    transform = np.zeros((6, 6), dtype=np.float64)
    transform[:3, :3] = basis_matrix
    transform[3:, 3:] = axial_basis_matrix(basis_matrix)
    return (transform @ matrix @ transform.T).reshape(36).astype(float).tolist()


def _transform_covariance3(covariance: Sequence[float], basis_matrix: np.ndarray) -> list[float]:
    matrix = np.asarray(covariance, dtype=np.float64).reshape(3, 3)
    return (basis_matrix @ matrix @ basis_matrix.T).reshape(9).astype(float).tolist()


def _normalized_frame(frame_id: str | None) -> str:
    return (frame_id or "").strip().lower()


def _is_controller_world_frame(frame_id: str | None) -> bool:
    return _normalized_frame(frame_id) == "controller_world"


def _is_controller_body_frame(frame_id: str | None) -> bool:
    return _normalized_frame(frame_id) == "controller_body"


class ControllerStateAdapter(Node):
    """Convert real pool-world fused state into controller-native topics."""

    def __init__(self) -> None:
        super().__init__("controller_state_adapter")

        self._basis_matrix = build_basis_matrix(
            self.declare_parameter("ros_to_controller_basis_indices", [0, 2, 1]).value,
            self.declare_parameter("ros_to_controller_basis_signs", [1.0, 1.0, 1.0]).value,
        )
        self._water_surface_z_m = float(self.declare_parameter("water_surface_z_m", 0.98).value)
        self._controller_world_frame_id = str(
            self.declare_parameter("controller_world_frame_id", "controller_world").value
        )
        self._controller_body_frame_id = str(
            self.declare_parameter("controller_body_frame_id", "controller_body").value
        )

        input_pose_topic = str(self.declare_parameter("input_pose_topic", "/finsrov/pose").value)
        input_imu_topic = str(self.declare_parameter("input_imu_topic", "/finsrov/imu_link").value)
        input_depth_topic = str(self.declare_parameter("input_depth_topic", "/finsrov/depth_link").value)
        input_dvl_topic = str(self.declare_parameter("input_dvl_topic", "/finsrov/dvl_link").value)
        input_status_topic = str(self.declare_parameter("input_status_topic", "/finsrov/state/status").value)

        self._output_pose_topic = str(self.declare_parameter("output_pose_topic", "/finsrov/controller/pose").value)
        self._output_imu_topic = str(self.declare_parameter("output_imu_topic", "/finsrov/controller/imu").value)
        self._output_depth_topic = str(
            self.declare_parameter("output_depth_topic", "/finsrov/controller/depth").value
        )
        self._output_dvl_topic = str(self.declare_parameter("output_dvl_topic", "/finsrov/controller/dvl").value)
        self._output_status_topic = str(
            self.declare_parameter("output_status_topic", "/finsrov/controller/state/status").value
        )

        self._pose_pub = self.create_publisher(PoseWithCovarianceStamped, self._output_pose_topic, 10)
        self._imu_pub = self.create_publisher(Imu, self._output_imu_topic, 10)
        self._depth_pub = self.create_publisher(PoseWithCovarianceStamped, self._output_depth_topic, 10)
        self._dvl_pub = self.create_publisher(TwistWithCovarianceStamped, self._output_dvl_topic, 10)
        self._status_pub = self.create_publisher(String, self._output_status_topic, 10)

        self.create_subscription(PoseWithCovarianceStamped, input_pose_topic, self._pose_callback, 10)
        self.create_subscription(Imu, input_imu_topic, self._imu_callback, 10)
        self.create_subscription(PoseWithCovarianceStamped, input_depth_topic, self._depth_callback, 10)
        self.create_subscription(TwistWithCovarianceStamped, input_dvl_topic, self._dvl_callback, 10)
        self.create_subscription(String, input_status_topic, self._status_callback, 10)

        self.get_logger().info(
            "controller_state_adapter started: "
            f"pool_pose={input_pose_topic} -> {self._output_pose_topic}, "
            f"water_surface_z_m={self._water_surface_z_m:.3f}"
        )

    def pool_position_to_controller(self, position_pool: Sequence[float]) -> np.ndarray:
        position = reorder_vector(position_pool, self._basis_matrix)
        position[1] -= np.float32(self._water_surface_z_m)
        return position.astype(np.float32, copy=False)

    def _position_to_controller_world(self, position: Sequence[float], frame_id: str | None) -> np.ndarray:
        array = np.asarray(position, dtype=np.float32).reshape(3)
        if _is_controller_world_frame(frame_id):
            return array.astype(np.float32, copy=True)
        return self.pool_position_to_controller(array)

    def _vector_to_controller_body(self, value: Sequence[float], frame_id: str | None) -> np.ndarray:
        array = np.asarray(value, dtype=np.float32).reshape(3)
        if _is_controller_body_frame(frame_id):
            return array.astype(np.float32, copy=True)
        return reorder_vector(array, self._basis_matrix)

    def _quat_to_controller(self, quat_xyzw: Sequence[float], frame_id: str | None) -> np.ndarray:
        quat = np.asarray(quat_xyzw, dtype=np.float32).reshape(4)
        if _is_controller_world_frame(frame_id) or _is_controller_body_frame(frame_id):
            return quat.astype(np.float32, copy=True)
        return transform_quat_basis(quat, self._basis_matrix)

    def _pose_callback(self, msg: PoseWithCovarianceStamped) -> None:
        out = deepcopy(msg)
        out.header.frame_id = self._controller_world_frame_id
        position_controller = self._position_to_controller_world(_vec3_from_position(msg.pose.pose.position), msg.header.frame_id)
        _write_position(out.pose.pose.position, position_controller)
        quat_pool = [
            msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w,
        ]
        quat_controller = self._quat_to_controller(quat_pool, msg.header.frame_id)
        out.pose.pose.orientation.x = float(quat_controller[0])
        out.pose.pose.orientation.y = float(quat_controller[1])
        out.pose.pose.orientation.z = float(quat_controller[2])
        out.pose.pose.orientation.w = float(quat_controller[3])
        if _is_controller_world_frame(msg.header.frame_id):
            out.pose.covariance = list(msg.pose.covariance)
        else:
            out.pose.covariance = _transform_covariance(msg.pose.covariance, self._basis_matrix)
        self._pose_pub.publish(out)

    def _imu_callback(self, msg: Imu) -> None:
        out = deepcopy(msg)
        out.header.frame_id = self._controller_body_frame_id
        quat_pool = [msg.orientation.x, msg.orientation.y, msg.orientation.z, msg.orientation.w]
        quat_controller = self._quat_to_controller(quat_pool, msg.header.frame_id)
        out.orientation.x = float(quat_controller[0])
        out.orientation.y = float(quat_controller[1])
        out.orientation.z = float(quat_controller[2])
        out.orientation.w = float(quat_controller[3])
        angular_velocity = _vec3_from_vector(msg.angular_velocity)
        if _is_controller_body_frame(msg.header.frame_id):
            angular_velocity_controller = angular_velocity
        else:
            angular_velocity_controller = transform_axial_vector(angular_velocity, self._basis_matrix)
        _write_vector(out.angular_velocity, angular_velocity_controller)
        _write_vector(
            out.linear_acceleration,
            self._vector_to_controller_body(_vec3_from_vector(msg.linear_acceleration), msg.header.frame_id),
        )
        if _is_controller_body_frame(msg.header.frame_id):
            out.orientation_covariance = list(msg.orientation_covariance)
            out.angular_velocity_covariance = list(msg.angular_velocity_covariance)
            out.linear_acceleration_covariance = list(msg.linear_acceleration_covariance)
        else:
            out.orientation_covariance = transform_axial_covariance(msg.orientation_covariance, self._basis_matrix)
            out.angular_velocity_covariance = transform_axial_covariance(
                msg.angular_velocity_covariance,
                self._basis_matrix,
            )
            out.linear_acceleration_covariance = _transform_covariance3(
                msg.linear_acceleration_covariance,
                self._basis_matrix,
            )
        self._imu_pub.publish(out)

    def _depth_callback(self, msg: PoseWithCovarianceStamped) -> None:
        out = deepcopy(msg)
        out.header.frame_id = self._controller_world_frame_id
        position_controller = self._position_to_controller_world(_vec3_from_position(msg.pose.pose.position), msg.header.frame_id)
        _write_position(out.pose.pose.position, position_controller)
        if _is_controller_world_frame(msg.header.frame_id):
            out.pose.covariance = list(msg.pose.covariance)
        else:
            out.pose.covariance = _transform_covariance(msg.pose.covariance, self._basis_matrix)
        self._depth_pub.publish(out)

    def _dvl_callback(self, msg: TwistWithCovarianceStamped) -> None:
        out = deepcopy(msg)
        out.header.frame_id = self._controller_body_frame_id
        _write_vector(
            out.twist.twist.linear,
            self._vector_to_controller_body(_vec3_from_vector(msg.twist.twist.linear), msg.header.frame_id),
        )
        angular_velocity = _vec3_from_vector(msg.twist.twist.angular)
        if _is_controller_body_frame(msg.header.frame_id):
            angular_velocity_controller = angular_velocity
        else:
            angular_velocity_controller = transform_axial_vector(angular_velocity, self._basis_matrix)
        _write_vector(out.twist.twist.angular, angular_velocity_controller)
        if _is_controller_body_frame(msg.header.frame_id):
            out.twist.covariance = list(msg.twist.covariance)
        else:
            out.twist.covariance = _transform_covariance(msg.twist.covariance, self._basis_matrix)
        self._dvl_pub.publish(out)

    def _status_callback(self, msg: String) -> None:
        out = String()
        try:
            payload = json.loads(msg.data)
            if not isinstance(payload, dict):
                raise ValueError("status payload is not an object")
        except Exception:
            payload = {"raw_status": msg.data}
        payload["coordinate_convention"] = "controller_x_forward_y_up_z_left"
        payload["state_source"] = "raw_fusion_controller_adapter"
        payload["water_surface_z_m"] = self._water_surface_z_m
        payload["controller_pose_topic"] = self._output_pose_topic
        payload["controller_imu_topic"] = self._output_imu_topic
        payload["controller_depth_topic"] = self._output_depth_topic
        payload["controller_dvl_topic"] = self._output_dvl_topic
        out.data = json.dumps(payload, separators=(",", ":"))
        self._status_pub.publish(out)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerStateAdapter()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
