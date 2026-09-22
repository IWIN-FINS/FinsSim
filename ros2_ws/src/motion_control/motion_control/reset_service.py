from __future__ import annotations

import time
from pathlib import Path

import rclpy
import yaml
from ament_index_python.packages import get_package_share_directory
from rclpy.executors import ExternalShutdownException
from rclpy.node import Node
from std_msgs.msg import Empty, Float32MultiArray
from std_srvs.srv import Trigger


def _load_installed_default_params() -> dict:
    try:
        config_path = (
            Path(get_package_share_directory("motion_control"))
            / "config"
            / "controller.yaml"
        )
    except Exception:
        return {}

    if not config_path.is_file():
        return {}

    try:
        data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}

    params = data.get("motion_controller", {}).get("ros__parameters", {})
    return params if isinstance(params, dict) else {}


def _string_param(node: Node, name: str, default: str) -> str:
    return str(node.declare_parameter(name, default).value).strip()


class ResetServiceNode(Node):
    def __init__(self) -> None:
        super().__init__("reset_service")
        config_defaults = _load_installed_default_params()

        vehicle_name = _string_param(self, "vehicle_name", str(config_defaults.get("vehicle_name", "FinsROV")))
        self._reset_topic = _string_param(
            self,
            "reset_topic",
            str(config_defaults.get("reset_topic", f"/{vehicle_name}/reset")),
        )
        default_thruster_topic = str(
            config_defaults.get("thruster_topic", config_defaults.get("pwm_topic", "/finsrov/thrusters_out"))
        )
        legacy_pwm_topic = _string_param(
            self,
            "pwm_topic",
            default_thruster_topic,
        )
        self._thruster_topic = _string_param(self, "thruster_topic", "") or legacy_pwm_topic or default_thruster_topic
        self._cancel_topic = _string_param(
            self,
            "command_cancel_topic",
            str(config_defaults.get("command_cancel_topic", "/motion_controller/command/cancel")),
        )
        self._reset_service_name = _string_param(
            self,
            "reset_service_name",
            str(config_defaults.get("reset_service_name", "/motion_controller/reset_vehicle")),
        )
        self._publish_count = int(self.declare_parameter("reset_publish_count", 3).value)
        self._publish_interval_sec = float(self.declare_parameter("reset_publish_interval_sec", 0.1).value)

        self._reset_pub = self.create_publisher(Float32MultiArray, self._reset_topic, 10)
        self._thruster_pub = self.create_publisher(Float32MultiArray, self._thruster_topic, 10)
        self._cancel_pub = self.create_publisher(Empty, self._cancel_topic, 10)
        self.create_service(Trigger, self._reset_service_name, self._handle_reset)

        self.get_logger().info(
            f"reset service started: service={self._reset_service_name}, reset_topic={self._reset_topic}, "
            f"thruster_topic={self._thruster_topic}, cancel_topic={self._cancel_topic}"
        )

    def _handle_reset(self, _request: Trigger.Request, response: Trigger.Response) -> Trigger.Response:
        reset_msg = Float32MultiArray()
        reset_msg.data = [1.0]
        zero_pwm_msg = Float32MultiArray()
        zero_pwm_msg.data = [0.0] * 8
        cancel_msg = Empty()

        publish_count = max(self._publish_count, 1)
        publish_interval_sec = max(self._publish_interval_sec, 0.0)

        for _ in range(publish_count):
            self._cancel_pub.publish(cancel_msg)
            self._thruster_pub.publish(zero_pwm_msg)
            self._reset_pub.publish(reset_msg)
            rclpy.spin_once(self, timeout_sec=0.01)
            if publish_interval_sec > 0.0:
                time.sleep(publish_interval_sec)

        response.success = True
        response.message = (
            f"published reset to `{self._reset_topic}`, zero thrusters to `{self._thruster_topic}`, "
            f"and cancel to `{self._cancel_topic}`"
        )
        self.get_logger().info(response.message)
        return response


def main(args: list[str] | None = None) -> None:
    rclpy.init(args=args)
    node = ResetServiceNode()
    try:
        rclpy.spin(node)
    except (KeyboardInterrupt, ExternalShutdownException):
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
