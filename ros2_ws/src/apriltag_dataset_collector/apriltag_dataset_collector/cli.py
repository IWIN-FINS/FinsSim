"""One-shot CLI client for the native AprilTag dataset capture interface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from std_msgs.msg import String
import yaml

from .capture_protocol import TRUTH_FIELDS, build_capture_request, normalize_dataset_root, request_json


def _default_config() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("apriltag_dataset_collector")) / "config" / "apriltag_dataset_collector.yaml"
    except Exception:
        return Path(__file__).resolve().parents[1] / "config" / "apriltag_dataset_collector.yaml"


def _load_config(path: Path) -> dict:
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = loaded.get("collector", {}).get("ros__parameters", {}) if isinstance(loaded, dict) else {}
    if not isinstance(params, dict):
        raise ValueError(f"invalid collector config: {path}")
    return params


class CaptureCli(Node):
    def __init__(self, config: dict, request: dict) -> None:
        super().__init__("apriltag_dataset_capture")
        self._request_id = str(request["request_id"])
        self._result: dict | None = None
        self._request_pub = self.create_publisher(String, str(config["dataset_capture_request_topic"]), 10)
        self.create_subscription(String, str(config["dataset_capture_result_topic"]), self._on_result, 10)
        self._detector_node = str(config["detector_node"])

    def _on_result(self, message: String) -> None:
        try:
            parsed = json.loads(message.data)
        except json.JSONDecodeError:
            return
        if parsed.get("request_id") == self._request_id:
            self._result = parsed

    def set_dataset_root(self, root: str, timeout_sec: float) -> None:
        client = self.create_client(SetParameters, f"{self._detector_node}/set_parameters")
        if not client.wait_for_service(timeout_sec=timeout_sec):
            raise RuntimeError(f"dataset detector parameter service unavailable: {self._detector_node}")
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="dataset_root",
                value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=normalize_dataset_root(root)),
            )
        ]
        future = client.call_async(request)
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_sec)
        response = future.result()
        if response is None or not response.results or not response.results[0].successful:
            reason = "unknown"
            if response is not None and response.results:
                reason = response.results[0].reason
            raise RuntimeError(f"failed to set detector dataset_root: {reason}")

    def publish_when_connected(self, payload: str, timeout_sec: float) -> None:
        deadline = self.get_clock().now().nanoseconds + int(timeout_sec * 1e9)
        while self._request_pub.get_subscription_count() == 0 and self.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(self, timeout_sec=0.1)
        if self._request_pub.get_subscription_count() == 0:
            raise RuntimeError("no native AprilTag detector subscribes to the capture request topic")
        message = String()
        message.data = payload
        self._request_pub.publish(message)


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Request one raw-frame AprilTag PnP dataset capture.")
    parser.add_argument("--config", type=Path, default=_default_config())
    parser.add_argument("--session", required=True)
    parser.add_argument("--operator", default="")
    parser.add_argument("--tag-id", type=int, default=None)
    parser.add_argument("--note", default="")
    parser.add_argument("--dataset-root", default=None, help="Runtime-only override for native detector dataset_root")
    parser.add_argument("--timeout", type=float, default=12.0)
    for field in TRUTH_FIELDS:
        parser.add_argument("--" + field.replace("_", "-"), type=float, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = _load_config(args.config)
    truth = {field: getattr(args, field) for field in TRUTH_FIELDS}
    request = build_capture_request(
        session_id=args.session, operator_name=args.operator, tag_id=args.tag_id, truth=truth, note=args.note
    )
    rclpy.init()
    node = CaptureCli(config, request)
    try:
        node.set_dataset_root(args.dataset_root or str(config["dataset_root"]), args.timeout)
        node.publish_when_connected(request_json(request), args.timeout)
        deadline = node.get_clock().now().nanoseconds + int(args.timeout * 1e9)
        while node._result is None and node.get_clock().now().nanoseconds < deadline:
            rclpy.spin_once(node, timeout_sec=0.1)
        if node._result is None:
            raise RuntimeError("timed out waiting for native capture result")
        print(json.dumps(node._result, indent=2, ensure_ascii=False))
        if not bool(node._result.get("success")):
            raise SystemExit(2)
    finally:
        node.destroy_node()
        rclpy.shutdown()
