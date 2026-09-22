"""Standalone PyQt6 UI for collecting AprilTag PnP calibration/benchmark data."""

from __future__ import annotations

import glob
import json
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import cv2
from msgs.msg import AprilTagDetection3DArray
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import rclpy
from rcl_interfaces.msg import Parameter, ParameterType, ParameterValue
from rcl_interfaces.srv import SetParameters
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import yaml

from .capture_protocol import TRUTH_FIELDS, build_capture_request, normalize_dataset_root, request_json


def _default_config() -> Path:
    try:
        from ament_index_python.packages import get_package_share_directory

        return Path(get_package_share_directory("apriltag_dataset_collector")) / "config" / "apriltag_dataset_collector.yaml"
    except Exception:
        return Path(__file__).resolve().parents[1] / "config" / "apriltag_dataset_collector.yaml"


def load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    params = payload.get("collector", {}).get("ros__parameters", {}) if isinstance(payload, dict) else {}
    if not isinstance(params, dict):
        raise ValueError(f"invalid collector configuration: {path}")
    return params


def decode_compressed(message: CompressedImage) -> np.ndarray | None:
    encoded = np.frombuffer(message.data, dtype=np.uint8)
    return cv2.imdecode(encoded, cv2.IMREAD_COLOR) if encoded.size else None


def discover_camera_devices() -> list[tuple[str, str]]:
    values: list[str] = []
    values.extend(sorted(glob.glob("/dev/v4l/by-id/*video-index0")))
    values.extend(sorted(glob.glob("/dev/video*")))
    unique: list[str] = []
    for value in values:
        if value not in unique:
            unique.append(value)
    return [(value, f"{Path(value).name}: {value}") for value in unique]


class CollectorNode(Node):
    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__("apriltag_dataset_collector_gui")
        self.config = config
        self.detector_node = str(config["detector_node"])
        self.latest_image: np.ndarray | None = None
        self.latest_status: dict[str, Any] = {}
        self.latest_camera_status: dict[str, Any] = {}
        self.latest_result: dict[str, Any] = {}
        self.latest_tags: dict[int, dict[str, float]] = {}
        self.last_image_at = 0.0
        self.last_status_at = 0.0
        self.last_camera_status_at = 0.0
        self.last_result_at = 0.0
        self.request_pub = self.create_publisher(String, str(config["dataset_capture_request_topic"]), 10)
        self.camera_control_pub = self.create_publisher(String, str(config["camera_control_topic"]), 10)
        self.create_subscription(CompressedImage, str(config["debug_image_topic"]), self._on_image, 10)
        self.create_subscription(String, str(config["vision_status_topic"]), self._on_status, 10)
        self.create_subscription(String, str(config["camera_status_topic"]), self._on_camera_status, 10)
        self.create_subscription(String, str(config["dataset_capture_result_topic"]), self._on_result, 10)
        self.create_subscription(AprilTagDetection3DArray, str(config["pnp_topic"]), self._on_tags, 10)
        self.parameter_client = self.create_client(SetParameters, f"{self.detector_node}/set_parameters")

    @staticmethod
    def _json(payload: str) -> dict[str, Any]:
        try:
            parsed = json.loads(payload)
            return parsed if isinstance(parsed, dict) else {"raw": payload}
        except json.JSONDecodeError:
            return {"raw": payload}

    def _on_image(self, message: CompressedImage) -> None:
        self.latest_image = decode_compressed(message)
        self.last_image_at = time.monotonic()

    def _on_status(self, message: String) -> None:
        self.latest_status = self._json(message.data)
        self.last_status_at = time.monotonic()

    def _on_camera_status(self, message: String) -> None:
        self.latest_camera_status = self._json(message.data)
        self.last_camera_status_at = time.monotonic()

    def _on_result(self, message: String) -> None:
        self.latest_result = self._json(message.data)
        self.last_result_at = time.monotonic()

    def _on_tags(self, message: AprilTagDetection3DArray) -> None:
        self.latest_tags = {
            int(item.tag_id): {
                "x": float(item.pose.pose.position.x),
                "y": float(item.pose.pose.position.y),
                "z": float(item.pose.pose.position.z),
                "margin": float(item.decision_margin),
                "reprojection": float(item.reprojection_error_px),
            }
            for item in message.detections
        }

    def send_camera_control(self, command: dict[str, Any]) -> None:
        message = String()
        message.data = json.dumps(command, separators=(",", ":"), allow_nan=False)
        self.camera_control_pub.publish(message)

    def publish_capture(self, request: dict[str, Any]) -> None:
        message = String()
        message.data = request_json(request)
        self.request_pub.publish(message)

    def set_root_async(self, root: str):
        if not self.parameter_client.service_is_ready():
            self.parameter_client.wait_for_service(timeout_sec=0.2)
        if not self.parameter_client.service_is_ready():
            raise RuntimeError(f"native detector parameter service unavailable: {self.detector_node}")
        request = SetParameters.Request()
        request.parameters = [
            Parameter(
                name="dataset_root",
                value=ParameterValue(type=ParameterType.PARAMETER_STRING, string_value=normalize_dataset_root(root)),
            )
        ]
        return self.parameter_client.call_async(request)


class CollectorWindow(QMainWindow):
    def __init__(self, node: CollectorNode) -> None:
        super().__init__()
        self.node = node
        self._pending_capture: dict[str, Any] | None = None
        self._pending_root_future = None
        self.setWindowTitle("FinsROV AprilTag Dataset Collector")
        self.resize(1480, 900)
        self._build_ui()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)
        self._refresh_devices()

    def _build_ui(self) -> None:
        central = QWidget()
        layout = QHBoxLayout(central)
        self.setCentralWidget(central)

        self.image = QLabel("Waiting for native debug image")
        self.image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image.setMinimumSize(760, 560)
        self.image.setStyleSheet("background:#202020; color:#cccccc;")
        layout.addWidget(self.image, 3)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.addWidget(self._camera_box())
        side_layout.addWidget(self._dataset_box())
        self.tag_table = QTableWidget(0, 6)
        self.tag_table.setHorizontalHeaderLabels(["Tag", "X", "Y", "Z", "Margin", "Reproj px"])
        side_layout.addWidget(self.tag_table, 2)
        self.status = QPlainTextEdit()
        self.status.setReadOnly(True)
        self.status.setMinimumHeight(200)
        side_layout.addWidget(self.status, 2)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(side)
        scroll.setMinimumWidth(470)
        layout.addWidget(scroll, 2)

    def _camera_box(self) -> QGroupBox:
        box = QGroupBox("Camera Control")
        layout = QGridLayout(box)
        self.device = QComboBox()
        refresh = QPushButton("Refresh")
        refresh.clicked.connect(self._refresh_devices)
        self.width = QSpinBox(); self.width.setRange(1, 7680); self.width.setValue(int(self.node.config["default_width"]))
        self.height = QSpinBox(); self.height.setRange(1, 4320); self.height.setValue(int(self.node.config["default_height"]))
        self.fps = QDoubleSpinBox(); self.fps.setRange(0.1, 240.0); self.fps.setValue(float(self.node.config["default_fps"]))
        self.fourcc = QLineEdit(str(self.node.config["default_fourcc"]))
        apply = QPushButton("Open / Apply")
        apply.clicked.connect(self._apply_camera)
        stop = QPushButton("Stop Publishing")
        stop.clicked.connect(lambda: self.node.send_camera_control({"enabled": False}))
        layout.addWidget(self.device, 0, 0, 1, 3); layout.addWidget(refresh, 0, 3)
        layout.addWidget(QLabel("Width"), 1, 0); layout.addWidget(self.width, 1, 1)
        layout.addWidget(QLabel("Height"), 1, 2); layout.addWidget(self.height, 1, 3)
        layout.addWidget(QLabel("FPS"), 2, 0); layout.addWidget(self.fps, 2, 1)
        layout.addWidget(QLabel("FourCC"), 2, 2); layout.addWidget(self.fourcc, 2, 3)
        layout.addWidget(apply, 3, 0, 1, 2); layout.addWidget(stop, 3, 2, 1, 2)
        return box

    def _dataset_box(self) -> QGroupBox:
        box = QGroupBox("AprilTag Dataset Capture")
        layout = QVBoxLayout(box)
        form = QFormLayout()
        self.root = QLineEdit(str(self.node.config["dataset_root"]))
        self.session = QLineEdit(time.strftime("%Y%m%d_%H%M%S"))
        self.operator = QLineEdit()
        self.tag_id = QLineEdit(); self.tag_id.setPlaceholderText("empty = best configured target")
        self.truth: dict[str, QLineEdit] = {}
        form.addRow("Dataset root", self.root)
        form.addRow("Session", self.session)
        form.addRow("Operator", self.operator)
        form.addRow("Tag ID", self.tag_id)
        for field, label in (("x_m", "X [m]"), ("y_m", "Y [m]"), ("z_m", "Z [m]"),
                             ("roll_deg", "Roll [deg]"), ("pitch_deg", "Pitch [deg]"), ("yaw_deg", "Yaw [deg]")):
            edit = QLineEdit(); edit.setPlaceholderText("empty / null")
            self.truth[field] = edit
            form.addRow(label, edit)
        self.note = QLineEdit()
        form.addRow("Note", self.note)
        layout.addLayout(form)
        actions = QHBoxLayout()
        apply_root = QPushButton("Apply Root")
        apply_root.clicked.connect(lambda: self._apply_root_then(None))
        capture = QPushButton("Capture Next Valid PnP Frame")
        capture.clicked.connect(self._capture)
        actions.addWidget(apply_root); actions.addWidget(capture)
        layout.addLayout(actions)
        self.capture_result = QLabel("No capture result")
        self.capture_result.setWordWrap(True)
        self.capture_result.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addWidget(self.capture_result)
        return box

    def _refresh_devices(self) -> None:
        previous = self.device.currentData()
        self.device.clear()
        for path, label in discover_camera_devices():
            self.device.addItem(label, path)
        if previous:
            index = self.device.findData(previous)
            if index >= 0:
                self.device.setCurrentIndex(index)

    def _apply_camera(self) -> None:
        selected = self.device.currentData()
        if not selected:
            self.capture_result.setText("No camera device selected")
            return
        self.node.send_camera_control({
            "device": str(selected), "width": int(self.width.value()), "height": int(self.height.value()),
            "fps": float(self.fps.value()), "fourcc": self.fourcc.text().strip() or "MJPG", "enabled": True,
        })

    def _capture(self) -> None:
        try:
            request = build_capture_request(
                session_id=self.session.text(), operator_name=self.operator.text(), tag_id=self.tag_id.text(),
                truth={field: edit.text() for field, edit in self.truth.items()}, note=self.note.text(),
            )
        except ValueError as exc:
            self.capture_result.setText(f"Invalid capture input: {exc}")
            return
        self._apply_root_then(request)

    def _apply_root_then(self, request: dict[str, Any] | None) -> None:
        try:
            self._pending_root_future = self.node.set_root_async(self.root.text())
            self._pending_capture = request
            self.capture_result.setText("Applying dataset root...")
        except (RuntimeError, ValueError) as exc:
            self.capture_result.setText(str(exc))

    def _tick(self) -> None:
        rclpy.spin_once(self.node, timeout_sec=0.0)
        self._finish_root_update()
        self._update_view()

    def _finish_root_update(self) -> None:
        future = self._pending_root_future
        if future is None or not future.done():
            return
        self._pending_root_future = None
        try:
            response = future.result()
            successful = bool(response and response.results and response.results[0].successful)
            if not successful:
                reason = response.results[0].reason if response and response.results else "unknown"
                raise RuntimeError(reason)
            if self._pending_capture is None:
                self.capture_result.setText(f"Dataset root active: {normalize_dataset_root(self.root.text())}")
            else:
                request = self._pending_capture
                self._pending_capture = None
                self.node.publish_capture(request)
                self.capture_result.setText(f"Capture queued: {request['request_id']}")
        except Exception as exc:
            self._pending_capture = None
            self.capture_result.setText(f"Dataset root update failed: {exc}")

    def _update_view(self) -> None:
        if self.node.latest_image is not None:
            rgb = cv2.cvtColor(self.node.latest_image, cv2.COLOR_BGR2RGB)
            height, width, channels = rgb.shape
            image = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
            self.image.setPixmap(QPixmap.fromImage(image.copy()).scaled(
                self.image.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation))
        self.tag_table.setRowCount(len(self.node.latest_tags))
        for row, (tag_id, value) in enumerate(sorted(self.node.latest_tags.items())):
            for column, cell in enumerate((tag_id, value["x"], value["y"], value["z"], value["margin"], value["reprojection"])):
                self.tag_table.setItem(row, column, QTableWidgetItem(str(cell) if column == 0 else f"{cell:.4f}"))
        if self.node.latest_result:
            result = self.node.latest_result
            label = "OK" if result.get("success") else "FAILED"
            self.capture_result.setText(
                f"{label}: {result.get('reason')}\n{result.get('message')}\n{result.get('image_raw_path')}")
        status = self.node.latest_status
        camera = self.node.latest_camera_status
        self.status.setPlainText(
            "Native detector\n"
            f"detected={status.get('detected')} ids={status.get('detected_ids')} target={status.get('target_tag_ids')}\n"
            f"PnP={status.get('pnp_valid')} reproj={status.get('pnp_reprojection_error_px')} px\n"
            f"camera={camera.get('device')} opened={camera.get('opened')} "
            f"{camera.get('actual_width')}x{camera.get('actual_height')} @ {camera.get('actual_fps')} Hz\n"
            f"detector error={camera.get('last_error')}\n"
            f"topics: request={self.node.config['dataset_capture_request_topic']}\n"
            f"result={self.node.config['dataset_capture_result_topic']}"
        )

    def closeEvent(self, event) -> None:
        self._timer.stop()
        super().closeEvent(event)


def _parse_args(argv: Sequence[str] | None = None):
    import argparse

    parser = argparse.ArgumentParser(description="Open the standalone FinsROV AprilTag dataset collector GUI.")
    parser.add_argument("--config", type=Path, default=_default_config())
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    config = load_config(args.config)
    rclpy.init(args=None)
    node = CollectorNode(config)
    app = QApplication(sys.argv)
    window = CollectorWindow(node)
    window.show()
    try:
        app.exec()
    finally:
        node.destroy_node()
        rclpy.shutdown()
