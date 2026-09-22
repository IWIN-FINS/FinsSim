from __future__ import annotations

import glob
import json
import os
import subprocess
import sys
import time
from collections import deque

import cv2
import numpy as np
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor, QImage, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QComboBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
import rclpy
from msgs.msg import AprilTagDetection3DArray
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String


class MonitorNode(Node):
    def __init__(self) -> None:
        super().__init__("perception_monitor")
        self.image_topic = str(self.declare_parameter("image_topic", "/finsrov/camera/image_raw").value)
        self.debug_image_topic = str(
            self.declare_parameter("debug_image_topic", "/finsrov/camera/debug/compressed").value
        )
        self.pnp_topic = str(
            self.declare_parameter("pnp_topic", "/finsrov/vision/tag_poses_3d_camera").value
        )
        self.status_topic = str(self.declare_parameter("status_topic", "/finsrov/vision/status").value)
        self.state_status_topic = str(self.declare_parameter("state_status_topic", "/finsrov/state/status").value)
        self.camera_control_topic = str(
            self.declare_parameter("camera_control_topic", "/finsrov/camera/control").value
        )
        self.camera_status_topic = str(self.declare_parameter("camera_status_topic", "/finsrov/camera/status").value)

        self.latest_bgr: np.ndarray | None = None
        self.latest_pnp_tags: dict[int, dict] = {}
        self.latest_status: dict = {}
        self.latest_state_status: dict = {}
        self.latest_camera_status: dict = {}
        self.image_count = 0
        self.pnp_count = 0
        self.status_count = 0
        self.state_status_count = 0
        self.camera_status_count = 0
        self.last_image_time = 0.0
        self.last_pnp_time = 0.0
        self.last_status_time = 0.0
        self.last_state_status_time = 0.0
        self.last_camera_status_time = 0.0
        self.image_times: deque[float] = deque(maxlen=240)
        self.pnp_times: deque[float] = deque(maxlen=240)
        self.status_times: deque[float] = deque(maxlen=240)
        self.state_status_times: deque[float] = deque(maxlen=240)
        self.camera_status_times: deque[float] = deque(maxlen=120)

        self.camera_control_pub = self.create_publisher(String, self.camera_control_topic, 10)
        self.create_subscription(Image, self.image_topic, self._image_callback, 10)
        self.create_subscription(CompressedImage, self.debug_image_topic, self._compressed_image_callback, 10)
        self.create_subscription(AprilTagDetection3DArray, self.pnp_topic, self._pnp_callback, 10)
        self.create_subscription(String, self.status_topic, self._status_callback, 10)
        self.create_subscription(String, self.state_status_topic, self._state_status_callback, 10)
        self.create_subscription(String, self.camera_status_topic, self._camera_status_callback, 10)

    def _image_callback(self, msg: Image) -> None:
        self.latest_bgr = image_msg_to_bgr(msg)
        self.image_count += 1
        self.last_image_time = time.monotonic()
        self.image_times.append(self.last_image_time)

    def _compressed_image_callback(self, msg: CompressedImage) -> None:
        self.latest_bgr = compressed_image_msg_to_bgr(msg)
        self.image_count += 1
        self.last_image_time = time.monotonic()
        self.image_times.append(self.last_image_time)

    def _pnp_callback(self, msg: AprilTagDetection3DArray) -> None:
        now = time.monotonic()
        stamp_sec = float(msg.header.stamp.sec) + float(msg.header.stamp.nanosec) * 1e-9
        tags: dict[int, dict] = {}
        for detection in msg.detections:
            pose = detection.pose.pose
            tag_id = int(detection.tag_id)
            tags[tag_id] = {
                "tag_id": tag_id,
                "stamp_sec": stamp_sec,
                "last_seen": now,
                "decision_margin": float(detection.decision_margin),
                "reprojection_error_px": float(detection.reprojection_error_px),
                "camera_xyz": [
                    float(pose.position.x),
                    float(pose.position.y),
                    float(pose.position.z),
                ],
                "camera_quat_xyzw": [
                    float(pose.orientation.x),
                    float(pose.orientation.y),
                    float(pose.orientation.z),
                    float(pose.orientation.w),
                ],
            }
        for tag_id, payload in tags.items():
            self.latest_pnp_tags[tag_id] = payload
        self.pnp_count += 1
        self.last_pnp_time = now
        self.pnp_times.append(now)

    def _status_callback(self, msg: String) -> None:
        try:
            self.latest_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_status = {"raw": msg.data}
        self.status_count += 1
        self.last_status_time = time.monotonic()
        self.status_times.append(self.last_status_time)

    def _state_status_callback(self, msg: String) -> None:
        try:
            self.latest_state_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_state_status = {"raw": msg.data}
        self.state_status_count += 1
        self.last_state_status_time = time.monotonic()
        self.state_status_times.append(self.last_state_status_time)

    def _camera_status_callback(self, msg: String) -> None:
        try:
            self.latest_camera_status = json.loads(msg.data)
        except json.JSONDecodeError:
            self.latest_camera_status = {"raw": msg.data}
        self.camera_status_count += 1
        self.last_camera_status_time = time.monotonic()
        self.camera_status_times.append(self.last_camera_status_time)

    def send_camera_control(self, command: dict) -> None:
        msg = String()
        msg.data = json.dumps(command, separators=(",", ":"))
        self.camera_control_pub.publish(msg)


class PerceptionMonitor(QMainWindow):
    def __init__(self, node: MonitorNode) -> None:
        super().__init__()
        self._node = node
        self.setWindowTitle("FinsROV Perception Monitor")
        self.resize(1280, 720)

        central = QWidget()
        layout = QGridLayout(central)
        self._image_label = QLabel("Waiting for /finsrov/camera/image_raw")
        self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._image_label.setMinimumSize(640, 360)
        self._image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._image_label.setStyleSheet("background: #111; color: #ddd;")

        self._status_label = QLabel()
        self._status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._status_label.setMinimumWidth(260)
        self._status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._status_label.setStyleSheet("font-family: monospace; font-size: 13px;")

        self._tag_table = QTableWidget(0, 10)
        self._tag_table.setHorizontalHeaderLabels(
            ["ID", "Age", "Target", "Margin", "Err", "Cam X", "Cam Y", "Cam Z", "Used", "State"]
        )
        self._tag_table.verticalHeader().setVisible(False)
        self._tag_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._tag_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self._tag_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._tag_table.horizontalHeader().setStretchLastSection(True)
        self._tag_table.setMinimumHeight(170)
        self._tag_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self._camera_status_label = QLabel()
        self._camera_status_label.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self._camera_status_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self._camera_status_label.setStyleSheet("font-family: monospace; font-size: 13px;")

        middle_panel = QWidget()
        middle_panel.setMinimumWidth(330)
        middle_layout = QVBoxLayout(middle_panel)
        middle_layout.addWidget(self._build_camera_controls())
        middle_layout.addWidget(self._make_scroll_area(self._camera_status_label), 1)
        middle_layout.addWidget(self._tag_table, 2)

        layout.addWidget(self._image_label, 0, 0)
        layout.addWidget(middle_panel, 0, 1)
        layout.addWidget(self._make_scroll_area(self._status_label), 0, 2)
        layout.setColumnStretch(0, 4)
        layout.setColumnStretch(1, 2)
        layout.setColumnStretch(2, 2)
        self.setCentralWidget(central)

        self._refresh_devices()

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def _make_scroll_area(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setFrameShape(QScrollArea.Shape.NoFrame)
        area.setWidget(widget)
        return area

    def _build_camera_controls(self) -> QGroupBox:
        box = QGroupBox("Camera Control")
        layout = QVBoxLayout(box)

        self._device_combo = QComboBox()
        self._device_combo.setMinimumWidth(310)
        self._refresh_button = QPushButton("Refresh")
        self._refresh_button.clicked.connect(self._refresh_devices)

        device_row = QHBoxLayout()
        device_row.addWidget(self._device_combo, 1)
        device_row.addWidget(self._refresh_button)
        layout.addLayout(device_row)

        self._width_spin = QSpinBox()
        self._width_spin.setRange(160, 7680)
        self._width_spin.setValue(1280)
        self._height_spin = QSpinBox()
        self._height_spin.setRange(120, 4320)
        self._height_spin.setValue(720)
        self._fps_spin = QDoubleSpinBox()
        self._fps_spin.setRange(1.0, 240.0)
        self._fps_spin.setDecimals(1)
        self._fps_spin.setValue(60.0)
        self._fourcc_edit = QLineEdit("MJPG")
        self._fourcc_edit.setMaxLength(4)

        format_row = QGridLayout()
        format_row.addWidget(QLabel("W"), 0, 0)
        format_row.addWidget(self._width_spin, 0, 1)
        format_row.addWidget(QLabel("H"), 0, 2)
        format_row.addWidget(self._height_spin, 0, 3)
        format_row.addWidget(QLabel("FPS"), 1, 0)
        format_row.addWidget(self._fps_spin, 1, 1)
        format_row.addWidget(QLabel("FourCC"), 1, 2)
        format_row.addWidget(self._fourcc_edit, 1, 3)
        layout.addLayout(format_row)

        self._apply_button = QPushButton("Open / Apply")
        self._apply_button.clicked.connect(self._apply_camera)
        self._stop_button = QPushButton("Stop Publishing")
        self._stop_button.clicked.connect(self._stop_camera)

        action_row = QHBoxLayout()
        action_row.addWidget(self._apply_button)
        action_row.addWidget(self._stop_button)
        layout.addLayout(action_row)
        return box

    def _tick(self) -> None:
        try:
            rclpy.spin_once(self._node, timeout_sec=0.0)
        except KeyboardInterrupt:
            app = QApplication.instance()
            if app is not None:
                app.quit()
            return
        self._update_image()
        self._update_status()
        self._update_tag_table()
        self._update_camera_status()

    def _refresh_devices(self) -> None:
        previous = self._device_combo.currentData() if hasattr(self, "_device_combo") else None
        devices = discover_camera_devices()
        self._device_combo.clear()
        for path, label in devices:
            self._device_combo.addItem(label, path)
        if previous:
            for index in range(self._device_combo.count()):
                if self._device_combo.itemData(index) == previous:
                    self._device_combo.setCurrentIndex(index)
                    break
        elif self._device_combo.count() > 0:
            self._device_combo.setCurrentIndex(0)

    def _apply_camera(self) -> None:
        device = self._device_combo.currentData()
        if not device:
            return
        self._node.send_camera_control(
            {
                "device": str(device),
                "width": int(self._width_spin.value()),
                "height": int(self._height_spin.value()),
                "fps": float(self._fps_spin.value()),
                "fourcc": self._fourcc_edit.text().strip() or "MJPG",
                "enabled": True,
            }
        )

    def _stop_camera(self) -> None:
        self._node.send_camera_control({"enabled": False})

    def _update_image(self) -> None:
        if self._node.latest_bgr is None:
            return
        frame = self._node.latest_bgr.copy()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, channels = rgb.shape
        qimage = QImage(rgb.data, width, height, channels * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        self._image_label.setPixmap(
            pixmap.scaled(
                self._image_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _update_status(self) -> None:
        now = time.monotonic()
        status = self._node.latest_status
        lines = [
            "Topics",
            f"image:  {self._node.image_topic}",
            f"debug:  {self._node.debug_image_topic}",
            f"pnp:    {self._node.pnp_topic}",
            f"status: {self._node.status_topic}",
            f"fusion: {self._node.state_status_topic}",
            "",
            "Counters",
            f"images: {self._node.image_count} ({estimate_rate(self._node.image_times):.1f} Hz)",
            f"pnp:    {self._node.pnp_count} ({estimate_rate(self._node.pnp_times):.1f} Hz)",
            f"status: {self._node.status_count} ({estimate_rate(self._node.status_times):.1f} Hz)",
            f"fusion: {self._node.state_status_count} ({estimate_rate(self._node.state_status_times):.1f} Hz)",
            "",
            "Age",
            f"image:  {format_age(now - self._node.last_image_time)}",
            f"pnp:    {format_age(now - self._node.last_pnp_time)}",
            f"status: {format_age(now - self._node.last_status_time)}",
            f"fusion: {format_age(now - self._node.last_state_status_time)}",
            "",
            "Detection",
            f"detected: {status.get('detected')}",
            f"tag_id:   {status.get('tag_id')}",
            f"family:   {status.get('detected_family')}",
            f"ids:      {status.get('detected_ids')}",
            f"targets:  {status.get('target_tag_ids')}",
            f"pixel:    {status.get('pixel_xy')}",
            f"world:    {status.get('world_xy')}",
            f"pnp ok:   {status.get('pnp_valid')}",
            f"pnp xyz:  {format_sequence(status.get('pnp_camera_xyz'), precision=4)} m",
            f"pnp quat: {format_sequence(status.get('pnp_camera_quat_xyzw'), precision=4)}",
            f"pnp rpy:  {format_sequence(status.get('pnp_camera_rpy_deg'), precision=2)} deg",
            f"pnp err:  {format_optional_float(status.get('pnp_reprojection_error_px'))} px",
            f"pnp edge: {format_optional_float(status.get('pnp_mean_edge_px'))} px",
            f"pnp z~:   {format_optional_float(status.get('pnp_edge_z_estimate_m'))} m",
            "",
            "Fusion",
            f"ready:   {self._node.latest_state_status.get('ready')}",
            f"fresh:   V={self._node.latest_state_status.get('vision_fresh')} "
            f"I={self._node.latest_state_status.get('imu_fresh')} "
            f"D={self._node.latest_state_status.get('depth_fresh')}",
            f"used:    {self._node.latest_state_status.get('used_tag_ids')}",
            f"reject:  {self._node.latest_state_status.get('reject_reason')}",
        ]
        self._status_label.setText("\n".join(lines))

    def _update_tag_table(self) -> None:
        now = time.monotonic()
        status = self._node.latest_status
        state_status = self._node.latest_state_status
        used_ids = set()
        raw_used = state_status.get("used_tag_ids")
        if isinstance(raw_used, list):
            used_ids = {int(item) for item in raw_used if isinstance(item, (int, float, str)) and str(item).lstrip("-").isdigit()}

        detail_by_id: dict[int, dict] = {}
        details = status.get("detections_detail")
        if isinstance(details, list):
            for detail in details:
                if isinstance(detail, dict) and detail.get("tag_id") is not None:
                    try:
                        detail_by_id[int(detail["tag_id"])] = detail
                    except (TypeError, ValueError):
                        pass

        stale_after = 1.0
        remove_after = 3.0
        for tag_id in list(self._node.latest_pnp_tags):
            if now - float(self._node.latest_pnp_tags[tag_id].get("last_seen", 0.0)) > remove_after:
                del self._node.latest_pnp_tags[tag_id]

        tag_ids = sorted(set(self._node.latest_pnp_tags) | set(detail_by_id))
        self._tag_table.setRowCount(len(tag_ids))
        for row, tag_id in enumerate(tag_ids):
            pnp = self._node.latest_pnp_tags.get(tag_id, {})
            detail = detail_by_id.get(tag_id, {})
            age = now - float(pnp.get("last_seen", self._node.last_status_time))
            stale = age > stale_after
            xyz = pnp.get("camera_xyz") or detail.get("pnp_camera_xyz") or [None, None, None]
            used = tag_id in used_ids
            row_values = [
                str(tag_id),
                format_age(age),
                "yes" if bool(detail.get("target_match", False)) else "",
                format_optional_float(pnp.get("decision_margin", detail.get("decision_margin"))),
                format_optional_float(pnp.get("reprojection_error_px", detail.get("pnp_reprojection_error_px"))),
                format_optional_float(xyz[0] if isinstance(xyz, list) and len(xyz) > 0 else None),
                format_optional_float(xyz[1] if isinstance(xyz, list) and len(xyz) > 1 else None),
                format_optional_float(xyz[2] if isinstance(xyz, list) and len(xyz) > 2 else None),
                "yes" if used else "",
                "stale" if stale else "live",
            ]
            for col, value in enumerate(row_values):
                item = QTableWidgetItem(value)
                if used:
                    item.setBackground(QColor(30, 90, 50))
                    item.setForeground(QColor(245, 245, 245))
                elif stale:
                    item.setBackground(QColor(70, 70, 70))
                    item.setForeground(QColor(230, 230, 230))
                self._tag_table.setItem(row, col, item)

    def _update_camera_status(self) -> None:
        now = time.monotonic()
        status = self._node.latest_camera_status
        lines = [
            "Camera",
            f"control: {self._node.camera_control_topic}",
            f"status:  {self._node.camera_status_topic}",
            f"age:     {format_age(now - self._node.last_camera_status_time)}",
            f"count:   {self._node.camera_status_count}",
            "",
            f"enabled: {status.get('enabled')}",
            f"opened:  {status.get('opened')}",
            f"device:  {status.get('device')}",
            f"size:    {status.get('width')}x{status.get('height')}",
            f"fps:     {status.get('fps')}",
            f"fourcc:  {status.get('fourcc')}",
            f"actual:  {status.get('actual_width')}x{status.get('actual_height')} "
            f"@ {format_optional_float(status.get('actual_fps'))} Hz {status.get('actual_fourcc')}",
            f"frames:  {status.get('frame_count')}",
            f"pub hz:  {format_optional_float(status.get('publish_fps'))}",
            f"gui hz:  {estimate_rate(self._node.image_times):.1f}",
            f"mean:    {format_optional_float(status.get('last_frame_mean'))}",
            f"min/max: {status.get('last_frame_min')}/{status.get('last_frame_max')}",
            f"error:   {status.get('last_error')}",
            "",
        ]
        self._camera_status_label.setText("\n".join(lines))


def format_age(age: float) -> str:
    if age > 1e8:
        return "never"
    return f"{age:.3f}s"


def format_optional_float(value: object) -> str:
    if value is None:
        return "None"
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return str(value)


def format_sequence(value: object, precision: int = 3) -> str:
    if not isinstance(value, (list, tuple)):
        return "None"
    try:
        return "[" + ", ".join(f"{float(item):.{precision}f}" for item in value) + "]"
    except (TypeError, ValueError):
        return str(value)


def estimate_rate(times: deque[float]) -> float:
    if len(times) < 2:
        return 0.0
    elapsed = times[-1] - times[0]
    if elapsed <= 0.0:
        return 0.0
    return (len(times) - 1) / elapsed


def discover_camera_devices() -> list[tuple[str, str]]:
    candidates: list[str] = []
    candidates.extend(sorted(glob.glob("/dev/v4l/by-id/*video-index0")))
    candidates.extend(sorted(glob.glob("/dev/video*")))

    devices: list[tuple[str, str]] = []
    seen_realpaths: set[str] = set()
    for candidate in candidates:
        realpath = os.path.realpath(candidate)
        if realpath in seen_realpaths:
            continue
        if not is_video_capture_device(candidate):
            continue
        seen_realpaths.add(realpath)
        label = candidate
        if candidate.startswith("/dev/v4l/by-id/"):
            label = f"{candidate} -> {realpath}"
        devices.append((candidate, label))
    return devices


def is_video_capture_device(path: str) -> bool:
    try:
        result = subprocess.run(
            ["v4l2-ctl", "--device", path, "--all"],
            check=False,
            capture_output=True,
            text=True,
            timeout=1.0,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return os.path.basename(path).startswith("video")
    if result.returncode != 0:
        return False
    output = result.stdout + result.stderr
    if "Format Video Capture:" in output:
        return True
    device_caps = output.split("Device Caps", 1)[-1].split("Media Driver Info", 1)[0]
    return "Video Capture" in device_caps


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    if msg.encoding == "bgr8":
        return np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3).copy()
    if msg.encoding == "rgb8":
        rgb = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if msg.encoding in {"mono8", "8UC1"}:
        mono = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width)
        return cv2.cvtColor(mono, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


def compressed_image_msg_to_bgr(msg: CompressedImage) -> np.ndarray:
    data = np.frombuffer(msg.data, dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"failed to decode compressed image format {msg.format!r}")
    return image


def main(args=None) -> None:
    rclpy.init(args=args)
    app = QApplication(sys.argv)
    node = MonitorNode()
    window = PerceptionMonitor(node)
    window.show()
    try:
        app.exec()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()


if __name__ == "__main__":
    main()
