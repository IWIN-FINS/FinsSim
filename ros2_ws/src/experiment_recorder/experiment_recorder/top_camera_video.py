"""Write an unannotated overhead-camera topic to one MP4 file.

The node deliberately subscribes to the native detector's *raw* compressed
image topic instead of opening a V4L2 device.  It therefore neither competes
with AprilTag detection for the camera nor records debug overlays.  The T2
runner owns its process lifetime, but this command is also useful for a
standalone optical record during another hardware experiment.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from pathlib import Path
import time
from typing import Any, Sequence

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from sensor_msgs.msg import CompressedImage

from .manifest import now_iso, write_json


DEFAULT_TOPIC = "/finsrov/camera/raw/compressed"


def metadata_path_for_video(output_path: Path) -> Path:
    """Return the stable sidecar name used for readiness and provenance."""

    return output_path.with_suffix(".metadata.json")


@dataclass(frozen=True)
class VideoRecordingSpec:
    """Validated, non-control parameters for one raw-camera video."""

    image_topic: str
    output_path: Path
    fps: float
    codec: str

    @classmethod
    def create(
        cls,
        *,
        image_topic: str,
        output_path: Path,
        fps: float,
        codec: str,
    ) -> "VideoRecordingSpec":
        topic = str(image_topic).strip()
        if not topic.startswith("/"):
            raise ValueError("image topic must be an absolute ROS topic")
        path = Path(output_path).expanduser().resolve()
        if path.suffix.lower() != ".mp4":
            raise ValueError("output path must use the .mp4 suffix")
        if not math.isfinite(float(fps)) or float(fps) <= 0.0:
            raise ValueError("fps must be a positive finite number")
        fourcc = str(codec).strip()
        if len(fourcc) != 4:
            raise ValueError("codec must be a four-character OpenCV FourCC (for example mp4v)")
        return cls(image_topic=topic, output_path=path, fps=float(fps), codec=fourcc)


class RawTopCameraVideoRecorder(Node):
    """Read-only compressed-image subscriber that writes an MP4 without overlays."""

    def __init__(self, spec: VideoRecordingSpec) -> None:
        super().__init__("finsrov_top_camera_video_recorder")
        self._spec = spec
        self._metadata_path = metadata_path_for_video(spec.output_path)
        self._writer: cv2.VideoWriter | None = None
        self._writer_failed = False
        self._frame_count = 0
        self._decode_failures = 0
        self._size_mismatches = 0
        self._first_header_stamp_sec: float | None = None
        self._last_header_stamp_sec: float | None = None
        self._first_received_monotonic: float | None = None
        self._last_received_monotonic: float | None = None
        self._resolution: tuple[int, int] | None = None
        self._closed = False

        spec.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_metadata("waiting_for_frame")
        self.create_subscription(CompressedImage, spec.image_topic, self._image_callback, 10)
        self.get_logger().info(
            f"recording unannotated overhead frames from {spec.image_topic} "
            f"-> {spec.output_path} at {spec.fps:.1f} fps"
        )

    @property
    def metadata_path(self) -> Path:
        return self._metadata_path

    def _header_stamp_sec(self, message: CompressedImage) -> float | None:
        stamp = message.header.stamp
        value = float(stamp.sec) + float(stamp.nanosec) * 1e-9
        return value if value > 0.0 else None

    def _write_metadata(self, status: str, *, error: str | None = None) -> None:
        duration = None
        if self._first_header_stamp_sec is not None and self._last_header_stamp_sec is not None:
            duration = max(0.0, self._last_header_stamp_sec - self._first_header_stamp_sec)
        source_fps_estimate = None
        if duration is not None and duration > 0.0 and self._frame_count > 1:
            source_fps_estimate = (self._frame_count - 1) / duration
        payload: dict[str, Any] = {
            "schema_version": 1,
            "status": status,
            "created_at": now_iso(),
            "source": {
                "topic": self._spec.image_topic,
                "message_type": "sensor_msgs/msg/CompressedImage",
                "annotation": "none; detector raw image before AprilTag debug rendering",
            },
            "output": {
                "path": str(self._spec.output_path),
                "codec": self._spec.codec,
                "nominal_fps": self._spec.fps,
                "resolution_px": list(self._resolution) if self._resolution is not None else None,
            },
            "frames_written": self._frame_count,
            "source_fps_estimate": source_fps_estimate,
            "decode_failures": self._decode_failures,
            "resolution_mismatches": self._size_mismatches,
            "first_header_stamp_sec": self._first_header_stamp_sec,
            "last_header_stamp_sec": self._last_header_stamp_sec,
            "source_duration_sec": duration,
        }
        if error is not None:
            payload["error"] = error
        write_json(self._metadata_path, payload)

    def _open_writer(self, image: np.ndarray) -> bool:
        height, width = image.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*self._spec.codec)
        writer = cv2.VideoWriter(str(self._spec.output_path), fourcc, self._spec.fps, (width, height))
        if not writer.isOpened():
            self.get_logger().error(
                f"cannot open MP4 writer at {self._spec.output_path} "
                f"(codec={self._spec.codec}, size={width}x{height})"
            )
            self._writer_failed = True
            self._write_metadata("failed", error="cv2.VideoWriter could not be opened")
            return False
        self._writer = writer
        self._resolution = (width, height)
        return True

    def _image_callback(self, message: CompressedImage) -> None:
        if self._writer_failed:
            return
        encoded = np.frombuffer(bytes(message.data), dtype=np.uint8)
        image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            self._decode_failures += 1
            if self._decode_failures == 1 or self._decode_failures % 50 == 0:
                self.get_logger().warning("could not decode compressed raw camera frame")
            return
        if self._writer is None and not self._open_writer(image):
            return
        assert self._writer is not None
        height, width = image.shape[:2]
        if self._resolution != (width, height):
            self._size_mismatches += 1
            self.get_logger().warning(
                f"skipping raw camera frame with changed resolution {width}x{height}; "
                f"expected {self._resolution}"
            )
            return
        self._writer.write(image)
        self._frame_count += 1
        header_stamp = self._header_stamp_sec(message)
        if header_stamp is not None:
            if self._first_header_stamp_sec is None:
                self._first_header_stamp_sec = header_stamp
            self._last_header_stamp_sec = header_stamp
        now = time.monotonic()
        if self._first_received_monotonic is None:
            self._first_received_monotonic = now
        self._last_received_monotonic = now
        if self._frame_count == 1:
            self._write_metadata("recording")
            self.get_logger().info("first unannotated camera frame written")
        else:
            # The T2 runner reads this compact sidecar before dispatching a
            # trajectory, so it can reject a stale 0.5 Hz raw publisher rather
            # than creating a time-distorted MP4 at the requested 10 Hz.
            self._write_metadata("recording")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        status = "failed" if self._writer_failed else ("complete" if self._frame_count > 0 else "no_frames")
        self._write_metadata(status)
        self.get_logger().info(
            f"top-camera video finalized: status={status} frames={self._frame_count} "
            f"output={self._spec.output_path}"
        )


def _parse_args(argv: Sequence[str] | None = None) -> VideoRecordingSpec:
    parser = argparse.ArgumentParser(
        description="Record the unannotated FinsROV overhead raw-camera topic to one MP4 file."
    )
    parser.add_argument("--output", required=True, type=Path, help="Per-trial .mp4 output path.")
    parser.add_argument("--topic", default=DEFAULT_TOPIC, help="Unannotated CompressedImage source topic.")
    parser.add_argument("--fps", type=float, default=10.0, help="MP4 frame rate; match the raw publisher rate.")
    parser.add_argument("--codec", default="mp4v", help="Four-character OpenCV video codec (default: mp4v).")
    args = parser.parse_args(argv)
    try:
        return VideoRecordingSpec.create(
            image_topic=args.topic,
            output_path=args.output,
            fps=args.fps,
            codec=args.codec,
        )
    except ValueError as exc:
        parser.error(str(exc))
        raise AssertionError("argparse.error does not return")  # pragma: no cover


def main(argv: Sequence[str] | None = None) -> None:
    spec = _parse_args(argv)
    rclpy.init(args=None, signal_handler_options=SignalHandlerOptions.NO)
    node: RawTopCameraVideoRecorder | None = None
    try:
        node = RawTopCameraVideoRecorder(spec)
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        if node is not None:
            node.close()
            node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
