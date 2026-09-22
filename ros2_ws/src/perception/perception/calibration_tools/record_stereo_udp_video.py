from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from ._stereo_udp import StereoUdpSideBySideCapture


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a side-by-side stereo UDP stream and split into left/right videos.")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--latency-ms", type=int, default=60)
    parser.add_argument("--duration-sec", type=float, default=30.0)
    parser.add_argument("--fps", type=float, default=20.0, help="Output video FPS.")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--show-preview", action="store_true", help="Display a live preview while recording.")
    parser.add_argument("--preview-scale", type=float, default=1.0, help="Preview window scale factor.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = StereoUdpSideBySideCapture(port=args.port, latency_ms=args.latency_ms)
    capture.open()

    combined_writer = None
    left_writer = None
    right_writer = None
    frame_count = 0
    start_time = time.monotonic()
    end_time = start_time + args.duration_sec

    try:
        print(f"recording stereo UDP stream on port {args.port} for {args.duration_sec:.1f}s")
        print(f"output: {output_dir}")
        while time.monotonic() < end_time:
            stereo = capture.read()
            if stereo is None:
                continue

            if combined_writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                combined_path = output_dir / "stereo_combined.mp4"
                left_path = output_dir / "stereo_left.mp4"
                right_path = output_dir / "stereo_right.mp4"
                combined_writer = cv2.VideoWriter(str(combined_path), fourcc, args.fps, (stereo.combined.shape[1], stereo.combined.shape[0]))
                left_writer = cv2.VideoWriter(str(left_path), fourcc, args.fps, (stereo.left.shape[1], stereo.left.shape[0]))
                right_writer = cv2.VideoWriter(str(right_path), fourcc, args.fps, (stereo.right.shape[1], stereo.right.shape[0]))
                if not combined_writer.isOpened() or not left_writer.isOpened() or not right_writer.isOpened():
                    raise RuntimeError("failed to open output video writer")

            combined_writer.write(stereo.combined)
            left_writer.write(stereo.left)
            right_writer.write(stereo.right)
            frame_count += 1

            if args.show_preview:
                display = stereo.combined
                if args.preview_scale != 1.0:
                    display = cv2.resize(
                        display,
                        (int(display.shape[1] * args.preview_scale), int(display.shape[0] * args.preview_scale)),
                        interpolation=cv2.INTER_AREA,
                    )
                elapsed = time.monotonic() - start_time
                cv2.putText(
                    display,
                    f"REC {elapsed:05.1f}s  frames={frame_count}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    2,
                )
                cv2.imshow("record_stereo_udp_video", display)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    print("recording interrupted by user")
                    break
    finally:
        capture.close()
        if combined_writer is not None:
            combined_writer.release()
        if left_writer is not None:
            left_writer.release()
        if right_writer is not None:
            right_writer.release()
        if args.show_preview:
            cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_time
    metadata_path = output_dir / "recording_info.txt"
    metadata_path.write_text(
        "\n".join(
            [
                f"port={args.port}",
                f"latency_ms={args.latency_ms}",
                f"requested_duration_sec={args.duration_sec}",
                f"output_fps={args.fps}",
                f"frames_written={frame_count}",
                f"elapsed_sec={elapsed:.3f}",
                f"recorded_at={datetime.now().isoformat(timespec='seconds')}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"done: wrote {frame_count} frames to {output_dir}")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ros2_ws_dir = Path(__file__).resolve().parents[4]
    return ros2_ws_dir / "data" / "calibration_videos" / f"stereo_record_{stamp}"


if __name__ == "__main__":
    main()
