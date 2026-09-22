from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture chessboard images for camera calibration.")
    parser.add_argument("--device", default="/dev/v4l/by-id/usb-DSJ_USB_Camera_200901010001-video-index0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pattern-cols", type=int, default=9, help="Number of inner chessboard corners per row.")
    parser.add_argument("--pattern-rows", type=int, default=6, help="Number of inner chessboard corners per column.")
    parser.add_argument("--max-images", type=int, default=40)
    parser.add_argument("--min-interval-sec", type=float, default=0.8)
    parser.add_argument("--auto", action="store_true", help="Automatically save when a chessboard is detected.")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)

    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if len(args.fourcc) >= 4:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise SystemExit(f"failed to open camera {args.device}")

    actual_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = float(capture.get(cv2.CAP_PROP_FPS))
    print(f"camera opened: {args.device}")
    print(f"actual: {actual_width}x{actual_height} @ {actual_fps:.2f} fps")
    print(f"output: {output_dir}")
    print("keys: space/c=save, q/esc=quit")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    saved_count = 0
    last_save_time = 0.0

    try:
        while saved_count < args.max_images:
            ok, frame = capture.read()
            if not ok or frame is None:
                print("camera read failed")
                time.sleep(0.1)
                continue

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, pattern_size, corners, found)

            now = time.monotonic()
            should_save = bool(args.auto and found and now - last_save_time >= args.min_interval_sec)

            key = -1
            if not args.no_gui:
                label = f"saved {saved_count}/{args.max_images} found={found}"
                cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                cv2.imshow("capture_chessboard_images", display)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
                if key in {ord(" "), ord("c")}:
                    should_save = True

            if should_save:
                path = output_dir / f"chessboard_{saved_count:04d}.png"
                cv2.imwrite(str(path), frame)
                saved_count += 1
                last_save_time = now
                print(f"saved {path}")

            if args.no_gui and not args.auto:
                time.sleep(0.05)
    finally:
        capture.release()
        if not args.no_gui:
            cv2.destroyAllWindows()

    print(f"done: saved {saved_count} images to {output_dir}")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return Path("calibration_images") / f"rgb_chessboard_{stamp}"


if __name__ == "__main__":
    main()
