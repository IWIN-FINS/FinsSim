from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path

import cv2

from ._stereo_udp import StereoUdpSideBySideCapture


def main() -> None:
    parser = argparse.ArgumentParser(description="Capture stereo chessboard image pairs from UDP side-by-side stream.")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--latency-ms", type=int, default=60)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--pattern-cols", type=int, default=9, help="Number of inner chessboard corners per row.")
    parser.add_argument("--pattern-rows", type=int, default=6, help="Number of inner chessboard corners per column.")
    parser.add_argument("--max-pairs", type=int, default=40)
    parser.add_argument("--min-interval-sec", type=float, default=0.8)
    parser.add_argument("--auto", action="store_true", help="Automatically save when the board is detected in both views.")
    parser.add_argument("--no-gui", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir()
    left_dir = output_dir / "left"
    right_dir = output_dir / "right"
    preview_dir = output_dir / "preview"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)
    preview_dir.mkdir(parents=True, exist_ok=True)

    capture = StereoUdpSideBySideCapture(port=args.port, latency_ms=args.latency_ms)
    capture.open()
    print(f"stereo stream opened on UDP {args.port}")
    print(f"output: {output_dir}")
    print("keys: space/c=save, q/esc=quit")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    saved_count = 0
    last_save_time = 0.0

    try:
        while saved_count < args.max_pairs:
            stereo = capture.read()
            if stereo is None:
                continue

            left_gray = cv2.cvtColor(stereo.left, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(stereo.right, cv2.COLOR_BGR2GRAY)
            found_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size)
            found_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size)

            display_left = stereo.left.copy()
            display_right = stereo.right.copy()
            if found_left:
                corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display_left, pattern_size, corners_left, True)
            if found_right:
                corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
                cv2.drawChessboardCorners(display_right, pattern_size, corners_right, True)

            display = cv2.hconcat([display_left, display_right])
            status = f"saved {saved_count}/{args.max_pairs} left={found_left} right={found_right}"
            cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

            now = time.monotonic()
            should_save = bool(
                args.auto and found_left and found_right and now - last_save_time >= args.min_interval_sec
            )

            key = -1
            if not args.no_gui:
                cv2.imshow("capture_stereo_chessboard_images", display)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
                if key in {ord(" "), ord("c")}:
                    should_save = True

            if should_save:
                if not (found_left and found_right):
                    print("skip save: chessboard must be found in both left and right images")
                    continue
                left_path = left_dir / f"left_{saved_count:04d}.png"
                right_path = right_dir / f"right_{saved_count:04d}.png"
                preview_path = preview_dir / f"pair_{saved_count:04d}.png"
                cv2.imwrite(str(left_path), stereo.left)
                cv2.imwrite(str(right_path), stereo.right)
                cv2.imwrite(str(preview_path), display)
                saved_count += 1
                last_save_time = now
                print(f"saved pair {saved_count:04d}: {left_path.name}, {right_path.name}")

            if args.no_gui and not args.auto:
                time.sleep(0.05)
    finally:
        capture.close()
        if not args.no_gui:
            cv2.destroyAllWindows()

    print(f"done: saved {saved_count} stereo pairs to {output_dir}")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    ros2_ws_dir = Path(__file__).resolve().parents[4]
    return ros2_ws_dir / "data" / "calibration_images" / f"stereo_chessboard_{stamp}"


if __name__ == "__main__":
    main()
