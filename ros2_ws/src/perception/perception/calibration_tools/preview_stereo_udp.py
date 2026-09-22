from __future__ import annotations

import argparse

import cv2

from ._stereo_udp import StereoUdpSideBySideCapture


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview side-by-side stereo UDP stream.")
    parser.add_argument("--port", type=int, default=5600)
    parser.add_argument("--latency-ms", type=int, default=60)
    parser.add_argument("--scale", type=float, default=1.0, help="Display scale for the preview window.")
    args = parser.parse_args()

    capture = StereoUdpSideBySideCapture(port=args.port, latency_ms=args.latency_ms)
    capture.open()
    print(f"previewing stereo UDP stream on port {args.port}")
    print("keys: q/esc=quit")

    try:
        while True:
            frame = capture.read()
            if frame is None:
                continue
            display = frame.combined
            if args.scale != 1.0:
                display = cv2.resize(
                    display,
                    (int(display.shape[1] * args.scale), int(display.shape[0] * args.scale)),
                    interpolation=cv2.INTER_AREA,
                )
            cv2.imshow("preview_stereo_udp", display)
            key = cv2.waitKey(1) & 0xFF
            if key in {ord("q"), 27}:
                break
    finally:
        capture.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
