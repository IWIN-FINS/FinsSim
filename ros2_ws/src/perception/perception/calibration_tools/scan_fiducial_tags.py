from __future__ import annotations

import argparse
import time

import cv2
import numpy as np


DICTIONARIES = {
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
    "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
    "DICT_4X4_250": cv2.aruco.DICT_4X4_250,
    "DICT_4X4_1000": cv2.aruco.DICT_4X4_1000,
    "DICT_5X5_50": cv2.aruco.DICT_5X5_50,
    "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
    "DICT_5X5_250": cv2.aruco.DICT_5X5_250,
    "DICT_5X5_1000": cv2.aruco.DICT_5X5_1000,
    "DICT_6X6_50": cv2.aruco.DICT_6X6_50,
    "DICT_6X6_100": cv2.aruco.DICT_6X6_100,
    "DICT_6X6_250": cv2.aruco.DICT_6X6_250,
    "DICT_6X6_1000": cv2.aruco.DICT_6X6_1000,
    "DICT_7X7_50": cv2.aruco.DICT_7X7_50,
    "DICT_7X7_100": cv2.aruco.DICT_7X7_100,
    "DICT_7X7_250": cv2.aruco.DICT_7X7_250,
    "DICT_7X7_1000": cv2.aruco.DICT_7X7_1000,
    "DICT_APRILTAG_16H5": cv2.aruco.DICT_APRILTAG_16H5,
    "DICT_APRILTAG_25H9": cv2.aruco.DICT_APRILTAG_25H9,
    "DICT_APRILTAG_36H10": cv2.aruco.DICT_APRILTAG_36H10,
    "DICT_APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36H11,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan camera frames across ArUco/AprilTag dictionaries.")
    parser.add_argument("--device", default="/dev/v4l/by-id/usb-DSJ_USB_Camera_200901010001-video-index0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--no-gui", action="store_true")
    parser.add_argument("--print-every-sec", type=float, default=1.0)
    args = parser.parse_args()

    detectors = {}
    for name, dictionary_id in DICTIONARIES.items():
        dictionary = cv2.aruco.getPredefinedDictionary(dictionary_id)
        params = cv2.aruco.DetectorParameters()
        params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG if "APRILTAG" in name else cv2.aruco.CORNER_REFINE_SUBPIX
        detectors[name] = cv2.aruco.ArucoDetector(dictionary, params)

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
    print("keys: q/esc=quit")

    last_print = 0.0
    last_seen: dict[str, list[int]] = {}

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                time.sleep(0.05)
                continue

            detections = detect_all(frame, detectors)
            if detections:
                last_seen = {name: ids for name, ids, _ in detections}

            now = time.monotonic()
            if now - last_print >= args.print_every_sec:
                last_print = now
                if detections:
                    print("detected:", format_detections(detections))
                else:
                    print("detected: none", f"last_seen={last_seen}" if last_seen else "")

            if not args.no_gui:
                display = frame.copy()
                draw_best_detection(display, detections)
                cv2.putText(
                    display,
                    format_detections(detections) if detections else "detected: none",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0) if detections else (0, 0, 255),
                    2,
                )
                cv2.imshow("scan_fiducial_tags", display)
                key = cv2.waitKey(1) & 0xFF
                if key in {ord("q"), 27}:
                    break
    finally:
        capture.release()
        if not args.no_gui:
            cv2.destroyAllWindows()


def detect_all(frame: np.ndarray, detectors: dict[str, cv2.aruco.ArucoDetector]):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    detections = []
    for name, detector in detectors.items():
        corners, ids, _ = detector.detectMarkers(gray)
        if ids is None or len(ids) == 0:
            continue
        detections.append((name, [int(v[0]) for v in ids], corners))
    detections.sort(key=lambda item: len(item[1]), reverse=True)
    return detections


def format_detections(detections) -> str:
    parts = [f"{name}: ids={ids}" for name, ids, _ in detections[:4]]
    return " | ".join(parts)


def draw_best_detection(frame: np.ndarray, detections) -> None:
    if not detections:
        return
    name, ids, corners = detections[0]
    id_array = np.asarray([[tag_id] for tag_id in ids], dtype=np.int32)
    cv2.aruco.drawDetectedMarkers(frame, corners, id_array)
    cv2.putText(frame, name, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)


if __name__ == "__main__":
    main()
