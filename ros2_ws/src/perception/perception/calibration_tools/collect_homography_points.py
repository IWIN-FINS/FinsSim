from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


TAG_DICTIONARIES = {
    "DICT_APRILTAG_16H5": cv2.aruco.DICT_APRILTAG_16H5,
    "DICT_APRILTAG_25H9": cv2.aruco.DICT_APRILTAG_25H9,
    "DICT_APRILTAG_36H10": cv2.aruco.DICT_APRILTAG_36H10,
    "DICT_APRILTAG_36H11": cv2.aruco.DICT_APRILTAG_36H11,
    "DICT_4X4_50": cv2.aruco.DICT_4X4_50,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect AprilTag pixel/world pairs and write homography YAML.")
    parser.add_argument("--device", default="/dev/v4l/by-id/usb-DSJ_USB_Camera_200901010001-video-index0")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--fourcc", default="MJPG")
    parser.add_argument("--tag-family", default="DICT_APRILTAG_36H11", choices=sorted(TAG_DICTIONARIES))
    parser.add_argument("--target-tag-id", type=int, default=-1)
    parser.add_argument("--output", required=True)
    parser.add_argument("--load-existing", action="store_true")
    args = parser.parse_args()

    image_points: list[list[float]] = []
    world_points: list[list[float]] = []
    output_path = Path(args.output)
    if args.load_existing and output_path.exists():
        image_points, world_points = load_existing_points(output_path)
        print(f"loaded {len(image_points)} existing point pairs from {output_path}")

    dictionary = cv2.aruco.getPredefinedDictionary(TAG_DICTIONARIES[args.tag_family])
    parameters = cv2.aruco.DetectorParameters()
    parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_APRILTAG
    detector = cv2.aruco.ArucoDetector(dictionary, parameters)

    capture = cv2.VideoCapture(args.device, cv2.CAP_V4L2)
    if len(args.fourcc) >= 4:
        capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*args.fourcc[:4]))
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    capture.set(cv2.CAP_PROP_FPS, args.fps)
    if not capture.isOpened():
        raise SystemExit(f"failed to open camera {args.device}")

    print(f"camera opened: {args.device}")
    print("keys: c=record current tag center, u=undo, s=save, q/esc=quit")
    print("when recording, enter world x y in meters in the terminal")

    try:
        while True:
            ok, frame = capture.read()
            if not ok or frame is None:
                continue

            detected = detect_tag_center(frame, detector, args.target_tag_id)
            display = frame.copy()
            if detected is not None:
                tag_id, center, corners = detected
                draw_corners = corners.reshape(1, 4, 2).astype(np.float32)
                cv2.aruco.drawDetectedMarkers(display, [draw_corners], np.asarray([[tag_id]], dtype=np.int32))
                cv2.drawMarker(
                    display,
                    (int(round(center[0])), int(round(center[1]))),
                    (0, 255, 0),
                    cv2.MARKER_CROSS,
                    32,
                    2,
                )

            label = f"pairs={len(image_points)} detected={detected[0] if detected else None}"
            cv2.putText(display, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv2.imshow("collect_homography_points", display)
            key = cv2.waitKey(1) & 0xFF

            if key in {ord("q"), 27}:
                break
            if key == ord("u"):
                if image_points:
                    removed = (image_points.pop(), world_points.pop())
                    print(f"removed last pair: {removed}")
                continue
            if key == ord("s"):
                write_homography_yaml(output_path, image_points, world_points)
                print(f"saved {output_path}")
                continue
            if key == ord("c"):
                if detected is None:
                    print("no target tag detected")
                    continue
                world_xy = prompt_world_xy()
                if world_xy is None:
                    continue
                _, center, _ = detected
                image_points.append([float(center[0]), float(center[1])])
                world_points.append([float(world_xy[0]), float(world_xy[1])])
                print(f"added image={image_points[-1]} world={world_points[-1]}")
    finally:
        capture.release()
        cv2.destroyAllWindows()

    if len(image_points) >= 4:
        write_homography_yaml(output_path, image_points, world_points)
        print(f"saved {output_path}")
    else:
        print(f"not enough points to save homography: {len(image_points)} pairs")


def detect_tag_center(frame: np.ndarray, detector, target_tag_id: int):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = detector.detectMarkers(gray)
    if ids is None or len(ids) == 0:
        return None
    for index, tag_id_array in enumerate(ids):
        tag_id = int(tag_id_array[0])
        if target_tag_id >= 0 and tag_id != target_tag_id:
            continue
        tag_corners = np.asarray(corners[index], dtype=np.float64).reshape(4, 2)
        center = tag_corners.mean(axis=0)
        return tag_id, center, tag_corners
    return None


def prompt_world_xy() -> tuple[float, float] | None:
    text = input("world x y [m], empty to cancel: ").strip()
    if not text:
        return None
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        print("expected exactly two numbers")
        return None
    try:
        return float(parts[0]), float(parts[1])
    except ValueError:
        print("invalid number")
        return None


def load_existing_points(path: Path) -> tuple[list[list[float]], list[list[float]]]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return list(data.get("image_points", [])), list(data.get("world_points", []))


def write_homography_yaml(path: Path, image_points: list[list[float]], world_points: list[list[float]]) -> None:
    if len(image_points) != len(world_points):
        raise ValueError("image_points and world_points length mismatch")
    if len(image_points) < 4:
        raise ValueError("at least 4 point pairs are required")
    homography, mask = cv2.findHomography(
        np.asarray(image_points, dtype=np.float64),
        np.asarray(world_points, dtype=np.float64),
        method=0,
    )
    if homography is None or mask is None:
        raise ValueError("failed to compute homography")
    payload = {
        "image_points": image_points,
        "world_points": world_points,
        "pixel_to_world_homography": [float(v) for v in homography.reshape(-1)],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, sort_keys=False, allow_unicode=True)


if __name__ == "__main__":
    main()
