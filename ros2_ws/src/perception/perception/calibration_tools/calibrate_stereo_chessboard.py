from __future__ import annotations

import argparse
import glob
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate a stereo camera from chessboard image pairs.")
    parser.add_argument("--left-images", required=True, help="Glob for left images, for example '.../left/*.png'.")
    parser.add_argument("--right-images", required=True, help="Glob for right images, for example '.../right/*.png'.")
    parser.add_argument("--pattern-cols", type=int, default=9, help="Number of inner chessboard corners per row.")
    parser.add_argument("--pattern-rows", type=int, default=6, help="Number of inner chessboard corners per column.")
    parser.add_argument("--square-size", type=float, required=True, help="Chessboard square size in meters.")
    parser.add_argument("--camera-name", default="finsrov_stereo_udp5600")
    parser.add_argument("--output", required=True, help="Output stereo calibration YAML path.")
    parser.add_argument("--preview-dir", default="", help="Optional directory for detection and rectification previews.")
    parser.add_argument(
        "--refine-intrinsics",
        action="store_true",
        help="Allow stereoCalibrate to refine intrinsics instead of fixing monocular calibration results.",
    )
    args = parser.parse_args()

    left_paths = sorted(glob.glob(args.left_images))
    right_paths = sorted(glob.glob(args.right_images))
    if not left_paths:
        raise SystemExit(f"no left images matched {args.left_images!r}")
    if not right_paths:
        raise SystemExit(f"no right images matched {args.right_images!r}")
    if len(left_paths) != len(right_paths):
        raise SystemExit(f"left/right image count mismatch: {len(left_paths)} vs {len(right_paths)}")

    preview_dir = Path(args.preview_dir) if args.preview_dir else Path(args.output).with_suffix("")
    preview_dir = preview_dir.parent / f"{preview_dir.name}_preview"
    detection_dir = preview_dir / "detections"
    rectified_dir = preview_dir / "rectified"
    detection_dir.mkdir(parents=True, exist_ok=True)
    rectified_dir.mkdir(parents=True, exist_ok=True)

    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_template = np.zeros((args.pattern_cols * args.pattern_rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.pattern_cols, 0 : args.pattern_rows].T.reshape(-1, 2)
    object_template *= float(args.square_size)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    object_points: list[np.ndarray] = []
    left_points: list[np.ndarray] = []
    right_points: list[np.ndarray] = []
    valid_pairs: list[tuple[str, str]] = []
    image_size: tuple[int, int] | None = None

    for left_path, right_path in zip(left_paths, right_paths):
        left = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right = cv2.imread(right_path, cv2.IMREAD_COLOR)
        if left is None or right is None:
            print(f"skip unreadable pair: {left_path} | {right_path}")
            continue
        left_gray = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (left_gray.shape[1], left_gray.shape[0])
        elif image_size != (left_gray.shape[1], left_gray.shape[0]) or image_size != (
            right_gray.shape[1],
            right_gray.shape[0],
        ):
            print(f"skip size mismatch: {left_path} | {right_path}")
            continue

        found_left, corners_left = cv2.findChessboardCorners(left_gray, pattern_size)
        found_right, corners_right = cv2.findChessboardCorners(right_gray, pattern_size)
        if not (found_left and found_right):
            print(f"not found in both views: {left_path} | {right_path}")
            continue

        corners_left = cv2.cornerSubPix(left_gray, corners_left, (11, 11), (-1, -1), criteria)
        corners_right = cv2.cornerSubPix(right_gray, corners_right, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        left_points.append(corners_left)
        right_points.append(corners_right)
        valid_pairs.append((left_path, right_path))
        print(f"found pair: {left_path} | {right_path}")

        left_vis = left.copy()
        right_vis = right.copy()
        cv2.drawChessboardCorners(left_vis, pattern_size, corners_left, True)
        cv2.drawChessboardCorners(right_vis, pattern_size, corners_right, True)
        combined = cv2.hconcat([left_vis, right_vis])
        cv2.imwrite(str(detection_dir / f"pair_{len(valid_pairs)-1:04d}.png"), combined)

    if image_size is None:
        raise SystemExit("no readable stereo pairs")
    if len(valid_pairs) < 8:
        raise SystemExit(f"need at least 8 valid stereo pairs, got {len(valid_pairs)}")

    left_rms, K_left, D_left, left_rvecs, left_tvecs = cv2.calibrateCamera(
        object_points,
        left_points,
        image_size,
        None,
        None,
    )
    right_rms, K_right, D_right, right_rvecs, right_tvecs = cv2.calibrateCamera(
        object_points,
        right_points,
        image_size,
        None,
        None,
    )

    stereo_flags = 0 if args.refine_intrinsics else cv2.CALIB_FIX_INTRINSIC
    stereo_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-5)
    stereo_rms, K_left, D_left, K_right, D_right, R, T, E, F = cv2.stereoCalibrate(
        object_points,
        left_points,
        right_points,
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        criteria=stereo_criteria,
        flags=stereo_flags,
    )

    R1, R2, P1, P2, Q, roi1, roi2 = cv2.stereoRectify(
        K_left,
        D_left,
        K_right,
        D_right,
        image_size,
        R,
        T,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0.0,
    )

    map_l1, map_l2 = cv2.initUndistortRectifyMap(K_left, D_left, R1, P1, image_size, cv2.CV_32FC1)
    map_r1, map_r2 = cv2.initUndistortRectifyMap(K_right, D_right, R2, P2, image_size, cv2.CV_32FC1)
    for idx, (left_path, right_path) in enumerate(valid_pairs[:5]):
        left = cv2.imread(left_path, cv2.IMREAD_COLOR)
        right = cv2.imread(right_path, cv2.IMREAD_COLOR)
        left_rect = cv2.remap(left, map_l1, map_l2, cv2.INTER_LINEAR)
        right_rect = cv2.remap(right, map_r1, map_r2, cv2.INTER_LINEAR)
        combined = cv2.hconcat([left_rect, right_rect])
        for y in range(0, combined.shape[0], 40):
            cv2.line(combined, (0, y), (combined.shape[1], y), (0, 255, 0), 1)
        cv2.imwrite(str(rectified_dir / f"rectified_{idx:04d}.png"), combined)

    baseline_m = abs(float(P2[0, 3] / P2[0, 0]))
    output = {
        "camera_name": args.camera_name,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size_m": float(args.square_size),
        "valid_pair_count": len(valid_pairs),
        "input_pair_count": min(len(left_paths), len(right_paths)),
        "monocular_rms_px": {
            "left": float(left_rms),
            "right": float(right_rms),
        },
        "stereo_rms_px": float(stereo_rms),
        "left_camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(v) for v in K_left.reshape(-1)],
        },
        "right_camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(v) for v in K_right.reshape(-1)],
        },
        "left_distortion_coefficients": {
            "rows": 1,
            "cols": int(D_left.size),
            "data": [float(v) for v in D_left.reshape(-1)],
        },
        "right_distortion_coefficients": {
            "rows": 1,
            "cols": int(D_right.size),
            "data": [float(v) for v in D_right.reshape(-1)],
        },
        "stereo_extrinsics": {
            "R": matrix_to_rows(R),
            "T": [float(v) for v in T.reshape(-1)],
            "E": matrix_to_rows(E),
            "F": matrix_to_rows(F),
            "baseline_m": baseline_m,
        },
        "stereo_rectification": {
            "R1": matrix_to_rows(R1),
            "R2": matrix_to_rows(R2),
            "P1": matrix_to_rows(P1),
            "P2": matrix_to_rows(P2),
            "Q": matrix_to_rows(Q),
            "roi_left": [int(v) for v in roi1],
            "roi_right": [int(v) for v in roi2],
        },
        "depth_estimation_config": {
            "camera": {
                "fx": float(P1[0, 0]),
                "fy": float(P1[1, 1]),
                "cx": float(P1[0, 2]),
                "cy": float(P1[1, 2]),
                "baseline_m": baseline_m,
            },
            "rectification": {
                "enabled": True,
                "calibration_image_width": int(image_size[0]),
                "calibration_image_height": int(image_size[1]),
                "alpha": 0.0,
                "zero_disparity": True,
                "K_left": matrix_to_rows(K_left),
                "K_right": matrix_to_rows(K_right),
                "D_left": [float(v) for v in D_left.reshape(-1)],
                "D_right": [float(v) for v in D_right.reshape(-1)],
                "R": matrix_to_rows(R),
                "T": [float(v) for v in T.reshape(-1)],
            },
        },
        "valid_pairs": [{"left": left, "right": right} for left, right in valid_pairs],
        "preview_dir": str(preview_dir),
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)

    print(f"wrote {output_path}")
    print(f"valid pairs: {len(valid_pairs)}/{min(len(left_paths), len(right_paths))}")
    print(f"left RMS: {left_rms:.4f} px, right RMS: {right_rms:.4f} px, stereo RMS: {stereo_rms:.4f} px")
    print(
        "rectified intrinsics: "
        f"fx={P1[0,0]:.3f}, fy={P1[1,1]:.3f}, cx={P1[0,2]:.3f}, cy={P1[1,2]:.3f}, "
        f"baseline={baseline_m:.6f} m"
    )
    print(f"preview images: {preview_dir}")


def matrix_to_rows(matrix: np.ndarray) -> list[list[float]]:
    return [[float(v) for v in row] for row in np.asarray(matrix, dtype=np.float64)]


if __name__ == "__main__":
    main()
