from __future__ import annotations

import argparse
import glob
import math
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate camera intrinsics from chessboard images.")
    parser.add_argument("--images", required=True, help="Glob pattern, for example 'calibration_images/run/*.png'.")
    parser.add_argument("--pattern-cols", type=int, default=9, help="Number of inner chessboard corners per row.")
    parser.add_argument("--pattern-rows", type=int, default=6, help="Number of inner chessboard corners per column.")
    parser.add_argument("--square-size", type=float, required=True, help="Chessboard square size in meters.")
    parser.add_argument("--camera-name", default="finsrov_overhead_rgb_camera")
    parser.add_argument("--output", required=True, help="Output YAML path.")
    parser.add_argument("--preview-dir", default="", help="Optional directory for detected-corner preview images.")
    parser.add_argument(
        "--free-aspect-ratio",
        action="store_true",
        help=(
            "Allow fx and fy to be estimated independently. By default the script assumes square pixels "
            "and fixes fx/fy aspect ratio, which is safer for USB cameras and weak calibration datasets."
        ),
    )
    parser.add_argument(
        "--initial-fov-deg",
        type=float,
        default=90.0,
        help="Initial horizontal FOV guess used when fx/fy aspect ratio is fixed. Default: 90 deg.",
    )
    parser.add_argument(
        "--zero-tangent-dist",
        action="store_true",
        help="Force tangential distortion p1/p2 to zero.",
    )
    parser.add_argument(
        "--fix-k3",
        action="store_true",
        help="Force radial distortion k3 to zero.",
    )
    args = parser.parse_args()

    image_paths = sorted(glob.glob(args.images))
    if not image_paths:
        raise SystemExit(f"no images matched {args.images!r}")

    pattern_size = (args.pattern_cols, args.pattern_rows)
    object_template = np.zeros((args.pattern_cols * args.pattern_rows, 3), np.float32)
    object_template[:, :2] = np.mgrid[0 : args.pattern_cols, 0 : args.pattern_rows].T.reshape(-1, 2)
    object_template *= float(args.square_size)

    object_points: list[np.ndarray] = []
    image_points: list[np.ndarray] = []
    image_size: tuple[int, int] | None = None
    valid_paths: list[str] = []

    preview_dir = Path(args.preview_dir) if args.preview_dir else None
    if preview_dir is not None:
        preview_dir.mkdir(parents=True, exist_ok=True)

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    for path in image_paths:
        image = cv2.imread(path, cv2.IMREAD_COLOR)
        if image is None:
            print(f"skip unreadable image: {path}")
            continue
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        if image_size is None:
            image_size = (gray.shape[1], gray.shape[0])
        elif image_size != (gray.shape[1], gray.shape[0]):
            print(f"skip size mismatch: {path}")
            continue

        found, corners = cv2.findChessboardCorners(gray, pattern_size)
        if not found:
            print(f"not found: {path}")
            continue

        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        object_points.append(object_template.copy())
        image_points.append(corners)
        valid_paths.append(path)
        print(f"found: {path}")

        if preview_dir is not None:
            preview = image.copy()
            cv2.drawChessboardCorners(preview, pattern_size, corners, True)
            cv2.imwrite(str(preview_dir / Path(path).name), preview)

    if image_size is None:
        raise SystemExit("no readable images")
    if len(object_points) < 5:
        raise SystemExit(f"need at least 5 valid chessboard images, got {len(object_points)}")

    flags = 0
    initial_camera_matrix = None
    if not args.free_aspect_ratio:
        initial_focal = image_size[0] / (2.0 * math.tan(math.radians(args.initial_fov_deg) * 0.5))
        initial_camera_matrix = np.array(
            [
                [initial_focal, 0.0, image_size[0] * 0.5],
                [0.0, initial_focal, image_size[1] * 0.5],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        flags |= cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO
    if args.zero_tangent_dist:
        flags |= cv2.CALIB_ZERO_TANGENT_DIST
    if args.fix_k3:
        flags |= cv2.CALIB_FIX_K3

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        object_points,
        image_points,
        image_size,
        initial_camera_matrix,
        None,
        flags=flags,
    )
    per_view_errors = compute_per_view_errors(object_points, image_points, rvecs, tvecs, camera_matrix, dist_coeffs)
    quality = compute_dataset_quality(image_points, image_size)
    fov_x = 2.0 * math.degrees(math.atan(image_size[0] / (2.0 * camera_matrix[0, 0])))
    fov_y = 2.0 * math.degrees(math.atan(image_size[1] / (2.0 * camera_matrix[1, 1])))

    output = {
        "camera_name": args.camera_name,
        "image_width": int(image_size[0]),
        "image_height": int(image_size[1]),
        "camera_matrix": {
            "rows": 3,
            "cols": 3,
            "data": [float(v) for v in camera_matrix.reshape(-1)],
        },
        "distortion_model": "plumb_bob",
        "distortion_coefficients": {
            "rows": 1,
            "cols": int(dist_coeffs.size),
            "data": [float(v) for v in dist_coeffs.reshape(-1)],
        },
        "reprojection_error_px": float(rms),
        "valid_image_count": len(valid_paths),
        "input_image_count": len(image_paths),
        "pattern_cols": args.pattern_cols,
        "pattern_rows": args.pattern_rows,
        "square_size_m": float(args.square_size),
        "calibration_flags": {
            "free_aspect_ratio": bool(args.free_aspect_ratio),
            "fix_aspect_ratio": not bool(args.free_aspect_ratio),
            "initial_fov_deg": float(args.initial_fov_deg),
            "zero_tangent_dist": bool(args.zero_tangent_dist),
            "fix_k3": bool(args.fix_k3),
        },
        "estimated_fov_deg": {
            "horizontal": float(fov_x),
            "vertical": float(fov_y),
        },
        "dataset_quality": quality,
        "valid_images": valid_paths,
        "per_view_reprojection_errors_px": per_view_errors,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(output, f, sort_keys=False, allow_unicode=True)

    print(f"wrote {output_path}")
    print(f"valid images: {len(valid_paths)}/{len(image_paths)}")
    print(f"RMS reprojection error: {rms:.4f} px")
    print(f"camera matrix fx={camera_matrix[0,0]:.3f}, fy={camera_matrix[1,1]:.3f}, "
          f"cx={camera_matrix[0,2]:.3f}, cy={camera_matrix[1,2]:.3f}")
    print(f"estimated FOV: horizontal={fov_x:.2f} deg, vertical={fov_y:.2f} deg")
    print(
        "dataset quality: corner coverage="
        f"{quality['corner_coverage_x']:.2f}x{quality['corner_coverage_y']:.2f}, "
        f"mean board span={quality['mean_board_span_x_px']:.1f}x{quality['mean_board_span_y_px']:.1f}px, "
        f"span variation={quality['board_span_variation_x_px']:.1f}x{quality['board_span_variation_y_px']:.1f}px"
    )
    if args.free_aspect_ratio:
        aspect_error = abs(camera_matrix[0, 0] / camera_matrix[1, 1] - 1.0)
        if aspect_error > 0.1:
            print(
                "WARNING: fx/fy differ by more than 10%. Most USB cameras have square pixels; "
                "try rerunning without --free-aspect-ratio."
            )
    if quality["board_span_variation_x_px"] < 80.0 or quality["board_span_variation_y_px"] < 80.0:
        print(
            "WARNING: chessboard scale variation is low. Capture images with the board at several distances "
            "and strong tilts to make focal length observable."
        )


def compute_per_view_errors(
    object_points: list[np.ndarray],
    image_points: list[np.ndarray],
    rvecs,
    tvecs,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> list[float]:
    errors: list[float] = []
    for obj, img, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(img, projected, cv2.NORM_L2) / np.sqrt(len(projected))
        errors.append(float(error))
    return errors


def compute_dataset_quality(image_points: list[np.ndarray], image_size: tuple[int, int]) -> dict:
    all_points = np.vstack([points.reshape(-1, 2) for points in image_points])
    spans = []
    centers = []
    for points in image_points:
        pts = points.reshape(-1, 2)
        spans.append([float(np.ptp(pts[:, 0])), float(np.ptp(pts[:, 1]))])
        centers.append([float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1]))])
    spans_array = np.asarray(spans, dtype=np.float64)
    centers_array = np.asarray(centers, dtype=np.float64)
    return {
        "corner_coverage_x": float((np.max(all_points[:, 0]) - np.min(all_points[:, 0])) / image_size[0]),
        "corner_coverage_y": float((np.max(all_points[:, 1]) - np.min(all_points[:, 1])) / image_size[1]),
        "center_coverage_x": float((np.max(centers_array[:, 0]) - np.min(centers_array[:, 0])) / image_size[0]),
        "center_coverage_y": float((np.max(centers_array[:, 1]) - np.min(centers_array[:, 1])) / image_size[1]),
        "mean_board_span_x_px": float(np.mean(spans_array[:, 0])),
        "mean_board_span_y_px": float(np.mean(spans_array[:, 1])),
        "board_span_variation_x_px": float(np.max(spans_array[:, 0]) - np.min(spans_array[:, 0])),
        "board_span_variation_y_px": float(np.max(spans_array[:, 1]) - np.min(spans_array[:, 1])),
    }


if __name__ == "__main__":
    main()
