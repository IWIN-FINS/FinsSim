from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate image/world point quality in a homography YAML file.")
    parser.add_argument("yaml_file", help="YAML containing image_points and world_points.")
    parser.add_argument("--ransac", action="store_true", help="Use RANSAC for the main fit.")
    parser.add_argument("--ransac-threshold", type=float, default=3.0, help="RANSAC reprojection threshold in world units.")
    args = parser.parse_args()

    image_points, world_points = load_points(Path(args.yaml_file))
    method = cv2.RANSAC if args.ransac else 0
    homography, mask = cv2.findHomography(image_points, world_points, method=method, ransacReprojThreshold=args.ransac_threshold)
    if homography is None:
        raise SystemExit("failed to compute homography")

    predicted = apply_homography(homography, image_points)
    errors = np.linalg.norm(predicted - world_points, axis=1)
    pixel_area = convex_hull_area(image_points)
    world_area = convex_hull_area(world_points)

    print("Homography quality report")
    print(f"file: {args.yaml_file}")
    print(f"point_count: {len(image_points)}")
    print(f"method: {'RANSAC' if args.ransac else 'direct least squares'}")
    print()
    print("Coverage")
    print(f"image_bbox: u=[{image_points[:,0].min():.3f}, {image_points[:,0].max():.3f}], "
          f"v=[{image_points[:,1].min():.3f}, {image_points[:,1].max():.3f}]")
    print(f"world_bbox: x=[{world_points[:,0].min():.4f}, {world_points[:,0].max():.4f}], "
          f"y=[{world_points[:,1].min():.4f}, {world_points[:,1].max():.4f}]")
    print(f"image_convex_hull_area_px2: {pixel_area:.3f}")
    print(f"world_convex_hull_area_m2: {world_area:.6f}")
    if pixel_area <= 1e-6 or world_area <= 1e-9:
        print("WARNING: points are nearly collinear or area is too small; homography will be unstable.")
    print()
    print("Fit residuals on calibration points")
    print(f"mean_error_m: {errors.mean():.6f}")
    print(f"max_error_m: {errors.max():.6f}")
    print(f"rmse_m: {np.sqrt(np.mean(errors ** 2)):.6f}")
    print()
    print("Per-point residuals")
    for index, (image, world, estimate, error) in enumerate(zip(image_points, world_points, predicted, errors)):
        inlier = ""
        if mask is not None:
            inlier = f" inlier={int(mask[index][0])}"
        print(
            f"{index:02d}: image=[{image[0]:.3f}, {image[1]:.3f}] "
            f"world=[{world[0]:+.4f}, {world[1]:+.4f}] "
            f"estimated=[{estimate[0]:+.4f}, {estimate[1]:+.4f}] "
            f"error={error:.6f} m{inlier}"
        )

    print()
    print("Homography matrix pixel_to_world")
    for row in homography:
        print("  " + " ".join(f"{value:+.12e}" for value in row))

    if len(image_points) == 4:
        print()
        print("NOTE: exactly 4 point pairs determine a homography exactly.")
        print("      The residuals above can be near zero even when the mapping is poor elsewhere.")
        print("      Add 6-12 points across the full working area to evaluate real quality.")
    elif len(image_points) >= 5:
        print()
        print("Leave-one-out validation")
        loo_errors = leave_one_out_errors(image_points, world_points)
        print(f"loo_mean_error_m: {loo_errors.mean():.6f}")
        print(f"loo_max_error_m: {loo_errors.max():.6f}")
        print(f"loo_rmse_m: {np.sqrt(np.mean(loo_errors ** 2)):.6f}")
        for index, error in enumerate(loo_errors):
            print(f"{index:02d}: loo_error={error:.6f} m")


def load_points(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if "image_points" not in data or "world_points" not in data:
        raise SystemExit("YAML must contain image_points and world_points")

    image_points = np.asarray(data["image_points"], dtype=np.float64)
    world_points = np.asarray(data["world_points"], dtype=np.float64)
    if image_points.ndim != 2 or image_points.shape[1] != 2:
        raise SystemExit(f"image_points must be Nx2, got {image_points.shape}")
    if world_points.ndim != 2 or world_points.shape[1] != 2:
        raise SystemExit(f"world_points must be Nx2, got {world_points.shape}")
    if len(image_points) != len(world_points):
        raise SystemExit("image_points and world_points must have the same length")
    if len(image_points) < 4:
        raise SystemExit("at least 4 point pairs are required")
    return image_points, world_points


def apply_homography(homography: np.ndarray, points: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    mapped = (homography @ homogeneous.T).T
    scale = mapped[:, 2:3]
    if np.any(np.abs(scale) < 1e-12):
        raise SystemExit("homography produced a near-zero homogeneous scale")
    return mapped[:, :2] / scale


def convex_hull_area(points: np.ndarray) -> float:
    hull = cv2.convexHull(points.astype(np.float32))
    return float(cv2.contourArea(hull))


def leave_one_out_errors(image_points: np.ndarray, world_points: np.ndarray) -> np.ndarray:
    errors = []
    for heldout in range(len(image_points)):
        train_mask = np.ones(len(image_points), dtype=bool)
        train_mask[heldout] = False
        homography, _ = cv2.findHomography(image_points[train_mask], world_points[train_mask], method=0)
        if homography is None:
            errors.append(float("nan"))
            continue
        estimate = apply_homography(homography, image_points[heldout : heldout + 1])[0]
        errors.append(float(np.linalg.norm(estimate - world_points[heldout])))
    return np.asarray(errors, dtype=np.float64)


if __name__ == "__main__":
    main()
