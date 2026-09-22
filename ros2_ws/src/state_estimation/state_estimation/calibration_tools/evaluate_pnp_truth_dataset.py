from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from state_estimation.transforms import (
    matrix_to_pose,
    pose_matrix,
    quat_to_rpy_xyzw,
    rpy_to_quat_xyzw,
)


def _transform_from_node(node: dict[str, Any]) -> np.ndarray:
    if "matrix" in node:
        return np.asarray(node["matrix"], dtype=np.float64).reshape(4, 4)
    translation = node.get("translation_xyz", [0.0, 0.0, 0.0])
    if "rotation_xyzw" in node:
        rotation = node["rotation_xyzw"]
    elif "rotation_rpy_deg" in node:
        rotation = rpy_to_quat_xyzw(*[math.radians(float(value)) for value in node["rotation_rpy_deg"]])
    else:
        rotation = [0.0, 0.0, 0.0, 1.0]
    return pose_matrix(translation, rotation)


def _load_t_world_camera(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as stream:
        data = yaml.safe_load(stream) or {}
    if "T_world_camera" not in data:
        raise ValueError(f"{path} has no T_world_camera")
    return _transform_from_node(data["T_world_camera"])


def _truth_to_ros(
    row: dict[str, str],
    origin_xy: tuple[float, float],
    z_mode: str,
    water_surface_z_m: float,
) -> np.ndarray:
    x = float(row["truth_x_m"]) - origin_xy[0]
    y = float(row["truth_y_m"]) - origin_xy[1]
    z_raw = float(row["truth_z_m"])
    if z_mode == "depth_from_surface":
        z = water_surface_z_m - z_raw
    elif z_mode in ("negative_depth", "depth_down"):
        z = -z_raw
    elif z_mode == "z_up":
        z = z_raw
    else:
        raise ValueError(f"unsupported truth z mode: {z_mode}")
    return np.array([x, y, z], dtype=np.float64)


def _camera_tag_from_row(row: dict[str, str]) -> np.ndarray:
    translation = [
        float(row["pnp_camera_x_m"]),
        float(row["pnp_camera_y_m"]),
        float(row["pnp_camera_z_m"]),
    ]
    rotation_xyzw = [
        float(row["pnp_quat_x"]),
        float(row["pnp_quat_y"]),
        float(row["pnp_quat_z"]),
        float(row["pnp_quat_w"]),
    ]
    return pose_matrix(translation, rotation_xyzw)


def _fit_rigid_transform(source_points: np.ndarray, target_points: np.ndarray) -> np.ndarray:
    source_centroid = np.mean(source_points, axis=0)
    target_centroid = np.mean(target_points, axis=0)
    source_centered = source_points - source_centroid
    target_centered = target_points - target_centroid
    covariance = source_centered.T @ target_centered
    u, _, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0.0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def _summarize_errors(errors: np.ndarray) -> dict[str, Any]:
    norms = np.linalg.norm(errors, axis=1)
    return {
        "count": int(errors.shape[0]),
        "rmse_xyz_m": np.sqrt(np.mean(errors * errors, axis=0)).tolist(),
        "rmse_norm_m": float(np.sqrt(np.mean(norms * norms))),
        "mean_error_xyz_m": np.mean(errors, axis=0).tolist(),
        "median_norm_m": float(np.median(norms)),
        "p95_norm_m": float(np.percentile(norms, 95.0)),
        "max_norm_m": float(np.max(norms)),
    }


def _format_vector(values, precision: int = 6) -> str:
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in values) + "]"


def _transform_summary(transform: np.ndarray) -> dict[str, Any]:
    translation, quat = matrix_to_pose(transform)
    roll, pitch, yaw = quat_to_rpy_xyzw(quat)
    return {
        "translation_xyz": translation.tolist(),
        "rotation_xyzw": quat.tolist(),
        "rotation_rpy_deg": [math.degrees(roll), math.degrees(pitch), math.degrees(yaw)],
    }


def _print_summary(report: dict[str, Any]) -> None:
    print("=== Dataset ===")
    print(f"samples_csv: {report['samples_csv']}")
    print(f"used_samples: {report['used_samples']}")
    print(f"truth_frame_conversion: calibration_left_bottom -> ROS center")
    print(f"origin_xy_calibration_in_ros: {report['origin_xy_calibration_in_ros']}")
    print(f"truth_z_mode: {report['truth_z_mode']}")
    print(f"water_surface_z_m: {report['water_surface_z_m']}")
    print()

    print("=== Current T_world_camera Evaluation ===")
    current = report["current"]
    print(f"rmse_xyz_m: {_format_vector(current['rmse_xyz_m'])}")
    print(f"rmse_norm_m: {current['rmse_norm_m']:.6f}")
    print(f"mean_error_xyz_m: {_format_vector(current['mean_error_xyz_m'])}")
    print(f"median_norm_m: {current['median_norm_m']:.6f}")
    print(f"p95_norm_m: {current['p95_norm_m']:.6f}")
    print(f"max_norm_m: {current['max_norm_m']:.6f}")
    print()

    print("=== Best Rigid Fit From Dataset ===")
    fit = report["fit"]
    print(f"rmse_xyz_m: {_format_vector(fit['rmse_xyz_m'])}")
    print(f"rmse_norm_m: {fit['rmse_norm_m']:.6f}")
    print(f"mean_error_xyz_m: {_format_vector(fit['mean_error_xyz_m'])}")
    print(f"median_norm_m: {fit['median_norm_m']:.6f}")
    print(f"p95_norm_m: {fit['p95_norm_m']:.6f}")
    print(f"max_norm_m: {fit['max_norm_m']:.6f}")
    print()

    fit_tf = report["fit_t_world_camera"]
    print("fit T_world_camera YAML snippet:")
    print("T_world_camera:")
    print("  translation_xyz:")
    for value in fit_tf["translation_xyz"]:
        print(f"  - {value:.12f}")
    print("  rotation_xyzw:")
    for value in fit_tf["rotation_xyzw"]:
        print(f"  - {value:.12f}")
    print(f"# rotation_rpy_deg: {_format_vector(fit_tf['rotation_rpy_deg'], 3)}")
    print()

    delta = report["delta_fit_from_current"]
    print("=== Delta: T_fit * inverse(T_current) ===")
    print(f"translation_xyz_m: {_format_vector(delta['translation_xyz'])}")
    print(f"rotation_rpy_deg: {_format_vector(delta['rotation_rpy_deg'], 3)}")
    print()

    print("=== Worst Current Samples ===")
    for item in report["worst_current_samples"]:
        print(
            f"{item['sample_id']} truth={_format_vector(item['truth_ros_xyz_m'])} "
            f"pred={_format_vector(item['current_pred_xyz_m'])} "
            f"err={_format_vector(item['current_error_xyz_m'])} "
            f"norm={item['current_error_norm_m']:.6f} reproj={item['pnp_reprojection_error_px']:.3f}"
        )


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    dataset = Path(args.dataset).expanduser().resolve()
    samples_csv = dataset if dataset.name.endswith(".csv") else dataset / "samples.csv"
    extrinsics = Path(args.extrinsics).expanduser()
    if not extrinsics.is_absolute():
        extrinsics = Path.cwd() / extrinsics
    extrinsics = extrinsics.resolve()

    t_world_camera = _load_t_world_camera(extrinsics)
    rows = list(csv.DictReader(samples_csv.open("r", encoding="utf-8")))

    origin_xy = (float(args.origin_xy[0]), float(args.origin_xy[1]))
    records = []
    camera_points = []
    truth_points = []
    current_predictions = []
    current_errors = []
    grouped_errors: dict[tuple[float, float, float], list[np.ndarray]] = defaultdict(list)

    for row in rows:
        if not row.get("truth_x_m") or not row.get("truth_y_m") or not row.get("truth_z_m"):
            continue
        reprojection = float(row.get("pnp_reprojection_error_px") or 0.0)
        if args.max_reprojection_error_px is not None and reprojection > args.max_reprojection_error_px:
            continue
        t_camera_tag = _camera_tag_from_row(row)
        camera_tag_position = t_camera_tag[:3, 3].copy()
        truth_ros = _truth_to_ros(row, origin_xy, args.truth_z_mode, float(args.water_surface_z_m))
        current_pred = (t_world_camera @ t_camera_tag)[:3, 3]
        current_error = current_pred - truth_ros

        camera_points.append(camera_tag_position)
        truth_points.append(truth_ros)
        current_predictions.append(current_pred)
        current_errors.append(current_error)
        key = (
            float(row["truth_x_m"]),
            float(row["truth_y_m"]),
            float(row["truth_z_m"]),
        )
        grouped_errors[key].append(current_error)
        records.append(
            {
                "sample_id": row["sample_id"],
                "tag_id": int(row["tag_id"]),
                "pnp_reprojection_error_px": reprojection,
                "truth_ros_xyz_m": truth_ros,
                "current_pred_xyz_m": current_pred,
                "current_error_xyz_m": current_error,
                "current_error_norm_m": float(np.linalg.norm(current_error)),
            }
        )

    if len(records) < 3:
        raise ValueError("not enough valid samples to evaluate")

    camera_points_np = np.asarray(camera_points, dtype=np.float64)
    truth_points_np = np.asarray(truth_points, dtype=np.float64)
    current_errors_np = np.asarray(current_errors, dtype=np.float64)
    t_world_camera_fit = _fit_rigid_transform(camera_points_np, truth_points_np)
    fit_predictions = (t_world_camera_fit[:3, :3] @ camera_points_np.T).T + t_world_camera_fit[:3, 3]
    fit_errors = fit_predictions - truth_points_np

    delta = t_world_camera_fit @ np.linalg.inv(t_world_camera)
    delta_summary = _transform_summary(delta)

    group_summaries = []
    for key, errors in grouped_errors.items():
        values = np.asarray(errors, dtype=np.float64)
        mean_error = np.mean(values, axis=0)
        group_summaries.append(
            {
                "truth_calibration_xyz_m": list(key),
                "sample_count": int(values.shape[0]),
                "mean_current_error_xyz_m": mean_error.tolist(),
                "mean_current_error_norm_m": float(np.linalg.norm(mean_error)),
            }
        )
    group_summaries.sort(key=lambda item: item["mean_current_error_norm_m"], reverse=True)

    worst = sorted(records, key=lambda item: item["current_error_norm_m"], reverse=True)[: args.worst_count]

    report = {
        "samples_csv": str(samples_csv),
        "extrinsics": str(extrinsics),
        "used_samples": len(records),
        "origin_xy_calibration_in_ros": list(origin_xy),
        "truth_z_mode": args.truth_z_mode,
        "water_surface_z_m": float(args.water_surface_z_m),
        "current_t_world_camera": _transform_summary(t_world_camera),
        "current": _summarize_errors(current_errors_np),
        "fit_t_world_camera": _transform_summary(t_world_camera_fit),
        "fit": _summarize_errors(fit_errors),
        "delta_fit_from_current": delta_summary,
        "worst_current_samples": [
            {
                **item,
                "truth_ros_xyz_m": item["truth_ros_xyz_m"].tolist(),
                "current_pred_xyz_m": item["current_pred_xyz_m"].tolist(),
                "current_error_xyz_m": item["current_error_xyz_m"].tolist(),
            }
            for item in worst
        ],
        "worst_truth_points": group_summaries[: args.worst_count],
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate T_world_camera against an AprilTag PnP truth dataset."
    )
    parser.add_argument(
        "--dataset",
        required=True,
        help="Dataset session directory or samples.csv path.",
    )
    parser.add_argument(
        "--extrinsics",
        default="src/state_estimation/config/state_fusion_extrinsics.yaml",
        help="YAML containing T_world_camera.",
    )
    parser.add_argument(
        "--origin-xy",
        nargs=2,
        type=float,
        default=(2.0, 1.0),
        metavar=("X", "Y"),
        help="Calibration-frame coordinate of the current ROS/pool_world origin.",
    )
    parser.add_argument(
        "--truth-z-mode",
        choices=("depth_from_surface", "z_up", "negative_depth", "depth_down"),
        default="depth_from_surface",
        help=(
            "How to convert truth_z_m to ROS z. depth_from_surface means "
            "ros_z=water_surface_z_m-truth_z_m. negative_depth/depth_down "
            "keeps the old ros_z=-truth_z_m convention."
        ),
    )
    parser.add_argument(
        "--water-surface-z-m",
        type=float,
        default=1.0,
        help="Water-surface height in bottom-origin pool_world, used by truth-z-mode=depth_from_surface.",
    )
    parser.add_argument(
        "--max-reprojection-error-px",
        type=float,
        default=None,
        help="Discard samples above this stored PnP reprojection error.",
    )
    parser.add_argument("--worst-count", type=int, default=8)
    parser.add_argument("--json-output", default="")
    args = parser.parse_args()

    report = evaluate(args)
    _print_summary(report)
    if args.json_output:
        output = Path(args.json_output).expanduser()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print()
        print(f"wrote JSON report: {output}")


if __name__ == "__main__":
    main()
