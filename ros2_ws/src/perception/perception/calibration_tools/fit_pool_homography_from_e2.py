"""Fit a fixed-plane pool homography from an audited E2 native replay.

The source annotation is kept immutable.  This tool maps its recorded
AprilTag outer-lower-left grid coordinate to the AprilTag centre before
fitting the pixel-to-pool-world homography used by ``direct_apriltag_node``.
It fits only a 2-D fixed-depth diagnostic map; it does not modify camera
intrinsics, the refractive model, or ``T_world_camera``.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
from typing import Iterable, Sequence

import cv2
import numpy as np
import yaml

from perception.calibration import load_pixel_to_world_homography


@dataclass(frozen=True)
class PointPair:
    annotation_x_m: float
    annotation_y_m: float
    truth_x_m: float
    truth_y_m: float
    pixel_u: float
    pixel_v: float
    image_count: int
    grid_ix: int
    grid_iy: int
    split: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float(row: dict[str, str], field: str) -> float:
    try:
        return float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"source samples.csv has invalid {field!r}: {row.get(field)!r}") from exc


def _load_detector_pixels(events_path: Path) -> dict[int, tuple[float, float]]:
    pixels: dict[int, tuple[float, float]] = {}
    with events_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid JSON in {events_path}:{line_number}: {exc}") from exc
            if event.get("stream") != "native_detector_status":
                continue
            index = event.get("replay_index")
            pixel = event.get("pixel_xy")
            if not isinstance(index, int) or not isinstance(pixel, list) or len(pixel) != 2:
                continue
            try:
                value = (float(pixel[0]), float(pixel[1]))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(value).all():
                continue
            if index in pixels:
                raise SystemExit(f"duplicate native detector replay_index {index} in {events_path}")
            pixels[index] = value
    if not pixels:
        raise SystemExit(f"no native_detector_status pixel_xy observations found in {events_path}")
    return pixels


def _grid_index(value: float, origin: float, step: float, label: str) -> int:
    index = int(round((value - origin) / step))
    if not np.isclose(value, origin + index * step, rtol=0.0, atol=step * 1e-3):
        raise SystemExit(
            f"annotation {label}={value:.9f} does not match the configured grid origin={origin:.9f}, step={step:.9f}"
        )
    return index


def _split_name(ix: int, iy: int, modulus: int, heldout_remainder: int) -> str:
    return "heldout" if (ix + 2 * iy) % modulus == heldout_remainder else "train"


def _build_point_pairs(
    samples_path: Path,
    pixels: dict[int, tuple[float, float]],
    *,
    grid_to_pool_offset: tuple[float, float],
    tag_center_offset: tuple[float, float],
    grid_origin: tuple[float, float],
    grid_step_m: float,
    heldout_modulus: int,
    heldout_remainder: int,
) -> list[PointPair]:
    if grid_step_m <= 0.0:
        raise SystemExit("grid-step-m must be positive")
    if heldout_modulus < 2 or not 0 <= heldout_remainder < heldout_modulus:
        raise SystemExit("heldout split must use modulus >= 2 and a remainder in [0, modulus)")

    groups: dict[tuple[float, float], list[tuple[float, float]]] = {}
    with samples_path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise SystemExit(f"source samples file is empty: {samples_path}")
    missing = [index for index in range(len(rows)) if index not in pixels]
    if missing:
        preview = ", ".join(str(value) for value in missing[:8])
        raise SystemExit(f"native detector pixels are missing for {len(missing)} replay frames (first: {preview})")

    for replay_index, row in enumerate(rows):
        annotation = (_as_float(row, "truth_x_m"), _as_float(row, "truth_y_m"))
        groups.setdefault(annotation, []).append(pixels[replay_index])

    pairs: list[PointPair] = []
    for annotation, observation_pixels in sorted(groups.items()):
        ix = _grid_index(annotation[0], grid_origin[0], grid_step_m, "x")
        iy = _grid_index(annotation[1], grid_origin[1], grid_step_m, "y")
        center_truth = (
            annotation[0] + grid_to_pool_offset[0] + tag_center_offset[0],
            annotation[1] + grid_to_pool_offset[1] + tag_center_offset[1],
        )
        pixel_array = np.asarray(observation_pixels, dtype=np.float64)
        pairs.append(
            PointPair(
                annotation_x_m=annotation[0],
                annotation_y_m=annotation[1],
                truth_x_m=center_truth[0],
                truth_y_m=center_truth[1],
                pixel_u=float(np.median(pixel_array[:, 0])),
                pixel_v=float(np.median(pixel_array[:, 1])),
                image_count=len(observation_pixels),
                grid_ix=ix,
                grid_iy=iy,
                split=_split_name(ix, iy, heldout_modulus, heldout_remainder),
            )
        )
    if len(pairs) < 8:
        raise SystemExit(f"need at least 8 distinct truth points, got {len(pairs)}")
    return pairs


def _arrays(pairs: Iterable[PointPair]) -> tuple[np.ndarray, np.ndarray]:
    selected = list(pairs)
    image = np.asarray([[pair.pixel_u, pair.pixel_v] for pair in selected], dtype=np.float64)
    world = np.asarray([[pair.truth_x_m, pair.truth_y_m] for pair in selected], dtype=np.float64)
    return image, world


def _fit(image: np.ndarray, world: np.ndarray) -> np.ndarray:
    if len(image) < 4:
        raise SystemExit("homography fitting requires at least four point pairs")
    homography, _ = cv2.findHomography(image, world, method=0)
    if homography is None:
        raise SystemExit("cv2.findHomography failed")
    return np.asarray(homography, dtype=np.float64)


def _apply(homography: np.ndarray, image: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([image, np.ones(len(image), dtype=np.float64)])
    projected = (homography @ homogeneous.T).T
    scale = projected[:, 2:3]
    if np.any(np.abs(scale) < 1e-12):
        raise SystemExit("homography produced a near-zero homogeneous scale")
    return projected[:, :2] / scale


def _metrics(homography: np.ndarray, pairs: Sequence[PointPair]) -> dict[str, object]:
    if not pairs:
        return {"n": 0, "rmse_m": None, "mean_error_m": None, "p95_error_m": None, "bias_x_m": None, "bias_y_m": None}
    image, truth = _arrays(pairs)
    predicted = _apply(homography, image)
    residual = predicted - truth
    errors = np.linalg.norm(residual, axis=1)
    return {
        "n": int(len(pairs)),
        "rmse_m": float(np.sqrt(np.mean(errors**2))),
        "mean_error_m": float(np.mean(errors)),
        "p95_error_m": float(np.quantile(errors, 0.95, method="linear")),
        "max_error_m": float(np.max(errors)),
        "bias_x_m": float(np.mean(residual[:, 0])),
        "bias_y_m": float(np.mean(residual[:, 1])),
    }


def _prediction_rows(homography: np.ndarray, pairs: Sequence[PointPair]) -> list[dict[str, object]]:
    image, truth = _arrays(pairs)
    predicted = _apply(homography, image)
    rows: list[dict[str, object]] = []
    for pair, estimate, target in zip(pairs, predicted, truth):
        dx = float(estimate[0] - target[0])
        dy = float(estimate[1] - target[1])
        rows.append(
            {
                **asdict(pair),
                "estimate_x_m": float(estimate[0]),
                "estimate_y_m": float(estimate[1]),
                "error_x_m": dx,
                "error_y_m": dy,
                "horizontal_error_m": float(np.hypot(dx, dy)),
            }
        )
    return rows


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_yaml(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit a candidate fixed-plane pool homography from an E2 native AprilTag replay."
    )
    parser.add_argument("--e2-session", required=True, type=Path, help="E2 run directory containing raw/ros_events.ndjson and input/source_samples.csv")
    parser.add_argument("--output-dir", required=True, type=Path, help="New directory for point pairs, predictions, and report")
    parser.add_argument("--output-yaml", required=True, type=Path, help="Candidate YAML written with the full-data homography")
    parser.add_argument("--grid-to-pool-offset", nargs=2, type=float, default=(-2.0, -1.0), metavar=("X_M", "Y_M"))
    parser.add_argument("--tag-center-offset", nargs=2, type=float, default=(0.06, 0.06), metavar=("X_M", "Y_M"))
    parser.add_argument("--tag-outer-side-m", type=float, default=0.12)
    parser.add_argument("--marker-length-m", type=float, default=0.098)
    parser.add_argument("--grid-origin", nargs=2, type=float, default=(0.2, 0.2), metavar=("X_M", "Y_M"))
    parser.add_argument("--grid-step-m", type=float, default=0.2)
    parser.add_argument("--heldout-modulus", type=int, default=5)
    parser.add_argument("--heldout-remainder", type=int, default=0)
    parser.add_argument("--current-homography", type=Path, default=None, help="Optional existing runtime YAML to evaluate on the held-out points")
    parser.add_argument("--overwrite", action="store_true", help="Replace only explicit output-dir and output-yaml targets")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    session = args.e2_session.resolve()
    events_path = session / "raw" / "ros_events.ndjson"
    samples_path = session / "input" / "source_samples.csv"
    if not events_path.is_file() or not samples_path.is_file():
        raise SystemExit("--e2-session must contain raw/ros_events.ndjson and input/source_samples.csv")
    output_dir = args.output_dir.resolve()
    output_yaml = args.output_yaml.resolve()
    if (output_dir.exists() or output_yaml.exists()) and not args.overwrite:
        raise SystemExit("output target already exists; choose a new target or pass --overwrite explicitly")
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=False)

    pixels = _load_detector_pixels(events_path)
    pairs = _build_point_pairs(
        samples_path,
        pixels,
        grid_to_pool_offset=tuple(args.grid_to_pool_offset),
        tag_center_offset=tuple(args.tag_center_offset),
        grid_origin=tuple(args.grid_origin),
        grid_step_m=float(args.grid_step_m),
        heldout_modulus=int(args.heldout_modulus),
        heldout_remainder=int(args.heldout_remainder),
    )
    train = [pair for pair in pairs if pair.split == "train"]
    heldout = [pair for pair in pairs if pair.split == "heldout"]
    if len(train) < 4 or len(heldout) < 4:
        raise SystemExit(f"split leaves too few pairs: train={len(train)}, heldout={len(heldout)}")

    train_homography = _fit(*_arrays(train))
    final_homography = _fit(*_arrays(pairs))
    metrics = {
        "train": _metrics(train_homography, train),
        "heldout": _metrics(train_homography, heldout),
        "full_data": _metrics(final_homography, pairs),
    }
    current_metrics: dict[str, object] | None = None
    current_homography_path: Path | None = None
    if args.current_homography is not None:
        current_homography_path = args.current_homography.resolve()
        current_metrics = _metrics(load_pixel_to_world_homography(current_homography_path), heldout)

    point_rows = [asdict(pair) for pair in pairs]
    prediction_rows = _prediction_rows(train_homography, train) + _prediction_rows(train_homography, heldout)
    _write_csv(output_dir / "point_pairs.csv", point_rows)
    _write_csv(output_dir / "heldout_predictions.csv", prediction_rows)

    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    report = {
        "schema_version": 1,
        "created_at": now,
        "purpose": "E2 fixed-bottom-plane pixel-to-pool-world homography calibration and held-out audit",
        "scope": {
            "output_dimensions": ["pool_world_x", "pool_world_y"],
            "tag_plane": "archived bottom plane; water depth 1 m",
            "not_valid_for": ["variable depth", "3D pose", "Snell pose", "state fusion", "yaw correction"],
        },
        "input": {
            "e2_session": str(session),
            "events_path": str(events_path),
            "events_sha256": _sha256(events_path),
            "samples_path": str(samples_path),
            "samples_sha256": _sha256(samples_path),
            "native_detector_frames": len(pixels),
            "unique_truth_positions": len(pairs),
        },
        "truth_definition": {
            "annotation_frame": "AprilTag outer-lower-left grid coordinate",
            "grid_to_pool_world_offset_m": list(args.grid_to_pool_offset),
            "tag_center_offset_m": list(args.tag_center_offset),
            "tag_outer_side_m": float(args.tag_outer_side_m),
            "marker_length_m": float(args.marker_length_m),
        },
        "split": {
            "method": "spatial_modulo",
            "grid_origin_m": list(args.grid_origin),
            "grid_step_m": float(args.grid_step_m),
            "heldout_modulus": int(args.heldout_modulus),
            "heldout_remainder": int(args.heldout_remainder),
            "train_point_count": len(train),
            "heldout_point_count": len(heldout),
        },
        "metrics_m": metrics,
        "current_runtime_homography_heldout_metrics_m": current_metrics,
        "final_pixel_to_world_homography": final_homography.reshape(-1).tolist(),
    }
    (output_dir / "fit_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    candidate = {
        "schema_version": 1,
        "description": "E2-calibrated fixed-bottom-plane AprilTag-center pixel-to-pool-world mapping.",
        "pixel_to_world_homography": [float(value) for value in final_homography.reshape(-1)],
        "calibration_scope": {
            "output_frame": "pool_world",
            "plane": "archived bottom AprilTag plane at 1 m water depth",
            "valid_for": "native direct_apriltag_node centre-pixel debug mapping only",
            "not_valid_for": "variable-depth localisation, refractive 6D pose, or state fusion",
        },
        "truth_reference": {
            "annotation": "AprilTag outer-lower-left grid point",
            "grid_to_pool_world_offset_m": [float(value) for value in args.grid_to_pool_offset],
            "tag_center_offset_m": [float(value) for value in args.tag_center_offset],
            "tag_outer_side_m": float(args.tag_outer_side_m),
            "marker_length_m": float(args.marker_length_m),
        },
        "provenance": {
            "e2_session": str(session),
            "fit_report": str((output_dir / "fit_report.json")),
            "source_samples_sha256": _sha256(samples_path),
            "native_events_sha256": _sha256(events_path),
            "fitted_at": now,
        },
        "heldout_validation_m": metrics["heldout"],
    }
    _write_yaml(output_yaml, candidate)
    print(json.dumps({"output_dir": str(output_dir), "candidate_yaml": str(output_yaml), "metrics_m": metrics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
