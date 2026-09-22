from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


def _as_point_array(values: Any, *, dims: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != dims:
        raise ValueError(f"expected Nx{dims} point array, got shape {array.shape}")
    return array


def load_pixel_to_world_homography(path: str | Path) -> np.ndarray:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if "pixel_to_world_homography" in data:
        matrix = np.asarray(data["pixel_to_world_homography"], dtype=np.float64)
        if matrix.shape == (9,):
            matrix = matrix.reshape(3, 3)
        if matrix.shape != (3, 3):
            raise ValueError("pixel_to_world_homography must be a 3x3 matrix or 9 values")
        return matrix

    if "image_points" in data and "world_points" in data:
        image_points = _as_point_array(data["image_points"], dims=2)
        world_points = _as_point_array(data["world_points"], dims=2)
        if image_points.shape[0] < 4 or world_points.shape[0] < 4:
            raise ValueError("at least 4 image/world points are required for homography")
        if image_points.shape[0] != world_points.shape[0]:
            raise ValueError("image_points and world_points must have the same length")
        homography, mask = cv2.findHomography(image_points, world_points, method=0)
        if homography is None or mask is None:
            raise ValueError(f"failed to compute homography from {path}")
        return homography.astype(np.float64, copy=False)

    raise ValueError(f"{path} must contain pixel_to_world_homography or image_points/world_points")


def pixel_to_world_xy(homography: np.ndarray, pixel_xy: tuple[float, float] | np.ndarray) -> tuple[float, float]:
    h = np.asarray(homography, dtype=np.float64).reshape(3, 3)
    p = np.asarray([float(pixel_xy[0]), float(pixel_xy[1]), 1.0], dtype=np.float64)
    world = h @ p
    if abs(float(world[2])) < 1e-12:
        raise ValueError("homogeneous world scale is zero")
    world = world / world[2]
    return float(world[0]), float(world[1])
