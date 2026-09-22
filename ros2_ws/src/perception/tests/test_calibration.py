import numpy as np
import yaml

from perception.calibration import load_pixel_to_world_homography, pixel_to_world_xy


def test_pixel_to_world_xy_identity():
    h = np.eye(3)
    assert pixel_to_world_xy(h, (2.0, 3.0)) == (2.0, 3.0)


def test_load_homography_from_point_pairs(tmp_path):
    path = tmp_path / "homography.yaml"
    data = {
        "image_points": [[0, 0], [10, 0], [10, 10], [0, 10]],
        "world_points": [[1, 2], [3, 2], [3, 6], [1, 6]],
    }
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    h = load_pixel_to_world_homography(path)
    assert pixel_to_world_xy(h, (5.0, 5.0)) == pytest_approx_tuple((2.0, 4.0))


def pytest_approx_tuple(values):
    import pytest

    return pytest.approx(values)
