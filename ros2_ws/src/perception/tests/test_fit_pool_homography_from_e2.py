import numpy as np

from perception.calibration_tools.fit_pool_homography_from_e2 import _apply, _fit, _metrics, PointPair


def test_fit_recovers_projective_mapping_and_reports_heldout_error():
    expected = np.array(
        [[0.002, -0.0003, -1.8], [0.0002, -0.0015, 0.8], [0.000001, -0.000002, 1.0]], dtype=np.float64
    )
    image = np.array([[100, 100], [1000, 100], [100, 600], [1000, 600], [600, 350]], dtype=np.float64)
    world = _apply(expected, image)
    fitted = _fit(image[:4], world[:4])
    assert np.allclose(_apply(fitted, image), world, atol=1e-9)
    pairs = [
        PointPair(0.0, 0.0, float(target[0]), float(target[1]), float(pixel[0]), float(pixel[1]), 1, 0, 0, "heldout")
        for pixel, target in zip(image, world)
    ]
    assert _metrics(fitted, pairs)["rmse_m"] < 1e-7
