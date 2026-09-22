from pathlib import Path

import pytest

from experiment_recorder.apriltag_localization import _load_samples


def test_e2_truth_converts_outer_lower_left_to_tag_center(tmp_path: Path):
    """E2 evaluates the same tag-centre point published by PnP/Snell."""

    source = tmp_path / "session"
    image_dir = source / "images" / "raw"
    image_dir.mkdir(parents=True)
    (image_dir / "sample_001.png").touch()
    (source / "samples.csv").write_text(
        "sample_id,truth_x_m,truth_y_m\n"
        "sample_001,0.25,-0.40\n",
        encoding="utf-8",
    )

    samples = _load_samples(
        source,
        "samples.csv",
        {"x_scale": 1.0, "x_offset_m": -2.0, "y_scale": 1.0, "y_offset_m": -1.0},
        {"x_offset_m": 0.06, "y_offset_m": 0.06, "tag_outer_side_m": 0.12},
        limit=None,
    )

    assert len(samples) == 1
    assert samples[0].annotation_truth_x_m == 0.25
    assert samples[0].annotation_truth_y_m == -0.40
    assert samples[0].truth_x_m == pytest.approx(-1.69)
    assert samples[0].truth_y_m == pytest.approx(-1.34)
