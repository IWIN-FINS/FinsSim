from pathlib import Path

import pytest

from experiment_recorder.top_camera_video import VideoRecordingSpec, metadata_path_for_video


def test_video_spec_requires_an_absolute_raw_topic_and_mp4_output(tmp_path: Path) -> None:
    spec = VideoRecordingSpec.create(
        image_topic="/finsrov/camera/raw/compressed",
        output_path=tmp_path / "video" / "top_camera_raw.mp4",
        fps=10.0,
        codec="mp4v",
    )
    assert spec.output_path == (tmp_path / "video" / "top_camera_raw.mp4").resolve()
    assert metadata_path_for_video(spec.output_path).name == "top_camera_raw.metadata.json"

    with pytest.raises(ValueError, match="absolute"):
        VideoRecordingSpec.create(
            image_topic="finsrov/camera/raw/compressed",
            output_path=tmp_path / "video.mp4",
            fps=10.0,
            codec="mp4v",
        )
    with pytest.raises(ValueError, match=".mp4"):
        VideoRecordingSpec.create(
            image_topic="/finsrov/camera/raw/compressed",
            output_path=tmp_path / "video.avi",
            fps=10.0,
            codec="mp4v",
        )


def test_video_spec_rejects_invalid_fps_and_fourcc(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="positive finite"):
        VideoRecordingSpec.create(
            image_topic="/finsrov/camera/raw/compressed",
            output_path=tmp_path / "video.mp4",
            fps=0.0,
            codec="mp4v",
        )
    with pytest.raises(ValueError, match="four-character"):
        VideoRecordingSpec.create(
            image_topic="/finsrov/camera/raw/compressed",
            output_path=tmp_path / "video.mp4",
            fps=10.0,
            codec="h2640",
        )
