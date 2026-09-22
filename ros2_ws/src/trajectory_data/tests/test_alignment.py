import pytest

from trajectory_data.alignment import TimedRecord, bracket_records, interpolate_payload, previous_record, slerp
from trajectory_data.analyzer import _pool_bounds_report
from trajectory_data.session import prepare_session_directory, session_directory


def test_command_hold_never_uses_future_command():
    records = [
        TimedRecord(1.0, {"command": [1.0]}, 1.0),
        TimedRecord(2.0, {"command": [2.0]}, 2.0),
    ]
    assert previous_record(records, 1.5).payload["command"] == [1.0]
    assert previous_record(records, 0.9) is None


def test_bracket_records_handles_the_final_sample():
    records = [
        TimedRecord(1.0, {"sample": 1}, 1.0),
        TimedRecord(2.0, {"sample": 2}, 2.0),
        TimedRecord(3.0, {"sample": 3}, 3.0),
    ]
    first, second = bracket_records(records, 3.0)
    assert (first.time_sec, second.time_sec) == (2.0, 3.0)


def test_linear_interpolation_and_nearest_age():
    records = [
        TimedRecord(0.0, {"velocity": [0.0, 2.0, 4.0]}, 0.0),
        TimedRecord(1.0, {"velocity": [2.0, 4.0, 6.0]}, 1.0),
    ]
    payload, age = interpolate_payload(records, 0.25, vector_fields=("velocity",))
    assert payload is not None
    assert payload["velocity"] == [0.5, 2.5, 4.5]
    assert age == 0.25


def test_slerp_normalizes_interpolated_quaternion():
    quaternion = slerp([0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 1.0, 0.0], 0.5)
    assert abs(sum(value * value for value in quaternion) - 1.0) < 1e-8


def test_overwrite_replaces_only_the_requested_session_directory(tmp_path):
    output_root = tmp_path / "sessions"
    session_dir = session_directory(output_root, "trial_001")
    prepare_session_directory(session_dir, overwrite=False)
    (session_dir / "old.txt").write_text("old", encoding="utf-8")
    sibling = output_root / "trial_002"
    sibling.mkdir(parents=True)
    (sibling / "keep.txt").write_text("keep", encoding="utf-8")

    prepare_session_directory(session_dir, overwrite=True)

    assert session_dir.is_dir()
    assert not (session_dir / "old.txt").exists()
    assert (sibling / "keep.txt").read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("session_id", ("", ".", "..", "../outside", "/tmp/outside", "nested/name"))
def test_session_id_must_be_a_single_directory_name(tmp_path, session_id):
    with pytest.raises(ValueError):
        session_directory(tmp_path, session_id)


def test_pool_bounds_only_enforces_y_lower_bound():
    records = [
        TimedRecord(1.0, {"position_xyz": [0.0, 0.5, 0.0]}, 1.0),
        TimedRecord(2.0, {"position_xyz": [2.2, -1.2, 1.1]}, 2.0),
    ]

    report, events = _pool_bounds_report(records, "fused_pose")

    assert report["in_bounds_samples"] == 1
    assert report["out_of_bounds_samples"] == 1
    assert report["by_axis"]["y"]["above_max_samples"] == 0
    assert set(events[0]["violated_axes"]) == {"x", "y", "z"}
