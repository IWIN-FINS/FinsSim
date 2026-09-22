from pathlib import Path

from experiment_recorder.apriltag_utils import next_repeat, position_id


def test_position_id_is_stable_and_signed():
    assert position_id(0.25, -0.4) == "P_XP0p250_YN0p400"


def test_next_repeat_counts_matching_truth_rows(tmp_path: Path):
    session = tmp_path / "20260822_120000_E1" / "external_truth"
    session.mkdir(parents=True)
    (session / "truth_xy.csv").write_text(
        "position_id,repeat,truth_x_m,truth_y_m,truth_frame,measurement_method,uncertainty_m,notes\n"
        "P01,1,0.250,-0.400,pool_world,survey_grid,0.005,\n",
        encoding="utf-8",
    )
    assert next_repeat(tmp_path, 0.25, -0.4) == 2
    assert next_repeat(tmp_path, 0.8, -0.4) == 1
