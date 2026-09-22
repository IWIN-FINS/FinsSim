import pytest

from motion_control.goal_sender import _build_goal_parser


def test_position_goal_requires_controller_world_or_body_frame():
    parser = _build_goal_parser()

    assert parser.parse_args(["--frame", "controller_world", "--x", "1", "--y", "0", "--z", "0"]).frame == (
        "controller_world"
    )
    assert parser.parse_args(["--frame", "controller_body", "--x", "1", "--y", "0", "--z", "0"]).frame == (
        "controller_body"
    )
    with pytest.raises(SystemExit):
        parser.parse_args(["--frame", "world", "--x", "1", "--y", "0", "--z", "0"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--frame", "body", "--x", "1", "--y", "0", "--z", "0"])
