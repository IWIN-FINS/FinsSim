import pytest

from motion_control.controller_node import _update_goal_hold_state, _update_stuck_hold_state


def test_goal_hold_resets_when_outside_tolerance():
    reached, entered_sec, held_duration_sec = _update_goal_hold_state(
        within_tolerance=False,
        now_sec=12.0,
        entered_sec=5.0,
        hold_time_sec=10.0,
    )

    assert reached is False
    assert entered_sec is None
    assert held_duration_sec == 0.0


def test_goal_hold_starts_countdown_on_first_within_tolerance_sample():
    reached, entered_sec, held_duration_sec = _update_goal_hold_state(
        within_tolerance=True,
        now_sec=12.0,
        entered_sec=None,
        hold_time_sec=10.0,
    )

    assert reached is False
    assert entered_sec == 12.0
    assert held_duration_sec == 0.0


def test_goal_hold_requires_full_hold_duration():
    reached, entered_sec, held_duration_sec = _update_goal_hold_state(
        within_tolerance=True,
        now_sec=22.1,
        entered_sec=12.0,
        hold_time_sec=10.0,
    )

    assert reached is True
    assert entered_sec == 12.0
    assert held_duration_sec == pytest.approx(10.1)


def test_stuck_hold_resets_when_candidate_is_false():
    stuck, entered_sec, best_error, held_duration_sec = _update_stuck_hold_state(
        candidate=False,
        now_sec=5.0,
        position_error=0.3,
        entered_sec=1.0,
        best_error=0.2,
        progress_epsilon_m=0.02,
        trigger_duration_sec=2.0,
    )

    assert stuck is False
    assert entered_sec is None
    assert best_error is None
    assert held_duration_sec == 0.0


def test_stuck_hold_resets_timer_on_real_progress():
    stuck, entered_sec, best_error, held_duration_sec = _update_stuck_hold_state(
        candidate=True,
        now_sec=5.0,
        position_error=0.15,
        entered_sec=1.0,
        best_error=0.2,
        progress_epsilon_m=0.02,
        trigger_duration_sec=2.0,
    )

    assert stuck is False
    assert entered_sec == 5.0
    assert best_error == 0.15
    assert held_duration_sec == 0.0


def test_stuck_hold_triggers_after_duration_without_progress():
    stuck, entered_sec, best_error, held_duration_sec = _update_stuck_hold_state(
        candidate=True,
        now_sec=3.2,
        position_error=0.21,
        entered_sec=1.0,
        best_error=0.2,
        progress_epsilon_m=0.02,
        trigger_duration_sec=2.0,
    )

    assert stuck is True
    assert entered_sec == 1.0
    assert best_error == 0.2
    assert held_duration_sec == pytest.approx(2.2)
