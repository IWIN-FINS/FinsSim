import numpy as np

from finssim_rl.training.one_chase_one_eval_metrics import OneChaseOneDistanceTracker


def _obs(distance: float) -> np.ndarray:
    value = np.zeros(14, dtype=np.float32)
    value[12] = distance
    return value


def test_tracker_uses_terminal_observation_for_final_distance() -> None:
    tracker = OneChaseOneDistanceTracker(num_envs=2)
    tracker.reset(np.stack([_obs(8.0), _obs(6.0)]))

    tracker.record_transition(0, np.stack([_obs(5.0), _obs(4.0)]), False)
    completed = tracker.record_transition(
        0,
        np.stack([_obs(9.0), _obs(3.0)]),
        True,
        {"terminal_observation": _obs(2.0)},
    )

    assert completed is not None
    assert completed.minimum_m == 2.0
    assert completed.final_m == 2.0
    assert tracker.summary() == {
        "eval/mean_min_distance_to_prey_m": 2.0,
        "eval/mean_final_distance_to_prey_m": 2.0,
        "eval/best_min_distance_to_prey_m": 2.0,
        "eval/worst_final_distance_to_prey_m": 2.0,
    }


def test_tracker_preserves_each_area_independently() -> None:
    tracker = OneChaseOneDistanceTracker(num_envs=2)
    tracker.reset(np.stack([_obs(7.0), _obs(9.0)]))

    tracker.record_transition(0, np.stack([_obs(4.0), _obs(3.0)]), False)
    tracker.record_transition(1, np.stack([_obs(4.0), _obs(3.0)]), False)
    first = tracker.record_transition(0, np.stack([_obs(1.0), _obs(2.0)]), True, {})
    second = tracker.record_transition(1, np.stack([_obs(1.0), _obs(2.0)]), True, {})

    assert first is not None and (first.minimum_m, first.final_m) == (1.0, 1.0)
    assert second is not None and (second.minimum_m, second.final_m) == (2.0, 2.0)
