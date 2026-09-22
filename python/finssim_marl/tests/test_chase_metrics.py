import numpy as np
import pytest

from finssim_marl.utils.chase_metrics import (
    nearest_chaser_prey_distance_m,
    summarize_episode_distances,
)


def test_nearest_chaser_prey_distance_uses_only_chaser_relative_positions():
    observations = np.zeros((2, 4, 30), dtype=np.float32)
    observations[0, 0, 6:9] = (3.0, 4.0, 0.0)
    observations[0, 1, 6:9] = (1.0, 2.0, 2.0)
    observations[0, 2, 6:9] = (0.5, 0.0, 0.0)
    observations[0, 3, 6:9] = (0.01, 0.0, 0.0)  # Prey's layout is unrelated.
    observations[1, 0, 6:9] = (6.0, 8.0, 0.0)
    observations[1, 1, 6:9] = (0.0, 0.0, 3.0)
    observations[1, 2, 6:9] = (0.0, 4.0, 0.0)

    distances = nearest_chaser_prey_distance_m(observations, role_ids=(0, 1, 1, 2))

    assert np.allclose(distances, (0.5, 3.0))


def test_summarize_episode_distances_reports_mean_distance_progress():
    metrics = summarize_episode_distances(
        initial_distance_m=(8.0, 6.0),
        min_distance_m=(1.0, 2.0),
        final_distance_m=(2.0, 3.0),
    )

    assert metrics == {
        "initial_nearest_chaser_distance_to_prey_m": 7.0,
        "min_nearest_chaser_distance_to_prey_m": 1.5,
        "final_nearest_chaser_distance_to_prey_m": 2.5,
        "nearest_chaser_distance_reduction_m": 4.5,
    }


def test_nearest_chaser_prey_distance_rejects_missing_chasers():
    with pytest.raises(ValueError, match="finite chaser"):
        nearest_chaser_prey_distance_m(np.zeros((1, 1, 30)), role_ids=(2,))
