from __future__ import annotations

import numpy as np

from finssim_rl.models.traditional_trajectory_tracking import (
    T2_ALLOCATOR_BODY_WRENCH_LIMITS,
    TraditionalTrajectoryTrackingTrainingConfig,
    TraditionalTrajectoryTrackingWrenchModel,
    as_trajectory_observation_batch,
)
from finssim_rl.training.config import get_config


def test_trajectory_baseline_config_is_registered_without_checkpoint():
    config = get_config("traditional_trajectory_tracking_wrench_pd")

    assert config.requires_checkpoint is False
    assert config.model_type == "TRADITIONAL_TRAJECTORY_TRACKING_WRENCH_PD"
    assert config.env_config.parallel_mode == "multi_area"


def test_trajectory_preview_pd_uses_unity_30d_layout_and_allocator():
    config = TraditionalTrajectoryTrackingTrainingConfig(
        position_kp=(2.0, 3.0, 4.0),
        velocity_kd=(5.0, 6.0, 7.0),
        yaw_kp=2.0,
        yaw_rate_kd=0.0,
        output_slew_rate_limit=0.0,
    )
    model = TraditionalTrajectoryTrackingWrenchModel(config, device="cpu")
    observation = np.zeros(30, dtype=np.float32)
    observation[0:3] = (0.5, -0.25, 0.125)  # Preview offset normalized by 3 m.
    observation[12:15] = (0.4, 0.2, -0.1)  # Desired body velocity.
    observation[15:18] = (0.1, -0.3, 0.5)  # Measured body velocity.
    observation[21:24] = (0.0, 1.0, 0.0)
    observation[24:26] = (1.0, 0.0)  # Reference tangent is body-left; yaw command is right.

    action, _ = model.predict(observation)

    np.testing.assert_allclose(
        model.last_diagnostics["wrench"],
        np.array([4.5, 0.75, -2.7, 0.0, -T2_ALLOCATOR_BODY_WRENCH_LIMITS[4], 0.0], dtype=np.float32),
        atol=1e-6,
    )
    assert action.shape == (8,)
    assert np.all(np.isfinite(action))
    assert np.all(np.abs(action) <= 1.0)


def test_trajectory_baseline_rejects_non_t2_observation_shape():
    try:
        as_trajectory_observation_batch(np.zeros(16, dtype=np.float32))
    except ValueError as exc:
        assert "30D" in str(exc)
    else:
        raise AssertionError("expected 16D HoldForPosition observation to be rejected")
