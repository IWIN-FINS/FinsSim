from __future__ import annotations

from types import SimpleNamespace

import pytest

from finssim_rl.training.utils import (
    apply_continuation_learning_rate_schedule,
    cosine_warm_restarts_schedule,
)


def test_cosine_warm_restarts_uses_continuation_local_progress() -> None:
    schedule = cosine_warm_restarts_schedule(
        1e-4,
        1e-5,
        start_progress_remaining=0.2,
        cycles=2,
    )

    assert schedule(0.2) == pytest.approx(1e-4)
    assert schedule(0.15) == pytest.approx(5.5e-5)
    assert schedule(0.1) == pytest.approx(1e-4)
    assert schedule(0.0) == pytest.approx(1e-5)


def test_apply_continuation_schedule_replaces_checkpoint_schedule() -> None:
    model = SimpleNamespace(
        num_timesteps=16_384_000,
        learning_rate=None,
        lr_schedule=lambda _: 1e-5,
        policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 1e-5}])),
    )
    config = SimpleNamespace(
        learning_rate_schedule="cosine_warm_restarts",
        learning_rate_peak=1e-4,
        learning_rate_min=1e-5,
        learning_rate_warm_restart_cycles=2,
        total_timesteps=3_276_800,
    )

    apply_continuation_learning_rate_schedule(model, config)

    start_progress = 3_276_800 / (16_384_000 + 3_276_800)
    assert model.lr_schedule(start_progress) == pytest.approx(1e-4)
    assert model.policy.optimizer.param_groups[0]["lr"] == pytest.approx(1e-4)


def test_continuation_schedule_can_use_resume_steps_remaining_to_target() -> None:
    model = SimpleNamespace(
        num_timesteps=37_500_000,
        learning_rate=None,
        lr_schedule=lambda _: 1e-5,
        policy=SimpleNamespace(optimizer=SimpleNamespace(param_groups=[{"lr": 1e-5}])),
    )
    config = SimpleNamespace(
        learning_rate_schedule="cosine_warm_restarts",
        learning_rate_peak=1e-4,
        learning_rate_min=1e-5,
        learning_rate_warm_restart_cycles=2,
        total_timesteps=100_000_000,
    )

    apply_continuation_learning_rate_schedule(
        model,
        config,
        additional_steps=62_500_000,
    )

    start_progress = 62_500_000 / 100_000_000
    assert model.lr_schedule(start_progress) == pytest.approx(1e-4)
