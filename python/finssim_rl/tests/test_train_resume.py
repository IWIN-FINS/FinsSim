from types import SimpleNamespace

from scripts.train import _resolve_training_steps


def test_resume_only_trains_steps_remaining_to_configured_target() -> None:
    args = SimpleNamespace(resume=True)

    assert _resolve_training_steps(args, 100_000_000, 37_500_000) == 62_500_000


def test_resume_at_or_past_configured_target_does_not_train_more() -> None:
    args = SimpleNamespace(resume=True)

    assert _resolve_training_steps(args, 100_000_000, 100_000_000) == 0
    assert _resolve_training_steps(args, 100_000_000, 120_000_000) == 0


def test_init_checkpoint_keeps_an_explicit_invocation_budget() -> None:
    args = SimpleNamespace(resume=False)

    assert _resolve_training_steps(args, 10_000_000, 37_500_000) == 10_000_000
