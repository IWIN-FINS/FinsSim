"""Small registry helpers for FinsSim BenchMARL integration."""

from __future__ import annotations

from .tasks import FinsSimBenchMARLTask


def available_tasks() -> list[str]:
    """Return stable task names exposed by the FinsSim BenchMARL integration."""
    return [task.name.lower() for task in FinsSimBenchMARLTask]
