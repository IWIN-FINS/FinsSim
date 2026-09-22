"""BenchMARL experiment construction helpers.

This module keeps BenchMARL orchestration isolated from the existing hand-written
MAPPO runner. Full algorithm/model configuration is intentionally passed in by
callers so FinsSim config loading stays the source of truth.
"""

from __future__ import annotations

from typing import Any

from .tasks import build_task_from_config


def build_task(config: dict[str, Any]):
    """Build a BenchMARL task object for FinsSim."""
    return build_task_from_config(config)
