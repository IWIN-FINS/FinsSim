"""Generic ML-Agents ``StatsSideChannel`` extraction for Unity environments."""

from __future__ import annotations

from typing import Any


def unity_stat_aggregation_kind(aggregation: Any) -> str:
    """Normalize ML-Agents aggregation enums and legacy integer values."""
    name = str(getattr(aggregation, "name", "")).replace("_", "").upper()
    if name == "MOSTRECENT":
        return "most_recent"
    if name == "SUM":
        return "sum"

    value = getattr(aggregation, "value", aggregation)
    try:
        numeric_value = int(value)
    except (TypeError, ValueError):
        return "mean"
    if numeric_value == 1:
        return "most_recent"
    if numeric_value == 2:
        return "sum"
    return "mean"


def find_unity_stats_channel(env: Any):
    """Find the StatsSideChannel attached to an environment or its wrapper."""
    visited: set[int] = set()
    current = env
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        channel = getattr(current, "_finssim_stats_channel", None)
        if channel is not None:
            return channel
        current = getattr(current, "venv", None)
    return None


def drain_unity_stats(env: Any) -> dict[str, float]:
    """Return aggregated custom Unity stats from one StatsSideChannel window.

    Metric names and aggregation behavior are defined by Unity. This adapter is
    intentionally task-agnostic so any current or future Unity task can publish
    its StatsRecorder keys without a trainer-side prefix whitelist.
    """
    channel = find_unity_stats_channel(env)
    if channel is None:
        return {}

    raw_stats = channel.get_and_reset_stats()
    metrics: dict[str, float] = {}
    for key, samples in raw_stats.items():
        if not samples:
            continue
        values = [float(value) for value, _aggregation in samples]
        aggregation = unity_stat_aggregation_kind(samples[-1][1])
        if aggregation == "most_recent":
            metrics[str(key)] = values[-1]
        elif aggregation == "sum":
            metrics[str(key)] = sum(values)
        else:  # Average and Histogram are represented by their scalar mean.
            metrics[str(key)] = sum(values) / len(values)
    return metrics


__all__ = [
    "drain_unity_stats",
    "find_unity_stats_channel",
    "unity_stat_aggregation_kind",
]
