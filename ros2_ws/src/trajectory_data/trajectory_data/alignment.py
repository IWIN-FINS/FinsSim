from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
import math
from typing import Any, Iterable, Sequence

import numpy as np


def stamp_to_sec(stamp: Any) -> float:
    return float(stamp.sec) + float(stamp.nanosec) * 1e-9


def safe_json(value: str) -> dict[str, Any]:
    import json

    try:
        decoded = json.loads(value)
    except (TypeError, ValueError):
        return {}
    return decoded if isinstance(decoded, dict) else {}


def slerp(first: Sequence[float], second: Sequence[float], alpha: float) -> list[float]:
    first_q = np.asarray(first, dtype=float)
    second_q = np.asarray(second, dtype=float)
    first_norm = float(np.linalg.norm(first_q))
    second_norm = float(np.linalg.norm(second_q))
    if first_norm <= 1e-12 or second_norm <= 1e-12:
        return [float(value) for value in first_q]
    first_q /= first_norm
    second_q /= second_norm
    dot = float(np.clip(np.dot(first_q, second_q), -1.0, 1.0))
    if dot < 0.0:
        second_q = -second_q
        dot = -dot
    if dot > 0.9995:
        result = first_q + float(alpha) * (second_q - first_q)
        result /= max(float(np.linalg.norm(result)), 1e-12)
        return [float(value) for value in result]
    theta = math.acos(dot)
    sin_theta = math.sin(theta)
    first_weight = math.sin((1.0 - float(alpha)) * theta) / sin_theta
    second_weight = math.sin(float(alpha) * theta) / sin_theta
    return [float(value) for value in first_weight * first_q + second_weight * second_q]


def linear(first: Sequence[float], second: Sequence[float], alpha: float) -> list[float]:
    first_values = np.asarray(first, dtype=float)
    second_values = np.asarray(second, dtype=float)
    return [float(value) for value in first_values + float(alpha) * (second_values - first_values)]


@dataclass(frozen=True)
class TimedRecord:
    time_sec: float
    payload: dict[str, Any]
    bag_time_sec: float


def sorted_records(records: Iterable[TimedRecord]) -> list[TimedRecord]:
    return sorted(records, key=lambda record: record.time_sec)


def previous_record(records: Sequence[TimedRecord], time_sec: float) -> TimedRecord | None:
    index = bisect_right(records, time_sec, key=lambda record: record.time_sec) - 1
    return records[index] if index >= 0 else None


def bracket_records(records: Sequence[TimedRecord], time_sec: float) -> tuple[TimedRecord, TimedRecord] | None:
    if len(records) < 2 or time_sec < records[0].time_sec or time_sec > records[-1].time_sec:
        return None
    right_index = bisect_right(records, time_sec, key=lambda record: record.time_sec)
    if right_index >= len(records):
        return records[-2], records[-1]
    return records[right_index - 1], records[right_index]


def interpolate_payload(
    records: Sequence[TimedRecord],
    time_sec: float,
    *,
    vector_fields: Sequence[str],
    quaternion_fields: Sequence[str] = (),
) -> tuple[dict[str, Any] | None, float]:
    bracket = bracket_records(records, time_sec)
    if bracket is None:
        nearest = previous_record(records, time_sec)
        if nearest is None:
            return None, math.inf
        return dict(nearest.payload), abs(time_sec - nearest.time_sec)
    first, second = bracket
    span = max(second.time_sec - first.time_sec, 1e-12)
    alpha = (time_sec - first.time_sec) / span
    result = dict(first.payload)
    for field in vector_fields:
        if field in first.payload and field in second.payload:
            result[field] = linear(first.payload[field], second.payload[field], alpha)
    for field in quaternion_fields:
        if field in first.payload and field in second.payload:
            result[field] = slerp(first.payload[field], second.payload[field], alpha)
    return result, min(abs(time_sec - first.time_sec), abs(second.time_sec - time_sec))


def rate_summary(records: Sequence[TimedRecord]) -> dict[str, float | int | None]:
    if len(records) < 2:
        return {"samples": len(records), "rate_hz": None, "period_median_sec": None, "period_p95_sec": None}
    periods = np.diff(np.asarray([record.time_sec for record in records], dtype=float))
    periods = periods[periods > 0.0]
    if periods.size == 0:
        return {"samples": len(records), "rate_hz": None, "period_median_sec": None, "period_p95_sec": None}
    return {
        "samples": len(records),
        "rate_hz": float(1.0 / np.median(periods)),
        "period_median_sec": float(np.median(periods)),
        "period_p95_sec": float(np.quantile(periods, 0.95)),
    }
