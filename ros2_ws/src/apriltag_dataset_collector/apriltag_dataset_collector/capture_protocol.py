"""Shared request parsing and serialization for native AprilTag dataset capture."""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any


TRUTH_FIELDS = ("x_m", "y_m", "z_m", "roll_deg", "pitch_deg", "yaw_deg")


def parse_nullable_float(value: str | float | int | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (float, int)):
        return float(value)
    normalized = value.strip()
    if not normalized or normalized.lower() in {"none", "null"}:
        return None
    return float(normalized)


def parse_optional_tag_id(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    normalized = value.strip()
    return int(normalized) if normalized else None


def build_capture_request(
    *,
    session_id: str,
    operator_name: str = "",
    tag_id: str | int | None = None,
    truth: dict[str, str | float | int | None] | None = None,
    note: str = "",
    request_id: str | None = None,
) -> dict[str, Any]:
    truth = truth or {}
    request: dict[str, Any] = {
        "request_id": request_id or uuid.uuid4().hex,
        "session_id": session_id.strip() or time.strftime("%Y%m%d_%H%M%S"),
        "operator": operator_name.strip(),
        "truth_world": {field: parse_nullable_float(truth.get(field)) for field in TRUTH_FIELDS},
        "note": note.strip(),
    }
    selected_tag = parse_optional_tag_id(tag_id)
    if selected_tag is not None:
        request["tag_id"] = selected_tag
    return request


def request_json(request: dict[str, Any]) -> str:
    return json.dumps(request, separators=(",", ":"), allow_nan=False)


def normalize_dataset_root(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("dataset root must not be empty")
    return str(Path(normalized).expanduser())
