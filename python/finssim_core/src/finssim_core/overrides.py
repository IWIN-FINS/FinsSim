from __future__ import annotations

import dataclasses
from dataclasses import fields, is_dataclass
from typing import Any, TypeVar

T = TypeVar("T")


def merge_config_layers(*layers: dict[str, Any] | None) -> dict[str, Any]:
    """Merge config dictionaries from lowest to highest priority."""
    merged: dict[str, Any] = {}
    for layer in layers:
        if not layer:
            continue
        merged.update(layer)
    return merged


def apply_dataclass_overrides(instance: T, overrides: dict[str, Any], *, strict: bool = True) -> T:
    """Return a dataclass copy with same-name fields overridden."""
    if not is_dataclass(instance):
        raise TypeError("apply_dataclass_overrides expects a dataclass instance")

    field_names = {field.name for field in fields(instance)}
    unknown = sorted(set(overrides) - field_names)
    if unknown and strict:
        raise ValueError(f"Unknown override fields: {', '.join(unknown)}")

    selected = {key: value for key, value in overrides.items() if key in field_names}
    return dataclasses.replace(instance, **selected)
