from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """Backend-neutral experiment metadata from a FinsSim YAML file."""

    name: str
    seed: int = 42
    tags: tuple[str, ...] = field(default_factory=tuple)
    output_dir: str | None = None

    @classmethod
    def from_mapping(cls, data: dict[str, Any] | None) -> "ExperimentConfig":
        if data is None:
            raise ValueError("Missing required 'experiment' section")
        if not data.get("name"):
            raise ValueError("experiment.name is required")

        tags = data.get("tags", ())
        if tags is None:
            tags = ()
        if not isinstance(tags, (list, tuple)):
            raise ValueError("experiment.tags must be a list")

        return cls(
            name=str(data["name"]),
            seed=int(data.get("seed", cls.seed)),
            tags=tuple(str(tag) for tag in tags),
            output_dir=data.get("output_dir"),
        )
