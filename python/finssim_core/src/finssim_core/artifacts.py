from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ArtifactLayout:
    """Resolved artifact paths for one backend run."""

    root: Path
    run_dir: Path
    logs_dir: Path
    checkpoints_dir: Path
    tensorboard_dir: Path


def default_artifacts_root(workspace_root: str | Path | None = None) -> Path:
    """Return the default artifact root for FinsSim-managed runs."""
    root = Path.cwd() if workspace_root is None else Path(workspace_root)
    return root / "artifacts"


def build_run_dir(
    backend: str,
    run_name: str,
    artifacts_root: str | Path | None = None,
) -> Path:
    """Return artifacts/runs/{backend}/{run_name} without creating it."""
    root = Path("artifacts") if artifacts_root is None else Path(artifacts_root)
    return root / "runs" / backend / run_name


def create_artifact_layout(
    backend: str,
    run_name: str,
    artifacts_root: str | Path | None = None,
    *,
    create: bool = True,
) -> ArtifactLayout:
    """Resolve and optionally create the standard artifact layout for a run."""
    root = Path("artifacts") if artifacts_root is None else Path(artifacts_root)
    run_dir = build_run_dir(backend, run_name, root)
    layout = ArtifactLayout(
        root=root,
        run_dir=run_dir,
        logs_dir=run_dir / "logs",
        checkpoints_dir=run_dir / "checkpoints",
        tensorboard_dir=run_dir / "tensorboard",
    )

    if create:
        for directory in (
            layout.run_dir,
            layout.logs_dir,
            layout.checkpoints_dir,
            layout.tensorboard_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)

    return layout
