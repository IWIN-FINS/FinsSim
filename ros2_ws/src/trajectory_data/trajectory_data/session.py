from __future__ import annotations

from pathlib import Path
import shutil


def session_directory(output_root: Path, session_id: str) -> Path:
    """Resolve one safe, single-directory session path below output_root."""

    name = str(session_id).strip()
    candidate = Path(name)
    if not name or candidate.is_absolute() or candidate.name != name or name in {".", ".."}:
        raise ValueError("--session-id must be a non-empty single directory name")
    root = output_root.expanduser().resolve()
    session_dir = (root / candidate).resolve()
    try:
        session_dir.relative_to(root)
    except ValueError as exc:
        raise ValueError("--session-id must resolve below --output-root") from exc
    return session_dir


def prepare_session_directory(session_dir: Path, *, overwrite: bool) -> None:
    if not session_dir.exists():
        session_dir.mkdir(parents=True)
        return
    if not overwrite:
        raise FileExistsError(f"session directory already exists: {session_dir}")
    if not session_dir.is_dir():
        raise ValueError(f"refusing to overwrite non-directory session path: {session_dir}")
    shutil.rmtree(session_dir)
    session_dir.mkdir(parents=True)
