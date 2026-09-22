"""Manifest and safe-path helpers shared by recorder commands."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
from pathlib import Path
import subprocess
from typing import Any


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def safe_session_dir(root: Path, session_id: str) -> Path:
    name = str(session_id).strip()
    candidate = Path(name)
    if not name or candidate.is_absolute() or candidate.name != name or name in {".", ".."}:
        raise ValueError("--session-id must be a non-empty single directory name")
    resolved_root = root.expanduser().resolve()
    session = (resolved_root / candidate).resolve()
    try:
        session.relative_to(resolved_root)
    except ValueError as exc:
        raise ValueError("--session-id must resolve below --output-root") from exc
    return session


def file_sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_revision(repo_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    return payload

