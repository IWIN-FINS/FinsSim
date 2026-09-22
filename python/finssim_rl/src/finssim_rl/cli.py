from __future__ import annotations

import runpy
import sys
from pathlib import Path

_SCRIPTS = {
    "train": "train.py",
    "eval": "eval.py",
    "export": "export_to_onnx.py",
}


def _script_path(command: str) -> Path:
    project_root = Path(__file__).resolve().parents[2]
    return project_root / "scripts" / _SCRIPTS[command]


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in {"-h", "--help"}:
        commands = ", ".join(sorted(_SCRIPTS))
        print(f"Usage: finssim-rl <command> [args]\n\nCommands: {commands}")
        return

    command = sys.argv[1]
    if command not in _SCRIPTS:
        commands = ", ".join(sorted(_SCRIPTS))
        raise SystemExit(f"Unknown finssim-rl command '{command}'. Available: {commands}")

    script = _script_path(command)
    sys.argv = [str(script), *sys.argv[2:]]
    runpy.run_path(str(script), run_name="__main__")
