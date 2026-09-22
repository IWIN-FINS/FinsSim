"""Convenience E1 recorder for the horizontal AprilTag accuracy experiment.

The operator supplies only the surveyed ``pool_world`` truth ``x`` and ``y``.
All other recording choices come from ``config/e1_e2_apriltag.yaml``.  This
entry point deliberately does not launch perception, fusion, or hardware: the
startup block in the YAML is a versioned checklist, not an auto-start policy.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Sequence

import yaml

from . import recorder
from .apriltag_utils import next_repeat, position_id
from .progress import log_stage


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


def _default_config() -> Path:
    # Source-tree fallback.  The installed package uses the same relative
    # config path through setup.py's share/<package>/config installation.
    source_config = Path(__file__).resolve().parents[1] / "config" / "e1_e2_apriltag.yaml"
    try:
        from ament_index_python.packages import get_package_share_directory

        installed = Path(get_package_share_directory("experiment_recorder")) / "config" / "e1_e2_apriltag.yaml"
        if installed.is_file():
            return installed
    except Exception:
        pass
    return source_config


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise SystemExit(f"E1/E2 config not found: {path}") from exc
    except yaml.YAMLError as exc:
        raise SystemExit(f"invalid E1/E2 YAML config {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"E1/E2 config must contain a YAML mapping: {path}")
    return payload


def _resolve_repo_path(value: str | Path, repo_root: Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _new_session_id(output_root: Path, experiment_id: str) -> str:
    base = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{experiment_id}"
    candidate = base
    suffix = 2
    while (output_root / candidate).exists():
        candidate = f"{base}_{suffix:02d}"
        suffix += 1
    return candidate


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record E1 AprilTag horizontal accuracy; only truth x/y are required."
    )
    parser.add_argument("--truth-x", required=True, type=float, help="Surveyed pool_world x [m].")
    parser.add_argument("--truth-y", required=True, type=float, help="Surveyed pool_world y [m].")
    parser.add_argument("--config-file", type=Path, default=None, help="Override e1_e2_apriltag.yaml.")
    parser.add_argument("--duration", type=float, default=None, help="Override recording duration; <=0 means Ctrl-C.")
    parser.add_argument("--max-bag-duration", type=float, default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = _repo_root()
    config_file = (args.config_file or _default_config()).expanduser().resolve()
    config = _load_config(config_file)
    experiment = config.get("experiment", {})
    if not isinstance(experiment, dict):
        raise SystemExit("experiment must be a YAML mapping")
    experiment_id = str(experiment.get("online_id", "E1"))
    profile = str(experiment.get("recorder_profile", "apriltag"))
    output_root = _resolve_repo_path(experiment.get("output_root", "ros2_ws/data/experiments/E1"), repo_root)
    precision = int(experiment.get("position_id_precision", 3))
    position_id_value = position_id(args.truth_x, args.truth_y)
    repeat = next_repeat(output_root, args.truth_x, args.truth_y, precision)
    session_id = _new_session_id(output_root, experiment_id)

    recording = config.get("recording", {})
    if not isinstance(recording, dict):
        recording = {}
    duration = float(args.duration if args.duration is not None else recording.get("duration_sec", 0.0))
    max_bag_duration = float(
        args.max_bag_duration
        if args.max_bag_duration is not None
        else recording.get("max_bag_duration_sec", 0.0)
    )

    config_paths: list[Path] = [config_file]
    configured_paths = config.get("configs", {})
    if isinstance(configured_paths, dict):
        for value in configured_paths.values():
            if isinstance(value, str):
                config_paths.append(_resolve_repo_path(value, repo_root))

    metadata = {
        "experiment_mode": "E1_online",
        "setup_config": str(config_file),
        "truth_input": "cli",
        "position_id": position_id_value,
        "repeat": repeat,
        "truth_xy": [args.truth_x, args.truth_y],
        "truth_frame": str(experiment.get("truth_frame", "pool_world")),
        "truth_measurement_method": str(experiment.get("truth_measurement_method", "survey_grid")),
    }
    label = args.label or f"{position_id_value}_repeat_{repeat:02d}"
    forwarded = [
        "--experiment-id",
        experiment_id,
        "--profile",
        profile,
        "--session-id",
        session_id,
        "--label",
        label,
        "--output-root",
        str(output_root),
        "--duration",
        str(duration),
        "--max-bag-duration",
        str(max_bag_duration),
        "--truth-x",
        str(args.truth_x),
        "--truth-y",
        str(args.truth_y),
        "--truth-frame",
        str(experiment.get("truth_frame", "pool_world")),
        "--position-id",
        position_id_value,
        "--repeat",
        str(repeat),
        "--truth-method",
        str(experiment.get("truth_measurement_method", "survey_grid")),
        "--metadata-json",
        json.dumps(metadata, ensure_ascii=True, sort_keys=True),
    ]
    for path in config_paths:
        forwarded.extend(["--config", str(path)])
    if args.dry_run:
        forwarded.append("--dry-run")
    if args.overwrite:
        forwarded.append("--overwrite")
    log_stage(
        f"E1 localization acquisition prepared: point={position_id_value} repeat={repeat} "
        f"session={session_id} truth=pool_world[{args.truth_x:.6f}, {args.truth_y:.6f}] m"
    )
    recorder.main(forwarded)


if __name__ == "__main__":
    main()
