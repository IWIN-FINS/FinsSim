from __future__ import annotations

import argparse
from pathlib import Path

from .analysis import analyze_session
from .manifest import now_iso, read_json, write_json


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility command for a recorder session. Recording now runs "
            "analysis automatically; this command only re-runs it and updates "
            "external-file bookkeeping."
        )
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--status", choices=("complete", "needs_review", "blocked"), default="needs_review")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()
    session_dir = args.session_dir.expanduser().resolve()
    manifest_path = session_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"manifest not found: {manifest_path}")
    manifest = read_json(manifest_path)
    manifest["status"] = args.status
    manifest["finalized_at"] = now_iso()
    manifest["finalize_notes"] = args.notes
    manifest["bag_metadata_exists"] = (session_dir / "raw" / "rosbag2" / "metadata.yaml").is_file()
    manifest["external_truth_present"] = (session_dir / "external_truth").exists()
    manifest["video_present"] = (session_dir / "video").exists() and any((session_dir / "video").iterdir())
    report = analyze_session(session_dir, force=True)
    manifest["analysis"] = {
        "enabled": True,
        "status": report.get("status", "needs_review"),
        "report": "derived/analysis_report.json",
        "plots": report.get("plots", []),
        "warnings": report.get("warnings", []),
    }
    write_json(manifest_path, manifest)
    print(f"finalized {manifest.get('experiment_id', '<unknown>')}/{manifest.get('session_id', session_dir.name)}: {args.status}")


if __name__ == "__main__":
    main()
