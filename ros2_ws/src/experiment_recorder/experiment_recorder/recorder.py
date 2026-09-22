from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import time
from typing import Any, Sequence

import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions

from msgs.msg import TrajectoryEvent

from .analysis import analyze_session
from .layout import ensure_trial_data_dirs
from .manifest import file_sha256, git_revision, now_iso, safe_session_dir, write_json
from .progress import log_stage
from .profiles import topics_for_profile


def _repo_root() -> Path:
    configured = os.environ.get("FINSSIM_REPO_ROOT", "").strip()
    if configured:
        return Path(configured).resolve()
    for parent in Path(__file__).resolve().parents:
        if (parent / "ros2_ws").is_dir() and (parent / "pyproject.toml").is_file():
            return parent
    return Path.cwd().resolve()


class EventPublisher(Node):
    def __init__(self, topic: str) -> None:
        super().__init__("finsrov_experiment_event_publisher")
        self._publisher = self.create_publisher(TrajectoryEvent, topic, 10)

    def publish(self, session_id: str, event: str, label: str, metadata: dict[str, Any]) -> None:
        message = TrajectoryEvent()
        message.header.stamp = self.get_clock().now().to_msg()
        message.session_id = session_id
        message.event = event
        message.label = label
        message.metadata_json = json.dumps(metadata, ensure_ascii=True, sort_keys=True)
        self._publisher.publish(message)


def _publish_repeated(node: EventPublisher, session_id: str, event: str, label: str, metadata: dict[str, Any]) -> None:
    for _ in range(3):
        if not rclpy.ok():
            return
        node.publish(session_id, event, label, metadata)
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.1)


def _stop_recorder_process(child: subprocess.Popen[str]) -> None:
    """Stop the rosbag command without assuming ownership of a terminal group.

    ``record_experiment`` can run standalone from a user terminal or as a
    child of ``_ManagedProcess``.  The bag process therefore deliberately
    stays in its parent's process group; a forced experiment shutdown then
    terminates both recorder and bag together.  For a normal recorder exit we
    signal only the direct rosbag process, never the caller's terminal group.
    """

    if child.poll() is not None:
        return
    try:
        child.send_signal(signal.SIGINT)
    except ProcessLookupError:
        return
    try:
        child.wait(timeout=30.0)
    except subprocess.TimeoutExpired:
        child.terminate()
        try:
            child.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=5.0)


def _parse_metadata(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise SystemExit(f"--metadata-json must be a JSON object: {exc}") from exc
    if not isinstance(value, dict):
        raise SystemExit("--metadata-json must be a JSON object")
    return value


def _strategy_slug(value: object) -> str:
    """Return a filesystem-safe, human-readable strategy identifier."""

    text = str(value or "UNSPECIFIED").strip().upper()
    result = "".join(char if char.isalnum() or char in "-_" else "_" for char in text)
    return result.strip("-_") or "UNSPECIFIED"


def _session_id_with_strategy(session_id: str, strategy: str) -> str:
    """Avoid appending a multi-word strategy that is already in the ID."""

    return session_id if strategy.upper() in session_id.upper() else f"{session_id}_{strategy}"


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record one FinsROV experiment session with rosbag2 and a manifest.")
    parser.add_argument("--experiment-id", required=True, help="E1, E2, E4, E5, E6, E7, E9, or E10.")
    parser.add_argument("--profile", required=True, choices=("apriltag", "t1", "t2", "t1_sim", "t2_sim", "full"))
    parser.add_argument("--session-id", default=None)
    parser.add_argument(
        "--strategy",
        default=None,
        help="Controller/policy name included in the session directory (for example PID or PPO).",
    )
    parser.add_argument("--label", default="")
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--duration", type=float, default=0.0, help="Seconds; <=0 records until Ctrl-C.")
    parser.add_argument("--max-bag-duration", type=float, default=0.0)
    parser.add_argument("--config", action="append", default=[], type=Path, help="Resolved config to hash; repeatable.")
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--metadata-json", default="{}")
    parser.add_argument("--truth-x", type=float, default=None, help="External pool_world truth x in metres.")
    parser.add_argument("--truth-y", type=float, default=None, help="External pool_world truth y in metres.")
    parser.add_argument("--truth-frame", default="pool_world", help="Frame for external truth; E1/E2 use pool_world.")
    parser.add_argument("--position-id", default="", help="Stable external-truth position identifier.")
    parser.add_argument("--repeat", type=int, default=1, help="Repeat number for the external-truth row.")
    parser.add_argument("--truth-method", default="survey_grid")
    parser.add_argument("--truth-uncertainty", type=float, default=None)
    parser.add_argument("--truth-notes", default="")
    parser.add_argument(
        "--prepared-session",
        action="store_true",
        help="Allow a runner-created session directory containing only trial-local runtime logs.",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--no-auto-analysis",
        action="store_true",
        help="Do not analyse this session after rosbag2 stops (use only for debugging).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if (args.truth_x is None) != (args.truth_y is None):
        parser.error("--truth-x and --truth-y must be supplied together")
    if args.repeat < 1:
        parser.error("--repeat must be >= 1")
    if args.prepared_session and args.overwrite:
        parser.error("--prepared-session and --overwrite cannot be used together")
    return args


def _truth_metadata(args: argparse.Namespace) -> dict[str, Any]:
    """Return auditable truth metadata, or an empty dict for non-truth runs."""

    if args.truth_x is None or args.truth_y is None:
        return {}
    return {
        "truth_xy": [float(args.truth_x), float(args.truth_y)],
        "truth_frame": str(args.truth_frame),
        "position_id": str(args.position_id or "P_AUTO"),
        "repeat": int(args.repeat),
        "measurement_method": str(args.truth_method),
        "uncertainty_m": args.truth_uncertainty,
        "truth_notes": str(args.truth_notes),
    }


def _write_truth_csv(session_dir: Path, args: argparse.Namespace) -> Path | None:
    """Write one external horizontal-truth row for an E1/E2 session."""

    truth = _truth_metadata(args)
    if not truth:
        return None
    path = session_dir / "external_truth" / "truth_xy.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "position_id",
        "repeat",
        "truth_x_m",
        "truth_y_m",
        "truth_frame",
        "measurement_method",
        "uncertainty_m",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerow(
            {
                "position_id": truth["position_id"],
                "repeat": truth["repeat"],
                "truth_x_m": f"{args.truth_x:.6f}",
                "truth_y_m": f"{args.truth_y:.6f}",
                "truth_frame": truth["truth_frame"],
                "measurement_method": truth["measurement_method"],
                "uncertainty_m": "" if args.truth_uncertainty is None else f"{args.truth_uncertainty:.6f}",
                "notes": args.truth_notes,
            }
        )
    return path


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    repo_root = _repo_root()
    metadata = _parse_metadata(args.metadata_json)
    strategy = _strategy_slug(args.strategy or metadata.get("strategy") or metadata.get("controller"))
    if args.session_id:
        # Keep manually supplied IDs readable while guaranteeing that the
        # strategy is present in the on-disk session name as well as metadata.
        # Strategies commonly contain underscores, so testing individual path
        # components would append an already-present slug a second time.
        session_id = _session_id_with_strategy(str(args.session_id), strategy)
    else:
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + f"_{args.experiment_id}_{strategy}"
    output_root = args.output_root or repo_root / "ros2_ws" / "data" / "experiments" / args.experiment_id
    try:
        session_dir = safe_session_dir(output_root, session_id)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if session_dir.exists() and not args.overwrite and not args.prepared_session:
        raise SystemExit(f"session already exists: {session_dir}; use --overwrite only after checking the target")
    if args.prepared_session:
        if not session_dir.is_dir() or (session_dir / "manifest.json").exists():
            raise SystemExit("--prepared-session requires a new runner-created session directory without manifest.json")
        unexpected = [child.name for child in session_dir.iterdir() if child.name != "logs"]
        if unexpected:
            raise SystemExit(
                "--prepared-session session directory may contain only logs/ before recording; "
                f"found {sorted(unexpected)}"
            )

    topics = topics_for_profile(args.profile)
    bag_dir = session_dir / "raw" / "rosbag2"
    command = ["ros2", "bag", "record", "--output", str(bag_dir), "--storage", "sqlite3"]
    if args.max_bag_duration > 0.0:
        command.extend(["--max-bag-duration", str(args.max_bag_duration)])
    command.extend(topics)
    metadata["strategy"] = strategy
    truth_metadata = _truth_metadata(args)
    for key, value in truth_metadata.items():
        metadata.setdefault(key, value)
    config_hashes = {str(path): file_sha256(path) for path in args.config}
    manifest = {
        "schema_version": 2,
        "experiment_id": str(args.experiment_id),
        "profile": str(args.profile),
        "strategy": strategy,
        "session_id": session_id,
        "label": str(args.label),
        "created_at": now_iso(),
        "status": "planned" if args.dry_run else "recording",
        "git_revision": git_revision(repo_root),
        "config_sha256": config_hashes,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "checkpoint_sha256": file_sha256(args.checkpoint) if args.checkpoint else None,
        "metadata": metadata,
        "recording": {
            "storage": "sqlite3",
            "compression": "none",
            "duration_sec": float(args.duration),
            "topics": list(topics),
            "command": command,
        },
        "started_at": None,
        "ended_at": None,
        "events_topic": "/finsrov/experiment/event",
        "analysis": {
            "enabled": not args.no_auto_analysis,
            "status": "pending" if not args.no_auto_analysis else "disabled",
            "report": "derived/analysis_report.json",
        },
        "artifacts": {
            "raw_rosbag": "raw/rosbag2",
            "derived": "derived",
            "runtime_logs": {
                "local": ["logs/recorder.log"] if args.prepared_session else [],
                "shared": [],
            },
        },
    }
    if truth_metadata:
        manifest["external_truth"] = {
            "file": "external_truth/truth_xy.csv",
            "source": "cli",
            "dimensions": ["pool_world_x", "pool_world_y"],
        }

    if args.dry_run:
        print(json.dumps(manifest, ensure_ascii=True, indent=2, sort_keys=True))
        print("rosbag command:")
        print(" ".join(command))
        return

    if session_dir.exists() and args.overwrite:
        for child in session_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
    session_dir.mkdir(parents=True, exist_ok=True)
    ensure_trial_data_dirs(session_dir, external_truth=bool(truth_metadata))
    write_json(session_dir / "manifest.json", manifest)
    _write_truth_csv(session_dir, args)

    # Keep Python's KeyboardInterrupt alive long enough to close rosbag2 and
    # write the session manifest. The default rclpy handler invalidates the
    # context before this ``try/finally`` can perform that cleanup.
    rclpy.init(signal_handler_options=SignalHandlerOptions.NO)
    node = EventPublisher("/finsrov/experiment/event")
    child: subprocess.Popen[str] | None = None
    recording_error: Exception | None = None
    manifest["started_at"] = now_iso()
    write_json(session_dir / "manifest.json", manifest)
    try:
        # Do not create a detached session for rosbag2. When launched by a
        # T1/T2 runner, the runner already owns this process group; its forced
        # shutdown must therefore reach this child as well.
        child = subprocess.Popen(command, text=True)
        time.sleep(0.75)
        if child.poll() is not None:
            raise RuntimeError(f"rosbag exited immediately with code {child.returncode}")
        _publish_repeated(node, session_id, "session_start", args.label, metadata)
        if truth_metadata:
            # The CSV remains the source of truth; this marker only provides a
            # timestamp for aligning the entered pose with the bag.
            _publish_repeated(node, session_id, "truth_xy", args.position_id or "P_AUTO", metadata)
        log_stage(
            f"recording started: experiment={args.experiment_id} session={session_id} "
            f"profile={args.profile} output={session_dir}",
            logger=node.get_logger(),
        )
        start = time.monotonic()
        while rclpy.ok() and child.poll() is None:
            if args.duration > 0.0 and time.monotonic() - start >= args.duration:
                break
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        log_stage("recording interrupted by operator; finalizing rosbag before analysis", logger=node.get_logger())
    except Exception as exc:
        recording_error = exc
        node.get_logger().error(f"recording failed: {exc}")
    finally:
        if rclpy.ok():
            try:
                _publish_repeated(node, session_id, "session_stop", args.label, metadata)
            except Exception as exc:  # pragma: no cover - shutdown safety path
                node.get_logger().warning(f"could not publish session_stop: {exc}")
        if child is not None:
            _stop_recorder_process(child)
        rosbag_ok = child is not None and child.returncode in (0, 130, 2)
        manifest["status"] = "complete" if recording_error is None and rosbag_ok else "rosbag_failed"
        if recording_error is not None:
            manifest["error"] = repr(recording_error)
        manifest["ended_at"] = now_iso()
        write_json(session_dir / "manifest.json", manifest)
        if not args.no_auto_analysis:
            # Analysis runs only after rosbag2 has received its stop signal, so
            # SQLite metadata and all committed messages are available.  It is
            # deliberately best-effort: a plotting/import problem is recorded
            # in the report without changing a valid recording into a failure.
            try:
                log_stage(
                    f"automatic analysis started: session={session_id} input={session_dir / 'raw'}",
                    logger=node.get_logger(),
                )
                report = analyze_session(session_dir)
                manifest["analysis"] = {
                    "enabled": True,
                    "status": report.get("status", "needs_review"),
                    "report": "derived/analysis_report.json",
                    "plots": report.get("plots", []),
                    "warnings": report.get("warnings", []),
                }
                log_stage(
                    f"automatic analysis complete: session={session_id} "
                    f"status={manifest['analysis']['status']} report={session_dir / 'derived' / 'analysis_report.json'} "
                    f"plots={len(manifest['analysis']['plots'])}",
                    logger=node.get_logger(),
                )
            except Exception as exc:  # pragma: no cover - shutdown safety path
                manifest["analysis"] = {
                    "enabled": True,
                    "status": "needs_review",
                    "report": "derived/analysis_report.json",
                    "error": repr(exc),
                }
                log_stage(
                    f"automatic analysis needs review: session={session_id} error={exc!r}",
                    logger=node.get_logger(),
                )
            write_json(session_dir / "manifest.json", manifest)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if recording_error is not None:
        raise SystemExit(f"experiment recording failed; see {session_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
