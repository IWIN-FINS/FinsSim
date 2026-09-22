"""Derive explicit GoalYaw episode manifests from recorded trajectory events."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

from .contracts import PROTOCOL_VERSION


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    value = row.get("metadata", {})
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        parsed = json.loads(value)
        if isinstance(parsed, dict):
            return parsed
    return {}


def build_episode_manifest(rows: list[dict[str, Any]], *, session_id: str | None = None) -> list[dict[str, Any]]:
    """Create non-overlapping canonical episode intervals from raw event rows."""

    starts: dict[str, dict[str, Any]] = {}
    stops: dict[str, dict[str, Any]] = {}
    for row in sorted(rows, key=lambda item: float(item["source_time_sec"])):
        if session_id is not None and str(row.get("session_id", "")) != session_id:
            continue
        metadata = _metadata(row)
        if metadata.get("protocol") != PROTOCOL_VERSION:
            continue
        episode_id = str(metadata.get("episode_id", row.get("label", ""))).strip()
        if not episode_id:
            raise ValueError("GoalYaw event has no episode_id")
        event = str(row.get("event", ""))
        if event == "goal_yaw_episode_start":
            starts.setdefault(episode_id, {**row, "metadata": metadata})
        elif event == "goal_yaw_episode_stop":
            stops[episode_id] = {**row, "metadata": metadata}

    episodes: list[dict[str, Any]] = []
    for episode_id, start in starts.items():
        stop = stops.get(episode_id)
        if stop is None:
            raise ValueError(f"GoalYaw episode '{episode_id}' has no stop event")
        start_time = float(start["source_time_sec"])
        end_time = float(stop["source_time_sec"])
        if end_time <= start_time:
            raise ValueError(f"GoalYaw episode '{episode_id}' has non-positive duration")
        metadata = start["metadata"]
        position = metadata.get("goal_position_world_xyz")
        orientation = metadata.get("goal_orientation_xyzw")
        if not isinstance(position, list) or len(position) != 3 or not isinstance(orientation, list) or len(orientation) != 4:
            raise ValueError(f"GoalYaw episode '{episode_id}' has an invalid goal pose")
        episodes.append(
            {
                "episode_id": episode_id,
                "start_time_sec": start_time,
                "end_time_sec": end_time,
                "goal_frame_id": str(metadata.get("goal_frame_id", "")),
                "goal_position_world_xyz": [float(value) for value in position],
                "goal_orientation_xyzw": [float(value) for value in orientation],
                "outcome": str(stop["metadata"].get("outcome", "")),
                "terminal": True,
            }
        )
    episodes.sort(key=lambda item: float(item["start_time_sec"]))
    for previous, current in zip(episodes, episodes[1:]):
        if float(current["start_time_sec"]) < float(previous["end_time_sec"]):
            raise ValueError("GoalYaw episodes overlap; record one vehicle task at a time")
    if not episodes:
        raise ValueError("no GoalYawIRL protocol events were found")
    return episodes


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build GoalYawIRL JSONL episode manifests from raw trajectory_event.parquet.")
    parser.add_argument("--events-parquet", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    events_path = args.events_parquet.expanduser().resolve()
    destination = args.output.expanduser().resolve()
    if destination.exists() and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing manifest: {destination}")
    try:
        rows = pq.read_table(events_path).to_pylist()
        episodes = build_episode_manifest(rows, session_id=args.session_id)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise SystemExit(str(exc)) from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        "".join(json.dumps(episode, ensure_ascii=True, sort_keys=True) + "\n" for episode in episodes),
        encoding="utf-8",
    )
    print(f"wrote {len(episodes)} GoalYawIRL episode manifests to {destination}")


if __name__ == "__main__":
    main()
