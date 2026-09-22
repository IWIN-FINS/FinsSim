from __future__ import annotations

from irl_data.contracts import PROTOCOL_VERSION
from irl_data.episode_manifest import build_episode_manifest


def test_manifest_uses_explicit_start_stop_and_goal() -> None:
    rows = [
        {
            "source_time_sec": 10.0,
            "session_id": "session-a",
            "event": "goal_yaw_episode_start",
            "label": "episode-1",
            "metadata": {
                "protocol": PROTOCOL_VERSION,
                "episode_id": "episode-1",
                "goal_frame_id": "map",
                "goal_position_world_xyz": [1.0, 2.0, 3.0],
                "goal_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
        },
        {
            "source_time_sec": 14.0,
            "session_id": "session-a",
            "event": "goal_yaw_episode_stop",
            "label": "episode-1",
            "metadata": {"protocol": PROTOCOL_VERSION, "episode_id": "episode-1", "outcome": "success"},
        },
    ]
    episodes = build_episode_manifest(rows, session_id="session-a")
    assert episodes == [
        {
            "episode_id": "episode-1",
            "start_time_sec": 10.0,
            "end_time_sec": 14.0,
            "goal_frame_id": "map",
            "goal_position_world_xyz": [1.0, 2.0, 3.0],
            "goal_orientation_xyzw": [0.0, 0.0, 0.0, 1.0],
            "outcome": "success",
            "terminal": True,
        }
    ]
