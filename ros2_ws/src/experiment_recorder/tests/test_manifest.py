from pathlib import Path

import pytest

from experiment_recorder.manifest import safe_session_dir
from experiment_recorder.recorder import _session_id_with_strategy


def test_safe_session_dir_rejects_paths(tmp_path: Path):
    with pytest.raises(ValueError):
        safe_session_dir(tmp_path, "../escape")
    with pytest.raises(ValueError):
        safe_session_dir(tmp_path, "/absolute")


def test_safe_session_dir_is_below_root(tmp_path: Path):
    assert safe_session_dir(tmp_path, "session_001") == (tmp_path / "session_001").resolve()


def test_session_id_does_not_repeat_an_existing_multiword_strategy_slug():
    session_id = "20260823_E7_ppo_isaac_reproj_P01_R01"
    assert _session_id_with_strategy(session_id, "PPO_ISAAC_REPROJ") == session_id
