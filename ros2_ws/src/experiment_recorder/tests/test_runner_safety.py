import os
from pathlib import Path
import sys
import time

from experiment_recorder.runner import (
    _ManagedProcess,
    _bridge_enabled_confirmed,
    _mcu_disarm_confirmed,
    _record_command,
    _session_strategy_slug,
)
from experiment_recorder.t2_runner import _global_motion_controller_nodes


def _safe_status() -> dict[str, object]:
    return {
        "enabled": False,
        "mcu_diagnostics_supported": True,
        "mcu_has_received_command": True,
        "mcu_direct_thrusters_enabled": False,
        "mcu_target_rpm_nonzero": False,
        "mcu_target_throttle_nonzero": False,
    }


def test_mcu_disarm_requires_fresh_firmware_status() -> None:
    assert not _mcu_disarm_confirmed(
        _safe_status(), status_sequence=4, status_sequence_before_disarm=4
    )
    assert _mcu_disarm_confirmed(
        _safe_status(), status_sequence=5, status_sequence_before_disarm=4
    )


def test_mcu_disarm_rejects_armed_or_nonzero_status() -> None:
    unsafe = _safe_status()
    unsafe["mcu_target_rpm_nonzero"] = True
    assert not _mcu_disarm_confirmed(
        unsafe, status_sequence=5, status_sequence_before_disarm=4
    )

    no_diagnostics = _safe_status()
    no_diagnostics["mcu_diagnostics_supported"] = False
    assert not _mcu_disarm_confirmed(
        no_diagnostics, status_sequence=5, status_sequence_before_disarm=4
    )


def test_bridge_enable_requires_a_fresh_enabled_status() -> None:
    assert not _bridge_enabled_confirmed(
        {"enabled": True}, status_sequence=4, status_sequence_before_request=4
    )
    assert not _bridge_enabled_confirmed(
        {"enabled": False}, status_sequence=5, status_sequence_before_request=4
    )
    assert _bridge_enabled_confirmed(
        {"enabled": True}, status_sequence=5, status_sequence_before_request=4
    )


def test_session_strategy_slug_uses_protocol_alias_and_bounds_unknown_profiles() -> None:
    assert _session_strategy_slug(
        "ppo_isaaclab_finsrov_hold_for_position_reprojected_physical_wrench",
        {"ppo_isaaclab_finsrov_hold_for_position_reprojected_physical_wrench": "ppo_isaac_reproj"},
    ) == "ppo_isaac_reproj"
    assert len(_session_strategy_slug("ppo_" + "long_" * 20, {})) <= 48


def test_deferred_trial_recording_disables_per_trial_auto_analysis(tmp_path: Path) -> None:
    command = _record_command(
        experiment_id="E7",
        strategy="ppo_test",
        session_id="test_session",
        output_root=tmp_path,
        profile="t1",
        metadata={},
        config_files=[],
        checkpoint=None,
        defer_analysis=True,
    )

    assert "--no-auto-analysis" in command


def test_managed_process_stops_children_when_launch_leader_has_already_exited(tmp_path: Path) -> None:
    """Regression: a dead ros2-launch leader must not leak a controller child."""

    child_code = "import time; time.sleep(60)"
    launcher_code = (
        "import subprocess, sys; "
        f"child = subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print(child.pid, flush=True)"
    )
    managed = _ManagedProcess([sys.executable, "-c", launcher_code], tmp_path / "leader.log")
    child_pid: int | None = None
    try:
        managed.process.wait(timeout=2.0)
        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline and not (tmp_path / "leader.log").read_text().strip():
            time.sleep(0.02)
        child_pid = int((tmp_path / "leader.log").read_text().strip())
        assert managed._group_exists()

        managed.stop(0.2)
        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and Path(f"/proc/{child_pid}").exists():
            time.sleep(0.02)
        assert not Path(f"/proc/{child_pid}").exists()
    finally:
        # Keep the test self-cleaning if an assertion fails midway.
        try:
            os.killpg(managed.process.pid, 9)
        except ProcessLookupError:
            pass
        if child_pid is not None:
            try:
                os.kill(child_pid, 9)
            except ProcessLookupError:
                pass


def test_t2_detects_only_a_global_real_motion_controller() -> None:
    assert _global_motion_controller_nodes([
        ("motion_controller", "/"),
        ("motion_controller", "/sim"),
        ("other_node", "/"),
    ]) == ["/motion_controller"]
