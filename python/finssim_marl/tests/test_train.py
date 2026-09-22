"""
Pytest tests for training script.

Usage:
    pytest tests/test_train.py -v
    pytest tests/test_train.py -v -k "test_training_runs"  # Run specific test

Requirements for integration tests:
    - DISPLAY environment variable must be set (or use xvfb-run for headless)
    - Unity environment binary must be available
    - .venv virtual environment will be activated automatically via conftest
"""

import os
import subprocess
import sys


def test_training_command_runs():
    """Test that the training command from train_command_line.md executes without immediate errors.

    This is an integration smoke test that verifies the training script
    can be invoked with the expected arguments. It does NOT run a full
    training session (which would take hours).
    """
    # Get paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_script = os.path.join(base_dir, "scripts", "train.py")

    # Build command - simulate what train_command_line.md shows
    cmd = [
        sys.executable,
        train_script,
        "--config", "chasing_3_chase_1_test",
        "--exp-name", "pytest_smoke_test",
        "--env_base_port", "2422",
    ]

    # Run with short timestep limit for smoke test
    env = os.environ.copy()
    env["DISPLAY"] = env.get("DISPLAY", ":0")

    # Use minimal timesteps to verify script starts correctly
    result = subprocess.run(
        cmd,
        cwd=base_dir,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,  # Should fail fast if something is wrong
    )

    # Check for immediate failures (import errors, missing deps, etc.)
    assert result.returncode == 0, (
        f"Training script failed with return code {result.returncode}\n"
        f"STDOUT: {result.stdout}\n"
        f"STDERR: {result.stderr}"
    )