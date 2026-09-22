"""
Pytest configuration for UnderwaterMARL tests.

Sets up common fixtures and environment handling.
"""

import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def pytest_configure(config):
    """Configure pytest with common settings."""
    # Ensure .venv python is used when available
    venv_python = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        ".venv",
        "bin",
        "python"
    )
    if os.path.exists(venv_python) and sys.executable != venv_python:
        # Suggest using venv python
        pass
