"""Human-readable experiment-stage messages.

ROS2's native logger already provides the standard ``[INFO]`` prefix.  This
helper therefore standardises the useful stage text---rather than inventing a
second logging format---and keeps standalone analysis commands equally clear.
"""

from __future__ import annotations

from typing import Any


def log_stage(message: str, *, logger: Any | None = None) -> None:
    """Emit one concise operator-facing stage message.

    ``logger`` is intentionally duck-typed: passing an rclpy logger keeps the
    message in ROS2's normal log stream, while standalone tools such as the
    analyser print to the invoking terminal.  The function never changes the
    message text or suppresses an exception from an unavailable optional
    logger; those would hide the actual experiment state from the operator.
    """

    if logger is None:
        print(f"[INFO] {message}", flush=True)
    else:
        logger.info(message)
