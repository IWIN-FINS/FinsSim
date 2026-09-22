"""Dependency-light helpers for the E1 position/repeat convention."""

from __future__ import annotations

import csv
from pathlib import Path


def axis_token(value: float) -> str:
    sign = "N" if value < 0 else "P"
    return sign + f"{abs(value):.3f}".replace(".", "p")


def position_id(x: float, y: float) -> str:
    return f"P_X{axis_token(x)}_Y{axis_token(y)}"


def next_repeat(output_root: Path, x: float, y: float, precision: int = 3) -> int:
    """Count completed truth rows at this rounded position and return next repeat."""

    tolerance = 0.5 * 10 ** (-precision)
    repeats: list[int] = []
    if not output_root.is_dir():
        return 1
    for truth_file in output_root.glob("*/external_truth/truth_xy.csv"):
        try:
            with truth_file.open(encoding="utf-8", newline="") as stream:
                for row in csv.DictReader(stream):
                    if abs(float(row["truth_x_m"]) - x) <= tolerance and abs(float(row["truth_y_m"]) - y) <= tolerance:
                        repeats.append(int(row.get("repeat", "0")))
        except (OSError, KeyError, TypeError, ValueError):
            # An incomplete truth file remains auditable but must not block a
            # new run.
            continue
    return max(repeats, default=0) + 1
