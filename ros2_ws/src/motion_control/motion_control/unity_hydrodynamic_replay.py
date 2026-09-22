from __future__ import annotations

import argparse
import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


THRUSTER_COLUMNS = (
    "force_cmd_V_LF_n",
    "force_cmd_V_LB_n",
    "force_cmd_V_RB_n",
    "force_cmd_V_RF_n",
    "force_cmd_H_LF_n",
    "force_cmd_H_LB_n",
    "force_cmd_H_RB_n",
    "force_cmd_H_RF_n",
)

FORCE_FROM_RPM_COLUMNS = tuple(column.replace("force_cmd_", "force_from_rpm_") for column in THRUSTER_COLUMNS)


@dataclass(frozen=True)
class ReplayRow:
    source_time_sec: float
    relative_time_sec: float
    trial_index: int
    phase: str
    level: float
    force_n: tuple[float, ...]


def _float(value: str | None, default: float = 0.0) -> float:
    try:
        result = float(value or default)
    except (TypeError, ValueError):
        return default
    return result if math.isfinite(result) else default


def _read_trial_rows(
    csv_path: Path,
    trial_indices: list[int] | None,
    force_source: str,
) -> list[list[ReplayRow]]:
    if force_source not in {"command", "force_from_rpm"}:
        raise ValueError(f"unsupported force source: {force_source}")
    force_columns = THRUSTER_COLUMNS if force_source == "command" else FORCE_FROM_RPM_COLUMNS
    grouped: dict[int, list[tuple[float, str, float, tuple[float, ...]]]] = {}
    with csv_path.open("r", newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        missing = [column for column in force_columns if column not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"{csv_path} is missing thruster columns: {missing}")
        for raw in reader:
            trial = int(_float(raw.get("trial_index"), -1.0))
            if trial < 0:
                continue
            if trial_indices is not None and trial not in trial_indices:
                continue
            source_time = _float(raw.get("time_sec"))
            force = tuple(_float(raw.get(column)) for column in force_columns)
            grouped.setdefault(trial, []).append(
                (source_time, str(raw.get("phase") or "unknown"), _float(raw.get("level")), force)
            )

    trials: list[list[ReplayRow]] = []
    for trial in sorted(grouped):
        raw_rows = sorted(grouped[trial], key=lambda item: item[0])
        if not raw_rows:
            continue
        first_time = raw_rows[0][0]
        trials.append(
            [
                ReplayRow(
                    source_time_sec=source_time,
                    relative_time_sec=max(0.0, source_time - first_time),
                    trial_index=trial,
                    phase=phase,
                    level=level,
                    force_n=force,
                )
                for source_time, phase, level, force in raw_rows
            ]
        )
    if not trials:
        raise ValueError(f"No non-negative trial_index rows found in {csv_path}")
    return trials


class UnityHydrodynamicReplay(Node):
    def __init__(
        self,
        trials: list[list[ReplayRow]],
        output_dir: Path,
        topic: str,
        reset_topic: str,
        settle_sec: float,
        between_trials_sec: float,
        reset_count: int,
        rate_hz: float,
        force_source: str,
    ) -> None:
        super().__init__("unity_hydrodynamic_replay")
        self._trials = trials
        self._output_dir = output_dir
        self._topic = topic
        self._reset_topic = reset_topic
        self._settle_sec = max(0.0, settle_sec)
        self._between_trials_sec = max(0.0, between_trials_sec)
        self._reset_count = max(1, reset_count)
        self._force_source = force_source
        self._publisher = self.create_publisher(Float32MultiArray, topic, 10)
        self._reset_publisher = self.create_publisher(Float32MultiArray, reset_topic, 10)
        self._start_monotonic = time.monotonic()
        self._phase_start_monotonic = self._start_monotonic
        self._trial_number = -1
        self._row_index = 0
        self._state = "initial_reset"
        self._zero_sent = 0
        self._done = False
        self._last_reset_monotonic = 0.0
        self._timer = self.create_timer(1.0 / max(1.0, rate_hz), self._tick)
        self._file = (output_dir / "replay_commands.csv").open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._file,
            fieldnames=[
                "wall_time_sec",
                "elapsed_sec",
                "trial_index",
                "phase",
                "level",
                *THRUSTER_COLUMNS,
            ],
        )
        self._writer.writeheader()
        self._write_manifest()
        self.get_logger().info(
            f"replaying {len(trials)} trials from {trials[0][0].trial_index}..{trials[-1][0].trial_index} "
            f"to `{topic}` using force_source={force_source}; output=`{output_dir}`"
        )

    def _write_manifest(self) -> None:
        manifest = {
            "source": "real hydrodynamic identification CSV",
            "topic": self._topic,
            "reset_topic": self._reset_topic,
            "force_source": self._force_source,
            "force_columns": list(THRUSTER_COLUMNS if self._force_source == "command" else FORCE_FROM_RPM_COLUMNS),
            "trial_indices": [trial[0].trial_index for trial in self._trials],
            "settle_sec": self._settle_sec,
            "between_trials_sec": self._between_trials_sec,
            "note": "selected force values are replayed directly; no allocator or force curve is applied",
        }
        (self._output_dir / "replay_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=True) + "\n", encoding="utf-8"
        )

    def _publish(self, values: tuple[float, ...]) -> None:
        self._publisher.publish(Float32MultiArray(data=[float(value) for value in values]))

    def _publish_reset(self) -> None:
        self._reset_publisher.publish(Float32MultiArray(data=[1.0]))
        self._last_reset_monotonic = time.monotonic()

    def _write_command(self, row: ReplayRow, elapsed: float) -> None:
        self._writer.writerow(
            {
                "wall_time_sec": f"{time.time():.9f}",
                "elapsed_sec": f"{elapsed:.9f}",
                "trial_index": row.trial_index,
                "phase": row.phase,
                "level": f"{row.level:.9f}",
                **{column: f"{value:.9f}" for column, value in zip(THRUSTER_COLUMNS, row.force_n)},
            }
        )

    def _start_trial(self) -> None:
        self._trial_number += 1
        self._row_index = 0
        self._phase_start_monotonic = time.monotonic()
        self._state = "replaying"
        trial = self._trials[self._trial_number]
        self.get_logger().info(
            f"trial {trial[0].trial_index}: level={trial[0].level:g}, rows={len(trial)}, "
            f"duration={trial[-1].relative_time_sec:.2f}s"
        )

    def _finish(self) -> None:
        if self._done:
            return
        self._publish((0.0,) * 8)
        self._file.flush()
        self._file.close()
        self._done = True
        self.get_logger().info("replay complete; zero force is being held")

    def _tick(self) -> None:
        if self._done:
            return
        now = time.monotonic()
        if self._state == "initial_reset":
            if now - self._last_reset_monotonic >= 0.05:
                self._publish_reset()
                self._zero_sent += 1
            if self._zero_sent >= self._reset_count and now - self._start_monotonic >= self._settle_sec:
                self._start_trial()
            else:
                self._publish((0.0,) * 8)
            return

        if self._state == "between_trials":
            self._publish((0.0,) * 8)
            if now - self._phase_start_monotonic >= self._between_trials_sec:
                self._publish_reset()
                self._zero_sent = 0
                self._state = "resetting"
            return

        if self._state == "resetting":
            self._publish((0.0,) * 8)
            if now - self._last_reset_monotonic >= self._settle_sec:
                self._start_trial()
            return

        trial = self._trials[self._trial_number]
        elapsed = now - self._phase_start_monotonic
        while self._row_index + 1 < len(trial) and trial[self._row_index + 1].relative_time_sec <= elapsed:
            self._row_index += 1
        row = trial[self._row_index]
        self._publish(row.force_n)
        self._write_command(row, now - self._start_monotonic)
        if elapsed >= trial[-1].relative_time_sec:
            if self._trial_number + 1 >= len(self._trials):
                self._finish()
            else:
                self._phase_start_monotonic = now
                self._state = "between_trials"

    def close(self) -> None:
        self._publish((0.0,) * 8)
        if not self._file.closed:
            self._file.flush()
            self._file.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay measured 8-thruster forces into Unity Fossen hydrodynamics.")
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--topic", default="/sim/finsrov/thrusters_out")
    parser.add_argument("--reset-topic", default="/sim/finsrov/reset")
    parser.add_argument("--trial", type=int, action="append", dest="trials", default=None)
    parser.add_argument("--trial-limit", type=int, default=0)
    parser.add_argument("--settle-sec", type=float, default=1.0)
    parser.add_argument("--between-trials-sec", type=float, default=1.0)
    parser.add_argument("--reset-count", type=int, default=3)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument(
        "--force-source",
        choices=("command", "force_from_rpm"),
        default="command",
        help="CSV force columns to replay; use force_from_rpm for a like-for-like physical-force comparison",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected = sorted(set(args.trials)) if args.trials else None
    trials = _read_trial_rows(args.csv, selected, args.force_source)
    if args.trial_limit > 0:
        trials = trials[: args.trial_limit]
    rclpy.init()
    node = UnityHydrodynamicReplay(
        trials,
        args.output_dir,
        args.topic,
        args.reset_topic,
        args.settle_sec,
        args.between_trials_sec,
        args.reset_count,
        args.rate,
        args.force_source,
    )
    try:
        while rclpy.ok() and not node._done:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        pass
    finally:
        node.close()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
