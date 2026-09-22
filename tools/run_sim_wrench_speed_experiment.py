#!/usr/bin/env python3
"""Measure full-scale FinsROV velocity response through ROS2 and the Unity gRPC adapter.

Run from ``ros2_ws`` so ``rclpy`` and the installed ``send_wrench_action``
entry point use the repository's ROS/uv environment:

  ./scripts/run_ros2_uv.sh python ../tools/run_sim_wrench_speed_experiment.py

This tool deliberately sends the existing ``send_wrench_action`` command. It
does not bypass the allocator, publish direct thruster forces, reset Unity, or
activate any scene component.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Callable

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TwistWithCovarianceStamped
from rclpy.node import Node
from sensor_msgs.msg import Imu
from std_msgs.msg import Float32MultiArray


SIM_THRUSTER_TOPIC = "/sim/finsrov/thrusters_out"
SIM_DVL_TOPIC = "/sim/finsrov/controller/dvl"
SIM_IMU_TOPIC = "/sim/finsrov/controller/imu"
SIM_POSE_TOPIC = "/sim/finsrov/controller/pose"
REPO_ROOT = Path(__file__).resolve().parents[1]
SIM_WRENCH_CONFIG = REPO_ROOT / "ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose_sim.yaml"


@dataclass(frozen=True)
class Trial:
    name: str
    action: tuple[float, float, float, float, float, float]
    source: str
    component: int
    unit: str

    @property
    def sign(self) -> float:
        value = self.action[[index for index, item in enumerate(self.action) if item != 0.0][0]]
        return 1.0 if value > 0.0 else -1.0


# Policy action order: [surge, sway, heave, roll, pitch, yaw].
# DVL is controller body [forward, up, left]; IMU follows the same body axes.
ALL_TRIALS = (
    Trial("surge_pos", (1, 0, 0, 0, 0, 0), "linear", 0, "m/s"),
    Trial("surge_neg", (-1, 0, 0, 0, 0, 0), "linear", 0, "m/s"),
    Trial("sway_pos", (0, 1, 0, 0, 0, 0), "linear", 2, "m/s"),
    Trial("sway_neg", (0, -1, 0, 0, 0, 0), "linear", 2, "m/s"),
    Trial("heave_pos", (0, 0, 1, 0, 0, 0), "linear", 1, "m/s"),
    Trial("heave_neg", (0, 0, -1, 0, 0, 0), "linear", 1, "m/s"),
    Trial("roll_pos", (0, 0, 0, 1, 0, 0), "angular", 0, "rad/s"),
    Trial("roll_neg", (0, 0, 0, -1, 0, 0), "angular", 0, "rad/s"),
    Trial("pitch_pos", (0, 0, 0, 0, 1, 0), "angular", 2, "rad/s"),
    Trial("pitch_neg", (0, 0, 0, 0, -1, 0), "angular", 2, "rad/s"),
    Trial("yaw_pos", (0, 0, 0, 0, 0, 1), "angular", 1, "rad/s"),
    Trial("yaw_neg", (0, 0, 0, 0, 0, -1), "angular", 1, "rad/s"),
)
PLANAR_TRIALS = tuple(trial for trial in ALL_TRIALS if trial.name.split("_")[0] in {"surge", "sway", "yaw"})


class ResponseRecorder(Node):
    def __init__(self) -> None:
        super().__init__("sim_wrench_speed_experiment")
        self.linear_samples: list[dict[str, object]] = []
        self.angular_samples: list[dict[str, object]] = []
        self.pose_samples: list[dict[str, object]] = []
        self.command_samples: list[dict[str, object]] = []
        self._zero_publisher = self.create_publisher(Float32MultiArray, SIM_THRUSTER_TOPIC, 10)
        self.create_subscription(TwistWithCovarianceStamped, SIM_DVL_TOPIC, self._on_dvl, 20)
        self.create_subscription(Imu, SIM_IMU_TOPIC, self._on_imu, 20)
        self.create_subscription(PoseWithCovarianceStamped, SIM_POSE_TOPIC, self._on_pose, 20)
        self.create_subscription(Float32MultiArray, SIM_THRUSTER_TOPIC, self._on_command, 20)

    @staticmethod
    def _now() -> float:
        return time.monotonic()

    def _on_dvl(self, msg: TwistWithCovarianceStamped) -> None:
        linear = msg.twist.twist.linear
        self.linear_samples.append({"t": self._now(), "x": linear.x, "y": linear.y, "z": linear.z})

    def _on_imu(self, msg: Imu) -> None:
        angular = msg.angular_velocity
        self.angular_samples.append({"t": self._now(), "x": angular.x, "y": angular.y, "z": angular.z})

    def _on_pose(self, msg: PoseWithCovarianceStamped) -> None:
        position = msg.pose.pose.position
        self.pose_samples.append({"t": self._now(), "x": position.x, "y": position.y, "z": position.z})

    def _on_command(self, msg: Float32MultiArray) -> None:
        values = [float(value) for value in msg.data]
        if len(values) == 8:
            self.command_samples.append({"t": self._now(), "values": values})

    def publish_zero(self, count: int = 3) -> None:
        zero = Float32MultiArray(data=[0.0] * 8)
        for _ in range(count):
            self._zero_publisher.publish(zero)
            rclpy.spin_once(self, timeout_sec=0.05)
            time.sleep(0.05)


def _spin_for(node: Node, duration_sec: float) -> None:
    deadline = time.monotonic() + duration_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)


def _wait_until(node: Node, predicate: Callable[[], bool], timeout_sec: float, description: str) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        if predicate():
            return
    raise RuntimeError(f"timed out waiting for {description}")


def _numeric_samples(samples: list[dict[str, object]], start: int, begin: float, end: float) -> list[dict[str, object]]:
    return [sample for sample in samples[start:] if begin <= float(sample["t"]) <= end]


def _median_vector(samples: list[dict[str, object]]) -> list[float]:
    if not samples:
        return [float("nan")] * 3
    return [float(median([float(sample[axis]) for sample in samples])) for axis in ("x", "y", "z")]


def _percentile(values: list[float], fraction: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = int(round((len(ordered) - 1) * fraction))
    return float(ordered[index])


def _write_csv(path: Path, rows: list[dict[str, object]], headings: tuple[str, ...]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=headings)
        writer.writeheader()
        writer.writerows(rows)


def _trial_metrics(
    trial: Trial,
    samples: list[dict[str, object]],
    baseline: list[dict[str, object]],
    active_start: float,
    active_end: float,
    ramp_time: float,
    steady_window: float,
) -> dict[str, object]:
    measure_start = active_start + ramp_time
    measured = [sample for sample in samples if measure_start <= float(sample["t"]) <= active_end]
    if not measured:
        raise RuntimeError(f"{trial.name}: no {trial.source} samples while command was active")
    steady_start = max(measure_start, active_end - steady_window)
    steady = [sample for sample in measured if float(sample["t"]) >= steady_start]
    primary_axis = ("x", "y", "z")[trial.component]
    primary = [float(sample[primary_axis]) for sample in measured]
    aligned = [trial.sign * value for value in primary]
    cross_axes = [axis for axis in ("x", "y", "z") if axis != primary_axis]
    cross_peak = max(abs(float(sample[axis])) for sample in measured for axis in cross_axes)
    return {
        "trial": trial.name,
        "policy_action": list(trial.action),
        "source": trial.source,
        "primary_axis": primary_axis,
        "unit": trial.unit,
        "active_duration_sec": active_end - active_start,
        "sample_count": len(measured),
        "baseline_median": _median_vector(baseline),
        "peak_abs": max(abs(value) for value in primary),
        "p95_abs": _percentile([abs(value) for value in primary], 0.95),
        "peak_along_command": max(aligned),
        "steady_median_along_command": median(
            trial.sign * float(sample[primary_axis]) for sample in steady
        ),
        "cross_axis_peak_abs": cross_peak,
    }


def _sender_command(args: argparse.Namespace, action: tuple[float, ...]) -> list[str]:
    return [
        "ros2",
        "run",
        "motion_control",
        "send_wrench_action",
        "--action",
        *(str(value) for value in action),
        "--duration",
        str(args.hold_sec),
        "--ramp-time",
        str(args.ramp_sec),
        "--rate",
        str(args.rate_hz),
        "--topic",
        SIM_THRUSTER_TOPIC,
        "--config",
        str(args.config),
        "--match-timeout",
        "2",
        "--zero-count",
        "5",
    ]


def _run_trial(node: ResponseRecorder, trial: Trial, args: argparse.Namespace, output_dir: Path) -> dict[str, object]:
    _spin_for(node, args.settle_sec)
    sample_source = node.linear_samples if trial.source == "linear" else node.angular_samples
    before_sample = len(sample_source)
    before_command = len(node.command_samples)
    baseline_start = time.monotonic() - min(args.settle_sec, 2.0)
    baseline = [sample for sample in sample_source if float(sample["t"]) >= baseline_start]
    command = _sender_command(args, trial.action)
    started = time.monotonic()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    try:
        while process.poll() is None:
            rclpy.spin_once(node, timeout_sec=0.05)
    except BaseException:
        process.terminate()
        process.wait(timeout=3)
        raise
    sender_log, _ = process.communicate(timeout=3)
    (output_dir / f"{trial.name}_sender.log").write_text(sender_log or "", encoding="utf-8")
    if process.returncode != 0:
        raise RuntimeError(f"{trial.name}: send_wrench_action failed with exit code {process.returncode}")
    _spin_for(node, 0.35)
    command_samples = node.command_samples[before_command:]
    active = [
        sample
        for sample in command_samples
        if max(abs(value) for value in sample["values"]) > 1e-4
    ]
    if not active:
        raise RuntimeError(f"{trial.name}: no nonzero command returned on {SIM_THRUSTER_TOPIC}")
    result = _trial_metrics(
        trial,
        sample_source[before_sample:],
        baseline,
        float(active[0]["t"]),
        float(active[-1]["t"]),
        args.ramp_sec,
        args.steady_window_sec,
    )
    result["sender_started_monotonic"] = started
    return result


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--axes", choices=("all", "planar"), default="all")
    parser.add_argument("--hold-sec", type=float, default=8.0, help="Full-scale hold duration after the ramp.")
    parser.add_argument("--ramp-sec", type=float, default=1.0, help="Zero-to-full action ramp duration.")
    parser.add_argument("--settle-sec", type=float, default=3.0, help="Zero-command settling duration between trials.")
    parser.add_argument("--steady-window-sec", type=float, default=2.0)
    parser.add_argument("--rate-hz", type=float, default=20.0)
    parser.add_argument("--config", type=Path, default=SIM_WRENCH_CONFIG)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/runs/diagnostics/sim_wrench_speed",
        help="Parent directory for timestamped raw CSV and summary files.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if min(args.hold_sec, args.ramp_sec, args.settle_sec, args.steady_window_sec) < 0.0 or args.rate_hz <= 0.0:
        raise SystemExit("durations must be non-negative and --rate-hz must be positive")
    if not args.config.is_file():
        raise SystemExit(f"wrench config does not exist: {args.config}")
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)
    trials = ALL_TRIALS if args.axes == "all" else PLANAR_TRIALS

    rclpy.init(args=None)
    node = ResponseRecorder()
    summaries: list[dict[str, object]] = []
    try:
        _wait_until(node, lambda: bool(node.linear_samples), 5.0, SIM_DVL_TOPIC)
        _wait_until(node, lambda: bool(node.angular_samples), 5.0, SIM_IMU_TOPIC)
        _wait_until(node, lambda: bool(node.pose_samples), 5.0, SIM_POSE_TOPIC)
        print(f"streams ready; running {len(trials)} full-scale trials through {SIM_THRUSTER_TOPIC}", flush=True)
        for trial in trials:
            print(f"[{trial.name}] sending {trial.action}", flush=True)
            summary = _run_trial(node, trial, args, output_dir)
            summaries.append(summary)
            print(
                f"[{trial.name}] peak={summary['peak_abs']:.4f} {trial.unit}; "
                f"steady={summary['steady_median_along_command']:.4f} {trial.unit}; "
                f"cross={summary['cross_axis_peak_abs']:.4f} {trial.unit}",
                flush=True,
            )
    finally:
        node.publish_zero()
        _write_csv(output_dir / "linear_velocity.csv", node.linear_samples, ("t", "x", "y", "z"))
        _write_csv(output_dir / "angular_velocity.csv", node.angular_samples, ("t", "x", "y", "z"))
        _write_csv(output_dir / "pose.csv", node.pose_samples, ("t", "x", "y", "z"))
        command_rows = [
            {"t": sample["t"], **{f"thruster_{index + 1}": value for index, value in enumerate(sample["values"])}}
            for sample in node.command_samples
        ]
        _write_csv(output_dir / "thruster_command.csv", command_rows, ("t", *(f"thruster_{index}" for index in range(1, 9))))
        _write_csv(
            output_dir / "summary.csv",
            summaries,
            (
                "trial", "policy_action", "source", "primary_axis", "unit", "active_duration_sec", "sample_count",
                "baseline_median", "peak_abs", "p95_abs", "peak_along_command", "steady_median_along_command",
                "cross_axis_peak_abs", "sender_started_monotonic",
            ),
        )
        (output_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
        node.destroy_node()
        rclpy.shutdown()
    print(f"results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
