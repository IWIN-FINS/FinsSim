from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from hardware_bridge.direct_thruster_sender import CANONICAL_NAMES
THRUSTER_COUNT = len(CANONICAL_NAMES)
DEFAULT_COMMAND_TOPIC = "/finsrov/thrusters_out"
DEFAULT_RPM_TOPIC = "/finsrov/hardware/motor_rpm_raw"
DEFAULT_OUTPUT_ROOT = Path(
    "./ros2_ws/data/thruster/"
    "rpm_closed_loop/finsrov_v4_pro1"
)


@dataclass(frozen=True)
class Phase:
    name: str
    command: float
    duration_sec: float


@dataclass
class RpmSample:
    elapsed_sec: float
    phase: str
    command_norm: float
    target_rpm: float
    rpm: list[float]
    rpm_received_age_sec: float | None


def parse_amplitudes(value: str) -> list[float]:
    result = [float(item.strip()) for item in value.split(",") if item.strip()]
    if not result:
        raise ValueError("at least one amplitude is required")
    if any(not math.isfinite(item) or item <= 0.0 or item > 1.0 for item in result):
        raise ValueError("amplitudes must be finite and in (0, 1]")
    return result


def build_phases(
    amplitudes: Sequence[float],
    *,
    settle_sec: float,
    active_sec: float,
    rest_sec: float,
    include_negative: bool,
) -> list[Phase]:
    phases = [Phase("initial_zero", 0.0, settle_sec)]
    for index, amplitude in enumerate(amplitudes, start=1):
        phases.append(Phase(f"positive_{index:02d}", float(amplitude), active_sec))
        phases.append(Phase(f"rest_after_positive_{index:02d}", 0.0, rest_sec))
        if include_negative:
            phases.append(Phase(f"negative_{index:02d}", -float(amplitude), active_sec))
            phases.append(Phase(f"rest_after_negative_{index:02d}", 0.0, rest_sec))
    return phases


def _median(values: Sequence[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else default


def summarize_phase(
    samples: Sequence[RpmSample],
    phase: Phase,
    *,
    motor_index: int,
    target_rpm_max: float,
) -> dict[str, float | int | str | None]:
    active = [sample for sample in samples if sample.phase == phase.name]
    if not active:
        return {
            "phase": phase.name,
            "command_norm": phase.command,
            "target_rpm": phase.command * target_rpm_max,
            "sample_count": 0,
        }

    target = phase.command * target_rpm_max
    values = [sample.rpm[motor_index] for sample in active]
    tail_count = max(3, int(len(values) * 0.25))
    tail = values[-tail_count:]
    baseline = 0.0
    steady = _median(tail)
    peak = max(values) if target >= 0.0 else min(values)
    response = abs(steady - baseline)
    overshoot = max(0.0, abs(peak - baseline) - response)
    overshoot_ratio = overshoot / max(response, 1.0)
    error = steady - target

    t10 = None
    t90 = None
    if abs(response) >= 10.0:
        low = abs(response) * 0.1
        high = abs(response) * 0.9
        for sample in active:
            magnitude = abs(sample.rpm[motor_index] - baseline)
            if t10 is None and magnitude >= low:
                t10 = sample.elapsed_sec
            if t90 is None and magnitude >= high:
                t90 = sample.elapsed_sec
                break

    return {
        "phase": phase.name,
        "command_norm": phase.command,
        "target_rpm": target,
        "sample_count": len(active),
        "steady_rpm": steady,
        "peak_rpm": peak,
        "steady_error_rpm": error,
        "steady_abs_error_rpm": abs(error),
        "overshoot_ratio": overshoot_ratio,
        "t10_sec": t10,
        "t90_sec": t90,
        "rise_10_90_sec": None if t10 is None or t90 is None else t90 - t10,
    }


def save_csv(samples: Sequence[RpmSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "elapsed_sec",
        "phase",
        "command_norm",
        "target_rpm",
        "rpm_received_age_sec",
        *[f"rpm_{name}" for name in CANONICAL_NAMES],
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            row = {
                "elapsed_sec": f"{sample.elapsed_sec:.6f}",
                "phase": sample.phase,
                "command_norm": f"{sample.command_norm:.6f}",
                "target_rpm": f"{sample.target_rpm:.3f}",
                "rpm_received_age_sec": "" if sample.rpm_received_age_sec is None else f"{sample.rpm_received_age_sec:.6f}",
            }
            row.update({f"rpm_{name}": f"{value:.3f}" for name, value in zip(CANONICAL_NAMES, sample.rpm)})
            writer.writerow(row)


def save_plot(
    samples: Sequence[RpmSample],
    phases: Sequence[Phase],
    path: Path,
    *,
    motor_index: int,
    target_rpm_max: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        return

    times = [sample.elapsed_sec for sample in samples]
    target = [sample.target_rpm for sample in samples]
    actual = [sample.rpm[motor_index] for sample in samples]
    error = [actual_value - target_value for actual_value, target_value in zip(actual, target)]

    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    axes[0].plot(times, target, color="tab:orange", linewidth=1.8, label="target RPM")
    axes[0].plot(times, actual, color="tab:blue", linewidth=1.2, label="measured RPM")
    axes[0].set_ylabel("RPM")
    axes[0].set_title(f"{CANONICAL_NAMES[motor_index]} direct RPM closed-loop response")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="upper right")

    axes[1].plot(times, error, color="tab:red", linewidth=1.2, label="RPM error")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("actual - target (RPM)")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="upper right")

    elapsed = 0.0
    for phase in phases:
        start = elapsed
        elapsed += phase.duration_sec
        if phase.command != 0.0:
            for axis in axes:
                axis.axvspan(start, elapsed, color="tab:green", alpha=0.05)
                axis.text(
                    (start + elapsed) * 0.5,
                    0.96,
                    f"{phase.command:+.2f}",
                    transform=axis.get_xaxis_transform(),
                    ha="center",
                    va="top",
                    fontsize=8,
                )

    for axis in axes:
        axis.set_xlim(left=0.0)
        axis.tick_params(axis="both", which="both", direction="in")
    fig.savefig(path, dpi=160)
    plt.close(fig)


class RpmIdentificationNode(Node):
    def __init__(self, args: argparse.Namespace, phases: Sequence[Phase]) -> None:
        super().__init__("thruster_rpm_closed_loop_identification")
        self._args = args
        self._phases = list(phases)
        self._publisher = self.create_publisher(Float32MultiArray, args.command_topic, 10)
        self.create_subscription(Float32MultiArray, args.rpm_topic, self._rpm_callback, 20)
        self._created_time = time.monotonic()
        self._start_time: float | None = None
        self._phase_started: float | None = None
        self._started = False
        self._phase_index = 0
        self._last_rpm = [0.0] * THRUSTER_COUNT
        self._last_rpm_time: float | None = None
        self.samples: list[RpmSample] = []
        self.finished = False
        self._timer = self.create_timer(1.0 / args.rate_hz, self._on_timer)
        self.get_logger().warn(
            "direct RPM identification started: "
            f"motor={CANONICAL_NAMES[args.motor_index]}, amplitudes={args.amplitudes}, "
            f"target_rpm_max={args.target_rpm_max:.1f}, phases={len(self._phases)}"
        )

    @property
    def start_time(self) -> float:
        return self._start_time if self._start_time is not None else self._created_time

    def _rpm_callback(self, message: Float32MultiArray) -> None:
        if len(message.data) != THRUSTER_COUNT:
            return
        self._last_rpm = [float(value) if math.isfinite(float(value)) else 0.0 for value in message.data]
        self._last_rpm_time = time.monotonic()

    def _publish_command(self, command: float, *, enabled: bool) -> None:
        values = [0.0] * THRUSTER_COUNT
        values[self._args.motor_index] = command if enabled else 0.0
        message = Float32MultiArray()
        message.data = values
        self._publisher.publish(message)

    def _on_timer(self) -> None:
        now = time.monotonic()
        if not self._started:
            subscriber_count = self._publisher.get_subscription_count()
            if subscriber_count == 0:
                if now - self._created_time >= self._args.wait_for_subscriber_sec:
                    self.get_logger().error(
                        f"no subscriber matched {self._args.command_topic} within "
                        f"{self._args.wait_for_subscriber_sec:.1f}s; aborting without motor command"
                    )
                    self.finished = True
                    self._timer.cancel()
                return
            self._start_time = now
            self._phase_started = now
            self._started = True
            self.get_logger().info(
                f"command subscriber matched ({subscriber_count}); starting timed sequence"
            )

        assert self._start_time is not None
        assert self._phase_started is not None
        elapsed = now - self._start_time
        if self._phase_index >= len(self._phases):
            self._publish_command(0.0, enabled=False)
            self.get_logger().info(f"identification finished: {len(self.samples)} samples")
            self._timer.cancel()
            self.finished = True
            return

        phase = self._phases[self._phase_index]
        phase_elapsed = now - self._phase_started
        if phase_elapsed >= phase.duration_sec:
            self._phase_index += 1
            self._phase_started = now
            if self._phase_index >= len(self._phases):
                self._publish_command(0.0, enabled=False)
                self._timer.cancel()
                self.finished = True
                return
            phase = self._phases[self._phase_index]

        command = phase.command
        self._publish_command(command, enabled=abs(command) > 1e-9)
        age = None if self._last_rpm_time is None else max(0.0, now - self._last_rpm_time)
        self.samples.append(
            RpmSample(
                elapsed_sec=elapsed,
                phase=phase.name,
                command_norm=command,
                target_rpm=command * self._args.target_rpm_max,
                rpm=list(self._last_rpm),
                rpm_received_age_sec=age,
            )
        )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the firmware RPM loop directly. The command is normalized RPM, "
            "not force and not c1-derived. Canonical order is "
            "[V_LF,V_LB,V_RB,V_RF,H_LF,H_LB,H_RB,H_RF]."
        )
    )
    parser.add_argument("--motor-index", type=int, default=4, choices=range(THRUSTER_COUNT))
    parser.add_argument("--amplitudes", default="0.10,0.20,0.40,0.60")
    parser.add_argument("--target-rpm-max", type=float, default=3000.0)
    parser.add_argument("--settle-sec", type=float, default=2.0)
    parser.add_argument("--active-sec", type=float, default=3.0)
    parser.add_argument("--rest-sec", type=float, default=1.5)
    parser.add_argument("--rate-hz", type=float, default=50.0)
    parser.add_argument("--wait-for-subscriber-sec", type=float, default=5.0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--no-negative", action="store_true")
    parser.add_argument("--command-topic", default=DEFAULT_COMMAND_TOPIC)
    parser.add_argument("--rpm-topic", default=DEFAULT_RPM_TOPIC)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    try:
        args.amplitudes = parse_amplitudes(args.amplitudes)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    if args.target_rpm_max <= 0.0 or args.rate_hz <= 0.0:
        raise SystemExit("target-rpm-max and rate-hz must be positive")
    if args.wait_for_subscriber_sec < 0.0:
        raise SystemExit("wait-for-subscriber-sec must be >= 0")
    if args.settle_sec < 0.0 or args.active_sec <= 0.0 or args.rest_sec < 0.0:
        raise SystemExit("settle-sec/rest-sec must be >= 0 and active-sec must be > 0")
    if args.repeats < 1:
        raise SystemExit("repeats must be >= 1")

    base_phases = build_phases(
        args.amplitudes,
        settle_sec=args.settle_sec,
        active_sec=args.active_sec,
        rest_sec=args.rest_sec,
        include_negative=not args.no_negative,
    )
    phases: list[Phase] = []
    for repeat in range(args.repeats):
        suffix = "" if args.repeats == 1 else f"_r{repeat + 1:02d}"
        phases.extend(Phase(f"{phase.name}{suffix}", phase.command, phase.duration_sec) for phase in base_phases)

    output_dir = args.output_dir
    if output_dir is None:
        timestamp = time.strftime("run%Y%m%d_%H%M%S")
        output_dir = DEFAULT_OUTPUT_ROOT / f"{CANONICAL_NAMES[args.motor_index]}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = RpmIdentificationNode(args, phases)
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warn("interrupted; publishing zero command")
        if rclpy.ok():
            node._publish_command(0.0, enabled=False)
    finally:
        if node.samples:
            summaries = [
                summarize_phase(
                    node.samples,
                    phase,
                    motor_index=args.motor_index,
                    target_rpm_max=args.target_rpm_max,
                )
                for phase in phases
            ]
            save_csv(node.samples, output_dir / "rpm_closed_loop.csv")
            save_plot(
                node.samples,
                phases,
                output_dir / "rpm_closed_loop_response.png",
                motor_index=args.motor_index,
                target_rpm_max=args.target_rpm_max,
            )
            metadata = {
                "motor_index": args.motor_index,
                "motor_name": CANONICAL_NAMES[args.motor_index],
                "amplitudes": args.amplitudes,
                "target_rpm_max": args.target_rpm_max,
                "command_topic": args.command_topic,
                "rpm_topic": args.rpm_topic,
                "phases": [phase.__dict__ for phase in phases],
                "phase_summary": summaries,
                "sample_count": len(node.samples),
                "note": "command is normalized_rpm; no c1/force conversion is used",
            }
            (output_dir / "summary.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            print(f"saved RPM closed-loop data to {output_dir}")
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
