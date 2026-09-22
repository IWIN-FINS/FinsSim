from __future__ import annotations

import argparse
import csv
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence

from hardware_bridge.direct_thruster_sender import CANONICAL_NAMES
from hardware_bridge.thruster_test_sender import THRUSTER_COUNT, _zeros


DEFAULT_THRUSTER_TOPIC = "/finsrov/thrusters_out"


@dataclass(frozen=True)
class SweepSample:
    phase: str
    command: float
    done: bool = False


@dataclass(frozen=True)
class SweepConfig:
    index: int
    amplitude: float
    pre_zero_sec: float
    marker_amplitude: float
    marker_on_sec: float
    marker_off_sec: float
    marker_count: int
    settle_sec: float
    hold_start_sec: float
    ramp_sec: float
    hold_end_sec: float
    post_zero_sec: float
    bidirectional: bool


def _positive_float(value: float, name: str, *, allow_zero: bool = True) -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")
    if allow_zero:
        if value < 0.0:
            raise ValueError(f"{name} must be >= 0")
    elif value <= 0.0:
        raise ValueError(f"{name} must be > 0")
    return value


def validate_sweep_config(config: SweepConfig) -> None:
    if config.index < 0 or config.index >= THRUSTER_COUNT:
        raise ValueError(f"--index must be in [0, {THRUSTER_COUNT - 1}], got {config.index}")
    if not math.isfinite(config.amplitude) or config.amplitude <= 0.0 or config.amplitude > 1.0:
        raise ValueError("--amplitude must be in (0, 1]")
    if not math.isfinite(config.marker_amplitude) or abs(config.marker_amplitude) > 1.0:
        raise ValueError("--marker-amplitude must be in [-1, 1]")
    if config.marker_count < 0:
        raise ValueError("--marker-count must be >= 0")
    _positive_float(config.pre_zero_sec, "--pre-zero")
    _positive_float(config.marker_on_sec, "--marker-on")
    _positive_float(config.marker_off_sec, "--marker-off")
    _positive_float(config.settle_sec, "--settle")
    _positive_float(config.hold_start_sec, "--hold-start")
    _positive_float(config.ramp_sec, "--ramp", allow_zero=False)
    _positive_float(config.hold_end_sec, "--hold-end")
    _positive_float(config.post_zero_sec, "--post-zero")


def total_duration(config: SweepConfig) -> float:
    validate_sweep_config(config)
    marker_time = config.marker_count * (config.marker_on_sec + config.marker_off_sec)
    one_way = config.hold_start_sec + config.ramp_sec + config.hold_end_sec
    sweep_time = one_way * (2 if config.bidirectional else 1)
    return config.pre_zero_sec + marker_time + config.settle_sec + sweep_time + config.post_zero_sec


def command_at_elapsed(elapsed_sec: float, config: SweepConfig) -> SweepSample:
    validate_sweep_config(config)
    t = max(0.0, float(elapsed_sec))

    if t < config.pre_zero_sec:
        return SweepSample("pre_zero", 0.0)
    t -= config.pre_zero_sec

    for marker_idx in range(config.marker_count):
        if t < config.marker_on_sec:
            return SweepSample(f"marker_{marker_idx + 1}_on", config.marker_amplitude)
        t -= config.marker_on_sec
        if t < config.marker_off_sec:
            return SweepSample(f"marker_{marker_idx + 1}_off", 0.0)
        t -= config.marker_off_sec

    if t < config.settle_sec:
        return SweepSample("settle_zero", 0.0)
    t -= config.settle_sec

    if t < config.hold_start_sec:
        return SweepSample("hold_negative", -config.amplitude)
    t -= config.hold_start_sec

    if t < config.ramp_sec:
        alpha = t / config.ramp_sec
        return SweepSample("ramp_negative_to_positive", -config.amplitude + 2.0 * config.amplitude * alpha)
    t -= config.ramp_sec

    if t < config.hold_end_sec:
        return SweepSample("hold_positive", config.amplitude)
    t -= config.hold_end_sec

    if config.bidirectional:
        if t < config.hold_start_sec:
            return SweepSample("hold_positive_return", config.amplitude)
        t -= config.hold_start_sec

        if t < config.ramp_sec:
            alpha = t / config.ramp_sec
            return SweepSample("ramp_positive_to_negative", config.amplitude - 2.0 * config.amplitude * alpha)
        t -= config.ramp_sec

        if t < config.hold_end_sec:
            return SweepSample("hold_negative_return", -config.amplitude)
        t -= config.hold_end_sec

    if t < config.post_zero_sec:
        return SweepSample("post_zero", 0.0)

    return SweepSample("done", 0.0, done=True)


def values_for_sample(sample: SweepSample, index: int) -> List[float]:
    values = _zeros()
    values[index] = sample.command
    return values


def _write_csv_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "wall_time_unix",
            "monotonic_time",
            "elapsed_sec",
            "phase",
            "target_index",
            "target_name",
            "target_command",
            *CANONICAL_NAMES,
        ]
    )


def _write_csv_row(
    writer: csv.writer,
    *,
    wall_time: float,
    monotonic_time: float,
    elapsed_sec: float,
    sample: SweepSample,
    config: SweepConfig,
) -> None:
    values = values_for_sample(sample, config.index)
    writer.writerow(
        [
            f"{wall_time:.6f}",
            f"{monotonic_time:.6f}",
            f"{elapsed_sec:.6f}",
            sample.phase,
            config.index,
            CANONICAL_NAMES[config.index],
            f"{sample.command:.6f}",
            *(f"{value:.6f}" for value in values),
        ]
    )


def create_sweep_node(
    *,
    topic: str,
    config: SweepConfig,
    rate_hz: float,
    csv_path: Path | None,
    send_stop_on_exit: bool,
):
    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    class ThrusterCurveSweep(Node):
        def __init__(self) -> None:
            super().__init__("thruster_curve_sweep")
            if rate_hz <= 0.0:
                raise ValueError("--rate must be > 0")

            self._pub = self.create_publisher(Float32MultiArray, topic, 10)
            self._config = config
            self._rate_hz = float(rate_hz)
            self._send_stop_on_exit = bool(send_stop_on_exit)
            self._start_monotonic = time.monotonic()
            self._csv_file = None
            self._csv_writer = None
            if csv_path is not None:
                csv_path.parent.mkdir(parents=True, exist_ok=True)
                self._csv_file = csv_path.open("w", newline="", encoding="utf-8")
                self._csv_writer = csv.writer(self._csv_file)
                _write_csv_header(self._csv_writer)

            duration = total_duration(self._config)
            self.get_logger().warn(
                "starting thruster curve sweep: "
                f"topic={topic}, index={config.index}({CANONICAL_NAMES[config.index]}), "
                f"amplitude=+/-{config.amplitude:.3f}, rate={self._rate_hz:.1f}Hz, "
                f"duration={duration:.2f}s, csv={str(csv_path) if csv_path else 'disabled'}"
            )
            self._timer = self.create_timer(1.0 / self._rate_hz, self._on_timer)

        def _on_timer(self) -> None:
            now_monotonic = time.monotonic()
            elapsed = now_monotonic - self._start_monotonic
            sample = command_at_elapsed(elapsed, self._config)
            if sample.done:
                if self._send_stop_on_exit:
                    self.publish_values(_zeros())
                self.get_logger().info(f"thruster curve sweep finished after {elapsed:.2f}s")
                self.close_csv()
                rclpy.shutdown()
                return

            values = values_for_sample(sample, self._config.index)
            self.publish_values(values)
            if self._csv_writer is not None:
                _write_csv_row(
                    self._csv_writer,
                    wall_time=time.time(),
                    monotonic_time=now_monotonic,
                    elapsed_sec=elapsed,
                    sample=sample,
                    config=self._config,
                )

        def publish_values(self, values: Sequence[float]) -> None:
            msg = Float32MultiArray()
            msg.data = list(values)
            self._pub.publish(msg)

        def close_csv(self) -> None:
            if self._csv_file is not None:
                self._csv_file.flush()
                self._csv_file.close()
                self._csv_file = None

        def destroy_node(self) -> bool:
            self.close_csv()
            return super().destroy_node()

    return ThrusterCurveSweep()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Publish an automated single-thruster sweep. "
            "Default topic is the normal hardware_bridge input: /finsrov/thrusters_out."
        )
    )
    parser.add_argument("--index", type=int, required=True, help="canonical thruster index [0..7]")
    parser.add_argument("--topic", default=DEFAULT_THRUSTER_TOPIC, help=f"output topic, default: {DEFAULT_THRUSTER_TOPIC}")
    parser.add_argument("--amplitude", type=float, default=1.0, help="sweep limit, default: 1.0")
    parser.add_argument("--rate", type=float, default=20.0, help="publish/log rate in Hz, default: 20")
    parser.add_argument("--pre-zero", type=float, default=5.0, help="initial zero-command duration, seconds")
    parser.add_argument("--marker-amplitude", type=float, default=0.2, help="sync marker command amplitude")
    parser.add_argument("--marker-on", type=float, default=0.5, help="sync marker on duration, seconds")
    parser.add_argument("--marker-off", type=float, default=0.5, help="sync marker off duration, seconds")
    parser.add_argument("--marker-count", type=int, default=2, help="number of sync marker pulses")
    parser.add_argument("--settle", type=float, default=3.0, help="zero-command settling time after markers")
    parser.add_argument("--hold-start", type=float, default=2.0, help="hold negative max before ramp, seconds")
    parser.add_argument("--ramp", type=float, default=60.0, help="linear ramp duration, seconds")
    parser.add_argument("--hold-end", type=float, default=2.0, help="hold positive max after ramp, seconds")
    parser.add_argument("--post-zero", type=float, default=5.0, help="final zero-command duration, seconds")
    parser.add_argument("--bidirectional", action="store_true", help="also sweep back from positive max to negative max")
    parser.add_argument("--csv", type=Path, default=None, help="write command timeline CSV")
    parser.add_argument("--no-stop-on-exit", action="store_true", help="do not publish a final all-zero frame")
    parser.add_argument("--print-plan", action="store_true", help="print total duration and exit without ROS publishing")
    return parser


def _config_from_args(args: argparse.Namespace) -> SweepConfig:
    return SweepConfig(
        index=args.index,
        amplitude=args.amplitude,
        pre_zero_sec=args.pre_zero,
        marker_amplitude=args.marker_amplitude,
        marker_on_sec=args.marker_on,
        marker_off_sec=args.marker_off,
        marker_count=args.marker_count,
        settle_sec=args.settle,
        hold_start_sec=args.hold_start,
        ramp_sec=args.ramp,
        hold_end_sec=args.hold_end,
        post_zero_sec=args.post_zero,
        bidirectional=args.bidirectional,
    )


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    config = _config_from_args(args)
    try:
        validate_sweep_config(config)
        duration = total_duration(config)
    except ValueError as exc:
        parser.error(str(exc))

    if args.print_plan:
        print(f"index={config.index} name={CANONICAL_NAMES[config.index]}")
        print(f"topic={args.topic}")
        print(f"duration_sec={duration:.3f}")
        print(f"sweep=-{config.amplitude:.3f} -> +{config.amplitude:.3f}")
        print(f"bidirectional={config.bidirectional}")
        return

    import rclpy

    rclpy.init()
    node = create_sweep_node(
        topic=args.topic,
        config=config,
        rate_hz=args.rate,
        csv_path=args.csv,
        send_stop_on_exit=not args.no_stop_on_exit,
    )
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        if not args.no_stop_on_exit:
            node.publish_values(_zeros())
        node.get_logger().info("thruster curve sweep interrupted")
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
