from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

from hardware_bridge.direct_thruster_sender import CANONICAL_NAMES
from hardware_bridge.thruster_test_sender import _zeros


THRUSTER_COUNT = len(CANONICAL_NAMES)
DEFAULT_TOPIC = "/finsrov/thrusters_out"
DEFAULT_RPM_TOPIC = "/finsrov/hardware/motor_rpm_raw"
DEFAULT_C1_CONFIG = Path(
    "./ros2_ws/src/"
    "thruster_curve_measurement/config/thruster_rpm_force_unity.json"
)


@dataclass(frozen=True)
class Sample:
    elapsed_sec: float
    phase: str
    command_n: float
    rpm: float | None
    rpm_age_sec: float | None
    omega_rad_s: float | None
    c1: float | None
    inferred_thrust_n: float | None


@dataclass(frozen=True)
class TransitionFit:
    label: str
    direction: str
    command_n: float
    t_start_sec: float
    t_end_sec: float
    baseline_thrust_n: float
    final_thrust_n: float
    gain_n_per_n: float
    delay_sec: float
    time_constant_sec: float
    rmse_n: float
    sample_count: int
    peak_thrust_n: float
    overshoot_ratio: float
    first_order_adequate: bool


def parse_float_list(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one comma-separated value")
    if any(not math.isfinite(value) for value in values):
        raise ValueError("all values must be finite")
    return values


def load_c1_values(path: Path) -> list[tuple[float, float]]:
    """Return (c1_positive, c1_negative) for canonical thrusters."""
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    motors = data.get("motors", {})
    result: list[tuple[float, float]] = []
    for index, name in enumerate(CANONICAL_NAMES):
        entry = None
        for motor in motors.values():
            if int(motor.get("canonical_index", -1)) == index:
                entry = motor
                break
        if entry is None:
            groups = data.get("identified_groups", {})
            group_name = "normal_propeller" if index % 2 == 0 else "reverse_propeller"
            entry = groups.get(group_name)
        if entry is None:
            raise ValueError(f"no c1 value found for canonical thruster {index} ({name})")
        positive = float(entry["c1_positive_omega"])
        negative = float(entry["c1_negative_omega"])
        if positive <= 0.0 or negative <= 0.0:
            raise ValueError(f"c1 values must be positive for thruster {index}")
        result.append((positive, negative))
    return result


def inferred_thrust(rpm: float, c1_positive: float, c1_negative: float) -> tuple[float, float, float]:
    omega = float(rpm) * (2.0 * math.pi / 60.0)
    c1 = c1_positive if omega >= 0.0 else c1_negative
    thrust = c1 * omega * abs(omega)
    return omega, c1, thrust


def _median(values: Iterable[float], default: float = 0.0) -> float:
    values = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(values) if values else default


def _linspace(start: float, stop: float, count: int) -> list[float]:
    if count <= 1:
        return [start]
    return [start + (stop - start) * index / (count - 1) for index in range(count)]


def fit_transition(
    samples: Sequence[Sample],
    *,
    label: str,
    direction: str,
    command_n: float,
    start_sec: float,
    end_sec: float,
) -> TransitionFit | None:
    """Fit y=y0+(y1-y0)(1-exp(-(t-Td)/tau)) after a known command edge."""
    valid = [
        sample
        for sample in samples
        if sample.inferred_thrust_n is not None and math.isfinite(sample.inferred_thrust_n)
    ]
    before = [
        sample.inferred_thrust_n
        for sample in valid
        if start_sec - 0.35 <= sample.elapsed_sec < start_sec - 0.03
    ]
    active = [
        sample
        for sample in valid
        if start_sec <= sample.elapsed_sec <= end_sec
    ]
    after = [
        sample.inferred_thrust_n
        for sample in valid
        if end_sec + 0.35 <= sample.elapsed_sec <= end_sec + 1.5
    ]
    if len(active) < 8:
        return None

    y0 = _median(before, 0.0)
    if direction == "rise":
        tail = [sample.inferred_thrust_n for sample in active if sample.elapsed_sec >= end_sec - 0.25]
        y1 = _median(tail, _median(sample.inferred_thrust_n for sample in active))
        fit_start = start_sec
        fit_end = end_sec
    else:
        head = [sample.inferred_thrust_n for sample in valid if start_sec - 0.25 <= sample.elapsed_sec < start_sec]
        y0 = _median(head, 0.0)
        y1 = _median(after, 0.0)
        fit_start = start_sec
        fit_end = min(end_sec + 1.5, samples[-1].elapsed_sec)
        active = [sample for sample in valid if fit_start <= sample.elapsed_sec <= fit_end]
        if len(active) < 8:
            return None

    delta = y1 - y0
    if abs(delta) < 1e-5 or abs(command_n) < 1e-6:
        return None

    duration = max(0.05, fit_end - fit_start)
    dt_candidates = [
        sample.elapsed_sec - previous.elapsed_sec
        for previous, sample in zip(active, active[1:])
        if sample.elapsed_sec > previous.elapsed_sec
    ]
    sample_dt = _median(dt_candidates, 0.02)
    max_delay = min(0.8, max(sample_dt, duration * 0.45))
    delay_candidates = _linspace(0.0, max_delay, 41)
    min_tau = max(sample_dt, 0.005)
    max_tau = max(0.1, duration * 2.0)
    tau_candidates = [
        math.exp(math.log(min_tau) + (math.log(max_tau) - math.log(min_tau)) * index / 40.0)
        for index in range(41)
    ]

    best: tuple[float, float, float] | None = None
    for delay in delay_candidates:
        for tau in tau_candidates:
            error_sum = 0.0
            for sample in active:
                shifted = sample.elapsed_sec - fit_start - delay
                response = 0.0 if shifted <= 0.0 else 1.0 - math.exp(-shifted / tau)
                predicted = y0 + delta * response
                error_sum += (sample.inferred_thrust_n - predicted) ** 2  # type: ignore[operator]
            if best is None or error_sum < best[0]:
                best = (error_sum, delay, tau)
    assert best is not None
    rmse = math.sqrt(best[0] / len(active))
    observed = [float(sample.inferred_thrust_n) for sample in active if sample.inferred_thrust_n is not None]
    peak = max(observed) if delta >= 0.0 else min(observed)
    overshoot = max(0.0, peak - y1) if delta >= 0.0 else max(0.0, y1 - peak)
    overshoot_ratio = overshoot / max(abs(delta), 1e-6)
    normalized_rmse = rmse / max(abs(delta), 1e-6)
    return TransitionFit(
        label=label,
        direction=direction,
        command_n=command_n,
        t_start_sec=fit_start,
        t_end_sec=fit_end,
        baseline_thrust_n=y0,
        final_thrust_n=y1,
        gain_n_per_n=delta / command_n,
        delay_sec=best[1],
        time_constant_sec=best[2],
        rmse_n=rmse,
        sample_count=len(active),
        peak_thrust_n=peak,
        overshoot_ratio=overshoot_ratio,
        first_order_adequate=overshoot_ratio <= 0.10 and normalized_rmse <= 0.10,
    )


def fit_pulses(samples: Sequence[Sample], pulse_edges: Sequence[tuple[str, float, float, float]]) -> list[TransitionFit]:
    fits: list[TransitionFit] = []
    for label, command_n, start_sec, end_sec in pulse_edges:
        direction = "rise" if command_n > 0.0 else "rise"
        fit = fit_transition(
            samples,
            label=label,
            direction=direction,
            command_n=command_n,
            start_sec=start_sec,
            end_sec=end_sec,
        )
        if fit is not None:
            fits.append(fit)
    return fits


def aggregate_fits(fits: Sequence[TransitionFit]) -> dict[str, dict[str, float | int]]:
    result: dict[str, dict[str, float | int]] = {}
    for sign in ("positive", "negative"):
        selected = [fit for fit in fits if (fit.command_n > 0.0) == (sign == "positive")]
        if not selected:
            continue
        result[sign] = {
            "sample_count": len(selected),
            "delay_sec_median": statistics.median(fit.delay_sec for fit in selected),
            "time_constant_sec_median": statistics.median(fit.time_constant_sec for fit in selected),
            "gain_n_per_n_median": statistics.median(fit.gain_n_per_n for fit in selected),
            "rmse_n_median": statistics.median(fit.rmse_n for fit in selected),
            "overshoot_ratio_median": statistics.median(fit.overshoot_ratio for fit in selected),
            "first_order_adequate": all(fit.first_order_adequate for fit in selected),
        }
    return result


def save_plot(
    samples: Sequence[Sample],
    fits: Sequence[TransitionFit],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_path.parent.mkdir(parents=True, exist_ok=True)
    valid = [sample for sample in samples if sample.inferred_thrust_n is not None]
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    axes[0].plot(
        [sample.elapsed_sec for sample in samples],
        [sample.command_n for sample in samples],
        label="command force",
        color="tab:orange",
    )
    axes[0].set_ylabel("command (N)")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].plot(
        [sample.elapsed_sec for sample in valid],
        [sample.inferred_thrust_n for sample in valid],
        label="inferred thrust from RPM",
        color="tab:blue",
    )
    for fit in fits:
        xs = _linspace(fit.t_start_sec, fit.t_end_sec, 100)
        delta = fit.final_thrust_n - fit.baseline_thrust_n
        ys = [
            fit.baseline_thrust_n
            + delta
            * (0.0 if x - fit.t_start_sec <= fit.delay_sec else 1.0 - math.exp(-(x - fit.t_start_sec - fit.delay_sec) / fit.time_constant_sec))
            for x in xs
        ]
        axes[1].plot(
            xs,
            ys,
            linewidth=2,
            label=(
                f"{fit.label}: Td={fit.delay_sec:.3f}s, tau={fit.time_constant_sec:.3f}s, "
                f"overshoot={fit.overshoot_ratio:.0%}"
            ),
        )
    axes[1].set_xlabel("time (s)")
    axes[1].set_ylabel("inferred thrust (N)")
    axes[1].grid(True)
    axes[1].legend()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_csv(samples: Sequence[Sample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "elapsed_sec",
                "phase",
                "command_n",
                "rpm",
                "rpm_age_sec",
                "omega_rad_s",
                "c1",
                "inferred_thrust_n",
            ]
        )
        for sample in samples:
            writer.writerow(
                [
                    f"{sample.elapsed_sec:.6f}",
                    sample.phase,
                    f"{sample.command_n:.6f}",
                    "" if sample.rpm is None else f"{sample.rpm:.6f}",
                    "" if sample.rpm_age_sec is None else f"{sample.rpm_age_sec:.6f}",
                    "" if sample.omega_rad_s is None else f"{sample.omega_rad_s:.9f}",
                    "" if sample.c1 is None else f"{sample.c1:.12g}",
                    "" if sample.inferred_thrust_n is None else f"{sample.inferred_thrust_n:.9f}",
                ]
            )


def write_summary(
    path: Path,
    *,
    index: int,
    c1_config: Path,
    commands: Sequence[float],
    fits: Sequence[TransitionFit],
    sample_count: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": "first_order_plus_pure_delay_on_inferred_thrust",
        "transfer_function": "K/(tau*s+1)*exp(-Td*s)",
        "input": "force_n command on /finsrov/thrusters_out",
        "output": "thrust inferred from actual RPM using existing c1 curve",
        "thruster_index": index,
        "thruster_name": CANONICAL_NAMES[index],
        "c1_config": str(c1_config),
        "commands_n": list(commands),
        "sample_count": sample_count,
        "transitions": [fit.__dict__ for fit in fits],
        "aggregate": aggregate_fits(fits),
        "limitations": [
            "Thrust is inferred from RPM and static c1; it is not a direct force measurement.",
            "The fit includes command publication, ESC and motor dynamics unless separately timestamped.",
            "Static c1 does not model inflow, voltage sag, propeller interaction or vehicle-body flow.",
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Identify a thruster first-order-plus-dead-time response using measured RPM and an existing c1 curve."
    )
    parser.add_argument("--index", type=int, required=True, help="canonical thruster index [0..7]")
    parser.add_argument("--topic", default=DEFAULT_TOPIC)
    parser.add_argument("--rpm-topic", default=DEFAULT_RPM_TOPIC)
    parser.add_argument("--c1-config", type=Path, default=DEFAULT_C1_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--commands", default="0.5,-0.5", help="force-N pulse values, e.g. 0.5,1.0,-0.5,-1.0")
    parser.add_argument("--pre-zero", type=float, default=2.0)
    parser.add_argument("--hold", type=float, default=2.5)
    parser.add_argument("--zero-rest", type=float, default=1.5)
    parser.add_argument("--post-zero", type=float, default=2.0)
    parser.add_argument("--rate", type=float, default=50.0)
    parser.add_argument("--rpm-stale-sec", type=float, default=0.25)
    parser.add_argument("--allow-other-publishers", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="print schedule without publishing")
    return parser


def _schedule(commands: Sequence[float], pre_zero: float, hold: float, zero_rest: float, post_zero: float):
    t = pre_zero
    edges: list[tuple[str, float, float, float]] = []
    phases: list[tuple[float, float, str, float]] = [(0.0, pre_zero, "pre_zero", 0.0)]
    for index, command in enumerate(commands):
        start = t
        end = start + hold
        edges.append((f"pulse_{index + 1}", command, start, end))
        phases.append((start, end, f"pulse_{index + 1}", command))
        phases.append((end, end + zero_rest, f"zero_{index + 1}", 0.0))
        t = end + zero_rest
    phases.append((t, t + post_zero, "post_zero", 0.0))
    return phases, edges, t + post_zero


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.index < 0 or args.index >= THRUSTER_COUNT:
        parser.error(f"--index must be in [0, {THRUSTER_COUNT - 1}]")
    try:
        commands = parse_float_list(args.commands)
        c1_values = load_c1_values(args.c1_config)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        parser.error(str(exc))
    if any(abs(command) < 1e-6 for command in commands):
        parser.error("--commands must not contain zero; zero is inserted between pulses")
    if args.pre_zero < 0 or args.hold <= 0 or args.zero_rest < 0 or args.post_zero < 0 or args.rate <= 0:
        parser.error("pre-zero/zero-rest/post-zero must be >=0, hold/rate must be >0")

    phases, pulse_edges, total_duration = _schedule(commands, args.pre_zero, args.hold, args.zero_rest, args.post_zero)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        print(f"thruster={args.index} {CANONICAL_NAMES[args.index]}")
        print(f"commands_n={list(commands)} total_duration_sec={total_duration:.3f}")
        for start, end, phase, command in phases:
            print(f"{start:.3f}..{end:.3f} {phase} command={command:+.3f} N")
        return

    import rclpy
    from rclpy.node import Node
    from std_msgs.msg import Float32MultiArray

    class DynamicIdentifier(Node):
        def __init__(self) -> None:
            super().__init__("thruster_dynamic_identification")
            self.publisher = self.create_publisher(Float32MultiArray, args.topic, 10)
            self.rpm_values: list[float] | None = None
            self.rpm_stamp = 0.0
            self.samples: list[Sample] = []
            self.started = time.monotonic()
            self.finished = False
            self.aborted = False
            self.checked_publishers = False
            self.timer = self.create_timer(1.0 / args.rate, self.tick)
            self.create_subscription(Float32MultiArray, args.rpm_topic, self.rpm_callback, 10)

        def rpm_callback(self, msg) -> None:
            values = [float(value) if math.isfinite(float(value)) else 0.0 for value in msg.data]
            if len(values) >= THRUSTER_COUNT:
                self.rpm_values = values[:THRUSTER_COUNT]
                self.rpm_stamp = time.monotonic()

        def current_phase(self, elapsed: float) -> tuple[str, float]:
            for start, end, phase, command in phases:
                if start <= elapsed < end:
                    return phase, command
            return "done", 0.0

        def publish(self, command: float) -> None:
            values = _zeros()
            values[args.index] = command
            msg = Float32MultiArray()
            msg.data = values
            self.publisher.publish(msg)

        def tick(self) -> None:
            elapsed = time.monotonic() - self.started
            if not self.checked_publishers and elapsed > 0.2:
                publishers = self.get_publishers_info_by_topic(args.topic)
                other_count = sum(1 for info in publishers if info.node_name != self.get_name())
                self.checked_publishers = True
                if other_count and not args.allow_other_publishers:
                    self.get_logger().error(
                        f"refusing test: {other_count} other publisher(s) detected on {args.topic}; "
                        "stop motion_controller first or pass --allow-other-publishers explicitly"
                    )
                    self.publish(0.0)
                    self.aborted = True
                    self.finished = True
                    return

            phase, command = self.current_phase(elapsed)
            self.publish(command)
            rpm = None
            age = None
            omega = None
            c1 = None
            thrust = None
            if self.rpm_values is not None:
                age = time.monotonic() - self.rpm_stamp
                if age <= args.rpm_stale_sec:
                    rpm = self.rpm_values[args.index]
                    omega, c1, thrust = inferred_thrust(rpm, *c1_values[args.index])
            self.samples.append(Sample(elapsed, phase, command, rpm, age, omega, c1, thrust))
            if elapsed >= total_duration:
                self.publish(0.0)
                self.finished = True
                self.timer.cancel()

    rclpy.init()
    node = DynamicIdentifier()
    try:
        while rclpy.ok() and not node.finished:
            rclpy.spin_once(node, timeout_sec=0.1)
    except KeyboardInterrupt:
        node.get_logger().warn("interrupted; publishing zero command")
        node.publish(0.0)
    finally:
        node.publish(0.0)
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

    if node.aborted:
        raise SystemExit(2)
    raw_path = output_dir / "raw.csv"
    summary_path = output_dir / "fit.json"
    plot_path = output_dir / "response.png"
    write_csv(node.samples, raw_path)
    fits = fit_pulses(node.samples, pulse_edges)
    write_summary(
        summary_path,
        index=args.index,
        c1_config=args.c1_config,
        commands=commands,
        fits=fits,
        sample_count=len(node.samples),
    )
    if any(sample.inferred_thrust_n is not None for sample in node.samples):
        try:
            save_plot(node.samples, fits, plot_path)
        except ImportError:
            node.get_logger().warning("matplotlib is unavailable; response.png was not generated")
    print(f"raw_csv={raw_path}")
    print(f"fit_json={summary_path}")
    print(f"response_plot={plot_path}")
    print(json.dumps(aggregate_fits(fits), indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
