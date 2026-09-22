from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import rclpy
from msgs.msg import HardwareTelemetry, ThrusterCommandEcho
from hardware_bridge.direct_thruster_sender import CANONICAL_NAMES
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray


THRUSTER_COUNT = len(CANONICAL_NAMES)
DEFAULT_COMMAND_TOPIC = "/finsrov/thrusters_out"
DEFAULT_RPM_TOPIC = "/finsrov/hardware/telemetry"
DEFAULT_ECHO_TOPIC = "/finsrov/hardware/thruster_cmd_echo"


@dataclass(frozen=True)
class Phase:
    name: str
    command_norm: float
    duration_sec: float


@dataclass(frozen=True)
class RpmSample:
    host_elapsed_sec: float
    host_received_sec: float
    phase: str
    command_norm: float
    target_rpm: float
    mcu_time_ms: float
    telemetry_sequence: int
    rpm: float


@dataclass(frozen=True)
class EchoSample:
    host_elapsed_sec: float
    host_received_sec: float
    mcu_time_ms: float
    telemetry_sequence: int
    command_count: int
    enabled: bool


@dataclass(frozen=True)
class StepFit:
    label: str
    phase: str
    input_command_norm: float
    input_target_rpm: float
    edge_mcu_time_ms: float
    baseline_rpm: float
    final_rpm: float
    gain_rpm_per_rpm: float
    delay_sec: float
    time_constant_sec: float
    rmse_rpm: float
    r2: float
    peak_rpm: float
    overshoot_ratio: float
    sample_count: int
    first_order_adequate: bool


def parse_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("value must be finite")
    return result


def build_phases(
    *,
    target_rpm: float,
    initial_zero_sec: float,
    active_sec: float,
    zero_sec: float,
) -> list[Phase]:
    command_norm = abs(target_rpm) / 3000.0
    return [
        Phase("initial_zero", 0.0, initial_zero_sec),
        Phase("positive", command_norm, active_sec),
        Phase("zero_after_positive", 0.0, zero_sec),
        Phase("negative", -command_norm, active_sec),
        Phase("zero_after_negative", 0.0, zero_sec),
    ]


def _median(values: Iterable[float], default: float = 0.0) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return statistics.median(finite) if finite else default


def _percentile(values: Sequence[float], fraction: float, default: float = 0.0) -> float:
    finite = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not finite:
        return default
    index = min(len(finite) - 1, max(0, int(round(fraction * (len(finite) - 1)))))
    return finite[index]


def _fit_model(t: float, delay: float, tau: float) -> float:
    if t <= delay:
        return 0.0
    return 1.0 - math.exp(-(t - delay) / max(tau, 1e-9))


def fit_step(
    samples: Sequence[RpmSample],
    *,
    phase: Phase,
    edge_mcu_time_ms: float,
    next_edge_mcu_time_ms: float,
    previous_edge_mcu_time_ms: float | None,
    target_rpm_max: float,
) -> StepFit | None:
    duration_sec = max(0.1, (next_edge_mcu_time_ms - edge_mcu_time_ms) / 1000.0)
    active = [
        sample
        for sample in samples
        if edge_mcu_time_ms <= sample.mcu_time_ms <= next_edge_mcu_time_ms
    ]
    if len(active) < 8:
        return None

    baseline_start = edge_mcu_time_ms - min(1000.0, max(250.0, duration_sec * 250.0))
    if previous_edge_mcu_time_ms is not None:
        baseline_start = max(baseline_start, previous_edge_mcu_time_ms)
    baseline_samples = [
        sample.rpm
        for sample in samples
        if baseline_start <= sample.mcu_time_ms < edge_mcu_time_ms - 20.0
    ]
    baseline = _median(baseline_samples, 0.0)
    tail_start = next_edge_mcu_time_ms - min(1000.0, max(250.0, duration_sec * 0.25 * 1000.0))
    tail_samples = [sample.rpm for sample in active if sample.mcu_time_ms >= tail_start]
    final = _median(tail_samples, _median(sample.rpm for sample in active))
    delta = final - baseline
    if abs(delta) < 1.0:
        return None

    sample_times = [(sample.mcu_time_ms - edge_mcu_time_ms) / 1000.0 for sample in active]
    min_dt = _median(
        [b - a for a, b in zip(sample_times, sample_times[1:]) if b > a],
        0.01,
    )
    max_delay = min(1.5, max(0.05, duration_sec * 0.6))
    delay_candidates = [
        max(0.0, max_delay * index / 80.0)
        for index in range(81)
    ]
    min_tau = max(0.005, min_dt * 0.5)
    max_tau = max(0.1, duration_sec * 2.0)
    tau_candidates = [
        math.exp(math.log(min_tau) + (math.log(max_tau) - math.log(min_tau)) * index / 80.0)
        for index in range(81)
    ]

    best: tuple[float, float, float] | None = None
    for delay in delay_candidates:
        for tau in tau_candidates:
            error = 0.0
            for sample, t in zip(active, sample_times):
                predicted = baseline + delta * _fit_model(t, delay, tau)
                error += (sample.rpm - predicted) ** 2
            if best is None or error < best[0]:
                best = (error, delay, tau)
    assert best is not None

    sse, delay, tau = best
    rmse = math.sqrt(sse / len(active))
    mean = statistics.mean(sample.rpm for sample in active)
    sst = sum((sample.rpm - mean) ** 2 for sample in active)
    r2 = 1.0 - sse / sst if sst > 1e-9 else 0.0
    observed = [sample.rpm for sample in active]
    peak = max(observed) if delta >= 0.0 else min(observed)
    overshoot = max(0.0, peak - final) if delta >= 0.0 else max(0.0, final - peak)
    overshoot_ratio = overshoot / max(abs(delta), 1.0)
    normalized_rmse = rmse / max(abs(delta), 1.0)
    return StepFit(
        label=phase.name,
        phase=phase.name,
        input_command_norm=phase.command_norm,
        input_target_rpm=phase.command_norm * target_rpm_max,
        edge_mcu_time_ms=edge_mcu_time_ms,
        baseline_rpm=baseline,
        final_rpm=final,
        gain_rpm_per_rpm=delta / (phase.command_norm * target_rpm_max)
        if abs(phase.command_norm) > 1e-9
        else 0.0,
        delay_sec=delay,
        time_constant_sec=tau,
        rmse_rpm=rmse,
        r2=r2,
        peak_rpm=peak,
        overshoot_ratio=overshoot_ratio,
        sample_count=len(active),
        first_order_adequate=(r2 >= 0.90 and normalized_rmse <= 0.10 and overshoot_ratio <= 0.10),
    )


def unwrap_mcu_times(
    rpm_records: Sequence[tuple[float, int, int]],
    echo_records: Sequence[tuple[float, int, int]],
) -> tuple[dict[int, float], dict[int, float]]:
    """Unwrap each stream's uint32 MCU clock in host-reception order.

    RPM telemetry and command echoes are sent as separate UDP streams. Their
    host arrival order is not guaranteed, so interleaving them before unwrap
    can incorrectly classify a valid sample as an out-of-order packet.
    """
    def unwrap(records: Sequence[tuple[float, int, int]]) -> dict[int, float]:
        values: dict[int, float] = {}
        last_raw: int | None = None
        epoch = 0
        for _, key, raw_value in sorted(records, key=lambda item: item[0]):
            raw = int(raw_value) & 0xFFFFFFFF
            if last_raw is not None:
                delta = (raw - last_raw) & 0xFFFFFFFF
                if delta > 0x80000000:
                    # An out-of-order packet cannot be used for timing.
                    continue
                if raw < last_raw:
                    epoch += 0x100000000
            values[key] = float(epoch + raw)
            last_raw = raw
        return values

    return unwrap(rpm_records), unwrap(echo_records)


def save_csv(samples: Sequence[RpmSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "host_elapsed_sec",
        "host_received_sec",
        "phase",
        "command_norm",
        "target_rpm",
        "mcu_time_ms",
        "telemetry_sequence",
        "rpm",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in samples:
            writer.writerow({
                "host_elapsed_sec": f"{sample.host_elapsed_sec:.6f}",
                "host_received_sec": f"{sample.host_received_sec:.6f}",
                "phase": sample.phase,
                "command_norm": f"{sample.command_norm:.6f}",
                "target_rpm": f"{sample.target_rpm:.3f}",
                "mcu_time_ms": f"{sample.mcu_time_ms:.3f}",
                "telemetry_sequence": sample.telemetry_sequence,
                "rpm": f"{sample.rpm:.6f}",
            })


def save_echo_csv(echoes: Sequence[EchoSample], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "host_elapsed_sec",
        "host_received_sec",
        "mcu_time_ms",
        "telemetry_sequence",
        "command_count",
        "enabled",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for sample in echoes:
            writer.writerow({
                "host_elapsed_sec": f"{sample.host_elapsed_sec:.6f}",
                "host_received_sec": f"{sample.host_received_sec:.6f}",
                "mcu_time_ms": f"{sample.mcu_time_ms:.3f}",
                "telemetry_sequence": sample.telemetry_sequence,
                "command_count": sample.command_count,
                "enabled": int(sample.enabled),
            })


def save_plot(
    samples: Sequence[RpmSample],
    phases: Sequence[Phase],
    fits: Sequence[StepFit],
    path: Path,
    *,
    target_rpm_max: float,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    if not samples:
        return
    base = min(sample.mcu_time_ms for sample in samples)
    times = [(sample.mcu_time_ms - base) / 1000.0 for sample in samples]
    target = [sample.target_rpm for sample in samples]
    actual = [sample.rpm for sample in samples]
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True, constrained_layout=True)
    axes[0].step(times, target, where="post", color="tab:orange", linewidth=1.8, label="target RPM")
    axes[0].plot(times, actual, color="tab:blue", linewidth=1.2, label="measured RPM")
    axes[0].set_ylabel("RPM")
    axes[0].set_title("MCU-time-axis first-order RPM identification")
    axes[0].grid(True, alpha=0.35)
    axes[0].legend(loc="upper right")

    residual = [sample.rpm - target_value for sample, target_value in zip(samples, target)]
    axes[1].plot(times, residual, color="tab:red", linewidth=1.1, label="RPM - target")
    axes[1].axhline(0.0, color="black", linewidth=0.8)
    axes[1].set_xlabel("MCU time from first sample (s)")
    axes[1].set_ylabel("RPM error")
    axes[1].grid(True, alpha=0.35)
    axes[1].legend(loc="upper right")

    elapsed = 0.0
    for phase in phases:
        start = elapsed
        elapsed += phase.duration_sec
        for axis in axes:
            axis.axvspan(start, elapsed, color="tab:green" if phase.command_norm else "tab:gray", alpha=0.04)
        if phase.command_norm:
            axes[0].text(
                (start + elapsed) * 0.5,
                0.96,
                f"{phase.command_norm * target_rpm_max:+.0f}",
                transform=axes[0].get_xaxis_transform(),
                ha="center",
                va="top",
                fontsize=8,
            )

    fit_text = " | ".join(
        f"{fit.label}: Td={fit.delay_sec:.3f}s, tau={fit.time_constant_sec:.3f}s, R2={fit.r2:.3f}"
        for fit in fits
    )
    if fit_text:
        axes[0].set_title(f"MCU-time-axis first-order RPM identification\n{fit_text}")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Identify target-RPM to measured-RPM first-order plus pure delay using MCU timestamps."
    )
    parser.add_argument("--motor-index", type=int, default=4, choices=range(THRUSTER_COUNT))
    parser.add_argument("--target-rpm", type=parse_float, default=1200.0)
    parser.add_argument("--target-rpm-max", type=parse_float, default=3000.0)
    parser.add_argument("--initial-zero-sec", type=parse_float, default=3.0)
    parser.add_argument("--active-sec", type=parse_float, default=8.0)
    parser.add_argument("--zero-sec", type=parse_float, default=8.0)
    parser.add_argument("--rate-hz", type=parse_float, default=50.0)
    parser.add_argument("--wait-for-subscriber-sec", type=parse_float, default=5.0)
    parser.add_argument("--command-topic", default=DEFAULT_COMMAND_TOPIC)
    parser.add_argument("--rpm-topic", default=DEFAULT_RPM_TOPIC)
    parser.add_argument("--echo-topic", default=DEFAULT_ECHO_TOPIC)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


class FirstOrderIdentifier(Node):
    def __init__(self, args: argparse.Namespace, phases: Sequence[Phase]) -> None:
        super().__init__("thruster_rpm_first_order_identification")
        self.args = args
        self.phases = list(phases)
        self.callback_group = ReentrantCallbackGroup()
        self.publisher = self.create_publisher(Float32MultiArray, args.command_topic, 10)
        self.create_subscription(
            HardwareTelemetry,
            args.rpm_topic,
            self.rpm_callback,
            100,
            callback_group=self.callback_group,
        )
        self.create_subscription(
            ThrusterCommandEcho,
            args.echo_topic,
            self.echo_callback,
            100,
            callback_group=self.callback_group,
        )
        self.created_host = time.monotonic()
        self.start_host: float | None = None
        self.phase_start_host: float | None = None
        self.phase_index = 0
        self.started = False
        self.finished = False
        self.samples_raw: list[tuple[float, str, float, float, int, int, float]] = []
        self.echoes_raw: list[tuple[float, int, int, int, bool]] = []
        self.phase_edges_host: list[tuple[str, float, float]] = []
        self._last_rpm_mcu_time: int | None = None
        self._last_echo_mcu_time: int | None = None
        self.timer = self.create_timer(
            1.0 / args.rate_hz,
            self.tick,
            callback_group=self.callback_group,
        )

    def rpm_callback(self, msg: HardwareTelemetry) -> None:
        # Do not let DDS backlog from before the experiment define the baseline.
        if not self.started:
            return
        now = time.monotonic()
        if len(msg.rpm) != THRUSTER_COUNT:
            return
        if self._last_rpm_mcu_time == int(msg.mcu_time_ms):
            return
        self._last_rpm_mcu_time = int(msg.mcu_time_ms)
        phase, command = self.current_phase(now)
        self.samples_raw.append(
            (
                now,
                phase.name,
                command,
                float(msg.mcu_time_ms),
                int(msg.telemetry_sequence),
                int(msg.mcu_time_ms),
                float(msg.rpm[self.args.motor_index]),
            )
        )

    def echo_callback(self, msg: ThrusterCommandEcho) -> None:
        if not self.started:
            return
        now = time.monotonic()
        if self._last_echo_mcu_time == int(msg.mcu_time_ms) and self.echoes_raw:
            return
        self._last_echo_mcu_time = int(msg.mcu_time_ms)
        self.echoes_raw.append(
            (
                now,
                int(msg.mcu_time_ms),
                int(msg.telemetry_sequence),
                int(msg.command_count),
                bool(msg.enabled),
            )
        )

    def current_phase(self, now: float) -> tuple[Phase, float]:
        if not self.started or self.phase_start_host is None:
            return self.phases[0], 0.0
        elapsed = now - self.start_host  # type: ignore[operator]
        cursor = 0.0
        for phase in self.phases:
            if cursor <= elapsed < cursor + phase.duration_sec:
                return phase, phase.command_norm
            cursor += phase.duration_sec
        return self.phases[-1], 0.0

    def publish(self, command_norm: float, enabled: bool) -> None:
        values = [0.0] * THRUSTER_COUNT
        values[self.args.motor_index] = command_norm if enabled else 0.0
        msg = Float32MultiArray()
        msg.data = values
        self.publisher.publish(msg)

    def tick(self) -> None:
        now = time.monotonic()
        if not self.started:
            if self.publisher.get_subscription_count() == 0:
                if now - self.created_host > self.args.wait_for_subscriber_sec:
                    self.get_logger().error("no matched thruster subscriber; aborting")
                    self.finished = True
                    self.timer.cancel()
                return
            self.started = True
            self.start_host = now
            self.phase_start_host = now
            self.phase_edges_host.append((self.phases[0].name, now, now + self.phases[0].duration_sec))
            self.get_logger().info(
                f"started motor={CANONICAL_NAMES[self.args.motor_index]} target={self.args.target_rpm:.1f} RPM"
            )

        assert self.start_host is not None
        elapsed = now - self.start_host
        cursor = 0.0
        selected_index = len(self.phases) - 1
        for index, phase in enumerate(self.phases):
            if cursor <= elapsed < cursor + phase.duration_sec:
                selected_index = index
                break
            cursor += phase.duration_sec

        if selected_index != self.phase_index:
            self.phase_index = selected_index
            phase_start = self.start_host + sum(phase.duration_sec for phase in self.phases[:selected_index])
            phase_end = phase_start + self.phases[selected_index].duration_sec
            self.phase_edges_host.append((self.phases[selected_index].name, phase_start, phase_end))

        phase = self.phases[self.phase_index]
        self.publish(phase.command_norm, enabled=abs(phase.command_norm) > 1e-9)
        total_duration = sum(item.duration_sec for item in self.phases)
        if elapsed >= total_duration:
            self.publish(0.0, enabled=False)
            self.finished = True
            self.timer.cancel()


def _make_results(node: FirstOrderIdentifier, args: argparse.Namespace, phases: Sequence[Phase]):
    if not node.samples_raw:
        raise RuntimeError("no MCU RPM samples received")
    rpm_records = [(item[0], index, int(item[5])) for index, item in enumerate(node.samples_raw)]
    echo_records = [(item[0], index, int(item[1])) for index, item in enumerate(node.echoes_raw)]
    rpm_times, echo_times = unwrap_mcu_times(rpm_records, echo_records)

    samples: list[RpmSample] = []
    for index, item in enumerate(node.samples_raw):
        mcu_time = rpm_times.get(index)
        if mcu_time is None:
            continue
        host_received, phase, command, _, sequence, _, rpm = item
        samples.append(
            RpmSample(
                host_elapsed_sec=host_received - node.start_host,  # type: ignore[operator]
                host_received_sec=host_received,
                phase=phase,
                command_norm=command,
                target_rpm=command * args.target_rpm_max,
                mcu_time_ms=mcu_time,
                telemetry_sequence=sequence,
                rpm=rpm,
            )
        )
    samples.sort(key=lambda sample: sample.mcu_time_ms)

    echoes: list[EchoSample] = []
    for index, item in enumerate(node.echoes_raw):
        mcu_time = echo_times.get(index)
        if mcu_time is None:
            continue
        host_received, raw_mcu, sequence, command_count, enabled = item
        echoes.append(
            EchoSample(
                host_elapsed_sec=host_received - node.start_host,  # type: ignore[operator]
                host_received_sec=host_received,
                mcu_time_ms=mcu_time,
                telemetry_sequence=sequence,
                command_count=command_count,
                enabled=enabled,
            )
        )
    echoes.sort(key=lambda echo: echo.host_received_sec)
    if len(echoes) < 10:
        raise RuntimeError(
            f"only {len(echoes)} command echo samples received; enable debug thruster echo before fitting"
        )

    edges: dict[str, float] = {}
    for name, start_host, end_host in node.phase_edges_host:
        candidates = [echo for echo in echoes if start_host <= echo.host_received_sec <= end_host]
        if candidates:
            edges[name] = candidates[0].mcu_time_ms
    if any(phase.name not in edges for phase in phases):
        missing = [phase.name for phase in phases if phase.name not in edges]
        raise RuntimeError(f"missing MCU command echo edge for phases: {missing}")

    # Re-label samples on the MCU clock. The ROS callback arrival time is only
    # used for diagnostics; it must not define the command phase.
    relabeled: list[RpmSample] = []
    for sample in samples:
        selected_phase = phases[-1]
        for index, phase in enumerate(phases):
            start = edges[phase.name]
            end = edges[phases[index + 1].name] if index + 1 < len(phases) else float("inf")
            if start <= sample.mcu_time_ms < end:
                selected_phase = phase
                break
        relabeled.append(
            RpmSample(
                host_elapsed_sec=sample.host_elapsed_sec,
                host_received_sec=sample.host_received_sec,
                phase=selected_phase.name,
                command_norm=selected_phase.command_norm,
                target_rpm=selected_phase.command_norm * args.target_rpm_max,
                mcu_time_ms=sample.mcu_time_ms,
                telemetry_sequence=sample.telemetry_sequence,
                rpm=sample.rpm,
            )
        )
    samples = relabeled

    fits: list[StepFit] = []
    for index, phase in enumerate(phases):
        if index + 1 >= len(phases):
            break
        fit = fit_step(
            samples,
            phase=phase,
            edge_mcu_time_ms=edges[phase.name],
            next_edge_mcu_time_ms=edges[phases[index + 1].name],
            previous_edge_mcu_time_ms=edges[phases[index - 1].name] if index > 0 else None,
            target_rpm_max=args.target_rpm_max,
        )
        if fit is not None and phase.command_norm != 0.0:
            fits.append(fit)

    return samples, echoes, fits, edges


def main() -> None:
    args = build_parser().parse_args()
    if args.target_rpm <= 0.0 or args.target_rpm > args.target_rpm_max:
        raise SystemExit("target-rpm must be in (0, target-rpm-max]")
    if args.rate_hz <= 0.0 or args.active_sec <= 0.0 or args.zero_sec <= 0.0:
        raise SystemExit("rate-hz must be positive and active/zero durations must be nonzero")
    phases = build_phases(
        target_rpm=args.target_rpm,
        initial_zero_sec=args.initial_zero_sec,
        active_sec=args.active_sec,
        zero_sec=args.zero_sec,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rclpy.init()
    node = FirstOrderIdentifier(args, phases)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    spin_thread = threading.Thread(target=executor.spin, name="rpm-identification-spin", daemon=True)
    spin_thread.start()
    error: Exception | None = None
    try:
        while rclpy.ok() and not node.finished:
            time.sleep(0.05)
    except KeyboardInterrupt:
        node.get_logger().warning("interrupted; publishing disabled zero")
    finally:
        executor.shutdown(timeout_sec=2.0)
        spin_thread.join(timeout=2.0)
        node.publish(0.0, enabled=False)
        try:
            samples, echoes, fits, edges = _make_results(node, args, phases)
            save_csv(samples, args.output_dir / "rpm_first_order.csv")
            save_echo_csv(echoes, args.output_dir / "command_echo.csv")
            save_plot(
                samples,
                phases,
                fits,
                args.output_dir / "rpm_first_order_response.png",
                target_rpm_max=args.target_rpm_max,
            )
            payload = {
                "model": "first_order_plus_pure_delay_on_mcu_timestamped_rpm",
                "transfer_function": "K/(tau*s+1)*exp(-Td*s)",
                "motor_index": args.motor_index,
                "motor_name": CANONICAL_NAMES[args.motor_index],
                "target_rpm": args.target_rpm,
                "target_rpm_max": args.target_rpm_max,
                "command_topic": args.command_topic,
                "rpm_topic": args.rpm_topic,
                "echo_topic": args.echo_topic,
                "phase_edges_mcu_time_ms": edges,
                "sample_count": len(samples),
                "echo_count": len(echoes),
                "fits": [fit.__dict__ for fit in fits],
                "notes": [
                    "RPM samples use MCU telemetry time, not ROS receive time.",
                    "Command edges use MCU receive time from ThrusterCommandEcho.",
                    "Old motor_rpm_raw Float32MultiArray is not used for fitting.",
                ],
            }
            (args.output_dir / "fit.json").write_text(
                json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
                encoding="utf-8",
            )
            print(f"saved first-order identification data to {args.output_dir}")
            for fit in fits:
                print(
                    f"{fit.label}: Td={fit.delay_sec:.4f}s tau={fit.time_constant_sec:.4f}s "
                    f"K={fit.gain_rpm_per_rpm:.4f} RMSE={fit.rmse_rpm:.2f} R2={fit.r2:.3f}"
                )
        except Exception as exc:  # preserve zero command and expose fit failure
            error = exc
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    if error is not None:
        raise SystemExit(str(error))


if __name__ == "__main__":
    main()
