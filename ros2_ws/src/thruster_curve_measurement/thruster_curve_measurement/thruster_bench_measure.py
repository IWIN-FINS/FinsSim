from __future__ import annotations

import argparse
import csv
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Sequence

from .force_sensor_modbus import ForceReading, ModbusForceSensor
from .thruster_bench_driver import SerialThrusterBenchDriver, ThrusterBenchStatus
from .thruster_curve_measure import (
    SignedQuadraticFit,
    SummaryPoint,
    fit_signed_quadratic,
    make_default_step_commands,
    parse_step_commands,
    save_discrete_plot,
    save_fit_plot,
    save_plot,
    write_fit_csv,
)
from .thruster_curve_sweep import (
    SweepConfig,
    command_at_elapsed,
    total_duration,
    validate_sweep_config,
)


CONFIG_SECTION = "thruster_bench_measure"
BENCH_TARGET_INDEX = 0
BENCH_TARGET_NAME = "single_thruster"


class _Log:
    def info(self, text: str) -> None:
        print(f"[INFO] {text}", flush=True)

    def warn(self, text: str) -> None:
        print(f"[WARN] {text}", flush=True)


def _config_key_to_dest(key: str) -> str:
    return key.replace("-", "_")


def _parser_dests(parser: argparse.ArgumentParser) -> set[str]:
    return {action.dest for action in parser._actions if action.dest != argparse.SUPPRESS}


def load_config_defaults(path: Path, parser: argparse.ArgumentParser) -> dict[str, object]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError("config file must contain a YAML mapping")

    data = raw.get(CONFIG_SECTION, raw)
    if not isinstance(data, dict):
        raise ValueError(f"config section '{CONFIG_SECTION}' must be a YAML mapping")

    valid_dests = _parser_dests(parser)
    defaults: dict[str, object] = {}
    unknown: list[str] = []
    for key, value in data.items():
        if not isinstance(key, str):
            raise ValueError("config keys must be strings")
        dest = _config_key_to_dest(key)
        if dest == "config":
            continue
        if dest not in valid_dests:
            unknown.append(key)
            continue
        defaults[dest] = value

    if unknown:
        joined = ", ".join(sorted(unknown))
        raise ValueError(f"unknown config key(s): {joined}")
    return defaults


def _path_from_config(value: object) -> Path | None:
    if value in (None, ""):
        return None
    if isinstance(value, Path):
        return value
    return Path(str(value))


def _write_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "wall_time_unix",
            "monotonic_time",
            "elapsed_sec",
            "phase",
            "target_index",
            "target_name",
            "target_command",
            "output_limit",
            "sensor_ok",
            "sensor_error",
            "raw_value",
            "signed_value",
            "force_n",
            "zero_offset_n",
            "force_zeroed_n",
            "sample_window",
            "point_index",
            "repeat_index",
            "status_ok",
            "status_enabled",
            "status_accepted",
            "status_reject_flags",
            "status_command",
            "status_applied",
            "status_rpm",
            "status_command_count",
        ]
    )


def _status_fields(status: ThrusterBenchStatus | None) -> list[object]:
    if status is None:
        return [0, "", "", "", "", "", "", ""]
    return [
        1,
        1 if status.enabled else 0,
        1 if status.accepted else 0,
        status.reject_flags,
        f"{status.command:.6f}",
        f"{status.applied:.6f}",
        f"{status.rpm:.6f}",
        status.command_count,
    ]


def _write_row(
    writer: csv.writer,
    *,
    wall_time: float,
    monotonic_time: float,
    elapsed_sec: float,
    phase: str,
    command: float,
    output_limit: float,
    reading: ForceReading,
    zero_offset_n: float,
    status: ThrusterBenchStatus | None,
    sample_window: bool = False,
    point_index: int | None = None,
    repeat_index: int | None = None,
) -> None:
    force = reading.force_n
    force_zeroed = force - zero_offset_n if force is not None else None
    writer.writerow(
        [
            f"{wall_time:.6f}",
            f"{monotonic_time:.6f}",
            f"{elapsed_sec:.6f}",
            phase,
            BENCH_TARGET_INDEX,
            BENCH_TARGET_NAME,
            f"{command:.6f}",
            f"{output_limit:.6f}",
            1 if reading.ok else 0,
            reading.error,
            "" if reading.raw_value is None else reading.raw_value,
            "" if reading.signed_value is None else reading.signed_value,
            "" if force is None else f"{force:.6f}",
            f"{zero_offset_n:.6f}",
            "" if force_zeroed is None else f"{force_zeroed:.6f}",
            1 if sample_window else 0,
            "" if point_index is None else point_index,
            "" if repeat_index is None else repeat_index,
            *_status_fields(status),
        ]
    )


def _write_summary_header(writer: csv.writer) -> None:
    writer.writerow(
        [
            "target_index",
            "target_name",
            "repeat_index",
            "point_index",
            "target_command",
            "sample_count",
            "force_mean_n",
            "force_median_n",
            "force_std_n",
            "force_min_n",
            "force_max_n",
        ]
    )


def _write_summary_rows(
    writer: csv.writer,
    *,
    samples: dict[tuple[int, int], list[float]],
    commands: Sequence[float],
) -> None:
    for repeat_index, point_index in sorted(samples):
        forces = samples[(repeat_index, point_index)]
        if not forces:
            continue
        command = commands[point_index]
        std = statistics.stdev(forces) if len(forces) > 1 else 0.0
        writer.writerow(
            [
                BENCH_TARGET_INDEX,
                BENCH_TARGET_NAME,
                repeat_index,
                point_index,
                f"{command:.6f}",
                len(forces),
                f"{statistics.mean(forces):.6f}",
                f"{statistics.median(forces):.6f}",
                f"{std:.6f}",
                f"{min(forces):.6f}",
                f"{max(forces):.6f}",
            ]
        )


def _load_summary_points(summary_csv: Path) -> list[SummaryPoint]:
    points: list[SummaryPoint] = []
    with summary_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("target_command") or not row.get("force_mean_n"):
                continue
            points.append(
                SummaryPoint(
                    command=float(row["target_command"]),
                    force_mean_n=float(row["force_mean_n"]),
                    force_median_n=float(row["force_median_n"]),
                    force_std_n=float(row["force_std_n"]),
                    sample_count=int(row["sample_count"]),
                    repeat_index=int(row["repeat_index"]),
                    point_index=int(row["point_index"]),
                )
            )
    return points


def _fit_points(points: Sequence[SummaryPoint], min_abs_command: float) -> list[SignedQuadraticFit]:
    fits: list[SignedQuadraticFit] = []
    for side in ("negative", "positive"):
        try:
            fits.append(fit_signed_quadratic(points, side=side, min_abs_command=min_abs_command))
        except ValueError as exc:
            print(f"[WARN] skipping {side} fit: {exc}", flush=True)
    return fits


def save_plot(csv_path: Path, plot_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    elapsed: list[float] = []
    command: list[float] = []
    force: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("sensor_ok") != "1" or row.get("force_zeroed_n", "") == "":
                continue
            elapsed.append(float(row["elapsed_sec"]))
            command.append(float(row["target_command"]))
            force.append(float(row["force_zeroed_n"]))

    if not elapsed:
        raise RuntimeError("no valid force samples available for plotting")

    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(9, 8), constrained_layout=True)
    axes[0].plot(elapsed, command, label="command")
    axes[0].set_xlabel("time (s)")
    axes[0].set_ylabel("command")
    axes[0].grid(True)
    axes[0].legend()
    axes[1].scatter(command, force, s=8, alpha=0.7)
    axes[1].set_xlabel("command")
    axes[1].set_ylabel("zeroed force (N)")
    axes[1].grid(True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def collect_zero_offset(
    *,
    log: _Log,
    driver: SerialThrusterBenchDriver,
    sensor: ModbusForceSensor,
    samples: int,
    sample_rate_hz: float,
) -> float:
    if samples <= 0:
        return 0.0

    period = 1.0 / max(sample_rate_hz, 0.1)
    forces: list[float] = []
    log.warn(f"collecting zero offset from {samples} sensor samples")
    for _ in range(samples):
        driver.command(0.0, enabled=False, limit=0.0)
        driver.read_status(timeout_sec=min(period, 0.02))
        reading = sensor.read()
        if reading.ok and reading.force_n is not None:
            forces.append(reading.force_n)
        else:
            log.warn(f"zero sample failed: {reading.error}")
        time.sleep(period)

    if not forces:
        raise RuntimeError("failed to collect any valid zero samples")
    zero = statistics.median(forces)
    log.warn(f"zero offset set to {zero:+.6f} N from {len(forces)} valid samples")
    return zero


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run standalone single-thruster bench measurement over serial while logging Modbus force data."
    )
    parser.add_argument("--config", type=Path, default=None, help="YAML defaults file; CLI arguments override it")
    parser.add_argument("--thruster-port", default=None, help="MCU serial port for ThrusterBench firmware")
    parser.add_argument("--thruster-baudrate", type=int, default=115200)
    parser.add_argument("--thruster-timeout", type=float, default=0.02)
    parser.add_argument("--output-limit", type=float, default=0.4, help="limit sent to MCU, clamped again by firmware")
    parser.add_argument("--mode", choices=("step", "ramp", "loop", "command"), default="step")
    parser.add_argument("--command", type=float, default=0.0, help="command-mode normalized command")
    parser.add_argument("--duration", type=float, default=3.0, help="command-mode duration, seconds")
    parser.add_argument("--amplitude", type=float, default=1.0)
    parser.add_argument("--rate", type=float, default=10.0)
    parser.add_argument("--pre-zero", type=float, default=5.0)
    parser.add_argument("--settle", type=float, default=3.0)
    parser.add_argument("--post-zero", type=float, default=5.0)
    parser.add_argument("--hold-start", type=float, default=2.0)
    parser.add_argument("--ramp", type=float, default=60.0)
    parser.add_argument("--hold-end", type=float, default=2.0)
    parser.add_argument("--bidirectional", action="store_true")
    parser.add_argument("--step", type=float, default=0.1)
    parser.add_argument("--step-hold", type=float, default=5.0)
    parser.add_argument("--step-sample", type=float, default=2.0)
    parser.add_argument("--step-rest", type=float, default=1.0)
    parser.add_argument("--step-commands", default="")
    parser.add_argument("--repeat-count", type=int, default=1)
    parser.add_argument("--repeat-rest", type=float, default=0.0)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    parser.add_argument("--plot", type=Path, default=None)
    parser.add_argument("--points-plot", type=Path, default=None)
    parser.add_argument("--fit-plot", type=Path, default=None)
    parser.add_argument("--fit-csv", type=Path, default=None)
    parser.add_argument("--fit-min-abs-command", type=float, default=0.0)
    parser.add_argument("--no-fit", action="store_true")
    parser.add_argument("--sensor-port", default=None)
    parser.add_argument("--sensor-baudrate", type=int, default=115200)
    parser.add_argument("--sensor-bytesize", type=int, default=8)
    parser.add_argument("--sensor-parity", default="N")
    parser.add_argument("--sensor-stopbits", type=int, default=1)
    parser.add_argument("--sensor-timeout", type=float, default=0.2)
    parser.add_argument("--sensor-address", type=int, default=205)
    parser.add_argument("--sensor-count", type=int, default=1)
    parser.add_argument("--sensor-device-id", type=int, default=1)
    parser.add_argument("--sensor-divisor", type=float, default=100.0)
    parser.add_argument("--sensor-scale", type=float, default=0.98)
    parser.add_argument("--sensor-offset-n", type=float, default=0.0)
    parser.add_argument("--zero-samples", type=int, default=20)
    parser.add_argument("--force-log-period", type=float, default=1.0)
    parser.add_argument("--no-stop-on-exit", action="store_true")
    parser.add_argument("--print-plan", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    argv = list(sys.argv[1:] if argv is None else argv)

    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, _ = config_parser.parse_known_args(argv)

    parser = _build_parser()
    if config_args.config is not None:
        try:
            defaults = load_config_defaults(config_args.config, parser)
        except ValueError as exc:
            parser.error(f"{config_args.config}: {exc}")
        parser.set_defaults(**defaults)

    args = parser.parse_args(argv)
    args.csv = _path_from_config(args.csv)
    args.summary_csv = _path_from_config(args.summary_csv)
    args.plot = _path_from_config(args.plot)
    args.points_plot = _path_from_config(args.points_plot)
    args.fit_plot = _path_from_config(args.fit_plot)
    args.fit_csv = _path_from_config(args.fit_csv)
    if args.thruster_port in (None, ""):
        parser.error("--thruster-port is required, either on the command line or in --config")
    if args.csv is None:
        parser.error("--csv is required, either on the command line or in --config")
    if args.sensor_port in (None, ""):
        parser.error("--sensor-port is required, either on the command line or in --config")
    if args.step_commands is None:
        args.step_commands = ""
    elif isinstance(args.step_commands, (list, tuple)):
        args.step_commands = ",".join(str(value) for value in args.step_commands)
    else:
        args.step_commands = str(args.step_commands)
    return args


def _commands_from_args(args: argparse.Namespace) -> tuple[float, ...]:
    return parse_step_commands(args.step_commands) if args.step_commands else make_default_step_commands(args.amplitude, args.step)


def _command_at_step_elapsed(elapsed_sec: float, args: argparse.Namespace, commands: Sequence[float]):
    t = max(0.0, float(elapsed_sec))
    if t < args.pre_zero:
        return "pre_zero", 0.0, False, None, None, False
    t -= args.pre_zero
    if t < args.settle:
        return "settle_zero", 0.0, False, None, None, False
    t -= args.settle

    per_point = args.step_hold + args.step_rest
    repeat_step_region = len(commands) * per_point
    for repeat_index in range(args.repeat_count):
        if t < repeat_step_region:
            point_index = int(t // per_point)
            local_t = t - point_index * per_point
            command = commands[point_index]
            if local_t < args.step_hold:
                sample_start = max(0.0, args.step_hold - args.step_sample)
                sample_window = local_t >= sample_start
                return (
                    "step_sample" if sample_window else "step_settle",
                    command,
                    sample_window,
                    point_index,
                    repeat_index,
                    False,
                )
            return "step_rest_zero", 0.0, False, point_index, repeat_index, False
        t -= repeat_step_region
        if repeat_index < args.repeat_count - 1:
            if t < args.repeat_rest:
                return "repeat_rest_zero", 0.0, False, None, repeat_index, False
            t -= args.repeat_rest

    if t < args.post_zero:
        return "post_zero", 0.0, False, None, None, False
    return "done", 0.0, False, None, None, True


def _step_total_duration(args: argparse.Namespace, commands: Sequence[float]) -> float:
    if args.repeat_count < 1:
        raise ValueError("--repeat-count must be >= 1")
    for name in ("pre_zero", "settle", "step_hold", "step_sample", "step_rest", "repeat_rest", "post_zero"):
        value = float(getattr(args, name))
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be >= 0")
    if args.step_hold <= 0.0:
        raise ValueError("--step-hold must be > 0")
    if args.step_sample <= 0.0 or args.step_sample > args.step_hold:
        raise ValueError("--step-sample must be in (0, step-hold]")
    return (
        args.pre_zero
        + args.settle
        + args.repeat_count * len(commands) * (args.step_hold + args.step_rest)
        + max(0, args.repeat_count - 1) * args.repeat_rest
        + args.post_zero
    )


def _ramp_config_from_args(args: argparse.Namespace) -> SweepConfig:
    loop_mode = args.mode == "loop"
    return SweepConfig(
        index=0,
        amplitude=args.amplitude,
        pre_zero_sec=args.pre_zero,
        marker_amplitude=0.0,
        marker_on_sec=0.0,
        marker_off_sec=0.0,
        marker_count=0,
        settle_sec=args.settle,
        hold_start_sec=0.0 if loop_mode else args.hold_start,
        ramp_sec=args.ramp,
        hold_end_sec=0.0 if loop_mode else args.hold_end,
        post_zero_sec=args.post_zero,
        bidirectional=True if loop_mode else args.bidirectional,
    )


def _command_mode_sample(elapsed: float, args: argparse.Namespace):
    if elapsed < args.pre_zero:
        return "pre_zero", 0.0, False
    if elapsed < args.pre_zero + args.duration:
        return "command_hold", args.command, False
    if elapsed < args.pre_zero + args.duration + args.post_zero:
        return "post_zero", 0.0, False
    return "done", 0.0, True


def _print_plan(args: argparse.Namespace, duration: float, commands: Sequence[float] | None) -> None:
    print(f"mode={args.mode}")
    print(f"thruster_port={args.thruster_port}")
    print(f"thruster_baudrate={args.thruster_baudrate}")
    print(f"output_limit={args.output_limit:.3f}")
    print(f"duration_sec={duration:.3f}")
    if commands is not None:
        print(f"commands={','.join(f'{value:.3f}' for value in commands)}")
    if args.mode in {"ramp", "loop"}:
        print(f"ramp_sec={args.ramp:.3f}")
        print(f"bidirectional={args.mode == 'loop' or args.bidirectional}")
    print(f"csv={args.csv}")
    print(f"sensor_port={args.sensor_port}")


def main() -> None:
    parser = _build_parser()
    args = parse_args()
    try:
        if args.mode == "step":
            commands = _commands_from_args(args)
            duration = _step_total_duration(args, commands)
            ramp_config = None
        elif args.mode in {"ramp", "loop"}:
            commands = None
            ramp_config = _ramp_config_from_args(args)
            validate_sweep_config(ramp_config)
            duration = total_duration(ramp_config)
        else:
            commands = None
            ramp_config = None
            if not math.isfinite(args.command) or args.command < -1.0 or args.command > 1.0:
                raise ValueError("--command must be in [-1, 1]")
            if args.duration <= 0.0:
                raise ValueError("--duration must be > 0")
            duration = args.pre_zero + args.duration + args.post_zero
    except ValueError as exc:
        parser.error(str(exc))

    if args.print_plan:
        _print_plan(args, duration, commands)
        return

    log = _Log()
    sensor = ModbusForceSensor(
        port=args.sensor_port,
        baudrate=args.sensor_baudrate,
        bytesize=args.sensor_bytesize,
        parity=args.sensor_parity,
        stopbits=args.sensor_stopbits,
        timeout=args.sensor_timeout,
        address=args.sensor_address,
        count=args.sensor_count,
        device_id=args.sensor_device_id,
        divisor=args.sensor_divisor,
        scale=args.sensor_scale,
        offset_n=args.sensor_offset_n,
    )
    if not sensor.connect():
        raise SystemExit(f"failed to open force sensor serial port: {args.sensor_port}")

    driver = SerialThrusterBenchDriver(
        port=args.thruster_port,
        baudrate=args.thruster_baudrate,
        timeout=args.thruster_timeout,
    )
    args.csv.parent.mkdir(parents=True, exist_ok=True)

    try:
        log.warn(
            "starting standalone thrust curve measurement: "
            f"mode={args.mode}, thruster={args.thruster_port}, sensor={args.sensor_port}, "
            f"duration={duration:.2f}s, csv={args.csv}"
        )
        zero_offset_n = collect_zero_offset(
            log=log,
            driver=driver,
            sensor=sensor,
            samples=args.zero_samples,
            sample_rate_hz=args.rate,
        )

        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            _write_header(writer)
            period = 1.0 / max(args.rate, 0.1)
            start = time.monotonic()
            next_tick = start
            next_force_log = start
            step_summary_samples: dict[tuple[int, int], list[float]] = {}

            while True:
                now = time.monotonic()
                if now < next_tick:
                    time.sleep(next_tick - now)
                now = time.monotonic()
                elapsed = now - start

                if args.mode == "step":
                    assert commands is not None
                    phase, command, sample_window, point_index, repeat_index, done = _command_at_step_elapsed(
                        elapsed, args, commands
                    )
                elif args.mode in {"ramp", "loop"}:
                    assert ramp_config is not None
                    sample = command_at_elapsed(elapsed, ramp_config)
                    phase = sample.phase
                    command = sample.command
                    sample_window = False
                    point_index = None
                    repeat_index = None
                    done = sample.done
                else:
                    phase, command, done = _command_mode_sample(elapsed, args)
                    sample_window = phase == "command_hold"
                    point_index = 0 if sample_window else None
                    repeat_index = 0 if sample_window else None

                if done:
                    break

                driver.command(command, enabled=abs(command) > 1e-9, limit=args.output_limit)
                status = driver.read_status(timeout_sec=min(period, 0.02))
                reading = sensor.read()
                force_zeroed = reading.force_n - zero_offset_n if reading.ok and reading.force_n is not None else None
                if args.force_log_period > 0.0 and now >= next_force_log:
                    if force_zeroed is None:
                        log.warn(f"force read failed: phase={phase}, command={command:+.3f}, error={reading.error}")
                    else:
                        applied = "n/a" if status is None else f"{status.applied:+.3f}"
                        log.info(
                            f"force_zeroed={force_zeroed:+.3f} N, raw_force={reading.force_n:+.3f} N, "
                            f"command={command:+.3f}, applied={applied}, phase={phase}"
                        )
                    next_force_log = now + args.force_log_period

                _write_row(
                    writer,
                    wall_time=time.time(),
                    monotonic_time=now,
                    elapsed_sec=elapsed,
                    phase=phase,
                    command=command,
                    output_limit=args.output_limit,
                    reading=reading,
                    zero_offset_n=zero_offset_n,
                    status=status,
                    sample_window=sample_window,
                    point_index=point_index,
                    repeat_index=repeat_index,
                )
                if (
                    args.mode == "step"
                    and sample_window
                    and point_index is not None
                    and repeat_index is not None
                    and reading.ok
                    and reading.force_n is not None
                ):
                    step_summary_samples.setdefault((repeat_index, point_index), []).append(force_zeroed)
                next_tick += period

        if not args.no_stop_on_exit:
            driver.stop()

        if args.mode == "step":
            assert commands is not None
            summary_csv = args.summary_csv or args.csv.with_name(f"{args.csv.stem}_summary{args.csv.suffix}")
            summary_csv.parent.mkdir(parents=True, exist_ok=True)
            with summary_csv.open("w", newline="", encoding="utf-8") as f:
                summary_writer = csv.writer(f)
                _write_summary_header(summary_writer)
                _write_summary_rows(summary_writer, samples=step_summary_samples, commands=commands)
            log.info(f"saved step summary CSV: {summary_csv}")
            if not args.no_fit:
                points = _load_summary_points(summary_csv)
                points_plot = args.points_plot or summary_csv.with_name(f"{summary_csv.stem}_points.png")
                fit_plot = args.fit_plot or summary_csv.with_name(f"{summary_csv.stem}_fit.png")
                fit_csv = args.fit_csv or summary_csv.with_name(f"{summary_csv.stem}_fit.csv")
                save_discrete_plot(points, points_plot)
                fits = _fit_points(points, args.fit_min_abs_command)
                if fits:
                    write_fit_csv(fit_csv, fits)
                    save_fit_plot(points, fits, fit_plot)
                    log.info(f"saved signed quadratic fit CSV: {fit_csv}")
                    log.info(f"saved signed quadratic fit plot: {fit_plot}")
                else:
                    log.warn("skipping fit outputs: no side had enough valid points")
                log.info(f"saved steady point plot: {points_plot}")
        if args.plot is not None:
            save_plot(args.csv, args.plot)
            log.info(f"saved thrust curve plot: {args.plot}")
        log.info(f"saved thrust curve CSV: {args.csv}")
    except KeyboardInterrupt:
        if not args.no_stop_on_exit:
            driver.stop()
        log.warn("thruster bench measurement interrupted")
    finally:
        sensor.close()
        driver.close()


if __name__ == "__main__":
    main()
