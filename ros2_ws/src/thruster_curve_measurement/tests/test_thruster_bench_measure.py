import pytest

from thruster_curve_measurement.thruster_bench_measure import (
    _build_parser,
    _ramp_config_from_args,
    load_config_defaults,
    parse_args,
)
from thruster_curve_measurement.thruster_curve_sweep import total_duration


def test_load_config_defaults_uses_bench_section(tmp_path):
    config = tmp_path / "bench.yaml"
    config.write_text(
        """
thruster_bench_measure:
  thruster_port: /dev/ttyUSB0
  sensor_port: /dev/ttyUSB1
  csv: /tmp/bench.csv
  output_limit: 0.2
  step_commands:
    - -0.1
    - 0.0
    - 0.1
""",
        encoding="utf-8",
    )

    defaults = load_config_defaults(config, _build_parser())

    assert defaults["thruster_port"] == "/dev/ttyUSB0"
    assert defaults["sensor_port"] == "/dev/ttyUSB1"
    assert defaults["csv"] == "/tmp/bench.csv"
    assert defaults["output_limit"] == pytest.approx(0.2)
    assert defaults["step_commands"] == [-0.1, 0.0, 0.1]


def test_parse_args_accepts_config_and_cli_override(tmp_path):
    config = tmp_path / "bench.yaml"
    config.write_text(
        """
thruster_bench_measure:
  thruster_port: /dev/ttyUSB0
  sensor_port: /dev/ttyUSB1
  csv: /tmp/from_config.csv
  output_limit: 0.2
""",
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--csv",
            "/tmp/from_cli.csv",
            "--print-plan",
        ]
    )

    assert args.thruster_port == "/dev/ttyUSB0"
    assert args.sensor_port == "/dev/ttyUSB1"
    assert str(args.csv) == "/tmp/from_cli.csv"
    assert args.output_limit == pytest.approx(0.2)
    assert args.print_plan


def test_loop_mode_forces_continuous_bidirectional_bench_ramp():
    args = parse_args(
        [
            "--thruster-port",
            "/dev/ttyUSB0",
            "--sensor-port",
            "/dev/ttyUSB1",
            "--csv",
            "/tmp/bench_loop.csv",
            "--mode",
            "loop",
            "--ramp",
            "30",
            "--hold-start",
            "9",
            "--hold-end",
            "9",
        ]
    )

    config = _ramp_config_from_args(args)

    assert config.bidirectional
    assert config.hold_start_sec == pytest.approx(0.0)
    assert config.hold_end_sec == pytest.approx(0.0)
    assert total_duration(config) == pytest.approx(args.pre_zero + args.settle + 60.0 + args.post_zero)
