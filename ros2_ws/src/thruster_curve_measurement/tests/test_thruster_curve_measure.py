import csv
import io

import pytest

from thruster_curve_measurement.force_sensor_modbus import ForceReading
from thruster_curve_measurement.thruster_curve_measure import (
    SignedQuadraticFit,
    StepConfig,
    SummaryPoint,
    _build_parser,
    _config_from_args,
    _write_header,
    _write_row,
    _write_summary_header,
    _write_summary_rows,
    fit_omega_quadratic,
    fit_signed_quadratic,
    load_raw_rpm_force_points,
    load_config_defaults,
    make_default_step_commands,
    parse_step_commands,
    parse_args,
    signed_quadratic_force,
    step_sample_at_elapsed,
    step_total_duration,
    values_for_index_command,
    write_omega_fit_csv,
    write_fit_csv,
)
from thruster_curve_measurement.thruster_curve_sweep import total_duration


def make_step_config(**overrides):
    values = dict(
        index=4,
        amplitude=1.0,
        step=0.5,
        commands=(-1.0, -0.5, 0.0, 0.5, 1.0),
        pre_zero_sec=1.0,
        settle_sec=1.0,
        hold_sec=5.0,
        sample_sec=2.0,
        rest_sec=1.0,
        repeat_rest_sec=0.0,
        post_zero_sec=1.0,
        repeat_count=1,
    )
    values.update(overrides)
    return StepConfig(**values)


def test_default_step_commands_cover_full_range():
    commands = make_default_step_commands(1.0, 0.25)

    assert commands[0] == pytest.approx(-1.0)
    assert commands[-1] == pytest.approx(1.0)
    assert 0.0 in commands
    assert len(commands) == 9


def test_parse_step_commands_validates_range():
    assert parse_step_commands("-1, -0.2, 0, 0.2, 1") == (-1.0, -0.2, 0.0, 0.2, 1.0)

    with pytest.raises(ValueError):
        parse_step_commands("-1.2,0,1")


def test_step_timing_uses_steady_sample_window_at_end_of_hold():
    config = make_step_config()
    assert step_total_duration(config) == pytest.approx(33.0)

    assert step_sample_at_elapsed(0.5, config).phase == "pre_zero"
    assert step_sample_at_elapsed(1.5, config).phase == "settle_zero"

    first_settle = step_sample_at_elapsed(2.5, config)
    assert first_settle.phase == "step_settle"
    assert first_settle.command == pytest.approx(-1.0)
    assert first_settle.point_index == 0
    assert first_settle.repeat_index == 0
    assert not first_settle.sample_window

    first_sample = step_sample_at_elapsed(5.2, config)
    assert first_sample.phase == "step_sample"
    assert first_sample.command == pytest.approx(-1.0)
    assert first_sample.sample_window

    rest = step_sample_at_elapsed(7.5, config)
    assert rest.phase == "step_rest_zero"
    assert rest.command == pytest.approx(0.0)

    second_point = step_sample_at_elapsed(8.1, config)
    assert second_point.point_index == 1
    assert second_point.command == pytest.approx(-0.5)


def test_step_repeat_indexes_advance_after_full_command_list():
    config = make_step_config(repeat_count=2, commands=(-1.0, 1.0))

    repeated = step_sample_at_elapsed(14.1, config)

    assert repeated.point_index == 0
    assert repeated.repeat_index == 1
    assert repeated.command == pytest.approx(-1.0)


def test_step_repeat_rest_inserts_zero_between_full_command_lists():
    config = make_step_config(repeat_count=2, commands=(-1.0, 1.0), repeat_rest_sec=10.0)
    assert step_total_duration(config) == pytest.approx(37.0)

    rest = step_sample_at_elapsed(14.1, config)
    assert rest.phase == "repeat_rest_zero"
    assert rest.command == pytest.approx(0.0)
    assert rest.repeat_index == 0

    repeated = step_sample_at_elapsed(24.1, config)
    assert repeated.phase == "step_settle"
    assert repeated.point_index == 0
    assert repeated.repeat_index == 1
    assert repeated.command == pytest.approx(-1.0)


def test_values_only_set_target_index():
    assert values_for_index_command(5, -0.4) == [0.0, 0.0, 0.0, 0.0, 0.0, -0.4, 0.0, 0.0]


def test_csv_rows_include_sample_metadata_and_zeroed_force():
    output = io.StringIO()
    writer = csv.writer(output)
    _write_header(writer)
    _write_row(
        writer,
        wall_time=1.0,
        monotonic_time=2.0,
        elapsed_sec=3.0,
        phase="step_sample",
        index=4,
        values=values_for_index_command(4, 0.5),
        reading=ForceReading(ok=True, raw_value=100, signed_value=100, force_n=2.5),
        zero_offset_n=0.5,
        rpm_values=[0.0, 0.0, 0.0, 0.0, 1200.0, 0.0, 0.0, 0.0],
        rpm_age_sec=0.1,
        sample_window=True,
        point_index=3,
        repeat_index=0,
    )

    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert rows[0]["target_command"] == "0.500000"
    assert rows[0]["force_zeroed_n"] == "2.000000"
    assert rows[0]["target_rpm"] == "1200.000000"
    assert float(rows[0]["target_omega_rad_s"]) == pytest.approx(125.663706, abs=1e-6)
    assert rows[0]["rpm_H_LF"] == "1200.000000"
    assert rows[0]["sample_window"] == "1"
    assert rows[0]["point_index"] == "3"


def test_summary_rows_report_statistics_per_step_point():
    output = io.StringIO()
    writer = csv.writer(output)
    _write_summary_header(writer)
    _write_summary_rows(
        writer,
        index=4,
        samples={(0, 1): [1.0, 2.0, 3.0]},
        rpm_samples={(0, 1): [1000.0, 1200.0, 1400.0]},
        commands=(-1.0, -0.5),
    )

    rows = list(csv.DictReader(io.StringIO(output.getvalue())))
    assert rows[0]["target_command"] == "-0.500000"
    assert rows[0]["sample_count"] == "3"
    assert rows[0]["force_mean_n"] == "2.000000"
    assert rows[0]["force_median_n"] == "2.000000"
    assert rows[0]["rpm_mean"] == "1200.000000"
    assert float(rows[0]["omega_mean_rad_s"]) == pytest.approx(125.663706, abs=1e-6)


def test_signed_quadratic_fit_recovers_positive_coefficients():
    points = [
        SummaryPoint(
            command=u,
            force_mean_n=3.0 * u * abs(u) + 2.0 * u,
            force_median_n=0.0,
            force_std_n=0.0,
            sample_count=10,
            repeat_index=0,
            point_index=i,
        )
        for i, u in enumerate((0.2, 0.5, 1.0))
    ]

    fit = fit_signed_quadratic(points, side="positive")

    assert fit.quadratic_coeff == pytest.approx(3.0)
    assert fit.linear_coeff == pytest.approx(2.0)
    assert fit.rmse_n == pytest.approx(0.0)
    assert signed_quadratic_force(0.75, fit) == pytest.approx(3.0 * 0.75 * 0.75 + 2.0 * 0.75)


def test_signed_quadratic_fit_recovers_negative_coefficients():
    points = [
        SummaryPoint(
            command=u,
            force_mean_n=4.0 * u * abs(u) + 1.0 * u,
            force_median_n=0.0,
            force_std_n=0.0,
            sample_count=10,
            repeat_index=0,
            point_index=i,
        )
        for i, u in enumerate((-1.0, -0.5, -0.2))
    ]

    fit = fit_signed_quadratic(points, side="negative")

    assert fit.quadratic_coeff == pytest.approx(4.0)
    assert fit.linear_coeff == pytest.approx(1.0)
    assert fit.command_min == pytest.approx(-1.0)
    assert fit.command_max == pytest.approx(-0.2)


def test_write_fit_csv_records_model(tmp_path):
    fit_csv = tmp_path / "fit.csv"
    write_fit_csv(
        fit_csv,
        [
            SignedQuadraticFit(
                side="positive",
                quadratic_coeff=3.0,
                linear_coeff=2.0,
                rmse_n=0.1,
                sample_count=5,
                command_min=0.1,
                command_max=1.0,
            )
        ],
    )

    with fit_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["side"] == "positive"
    assert "command * abs(command)" in rows[0]["model"]
    assert rows[0]["quadratic_coeff"] == "3"
    assert rows[0]["linear_coeff"] == "2"


def test_omega_quadratic_fit_recovers_c1_and_writes_csv(tmp_path):
    c1 = 0.003
    points = []
    for i, rpm in enumerate((-1200.0, -800.0, 800.0, 1200.0)):
        omega = rpm * 2.0 * 3.141592653589793 / 60.0
        points.append(
            SummaryPoint(
                command=rpm / 1200.0,
                force_mean_n=c1 * omega * abs(omega),
                force_median_n=0.0,
                force_std_n=0.0,
                sample_count=10,
                repeat_index=0,
                point_index=i,
                rpm_mean=rpm,
                rpm_std=1.0,
                omega_mean_rad_s=omega,
            )
        )

    fit = fit_omega_quadratic(points, side="all")
    assert fit.c1 == pytest.approx(c1)
    assert fit.rmse_n == pytest.approx(0.0)

    rpm_fit_csv = tmp_path / "rpm_fit.csv"
    write_omega_fit_csv(rpm_fit_csv, [fit])
    with rpm_fit_csv.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["side"] == "all"
    assert rows[0]["model"] == "force_n = c1 * omega_rad_s * abs(omega_rad_s)"
    assert float(rows[0]["c1"]) == pytest.approx(c1)


def test_raw_rpm_force_points_support_dense_rpm_sweep_fit(tmp_path):
    raw_csv = tmp_path / "raw.csv"
    with raw_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        _write_header(writer)
        for i, rpm in enumerate((-1200.0, -900.0, -600.0, 600.0, 900.0, 1200.0)):
            omega = rpm * 2.0 * 3.141592653589793 / 60.0
            force_n = 0.0025 * omega * abs(omega)
            _write_row(
                writer,
                wall_time=1.0 + i,
                monotonic_time=2.0 + i,
                elapsed_sec=float(i),
                phase="ramp_negative_to_positive" if i < 3 else "ramp_positive_to_negative",
                index=4,
                values=values_for_index_command(4, rpm / 1200.0),
                reading=ForceReading(ok=True, raw_value=0, signed_value=0, force_n=force_n),
                zero_offset_n=0.0,
                rpm_values=[0.0, 0.0, 0.0, 0.0, rpm, 0.0, 0.0, 0.0],
                rpm_age_sec=0.01,
            )

    points = load_raw_rpm_force_points(
        raw_csv,
        phases={"ramp_negative_to_positive", "ramp_positive_to_negative"},
    )
    fit = fit_omega_quadratic(points, side="all")

    assert len(points) == 6
    assert fit.c1 == pytest.approx(0.0025)
    assert fit.sample_count == 6


def test_load_config_defaults_accepts_hyphen_and_underscore_keys(tmp_path):
    config = tmp_path / "profile.yaml"
    config.write_text(
        """
thruster_curve_measure:
  step-hold: 4.0
  step_sample: 1.5
  sensor_port: /dev/ttyUSB9
""",
        encoding="utf-8",
    )

    defaults = load_config_defaults(config, _build_parser())

    assert defaults["step_hold"] == 4.0
    assert defaults["step_sample"] == 1.5
    assert defaults["sensor_port"] == "/dev/ttyUSB9"


def test_parse_args_uses_config_defaults_and_cli_overrides(tmp_path):
    output_dir = tmp_path / "curves"
    config = tmp_path / "profile.yaml"
    config.write_text(
        """
thruster_curve_measure:
  index: 4
  sensor_port: /dev/ttyUSB0
  output_dir: {output_dir}
  motor_name: M001
  amplitude: 0.5
  step_hold: 4.0
  step_sample: 1.0
  zero_samples: 12
  fit_min_abs_command: 0.05
  no_fit: true
  force_topic: /test/force
  force_log_period: 0.25
""".format(output_dir=output_dir),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--index",
            "5",
            "--motor-name",
            "M002",
        ]
    )

    assert args.index == 5
    assert args.topic == "/finsrov/thrusters_out"
    assert args.output_dir == output_dir
    assert args.motor_name == "M002"
    assert args.run_dir == output_dir / "M002"
    assert args.csv == output_dir / "M002" / "raw.csv"
    assert args.summary_csv == output_dir / "M002" / "summary.csv"
    assert args.plot == output_dir / "M002" / "curve.png"
    assert args.points_plot == output_dir / "M002" / "points.png"
    assert args.fit_csv == output_dir / "M002" / "fit.csv"
    assert args.fit_plot == output_dir / "M002" / "fit.png"
    assert args.rpm_fit_csv == output_dir / "M002" / "rpm_fit.csv"
    assert args.rpm_fit_plot == output_dir / "M002" / "rpm_fit.png"
    assert args.sensor_port == "/dev/ttyUSB0"
    assert args.amplitude == 0.5
    assert args.step_hold == 4.0
    assert args.step_sample == 1.0
    assert args.zero_samples == 12
    assert args.fit_min_abs_command == pytest.approx(0.05)
    assert args.no_fit
    assert args.force_topic == "/test/force"
    assert args.force_log_period == pytest.approx(0.25)


def test_parse_args_rejects_existing_output_dir_without_overwrite(tmp_path, capsys):
    run_dir = tmp_path / "curves" / "M001"
    run_dir.mkdir(parents=True)
    (run_dir / "raw.csv").write_text("existing\n", encoding="utf-8")

    with pytest.raises(SystemExit):
        parse_args(
            [
                "--index",
                "4",
                "--sensor-port",
                "/dev/ttyUSB0",
                "--output-dir",
                str(tmp_path / "curves"),
                "--motor-name",
                "M001",
            ]
        )

    assert "--overwrite" in capsys.readouterr().err


def test_parse_args_allows_existing_output_dir_with_overwrite(tmp_path):
    run_dir = tmp_path / "curves" / "M001"
    run_dir.mkdir(parents=True)
    (run_dir / "raw.csv").write_text("existing\n", encoding="utf-8")

    args = parse_args(
        [
            "--index",
            "4",
            "--sensor-port",
            "/dev/ttyUSB0",
            "--output-dir",
            str(tmp_path / "curves"),
            "--motor-name",
            "M001",
            "--overwrite",
        ]
    )

    assert args.overwrite
    assert args.run_dir == run_dir
    assert args.csv == run_dir / "raw.csv"


def test_parse_args_rejects_motor_name_path(tmp_path, capsys):
    with pytest.raises(SystemExit):
        parse_args(
            [
                "--index",
                "4",
                "--sensor-port",
                "/dev/ttyUSB0",
                "--output-dir",
                str(tmp_path / "curves"),
                "--motor-name",
                "group/M001",
            ]
        )

    assert "single directory name" in capsys.readouterr().err


def test_parse_args_accepts_legacy_csv_path():
    args = parse_args(
        [
            "--index",
            "4",
            "--sensor-port",
            "/dev/ttyUSB0",
            "--csv",
            "/tmp/from_cli.csv",
        ]
    )

    assert args.run_dir is None
    assert args.csv.name == "from_cli.csv"
    assert args.summary_csv.name == "from_cli_summary.csv"
    assert args.points_plot.name == "from_cli_summary_points.png"
    assert args.fit_csv.name == "from_cli_summary_fit.csv"
    assert args.fit_plot.name == "from_cli_summary_fit.png"


def test_loop_mode_forces_continuous_bidirectional_ramp(tmp_path):
    args = parse_args(
        [
            "--index",
            "4",
            "--sensor-port",
            "/dev/ttyUSB0",
            "--output-dir",
            str(tmp_path / "curves"),
            "--motor-name",
            "M001_loop",
            "--mode",
            "loop",
            "--amplitude",
            "1.0",
            "--ramp",
            "30",
            "--hold-start",
            "9",
            "--hold-end",
            "9",
        ]
    )

    config = _config_from_args(args)

    assert config.bidirectional
    assert config.hold_start_sec == pytest.approx(0.0)
    assert config.hold_end_sec == pytest.approx(0.0)
    assert total_duration(config) == pytest.approx(args.pre_zero + args.settle + 60.0 + args.post_zero)


def test_rpm_sweep_mode_forces_continuous_bidirectional_raw_fit(tmp_path):
    args = parse_args(
        [
            "--index",
            "4",
            "--sensor-port",
            "/dev/ttyUSB0",
            "--output-dir",
            str(tmp_path / "curves"),
            "--motor-name",
            "M001_rpm",
            "--mode",
            "rpm-sweep",
            "--ramp",
            "45",
            "--hold-start",
            "9",
            "--hold-end",
            "9",
        ]
    )

    config = _config_from_args(args)

    assert config.bidirectional
    assert config.hold_start_sec == pytest.approx(0.0)
    assert config.hold_end_sec == pytest.approx(0.0)
    assert total_duration(config) == pytest.approx(args.pre_zero + args.settle + 90.0 + args.post_zero)


def test_parse_args_ignores_ros_launch_arguments(tmp_path):
    output_dir = tmp_path / "curves"
    config = tmp_path / "profile.yaml"
    config.write_text(
        """
thruster_curve_measure:
  index: 4
  sensor_port: /dev/ttyUSB0
  output_dir: {output_dir}
  motor_name: M001
""".format(output_dir=output_dir),
        encoding="utf-8",
    )

    args = parse_args(
        [
            "--config",
            str(config),
            "--print-plan",
            "--ros-args",
            "-r",
            "__node:=thruster_curve_measure",
        ]
    )

    assert args.print_plan
    assert args.index == 4
    assert args.sensor_port == "/dev/ttyUSB0"
    assert args.run_dir == output_dir / "M001"
