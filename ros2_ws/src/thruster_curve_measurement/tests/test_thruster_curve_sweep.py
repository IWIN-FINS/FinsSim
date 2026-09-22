import pytest

from thruster_curve_measurement.thruster_curve_sweep import (
    SweepConfig,
    command_at_elapsed,
    total_duration,
    values_for_sample,
)


def make_config(**overrides):
    values = dict(
        index=4,
        amplitude=0.4,
        pre_zero_sec=1.0,
        marker_amplitude=0.2,
        marker_on_sec=0.5,
        marker_off_sec=0.5,
        marker_count=2,
        settle_sec=1.0,
        hold_start_sec=1.0,
        ramp_sec=4.0,
        hold_end_sec=1.0,
        post_zero_sec=1.0,
        bidirectional=False,
    )
    values.update(overrides)
    return SweepConfig(**values)


def test_total_duration_one_way():
    assert total_duration(make_config()) == pytest.approx(11.0)


def test_sync_markers_and_ramp():
    config = make_config()
    assert command_at_elapsed(0.25, config).phase == "pre_zero"
    assert command_at_elapsed(1.25, config).phase == "marker_1_on"
    assert command_at_elapsed(1.75, config).phase == "marker_1_off"
    assert command_at_elapsed(4.25, config).command == pytest.approx(-0.4)
    assert command_at_elapsed(6.0, config).command == pytest.approx(-0.2)
    assert command_at_elapsed(8.0, config).command == pytest.approx(0.2)
    assert command_at_elapsed(9.25, config).command == pytest.approx(0.4)


def test_bidirectional_duration_and_return_ramp():
    config = make_config(bidirectional=True)
    assert total_duration(config) == pytest.approx(17.0)
    sample = command_at_elapsed(11.0, config)
    assert sample.phase == "ramp_positive_to_negative"
    assert sample.command == pytest.approx(0.4)
    assert command_at_elapsed(13.0, config).command == pytest.approx(0.0)


def test_values_only_set_target_index():
    sample = command_at_elapsed(6.0, make_config(index=5))
    assert values_for_sample(sample, 5) == [0.0, 0.0, 0.0, 0.0, 0.0, -0.2, 0.0, 0.0]
