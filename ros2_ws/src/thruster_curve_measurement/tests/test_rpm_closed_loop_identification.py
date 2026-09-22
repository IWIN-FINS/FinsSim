from thruster_curve_measurement.thruster_rpm_closed_loop_identification import (
    build_phases,
    parse_amplitudes,
)


def test_rpm_identification_phase_sequence_has_zero_boundaries():
    phases = build_phases(
        parse_amplitudes("0.2,0.4"),
        settle_sec=2.0,
        active_sec=3.0,
        rest_sec=1.0,
        include_negative=True,
    )

    assert phases[0].name == "initial_zero"
    assert phases[0].command == 0.0
    assert [phase.command for phase in phases] == [0.0, 0.2, 0.0, -0.2, 0.0, 0.4, 0.0, -0.4, 0.0]


def test_parse_amplitudes_rejects_out_of_range_values():
    try:
        parse_amplitudes("0.2,1.1")
    except ValueError as exc:
        assert "(0, 1]" in str(exc)
    else:
        raise AssertionError("out-of-range amplitude was accepted")
