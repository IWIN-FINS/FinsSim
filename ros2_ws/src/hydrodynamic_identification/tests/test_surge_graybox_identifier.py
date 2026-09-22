from __future__ import annotations

import numpy as np

from hydrodynamic_identification.surge_graybox_identifier import Parameters, SurgeTrial, _simulate_trial


def test_zero_command_keeps_zero_state_without_bias() -> None:
    trial = SurgeTrial(
        trial_index=0,
        level_n=0.0,
        time_sec=np.linspace(0.0, 1.0, 21),
        phase=np.asarray(["rest"] * 21, dtype=object),
        command_n=np.zeros(21),
        rpm_wrench_n=np.zeros(21),
        velocity_mps=np.zeros(21),
    )
    values = np.asarray([25.0, 40.0, 10.0, 0.0, 1.0, 0.1, 0.05])
    wrench, velocity = _simulate_trial(trial, values)
    assert np.allclose(wrench, 0.0)
    assert np.allclose(velocity, 0.0)


def test_actuator_delay_defers_force_response() -> None:
    time = np.linspace(0.0, 1.0, 101)
    trial = SurgeTrial(
        trial_index=1,
        level_n=5.0,
        time_sec=time,
        phase=np.asarray(["excitation"] * time.size, dtype=object),
        command_n=np.full(time.size, 5.0),
        rpm_wrench_n=np.zeros(time.size),
        velocity_mps=np.zeros(time.size),
    )
    values = np.asarray([25.0, 40.0, 0.0, 0.0, 1.0, 0.05, 0.20])
    wrench, _ = _simulate_trial(trial, values)
    assert float(np.max(np.abs(wrench[time < 0.19]))) < 1e-9
    assert float(wrench[-1]) > 4.0
