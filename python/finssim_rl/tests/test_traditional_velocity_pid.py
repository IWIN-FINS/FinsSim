import os
import sys

import torch


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from finssim_rl.models.traditional_velocity_pid import (  # noqa: E402
    VelocityPIDFeedforwardESO,
    get_traditional_velocity_pid_configs,
)


def test_feedforward_changes_force_command() -> bool:
    ref = torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32)
    cur = torch.zeros_like(ref)

    pid_only = VelocityPIDFeedforwardESO(
        pid_gains=((0.0, 0.0, 0.0),) * 3,
        use_feedforward=False,
        use_eso=False,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    with_ff = VelocityPIDFeedforwardESO(
        pid_gains=((0.0, 0.0, 0.0),) * 3,
        use_feedforward=True,
        feedforward_gain=0.1,
        use_eso=False,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )

    tau_pid = pid_only(ref, cur, return_thrust=False)
    tau_ff = with_ff(ref, cur, return_thrust=False)
    return bool(torch.allclose(tau_pid[:, 0:3], torch.zeros_like(tau_pid[:, 0:3])) and tau_ff[0, 0] > 0.0)


def test_eso_counteracts_positive_velocity_disturbance() -> bool:
    controller = VelocityPIDFeedforwardESO(
        pid_gains=((0.0, 0.0, 0.0),) * 3,
        use_feedforward=False,
        use_eso=True,
        eso_gain=1.0,
        eso_bandwidth=4.0,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    ref = torch.zeros(1, 3)
    _ = controller(ref, torch.zeros(1, 3), return_thrust=False)
    tau = controller(ref, torch.tensor([[0.2, 0.0, 0.0]], dtype=torch.float32), return_thrust=False)
    return bool(tau[0, 0] < 0.0)


def test_feedforward_fades_out_when_same_direction_overspeed() -> bool:
    controller = VelocityPIDFeedforwardESO(
        pid_gains=((0.0, 0.0, 0.0),) * 3,
        use_feedforward=True,
        feedforward_gain=(0.15, 0.15, 0.15),
        feedforward_overspeed_deadband=0.03,
        use_eso=False,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    tau = controller(
        torch.tensor([[0.3, 0.0, 0.0]], dtype=torch.float32),
        torch.tensor([[0.6, 0.0, 0.0]], dtype=torch.float32),
        return_thrust=False,
    )
    return bool(torch.allclose(tau[:, 0], torch.zeros_like(tau[:, 0])))


def test_feedforward_output_is_limited() -> bool:
    controller = VelocityPIDFeedforwardESO(
        pid_gains=((0.0, 0.0, 0.0),) * 3,
        use_feedforward=True,
        feedforward_gain=(1.0, 1.0, 1.0),
        feedforward_output_limit_xyz=(1.2, 0.7, 1.2),
        use_eso=False,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    tau = controller(
        torch.tensor([[0.5, -0.5, 0.5]], dtype=torch.float32),
        torch.zeros(1, 3),
        return_thrust=False,
    )
    return bool(torch.all(tau[:, 0:3].abs() <= torch.tensor([[1.2001, 0.7001, 1.2001]])))


def test_pid_output_gain_scales_final_pid_contribution() -> bool:
    ref = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    cur = torch.zeros_like(ref)
    full = VelocityPIDFeedforwardESO(
        pid_gains=((2.0, 0.0, 0.0),) * 3,
        pid_output_gain=(1.0, 1.0, 1.0),
        use_feedforward=False,
        use_eso=False,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    scaled = VelocityPIDFeedforwardESO(
        pid_gains=((2.0, 0.0, 0.0),) * 3,
        pid_output_gain=(0.25, 0.25, 0.25),
        use_feedforward=False,
        use_eso=False,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    return bool(torch.allclose(scaled(ref, cur, return_thrust=False)[:, 0], 0.25 * full(ref, cur, return_thrust=False)[:, 0]))


def test_derivative_damps_measured_velocity_without_reference_kick() -> bool:
    controller = VelocityPIDFeedforwardESO(
        pid_gains=((0.0, 0.0, 1.0),) * 3,
        derivative_filter_alpha=1.0,
        use_feedforward=False,
        use_eso=False,
        output_limit=100.0,
        output_limit_xyz=(100.0, 100.0, 100.0),
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    ref = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32)
    first = controller(ref, torch.zeros_like(ref), return_thrust=False)
    second = controller(ref, torch.tensor([[0.1, 0.0, 0.0]], dtype=torch.float32), return_thrust=False)
    return bool(torch.allclose(first[:, 0], torch.zeros_like(first[:, 0])) and second[0, 0] < 0.0)


def test_traditional_configs_register_stages() -> bool:
    configs = get_traditional_velocity_pid_configs()
    expected = {
        "traditional_ff_velocity",
        "traditional_pid_velocity",
        "traditional_pid_ff_velocity",
        "traditional_pid_ff_eso_velocity",
    }
    if set(configs) != expected:
        return False

    pid = configs["traditional_pid_velocity"].training_config
    ff_only = configs["traditional_ff_velocity"].training_config
    ff = configs["traditional_pid_ff_velocity"].training_config
    ff_eso = configs["traditional_pid_ff_eso_velocity"].training_config
    return bool(
        not pid.use_feedforward
        and not pid.use_eso
        and pid.pid_output_gain == (1.0, 1.0, 1.0)
        and pid.pid_gains == ((2.40, 0.220, 0.045), (3.00, 0.280, 0.050), (2.70, 0.220, 0.045))
        and pid.derivative_filter_alpha == 0.08
        and pid.allocator_control_linear_range == 0.35
        and ff_only.use_feedforward
        and not ff_only.use_eso
        and not ff_only.use_feedforward_overspeed_gate
        and all(gain == (0.0, 0.0, 0.0) for gain in ff_only.pid_gains)
        and ff.use_feedforward
        and not ff.use_eso
        and ff.use_feedforward_overspeed_gate
        and ff.pid_output_gain == (0.25, 0.25, 0.25)
        and ff.pid_gains == ((0.00, 0.010, 0.00), (0.00, 0.010, 0.00), (0.00, 0.010, 0.00))
        and ff_eso.use_feedforward
        and ff_eso.use_eso
        and all(gain >= 0.0 for gain in ff_eso.feedforward_gain)
        and all(limit > 0.0 for limit in ff_eso.feedforward_output_limit_xyz)
        and ff_eso.eso_gain[1] == 0.0
        and ff_eso.eso_output_limit_xyz[0] > 0.0
        and ff_eso.reference_velocity_limit[1] <= 0.30
        and ff_eso.reference_velocity_rate_limit[1] > 0.0
    )


def test_reference_rate_limit_filters_step_command() -> bool:
    from finssim_rl.models.traditional_velocity_pid import TraditionalVelocityPIDModel

    controller = VelocityPIDFeedforwardESO(
        pid_gains=((0.0, 0.0, 0.0),) * 3,
        use_feedforward=False,
        use_eso=False,
        debug_print_interval=0,
        allocator_debug_print_interval=0,
    )
    model = TraditionalVelocityPIDModel(
        controller=controller,
        observation_format="velocity_normalized_body",
        reference_frame="normalized_body",
        linear_velocity_scale=(1.0, 1.0, 1.0),
        reference_velocity_limit=(1.0, 1.0, 1.0),
        reference_velocity_rate_limit=(0.2, 0.2, 0.2),
    )
    obs = torch.zeros(12, dtype=torch.float32).numpy()
    obs[0:3] = 1.0
    model.predict(obs, deterministic=True)
    return bool(model._filtered_ref_np is not None and torch.as_tensor(model._filtered_ref_np).abs().max() <= 0.0041)


def main() -> bool:
    tests = [
        ("feedforward changes force command", test_feedforward_changes_force_command),
        ("eso counteracts positive velocity disturbance", test_eso_counteracts_positive_velocity_disturbance),
        ("feedforward fades out on same-direction overspeed", test_feedforward_fades_out_when_same_direction_overspeed),
        ("feedforward output is limited", test_feedforward_output_is_limited),
        ("pid output gain scales final pid contribution", test_pid_output_gain_scales_final_pid_contribution),
        ("derivative damps measured velocity without reference kick", test_derivative_damps_measured_velocity_without_reference_kick),
        ("traditional configs register stages", test_traditional_configs_register_stages),
        ("reference rate limit filters step command", test_reference_rate_limit_filters_step_command),
    ]

    all_passed = True
    for name, fn in tests:
        passed = fn()
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
        all_passed = all_passed and passed
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
