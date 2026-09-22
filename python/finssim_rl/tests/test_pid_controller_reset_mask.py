import torch

from finssim_rl.models.pid_controller import PIDController


def test_pid_controller_partial_reset_only_clears_done_slots() -> None:
    controller = PIDController(device="cpu", use_thrust_allocator=False)

    error = torch.tensor(
        [
            [1.0, -0.5, 0.25, 0.0],
            [0.5, 0.25, -0.75, 0.0],
            [-0.25, 0.75, 0.5, 0.0],
        ],
        dtype=torch.float32,
    )
    pid_params = torch.ones((3, PIDController.PARAM_DIM), dtype=torch.float32)

    controller(error, pid_params, return_thrust=False)

    integral_before = controller.integral_state.clone()
    prev_before = controller.prev_error_state.clone()
    derivative_before = controller.derivative_filter_state.clone()

    controller.reset(torch.tensor([False, True, False]))

    assert torch.allclose(controller.integral_state[:, 1], torch.zeros_like(controller.integral_state[:, 1]))
    assert torch.allclose(controller.prev_error_state[:, 1], torch.zeros_like(controller.prev_error_state[:, 1]))
    assert torch.allclose(
        controller.derivative_filter_state[:, 1],
        torch.zeros_like(controller.derivative_filter_state[:, 1]),
    )

    assert torch.allclose(controller.integral_state[:, 0], integral_before[:, 0])
    assert torch.allclose(controller.integral_state[:, 2], integral_before[:, 2])
    assert torch.allclose(controller.prev_error_state[:, 0], prev_before[:, 0])
    assert torch.allclose(controller.prev_error_state[:, 2], prev_before[:, 2])
    assert torch.allclose(controller.derivative_filter_state[:, 0], derivative_before[:, 0])
    assert torch.allclose(controller.derivative_filter_state[:, 2], derivative_before[:, 2])


def test_pid_controller_applies_integral_decay_before_accumulating() -> None:
    controller = PIDController(
        device="cpu",
        use_thrust_allocator=False,
        integral_decay=0.5,
        dt=1.0,
    )
    error = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    pid_params = torch.zeros((1, PIDController.PARAM_DIM), dtype=torch.float32)

    controller(error, pid_params, return_thrust=False)
    assert torch.allclose(controller.integral_state[PIDController.SURGE, :], torch.tensor([1.0]))

    controller(error, pid_params, return_thrust=False)
    assert torch.allclose(controller.integral_state[PIDController.SURGE, :], torch.tensor([1.5]))


def test_pid_controller_clamps_integral_state() -> None:
    controller = PIDController(
        device="cpu",
        use_thrust_allocator=False,
        integral_decay=1.0,
        integral_state_limit=0.75,
        dt=1.0,
    )
    error = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    pid_params = torch.zeros((1, PIDController.PARAM_DIM), dtype=torch.float32)

    controller(error, pid_params, return_thrust=False)
    controller(error, pid_params, return_thrust=False)

    assert torch.allclose(controller.integral_state[PIDController.SURGE, :], torch.tensor([0.75]))


def test_pid_controller_blocks_integral_when_output_is_saturating() -> None:
    controller = PIDController(
        device="cpu",
        use_thrust_allocator=False,
        integral_decay=1.0,
        integral_state_limit=100.0,
        dt=1.0,
        output_range=(-1.0, 1.0),
        surge_output_limit=1.0,
    )
    error = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    pid_params = torch.zeros((1, PIDController.PARAM_DIM), dtype=torch.float32)
    pid_params[:, 3:6] = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)

    control = controller(error, pid_params, return_thrust=False)

    assert torch.allclose(control[:, 0], torch.tensor([1.0]))
    assert torch.allclose(controller.integral_state[PIDController.SURGE, :], torch.tensor([0.0]))


def test_pid_controller_derivative_filter_alpha_is_configurable() -> None:
    error = torch.tensor([[2.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    pid_params = torch.zeros((1, PIDController.PARAM_DIM), dtype=torch.float32)
    pid_params[:, 3:6] = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float32)

    filtered_out = PIDController(
        device="cpu",
        use_thrust_allocator=False,
        derivative_filter_alpha=0.0,
        dt=1.0,
    )(error, pid_params, return_thrust=False)
    unfiltered_out = PIDController(
        device="cpu",
        use_thrust_allocator=False,
        derivative_filter_alpha=1.0,
        dt=1.0,
    )(error, pid_params, return_thrust=False)

    assert torch.allclose(filtered_out[:, 0], torch.tensor([0.0]))
    assert torch.allclose(unfiltered_out[:, 0], torch.tensor([2.0]))


def test_pid_controller_limits_control_slew_rate() -> None:
    controller = PIDController(
        device="cpu",
        use_thrust_allocator=False,
        output_slew_rate_limit=5.0,
        dt=0.1,
    )
    error = torch.tensor([[10.0, 0.0, 0.0, 0.0]], dtype=torch.float32)
    pid_params = torch.zeros((1, PIDController.PARAM_DIM), dtype=torch.float32)
    pid_params[:, 3:6] = torch.tensor([100.0, 0.0, 0.0], dtype=torch.float32)

    first = controller(error, pid_params, return_thrust=False)
    second = controller(error, pid_params, return_thrust=False)

    assert torch.allclose(first[:, 0], torch.tensor([0.5]))
    assert torch.allclose(second[:, 0], torch.tensor([1.0]))
