import io
import math
import os
import sys
from contextlib import redirect_stdout

import numpy as np


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from finssim_rl.models.pid_controller import PIDController  # noqa: E402
from finssim_rl.models.traditional_position_pid import (  # noqa: E402
    TraditionalPositionPIDModel,
    build_error4_from_body_position_error,
    build_error4_from_obs,
)


def _build_pose20_obs() -> np.ndarray:
    obs = np.zeros(20, dtype=np.float32)
    obs[0:3] = np.array([1.0, -2.0, 0.5], dtype=np.float32)
    obs[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs[7:10] = np.array([0.2, -1.0, -0.1], dtype=np.float32)
    obs[10:14] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return obs


def _yaw_quat_deg(yaw_deg: float) -> np.ndarray:
    half_yaw = math.radians(yaw_deg) * 0.5
    return np.array([0.0, math.sin(half_yaw), 0.0, math.cos(half_yaw)], dtype=np.float32)


def _build_pose20_axis_target(target_xyz: np.ndarray) -> np.ndarray:
    obs = np.zeros(20, dtype=np.float32)
    obs[0:3] = np.asarray(target_xyz, dtype=np.float32)
    obs[3:7] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    obs[7:10] = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    obs[10:14] = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
    return obs


def test_controller_frame_position_error_keeps_axis_signs() -> None:
    cases = (
        (np.array([1.0, 0.0, 0.0], dtype=np.float32), np.array([1.0, 0.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 1.0, 0.0], dtype=np.float32), np.array([0.0, 1.0, 0.0], dtype=np.float32)),
        (np.array([0.0, 0.0, 1.0], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)),
        (np.array([0.0, 0.0, -1.0], dtype=np.float32), np.array([0.0, 0.0, -1.0], dtype=np.float32)),
    )
    for target, expected in cases:
        error4 = build_error4_from_obs(_build_pose20_axis_target(target))[0]
        assert np.allclose(error4[:3], expected)


def test_direct_body_position_error_keeps_axis_signs() -> None:
    body_errors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float32,
    )

    error4 = build_error4_from_body_position_error(body_errors)

    assert np.allclose(error4[:, :3], body_errors)
    assert np.allclose(error4[:, 3], 0.0)


def test_predict_from_body_position_error_maps_forward_up_left_to_control_axes() -> None:
    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        debug_print_interval=0,
    )
    model = TraditionalPositionPIDModel(
        controller=controller,
        pid_params=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        action_dim=6,
        enable_yaw_control=False,
        enable_logging=False,
    )
    body_errors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    actions, _ = model.predict_from_body_position_error(body_errors, deterministic=True)

    np.testing.assert_allclose(actions[:, 0], [1.0, 0.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(actions[:, 1], [0.0, 1.0, 0.0], atol=1e-6)
    np.testing.assert_allclose(actions[:, 2], [0.0, 0.0, 1.0], atol=1e-6)


def test_disable_logging_suppresses_pid_and_model_prints() -> None:
    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        debug_print_interval=1,
    )
    model = TraditionalPositionPIDModel(
        controller=controller,
        action_dim=6,
        enable_logging=False,
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        action, _ = model.predict(_build_pose20_obs(), deterministic=True)

    assert controller.debug_print_interval == 0
    assert np.asarray(action).shape == (6,)
    assert buffer.getvalue() == ""


def test_enable_logging_keeps_debug_output() -> None:
    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        debug_print_interval=1,
    )
    model = TraditionalPositionPIDModel(
        controller=controller,
        action_dim=6,
        enable_logging=True,
    )

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        action, _ = model.predict(_build_pose20_obs(), deterministic=True)

    output = buffer.getvalue()
    assert np.asarray(action).shape == (6,)
    assert "[PID Step 1]" in output
    assert "[TraditionalPositionPID INFO]" in output


def test_yaw_integral_respects_controller_decay() -> None:
    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        integral_decay=0.5,
    )
    model = TraditionalPositionPIDModel(
        controller=controller,
        yaw_pid_params=(0.0, 1.0, 0.0),
        action_dim=6,
        enable_logging=False,
    )

    model._compute_yaw_moment(np.array([10.0], dtype=np.float32))
    first = float(model._yaw_integral_np[0])
    model._compute_yaw_moment(np.array([10.0], dtype=np.float32))
    second = float(model._yaw_integral_np[0])

    assert np.isclose(first, 0.2)
    assert np.isclose(second, 0.3)


def test_yaw_integral_is_clamped() -> None:
    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        dt=1.0,
        integral_decay=1.0,
        yaw_deadband_deg=0.0,
        yaw_output_limit=0.0,
        yaw_integral_state_limit=5.0,
    )
    model = TraditionalPositionPIDModel(
        controller=controller,
        yaw_pid_params=(0.0, 1.0, 0.0),
        action_dim=6,
        enable_logging=False,
    )

    model._compute_yaw_moment(np.array([10.0], dtype=np.float32))
    model._compute_yaw_moment(np.array([10.0], dtype=np.float32))

    assert np.isclose(float(model._yaw_integral_np[0]), 5.0)


def test_yaw_integral_stops_when_output_is_saturating() -> None:
    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        dt=1.0,
        integral_decay=1.0,
        yaw_deadband_deg=0.0,
        yaw_output_limit=1.0,
        yaw_integral_state_limit=100.0,
    )
    model = TraditionalPositionPIDModel(
        controller=controller,
        yaw_pid_params=(1.0, 1.0, 0.0),
        action_dim=6,
        enable_logging=False,
    )

    yaw_moment = model._compute_yaw_moment(np.array([10.0], dtype=np.float32))

    assert np.isclose(float(yaw_moment[0]), 1.0)
    assert np.isclose(float(model._yaw_integral_np[0]), 0.0)


def test_disable_yaw_control_ignores_target_yaw_changes():
    obs_identity = _build_pose20_obs()
    obs_rotated = _build_pose20_obs()
    obs_rotated[3:7] = _yaw_quat_deg(90.0)

    def build_model() -> TraditionalPositionPIDModel:
        controller = PIDController(
            action_dim=6,
            device="cpu",
            use_thrust_allocator=False,
            debug_print_interval=0,
        )
        return TraditionalPositionPIDModel(
            controller=controller,
            yaw_pid_params=(1.0, 0.0, 0.0),
            enable_yaw_control=False,
            action_dim=6,
            enable_logging=False,
        )

    action_identity, _ = build_model().predict(obs_identity, deterministic=True)
    action_rotated, _ = build_model().predict(obs_rotated, deterministic=True)

    assert np.allclose(action_identity, action_rotated, atol=1e-6)
    assert float(np.asarray(action_identity)[4]) == 0.0
    assert float(np.asarray(action_rotated)[4]) == 0.0


def test_enable_yaw_control_changes_yaw_output():
    obs = _build_pose20_obs()
    obs[3:7] = _yaw_quat_deg(90.0)

    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        debug_print_interval=0,
    )
    model = TraditionalPositionPIDModel(
        controller=controller,
        yaw_pid_params=(1.0, 0.0, 0.0),
        enable_yaw_control=True,
        action_dim=6,
        enable_logging=False,
    )

    action, _ = model.predict(obs, deterministic=True)

    assert np.asarray(action).shape == (6,)
    assert not np.isclose(float(np.asarray(action)[4]), 0.0)


def main() -> bool:
    tests = [
        ("controller-frame position error keeps axis signs", test_controller_frame_position_error_keeps_axis_signs),
        ("direct body position error keeps axis signs", test_direct_body_position_error_keeps_axis_signs),
        (
            "predict from body position error maps forward/up/left to control axes",
            test_predict_from_body_position_error_maps_forward_up_left_to_control_axes,
        ),
        ("disable logging suppresses pid/model prints", test_disable_logging_suppresses_pid_and_model_prints),
        ("enable logging keeps debug output", test_enable_logging_keeps_debug_output),
        ("disable yaw control ignores target yaw changes", test_disable_yaw_control_ignores_target_yaw_changes),
        ("enable yaw control changes yaw output", test_enable_yaw_control_changes_yaw_output),
        ("yaw integral respects controller decay", test_yaw_integral_respects_controller_decay),
        ("yaw integral is clamped", test_yaw_integral_is_clamped),
        ("yaw integral stops when output is saturating", test_yaw_integral_stops_when_output_is_saturating),
    ]

    all_passed = True
    for name, fn in tests:
        try:
            fn()
            passed = True
        except Exception:
            passed = False
        status = "PASS" if passed else "FAIL"
        print(f"[{status}] {name}")
        all_passed = all_passed and passed
    return all_passed


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
