#!/usr/bin/env python3
"""Verify the 1Chase1 hierarchy subgoal -> PID frame contract.

This script is intentionally deterministic and does not need Unity to run. It
checks the exact algebra used across the Python hierarchy model, the
TraditionalPositionPID direct body-error path, and the Unity visualizer inverse
of ControllerBodyFrame.
"""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python" / "finssim_rl" / "src"))

from finssim_rl.models.pid_controller import PIDController  # noqa: E402
from finssim_rl.models.thrust_allocator import ThrustAllocator  # noqa: E402
from finssim_rl.models.traditional_position_pid import TraditionalPositionPIDModel  # noqa: E402
from finssim_rl.training.hierarchy_chase_config import (  # noqa: E402
    HIERARCHY_ACTOR_OBS_DIM,
    HierarchyChaseModel,
)


def unity_world_to_controller_body(
    world_delta: np.ndarray,
    *,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    return np.array(
        [
            float(np.dot(world_delta, forward)),
            float(np.dot(world_delta, up)),
            float(np.dot(world_delta, -right)),
        ],
        dtype=np.float32,
    )


def controller_body_to_unity_world(
    body_delta: np.ndarray,
    *,
    right: np.ndarray,
    up: np.ndarray,
    forward: np.ndarray,
) -> np.ndarray:
    return (
        forward * float(body_delta[0])
        + up * float(body_delta[1])
        - right * float(body_delta[2])
    ).astype(np.float32)


def assert_close(name: str, actual: np.ndarray, expected: np.ndarray, atol: float = 1e-6) -> None:
    if not np.allclose(actual, expected, atol=atol):
        raise AssertionError(f"{name}: expected {expected.tolist()}, got {actual.tolist()}")


def assert_signs(name: str, actual: np.ndarray, expected: np.ndarray) -> None:
    signs = np.sign(np.asarray(actual, dtype=np.float32))
    expected_signs = np.sign(np.asarray(expected, dtype=np.float32))
    if not np.array_equal(signs, expected_signs):
        raise AssertionError(f"{name}: expected signs {expected_signs.tolist()}, got {signs.tolist()} from {actual.tolist()}")


def verify_unity_frame_inverse() -> None:
    basis = {
        "right": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "forward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }
    expected_world = {
        "forward": np.array([0.0, 0.0, 1.0], dtype=np.float32),
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "left": np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    }
    body_axes = {
        "forward": np.array([1.0, 0.0, 0.0], dtype=np.float32),
        "up": np.array([0.0, 1.0, 0.0], dtype=np.float32),
        "left": np.array([0.0, 0.0, 1.0], dtype=np.float32),
    }

    for name, body_delta in body_axes.items():
        world_delta = controller_body_to_unity_world(body_delta, **basis)
        roundtrip = unity_world_to_controller_body(world_delta, **basis)
        assert_close(f"Unity body->world axis {name}", world_delta, expected_world[name])
        assert_close(f"Unity body roundtrip axis {name}", roundtrip, body_delta)


def verify_hierarchy_action_decode() -> None:
    model = object.__new__(HierarchyChaseModel)
    model.target_body_delta_limits = (1.5, 0.5, 1.5)
    obs = np.zeros((1, HIERARCHY_ACTOR_OBS_DIM), dtype=np.float32)

    cases = {
        "forward": (np.array([[1.0, 0.0, 0.0]], dtype=np.float32), np.array([[1.5, 0.0, 0.0]], dtype=np.float32)),
        "up": (np.array([[0.0, 1.0, 0.0]], dtype=np.float32), np.array([[0.0, 0.5, 0.0]], dtype=np.float32)),
        "left": (np.array([[0.0, 0.0, 1.0]], dtype=np.float32), np.array([[0.0, 0.0, 1.5]], dtype=np.float32)),
    }
    for name, (action, expected_body_error) in cases.items():
        body_error = model._build_low_level_pid_inputs(action, obs)
        assert_close(f"Hierarchy action decode {name}", body_error, expected_body_error)


def build_control_axis_pid() -> TraditionalPositionPIDModel:
    controller = PIDController(
        action_dim=6,
        device="cpu",
        use_thrust_allocator=False,
        output_range=(-1000.0, 1000.0),
        surge_output_limit=1000.0,
        sway_output_limit=1000.0,
        output_slew_rate_limit=0.0,
        debug_print_interval=0,
    )
    return TraditionalPositionPIDModel(
        controller=controller,
        pid_params=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0),
        enable_yaw_control=False,
        action_dim=6,
        enable_logging=False,
    )


def verify_pid_control_axes() -> None:
    model = build_control_axis_pid()
    body_errors = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    control_6d, _ = model.predict_from_body_position_error(body_errors, deterministic=True)
    expected = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    assert_close("PID direct body error -> control_6d", control_6d, expected)


def verify_allocator_axis_patterns() -> None:
    allocator = ThrustAllocator(
        device="cpu",
        control_axis_ranges=(1.0, 1.0, 1.0, 1.0, 1.0, 1.0),
        allocation_mode="empirical_thruster_mixer",
        debug_print_interval=0,
    )
    tau = torch.eye(6, dtype=torch.float32)[:3]
    thrust = allocator(tau).detach().cpu().numpy()

    assert_signs(
        "Allocator +Fx pattern [V1..V4,H1..H4]",
        thrust[0],
        np.array([0.0, 0.0, 0.0, 0.0, +1.0, +1.0, -1.0, -1.0], dtype=np.float32),
    )
    assert_signs(
        "Allocator +Fy pattern [V1..V4,H1..H4]",
        thrust[1],
        np.array([+1.0, +1.0, +1.0, +1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )
    assert_signs(
        "Allocator +Fz pattern [V1..V4,H1..H4]",
        thrust[2],
        np.array([0.0, 0.0, 0.0, 0.0, -1.0, +1.0, +1.0, -1.0], dtype=np.float32),
    )


def verify_config_requests_direct14() -> None:
    config_paths = [
        ROOT / "configs" / "rl" / "1chase1" / "hierarchy" / "distance_plus_subgoal.yaml",
        ROOT / "configs" / "rl" / "1chase1" / "hierarchy" / "distance_plus_subgoal_eval.yaml",
        ROOT / "configs" / "rl" / "1chase1" / "hierarchy" / "distance_only.yaml",
        ROOT / "configs" / "rl" / "1chase1" / "hierarchy" / "distance_only_eval.yaml",
    ]
    for path in config_paths:
        text = path.read_text(encoding="utf-8")
        if "finsim_1chase1_observation_mode: 14" not in text:
            raise AssertionError(f"{path} does not request 14D observation mode")
        if "- direct14" not in text:
            raise AssertionError(f"{path} does not pass -fins-1chase1-mode direct14")


def main() -> int:
    checks = [
        ("Unity frame inverse", verify_unity_frame_inverse),
        ("Hierarchy action decode", verify_hierarchy_action_decode),
        ("PID body error control axes", verify_pid_control_axes),
        ("Allocator axis patterns", verify_allocator_axis_patterns),
        ("Config direct14 request", verify_config_requests_direct14),
    ]
    for label, fn in checks:
        fn()
        print(f"[OK] {label}")

    print("\nExpected manual scene probes:")
    print("  subgoal [ 1, 0, 0] -> marker in front, PID Fx > 0")
    print("  subgoal [ 0, 1, 0] -> marker upward, PID Fy > 0")
    print("  subgoal [ 0, 0, 1] -> marker left, PID Fz > 0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
