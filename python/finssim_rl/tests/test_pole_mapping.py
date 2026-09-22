"""
Smoke tests for the current adaptive pole-placement PID mapping.

The production module cannot be imported directly in this environment because
the package import path initializes ML-Agents registries. These tests mirror the
small numerical mapping used by the current pole-placement path.
"""
import sys

import numpy as np


PID_PARAM_DIM = 9
TAU_MIN = 0.01
TAU_MAX = 2.710
TAU_EPS = 1e-6


def _stage_mask(stage: str) -> np.ndarray:
    mask = np.zeros(PID_PARAM_DIM, dtype=np.float32)
    if stage == "depth_only":
        mask[0:3] = 1.0
    elif stage == "outer_only":
        mask[3:9] = 1.0
    elif stage == "frozen":
        pass
    else:
        mask[:] = 1.0
    return mask


def _normalized_actions_to_poles(actions: np.ndarray) -> np.ndarray:
    actions = np.clip(np.asarray(actions, dtype=np.float32), -1.0, 1.0)
    ratio = (actions + 1.0) * 0.5
    log_tau = np.log(TAU_MIN) + ratio * (np.log(TAU_MAX) - np.log(TAU_MIN))
    return np.exp(log_tau).astype(np.float32)


def _poles_to_pid(tau_params: np.ndarray) -> np.ndarray:
    tau = np.maximum(np.asarray(tau_params, dtype=np.float32), TAU_EPS)
    if tau.ndim == 1:
        tau = tau.reshape(1, -1)
    groups = tau.reshape(tau.shape[0], -1, 3)
    tau1 = groups[:, :, 0]
    tau2 = groups[:, :, 1]
    tau3 = groups[:, :, 2]
    denom = np.maximum(tau1 * tau2 * tau3, TAU_EPS)

    kp = (tau1 + tau2 + tau3) / denom
    ki = 1.0 / denom
    kd = (tau1 * tau2 + tau2 * tau3 + tau3 * tau1) / denom
    return np.stack([kp, ki, kd], axis=2).reshape(tau.shape[0], PID_PARAM_DIM).astype(np.float32)


def map_policy_output_to_pid_params(actions: np.ndarray, stage: str, base: np.ndarray) -> np.ndarray:
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim == 1:
        actions = actions.reshape(1, -1)

    if actions.shape[1] < PID_PARAM_DIM:
        padded = np.zeros((actions.shape[0], PID_PARAM_DIM), dtype=np.float32)
        padded[:, :actions.shape[1]] = actions
        actions = padded
    elif actions.shape[1] > PID_PARAM_DIM:
        actions = actions[:, :PID_PARAM_DIM]

    pid = _poles_to_pid(_normalized_actions_to_poles(actions))
    base = np.asarray(base, dtype=np.float32).reshape(1, -1)
    mask = _stage_mask(stage).reshape(1, -1)
    result = pid * mask + base * (1.0 - mask)
    return result.astype(np.float32)


def test_pole_mapping_formula() -> bool:
    tau = np.array([[1.0, 2.0, 4.0] * 3], dtype=np.float32)
    pid = _poles_to_pid(tau)
    expected_first_group = np.array([
        (1.0 + 2.0 + 4.0) / 8.0,
        1.0 / 8.0,
        (1.0 * 2.0 + 2.0 * 4.0 + 4.0 * 1.0) / 8.0,
    ], dtype=np.float32)
    return np.allclose(pid[0, :3], expected_first_group)


def test_frozen_stage_returns_default_pid() -> bool:
    base = np.array([[200.0, 0.02, 10.0, 3.0, 0.05, 0.5, 8.0, 0.01, 0.5]])
    pid = map_policy_output_to_pid_params(np.zeros((1, PID_PARAM_DIM), dtype=np.float32), "frozen", base)
    return bool(np.allclose(pid, base))


def test_all_stage_uses_generated_pid_directly() -> bool:
    base = np.array([[200.0, 0.02, 10.0, 3.0, 0.05, 0.5, 8.0, 0.01, 0.5]])
    actions = np.zeros((1, PID_PARAM_DIM), dtype=np.float32)
    expected = _poles_to_pid(_normalized_actions_to_poles(actions))
    pid = map_policy_output_to_pid_params(actions, "all", base)
    return bool(np.allclose(pid, expected))


def test_tau_range_and_all_stage_adjusts_all_groups() -> bool:
    base = np.array([[200.0, 0.02, 10.0, 3.0, 0.05, 0.5, 8.0, 0.01, 0.5]])
    actions = np.array([[2.0, 0.0, -2.0] * 3], dtype=np.float32)
    tau = _normalized_actions_to_poles(actions)
    pid = map_policy_output_to_pid_params(actions, "all", base)

    tau_in_range = np.all((tau >= TAU_MIN - 1e-5) & (tau <= TAU_MAX + 1e-3))
    all_groups_changed = not np.allclose(pid, base)
    return bool(tau_in_range and all_groups_changed)


def test_stage_mask_keeps_horizontal_default_in_depth_only() -> bool:
    base = np.array([[200.0, 0.02, 10.0, 3.0, 0.05, 0.5, 8.0, 0.01, 0.5]])
    actions = np.ones((1, PID_PARAM_DIM), dtype=np.float32)
    pid = map_policy_output_to_pid_params(actions, "depth_only", base)

    horizontal_uses_base = np.allclose(pid[:, 3:9], base[:, 3:9])
    depth_changed = not np.allclose(pid[:, 0:3], base[:, 0:3])
    return bool(horizontal_uses_base and depth_changed)


def main() -> bool:
    tests = [
        ("pole formula", test_pole_mapping_formula),
        ("frozen stage default pid", test_frozen_stage_returns_default_pid),
        ("all stage uses generated pid", test_all_stage_uses_generated_pid_directly),
        ("tau range and all-stage adjustment", test_tau_range_and_all_stage_adjusts_all_groups),
        ("stage mask", test_stage_mask_keeps_horizontal_default_in_depth_only),
    ]

    all_passed = True
    for name, fn in tests:
        passed = fn()
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
        all_passed = all_passed and passed
    return all_passed


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
