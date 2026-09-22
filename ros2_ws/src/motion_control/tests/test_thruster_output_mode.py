from pathlib import Path

import numpy as np
import pytest
import yaml

from motion_control.controller_node import (
    THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N,
    THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N,
    THRUSTER_OUTPUT_WRENCH6D_FORCE_N,
    MotionControllerNode,
    _vec6_param,
    action_to_thruster_command,
    isaaclab_finsrov_calibrated_action_to_force_n,
    policy_action6d_to_allocator_input,
    rate_limit_policy_action,
    scale_allocator_body_wrench,
)
from motion_control.wrench_action_sender import wrench_action_to_thruster_command


PHYSICAL_HARDWARE_PROFILE = (
    Path(__file__).resolve().parents[1]
    / "config/FinsROV/ppo_wrench_for_pose_physical_wrench_allocator.yaml"
)


class RecordingAllocator:
    def __init__(self, output):
        self.output = np.asarray(output, dtype=np.float32)
        self.inputs = []
        self.device = "cpu"

    def __call__(self, wrench):
        if hasattr(wrench, "detach"):
            self.inputs.append(wrench.detach().cpu().numpy())
        else:
            self.inputs.append(np.asarray(wrench, dtype=np.float32))
        return self.output


class ParameterRecordingNode:
    def __init__(self):
        self.calls = []

    def declare_parameter(self, name, default):
        self.calls.append((name, default))
        return type("DeclaredParameter", (), {"value": default})()


def test_normalized_direct_keeps_clipped_action():
    command = action_to_thruster_command(
        [-2.0, -0.5, 0.0, 0.5, 2.0, 0.1, -0.2, 0.3],
        mode="normalized_direct",
        force_limit_positive=[1.0] * 8,
        force_limit_negative=[1.0] * 8,
    )

    assert command == pytest.approx([-1.0, -0.5, 0.0, 0.5, 1.0, 0.1, -0.2, 0.3])


def test_force_mode_uses_positive_and_negative_limits():
    command = action_to_thruster_command(
        [-1.0, -0.5, 0.0, 0.5, 1.0, 0.25, -0.25, 0.0],
        mode="force_n",
        force_limit_positive=[2.0] * 8,
        force_limit_negative=[4.0] * 8,
    )

    assert command == pytest.approx([-4.0, -2.0, 0.0, 1.0, 2.0, 0.5, -1.0, 0.0])
    assert command.dtype == np.float32


def test_force_alias_uses_newton_output_contract():
    command = action_to_thruster_command(
        [2.0, -2.0, 0.25, -0.25, 0.0, 0.5, -0.5, 1.0],
        mode="force",
        force_limit_positive=[10.0] * 8,
        force_limit_negative=[20.0] * 8,
    )

    assert command == pytest.approx([10.0, -20.0, 2.5, -5.0, 0.0, 5.0, -10.0, 10.0])


def test_isaaclab_finsrov_calibrated_thruster_actions_use_per_thruster_force_limits():
    positive = [8.474877, 8.797188, 8.797188, 8.474877, 8.797188, 8.474877, 8.797188, 8.474877]
    negative = [7.974983, 8.272793, 8.272793, 7.974983, 8.272793, 7.974983, 8.272793, 7.974983]
    force_n = isaaclab_finsrov_calibrated_action_to_force_n(
        [-2.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0],
        force_limit_positive=positive,
        force_limit_negative=negative,
    )

    assert force_n == pytest.approx(
        [-7.974983, -4.1363965, -2.06819825, 0.0, 2.199297, 4.2374385, 8.797188, 8.474877]
    )
    with pytest.raises(ValueError, match="requires 8 actions"):
        isaaclab_finsrov_calibrated_action_to_force_n(
            [0.0] * 6,
            force_limit_positive=positive,
            force_limit_negative=negative,
        )
    with pytest.raises(ValueError, match="custom force conversion"):
        action_to_thruster_command(
            [0.0] * 8,
            mode=THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N,
            force_limit_positive=positive,
            force_limit_negative=negative,
        )


def test_motion_controller_finsrov_calibrated_mode_applies_no_hidden_axis_scale():
    controller = MotionControllerNode.__new__(MotionControllerNode)
    controller._thruster_output_mode = THRUSTER_OUTPUT_ISAACLAB_FINSROV_CALIBRATED_THRUSTER8_FORCE_N
    controller._thruster_force_limit_positive = np.asarray(
        [8.474877, 8.797188, 8.797188, 8.474877, 8.797188, 8.474877, 8.797188, 8.474877],
        dtype=np.float32,
    )
    controller._thruster_force_limit_negative = np.asarray(
        [7.974983, 8.272793, 8.272793, 7.974983, 8.272793, 7.974983, 8.272793, 7.974983],
        dtype=np.float32,
    )
    controller._thruster_output_scale = np.ones(8, dtype=np.float32)
    controller._last_wrench6d_command = np.ones(6, dtype=np.float32)

    command = controller._action_to_thruster_command(
        np.asarray([-1.0, -0.5, 0.0, 0.1, 0.25, 0.5, 0.5, 1.0], dtype=np.float32)
    )

    assert command == pytest.approx([-7.974983, -4.1363965, 0.0, 0.8474877, 2.199297, 4.2374385, 4.398594, 8.474877])
    assert controller._last_wrench6d_command is None


def test_motion_controller_applies_per_thruster_output_scale():
    controller = MotionControllerNode.__new__(MotionControllerNode)
    controller._thruster_output_mode = "force_n"
    controller._thruster_force_limit_positive = np.asarray([10.0] * 8, dtype=np.float32)
    controller._thruster_force_limit_negative = np.asarray([20.0] * 8, dtype=np.float32)
    controller._thruster_output_scale = np.asarray([1.0, 0.5, 0.25, 0.0, 2.0, 1.0, 0.5, 1.0], dtype=np.float32)

    command = controller._action_to_thruster_command(
        np.asarray([1.0, 1.0, 1.0, 1.0, -0.5, -0.5, -0.5, -0.5], dtype=np.float32)
    )

    assert command == pytest.approx([10.0, 5.0, 2.5, 0.0, -20.0, -10.0, -5.0, -10.0])


def test_policy_action6d_to_allocator_input_uses_policy_axis_order():
    allocator_input = policy_action6d_to_allocator_input(
        [1.0, 0.5, -0.25, 0.25, -0.5, 1.0],
        wrench_limits=[20.0, 10.0, 30.0, 0.8, 0.4, 0.2],
    )

    assert allocator_input == pytest.approx([20.0, -7.5, 5.0, 0.2, 0.2, -0.2])


def test_policy_action6d_to_allocator_input_applies_per_axis_wrench_scale():
    allocator_input = policy_action6d_to_allocator_input(
        [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        wrench_limits=[20.0, 10.0, 30.0, 0.8, 0.4, 0.2],
        wrench_scale=[0.5, 0.25, 2.0, 0.0, 1.5, 0.1],
    )

    assert allocator_input == pytest.approx([10.0, 60.0, 2.5, 0.0, 0.02, 0.6])


def test_policy_action6d_allocator_body_order_has_no_legacy_axis_remap():
    allocator_input = policy_action6d_to_allocator_input(
        [1.0, 0.5, -0.25, 0.25, -0.5, 1.0],
        wrench_limits=[20.0, 10.0, 30.0, 0.8, 0.4, 0.2],
        policy_axis_order="allocator_body",
    )

    assert allocator_input == pytest.approx([20.0, 5.0, -7.5, 0.2, -0.2, 0.2])


def test_policy_action6d_to_allocator_input_rejects_negative_wrench_scale():
    with pytest.raises(ValueError, match="wrench_scale"):
        policy_action6d_to_allocator_input(
            [1.0] * 6,
            wrench_limits=[1.0] * 6,
            wrench_scale=[1.0, 1.0, -1.0, 1.0, 1.0, 1.0],
        )


def test_scale_allocator_body_wrench_uses_legacy_policy_axis_order():
    scaled = scale_allocator_body_wrench(
        [10.0, 20.0, 30.0, 4.0, 5.0, 6.0],
        wrench_scale=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    )

    # [Fx, Fy, Fz, Mx, My, Mz] is scaled through
    # [surge, sway, heave, roll, pitch, yaw].
    assert scaled == pytest.approx([1.0, 6.0, 6.0, 1.6, 3.0, 3.0])


def test_thruster8_reprojection_scales_projected_wrench_then_reallocates():
    controller = MotionControllerNode.__new__(MotionControllerNode)
    controller._thruster_output_mode = THRUSTER_OUTPUT_THRUSTER8_REPROJECTED_PHYSICAL_WRENCH_FORCE_N
    controller._thruster_force_limit_positive = np.asarray([10.0] * 8, dtype=np.float32)
    controller._thruster_force_limit_negative = np.asarray([20.0] * 8, dtype=np.float32)
    controller._thruster8_projection_input_scale = np.asarray([1.0, 0.5] + [1.0] * 6, dtype=np.float32)
    controller._wrench6d_wrench_scale = np.asarray([0.1, 0.2, 0.3, 0.0, 0.0, 0.01], dtype=np.float32)
    controller._wrench6d_policy_axis_order = "legacy_surge_sway_heave"
    controller._thruster_output_scale = np.asarray([1.0] * 8, dtype=np.float32)
    controller._last_wrench6d_command = None

    class ProjectionMatrix:
        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return np.eye(6, 8, dtype=np.float32)

    class ProjectionAllocator(RecordingAllocator):
        def __init__(self):
            super().__init__([0.25] * 8)
            self.B = ProjectionMatrix()

    controller._wrench6d_thrust_allocator = ProjectionAllocator()

    command = controller._action_to_thruster_command(
        np.asarray([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    # Raw action becomes [10, 5, 10, 10, 0, 0] N before B projection.
    # The legacy scale maps this body wrench to [1, 1.5, 2, 0, 0, 0].
    assert controller._last_wrench6d_command == pytest.approx([1.0, 1.5, 2.0, 0.0, 0.0, 0.0])
    assert controller._wrench6d_thrust_allocator.inputs[0] == pytest.approx([1.0, 1.5, 2.0, 0.0, 0.0, 0.0])
    assert command == pytest.approx([2.5] * 8)


def test_physical_hardware_profile_uses_documented_force_and_wrench_limits():
    profile = yaml.safe_load(PHYSICAL_HARDWARE_PROFILE.read_text(encoding="utf-8"))
    params = profile["motion_controller"]["ros__parameters"]

    assert params["wrench6d"]["allocation_mode"] == "physical_wrench_allocator"
    assert params["wrench6d"]["wrench_limits"] == pytest.approx(
        [18.586585, 18.068824, 25.636077, 3.791353, 2.535180, 6.947713]
    )
    assert params["wrench6d"]["wrench_scale"] == pytest.approx([1.0, 1.0, 1.0, 0.0, 0.0, 0.1])
    assert params["thruster_force_limits_n"]["positive"] == pytest.approx(
        [8.4749, 7.3809, 7.3809, 8.4749, 7.3809, 8.4749, 7.3809, 8.4749]
    )
    assert params["thruster_force_limits_n"]["negative"] == pytest.approx(
        [7.9750, 5.7618, 5.7618, 7.9750, 5.7618, 7.9750, 5.7618, 7.9750]
    )


def test_physical_hardware_profile_full_surge_reconstructs_the_requested_wrench():
    profile = yaml.safe_load(PHYSICAL_HARDWARE_PROFILE.read_text(encoding="utf-8"))
    params = profile["motion_controller"]["ros__parameters"]
    wrench, _, force_n = wrench_action_to_thruster_command(
        [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        wrench_limits=params["wrench6d"]["wrench_limits"],
        allocation_mode=params["wrench6d"]["allocation_mode"],
        force_limit_positive=params["thruster_force_limits_n"]["positive"],
        force_limit_negative=params["thruster_force_limits_n"]["negative"],
        output_scale=[1.0] * 8,
    )

    from finssim_rl.models.thrust_allocator import ThrustAllocator

    allocator = ThrustAllocator(
        allocation_mode="physical_wrench_allocator",
        physical_wrench_limits=params["wrench6d"]["wrench_limits"],
        thruster_force_limit_positive=params["thruster_force_limits_n"]["positive"],
        thruster_force_limit_negative=params["thruster_force_limits_n"]["negative"],
    )
    reconstructed = allocator.B.detach().cpu().numpy() @ force_n

    assert reconstructed == pytest.approx(wrench, abs=1e-4)


def test_rate_limit_policy_action_limits_change_per_second():
    limited = rate_limit_policy_action(
        [1.0, -1.0, 0.2],
        [0.0, 0.0, 0.0],
        delta_time_sec=0.1,
        max_rate_per_sec=1.5,
    )

    assert limited == pytest.approx([0.15, -0.15, 0.15])


def test_rate_limit_policy_action_zero_rate_preserves_action():
    action = rate_limit_policy_action([1.0, -1.0], [0.0, 0.0], 0.1, 0.0)

    assert action == pytest.approx([1.0, -1.0])


def test_vec6_param_converts_numpy_defaults_to_python_floats():
    node = ParameterRecordingNode()

    values = _vec6_param(
        node,
        "wrench6d.wrench_limits",
        np.asarray([20.0, 20.0, 20.0, 0.8, 0.8, 0.8], dtype=np.float32),
    )

    assert values == pytest.approx([20.0, 20.0, 20.0, 0.8, 0.8, 0.8])
    assert node.calls[0][0] == "wrench6d.wrench_limits"
    assert node.calls[0][1] == pytest.approx([20.0, 20.0, 20.0, 0.8, 0.8, 0.8])


def test_wrench6d_force_mode_requires_controller_allocator_path():
    with pytest.raises(ValueError, match="ThrustAllocator"):
        action_to_thruster_command(
            [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            mode="wrench6d_force_n",
            force_limit_positive=[10.0] * 8,
            force_limit_negative=[10.0] * 8,
        )


def test_motion_controller_wrench6d_mode_uses_shared_allocator_then_force_mode():
    controller = MotionControllerNode.__new__(MotionControllerNode)
    controller._thruster_output_mode = THRUSTER_OUTPUT_WRENCH6D_FORCE_N
    controller._thruster_force_limit_positive = np.asarray([10.0] * 8, dtype=np.float32)
    controller._thruster_force_limit_negative = np.asarray([20.0] * 8, dtype=np.float32)
    controller._thruster_output_scale = np.asarray([1.0] * 8, dtype=np.float32)
    controller._wrench6d_wrench_limits = np.asarray([1.0, 1.0, 4.0, 1.0, 1.0, 1.0], dtype=np.float32)
    controller._wrench6d_thrust_allocator = RecordingAllocator(
        [0.25, -0.25, 0.0, 1.0, -1.0, 0.5, -0.5, 0.0]
    )
    controller._last_wrench6d_command = None

    command = controller._action_to_thruster_command(
        np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    assert controller._last_wrench6d_command == pytest.approx([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    assert controller._wrench6d_thrust_allocator.inputs[0] == pytest.approx([0.0, 4.0, 0.0, 0.0, 0.0, 0.0])
    assert command == pytest.approx([2.5, -5.0, 0.0, 10.0, -20.0, 5.0, -10.0, 0.0])


def test_wrench6d_to_normalized_thruster_action_pads_allocator_output():
    controller = MotionControllerNode.__new__(MotionControllerNode)
    controller._wrench6d_thrust_allocator = RecordingAllocator([2.0, -2.0])

    action = MotionControllerNode._wrench6d_to_normalized_thruster_action(
        controller,
        np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    )

    assert action == pytest.approx([1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])


def test_motion_controller_wrench6d_mode_tracks_debug_wrench_and_scale():
    controller = MotionControllerNode.__new__(MotionControllerNode)
    controller._thruster_output_mode = THRUSTER_OUTPUT_WRENCH6D_FORCE_N
    controller._thruster_force_limit_positive = np.asarray([10.0] * 8, dtype=np.float32)
    controller._thruster_force_limit_negative = np.asarray([10.0] * 8, dtype=np.float32)
    controller._thruster_output_scale = np.asarray([1.0] * 8, dtype=np.float32)
    controller._wrench6d_wrench_limits = np.asarray([1.0, 1.0, 2.0, 1.0, 1.0, 1.0], dtype=np.float32)
    controller._wrench6d_thrust_allocator = RecordingAllocator([0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0])
    controller._last_wrench6d_command = None

    command = controller._action_to_thruster_command(
        np.asarray([0.0, 0.0, 1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    )

    assert controller._last_wrench6d_command == pytest.approx([0.0, 2.0, 0.0, 0.0, 0.0, 0.0])
    assert command == pytest.approx([5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 0.0, 0.0])


def test_direct_wrench_action_sender_matches_controller_wrench_path():
    wrench, normalized, command = wrench_action_to_thruster_command(
        [0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
        wrench_limits=[1.0] * 6,
        force_limit_positive=[10.0] * 8,
        force_limit_negative=[20.0] * 8,
        output_scale=[1.0] * 8,
    )

    assert wrench == pytest.approx([0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    assert normalized == pytest.approx([0.2494, 0.2494, 0.2494, 0.2494, 0.0, 0.0, 0.0, 0.0])
    assert command == pytest.approx([2.494, 2.494, 2.494, 2.494, 0.0, 0.0, 0.0, 0.0])


def test_pitch_allocator_uses_front_pair_against_rear_pair():
    _, normalized, _ = wrench_action_to_thruster_command(
        [0.0, 0.0, 0.0, 0.0, 0.2, 0.0],
        wrench_limits=[1.0] * 6,
        force_limit_positive=[10.0] * 8,
        force_limit_negative=[10.0] * 8,
        output_scale=[1.0] * 8,
    )

    # T1/T4 are front; T2/T3 are rear. A pure pitch action must not use
    # diagonal pairs because the vertical thrust directions are +body-Y.
    assert normalized == pytest.approx([0.37672, -0.37672, -0.37672, 0.37672, 0.0, 0.0, 0.0, 0.0])
