from __future__ import annotations

import argparse
from pathlib import Path
import time
from typing import Any, Sequence

import numpy as np
import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from .backend_loader import _ensure_finssim_rl_importable
from .controller_node import (
    DEFAULT_WRENCH6D_SCALE,
    action_to_thruster_command,
    policy_action6d_to_allocator_input,
)


# Default is the legacy unit-scale profile. New physical-wrench YAML profiles set
# wrench_limits in N/N*m and allocation_mode=physical_wrench_allocator explicitly.
DEFAULT_WRENCH_LIMITS = (1.0,) * 6
DEFAULT_FORCE_POSITIVE = (8.4749, 7.3809, 7.3809, 8.4749, 7.3809, 8.4749, 7.3809, 8.4749)
DEFAULT_FORCE_NEGATIVE = (7.9750, 5.7618, 5.7618, 7.9750, 5.7618, 7.9750, 5.7618, 7.9750)
DEFAULT_OUTPUT_SCALE = (1.0,) * 8


def _as_tuple(value: Any, length: int, name: str, default: Sequence[float]) -> tuple[float, ...]:
    if value is None:
        value = default
    values = tuple(float(item) for item in value)
    if len(values) != length:
        raise ValueError(f"{name} must contain {length} values, got {len(values)}")
    return values


def _load_wrench_profile(path: str) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "wrench_limits": DEFAULT_WRENCH_LIMITS,
        "wrench_scale": DEFAULT_WRENCH6D_SCALE,
        "allocation_mode": "empirical_thruster_mixer",
        "force_positive": DEFAULT_FORCE_POSITIVE,
        "force_negative": DEFAULT_FORCE_NEGATIVE,
        "output_scale": DEFAULT_OUTPUT_SCALE,
    }
    if not path:
        return profile

    config_path = Path(path).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as stream:
        root = yaml.safe_load(stream) or {}
    # Controller profiles use either the node-specific key or the ROS-wide
    # wildcard key.  Sim-only profiles intentionally use /** so the same
    # parameters can be applied regardless of the launched node name.
    node_config = root.get("motion_controller") or root.get("/**") or {}
    params = node_config.get("ros__parameters", {})
    wrench = params.get("wrench6d", {}) or {}
    force_limits = params.get("thruster_force_limits_n", {}) or {}
    profile.update(
        wrench_limits=_as_tuple(
            wrench.get("wrench_limits", wrench.get("action_gains")),
            6,
            "wrench6d.wrench_limits",
            DEFAULT_WRENCH_LIMITS,
        ),
        wrench_scale=_as_tuple(
            wrench.get("wrench_scale"),
            6,
            "wrench6d.wrench_scale",
            DEFAULT_WRENCH6D_SCALE,
        ),
        allocation_mode=str(wrench.get("allocation_mode", "empirical_thruster_mixer")),
        force_positive=_as_tuple(
            force_limits.get("positive"), 8,
            "thruster_force_limits_n.positive", DEFAULT_FORCE_POSITIVE,
        ),
        force_negative=_as_tuple(
            force_limits.get("negative"), 8,
            "thruster_force_limits_n.negative", DEFAULT_FORCE_NEGATIVE,
        ),
        output_scale=_as_tuple(
            params.get("thruster_output_scale"), 8,
            "thruster_output_scale", DEFAULT_OUTPUT_SCALE,
        ),
    )
    return profile


def wrench_action_to_thruster_command(
    action: Sequence[float],
    *,
    wrench_limits: Sequence[float],
    allocation_mode: str = "empirical_thruster_mixer",
    force_limit_positive: Sequence[float],
    force_limit_negative: Sequence[float],
    output_scale: Sequence[float],
    wrench_scale: Sequence[float] = DEFAULT_WRENCH6D_SCALE,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the same normalized action -> wrench -> force-N path.

    The returned allocator input is a physical wrench in [Fx, Fy, Fz, Mx,
    My, Mz] order and FinsROV body axes [forward, up, left].
    """
    action_array = np.clip(np.asarray(action, dtype=np.float32).reshape(6), -1.0, 1.0)
    allocator_input = policy_action6d_to_allocator_input(
        action_array,
        wrench_limits=wrench_limits,
        wrench_scale=wrench_scale,
    )

    # Resolve the sibling finssim_rl source tree in the same way as the
    # controller before importing the shared allocator.
    _ensure_finssim_rl_importable()
    from finssim_rl.models.thrust_allocator import ThrustAllocator

    allocator = ThrustAllocator(
        device="cpu",
        deadzone_comp=0.0,
        control_linear_range=1.0,
        control_axis_ranges=(1.0,) * 6,
        allocation_mode=allocation_mode,
        physical_wrench_limits=policy_action6d_to_allocator_input(
            np.ones(6, dtype=np.float32),
            wrench_limits=wrench_limits,
        ),
        thruster_force_limit_positive=force_limit_positive,
        thruster_force_limit_negative=force_limit_negative,
        debug_print_interval=0,
    )
    import torch

    wrench_tensor = torch.as_tensor(allocator_input, dtype=torch.float32, device=allocator.device)
    with torch.no_grad():
        normalized_thrusters = allocator(wrench_tensor).detach().cpu().numpy().astype(np.float32)
    force_command = action_to_thruster_command(
        normalized_thrusters,
        mode="force_n",
        force_limit_positive=force_limit_positive,
        force_limit_negative=force_limit_negative,
    )
    scaled_command = (
        force_command * np.asarray(output_scale, dtype=np.float32)
    ).astype(np.float32, copy=False)
    return allocator_input, normalized_thrusters, scaled_command


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Send one normalized 6D wrench meta-action through the same allocator "
            "and force-N path as motion_controller."
        )
    )
    parser.add_argument(
        "--action",
        nargs=6,
        type=float,
        required=True,
        metavar=("SURGE", "SWAY", "HEAVE", "ROLL", "PITCH", "YAW"),
        help="Normalized action in policy order [surge, sway, heave, roll, pitch, yaw], each in [-1, 1].",
    )
    parser.add_argument("--duration", type=float, default=5.0, help="Hold duration in seconds.")
    parser.add_argument(
        "--ramp-time",
        type=float,
        default=0.5,
        help="Ramp from zero to the requested action over this many seconds (0 for an immediate step).",
    )
    parser.add_argument("--rate", type=float, default=20.0, help="Publication rate in Hz.")
    parser.add_argument("--topic", default="/finsrov/thrusters_out", help="Thruster output topic.")
    parser.add_argument(
        "--config",
        default="",
        help="Optional wrench controller YAML. Defaults match ppo_wrench_for_pose.yaml.",
    )
    parser.add_argument(
        "--match-timeout",
        type=float,
        default=3.0,
        help="Seconds to wait for the hardware bridge subscriber.",
    )
    parser.add_argument(
        "--zero-count",
        type=int,
        default=5,
        help="Number of zero-force messages sent during shutdown.",
    )
    return parser


def _wait_for_subscriber(node: Node, publisher: Any, timeout_sec: float) -> None:
    deadline = time.monotonic() + max(float(timeout_sec), 0.0)
    while publisher.get_subscription_count() == 0 and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
        time.sleep(0.02)
    if publisher.get_subscription_count() == 0:
        node.get_logger().warning("no matched subscriber; the hardware bridge may not be running")


def _publish_command(publisher: Any, values: Sequence[float]) -> None:
    publisher.publish(Float32MultiArray(data=[float(value) for value in values]))


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    action = np.asarray(args.action, dtype=np.float32)
    if np.any(np.abs(action) > 1.0):
        raise SystemExit("each action component must be in [-1, 1]")
    if args.duration < 0.0 or args.ramp_time < 0.0 or args.rate <= 0.0:
        raise SystemExit("duration and ramp-time must be non-negative, and rate must be positive")

    try:
        profile = _load_wrench_profile(args.config)
        allocator_input, normalized, command = wrench_action_to_thruster_command(
            action,
            wrench_limits=profile["wrench_limits"],
            wrench_scale=profile["wrench_scale"],
            allocation_mode=str(profile["allocation_mode"]),
            force_limit_positive=profile["force_positive"],
            force_limit_negative=profile["force_negative"],
            output_scale=profile["output_scale"],
        )
    except (OSError, ValueError, ImportError) as exc:
        raise SystemExit(f"failed to prepare wrench action: {exc}") from exc

    rclpy.init(args=None)
    node = Node("send_wrench_action")
    publisher = node.create_publisher(Float32MultiArray, args.topic, 10)
    period = 1.0 / float(args.rate)
    zero = np.zeros(8, dtype=np.float32)
    started = time.monotonic()
    try:
        _wait_for_subscriber(node, publisher, args.match_timeout)
        node.get_logger().info(
            "wrench action: "
            f"policy=[{', '.join(f'{float(value):+.3f}' for value in action)}], "
            f"allocator_input=[{', '.join(f'{float(value):+.4f}' for value in allocator_input)}], "
            f"normalized_thrusters=[{', '.join(f'{float(value):+.4f}' for value in normalized)}], "
            f"force_n=[{', '.join(f'{float(value):+.4f}' for value in command)}]"
        )

        # Publish continuously so a stale command cannot outlive the test.
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= float(args.ramp_time) + float(args.duration):
                break
            fraction = 1.0 if args.ramp_time <= 0.0 else min(elapsed / float(args.ramp_time), 1.0)
            _publish_command(publisher, command * np.float32(fraction))
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(period)
    except KeyboardInterrupt:
        node.get_logger().warning("interrupted; sending zero force")
    finally:
        for _ in range(max(int(args.zero_count), 1)):
            _publish_command(publisher, zero)
            rclpy.spin_once(node, timeout_sec=0.0)
            time.sleep(min(period, 0.05))
        node.destroy_node()
        rclpy.shutdown()
