#!/usr/bin/env python3
"""Measure Unity's actual thruster wrench for several requested body Fx values.

The command follows the production PPO wrench path:
``policy action -> ThrustAllocator.A -> eight force-N commands``.
Unity reports the force it actually supplies to ``Rigidbody.AddForceAtPosition``
after the thruster delay and first-order response.  Thus this does not infer
force from vehicle acceleration, where buoyancy and hydrodynamics would hide a
thrust-allocation error.

Run while the foreground Unity Editor is playing and the gRPC adapter uses
``topic_prefix:=/sim``:

  cd ros2_ws
  ./scripts/run_ros2_uv.sh python ../tools/measure_sim_fx_allocation.py
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from statistics import mean, pstdev

import numpy as np
import rclpy
import yaml
from geometry_msgs.msg import TwistWithCovarianceStamped
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray

from motion_control.wrench_action_sender import (
    _load_wrench_profile,
    wrench_action_to_thruster_command,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = REPO_ROOT / "ros2_ws/src/motion_control/config/FinsROV/ppo_wrench_for_pose_sim.yaml"
DEFAULT_GEOMETRY = REPO_ROOT / "tools/finsrov_wrench_geometry.yaml"
DEFAULT_THRUSTER_TOPIC = "/sim/finsrov/thrusters_out"
DEFAULT_APPLIED_WRENCH_TOPIC = "/sim/finsrov/debug/thruster_applied_wrench"


def _load_forward_matrix(path: Path) -> np.ndarray:
    """Return B for tau=[Fx,Fy,Fz,Mx,My,Mz] = B @ force_N."""
    with path.open("r", encoding="utf-8") as stream:
        geometry = yaml.safe_load(stream)
    origin = np.asarray(geometry["reference_point"], dtype=float)
    columns: list[np.ndarray] = []
    for thruster in geometry["thrusters"]:
        position = np.asarray(thruster["position"], dtype=float) - origin
        direction = np.asarray(thruster["direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        columns.append(np.concatenate((direction, np.cross(position, direction))))
    return np.column_stack(columns)


class WrenchRecorder(Node):
    def __init__(self, *, thruster_topic: str, applied_wrench_topic: str) -> None:
        super().__init__("measure_sim_fx_allocation")
        self.samples: list[dict[str, float]] = []
        self.publisher = self.create_publisher(Float32MultiArray, thruster_topic, 10)
        self.create_subscription(TwistWithCovarianceStamped, applied_wrench_topic, self._on_wrench, 100)

    def _on_wrench(self, msg: TwistWithCovarianceStamped) -> None:
        wrench = msg.twist.twist
        self.samples.append(
            {
                "t": time.monotonic(),
                "fx": float(wrench.linear.x),
                "fy": float(wrench.linear.y),
                "fz": float(wrench.linear.z),
                "mx": float(wrench.angular.x),
                "my": float(wrench.angular.y),
                "mz": float(wrench.angular.z),
            }
        )

    def publish(self, force_n: np.ndarray) -> None:
        self.publisher.publish(Float32MultiArray(data=[float(item) for item in force_n]))


def _spin_for(node: Node, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.02)


def _wait_for_samples(node: WrenchRecorder, seconds: float) -> None:
    deadline = time.monotonic() + seconds
    while not node.samples and time.monotonic() < deadline:
        rclpy.spin_once(node, timeout_sec=0.05)
    if not node.samples:
        raise RuntimeError(
            "did not receive applied-wrench telemetry on the expected topic; "
            "ensure Unity has reloaded the FinsROV_Fossen prefab and is playing"
        )


def _send_hold(node: WrenchRecorder, force_n: np.ndarray, hold_sec: float, rate_hz: float) -> tuple[float, float]:
    start = time.monotonic()
    next_tick = start
    period = 1.0 / rate_hz
    while time.monotonic() - start < hold_sec:
        now = time.monotonic()
        if now >= next_tick:
            node.publish(force_n)
            next_tick += period
        rclpy.spin_once(node, timeout_sec=0.01)
    return start, time.monotonic()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--target-fx",
        type=float,
        nargs="+",
        default=[-5.0, -3.0, -1.0, 1.0, 3.0, 5.0],
        help="Body Fx targets (N) predicted with B before Unity execution.",
    )
    parser.add_argument("--hold-sec", type=float, default=2.0)
    parser.add_argument("--settle-sec", type=float, default=0.8)
    parser.add_argument("--steady-window-sec", type=float, default=0.7)
    parser.add_argument("--rate-hz", type=float, default=40.0)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--geometry", type=Path, default=DEFAULT_GEOMETRY)
    parser.add_argument("--thruster-topic", default=DEFAULT_THRUSTER_TOPIC)
    parser.add_argument("--applied-wrench-topic", default=DEFAULT_APPLIED_WRENCH_TOPIC)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts/runs/diagnostics/sim_fx_allocation",
    )
    args = parser.parse_args()
    if args.hold_sec <= 0 or args.settle_sec < 0 or args.steady_window_sec <= 0 or args.rate_hz <= 0:
        parser.error("hold, steady window, and rate must be positive; settle must be non-negative")
    if args.steady_window_sec > args.hold_sec:
        parser.error("steady window cannot exceed hold duration")
    if not args.config.is_file() or not args.geometry.is_file():
        parser.error("config and geometry must exist")
    return args


def main() -> int:
    args = _parse_args()
    profile = _load_wrench_profile(str(args.config))
    B = _load_forward_matrix(args.geometry)

    # The A-column is linear before clipping.  Determine its actual force-N
    # response using the same sender code that will publish the command.
    _, _, unit_force = wrench_action_to_thruster_command(
        (1, 0, 0, 0, 0, 0),
        action_gains=profile["action_gains"],
        force_limit_positive=profile["force_positive"],
        force_limit_negative=profile["force_negative"],
        output_scale=profile["output_scale"],
    )
    model_unit_wrench = B @ unit_force
    fx_per_action = float(model_unit_wrench[0])
    if abs(fx_per_action) < 1e-6:
        raise RuntimeError("the current surge allocation has no modelled Fx authority")

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = args.output_dir / timestamp
    output_dir.mkdir(parents=True, exist_ok=False)

    rclpy.init(args=None)
    node = WrenchRecorder(
        thruster_topic=args.thruster_topic,
        applied_wrench_topic=args.applied_wrench_topic,
    )
    rows: list[dict[str, object]] = []
    zero = np.zeros(8, dtype=np.float32)
    try:
        _wait_for_samples(node, 5.0)
        print(f"telemetry ready: {args.applied_wrench_topic}", flush=True)
        print(f"current +surge model B@f = {model_unit_wrench.tolist()}", flush=True)

        for target_fx in args.target_fx:
            action = float(target_fx) / fx_per_action
            if abs(action) > 1.0:
                raise RuntimeError(
                    f"target Fx={target_fx:g} N requires action={action:.4f}, outside [-1, 1]"
                )
            _, normalized, command = wrench_action_to_thruster_command(
                (action, 0, 0, 0, 0, 0),
                action_gains=profile["action_gains"],
                force_limit_positive=profile["force_positive"],
                force_limit_negative=profile["force_negative"],
                output_scale=profile["output_scale"],
            )
            predicted = B @ command
            first_sample = len(node.samples)
            active_start, active_end = _send_hold(node, command, args.hold_sec, args.rate_hz)
            steady_start = active_end - args.steady_window_sec
            steady = [sample for sample in node.samples[first_sample:] if steady_start <= sample["t"] <= active_end]
            if not steady:
                raise RuntimeError(f"Fx={target_fx:g}: no steady applied-wrench samples")
            applied_mean = np.asarray([mean(sample[key] for sample in steady) for key in ("fx", "fy", "fz", "mx", "my", "mz")])
            applied_std = np.asarray([pstdev(sample[key] for sample in steady) for key in ("fx", "fy", "fz", "mx", "my", "mz")])
            row = {
                "target_fx_n": float(target_fx),
                "policy_surge_action": action,
                "normalized_thrusters": normalized.tolist(),
                "command_force_n": command.tolist(),
                "model_wrench": predicted.tolist(),
                "applied_wrench_mean": applied_mean.tolist(),
                "applied_wrench_std": applied_std.tolist(),
                "fx_error_n": float(applied_mean[0] - target_fx),
                "fx_relative_error": float((applied_mean[0] - target_fx) / target_fx) if target_fx else 0.0,
                "steady_sample_count": len(steady),
            }
            rows.append(row)
            print(
                f"Fx target={target_fx:+.2f} N, model={predicted[0]:+.3f} N, "
                f"applied={applied_mean[0]:+.3f}+/-{applied_std[0]:.3f} N, "
                f"residual [Fy,Fz,My,Mz]=[{applied_mean[1]:+.3f}, {applied_mean[2]:+.3f}, "
                f"{applied_mean[4]:+.3f}, {applied_mean[5]:+.3f}]",
                flush=True,
            )
            _send_hold(node, zero, args.settle_sec, args.rate_hz)
    finally:
        for _ in range(5):
            node.publish(zero)
            _spin_for(node, 0.03)
        node.destroy_node()
        rclpy.shutdown()

    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("target_fx_n", "action", "model_fx_n", "applied_fx_mean_n", "applied_fx_std_n", "fx_error_n", "samples"))
        for row in rows:
            writer.writerow((
                row["target_fx_n"], row["policy_surge_action"], row["model_wrench"][0],
                row["applied_wrench_mean"][0], row["applied_wrench_std"][0], row["fx_error_n"],
                row["steady_sample_count"],
            ))
    print(f"results: {output_dir}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        raise SystemExit(130)
