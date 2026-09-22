"""Offline analysis and plotting for real/sim controller diagnostic recordings.

The recorder stores one CSV per ROS topic.  This module intentionally operates
only on those files and never creates ROS publishers or subscribers.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .backend_loader import load_backend


LOGGER = logging.getLogger(__name__)
ACTION_NAMES = ("surge", "sway", "heave", "roll", "pitch", "yaw")
CONTROLLER_ORIENTATION_NAMES = ("roll_x", "pitch_z", "yaw_y")
THRUSTER_NAMES = ("V_LF", "V_LB", "V_RB", "V_RF", "H_LF", "H_LB", "H_RB", "H_RF")
OBS_NAMES = (
    "target_offset_body_x_norm",
    "target_offset_body_y_norm",
    "target_offset_body_z_norm",
    "rot6d_00",
    "rot6d_10",
    "rot6d_20",
    "rot6d_01",
    "rot6d_11",
    "rot6d_21",
    "linear_velocity_body_x",
    "linear_velocity_body_y",
    "linear_velocity_body_z",
    "angular_velocity_body_x",
    "angular_velocity_body_y",
    "angular_velocity_body_z",
    "distance_norm",
)


def _read_rows(path: Path) -> tuple[np.ndarray, list[dict[str, Any]]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    times = np.asarray([float(row["wall_time_sec"]) for row in rows], dtype=float)
    payloads = [json.loads(row["payload_json"]) for row in rows]
    return times, payloads


def _find_csv(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    if not matches:
        LOGGER.warning("missing diagnostic CSV: %s/%s", directory, pattern)
        return None
    if len(matches) > 1:
        LOGGER.warning("multiple CSVs match %s; using %s", pattern, matches[0])
    return matches[0]


def _vector_topic(directory: Path, pattern: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = _find_csv(directory, pattern)
    if path is None:
        return None
    times, payloads = _read_rows(path)
    values = []
    for payload in payloads:
        data = payload.get("data", payload) if isinstance(payload, dict) else payload
        if isinstance(data, str):
            data = json.loads(data)
        values.append(np.asarray(data, dtype=float).reshape(-1))
    if not values:
        return None
    return times, np.vstack(values)


def _string_topic(directory: Path, pattern: str) -> tuple[np.ndarray, list[dict[str, Any]]] | None:
    path = _find_csv(directory, pattern)
    if path is None:
        return None
    times, payloads = _read_rows(path)
    decoded = []
    for payload in payloads:
        value = payload.get("data", payload) if isinstance(payload, dict) else payload
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except json.JSONDecodeError:
                value = {"data": value}
        decoded.append(value if isinstance(value, dict) else {})
    return times, decoded


def _pose_topic(directory: Path, pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = _find_csv(directory, pattern)
    if path is None:
        return None
    times, payloads = _read_rows(path)
    position = []
    quaternion = []
    for payload in payloads:
        position.append([payload["position"][axis] for axis in ("x", "y", "z")])
        quaternion.append([payload["orientation"][axis] for axis in ("x", "y", "z", "w")])
    return times, np.asarray(position, dtype=float), _quaternion_to_controller_rpy_deg(
        np.asarray(quaternion, dtype=float)
    )


def _imu_topic(directory: Path, pattern: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    path = _find_csv(directory, pattern)
    if path is None:
        return None
    times, payloads = _read_rows(path)
    angular = []
    acceleration = []
    for payload in payloads:
        angular.append([payload["angular_velocity"][axis] for axis in ("x", "y", "z")])
        acceleration.append([payload["linear_acceleration"][axis] for axis in ("x", "y", "z")])
    return times, np.asarray(angular, dtype=float), np.asarray(acceleration, dtype=float)


def _dvl_topic(directory: Path, pattern: str) -> tuple[np.ndarray, np.ndarray] | None:
    path = _find_csv(directory, pattern)
    if path is None:
        return None
    times, payloads = _read_rows(path)
    velocity = []
    for payload in payloads:
        velocity.append([payload["linear"][axis] for axis in ("x", "y", "z")])
    return times, np.asarray(velocity, dtype=float)


def _quaternion_to_controller_rpy_deg(quaternion: np.ndarray) -> np.ndarray:
    """Extract controller-frame ``[roll_x, pitch_z, yaw_y]`` in degrees.

    Controller orientation uses ``R = Ry(yaw_y) @ Rz(pitch_z) @ Rx(roll_x)``
    with ``x=forward, y=up, z=left``.  Applying the conventional ROS ZYX
    extraction here would swap the physical meaning of pitch and yaw and can
    create artificial +/-180 degree jumps in the diagnostic plots.
    """
    values = np.asarray(quaternion, dtype=np.float64).reshape(-1, 4)
    norms = np.linalg.norm(values, axis=1)
    if np.any(norms <= np.finfo(np.float64).eps):
        raise ValueError("cannot extract controller RPY from a zero quaternion")
    x, y, z, w = (values / norms[:, None]).T

    # Rotation-matrix entries for R = Ry(yaw_y) @ Rz(pitch_z) @ Rx(roll_x).
    r00 = 1.0 - 2.0 * (y * y + z * z)
    r10 = 2.0 * (x * y + w * z)
    r20 = 2.0 * (x * z - w * y)
    r11 = 1.0 - 2.0 * (x * x + z * z)
    r12 = 2.0 * (y * z - w * x)

    pitch_z = np.arcsin(np.clip(r10, -1.0, 1.0))
    yaw_y = np.arctan2(-r20, r00)
    roll_x = np.arctan2(-r12, r11)
    return np.degrees(np.column_stack((roll_x, pitch_z, yaw_y)))


def _relative_time(times: np.ndarray) -> np.ndarray:
    return times - times[0] if len(times) else times


def _stats(values: np.ndarray | None) -> dict[str, list[float]] | None:
    if values is None or values.size == 0:
        return None
    return {
        "min": np.nanmin(values, axis=0).tolist(),
        "max": np.nanmax(values, axis=0).tolist(),
        "mean_abs": np.nanmean(np.abs(values), axis=0).tolist(),
    }


def _reconstruct_raw_actions(
    observations: tuple[np.ndarray, np.ndarray] | None,
    checkpoint: str | None,
) -> np.ndarray | None:
    if observations is None:
        return None
    if not checkpoint:
        LOGGER.warning("checkpoint not supplied; raw action reconstruction skipped")
        return None
    backend = load_backend("ppo_wrench_for_pose_empirical_thruster_mixer", checkpoint_path=checkpoint, device="cpu")
    raw = []
    for observation in observations[1]:
        action = np.asarray(backend.predict_policy_action(observation), dtype=float).reshape(-1)
        raw.append(np.clip(action[:6], -1.0, 1.0))
    return np.vstack(raw)


def _write_csv(path: Path, header: Iterable[str], rows: Iterable[Iterable[Any]]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(list(header))
        writer.writerows(rows)


def _plot_meta_actions(path: Path, action_data: dict[str, Any], title: str) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes = axes.ravel()
    for index, name in enumerate(ACTION_NAMES):
        axis = axes[index]
        if action_data.get("raw") is not None:
            axis.plot(action_data["time_raw"], action_data["raw"][:, index], label="raw policy", lw=1.2)
        axis.plot(action_data["time_actual"], action_data["actual"][:, index], label="actual after limiter", lw=1.0)
        axis.axhline(0.0, color="black", lw=0.4)
        axis.set_ylabel(name)
        axis.set_ylim(-1.08, 1.08)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    axes[-2].set_xlabel("time since first action [s]")
    axes[-1].set_xlabel("time since first action [s]")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_thrusters_rpm(path: Path, thruster_data: dict[str, Any], title: str) -> None:
    fig, axes = plt.subplots(8, 1, figsize=(14, 18), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(thruster_data["time_command"], thruster_data["command"][:, index], label="thrusters_out / force command", lw=1.0)
        if thruster_data.get("rpm") is not None:
            rpm_axis = axis.twinx()
            rpm_axis.plot(thruster_data["time_rpm"], thruster_data["rpm"][:, index], color="tab:red", alpha=0.55, lw=0.75, label="rpm")
            rpm_axis.set_ylabel("RPM", color="tab:red")
            if index == 0:
                rpm_axis.legend(loc="upper right")
        axis.set_ylabel(f"{THRUSTER_NAMES[index]}\ncmd")
        axis.grid(alpha=0.2)
        if index == 0:
            axis.legend(loc="upper left")
    axes[-1].set_xlabel("time since first command [s]")
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.99))
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_pose(path: Path, pose_data: tuple[np.ndarray, np.ndarray, np.ndarray], title: str) -> None:
    times, position, rpy = pose_data
    times = _relative_time(times)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for index, name in enumerate(("x", "y", "z")):
        axes[0].plot(times, position[:, index], label=name)
    axes[0].axhline(0.0, color="black", lw=0.5)
    axes[0].axhline(-0.5, color="tab:orange", ls="--", lw=0.8, label="target y=-0.5")
    axes[0].set_ylabel("position [m]")
    axes[0].legend(ncol=4)
    axes[0].grid(alpha=0.25)
    for index, name in enumerate(CONTROLLER_ORIENTATION_NAMES):
        axes[1].plot(times, rpy[:, index], label=name)
    axes[1].axhline(90.0, color="tab:orange", ls="--", lw=0.8, label="target yaw_y=90")
    axes[1].set_ylabel("controller RPY [deg]")
    axes[1].set_xlabel("time since first pose [s]")
    axes[1].legend(ncol=4)
    axes[1].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _plot_velocity(path: Path, imu_data: tuple[np.ndarray, np.ndarray, np.ndarray] | None, dvl_data: tuple[np.ndarray, np.ndarray] | None, title: str) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    if dvl_data is not None:
        dvl_time, velocity = dvl_data
        dvl_time = _relative_time(dvl_time)
        for index, name in enumerate(("u/x", "v/y", "w/z")):
            axes[0].plot(dvl_time, velocity[:, index], label=name)
        if len(dvl_time) > 2:
            derivative = np.gradient(velocity, dvl_data[0], axis=0)
            for index, name in enumerate(("x", "y", "z")):
                axes[2].plot(dvl_time, derivative[:, index], label=f"d(DVL {name})/dt")
        axes[0].legend(ncol=3)
    axes[0].set_ylabel("DVL body velocity [m/s]")
    axes[0].grid(alpha=0.25)
    if imu_data is not None:
        imu_time, angular, acceleration = imu_data
        imu_time = _relative_time(imu_time)
        for index, name in enumerate(("p/x", "q/y", "r/z")):
            axes[1].plot(imu_time, angular[:, index], label=name)
        for index, name in enumerate(("x", "y", "z")):
            axes[2].plot(imu_time, acceleration[:, index], ls="--", alpha=0.6, label=f"IMU accel {name}")
        axes[1].legend(ncol=3)
    axes[1].set_ylabel("IMU angular rate [rad/s]")
    axes[1].grid(alpha=0.25)
    axes[2].set_ylabel("acceleration [m/s2]")
    axes[2].set_xlabel("time since first sample [s]")
    axes[2].legend(ncol=3, fontsize=8)
    axes[2].grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _plot_state_observation(path: Path, state_data: dict[str, Any], title: str) -> None:
    fig, axes = plt.subplots(4, 1, figsize=(14, 14), sharex=True)
    if state_data.get("policy") is not None:
        time, policy = state_data["policy"]
        time = _relative_time(time)
        axes[0].plot(time, policy["position_error_m"], label="position error norm")
        for index, name in enumerate(("x", "y", "z")):
            axes[1].plot(time, policy["error_body"][:, index], label=f"error body {name}")
            axes[2].plot(time, policy["linear_velocity_body"][:, index], label=f"linear v {name}")
        axes[0].legend()
        axes[1].legend(ncol=3)
        axes[2].legend(ncol=3)
    if state_data.get("observation") is not None:
        time, observation = state_data["observation"]
        time = _relative_time(time)
        for index, name in enumerate(OBS_NAMES[: observation.shape[1]]):
            axes[3].plot(time, observation[:, index], label=name)
        axes[3].legend(ncol=4, fontsize=7)
    axes[0].set_ylabel("error [m]")
    axes[1].set_ylabel("body error [m]")
    axes[2].set_ylabel("body velocity [m/s]")
    axes[3].set_ylabel("observation")
    axes[3].set_xlabel("time since first sample [s]")
    for axis in axes:
        axis.grid(alpha=0.25)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _load_dataset(directory: Path, prefix: str, checkpoint: str | None) -> dict[str, Any]:
    observations = _vector_topic(directory, f"{prefix}_debug_observation__*.csv")
    actions = _vector_topic(directory, f"{prefix}_debug_action__*.csv")
    raw = _reconstruct_raw_actions(observations, checkpoint)
    action_data = None
    if actions is not None:
        sample_count = min(
            len(actions[1]),
            len(raw) if raw is not None else len(actions[1]),
        )
        action_data = {
            "time_actual": _relative_time(actions[0][:sample_count]),
            "actual": actions[1][:sample_count, :6],
            "time_raw": _relative_time(observations[0][:sample_count]) if observations is not None else _relative_time(actions[0][:sample_count]),
            "raw": raw[:sample_count] if raw is not None else None,
        }

    command = _vector_topic(directory, f"{prefix}_thruster_command__*.csv")
    if command is None:
        command = _vector_topic(directory, f"{prefix}_debug_thruster_command__*.csv")
    rpm = _vector_topic(directory, f"{prefix}_hardware_rpm__*.csv")
    thruster_data = None
    if command is not None:
        thruster_data = {
            "time_command": _relative_time(command[0]),
            "command": command[1][:, :8],
            "time_rpm": _relative_time(rpm[0]) if rpm is not None else None,
            "rpm": rpm[1][:, :8] if rpm is not None else None,
        }

    pose = _pose_topic(directory, f"{prefix}_controller_pose__*.csv")
    imu = _imu_topic(directory, f"{prefix}_controller_imu__*.csv")
    dvl = _dvl_topic(directory, f"{prefix}_controller_dvl__*.csv")
    policy_rows = _string_topic(directory, f"{prefix}_debug_policy_info__*.csv")
    policy_data = None
    if policy_rows is not None:
        times, payloads = policy_rows
        def matrix(key: str) -> np.ndarray:
            return np.asarray([payload.get(key, [math.nan] * 3) for payload in payloads], dtype=float)
        policy_data = {
            "position_error_m": np.asarray([payload.get("position_error_m", math.nan) for payload in payloads], dtype=float),
            "error_body": matrix("error_body"),
            "linear_velocity_body": matrix("linear_velocity_body"),
        }

    return {
        "directory": directory,
        "observations": observations,
        "actions": actions,
        "action_data": action_data,
        "raw": raw,
        "wrench": _vector_topic(directory, f"{prefix}_debug_wrench6d__*.csv"),
        "thruster_data": thruster_data,
        "pose": pose,
        "imu": imu,
        "dvl": dvl,
        "policy": (policy_rows[0], policy_data) if policy_rows is not None else None,
    }


def _write_dataset_outputs(dataset: dict[str, Any], output_dir: Path, label: str) -> dict[str, Any]:
    action_data = dataset.get("action_data")
    if action_data is not None:
        rows = []
        for index in range(len(action_data["actual"])):
            raw = action_data["raw"][index] if action_data.get("raw") is not None else [math.nan] * 6
            rows.append([action_data["time_actual"][index], *raw, *action_data["actual"][index]])
        _write_csv(
            output_dir / f"{label}_raw_and_actual_meta_action_timeseries.csv",
            ["time_since_first_action_sec", *[f"raw_{name}" for name in ACTION_NAMES], *[f"actual_{name}" for name in ACTION_NAMES]],
            rows,
        )
        _plot_meta_actions(output_dir / f"{label}_raw_meta_action_vs_actual.png", action_data, f"{label}: raw and actual 6D meta action")

    thruster_data = dataset.get("thruster_data")
    if thruster_data is not None:
        _plot_thrusters_rpm(output_dir / f"{label}_thrusters_out_and_rpm.png", thruster_data, f"{label}: thrusters_out and measured RPM")

    pose = dataset.get("pose")
    if pose is not None:
        _plot_pose(output_dir / f"{label}_pose_rotation_vs_time.png", pose, f"{label}: pose and orientation")
        time, position, rpy = pose
        _write_csv(
            output_dir / f"{label}_pose_state_timeseries.csv",
            [
                "time_since_first_pose_sec",
                "pos_x",
                "pos_y",
                "pos_z",
                "roll_x_deg",
                "pitch_z_deg",
                "yaw_y_deg",
            ],
            ([time_value - time[0], *pos, *angles] for time_value, pos, angles in zip(time, position, rpy)),
        )

    imu, dvl = dataset.get("imu"), dataset.get("dvl")
    if imu is not None or dvl is not None:
        _plot_velocity(output_dir / f"{label}_velocity_angular_rate_acceleration.png", imu, dvl, f"{label}: velocity, angular rate, and acceleration")

    observations = dataset.get("observations")
    if observations is not None:
        _write_csv(
            output_dir / f"{label}_observation_pose16_rot6d_timeseries.csv",
            ["time_since_first_observation_sec", *OBS_NAMES[: observations[1].shape[1]]],
            ([time_value - observations[0][0], *values] for time_value, values in zip(*observations)),
        )

    state_data = {"observation": (observations[0], observations[1]) if observations is not None else None}
    if dataset.get("policy") is not None:
        state_data["policy"] = dataset["policy"]
        policy_time, policy = dataset["policy"]
        _write_csv(
            output_dir / f"{label}_policy_state_timeseries.csv",
            [
                "time_since_first_policy_sample_sec",
                "position_error_m",
                "error_body_x",
                "error_body_y",
                "error_body_z",
                "linear_velocity_body_x",
                "linear_velocity_body_y",
                "linear_velocity_body_z",
            ],
            (
                [time_value - policy_time[0], position_error, *error_body, *linear_velocity]
                for time_value, position_error, error_body, linear_velocity in zip(
                    policy_time,
                    policy["position_error_m"],
                    policy["error_body"],
                    policy["linear_velocity_body"],
                )
            ),
        )
    _plot_state_observation(output_dir / f"{label}_observation_and_controller_state.png", state_data, f"{label}: controller state and observation")

    return {
        "label": label,
        "directory": str(dataset["directory"]),
        "action_samples": int(len(action_data["actual"])) if action_data is not None else 0,
        "observation_samples": int(len(observations[1])) if observations is not None else 0,
        "raw_action_stats": _stats(action_data["raw"]) if action_data is not None and action_data.get("raw") is not None else None,
        "actual_action_stats": _stats(action_data["actual"]) if action_data is not None else None,
        "wrench_stats": _stats(dataset["wrench"][1]) if dataset.get("wrench") is not None else None,
        "thruster_stats": _stats(thruster_data["command"]) if thruster_data is not None else None,
        "rpm_stats": _stats(thruster_data["rpm"]) if thruster_data is not None and thruster_data.get("rpm") is not None else None,
    }


def _plot_comparison(output_dir: Path, real: dict[str, Any], sim: dict[str, Any]) -> None:
    real_action, sim_action = real.get("action_data"), sim.get("action_data")
    if real_action is not None and sim_action is not None:
        fig, axes = plt.subplots(3, 2, figsize=(15, 10))
        for index, (axis, name) in enumerate(zip(axes.ravel(), ACTION_NAMES)):
            if real_action.get("raw") is not None:
                axis.plot(
                    real_action["time_raw"],
                    real_action["raw"][:, index],
                    label="real raw",
                    color="tab:blue",
                    lw=0.9,
                )
            axis.plot(real_action["time_actual"], real_action["actual"][:, index], label="real actual", color="tab:orange", lw=0.9)
            if sim_action.get("raw") is not None:
                axis.plot(sim_action["time_raw"], sim_action["raw"][:, index], label="sim raw", color="tab:green", lw=0.9)
            axis.plot(sim_action["time_actual"], sim_action["actual"][:, index], label="sim actual", color="tab:red", lw=0.9)
            axis.axhline(0.0, color="black", lw=0.4)
            axis.set_title(name)
            axis.set_ylim(-1.08, 1.08)
            axis.grid(alpha=0.25)
        axes.ravel()[0].legend(ncol=2, fontsize=8)
        fig.suptitle("Real vs Unity: same goal 6D meta actions")
        fig.tight_layout()
        fig.savefig(output_dir / "real_vs_sim_meta_action.png", dpi=160)
        plt.close(fig)

    real_pose, sim_pose = real.get("pose"), sim.get("pose")
    if real_pose is not None and sim_pose is not None:
        fig, axes = plt.subplots(2, 1, figsize=(15, 9))
        for index, name in enumerate(("x", "y", "z")):
            axes[0].plot(_relative_time(real_pose[0]), real_pose[1][:, index], label=f"real {name}")
            axes[0].plot(_relative_time(sim_pose[0]), sim_pose[1][:, index], ls="--", label=f"sim {name}")
        axes[0].axhline(-0.5, color="tab:orange", ls=":", label="target y=-0.5")
        axes[0].set_ylabel("position [m]")
        axes[0].legend(ncol=4, fontsize=8)
        axes[0].grid(alpha=0.25)
        for index, name in enumerate(CONTROLLER_ORIENTATION_NAMES):
            axes[1].plot(_relative_time(real_pose[0]), real_pose[2][:, index], label=f"real {name}")
            axes[1].plot(_relative_time(sim_pose[0]), sim_pose[2][:, index], ls="--", label=f"sim {name}")
        axes[1].axhline(90.0, color="tab:orange", ls=":", label="target yaw_y=90")
        axes[1].set_ylabel("controller RPY [deg]")
        axes[1].set_xlabel("time since first recorded pose [s]")
        axes[1].legend(ncol=4, fontsize=8)
        axes[1].grid(alpha=0.25)
        fig.suptitle("Real vs Unity: controller pose for the same target experiment")
        fig.tight_layout()
        fig.savefig(output_dir / "real_vs_sim_pose_rotation.png", dpi=160)
        plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-dir", required=True, type=Path, help="Recorder directory containing real_* CSV files")
    parser.add_argument("--sim-dir", type=Path, help="Optional recorder directory containing sim_* CSV files")
    parser.add_argument("--output-dir", type=Path, help="Output directory; defaults to --real-dir")
    parser.add_argument(
        "--checkpoint",
        default="./artifacts/runs/rl/wrench/ppo_wrench_for_pose_v2_0809/checkpoints/best_model.zip",
        help="6D wrench PPO checkpoint used to reconstruct deterministic raw policy actions",
    )
    parser.add_argument("--no-raw-action", action="store_true", help="Do not load the checkpoint or reconstruct raw actions")
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logging.basicConfig(level=getattr(logging, args.log_level), format="[%(levelname)s] %(message)s")
    output_dir = args.output_dir or args.real_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = None if args.no_raw_action else args.checkpoint

    real = _load_dataset(args.real_dir, "real", checkpoint)
    real_summary = _write_dataset_outputs(real, output_dir, "real")
    summary: dict[str, Any] = {"real": real_summary, "checkpoint": checkpoint}

    if args.sim_dir is not None:
        sim = _load_dataset(args.sim_dir, "sim", checkpoint)
        sim_summary = _write_dataset_outputs(sim, output_dir, "sim")
        _plot_comparison(output_dir, real, sim)
        summary["sim"] = sim_summary

    (output_dir / "real_sim_analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    LOGGER.info("analysis outputs written to %s", output_dir)


if __name__ == "__main__":
    main()
