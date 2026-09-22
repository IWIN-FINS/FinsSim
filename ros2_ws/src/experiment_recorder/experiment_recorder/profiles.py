"""Versioned topic profiles used by the experiment recorder.

The recorder intentionally uses ``ros2 bag record`` instead of re-serializing
messages into custom CSV files.  This keeps the original message type, header
stamp, QoS metadata, and rosbag metadata available for later analysis.
"""

from __future__ import annotations

from collections.abc import Iterable


REAL_COMMON_TOPICS = (
    "/finsrov/experiment/event",
    "/finsrov/pose",
    # Present only for localization-noise ablations.  It is a parallel clean
    # controller-world conversion of raw fusion and never feeds control.
    "/finsrov/evaluation/pose",
    # Optional perturbed inputs.  These remain distinct from the raw-fusion
    # streams so a bag can audit what the policy saw without contaminating
    # the clean evaluation trajectory.
    "/finsrov/pose_noisy",
    "/finsrov/imu_link_noisy",
    # Full no-EKF state input branch.  These messages are distinct from both
    # the production EKF state and controller-world outputs, so a recorded
    # ablation can prove what was actually supplied to the controller.
    "/finsrov/ablation/no_ekf/pose",
    "/finsrov/ablation/no_ekf/imu_link",
    "/finsrov/ablation/no_ekf/depth_link",
    "/finsrov/ablation/no_ekf/dvl_link",
    "/finsrov/ablation/no_ekf/status",
    "/finsrov/controller/pose",
    "/finsrov/controller/imu",
    "/finsrov/controller/depth",
    "/finsrov/controller/dvl",
    "/finsrov/controller/state/status",
    "/finsrov/state/status",
    "/finsrov/vision/status",
    "/finsrov/vision/tag_poses_3d_camera",
    "/finsrov/vision/refracted_pose_6d",
    "/finsrov/vision/refracted_pose_6d_pure",
    "/finsrov/hardware/telemetry",
    "/finsrov/hardware/imu_raw",
    "/finsrov/hardware/depth_raw",
    "/finsrov/hardware/motor_rpm_raw",
    "/finsrov/hardware/thruster_cmd_echo",
    "/finsrov/hardware/status",
    "/tf_static",
)

# E1/E2 are horizontal-position experiments.  The pose messages themselves
# are retained because they are the implemented fusion outputs, but raw depth,
# IMU, RPM, and hardware telemetry are deliberately not recorded in this
# profile; offline analysis uses only pool_world x/y.
APRILTAG_TOPICS = (
    "/finsrov/experiment/event",
    "/finsrov/pose",
    "/finsrov/state/status",
    "/finsrov/vision/status",
    "/finsrov/vision/tag_poses_3d_camera",
    "/finsrov/vision/refracted_pose_6d",
    "/finsrov/vision/refracted_pose_6d_pure",
    "/tf_static",
)

REAL_CONTROL_TOPICS = (
    "/finsrov/thrusters_out",
    "/motion_controller/control_mode",
    "/motion_controller/command/position_controller_world",
    "/motion_controller/command/position_controller_body",
    "/motion_controller/command/pose_controller_body",
    "/motion_controller/command/trajectory",
    "/motion_controller/command/cancel",
    "/motion_controller/status/active_pose",
    "/motion_controller/status/error_body",
    "/motion_controller/status/reached",
    "/motion_controller/debug/observation",
    "/motion_controller/debug/action",
    "/motion_controller/debug/policy_action",
    "/motion_controller/debug/wrench6d",
    "/motion_controller/debug/thruster_command",
    "/motion_controller/debug/policy_info",
    "/motion_controller/debug/trajectory_reference",
)

SIM_COMMON_TOPICS = (
    "/finsrov/experiment/event",
    "/sim/finsrov/controller/pose",
    "/sim/finsrov/controller/imu",
    "/sim/finsrov/controller/depth",
    "/sim/finsrov/controller/dvl",
    "/sim/finsrov/controller/state/status",
    "/sim/finsrov/controller/state/status_stamped",
    "/sim/finsrov/debug/thruster_applied_wrench",
)

SIM_CONTROL_TOPICS = (
    "/sim/finsrov/thrusters_out",
    "/sim/motion_controller/command/position_controller_world",
    "/sim/motion_controller/command/position_controller_body",
    "/sim/motion_controller/command/pose_controller_body",
    "/sim/motion_controller/command/trajectory",
    "/sim/motion_controller/command/cancel",
    "/sim/motion_controller/status/active_pose",
    "/sim/motion_controller/status/error_body",
    "/sim/motion_controller/status/reached",
    "/sim/motion_controller/debug/observation",
    "/sim/motion_controller/debug/action",
    "/sim/motion_controller/debug/wrench6d",
    "/sim/motion_controller/debug/thruster_command",
    # Header-stamped controller output is the timing authority for accelerated
    # Unity trials; Float32MultiArray diagnostics have no /clock stamp.
    "/sim/motion_controller/debug/thruster_command_stamped",
    # The UInt64 payload is the exact Unity physics tick acknowledged by ROS2.
    "/sim/motion_controller/debug/control_tick_complete",
    "/sim/motion_controller/debug/policy_info",
    "/sim/motion_controller/debug/trajectory_reference",
    "/sim/motion_controller/debug/trajectory_reference_stamped",
)


def _unique(*groups: Iterable[str]) -> tuple[str, ...]:
    result: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for topic in group:
            if topic not in seen:
                seen.add(topic)
                result.append(topic)
    return tuple(result)


PROFILES: dict[str, tuple[str, ...]] = {
    # E1/E2.  Raw image topics are deliberately excluded; save selected image
    # clips separately when an optical audit is required.  This profile is
    # intentionally horizontal-only; see APRILTAG_TOPICS above.
    "apriltag": _unique(APRILTAG_TOPICS),
    # E4 planar response and E6/E7 T1 hardware runs.
    "t1": _unique(
        REAL_COMMON_TOPICS,
        REAL_CONTROL_TOPICS,
    ),
    # E10 T2 hardware run.
    "t2": _unique(
        REAL_COMMON_TOPICS,
        REAL_CONTROL_TOPICS,
    ),
    # E5/E9 when the ROS2 simulation bridge is active.
    "t1_sim": _unique(SIM_COMMON_TOPICS, SIM_CONTROL_TOPICS),
    "t2_sim": _unique(SIM_COMMON_TOPICS, SIM_CONTROL_TOPICS),
    "full": _unique(REAL_COMMON_TOPICS, REAL_CONTROL_TOPICS, SIM_COMMON_TOPICS, SIM_CONTROL_TOPICS),
}


EXPERIMENTS: dict[str, dict[str, str]] = {
    "E1": {"name": "AprilTag fixed-pose accuracy", "profile": "apriltag"},
    "E2": {"name": "AprilTag Snell/PnP/fusion ablation", "profile": "apriltag"},
    "E4": {"name": "Surge/sway/depth/yaw response", "profile": "t1"},
    "E5": {"name": "T1 simulation PID/PPO evaluation", "profile": "t1_sim"},
    "E6": {"name": "T1 hardware PID", "profile": "t1"},
    "E7": {"name": "T1 hardware PPO", "profile": "t1"},
    "E8": {"name": "T1 centre-point manual disturbance recovery", "profile": "t1"},
    "E9": {"name": "T2 simulation trajectory tracking", "profile": "t2_sim"},
    "E10": {"name": "T2 hardware trajectory tracking", "profile": "t2"},
}


def topics_for_profile(profile: str) -> tuple[str, ...]:
    """Return a validated, deterministic topic tuple for a profile."""

    key = str(profile).strip().lower()
    try:
        return PROFILES[key]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILES))
        raise ValueError(f"unknown profile {profile!r}; choose one of: {choices}") from exc
