import os
from glob import glob

from setuptools import find_packages, setup


package_name = "motion_control"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "docs"), glob(os.path.join("docs", "*.md"))),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "config", "FinsROV"), glob(os.path.join("config", "FinsROV", "*.yaml"))),
    ],
    install_requires=[
        "setuptools>=68",
        "numpy>=1.24,<2",
    ],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="ROS2 control node for sending motion goals to FinsSim vehicles with pluggable controller backends.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "controller_state_adapter = motion_control.controller_state_adapter:main",
            "motion_controller = motion_control.controller_node:main",
            "sim_truth_state_status = motion_control.sim_truth_state_status:main",
            "reset_service = motion_control.reset_service:main",
            "send_position_goal = motion_control.goal_sender:main",
            "send_position_yaw_goal = motion_control.goal_sender:main",
            "send_velocity_command = motion_control.goal_sender:velocity_main",
            "cancel_goal = motion_control.goal_sender:cancel_main",
            "cancel_position_goal = motion_control.goal_sender:cancel_main",
            "reset_vehicle = motion_control.goal_sender:reset_main",
            "set_control_mode = motion_control.goal_sender:control_mode_main",
            "record_real_sim_diagnostics = motion_control.real_sim_diagnostics_recorder:main",
            "replay_unity_hydrodynamics = motion_control.unity_hydrodynamic_replay:main",
            "analyze_real_sim_diagnostics = motion_control.analyze_real_sim_diagnostics:main",
            "send_wrench_action = motion_control.wrench_action_sender:main",
            "send_trajectory = motion_control.trajectory_sender:main",
            "send_t2_trajectory_goal = motion_control.trajectory_sender:t2_main",
        ],
    },
)
