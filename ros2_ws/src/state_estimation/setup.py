import os
from glob import glob

from setuptools import find_packages, setup


package_name = "state_estimation"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "docs"), glob(os.path.join("docs", "*.md"))),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools>=68", "numpy>=1.24"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="State fusion node for FinsROV real-vehicle ROS2 sensor topics.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "state_fusion = state_estimation.state_fusion_node:main",
            "calibrate_world_camera = state_estimation.calibration_tools.calibrate_world_camera:main",
            "evaluate_pnp_truth_dataset = state_estimation.calibration_tools.evaluate_pnp_truth_dataset:main",
            "fish_truth_left_camera = state_estimation.fish_truth_left_camera_node:main",
        ],
    },
)
