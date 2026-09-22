import os
from glob import glob

from setuptools import find_packages, setup


package_name = "teleop"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
        (os.path.join("share", package_name, "docs"), glob(os.path.join("docs", "*.md"))),
    ],
    install_requires=["setuptools>=68", "numpy>=1.24", "pygame>=2.5"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="Joystick teleoperation node for FinsROV ROS2 control.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "joystick_teleop = teleop.joystick_node:main",
        ],
    },
)
