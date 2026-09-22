import os
from glob import glob

from setuptools import find_packages, setup


package_name = "hardware_bridge"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml", "README.md"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools>=68", "numpy>=1.24"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="Full-duplex hardware bridge for FinsROV thruster commands and MCU telemetry.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "hardware_bridge = hardware_bridge.hardware_bridge_node:main",
            "send_thruster_test = hardware_bridge.thruster_test_sender:main",
            "send_thruster_force_test = hardware_bridge.thruster_force_test_sender:main",
            "send_direct_thrusters = hardware_bridge.direct_thruster_sender:main",
        ],
    },
)
