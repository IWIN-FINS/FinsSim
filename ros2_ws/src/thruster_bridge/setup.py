import os
from glob import glob

from setuptools import find_packages, setup


package_name = "thruster_bridge"


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
    install_requires=["setuptools>=68"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="Bridge normalized FinsROV thruster commands from ROS2 to the FineSUB V5Streamer protocol.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "thruster_streamer_bridge = thruster_bridge.thruster_streamer_bridge:main",
        ],
    },
)
