import os
from glob import glob

from setuptools import find_packages, setup


package_name = "thruster_curve_measurement"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.yaml"))),
        (os.path.join("share", package_name, "config"), glob(os.path.join("config", "*.json"))),
        (os.path.join("share", package_name, "docs"), glob(os.path.join("docs", "*.md"))),
        (os.path.join("share", package_name, "launch"), glob(os.path.join("launch", "*.launch.py"))),
    ],
    install_requires=["setuptools>=68", "pymodbus>=3.6,<4", "pyserial>=3.5,<4", "PyYAML>=6,<7"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="Automated FinsROV thruster curve measurement with Modbus force sensor logging.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "thruster_curve_sweep = thruster_curve_measurement.thruster_curve_sweep:main",
            "thruster_curve_measure = thruster_curve_measurement.thruster_curve_measure:main",
            "thruster_bench_measure = thruster_curve_measurement.thruster_bench_measure:main",
            "thruster_dynamic_identification = thruster_curve_measurement.thruster_dynamic_identification:main",
            "thruster_rpm_closed_loop_identification = thruster_curve_measurement.thruster_rpm_closed_loop_identification:main",
            "thruster_rpm_first_order_identification = thruster_curve_measurement.thruster_rpm_first_order_identification:main",
        ],
    },
)
