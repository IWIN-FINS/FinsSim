from glob import glob

from setuptools import find_packages, setup


package_name = "apriltag_dataset_collector"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/config", glob("config/*.yaml")),
        (f"share/{package_name}/docs", glob("docs/*.md")),
    ],
    install_requires=["setuptools>=68"],
    zip_safe=True,
    maintainer="FinsSim contributors",
    maintainer_email="support@openai.com",
    description="Standalone AprilTag dataset collection GUI and CLI for FinsROV.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "apriltag_dataset_collector_gui = apriltag_dataset_collector.gui:main",
            "apriltag_dataset_capture = apriltag_dataset_collector.cli:main",
        ],
    },
)
