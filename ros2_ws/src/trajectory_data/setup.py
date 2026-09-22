from glob import glob

from setuptools import find_packages, setup


package_name = "trajectory_data"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/docs", glob("docs/*.md")),
    ],
    install_requires=["setuptools>=68", "numpy>=1.24", "matplotlib>=3.8", "pyarrow>=17"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="Passive rosbag recording, trajectory export, and quality analysis for FinsSim vehicle profiles.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "record_teleop_trajectory = trajectory_data.recorder:main",
            "mark_teleop_trajectory = trajectory_data.marker:main",
            "export_teleop_trajectory = trajectory_data.exporter:main",
            "analyze_teleop_trajectory = trajectory_data.analyzer:main",
        ],
    },
)
