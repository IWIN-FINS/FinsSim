from setuptools import find_packages, setup


package_name = "irl_data"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools>=68", "pyarrow>=17"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="GoalYaw AIRL/GAIL episode protocol and manifest tools for FinsROV teleoperation data.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "publish_goal_yaw_irl_event = irl_data.protocol:main",
            "build_goal_yaw_irl_manifest = irl_data.episode_manifest:main",
        ],
    },
)
