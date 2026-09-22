from glob import glob

from setuptools import find_packages, setup


package_name = "experiment_recorder"


setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(include=[package_name, f"{package_name}.*"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/docs", glob("docs/*.md")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools>=68"],
    zip_safe=True,
    maintainer="FinsSim contributors",
    maintainer_email="fins@example.com",
    description="ROS 2 experiment recording, event marking, and metadata tools for FinsSim vehicle profiles.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "record_experiment = experiment_recorder.recorder:main",
            "record_apriltag_pose = experiment_recorder.apriltag:main",
            "mark_experiment_event = experiment_recorder.marker:main",
            "finalize_experiment = experiment_recorder.finalize:main",
            "analyze_experiment = experiment_recorder.analysis:main",
            "run_t1_experiment = experiment_recorder.runner:main",
            "run_t1_disturbance_experiment = experiment_recorder.t1_disturbance_runner:main",
            "run_t2_experiment = experiment_recorder.t2_runner:main",
            "run_t1_simulation = experiment_recorder.sim_runner:run_t1_main",
            "run_t2_simulation = experiment_recorder.sim_runner:run_t2_main",
            "compare_t1_experiments = experiment_recorder.t1_comparison:main",
            "compare_paper_experiments = experiment_recorder.paper_analysis:main",
            "run_apriltag_localization_evaluation = experiment_recorder.apriltag_localization:main",
            "record_apriltag_stability = experiment_recorder.apriltag_stability:main",
            "record_top_camera_video = experiment_recorder.top_camera_video:main",
        ],
    },
)
