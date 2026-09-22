import os
import site
import sys
from glob import glob

from setuptools import find_packages, setup
from setuptools.command.develop import develop as setuptools_develop


package_name = "hydrodynamic_identification"
distribution_name = package_name.replace("_", "-")


class ColconCompatibleDevelop(setuptools_develop):
    """Accept colcon's legacy editable-uninstall flags with new setuptools."""

    user_options = setuptools_develop.user_options + [
        ("uninstall", "u", "remove a previous editable install"),
        ("editable", "e", "accepted for compatibility with colcon"),
        ("build-directory=", "b", "accepted for compatibility with colcon"),
        ("script-dir=", None, "accepted for compatibility with ament_python setup.cfg"),
    ]
    boolean_options = setuptools_develop.boolean_options + ["uninstall", "editable"]

    def initialize_options(self):
        super().initialize_options()
        self.uninstall = False
        self.editable = False
        self.build_directory = None
        self.script_dir = None

    def run(self):
        if self.uninstall:
            self._remove_egg_link()
            return
        super().run()

    def _remove_egg_link(self):
        candidates = []
        try:
            candidates.extend(site.getsitepackages())
        except AttributeError:
            pass
        candidates.append(os.path.join(sys.prefix, "lib", f"python{sys.version_info.major}.{sys.version_info.minor}", "site-packages"))

        seen = set()
        for site_dir in candidates:
            if not site_dir or site_dir in seen:
                continue
            seen.add(site_dir)
            egg_link = os.path.join(site_dir, f"{distribution_name}.egg-link")
            if os.path.exists(egg_link):
                os.remove(egg_link)
            easy_install_pth = os.path.join(site_dir, "easy-install.pth")
            if os.path.exists(easy_install_pth):
                with open(easy_install_pth, "r", encoding="utf-8") as stream:
                    lines = stream.readlines()
                filtered = [line for line in lines if package_name not in line and distribution_name not in line]
                if filtered != lines:
                    with open(easy_install_pth, "w", encoding="utf-8") as stream:
                        stream.writelines(filtered)


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
    install_requires=["setuptools>=68", "PyYAML>=6,<7", "numpy>=1.24", "Pillow>=9", "matplotlib>=3.8,<4"],
    zip_safe=True,
    maintainer="OpenAI Codex",
    maintainer_email="support@openai.com",
    description="ROS2 node for FinsROV Fossen hydrodynamic coefficient identification with roll/pitch small-angle torque pulses.",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "hydrodynamic_identifier = hydrodynamic_identification.hydrodynamic_identifier_node:main",
            "analyze_yaw_response = hydrodynamic_identification.yaw_response_analyzer:main",
            "refit_hydrodynamic = hydrodynamic_identification.refit_hydrodynamic_identification:main",
            "fit_surge_graybox = hydrodynamic_identification.surge_graybox_identifier:main",
            "plot_hydrodynamic_force_segments = hydrodynamic_identification.plot_hydrodynamic_force_segments:main",
            "unity_fossen_replay_validator = hydrodynamic_identification.unity_fossen_replay_validator:main",
        ],
    },
    cmdclass={"develop": ColconCompatibleDevelop},
)
