import os
from glob import glob
from setuptools import find_packages, setup

package_name = 'grpc_ros_adapter'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(include=[package_name, f'{package_name}.*']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*_launch.py'))),
        (os.path.join('share', package_name, 'launch'), glob(os.path.join('launch', '*.launch'))),
        (os.path.join('share', package_name, 'config'), glob(os.path.join('config', '*.yaml'))),
    ],
    install_requires=[
        'setuptools>=68',
        'grpcio>=1.48.2,<2',
        'numpy>=1.24,<2',
        'protobuf>=3.20.3,<3.21',
        'PyYAML>=6,<7',
        'rospkg>=1.5,<2',
    ],
    zip_safe=True,
    maintainer='Labust',
    maintainer_email='labust@fer.hr',
    description='Bridges and translates messages between ROS and Unity using gRPC',
    license='Apache License 2.0',
    entry_points={
        'console_scripts': [
            f'server = {package_name}.server:main',
        ],
    },
)
