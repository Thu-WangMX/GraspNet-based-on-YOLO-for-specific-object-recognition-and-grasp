from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'perception_node'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # 包含launch文件
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        # 包含weights目录和里面的模型文件
        (os.path.join('share', package_name, 'weights'), glob('weights/*')),
        # 包含config目录和里面的yaml文件
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='hjj',
    maintainer_email='your_email@todo.com',
    description='A ROS2 package for real-time stain perception.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            # 这行是关键：它创建了一个名为 'perception_node' 的可执行文件
            # 它会去运行 src/perception_node/run_robot_perception_detect.py 文件中的 main 函数
            'perception_node = src.perception_node.run_robot_perception_detect:main',
        ],
    },
)