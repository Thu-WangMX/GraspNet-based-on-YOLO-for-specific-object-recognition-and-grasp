from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    
    # 获取realsense2_camera包的launch文件路径
    realsense_launch_path = os.path.join(
        get_package_share_directory('realsense2_camera'),
        'launch',
        'rs_launch.py'
    )

    # 包含并启动RealSense相机节点
    realsense_camera_node = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(realsense_launch_path),
        launch_arguments={'align_depth.enable': 'true'}.items()
    )

    # 定义我们自己的感知节点
    perception_node = Node(
        package='perception_node',
        executable='perception_node', # 这个名字来自setup.py的entry_points
        name='stain_perception_node',
        output='screen'
        # 你可以在这里通过parameters传递模型路径等参数，但目前我们先使用脚本内的默认值
    )
    
    return LaunchDescription([
        realsense_camera_node,
        perception_node
    ])