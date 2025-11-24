import pyrealsense2 as rs
import numpy as np

def get_realsense_intrinsics():
    """
    启动RealSense相机，获取颜色传感器的内参，并将其格式化为3x3矩阵。

    Returns:
        A tuple containing:
        - intrinsics_matrix (np.ndarray): 3x3的相机内参矩阵。
        - intrinsics_obj (rs.intrinsics): 原始的RealSense内参对象。
        Returns (None, None) if the camera cannot be started.
    """
    # 1. 初始化 RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 我们只需要颜色流的配置信息来获取内参，不需要启动深度流
    # 可以根据你的相机支持的分辨率和帧率进行修改
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    intrinsics_matrix = None
    intrinsics_obj = None

    try:
        # 2. 启动数据流
        profile = pipeline.start(config)
        
        # 3. 获取颜色流的配置和内参
        color_profile = profile.get_stream(rs.stream.color)
        intrinsics_obj = color_profile.as_video_stream_profile().get_intrinsics()
        
        # 4. 将内参数据提取并存入一个 3x3 NumPy 矩阵
        intrinsics_matrix = np.array([
            [intrinsics_obj.fx, 0, intrinsics_obj.ppx],
            [0, intrinsics_obj.fy, intrinsics_obj.ppy],
            [0, 0, 1]
        ])

    except RuntimeError as e:
        print(f"无法启动相机或获取数据: {e}")

    finally:
        # 5. 停止数据流
        pipeline.stop()
        print("相机已停止。")
        
    return intrinsics_matrix, intrinsics_obj


if __name__ == '__main__':
    # 调用函数来获取内参
    camera_matrix, rs_intrinsics = get_realsense_intrinsics()
    
    if camera_matrix is not None:
        print("\n--- 相机内参详情 ---")
        # 原始的内参对象包含所有参数
        print(f"宽度 (Width):      {rs_intrinsics.width}")
        print(f"高度 (Height):     {rs_intrinsics.height}")
        print(f"主点 X (ppx):    {rs_intrinsics.ppx}")
        print(f"主点 Y (ppy):    {rs_intrinsics.ppy}")
        print(f"焦距 X (fx):     {rs_intrinsics.fx}")
        print(f"焦距 Y (fy):     {rs_intrinsics.fy}")
        print(f"畸变模型 (coeffs): {rs_intrinsics.model}")
        print(f"畸变系数 (coeffs): {rs_intrinsics.coeffs}")
        
        print("\n--- 3x3 相机内参矩阵 ---")
        print(camera_matrix)