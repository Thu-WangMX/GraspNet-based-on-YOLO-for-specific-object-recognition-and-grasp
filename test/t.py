import pyrealsense2 as rs

def get_realsense_depth_scale():
    """
    启动RealSense相机并获取深度传感器的缩放因子 (depth_scale)。

    Returns:
        float: 深度缩放因子。如果无法启动相机则返回 None。
    """
    # 1. 初始化 RealSense Pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 必须启用深度流才能查询深度传感器的属性
    # 分辨率和帧率可以根据需要修改
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    depth_scale = None
    
    try:
        # 2. 启动数据流并获取配置信息
        profile = pipeline.start(config)
        
        # 3. 从配置中获取设备，并找到第一个深度传感器
        depth_sensor = profile.get_device().first_depth_sensor()
        
        # 4. 查询深度传感器的缩放因子
        depth_scale = depth_sensor.get_depth_scale()

    except RuntimeError as e:
        print(f"无法启动相机或获取数据: {e}")

    finally:
        # 5. 停止数据流
        pipeline.stop()
        print("相机已停止。")
        
    return depth_scale


if __name__ == '__main__':
    # 调用函数获取 factor_depth
    factor_depth = get_realsense_depth_scale()
    
    if factor_depth is not None:
        print("\n--- 深度传感器缩放因子 (factor_depth) ---")
        print(f"Depth Scale 的值为: {factor_depth}")
        
        print("\n--- 说明 ---")
        print("这个值意味着，深度图中的一个像素值，乘以这个缩放因子，就等于它在真实世界中的距离（单位：米）。")
        
        # 举例说明
        raw_depth_value = 1000 # 假设从深度图中读到的像素值为 1000
        distance_in_meters = raw_depth_value * factor_depth
        print(f"例如: 深度图中一个为 {raw_depth_value} 的像素值, 对应的真实距离是 {distance_in_meters:.3f} 米。")