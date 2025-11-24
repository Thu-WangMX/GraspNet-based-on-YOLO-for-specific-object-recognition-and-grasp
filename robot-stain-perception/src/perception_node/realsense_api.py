# src/perception_node/realsense_api.py

import numpy as np
import pyrealsense2 as rs
from collections import OrderedDict
import cv2

class RealsenseAPI:
    """
    一个封装了Intel RealSense D400系列相机高级功能的接口类。
    功能包括：多相机支持、传感器参数设置、以及深度图后处理滤波器链。
    """
    def __init__(self, height=480, width=640, fps=30, warm_start=60):
        self.height = height
        self.width = width
        self.fps = fps
        self.depth_scale = 0.0

        # 识别设备
        self.device_ls = []
        for c in rs.context().query_devices():
            self.device_ls.append(c.get_info(rs.camera_info(1)))

        if not self.device_ls:
            raise RuntimeError("未检测到 RealSense 设备。请检查相机连接。")

        # 启动数据流
        print(f"正在连接 RealSense 相机 ({len(self.device_ls)} found) ...")
        self.pipes = []
        self.profiles = OrderedDict()
        for i, device_id in enumerate(self.device_ls):
            pipe = rs.pipeline()
            config = rs.config()
            config.enable_device(device_id)
            config.enable_stream(rs.stream.depth, self.width, self.height, rs.format.z16, self.fps)
            config.enable_stream(rs.stream.color, self.width, self.height, rs.format.rgb8, self.fps)
            self.pipes.append(pipe)
            profile = pipe.start(config)
            self.profiles[device_id] = profile

            # 获取深度比例因子 (用于将 uint16 毫米单位转为 float 米单位)
            depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = depth_sensor.get_depth_scale()
            
            print(f"已连接到相机 {i+1} ({device_id})。深度比例因子: {self.depth_scale}")

        try:
            # 获取深度传感器并进行高级设置
            print("正在配置相机高级参数...")
            depth_sensor = self.profiles[self.device_ls[0]].get_device().first_depth_sensor()
            if depth_sensor.supports(rs.option.emitter_enabled):
                depth_sensor.set_option(rs.option.emitter_enabled, 1.0)
            if depth_sensor.supports(rs.option.laser_power):
                depth_sensor.set_option(rs.option.laser_power, depth_sensor.get_option_range(rs.option.laser_power).max)
        except Exception as e:
            print(f"  - 警告：设置红外投影器高级选项失败: {e}")

        self.align = rs.align(rs.stream.color)
        
        # 初始化滤波器链
        self._initialize_filters()
        
        # 相机预热
        print("相机预热中 (等待图像稳定)...")
        for _ in range(warm_start):
            self._get_frames()
        print("初始化完成。")

    def _initialize_filters(self):
        print("初始化深度滤波器链...")
        # 推荐的滤波器顺序和参数
        self.depth_to_disparity = rs.disparity_transform(True)
        self.spatial_filter = rs.spatial_filter()
        self.temporal_filter = rs.temporal_filter()
        self.hole_filling_filter = rs.hole_filling_filter(1)
        self.disparity_to_depth = rs.disparity_transform(False)
        
        # 你可以根据需要微调这些参数
        self.spatial_filter.set_option(rs.option.filter_magnitude, 2.0)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 20.0)
        
        self.temporal_filter.set_option(rs.option.filter_smooth_alpha, 0.4)
        self.temporal_filter.set_option(rs.option.filter_smooth_delta, 20.0)
        
        self.filter_chain = [
            self.depth_to_disparity,
            self.spatial_filter,
            self.temporal_filter,
            self.hole_filling_filter,
            self.disparity_to_depth
        ]
        print("  - 滤波器链配置完成。")

    def _apply_filters(self, depth_frame):
        frame = depth_frame
        for f in self.filter_chain:
            frame = f.process(frame)
        return frame

    def _get_frames(self):
        framesets = [pipe.wait_for_frames(5000) for pipe in self.pipes] # 5秒超时
        return [self.align.process(frameset) for frameset in framesets]

    def get_num_cameras(self):
        return len(self.device_ls)

    def get_frames(self):
        """
        获取一对对齐且经过滤波的RGB和深度图像。

        Returns:
            tuple(np.ndarray, np.ndarray) or (None, None): 
            返回 (rgb_image, depth_image_in_meters)。如果失败则返回 (None, None)。
            即使有多个相机，也只返回第一个相机的数据。
        """
        try:
            framesets = self._get_frames()
            if not framesets:
                return None, None

            # 假设只使用第一个相机
            first_frameset = framesets[0]
            
            # 获取并后处理深度帧
            depth_frame = first_frameset.get_depth_frame()
            filtered_depth_frame = self._apply_filters(depth_frame)
            depth_image_mm = np.asanyarray(filtered_depth_frame.get_data())
            # 将深度单位从毫米(uint16)转为米(float)
            depth_image_m = depth_image_mm.astype(np.float32) * self.depth_scale

            # 获取RGB帧
            color_frame = first_frameset.get_color_frame()
            # 从API获取的是RGB格式
            rgb_image = np.asanyarray(color_frame.get_data())
            # 将其转换为OpenCV常用的BGR格式
            bgr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            return bgr_image, depth_image_m

        except Exception as e:
            print(f"获取帧时出错: {e}")
            return None, None
    
    def close(self):
        """安全地关闭所有相机管道。"""
        print("正在关闭 RealSense 相机管道...")
        for pipe in self.pipes:
            pipe.stop()
            
            
    def get_intrinsics(self):
            """
            返回主相机（第一个设备）的相机内参。
            格式：一个 dict，至少包含 fx, fy, ppx, ppy，这几个键是
            perception_api_detect.py 里显式检查的。
            """
            if not hasattr(self, "profiles") or not self.profiles:
                raise RuntimeError("RealSense 尚未初始化，无法获取内参。")

            # 使用第一个相机的 profile
            device_id = self.device_ls[0]
            profile = self.profiles[device_id]

            # 取彩色相机的内参（与 bgr_image 对应）
            color_stream = profile.get_stream(rs.stream.color).as_video_stream_profile()
            intr = color_stream.get_intrinsics()

            camera_intrinsics = {
                "fx": intr.fx,
                "fy": intr.fy,
                "ppx": intr.ppx,   # 注意这里键名是 ppx / ppy
                "ppy": intr.ppy,
                "width": intr.width,
                "height": intr.height,
                "depth_scale": self.depth_scale,  # 你在 __init__ 里算好的
            }
            return camera_intrinsics