      
import os
import cv2
import pyrealsense2 as rs
import numpy as np
from datetime import datetime
import time

# ---------- 配置 ----------
ROOT_DIR = "mixed2"  # 根目录
NUM_IMAGES = 15                 # 总共采集张数
COLOR_RES = (1280, 720)        # 彩色分辨率
DEPTH_RES = (640, 480)         # 深度分辨率
FRAME_INTERVAL = 2           # 每帧采集间隔
# -----------------------------

os.makedirs(ROOT_DIR, exist_ok=True)

# ---------- 初始化 RealSense ----------
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, COLOR_RES[0], COLOR_RES[1], rs.format.bgr8, 30)
config.enable_stream(rs.stream.depth, DEPTH_RES[0], DEPTH_RES[1], rs.format.z16, 30)
profile = pipeline.start(config)

# ---------- 对齐对象 ----------
align_to = rs.stream.color
align = rs.align(align_to)

# ---------- 深度伪彩色对象 ----------
colorizer = rs.colorizer()

time.sleep(2)  # 等待相机自动曝光等稳定

try:
    for idx in range(1, NUM_IMAGES + 1):
        frames = pipeline.wait_for_frames()
        # 对齐深度到彩色
        aligned_frames = align.process(frames)

        color_frame = aligned_frames.get_color_frame()
        depth_frame = aligned_frames.get_depth_frame()

        if not color_frame or not depth_frame:
            continue

        # 转为 numpy
        color_img = np.asanyarray(color_frame.get_data())
        depth_img = np.asanyarray(depth_frame.get_data())           # 对齐后的深度
        depth_color_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())

        # 为每张图创建单独文件夹
        img_dir = os.path.join(ROOT_DIR, str(idx))
        os.makedirs(img_dir, exist_ok=True)

        # 文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        cv2.imwrite(os.path.join(img_dir, f"color_{timestamp}.png"), color_img)
        cv2.imwrite(os.path.join(img_dir, f"depth_{timestamp}.png"), depth_img)
        cv2.imwrite(os.path.join(img_dir, f"depth_color_{timestamp}.png"), depth_color_img)

        print(f"已采集 {idx}/{NUM_IMAGES} 张，保存到 {img_dir}")
        time.sleep(FRAME_INTERVAL)

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    print(f"\n✅ 总共采集 {NUM_IMAGES} 张图像，每张图像保存在单独文件夹下")

    