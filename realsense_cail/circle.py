import cv2
import numpy as np
import pyrealsense2 as rs

# 假设 extrinsic_matrix 是 world -> camera
extrinsic_matrix = np.array([
    [-0.05976104,  0.79598719, -0.60235622,  1.13315002],
    [ 0.99799063,  0.06037170, -0.01923423, -0.05149746],
    [ 0.02105507, -0.60229533, -0.79799563,  0.72158258],
    [ 0.        ,  0.        ,  0.        ,  1.        ]
])
# 预先计算它的逆，用于 camera->world
extrinsic_inv = np.linalg.inv(extrinsic_matrix)

pipeline = rs.pipeline()
config   = rs.config()
config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
config.enable_stream(rs.stream.color, 1280, 720, rs.format.bgr8, 30)
profile  = pipeline.start(config)

# 获取深度尺度（米/深度单位）
depth_sensor = profile.get_device().first_depth_sensor()
depth_scale  = depth_sensor.get_depth_scale()

# 对齐到 color
align      = rs.align(rs.stream.color)
point_2d   = None

def mouse_callback(event, x, y, flags, param):
    global point_2d
    if event == cv2.EVENT_LBUTTONDOWN:
        point_2d = (x, y)

cv2.namedWindow("RealSense")
cv2.setMouseCallback("RealSense", mouse_callback)

try:
    while True:
        frames        = pipeline.wait_for_frames()
        aligned       = align.process(frames)
        depth_frame   = aligned.get_depth_frame()
        color_frame   = aligned.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        intr = color_frame.profile.as_video_stream_profile().intrinsics
        fx, fy, cx, cy = intr.fx, intr.fy, intr.ppx, intr.ppy

        if point_2d is not None:
            x, y = point_2d
            raw_d = depth_image[y, x]
            if raw_d == 0:
                print("No depth data at this point.")
            else:
                d_m = raw_d * depth_scale  # 深度（米）
                # 生成相机坐标系下的点（米）
                X_cam = (x - cx) * d_m / fx
                Y_cam = (y - cy) * d_m / fy
                Z_cam = d_m
                point_cam = np.array([X_cam, Y_cam, Z_cam])
                print("Camera coords (m):", np.array2string(point_cam, separator=','))

                # 齐次坐标
                point_cam_h = np.hstack([point_cam, [1.0]])
                # 变换到世界系
                point_world_h = extrinsic_matrix @ point_cam_h
                # （如果最后一维不是 1，再除一下）
                point_world = point_world_h[:3] / point_world_h[3]
                print("World coords (m):", np.array2string(point_world, separator=','))

            point_2d = None

        cv2.imshow("RealSense", color_image)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
