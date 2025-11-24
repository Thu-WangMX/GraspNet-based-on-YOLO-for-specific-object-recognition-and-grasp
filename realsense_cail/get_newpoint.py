#!/usr/bin/env python3
import sys
import time
import logging
import math
import numpy as np
import cv2
import pyrealsense2 as rs
import urx

# —— Charuco 板参数 —— #
# (这部分保持不变, 确保与你的物理标定板匹配)
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    size=(9, 7),
    dictionary=CHARUCO_DICT,
    squareLength=0.023,     # 格子的物理边长（米）
    markerLength=0.017      # ArUco 子标记边长（米）
)
DETECT_PARAMS = cv2.aruco.DetectorParameters()

def get_aligned_images(pipeline):
    """从 RealSense 获取对齐后的 RGB 图和相机内参"""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None, None, None

    # 相机内参
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    camera_matrix = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1]
    ])
    dist_coeffs = np.array(intr.coeffs)

    color_image = np.asanyarray(color_frame.get_data())
    return color_image, camera_matrix, dist_coeffs

def draw_charuco_detection(img, camera_matrix, dist_coeffs):
    """
    在图像上检测并绘制ChArUco板的识别结果。
    返回: 绘制后的图像, 以及是否成功检测到。
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, CHARUCO_DICT, parameters=DETECT_PARAMS
    )

    detected = False
    if ids is not None and len(ids) > 0:
        cv2.aruco.drawDetectedMarkers(img, corners, ids)
        retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
            markerCorners=corners,
            markerIds=ids,
            image=gray,
            board=CHARUCO_BOARD
        )
        
        if retval > 4:
            cv2.aruco.drawDetectedCornersCharuco(img, charuco_corners, charuco_ids)
            # 尝试估计位姿以绘制坐标轴
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=CHARUCO_BOARD,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs,
                rvec=np.empty(1), # dummy values
                tvec=np.empty(1)  # dummy values
            )
            if success:
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                detected = True
                
    return img, detected

def main():
    logging.basicConfig(level=logging.INFO)
    
    # —— 初始化 RealSense —— #
    pipeline = rs.pipeline()
    cfg = rs.config()
    # 根据你的相机能力选择分辨率，640x480足够用于可视化
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    profile = pipeline.start(cfg)
    logging.info("RealSense camera initialized.")

    # —— 初始化机械臂 URX —— #
    try:
        robot = urx.Robot("192.168.101.101")
    except Exception as e:
        logging.error(f"Failed to connect to robot: {e}")
        pipeline.stop()
        return

    # !! 设置你的TCP !!
    # 这个TCP必须与你之后进行手眼标定和实际应用时使用的TCP完全一致
    robot.set_tcp((0,0,0.20,0, 0,0))
    robot.set_payload(0.5, (0,0,0))
    time.sleep(0.5)
    logging.info("Robot initialized at: %s", robot.getl())

    # 用于存储采集到的点位
    collected_points = []
    
    print("\n" + "="*50)
    print("准备开始采点...")
    print("操作说明:")
    print(" - 移动机器人，在实时视频窗口中观察标定板的识别情况。")
    print(" - 当你找到一个好的位姿（标定板被清晰识别，坐标轴稳定）时，")
    print("   请按下【空格键 (Spacebar)】来保存当前机器人的TCP坐标。")
    print(" - 按下【Q键】退出程序。")
    print("="*50 + "\n")

    try:
        while True:
            # 1. 获取相机图像和内参
            color_image, camera_matrix, dist_coeffs = get_aligned_images(pipeline)
            if color_image is None:
                continue

            # 2. 在图像上绘制检测结果
            vis_image, detected = draw_charuco_detection(color_image, camera_matrix, dist_coeffs)

            # 3. 显示图像窗口
            # 在窗口标题中加入提示信息
            window_title = "Live View | Press [SPACE] to save point, [Q] to quit"
            if not detected:
                window_title += " (Board NOT detected!)"
            cv2.imshow(window_title, vis_image)
            
            # 4. 检测按键
            key = cv2.waitKey(1) & 0xFF

            # 如果按下 'q' 或 ESC，则退出循环
            if key == ord('q') or key == 27:
                logging.info("Quit key pressed. Exiting...")
                break
            
            # 如果按下空格键，则保存点位
            if key == ord(' '):
                pose = robot.getl()
                collected_points.append(pose)
                logging.info(f"Point {len(collected_points)} saved: {pose}")

    finally:
        # 5. 清理和收尾工作
        print("\n" + "="*50)
        print("采点结束。总共采集了 {} 个点。".format(len(collected_points)))
        print("请将下面的坐标数组复制到你的标定脚本中:")
        print("points = np.array([")
        for point in collected_points:
            print(f"    {list(point)},")
        print("])")
        print("="*50 + "\n")

        cv2.destroyAllWindows()
        pipeline.stop()
        robot.close()
        logging.info("Program finished.")


if __name__ == "__main__":
    main()