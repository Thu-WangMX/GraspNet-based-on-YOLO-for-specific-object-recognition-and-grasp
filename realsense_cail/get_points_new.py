#!/usr/bin/env python3
import sys
import time
import logging
import math
import numpy as np
import cv2
import pyrealsense2 as rs
import rtde_receive
import json  # 1. 导入json模块

# --- Charuco 板参数 ---
# (此部分无改动)
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    size=(7, 9),
    dictionary=CHARUCO_DICT,
    squareLength=0.027,     # 格子的物理边长（米）
    markerLength=0.02     # ArUco子标记的物理边长（米）
)
DETECT_PARAMS = cv2.aruco.DetectorParameters()

def get_camera_frame(pipeline):
    """从RealSense相机获取彩色图像帧和内参"""
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    if not color_frame:
        return None, None, None

    intr = color_frame.profile.as_video_stream_profile().intrinsics
    camera_matrix = np.array([
        [intr.fx, 0, intr.ppx],
        [0, intr.fy, intr.ppy],
        [0, 0, 1]
    ])
    dist_coeffs = np.array(intr.coeffs)
    color_image = np.asanyarray(color_frame.get_data())
    
    return color_image, camera_matrix, dist_coeffs

def draw_charuco_visuals(img, camera_matrix, dist_coeffs):
    """在图像上检测并绘制ChArUco板的识别结果"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = cv2.aruco.detectMarkers(gray, CHARUCO_DICT, parameters=DETECT_PARAMS)

    is_detected = False
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
            # 尝试估计位姿以绘制坐标轴，提供更好的视觉反馈
            success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
                rvec=np.empty(1), # 占位符
                tvec=np.empty(1),  # 占位符
                charucoCorners=charuco_corners,
                charucoIds=charuco_ids,
                board=CHARUCO_BOARD,
                cameraMatrix=camera_matrix,
                distCoeffs=dist_coeffs
            )
            if success:
                cv2.drawFrameAxes(img, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
                is_detected = True
                
    return img, is_detected

def main():
    logging.basicConfig(level=logging.INFO)
    
    # --- 初始化 RealSense 相机 ---
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    logging.info("RealSense 相机初始化完成。")

    # --- 初始化机器人通信 ---
    try:
        rtde_r = rtde_receive.RTDEReceiveInterface("192.168.101.101")
        logging.info("已连接到机器人。")
    except Exception as e:
        logging.error(f"连接机器人失败: {e}")
        pipeline.stop()
        return

    collected_points = []
    
    print("\n" + "="*60)
    print(" 🤖 用于手眼标定的视觉辅助采点工具 👁️")
    print("-" * 60)
    print("操作说明:")
    print(" 1. 使用示教器手动移动机器人。")
    print(" 2. 观察实时视频窗口，找到一个能清晰检测到标定板的良好位姿。")
    print(" 3. 按下【空格键】保存当前机器人的TCP位姿。")
    print(" 4. 按下【Q键】完成采点并退出程序。")
    print("=" * 60 + "\n")
    
    try:
        while True:
            color_image, camera_matrix, dist_coeffs = get_camera_frame(pipeline)
            if color_image is None:
                continue

            vis_image, is_detected = draw_charuco_visuals(color_image, camera_matrix, dist_coeffs)
            
            window_title = "Live View | Press [SPACE] to Save | [Q] to Quit"
            if not is_detected:
                cv2.putText(vis_image, "BOARD NOT DETECTED", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.imshow(window_title, vis_image)
            
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                logging.info("按下 'Q' 键，正在结束采点...")
                break
            
            if key == ord(' '):
                pose = rtde_r.getActualTCPPose()
                collected_points.append(pose)
                logging.info(f"点 {len(collected_points)} 已保存: {pose}")

    finally:
        # 5. 清理并打印/保存最终结果
        cv2.destroyAllWindows()
        pipeline.stop()
        logging.info("程序结束。")

        print("\n" + "="*60)
        print(f"采集完成。总共保存了 {len(collected_points)} 个点。")

        if collected_points:
            # 2. 将采集到的点位写入JSON文件
            output_filename = "collected_robot_poses.json"
            try:
                with open(output_filename, 'w') as f:
                    # 使用json.dump写入文件，indent=4使其格式化，更易读
                    json.dump(collected_points, f, indent=4)
                logging.info(f"所有点位已成功保存到文件: {output_filename}")
            except Exception as e:
                logging.error(f"写入JSON文件失败: {e}")

            # (可选) 保留原有的终端打印输出，方便快速查看
            print("您可以将下面的数组复制到您的主标定脚本中:")
            print("\npoints = np.array([")
            for point in collected_points:
                formatted_point = ", ".join([f"{p:.8f}" for p in point])
                print(f"    [{formatted_point}],")
            print("])\n")
        
        print("="*60 + "\n")

if __name__ == "__main__":
    main()