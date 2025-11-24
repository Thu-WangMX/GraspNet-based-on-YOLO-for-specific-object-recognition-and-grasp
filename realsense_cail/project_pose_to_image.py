

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# 依赖:
# pip install numpy ur-rtde scipy opencv-python pyrealsense2

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import time
import cv2
import pyrealsense2 as rs

# ------------------- 用户配置 ------------------- #
# 机器人IP地址
ROBOT_IP = "192.168.101.101" # <-- 修改为您的机器人IP

# 从您的手眼标定结果中复制 T_E_C (相机在末端坐标系下的位姿)
T_effector_from_camera = np.array([
    [0.99975731, -0.01944023, -0.01036356, -0.03037332],
    [0.01905377, 0.99916404, -0.03616860, -0.09828749],
    [0.01105802, 0.03596235, 0.99929196, -0.10743959],
    [0.00000000, 0.00000000, 0.00000000, 1.00000000]
])
# 【新】输入您相机的内参矩阵 K 和图像分辨率
# 警告: 此处的分辨率必须与您标定K矩阵时使用的分辨率完全一致!
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480
K_matrix = np.array([
    [607.32080078125, 0.0,   324.86956787109375],  # fx, 0, cx
    [0.0,   607.1389770507812, 248.99806213378906], # 0, fy, cy
    [0.0,   0.0,   1.0]
])
# 机器人运动速度和加速度
SPEED = 0.2
ACCELERATION = 0.2
# ------------------------------------------------- #

def pose_to_matrix(pose_vec):
    """将UR的pose list [x,y,z,rx,ry,rz] 转换为 4x4 变换矩阵"""
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_rotvec(pose_vec[3:]).as_matrix()
    matrix[:3, 3] = pose_vec[:3]
    return matrix

def project_pose_to_image(T_B_Ea, T_B_Ec, T_E_C, K):
    """
    将机械臂在位姿C时的TCP，投影到机械臂在位姿A处拍摄的图像中。
    """
    P_Ec = np.array([0, 0, 0, 1])
    P_b = T_B_Ec @ P_Ec
    print(f"  > TCP at Pose C in Base Frame (P_b): {np.round(P_b[:3], 4)}")
    T_B_Ca = T_B_Ea @ T_E_C
    T_Ca_B = np.linalg.inv(T_B_Ca)
    P_Ca = T_Ca_B @ P_b
    print(f"  > TCP at Pose C in Camera A's Frame (P_Ca): {np.round(P_Ca[:3], 4)}")

    Z = P_Ca[2]
    if Z <= 0:
        print("  ✗ 错误: 目标点在相机后面，无法投影。")
        return None

    point_3d_in_camera = P_Ca[:3]
    pixel_coords_homogeneous = K @ point_3d_in_camera
    u = pixel_coords_homogeneous[0] / pixel_coords_homogeneous[2]
    v = pixel_coords_homogeneous[1] / pixel_coords_homogeneous[2]
    return (u, v)

def visualize_projection(pixel_coordinates):
    """
    启动相机，显示视频流，并在视频上绘制投影点。
    """
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, IMAGE_WIDTH, IMAGE_HEIGHT, rs.format.bgr8, 30)
    
    print("\n[INFO] 正在启动相机...")
    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"[ERROR] 无法启动相机，请检查相机连接: {e}")
        return

    print("[INFO] 相机已启动。在弹出的窗口中按 'q' 键退出。")

    if pixel_coordinates:
        point_to_draw = (int(pixel_coordinates[0]), int(pixel_coordinates[1]))
    else:
        point_to_draw = None

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())

            if point_to_draw:
                if 0 <= point_to_draw[0] < IMAGE_WIDTH and 0 <= point_to_draw[1] < IMAGE_HEIGHT:
                    cv2.circle(color_image, point_to_draw, 10, (0, 0, 255), 2)
                    cv2.circle(color_image, point_to_draw, 2, (0, 0, 255), -1)
                    cv2.putText(color_image, f"({point_to_draw[0]}, {point_to_draw[1]})",
                                (point_to_draw[0] + 15, point_to_draw[1] + 15),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                     cv2.putText(color_image, "Projected point is outside the image frame!",
                                (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            cv2.imshow("Live Projection - Press Q to Quit", color_image)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
    finally:
        print("[INFO] 正在关闭相机...")
        pipeline.stop()
        cv2.destroyAllWindows()

def main():
    """主函数，用于连接机器人并执行投影计算和可视化"""
    robot_control, robot_receive = None, None
    try:
        # 机器人连接仍然是必要的，用于后续的移动指令
        robot_control = rtde_control.RTDEControlInterface(ROBOT_IP)
        robot_receive = rtde_receive.RTDEReceiveInterface(ROBOT_IP) # 保留用于检查连接状态
        print("✓ 机器人连接成功。")
    except RuntimeError as e:
        print(f"✗ 机器人连接失败: {e}")
        return

    print("\n--- 机器人位姿投影与可视化程序 (自动模式) ---")
    print("1. 使用预设的位姿 (Pose A 和 Pose C) 进行计算。")
    print("2. 计算后，机器人将自动移动到 Pose A。")
    print("3. 启动相机进行可视化验证。")
    print("-" * 40)

    try:
        # ==================== 【修改部分】 ==================== #
        # 直接定义 Pose A 和 Pose C，不再需要用户交互
        pose_a_vec = np.array([4.500e-02, 4.160e-01, 3.670e-01, 3.141e+00, 1.600e-02, 2.000e-03])
        pose_c_vec = np.array([0.023686463739456362, 0.525527490096116, 0.02850107231040011, 3.141027196630571, 0.015997768360186695, 0.0019920486440520523])
        
        print(f"  > 使用预设 Pose A: {np.round(pose_a_vec, 3)}")
        print(f"  > 使用预设 Pose C: {np.round(pose_c_vec, 3)}")
        
        # 将位姿向量转换为变换矩阵
        T_B_Ea = pose_to_matrix(pose_a_vec)
        T_B_Ec = pose_to_matrix(pose_c_vec)
        # ====================================================== #
        
        # 步骤3: 执行投影计算
        print("\n--- 开始计算投影 ---")
        pixel_coordinates = project_pose_to_image(
            T_B_Ea=T_B_Ea, T_B_Ec=T_B_Ec,
            T_E_C=T_effector_from_camera, K=K_matrix
        )

        if pixel_coordinates:
            print("\n" + "="*40)
            print("✓ 计算完成!")
            print(f"  投影的像素坐标 (u, v) 是: ({pixel_coordinates[0]:.2f}, {pixel_coordinates[1]:.2f})")
            print("="*40)
        else:
            print("\n✗ 无法进行下一步，因为投影计算失败。")
            return

        # 步骤4: 自动移动机器人回 Pose A 并启动可视化
        print("\n[INFO] 准备自动将机器人移动回 Pose A 并启动相机...")
        time.sleep(2) # 短暂停顿，给用户阅读信息的时间
        
        print("[INFO] 正在将机器人移动回 Pose A...")
        robot_control.moveL(pose_a_vec, SPEED, ACCELERATION)
        print("✓ 机器人已到达 Pose A。")
        
        # 启动可视化
        visualize_projection(pixel_coordinates)

    finally:
        # 清理
        if robot_control and robot_control.isConnected():
            robot_control.stopScript()
        print("\n机器人连接已断开。程序退出。")

if __name__ == "__main__":
    main()
