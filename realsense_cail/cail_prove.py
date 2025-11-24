#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import time

# ------------------- 用户配置 ------------------- #
ROBOT_IP = "192.168.101.101" # <-- 修改为您的机器人IP

# 原始手眼标定矩阵 (相机 -> 150mm TCP)
T_oldTCP_from_camera = np.array([
    [0.99975731, -0.01944023, -0.01036356, -0.03037332],
    [0.01905377,  0.99916404, -0.03616860, -0.09828749],
    [0.01105802,  0.03596235,  0.99929196, -0.10743959],
    [0.00000000,  0.00000000,  0.00000000,  1.00000000]
])




T_E150_from_C = T_oldTCP_from_camera

#马桶刷
T_E630_from_E150 = np.array([
    [1, 0, 0,  0.0],
    [0, 1, 0,  0.0],
    [0, 0, 1, -0.480],  # 注意是负号
    [0, 0, 0,  1.0],
])

#夹爪
T_E630_from_E150 = np.array([
    [1, 0, 0,  0.0],
    [0, 1, 0,  0.0],
    [0, 0, 1, -0.05],  # 注意是负号
    [0, 0, 0,  1.0],
])


T_newTCP_from_camera = T_E630_from_E150 @ T_E150_from_C  # 替换你原来的那行
#print("1111")
#print(T_newTCP_from_camera)

# 机器人运动速度和加速度
SPEED = 0.1
ACCELERATION = 0.1
# ------------------------------------------------- #

def pose_to_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix

def main():
    try:
        robot_control = rtde_control.RTDEControlInterface(ROBOT_IP)
        robot_receive = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        print("✓ 机器人连接成功。")
    except RuntimeError as e:
        print(f"✗ 机器人连接失败: {e}")
        return

    # ################################################################## #
    # ## 关键提醒：请确保在运行此脚本前，已在机器人示教器上      ## #
    # ## 将当前激活的TCP设置为 Z=630mm (0.63m)。                 ## #
    # ################################################################## #
    print("\n[重要!] 请确保机器人控制器中激活的TCP Z值为 0.63 米。")

    print("\n--- 机器人移动控制程序 (坐标输入模式) ---")
    print("说明: 输入目标在相机坐标系下的坐标 (X,Y,Z)，机器人将移动TCP到该世界坐标。")
    print("-" * 40)

    try:
        while True:
            user_input = input("\n请输入目标在相机坐标系下的坐标 (X,Y,Z), 单位为米: ")
            if user_input.lower() in ['q', 'quit', 'exit']: break

            try:
                parts = user_input.replace(',', ' ').split()
                if len(parts) != 3: raise ValueError("需要输入3个数字。")
                
                point_in_camera_coords = [float(p) for p in parts]
                P_c = np.array(point_in_camera_coords + [1])
                print(f"  > 已接收相机坐标 (P_c): {P_c[:3]}")
            except ValueError as e:
                print(f"✗ 输入无效: {e}")
                continue

            current_pose_vec = robot_receive.getActualTCPPose()
            print("current_pose_vec",current_pose_vec)
            T_B_from_E = pose_to_matrix(current_pose_vec)     # ^B T_E
            T_E_from_C = T_newTCP_from_camera                 # ^E T_C
            T_B_from_C = T_B_from_E @ T_E_from_C              # ^B T_C
            P_b_homogeneous = T_B_from_C @ P_c                # ^B T_C * [P_c;1]
            P_b = P_b_homogeneous[:3]
                        
            print(f"  > 计算出的世界坐标 (P_b): {np.round(P_b, 4)}")

            # 3. 创建要移动到的目标位姿
            target_pose = [
                P_b[0], P_b[1], P_b[2],
                current_pose_vec[3], current_pose_vec[4], current_pose_vec[5]
            ]
            print(f"  > 准备移动到目标位姿: {np.round(target_pose, 4)}")

            # 4. 安全确认
            confirm = input("  ? 确认移动机器人? (y/n): ")
            if confirm.lower() == 'y':
                print("  > 正在移动机器人...")
                robot_control.moveL(target_pose, SPEED, ACCELERATION)
                print("  ✓ 移动完成。")
            else:
                print("  - 移动已取消。")
    
    finally:
        if 'robot_control' in locals() and robot_control.isConnected():
            robot_control.stopScript()
        print("\n程序退出，机器人连接已断开。")

if __name__ == "__main__":
    main()