#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import time


# ------------------- 用户配置 ------------------- #
ROBOT_IP = "192.168.101.101"  # <-- 修改为您的机器人IP

# 机器人运动速度和加速度
SPEED = 0.1
ACCELERATION = 0.1
Z_OFFSET = 0.05
# ------------------------------------------------- #

# ------------------- 手眼标定和工具变换 (来自您的原始代码) ------------------- #
# 原始手眼标定矩阵 (相机 -> 150mm TCP)
T_E150_from_C = np.array([
    [0.99975731, -0.01944023, -0.01036356, -0.03037332],
    [0.01905377,  0.99916404, -0.03616860, -0.09828749],
    [0.01105802,  0.03596235,  0.99929196, -0.10743959],
    [0.00000000,  0.00000000,  0.00000000,  1.00000000]
])

# 从 150mm TCP 到新TCP(630mm) 的变换
T_E630_from_E150 = np.array([
    [1, 0, 0,  0.0],
    [0, 1, 0,  0.0],
    [0, 0, 1, -0.050],  # 注意是负号
    [0, 0, 0,  1.0],
])

# 计算新的手眼标定矩阵 (相机 -> 新TCP)
T_newTCP_from_camera = T_E630_from_E150 @ T_E150_from_C

# ------------------- GraspNet 输出数据 ------------------- #
# 将GraspNet提供的平移和旋转数据填入此处
# 平移向量 (Translation): [x, y, z] in meters
grasp_translation_in_camera = np.array([ 0.05313859 ,-0.04731899 , 0.416      ])
# 旋转矩阵 (Rotation Matrix)
grasp_rotation_in_camera = np.array([
[ 0.37228444 ,-0.86275494 ,-0.3421378 ],
 [ 0.6352374,   0.5056224,  -0.5837975 ],
 [ 0.6766667  , 0.  ,        0.73628956]
])

# R_correct= np.array([
#     [ 0., -1.,  0.],
#     [ 1.,  0.,  0.],
#     [ 0.,  0.,  1.]
# ])

#graspnet绕z轴转-90度是我们的tcp
R_correct= np.array( [[ 0.,  1.,  0.],
 [-1.,  0.,  0.],
 [ 0.,  0.,  1.]]   )

grasp_rotation_in_camera = grasp_rotation_in_camera @ R_correct 
# ---------------------------------------------------------------- #

def pose_to_matrix(pose):
    """将 [x, y, z, rx, ry, rz] 格式的位姿转换为 4x4 齐次变换矩阵"""
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix

def matrix_to_pose(matrix):
    """将 4x4 齐次变换矩阵转换为 [x, y, z, rx, ry, rz] 格式的位姿"""
    pose = np.zeros(6)
    pose[:3] = matrix[:3, 3]
    pose[3:] = R.from_matrix(matrix[:3, :3]).as_rotvec()
    return pose

def main():
    try:
        robot_control = rtde_control.RTDEControlInterface(ROBOT_IP)
        robot_receive = rtde_receive.RTDEReceiveInterface(ROBOT_IP)
        print("✓ 机器人连接成功。")
    except RuntimeError as e:
        print(f"✗ 机器人连接失败: {e}")
        return

    print("\n[重要!] 请确保机器人控制器中激活的TCP Z值为 0.63 米。")
    print("\n--- 机器人抓取位姿移动程序 ---")
    print("说明: 使用GraspNet数据计算目标位姿，并移动机器人TCP。")
    print("-" * 40)

    try:
        # 1. 从GraspNet数据构建抓取位姿在相机坐标系下的变换矩阵 (^C T_G)
        #    这个矩阵描述了目标抓取姿态(G)相对于相机(C)的位置和方向
        T_C_from_G = np.eye(4)
        T_C_from_G[:3, :3] = grasp_rotation_in_camera
        T_C_from_G[:3, 3] = grasp_translation_in_camera
        
        # 2. 获取机器人当前TCP位姿，并计算相机在基座坐标系下的位姿变换 (^B T_C)
        current_pose_vec = robot_receive.getActualTCPPose()
        T_B_from_E = pose_to_matrix(current_pose_vec)     # ^B T_E (基座->TCP)
        T_E_from_C = T_newTCP_from_camera                 # ^E T_C (TCP->相机)
        T_B_from_C = T_B_from_E @ T_E_from_C              # ^B T_C (基座->相机)
        
        # 3. 计算最终抓取位姿在基座坐标系下的变换矩阵 (^B T_G)
        #    这是核心计算，将相机坐标系下的目标位姿转换到机器人基座坐标系下
        #    公式: ^B T_G = ^B T_C @ ^C T_G
        T_B_from_G_grasp = T_B_from_C @ T_C_from_G
        
        # # 4. 将最终的位姿矩阵转换为机器人可执行的 [x,y,z,rx,ry,rz] 格式
        # target_pose = matrix_to_pose(T_B_from_G_grasp)


          # ========================= 【新增代码：应用您的最终修正】 =========================
        print("  > 应用最终修正：绕目标TCP自身X轴旋转-90度...")

        # 创建一个绕X轴旋转-90度的4x4修正矩阵
        rot_x_neg90 = R.from_euler('x', -90, degrees=True)
        R_correct_final = np.eye(4)
        R_correct_final[:3, :3] = rot_x_neg90.as_matrix()
        print(f"  > 旋转矩阵 (R_correct_final): \n{R_correct_final}")

        # 通过右乘(@)将修正应用到目标位姿矩阵上，实现局部坐标系下的旋转
        T_B_from_G_final = T_B_from_G_grasp #@ R_correct_final
        # ============================================================================
        # # 4. 将【经过最终修正后】的位姿矩阵转换为机器人可执行的格式
        # target_pose = matrix_to_pose(T_B_from_G_final)

         # 创建一个沿Z轴平移的4x4修正矩阵
        z_offset_matrix = np.eye(4)
        z_offset_matrix[2, 3] = Z_OFFSET

        # 通过右乘将平移应用到目标位姿矩阵上（在局部坐标系下移动）
        T_B_from_G_with_offset = T_B_from_G_final @ z_offset_matrix
        # ============================================================================
        
        # 4. 将【深度修正后】的位姿矩阵转换为机器人可执行的格式
        target_pose = matrix_to_pose(T_B_from_G_with_offset)
        

        
        print(f"  > 计算出的目标位姿 (P_b): {np.round(target_pose, 4)}")

        # 5. 安全确认并执行
        confirm = input("  ? 确认移动机器人到目标位姿? (y/n): ")
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