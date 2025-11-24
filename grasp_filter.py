#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import time

# ========================= 用户配置与权重 =========================
ROBOT_IP = "192.168.101.101"
SPEED = 0.1
ACCELERATION = 0.1
Z_OFFSET = 0.02

# --- 加权分数权重 (可调整) ---
# 用于平衡原始分数和运动成本
WEIGHT_GRASPSCORE = 1.0  # GraspNet原始分数的权重
WEIGHT_DISTANCE = -0.5   # 平移距离的惩罚权重 (负值)
WEIGHT_ANGLE = -1.0    # 旋转角度的惩罚权重 (负值)
# =================================================================

# --- 手眼标定、工具变换和最终修正矩阵 (根据我们之前的调试结果) ---
T_E150_from_C = np.array([
    [0.99975731, -0.01944023, -0.01036356, -0.03037332],
    [0.01905377,  0.99916404, -0.03616860, -0.09828749],
    [0.01105802,  0.03596235,  0.99929196, -0.10743959],
    [0.00000000,  0.00000000,  0.00000000,  1.00000000]
])
T_E630_from_E150 = np.array([
    [1, 0, 0,  0.0], [0, 1, 0,  0.0], [0, 0, 1, -0.050], [0, 0, 0,  1.0],
])
T_newTCP_from_camera = T_E630_from_E150 @ T_E150_from_C

# 最终的复合修正矩阵 (Z轴-90度 -> X轴-90度)
rot_z_neg90 = R.from_euler('z', -90, degrees=True)
rot_x_neg90 = R.from_euler('x', -90, degrees=True)
final_fix_rotation = rot_z_neg90 * rot_x_neg90
R_correct_combined = final_fix_rotation.as_matrix()

# 镜像修正矩阵
mirror_fix_X = np.diag([-1, 1, 1, 1])
# -----------------------------------------------------------------

def pose_to_matrix(pose):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_rotvec(pose[3:]).as_matrix()
    matrix[:3, 3] = pose[:3]
    return matrix

def matrix_to_pose(matrix):
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

    print("\n--- 机器人智能抓取选择与执行程序 ---")
    try:
        # 1. 加载抓取候选位姿
        print("\n> 正在加载抓取候选位姿 from 'grasp_candidates.npz'...")
        candidates = np.load('grasp_candidates.npz')
        translations = candidates['translations']
        rotations = candidates['rotation_matrices']
        scores = candidates['scores']
        if len(translations) == 0:
            print("✗ 未找到任何候选抓取位姿。")
            return

        # 2. 获取机器人当前位姿 (只获取一次)
        current_pose_vec = robot_receive.getActualTCPPose()
        T_B_from_E_current = pose_to_matrix(current_pose_vec)
        T_B_from_C = T_B_from_E_current @ T_newTCP_from_camera
        
        best_weighted_score = -np.inf
        best_grasp_matrix = None
        best_grasp_index = -1

        print("\n> 正在为 %d 个候选位姿计算加权分数..." % len(translations))
        # 3. 遍历所有候选，计算加权分数
        for i in range(len(translations)):
            # --- 对每个候选位姿应用所有修正 ---
            T_C_from_G_raw = np.eye(4)
            T_C_from_G_raw[:3, :3] = rotations[i]
            T_C_from_G_raw[:3, 3] = translations[i]
            
            R_correct_mat = np.eye(4)
            R_correct_mat[:3,:3] = R_correct_combined
            T_C_from_G_tool_corrected = T_C_from_G_raw @ R_correct_mat
            
            T_C_from_G_final_corrected =  T_C_from_G_tool_corrected
            
            T_B_from_G_target = T_B_from_C @ T_C_from_G_final_corrected

            # --- 计算姿态变化幅度 ---
            # 平移距离
            dist_cost = np.linalg.norm(T_B_from_G_target[:3, 3] - T_B_from_E_current[:3, 3])
            
            # 旋转角度
            R_relative = T_B_from_E_current[:3, :3].T @ T_B_from_G_target[:3, :3]
            trace = np.trace(R_relative)
            angle_cost_rad = np.arccos(np.clip((trace - 1) / 2.0, -1.0, 1.0))

            # --- 计算加权总分 ---
            original_score = scores[i]
            weighted_score = (WEIGHT_GRASPSCORE * original_score) + \
                             (WEIGHT_DISTANCE * dist_cost) + \
                             (WEIGHT_ANGLE * angle_cost_rad)
            
            print(f"  - 候选 #{i+1}: GraspScore={original_score:.2f}, DistCost={dist_cost:.2f}m, AngleCost={np.rad2deg(angle_cost_rad):.1f}°, WeightedScore={weighted_score:.2f}")

            if weighted_score > best_weighted_score:
                best_weighted_score = weighted_score
                best_grasp_matrix = T_B_from_G_target
                best_grasp_index = i

        # 4. 选择最佳位姿并执行
        print(f"\n> 已选择最佳位姿: 候选 #{best_grasp_index + 1} (最高加权分数: {best_weighted_score:.2f})")
        
        # 应用Z轴深度偏移
        z_offset_matrix = np.eye(4)
        z_offset_matrix[2, 3] = Z_OFFSET
        T_B_from_G_final_with_offset = best_grasp_matrix @ z_offset_matrix
        
        target_pose = matrix_to_pose(T_B_from_G_final_with_offset)
        
        print(f"  > 最终目标位姿 (P_b): {np.round(target_pose, 4)}")

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