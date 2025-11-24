#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
新增筛选best位姿过程
"""

import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
import pyrealsense2 as rs
import time

# --- GraspNet相关库 ---
GRASPNET_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from trajectory_planner import Pose, TrajectoryPlanner
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# --- 机器人控制库 ---
from scipy.spatial.transform import Rotation as R
# 【修改】导入您提供的FlexivRobot控制类
from FlexivRobot import FlexivRobot

# ========================= 用户配置区 ========================= #

# ---- 机器人相关配置 ----
ROBOT_SN = 'Rizon4R-062032'      
GRIPPER_NAME = 'GripperFlexivModbus' 

ROBOT_SPEED = 0.1
ROBOT_ACC = 0.1

# ---- 夹爪相关配置 ----
GRIPPER_OPEN_WIDTH = 0.08  # 夹爪预抓取时张开的宽度(米)
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 10.0

# ---- 抓取流程配置 ----
PRE_GRASP_DISTANCE = 0.10 # 预抓取和抓取后抬升的距离(米)
Z_OFFSET = -0.06          # 抓取深度偏移量(米), 正值=更深

# ---- GraspNet模型相关配置 (保持不变) ----
CHECKPOINT_PATH = "/home/lrh/graspnet-baseline/checkpoints/checkpoint-rs.tar"
NUM_POINT = 20000
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01

W_GRASP_SCORE = 0.5  # GraspNet原始抓取分数的权重 (0.0 ~ 1.0)
W_ORIENTATION_SCORE = 0.5 # 姿态相似度分数的权重 (0.0 ~ 1.0)

# ---- 工作区掩码与相机内参 (保持不变) ----
WORKSPACE_MASK_PATH = "/home/lrh/graspnet-baseline/doc/example_data/workspace_mask_640x480.png"
INTRINSIC_MATRIX = np.array([
    [608.20166016, 0., 324.31015015],
    [0., 608.04638672, 243.1638031],
    [0., 0., 1.]
])
FACTOR_DEPTH = 1000.0

WAY_CHOICE = 2

# # ---- 手眼标定与工具TCP配置  ----

# T_E630_from_E150 = np.array([
#     [1, 0, 0,  0.0],
#     [0, 1, 0,  0.0],
#     [0, 0, 1, -0.050],
#     [0, 0, 0,  1.0],
# ])
# T_newTCP_from_camera = T_E630_from_E150 @ T_E150_from_C

T_newTCP_from_camera = np.array([[ 0.04685863, 0.99888241, -0.00618029, -0.11255110 ],
[ -0.99886706, 0.04690751, 0.00801801, 0.02457545 ],
[ 0.00829895, 0.00579757, 0.99994876, -0.20631265 ],
[ 0.00000000, 0.00000000, 0.00000000, 1.00000000 ]])



bin_pos = [0.177 , 0.3055 ,0.3468]
bin_euler_deg = [177.6441  ,  1.1419 ,-122.3548]
bin_pose = Pose.from_xyz_rpy(bin_pos, bin_euler_deg )
# ========================= 函数定义区========================= #

# --- GraspNet 相关函数 (完全不变) ---
def get_net():
    net = GraspNet(input_feature_dim=0, num_view=NUM_VIEW, num_angle=12, num_depth=4,
                   cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01, 0.02, 0.03, 0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(CHECKPOINT_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print(f"-> 模型已加载: {CHECKPOINT_PATH} (epoch: {start_epoch})")
    net.eval()
    return net

def get_data_from_realsense():
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)

    try:
        print("等待相机稳定...")
        time.sleep(2)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("无法从 RealSense 获取数据帧。")
        depth = np.asanyarray(depth_frame.get_data())
        color_bgr = np.asanyarray(color_frame.get_data())

         # 1. 定义一个保存图像的文件夹路径
        save_directory = "/home/lrh/graspnet-baseline/captured_img"

        # 2. 确保这个文件夹存在，如果不存在则创建它
        os.makedirs(save_directory, exist_ok=True)

        # 3. 保存Color图像 (BGR -> RGB)
        #    Pillow库处理的是RGB格式，而RealSense输出的是BGR，所以需要转换一下通道顺序。
        color_rgb = color_bgr[:, :, ::-1]
        color_image = Image.fromarray(color_rgb)
        color_image.save(os.path.join(save_directory, "color.png"))

        # 4. 保存Depth图像
        #    深度图是16位的单通道图像，直接保存为PNG格式可以保留其完整精度。
        depth_image = Image.fromarray(depth)
        depth_image.save(os.path.join(save_directory, "depth.png"))

        print(f"图像已成功保存到 '{save_directory}' 文件夹中。")
        
        print("✓ 成功捕获图像。")
        return color_rgb, depth
    finally:
        pipeline.stop()

def process_data(color_rgb, depth):
    workspace_mask = np.array(Image.open(WORKSPACE_MASK_PATH).resize((640, 480), Image.NEAREST))
    color_normalized = color_rgb.astype(np.float32) / 255.0
    camera = CameraInfo(640.0, 480.0, INTRINSIC_MATRIX[0][0], INTRINSIC_MATRIX[1][1],
                        INTRINSIC_MATRIX[0][2], INTRINSIC_MATRIX[1][2], FACTOR_DEPTH)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color_normalized[mask]
    if len(cloud_masked) >= NUM_POINT:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=False)
    else:
        idxs = np.random.choice(len(cloud_masked), NUM_POINT, replace=True)
    cloud_sampled = torch.from_numpy(cloud_masked[idxs][np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points = {'point_clouds': cloud_sampled}
    vis_cloud = o3d.geometry.PointCloud()
    vis_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    vis_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    return end_points, vis_cloud

def get_grasps(net, end_points):
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    return GraspGroup(gg_array)

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(np.array(cloud.points), voxel_size=VOXEL_SIZE)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=COLLISION_THRESH)
    return gg[~collision_mask]

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    best_grasp = gg[:1]
    grippers = best_grasp.to_open3d_geometry_list()
    print("显示最佳抓取位姿... (按 'q' 关闭窗口)")
    o3d.visualization.draw_geometries([cloud, *grippers])

# --- 机器人坐标变换函数 (完全不变) ---
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

def pose_euler_to_matrix(pose_euler_deg):
    """
    【新函数】将 [x, y, z, rx_deg, ry_deg, rz_deg] 格式的位姿(欧拉角为角度)
    转换为 4x4 齐次变换矩阵。
    注意：使用 'xyz' 内旋顺序，它等价于 'ZYX' 外旋。
    """
    matrix = np.eye(4)
    # 使用 from_euler，并明确指定单位为度
    matrix[:3, :3] = R.from_euler('xyz', pose_euler_deg[3:], degrees=True).as_matrix()
    matrix[:3, 3] = pose_euler_deg[:3]
    return matrix

def matrix_to_pose_euler(matrix):
    """
    【新函数】将 4x4 齐次变换矩阵转换为 [x, y, z, rx_deg, ry_deg, rz_deg] 格式的位姿。
    返回的欧拉角为角度制。
    """
    pose = np.zeros(6)
    pose[:3] = matrix[:3, 3]
    # 使用 as_euler，并明确指定单位为度
    pose[3:] = R.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)
    return pose

# ========================= 【新增】两种独立的抓取选择函数 =========================

def select_grasp_by_weighted_score(gg, T_B_from_E_start, R_correct_combined, T_newTCP_from_camera, z_offset_matrix):
    """
    方法一：加权分数法。
    结合GraspNet分数和机器人姿态相似度，选择综合得分最高的抓取。
    """
    print(f"--- 共检测到 {len(gg)} 个抓取位姿 ---")
    if len(gg) == 0:
        return None

    best_grasp = None
    max_weighted_score = -1.0
    
    # 预计算参考旋转矩阵和相机到基座的变换
    R_reference_in_base = T_B_from_E_start[:3, :3]
    T_B_from_C = T_B_from_E_start @ T_newTCP_from_camera
    
    print("--- 使用“加权分数法”评估所有抓取 ---")
    for i, grasp in enumerate(gg):
        # a. 计算在机器人基坐标系下的最终抓取位姿矩阵
        grasp_rotation_corrected = grasp.rotation_matrix @ R_correct_combined
        T_C_from_G_raw = np.eye(4)
        T_C_from_G_raw[:3, :3] = grasp_rotation_corrected
        T_C_from_G_raw[:3, 3] = grasp.translation
        T_B_from_G_grasp = T_B_from_C @ T_C_from_G_raw
        T_B_from_G_final = z_offset_matrix @ T_B_from_G_grasp
        
        # b. 提取候选姿态，并与参考姿态比较
        R_candidate_in_base = T_B_from_G_final[:3, :3]
        R_relative = R_candidate_in_base @ R_reference_in_base.T
        angle_diff_rad = R.from_matrix(R_relative).magnitude()

        # c. 计算分数
        orientation_score = 1.0 - (angle_diff_rad / np.pi)
        graspnet_score = grasp.score
        weighted_score = (W_GRASP_SCORE * graspnet_score) + (W_ORIENTATION_SCORE * orientation_score)
        
        print(f"  - 评估抓取 #{i+1}: grasp_score={graspnet_score:.4f}, orientation_score={orientation_score:.4f} -> weighted_score={weighted_score:.4f}")

        # d. 寻找最高分
        if weighted_score > max_weighted_score:
            max_weighted_score = weighted_score
            best_grasp = grasp

    if best_grasp:
        print(f"  ✓ 已找到最佳抓取! 最高加权分: {max_weighted_score:.4f}")
    return best_grasp


def select_grasp_by_angle_filter(gg, angle_threshold_deg=30.0):
    """
    方法二：排序+角度筛选法。
    选择GraspNet分数最高，且抓取方向与相机Z轴夹角小于阈值的抓取。
    """
    print(f"--- 共检测到 {len(gg)} 个抓取位姿 ---")
    if len(gg) == 0:
        return None

    gg.sort_by_score()
    print(f"--- 使用“角度筛选法”评估 {len(gg)} 个已排序的抓取 ---")

    for i, grasp in enumerate(gg):
        # a. 计算当前抓取与相机Z轴的夹角
        camera_z_axis = np.array([0, 0, 1])
        grasp_approach_vector = grasp.rotation_matrix[:, 0]
        dot_product = np.dot(camera_z_axis, grasp_approach_vector)
        angle_rad = np.arccos(np.clip(dot_product, -1.0, 1.0))
        angle_deg = np.degrees(angle_rad)
        print(f"  - 评估抓取 #{i+1}: 分数={grasp.score:.4f}, 与Z轴夹角={angle_deg:.2f}度")

        # b. 检查夹角是否小于阈值
        if angle_deg < angle_threshold_deg:
            print(f"  ✓ 已找到最佳抓取! (第 {i+1} 个)，夹角 {angle_deg:.2f} < {angle_threshold_deg} 度。")
            return grasp # 找到第一个就直接返回

        print(f"✗ 未能找到一个与Z轴夹角小于 {angle_threshold_deg} 度的方案。")
        #return None





def main():
    # ---- 1. 初始化 ----
    net = get_net()
    robot = FlexivRobot(ROBOT_SN, GRIPPER_NAME, frequency = 100.0, remote_control=True, gripper_init=True)
    robot.switch_PRIMITIVE_Mode()
    robot.move_tcp_home()
    robot.Move_gripper(GRIPPER_OPEN_WIDTH)
    _init_tcp_position, _init_tcp_rpy = robot.read_pose(Euler_flag=True)
    _init_tcp_position[0] -= 0.07
    _init_tcp_position[1] -= 0.1
    _init_tcp_position[2] += 0.05
    robot.MoveL(_init_tcp_position, _init_tcp_rpy)
    print("✓ 机器人初始化完成并进入PRIMITIVE模式。")

    try:
        while True:

            # --- 3. 感知 ---
            print("\n--------------- 步骤1: 场景感知 ---------------")
            color_rgb, depth = get_data_from_realsense()
            end_points, cloud = process_data(color_rgb, depth)
            gg = get_grasps(net, end_points)
            if COLLISION_THRESH > 0:
                gg = collision_detection(gg, cloud)

            if len(gg) == 0:
                print("✗ 本轮未检测到有效的抓取位姿。")
                continue
            
            # --- 4. 获取机器人位姿并选择最佳抓取 ---
            print("\n--------------- 步骤2: 筛选最佳抓取 ---------------")
            current_pos, current_euler_deg = robot.read_pose(Euler_flag=True)
            start_pose = Pose.from_xyz_rpy(current_pos, current_euler_deg)
            start_pose_vec = current_pos + current_euler_deg
            T_B_from_E_start = pose_euler_to_matrix(start_pose_vec)
            
            best_grasp = None
            if WAY_CHOICE == 1:
                # 方法一需要的常量矩阵
                rot_z_neg180 = R.from_euler('z', 180, degrees=True)
                rot_y_neg90 = R.from_euler('y', 90, degrees=True)
                final_fix_rotation = rot_y_neg90 * rot_z_neg180
                R_correct_combined = final_fix_rotation.as_matrix()
                z_offset_matrix = np.eye(4); z_offset_matrix[2, 3] = Z_OFFSET
                
                best_grasp = select_grasp_by_weighted_score(gg, T_B_from_E_start, R_correct_combined, T_newTCP_from_camera, z_offset_matrix)
            elif WAY_CHOICE == 2:
                best_grasp = select_grasp_by_angle_filter(gg, angle_threshold_deg=40.0)
            else:
                print("✗ 无效选项，请重新开始。")
                continue

            if best_grasp is None:
                print("✗ 未能根据所选方法找到合适的抓取位姿。")
                continue

            # --- 5. 对选出的 best_grasp 进行位姿计算 ---
            print(f"\n--------------- 步骤3: 计算机器人目标位姿 ---------------")
            print(f"  - 最终选定抓取: 分数={best_grasp.score:.4f}, 宽度={best_grasp.width:.4f} m")
            vis_grasps(gg, cloud) # 可视化最佳抓取

            # a. 定义常量变换矩阵 (与方法一共享)
            rot_z_neg180 = R.from_euler('z', 180, degrees=True)
            rot_y_neg90 = R.from_euler('y', 90, degrees=True)
            final_fix_rotation = rot_y_neg90 * rot_z_neg180
            R_correct_combined = final_fix_rotation.as_matrix()
            T_B_from_C = T_B_from_E_start @ T_newTCP_from_camera
            z_offset_matrix = np.eye(4); z_offset_matrix[2, 3] = Z_OFFSET

            # b. 计算最终机器人目标位姿
            grasp_rotation_corrected = best_grasp.rotation_matrix @ R_correct_combined
            T_C_from_G_raw = np.eye(4)
            T_C_from_G_raw[:3, :3] = grasp_rotation_corrected
            T_C_from_G_raw[:3, 3] = best_grasp.translation
            T_B_from_G_grasp = T_B_from_C @ T_C_from_G_raw
            T_B_from_G_final = z_offset_matrix @ T_B_from_G_grasp

            # --- 6. 执行抓取 ---
            print("\n--------------- 步骤4: 执行抓取序列 ---------------")
            final_pose_vec = matrix_to_pose_euler(T_B_from_G_final)
            final_pos, final_euler = final_pose_vec[:3].tolist(), final_pose_vec[3:].tolist()
            print(" final_euler前", final_euler)
            if( final_euler[2] > -90 and final_euler[2] < 90 ):
                print(" final_euler[2] > -90 and final_euler[2] < 90 ")
                final_euler[2] -= 180
            elif( final_euler[2] > 90 ):
                print(" final_euler[2] > 90 ")
                final_euler[2] -= 360
            elif( final_euler[2] < -270 ):
                print(" final_euler[2] < -270 ")
                final_euler[2] += 180
            print(" final_euler后", final_euler)
            goal_pose = Pose.from_xyz_rpy(final_pos, final_euler)

            print(f"  > 计算出的最终抓取位姿: pos={np.round(final_pos, 4)}, euler_deg={np.round(final_euler, 4)}")
            
            confirm = input("  ? 确认执行抓取序列? (y/n): ")
            if confirm.lower() == 'y':
                # ... (机器人移动代码保持不变) ...
                print("  > 正在打开夹爪...")
                robot.Move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE)
                trajectory_planner = TrajectoryPlanner()
                trajectory_planner.execute_cartesian_trajectory(
                    robot=robot, start_pose=start_pose, goal_pose=goal_pose, planner_name="sqp",
                    speed=0.2, acc=0.1, duration=10.0, num_samples=5, zoneRadius="Z100")
                
                print(f"  > 正在闭合夹爪...")
                robot.Move_gripper(0, GRIPPER_SPEED, GRIPPER_FORCE)

                print(f"  > 正在回到初始位置...")
                trajectory_planner.execute_cartesian_trajectory(
                    robot=robot, start_pose=goal_pose, goal_pose=start_pose, planner_name="sqp",
                    speed=0.2, acc=0.1, duration=10.0, num_samples=5, zoneRadius="Z100")
                print("\n  ✓ 抓取序列执行完毕！")
                

                print(f"  > 正在前往垃圾桶...")
                trajectory_planner.execute_cartesian_trajectory(
                    robot=robot, start_pose=start_pose, goal_pose=bin_pose, planner_name="sqp",
                    speed=0.2, acc=0.1, duration=10.0, num_samples=5, zoneRadius="Z100")
                print("  > 正在打开夹爪...")
                robot.Move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE)

                print(f"  > 正在回到初始位置...")
                trajectory_planner.execute_cartesian_trajectory(
                    robot=robot, start_pose=bin_pose, goal_pose=start_pose, planner_name="sqp",
                    speed=0.2, acc=0.1, duration=10.0, num_samples=5, zoneRadius="Z100")

            else:
                print("  - 移动已取消。")

    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        if 'robot' in locals():
            robot.Stop()
            print("\n程序退出，机器人已停止。")

if __name__ == '__main__':
    main()