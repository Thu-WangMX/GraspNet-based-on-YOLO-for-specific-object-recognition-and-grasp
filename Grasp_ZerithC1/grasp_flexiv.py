#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GraspNet与Flexiv机器人集成控制脚本
功能：
1. 从RealSense相机获取实时图像。
2. 使用GraspNet检测抓取位姿。
3. 对位姿进行坐标系修正。
4. 控制Flexiv机器人执行抓取。
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
from Grasp_ZerithC1.trajectory_planner import Pose, TrajectoryPlanner
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# --- 机器人控制库 ---
from scipy.spatial.transform import Rotation as R
# 【修改】导入您提供的FlexivRobot控制类
from Grasp_ZerithC1.FlexivRobot import FlexivRobot

# ========================= 用户配置区 ========================= #

# ---- 机器人相关配置 ----
ROBOT_SN =  'Rizon4s-062958'
GRIPPER_NAME ='Flexiv-GN01'

ROBOT_SPEED = 0.1
ROBOT_ACC = 0.1

# ---- 夹爪相关配置 ----
GRIPPER_OPEN_WIDTH = 0.1  # 夹爪预抓取时张开的宽度(米)
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 10.0

# ---- 抓取流程配置 ----
PRE_GRASP_DISTANCE = 0.10 # 预抓取和抓取后抬升的距离(米)
Z_OFFSET = -0.05          # 抓取深度偏移量(米), 正值=更深

# ---- GraspNet模型相关配置 (保持不变) ----
CHECKPOINT_PATH = "/home/lrh/graspnet-baseline/checkpoints/checkpoint-rs.tar"
NUM_POINT = 20000
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01

# ---- 工作区掩码与相机内参 (保持不变) ----
WORKSPACE_MASK_PATH = "/home/lrh/graspnet-baseline/doc/example_data/workspace_mask_640x480.png"
INTRINSIC_MATRIX = np.array([
    [608.20166016, 0., 324.31015015],
    [0., 608.04638672, 243.1638031],
    [0., 0., 1.]
])
FACTOR_DEPTH = 1000.0

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

# ========================= 主函数执行区 ========================= #
def main():
    # ---- 1. 初始化 ----
    net = get_net()
    # 使用FlexivRobot类进行初始化
    robot = FlexivRobot(ROBOT_SN, GRIPPER_NAME, frequency = 100.0, remote_control=True, gripper_init=False)
    # 切换到PRIMITIVE模式以执行MoveL等指令
    robot.switch_PRIMITIVE_Mode()
    print("✓ 机器人初始化完成并进入PRIMITIVE模式。")

    try:
        while True:
            # --- 2. 感知 ---
            input("\n按回车键开始新一轮抓取检测...")
            color_rgb, depth = get_data_from_realsense()
            end_points, cloud = process_data(color_rgb, depth)
            gg = get_grasps(net, end_points)
            if COLLISION_THRESH > 0:
                gg = collision_detection(gg, cloud)

            if len(gg) == 0:
                print("✗ 未检测到有效的抓取位姿。")
                continue

            # --- 3. 决策: 选择最佳抓取并应用所有坐标系修正  ---
            gg.sort_by_score()
            best_grasp = gg[0]
            grasp_translation_in_camera = best_grasp.translation
            grasp_rotation_in_camera = best_grasp.rotation_matrix
            
            print(f"--------------- 检测到最佳抓取 ---------------")
            print(f"  - 分数: {best_grasp.score:.4f}, 目标宽度: {best_grasp.width:.4f} m")
            vis_grasps(gg, cloud)

            # --- 坐标系修正) ---
            # 修正A: 局部工具坐标系复合修正
            # rot_z_neg90 = R.from_euler('z', -90, degrees=True)
            # rot_x_neg90 = R.from_euler('x', -90, degrees=True)
            # final_fix_rotation = rot_z_neg90 * rot_x_neg90
            # final_fix_rotation =  rot_x_neg90
            # R_correct_combined = final_fix_rotation.as_matrix()
            

            rot_z_neg180 = R.from_euler('z', 180, degrees=True)
            rot_y_neg90 = R.from_euler('y', 90, degrees=True)
            final_fix_rotation = rot_y_neg90 * rot_z_neg180
            R_correct_combined = final_fix_rotation.as_matrix()
            grasp_rotation_in_camera = grasp_rotation_in_camera @ R_correct_combined
            
            T_C_from_G_raw = np.eye(4)
            T_C_from_G_raw[:3, :3] = grasp_rotation_in_camera
            T_C_from_G_raw[:3, 3] = grasp_translation_in_camera

            # 修正B: 手眼标定镜像修正
            # mirror_fix_X = np.diag([-1, 1, 1, 1])
            # T_C_from_G_corrected = mirror_fix_X @ T_C_from_G_raw

            # --- 4. 计算机器人最终需要运动到的多个位姿 (接口修改) ---
            # 【修改】使用Flexiv的read_pose获取当前位姿
            current_pos, current_euler_deg = robot.read_pose(Euler_flag=True)
            start_pose = Pose.from_xyz_rpy(current_pos, current_euler_deg)
            current_pose_vec = current_pos + current_euler_deg
            
            T_B_from_E = pose_euler_to_matrix(current_pose_vec)
            
            
            
            # print("自己计算出的:",T_B_from_E)
            # current_pos = robot.read_pose(Euler_flag=False)
            # T_B_from_E_2 = robot.quat2matrix(current_pos[3:])
            # print("号哥计算的:",T_B_from_E_2)


            T_B_from_C = T_B_from_E @ T_newTCP_from_camera
            T_B_from_G_grasp = T_B_from_C @ T_C_from_G_raw # 如果使用镜像修正，这里用 T_C_from_G_corrected
            

            # 修正C: Z轴深度偏移
            z_offset_matrix = np.eye(4)
            z_offset_matrix[2, 3] = Z_OFFSET
            T_B_from_G_final = z_offset_matrix @ T_B_from_G_grasp

            
            # --- 计算预抓取和抓取后位姿 ---
            pre_grasp_offset_matrix = np.eye(4)
            pre_grasp_offset_matrix[2, 3] = -PRE_GRASP_DISTANCE
            T_B_from_G_pre_grasp = T_B_from_G_final @ pre_grasp_offset_matrix
            
            # 将矩阵转换为机器人可执行的 [位置, 欧拉角] 格式
            final_pose_vec = matrix_to_pose_euler(T_B_from_G_final)
            final_pos, final_euler = final_pose_vec[:3].tolist(), final_pose_vec[3:].tolist()
            goal_pose = Pose.from_xyz_rpy(final_pos, final_euler)
            pre_grasp_pose_vec = matrix_to_pose_euler(T_B_from_G_pre_grasp)
            pre_pos, pre_euler = pre_grasp_pose_vec[:3].tolist(), pre_grasp_pose_vec[3:].tolist()
            
            print(f"  > 计算出的预抓取位姿: pos={np.round(pre_pos, 4)}, euler_deg={np.round(pre_euler, 4)}")
            print(f"  > 计算出的最终抓取位姿: pos={np.round(final_pos, 4)}, euler_deg={np.round(final_euler, 4)}")

            # --- 5. 安全确认并执行完整抓取序列 (接口修改) ---
            confirm = input("  ? 确认执行抓取序列? (y/n): ")
            if confirm.lower() == 'y':
                
                 # b. 移动到预抓取位置
                #print(f"  > 正在移动到预抓取位姿...")
                #robot.MoveL(pre_pos, pre_euler, speed=ROBOT_SPEED, acc=ROBOT_ACC)

                #robot.MovePTP(pre_pos, pre_euler, 10)
                
                print("  > 正在打开夹爪...")
                robot.Move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE)

                # c. 移动到最终抓取位置
                print(f"  > 正在下降到抓取位姿...")
                #robot.MoveL(final_pos, final_euler, speed=ROBOT_SPEED, acc=ROBOT_ACC)
                #robot.MovePTP(final_pos, final_euler, 10)


                trajectory_planner = TrajectoryPlanner()

                # plan a trajectory using SQP
                sqp_trajectory = trajectory_planner.plan_with_sqp(
                        start_pose,
                        goal_pose,
                        duration=10,
                        num_samples=10,
                        max_velocity=2,
                    )
                for pose in sqp_trajectory:
                    print("sqp_trajectory pose:", pose)

                # plan a trajectory using S Curve
                s_curve_trajectory = trajectory_planner.plan_with_s_curve(
                        start_pose,
                        goal_pose,
                        duration=10,
                        num_samples=10,
                    )
                for pose in s_curve_trajectory:
                    print("S-Curve trajectory pose:", pose)

                # if no robot instance is available, set robot=None
                # plan a trajectory but not execute
                trajectory_planner.execute_cartesian_trajectory(
                    robot=robot,  # Replace with actual robot instance if available
                    start_pose=start_pose,
                    goal_pose=goal_pose,
                    planner_name="s_curve",  # or "s_curve". s_curve is faster
                    speed=0.2,
                    acc=0.1,
                    duration=10.0,   # total trajectory time, define the time step of each point as duration/num_samples
                    num_samples=5,
                    zoneRadius="Z100"
                )
                print("111111111111111")
               
                # d. 闭合夹爪
                print(f"  > 正在闭合夹爪至宽度 {best_grasp.width:.4f} m...")
                robot.Move_gripper(0, GRIPPER_SPEED, GRIPPER_FORCE)
                
                # e. 垂直向上移动，抬起物体
                print(f"  > 正在抬升物体...")
                # robot.MoveL(pre_pos, pre_euler, speed=ROBOT_SPEED/2, acc=ROBOT_ACC)

                trajectory_planner.execute_cartesian_trajectory(
                    robot=robot,  # Replace with actual robot instance if available
                    start_pose=goal_pose,
                    goal_pose=start_pose,
                    planner_name="s_curve",  # or "s_curve". s_curve is faster
                    speed=0.2,
                    acc=0.1,
                    duration=10.0,   # total trajectory time, define the time step of each point as duration/num_samples
                    num_samples=5,
                    zoneRadius="Z100"
                )
                
                print("\n  ✓ 抓取序列执行完毕！")
            else:
                print("  - 移动已取消。")

    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        if 'robot' in locals():
            # 【修改】使用Flexiv的Stop方法
            robot.Stop()
            print("\n程序退出，机器人已停止。")

if __name__ == '__main__':
    main()