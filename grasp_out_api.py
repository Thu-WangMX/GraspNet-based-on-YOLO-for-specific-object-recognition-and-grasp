#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
GraspNet与Flexiv机器人集成控制脚本（全流程函数封装版）
功能：
1. 控制机器人移动到预设的拍照位。
2. 从RealSense相机获取实时图像并保存。
3. 使用GraspNet检测抓取位姿。
4. 对位姿进行坐标系修正。
5. 控制Flexiv机器人执行一个完整的“抓取-移动-放置”任务。
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
# 当前这个文件所在的目录：/home/wmx/GraspNet-based-on-YOLO-for-specific-object-recognition-and-grasp
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# robot-stain-perception/tools 目录
TOOLS_DIR = os.path.join(ROOT_DIR, "robot-stain-perception", "tools")

# 加到 sys.path 里
if TOOLS_DIR not in sys.path:
    sys.path.append(TOOLS_DIR)

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from trajectory_planner import Pose, TrajectoryPlanner
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

# --- 自定义API ---
# 动态生成工作区掩码的API
sys.path.append('/home/wmx/graspnet-baseline/robot-stain-perception/tools')
from grasp_mask_api import generate_and_save_grasp_mask

# --- 机器人控制库 ---
from scipy.spatial.transform import Rotation as R
from FlexivRobot import FlexivRobot

# ========================= 用户配置区 (已从grasp_shuzhi.py更新) ========================= #

# ---- 机器人相关配置 ----
ROBOT_SN = 'Rizon4s-062958'
GRIPPER_NAME = 'Flexiv-GN01'

ROBOT_SPEED = 0.1
ROBOT_ACC = 0.1

# ---- 夹爪相关配置 ----
GRIPPER_OPEN_WIDTH = 0.1  # 夹爪预抓取时张开的宽度(米)
GRIPPER_SPEED = 0.1
GRIPPER_FORCE = 10.0

# ---- 抓取流程配置 ----
Z_OFFSET = -0.032       # 抓取深度偏移量(米), 正值=更深

# ---- GraspNet模型相关配置 ----
CHECKPOINT_PATH = "/home/wmx/GraspNet-based-on-YOLO-for-specific-object-recognition-and-grasp/checkpoints/checkpoint-rs.tar"
NUM_POINT = 20000
NUM_VIEW = 300
COLLISION_THRESH = 0.01
VOXEL_SIZE = 0.01

# ---- 相机内参 ----
INTRINSIC_MATRIX = np.array([
    [608.20166016, 0., 324.31015015],
    [0., 608.04638672, 243.1638031],
    [0., 0., 1.]
])
FACTOR_DEPTH = 1000.0

# ---- 手眼标定矩阵 ----
T_newTCP_from_camera = np.array([[ 0.04685863, 0.99888241, -0.00618029, -0.11255110 ],
[ -0.99886706, 0.04690751, 0.00801801, 0.02457545 ],
[ 0.00829895, 0.00579757, 0.99994876, -0.20631265 ],
[ 0.00000000, 0.00000000, 0.00000000, 1.00000000 ]])

# ---- 预设关键位姿 ----
# 抓取起始位姿 (拍照位)
start_pos = np.array([0.5127, -0.0077, -0.0288]) #TODO


start_euler_deg = np.array([[-179.4118  ,  3.4282, -177.0679 ]])
start_pose = Pose.from_xyz_rpy(start_pos.tolist(), start_euler_deg.tolist())

# 抓取起始位姿的抬起位姿 
binstart_pos_up = np.array([0.5167, 0.2731, 0.1]) #TODO
binstart_euler_deg_up = np.array([-1.799356e+02, -3.470000e-02, -1.759093e+02])
binstart_pose_up = Pose.from_xyz_rpy(binstart_pos_up.tolist(), binstart_euler_deg_up.tolist())

# 垃圾桶(放置位)位姿
bin_pos = np.array([0.8506, 0.2717, -0.0651]) #TODO
bin_euler_deg = np.array([178.7773, 12.6751, -167.8946])
bin_pose = Pose.from_xyz_rpy(bin_pos.tolist(), bin_euler_deg.tolist())

bin_pose_up = bin_pose.copy()
#bin_pose_up[2] += 0.1  # 放置点上抬

# ========================= 函数定义区 ========================= #

# --- GraspNet 相关函数 ---
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

# 【已更新】增加了保存图像的功能
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
        
        # 保存图像
        save_directory = "/home/wmx/graspnet-baseline/captured_img"
        os.makedirs(save_directory, exist_ok=True)
        color_rgb = color_bgr[:, :, ::-1]
        Image.fromarray(color_rgb).save(os.path.join(save_directory, "color.png"))
        Image.fromarray(depth).save(os.path.join(save_directory, "depth.png"))
        print(f"图像已成功保存到 '{save_directory}' 文件夹中。")
        
        print("✓ 成功捕获图像。")
        return color_rgb, depth
    finally:
        pipeline.stop()

# 【已更新】修正了mask的处理逻辑
def process_data(color_rgb, depth, workspace_mask_path):
    workspace_mask = np.array(Image.open(workspace_mask_path).resize((640, 480), Image.NEAREST))
    color_normalized = color_rgb.astype(np.float32) / 255.0
    camera = CameraInfo(640.0, 480.0, INTRINSIC_MATRIX[0][0], INTRINSIC_MATRIX[1][1],
                        INTRINSIC_MATRIX[0][2], INTRINSIC_MATRIX[1][2], FACTOR_DEPTH)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask = (workspace_mask > 0) & (depth > 0) # 使用 grasp_shuzhi.py 的逻辑
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
    grippers = gg[:1].to_open3d_geometry_list()
    print("显示最佳抓取位姿... (按 'q' 关闭窗口)")
    o3d.visualization.draw_geometries([cloud, *grippers])

def pose_euler_to_matrix(pose_euler_deg):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler('xyz', pose_euler_deg[3:], degrees=True).as_matrix()
    matrix[:3, 3] = pose_euler_deg[:3]
    return matrix

def matrix_to_pose_euler(matrix):
    pose = np.zeros(6)
    pose[:3] = matrix[:3, 3]
    pose[3:] = R.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)
    return pose

# ========================= 核心功能封装函数 (已完全重构) ========================= #

def perform_complete_grasp_cycle(net, robot, trajectory_planner, workspace_mask_path):
    """
    执行一个完整的抓取周期：感知 -> 决策 -> 抓取 -> 移动 -> 放置 -> 返回。

    Args:
        net (GraspNet): 已加载的GraspNet模型。
        robot (FlexivRobot): 机器人控制实例。
        trajectory_planner (TrajectoryPlanner): 轨迹规划器实例。
        workspace_mask_path (str): 工作区掩码文件的路径。

    Returns:
        bool: 如果整个序列成功执行，则返回 True，否则返回 False。
    """
    # --- 1. 感知 (在当前位置) ---
    color_rgb, depth = get_data_from_realsense()
    end_points, cloud = process_data(color_rgb, depth, workspace_mask_path)
    gg = get_grasps(net, end_points)
    if COLLISION_THRESH > 0:
        gg = collision_detection(gg, cloud)

    if len(gg) == 0:
        print("✗ 未检测到有效的抓取位姿。")
        return False

    # --- 2. 决策 (使用 grasp_shuzhi.py 的简化逻辑) ---
    gg.sort_by_score()
    best_grasp = gg[0]
    
    print(f"--------------- 检测到最佳抓取 ---------------")
    print(f"  - 分数: {best_grasp.score:.4f}, 目标宽度: {best_grasp.width:.4f} m")
    vis_grasps(gg, cloud)

    # --- 3. 坐标系变换，计算最终抓取位姿 ---
    # a. 局部工具坐标系复合修正
    rot_z_180 = R.from_euler('z', 180, degrees=True)
    rot_y_90 = R.from_euler('y', 90, degrees=True)
    R_correct_combined = (rot_y_90 * rot_z_180).as_matrix()
    
    grasp_rotation_corrected = best_grasp.rotation_matrix @ R_correct_combined
    
    T_C_from_G_raw = np.eye(4)
    T_C_from_G_raw[:3, :3] = grasp_rotation_corrected
    T_C_from_G_raw[:3, 3] = best_grasp.translation
    print('来自graspnet的位置:', T_C_from_G_raw[:3, 3])
    # T_C_from_G_raw[:2, 3] = center_pixel[0]
    # print('来自yolo的位置:', T_C_from_G_raw[:3, 3])

    # b. 计算相对于基座的抓取位姿
    current_pos, current_euler_deg = robot.read_pose(Euler_flag=True)
    current_pos = np.array(current_pos, dtype=float)
    current_euler_deg = np.array(current_euler_deg, dtype=float)
    T_B_from_E = pose_euler_to_matrix(np.concatenate([current_pos, current_euler_deg]))
    T_B_from_C = T_B_from_E @ T_newTCP_from_camera
    T_B_from_G_grasp = T_B_from_C @ T_C_from_G_raw
    
    # c. Z轴深度偏移
    z_offset_matrix = np.eye(4)
    z_offset_matrix[2, 3] = Z_OFFSET
    T_B_from_G_final = z_offset_matrix @ T_B_from_G_grasp

    # 将最终矩阵转换为机器人可执行的Pose对象
    final_pose_vec = matrix_to_pose_euler(T_B_from_G_final)
    # 强制使用当前工具姿态，只改变位置，以增加稳定性
    final_pos, final_euler = final_pose_vec[:3].tolist(), current_euler_deg.tolist()
    goal_pose = Pose.from_xyz_rpy(final_pos, final_euler)
    
    
    # goal_pose_up = goal_pose.copy()
    # goal_pose_up[2] += 0.1  # 抓取点上抬10cm以避障
    print(f"  > 计算出的最终抓取位姿: pos={np.round(final_pos, 4)}, euler_deg={np.round(final_euler, 4)}")
    
    # --- 4. 执行完整的抓取-放置流程 ---
    confirm = input("  ? 确认执行抓取序列? (y/n): ")
    if confirm.lower() == 'y':
        print("  > 正在打开夹爪...")
        robot.Move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE)

        print("  > 正在移动到抓取位姿...")
        trajectory_planner.execute_cartesian_trajectory(
            robot=robot, start_pose=start_pose, goal_pose=goal_pose,
            planner_name="s_curve", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )
    
        print(f"  > 正在闭合夹爪...")
        robot.Move_gripper(0, GRIPPER_SPEED, GRIPPER_FORCE)
        time.sleep(1) # 等待夹爪闭合稳定
        
        print("  > 正在抬升物体...")
        trajectory_planner.execute_cartesian_trajectory(
            robot=robot, start_pose=goal_pose, goal_pose=start_pose,
            planner_name="s_curve", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )
        robot.Move_gripper(0.1, GRIPPER_SPEED, GRIPPER_FORCE)
        time.sleep(1) # 等待夹爪闭合稳定
        # print("  > 正在前往垃圾桶...")
        # trajectory_planner.execute_cartesian_trajectory(
        #     robot=robot, start_pose=goal_pose_up, goal_pose=bin_pose_up,
        #     planner_name="sqp", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        # )



        # print("  > 正在返回抓取起始点...")
        # trajectory_planner.execute_cartesian_trajectory(
        #     robot=robot, start_pose=bin_pose_up, goal_pose=bin_pose,
        #     planner_name="sqp", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        # )

        # print("  > 正在扔入垃圾桶...")
        # robot.Move_gripper(GRIPPER_OPEN_WIDTH, GRIPPER_SPEED, GRIPPER_FORCE)
        # time.sleep(1) # 等待夹爪张开
        
        # trajectory_planner.execute_cartesian_trajectory(
        #     robot=robot, start_pose=bin_pose, goal_pose=start_pose,
        #     planner_name="sqp", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        # )


        print("\n  ✓ 完整抓取与放置序列执行完毕！")
        return True
    else:
        print("  - 移动已取消。")
        return False


# ========================= 主函数执行区 ========================= #
def Grasp_api(workspace_mask_path):
    """主API函数，负责初始化和循环执行抓取任务。"""
    # ---- 1. 初始化 ----
    net = get_net()
    robot = FlexivRobot(ROBOT_SN, GRIPPER_NAME, frequency=100.0, remote_control=True, gripper_init=False)
    robot.switch_PRIMITIVE_Mode()
    trajectory_planner = TrajectoryPlanner()
    print("✓ 所有模块初始化完成。")

    try:
        while True:
            input("\n按回车键开始新一轮完整抓取周期...")
            
            # --- 2. 移动到预设的拍照位置 ---
            print("-> 正在移动到拍照起始位姿...")
            # robot.MoveL(binstart_pos_up.tolist(), binstart_euler_deg_up.tolist(), speed=ROBOT_SPEED, acc=ROBOT_ACC)
            # robot.MoveL(start_pos.tolist(), start_euler_deg.tolist(), speed=ROBOT_SPEED, acc=ROBOT_ACC)

            # --- 3. 调用核心函数，执行从感知到放置的全过程 ---
            success = perform_complete_grasp_cycle(net, robot, trajectory_planner, workspace_mask_path)

            # --- 4. 根据结果打印信息 ---
            if success:
                print("\n==================== 任务成功 ====================")
            else:
                print("\n================== 任务未完成或被取消 ==================")

    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    finally:
        if 'robot' in locals():
            robot.Stop()
            print("\n程序退出，机器人已停止。")

if __name__ == '__main__':
    # --- 步骤1: 动态生成最新的工作区掩码 ---
    YOLO_MODEL_PATH = "/home/wmx/GraspNet-based-on-YOLO-for-specific-object-recognition-and-grasp/yolo8l_batch8_run1.pt"
    OUTPUT_MASK_PATH = "/home/wmx/GraspNet-based-on-YOLO-for-specific-object-recognition-and-grasp/generated_masks/mask.png"
    CONF_THRESHOLD = 0.5

    print("--- 正在生成工作区掩码 ---")
    saved_mask_path, completed_grasp  = generate_and_save_grasp_mask(
        model_weights_path=YOLO_MODEL_PATH,
        output_path=OUTPUT_MASK_PATH,
        confidence_threshold=CONF_THRESHOLD
    )

    if saved_mask_path and not completed_grasp:
        print(f"✓ 掩码已生成: {saved_mask_path}")
        # --- 步骤2: 启动抓取主程序 ---
        Grasp_api(saved_mask_path)
    elif completed_grasp:
        print("✓ 工作区已清洁，无需抓取。")
    else:
        print("✗ 错误：未能生成工作区掩码，无法启动抓取程序。")