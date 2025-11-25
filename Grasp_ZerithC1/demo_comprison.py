""" 
可视化best筛选的有效性
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import pyrealsense2 as rs
import torch
from graspnetAPI import GraspGroup
import time
import cv2 as cv

# 【新】从scipy库导入Rotation，用于姿态计算
from scipy.spatial.transform import Rotation as R

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

# ========================= 新增：与impro_grasp_flexiv.py同步的配置 =========================
# ---- 抓取决策权重 ----
W_GRASP_SCORE = 0.5
W_ORIENTATION_SCORE = 0.5

# ---- 抓取流程配置 ----
Z_OFFSET = -0.08

# ---- 手眼标定与工具TCP配置 ----
# T_newTCP_from_camera = np.array([[ 0.04685863, 0.99888241, -0.00618029, -0.11255110 ],
# [ -0.99886706, 0.04690751, 0.00801801, 0.02457545 ],
# [ 0.00829895, 0.00579757, 0.99994876, -0.20631265 ],
# [ 0.00000000, 0.00000000, 0.00000000, 1.00000000 ]])##TODO

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
# =====================================================================================

def get_net():
    # ... (此函数无变化) ...
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    net.eval()
    return net

def print_grasp_info(grasp):
    """
    打印GraspGroup对象中的详细抓取姿態数值。
    """
    # 按照分数排序并选择分数最高的5个进行打印

    """
    打印单个Grasp对象中的详细抓取姿態数值。
    """
    print(f"--------------- 抓取姿态数值 ---------------")
    
    # 直接打印传入的单个 grasp 对象的信息，不再需要排序和循环
    print(f"\n抓取详情:")
    # 1. 抓取分数 (越高越好)
    print(f"  - 抓取分数 (Score): {grasp.score:.4f}")
    
    # 2. 抓取宽度 (机械手需要张开的宽度，单位：米)
    print(f"  - 抓取宽度 (Width): {grasp.width:.4f} m")
    
    # 3. 抓取姿态的平移向量/位置 (x, y, z)
    print(f"  - 平移向量 (Translation): {grasp.translation}")
    
    # 4. 抓取姿态的旋转矩阵 (3x3 矩阵，描述了机械手的朝向)
    print(f"  - 旋转矩阵 (Rotation Matrix):\n{grasp.rotation_matrix}")
        
    print("---------------------------------------------------------")

def get_and_process_data(data_dir):
    # ... (此函数无变化) ...
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)
    align = rs.align(rs.stream.color)
    try:
        time.sleep(2)
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not depth_frame or not color_frame:
            raise RuntimeError("无法从 RealSense 获取数据帧。")
        depth = np.asanyarray(depth_frame.get_data())
        color_bgr = np.asanyarray(color_frame.get_data())
        color = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
    finally:
        pipeline.stop()
    workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask_640x480.png')))
    intrinsic = np.array([
        [608.20166016, 0., 324.31015015],
        [0., 608.04638672, 243.1638031],
        [0., 0., 1.]
    ])
    factor_depth = 1000.0
    camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    mask = (workspace_mask & (depth > 0))
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    return end_points, cloud

# --- 【新】从impro_grasp_flexiv.py移植的辅助函数 ---
def pose_euler_to_matrix(pose_euler_deg):
    matrix = np.eye(4)
    matrix[:3, :3] = R.from_euler('xyz', pose_euler_deg[3:], degrees=True).as_matrix()
    matrix[:3, 3] = pose_euler_deg[:3]
    return matrix

def get_grasps(net, end_points):
    # ... (此函数无变化) ...
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    # ... (此函数无变化) ...
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    # ... (此函数无变化) ...
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])


# --- 【核心修改】新的最佳抓取选择函数 ---
def select_best_grasp_consistent(gg, external_robot_pose_matrix):
    """
    严格按照 impro_grasp_flexiv.py 的逻辑，在模拟的机器人基坐标系下进行评分和选择。
    """
    if len(gg) == 0:
        return None

    # 1. 设置参考位姿和预计算变换矩阵
    T_B_from_E_start = external_robot_pose_matrix
    R_reference_in_base = T_B_from_E_start[:3, :3]
    T_B_from_C = T_B_from_E_start @ T_newTCP_from_camera
    z_offset_matrix = np.eye(4)
    z_offset_matrix[2, 3] = Z_OFFSET
    
    # flexiv ：修正抓取的旋转方向
    rot_z_neg180 = R.from_euler('z', 180, degrees=True)
    rot_y_neg90 = R.from_euler('y', 90, degrees=True)
    final_fix_rotation = rot_y_neg90 * rot_z_neg180
    R_correct_combined = final_fix_rotation.as_matrix()


    #ur：修正抓取的旋转方向
    rot_x_neg90 = R.from_euler('x', -90, degrees=True)
    R_correct_combined = rot_x_neg90.as_matrix()

    # 2. 遍历所有抓取，计算加权分数
    best_grasp = None
    max_weighted_score = -1.0

    for grasp in gg:
        # a. 获取相机系下的抓取位姿并应用局部修正
        grasp_rotation_corrected = grasp.rotation_matrix @ R_correct_combined
        T_C_from_G_raw = np.eye(4)
        T_C_from_G_raw[:3, :3] = grasp_rotation_corrected
        T_C_from_G_raw[:3, 3] = grasp.translation
        
        # b. 链式计算，得到在模拟机器人基坐标系下的最终抓取位姿
        T_B_from_G_grasp = T_B_from_C @ T_C_from_G_raw
        T_B_from_G_final = z_offset_matrix @ T_B_from_G_grasp
        
        # c. 提取候选姿态，并与参考姿态在基坐标系下比较
        R_candidate_in_base = T_B_from_G_final[:3, :3]
        R_relative = R_candidate_in_base @ R_reference_in_base.T
        angle_diff_rad = R.from_matrix(R_relative).magnitude()

        # d. 计算分数
        orientation_score = 1.0 - (angle_diff_rad / np.pi)
        graspnet_score = grasp.score
        weighted_score = (W_GRASP_SCORE * graspnet_score) + (W_ORIENTATION_SCORE * orientation_score)

        # e. 寻找最高分
        if weighted_score > max_weighted_score:
            max_weighted_score = weighted_score
            best_grasp = grasp

    print(f"\n--------------- 加权分数选出的最佳抓取 (机器人位姿一致) ---------------")
    if best_grasp is not None:
        print(f"  - 原始GraspNet分数: {best_grasp.score:.4f}")
        print(f"  - 最终加权总分: {max_weighted_score:.4f}")
        print_grasp_info(best_grasp)
    
    return best_grasp

def demo(data_dir):
    
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))

    # --- 【核心修改】在此处定义你从外部获取的机器人位姿 ---
    # 这是唯一需要手动修改的地方，用于模拟不同的机器人起始状态
    print("\n================== 手动输入区 ==================")
    # 格式: [x(m), y(m), z(m), rx(deg), ry(deg), rz(deg)] (内旋xyz欧拉角)
    external_robot_pose_vec = [0.5, 0.0, 0.4, 180, 0, 90] # 这是一个示例位姿
    T_B_from_E_start = pose_euler_to_matrix(external_robot_pose_vec)
    print(f"模拟的机器人起始位姿(向量): {external_robot_pose_vec}")
    print("--------------------------------------------------")

    # --- 方案1: 可视化 GraspNet 原始最高分抓取 ---
    print("\n================== 方案1: GraspNet 原始分数 ==================")
    # gg_original = gg.copy()
    gg.sort_by_score()
    print(f"  - 最高分: {gg[0].score:.4f}")
    input("按回车键，可视化【原始分数最高】的抓取...")
    vis_grasps(gg, cloud)

    # --- 方案2: 可视化与机器人位姿一致的加权分数“最佳”抓取 ---
    print("\n========== 方案2: GraspNet分数 + 机器人姿态分수 ==========")
    best_grasp_obj = select_best_grasp_consistent(gg, T_B_from_E_start)
    
    if best_grasp_obj is not None:
        # 创建一个新的 GraspGroup，只包含我们选出的这一个最佳抓取
        gg_selected = GraspGroup(best_grasp_obj.grasp_array[np.newaxis, :])
        input("按回车键，可视化【加权分数最高】的抓取...")
        vis_grasps(gg_selected, cloud)
    else:
        print("未能根据加权分数选出最佳抓取。")


if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)