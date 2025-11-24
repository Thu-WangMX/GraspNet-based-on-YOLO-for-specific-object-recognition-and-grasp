""" Demo to show prediction results.
    Author: chenxi-wang
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


def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    print("-> loaded checkpoint %s (epoch: %d)"%(cfgs.checkpoint_path, start_epoch))
    # set model to eval mode
    net.eval()
    return net


def print_grasp_info(gg):
    """
    打印GraspGroup对象中的详细抓取姿態数值。
    """
    # 按照分数排序并选择分数最高的5个进行打印
    gg.sort_by_score()
    top_grasps = gg[:1] # 只打印前5个作为示例
    
    print(f"--------------- 抓取姿态数值 (前 {len(top_grasps)} 个) ---------------")
    
    for i, grasp in enumerate(top_grasps):
        print(f"\n抓取 #{i+1}:")
        
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

    # load data
    # color = np.array(Image.open(os.path.join(data_dir, 'color1.png')), dtype=np.float32) / 255.0
    # depth = np.array(Image.open(os.path.join(data_dir, 'depth1.png')))
    
    
    # 替换为从 RealSense 实时读取
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

        # 将捕获的帧转换为NumPy数组，并处理成与原函数相同的格式
        depth = np.asanyarray(depth_frame.get_data())
        color_bgr = np.asanyarray(color_frame.get_data())

        # 1. 定义一个保存图像的文件夹路径
        save_directory = "/home/wmx/graspnet-baseline/captured_img"

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
        # BGR -> RGB 并归一化到 [0, 1]
        color = color_bgr[:, :, ::-1].astype(np.float32) / 255.0
    finally:
        pipeline.stop()


    # ========================= END: 修改部分 =========================

    mask_path = '/home/wmx/graspnet-baseline/mask.png'
    resized_path = os.path.join(data_dir, "workspace_mask_640x480.png")  # 也可以覆盖原图，见注释

    # 1) 读取并resize到 640x480（W,H），用最近邻不混类别
    img = Image.open(mask_path)
    img_resized = img.resize((640, 480), Image.NEAREST)

    # 2) 保存（若想直接覆盖原图，把下一行的 resized_path 改成 mask_path）
    img_resized.save(resized_path)

    # 3) 使用resize后的图像生成 workspace_mask
    workspace_mask = np.array(Image.open(resized_path))
    
    # workspace_mask = np.array(Image.open(os.path.join(data_dir, 'workspace_mask.png')))

    #meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
    #intrinsic = meta['intrinsic_matrix']
    
    intrinsic = np.array([
    [608.20166016  , 0.     ,    324.31015015],
    [  0.   ,      608.04638672, 243.1638031 ],
    [  0.     ,      0.      ,     1.        ]
    ])
    
    #factor_depth = meta['factor_depth']

    factor_depth = 1000
    
    # generate cloud
    camera = CameraInfo(640.0, 480.0, intrinsic[0][0], intrinsic[1][1], intrinsic[0][2], intrinsic[1][2], factor_depth)
    cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
    
    # get valid points
    mask = ((workspace_mask>0) & (depth > 0))
    
    cloud_masked = cloud[mask]
    color_masked = color[mask]
    
    
    # sample points
    if len(cloud_masked) >= cfgs.num_point:
        idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
    else:
        idxs1 = np.arange(len(cloud_masked))
        idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
        idxs = np.concatenate([idxs1, idxs2], axis=0)
    cloud_sampled = cloud_masked[idxs]
    color_sampled = color_masked[idxs]

    # convert data
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
    cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
    end_points = dict()
    cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    cloud_sampled = cloud_sampled.to(device)
    end_points['point_clouds'] = cloud_sampled
    end_points['cloud_colors'] = color_sampled

    return end_points, cloud



def save_grasps(gg, filename="grasp_candidates.npz", top_n=10):
    """
    保存分数最高的N组抓取位姿到.npz文件。
    """
    gg.sort_by_score()
    top_grasps = gg[:top_n]

    translations = np.array([g.translation for g in top_grasps])
    rotation_matrices = np.array([g.rotation_matrix for g in top_grasps])
    scores = np.array([g.score for g in top_grasps])
    widths = np.array([g.width for g in top_grasps])

    np.savez(filename, 
             translations=translations, 
             rotation_matrices=rotation_matrices, 
             scores=scores,
             widths=widths)
    print(f"\n已成功将 {len(top_grasps)} 组最佳抓取位姿保存到 '{filename}'")

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def vis_grasps(gg, cloud):
    gg.nms()
    gg.sort_by_score()
    gg = gg[:1]
    grippers = gg.to_open3d_geometry_list()
    o3d.visualization.draw_geometries([cloud, *grippers])

def demo(data_dir):
    net = get_net()
    end_points, cloud = get_and_process_data(data_dir)
    
    gg = get_grasps(net, end_points)
    if cfgs.collision_thresh > 0:
        gg = collision_detection(gg, np.array(cloud.points))
    print_grasp_info(gg)
    save_grasps(gg, top_n=10)
    vis_grasps(gg, cloud)



if __name__=='__main__':
    data_dir = 'doc/example_data'
    demo(data_dir)
