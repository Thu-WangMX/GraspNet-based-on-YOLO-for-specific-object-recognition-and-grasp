#!/usr/bin/env python3
# -*- coding: utf-8 -*-
## 卫生间外部垃圾抓取控制器 封装好的类

import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
import pyrealsense2 as rs
import time

# --- 依赖项导入 ---
GRASPNET_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from trajectory_planner import Pose, TrajectoryPlanner
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from scipy.spatial.transform import Rotation as R
from FlexivRobot import FlexivRobot

sys.path.append('/home/wmx/graspnet-baseline/robot-stain-perception/tools')
from grasp_mask_api import generate_and_save_grasp_mask

class GraspOutController:
    """
    一个封装了“卫生间外部”垃圾抓取全流程的控制器。
    """
    def __init__(self, robot_instance: FlexivRobot):
        """
        初始化GraspOutController。

        Args:
            robot_instance (FlexivRobot): 一个已经初始化好的FlexivRobot实例。
        """
        print("正在初始化 GraspOutController (卫生间外部)...")
        self.robot = robot_instance
        self.net = self._get_net()
        self.trajectory_planner = TrajectoryPlanner()
        self._load_configs()
        print("✓ GraspOutController 初始化成功。")

    def _load_configs(self):
        """将所有配置参数加载为类的属性。"""
        # --- 通用配置 ---
        self.ROBOT_SPEED = 0.1
        self.ROBOT_ACC = 0.1
        self.GRIPPER_OPEN_WIDTH = 0.1
        self.GRIPPER_SPEED = 0.1
        self.GRIPPER_FORCE = 10.0
        self.Z_OFFSET = -0.05
        self.NUM_POINT = 20000
        self.COLLISION_THRESH = 0.01
        self.VOXEL_SIZE = 0.01
        self.INTRINSIC_MATRIX = np.array([[608.20166016, 0., 324.31015015], [0., 608.04638672, 243.1638031], [0., 0., 1.]])#TODO
        self.FACTOR_DEPTH = 1000.0
        self.T_newTCP_from_camera = np.array([[ 0.04685863, 0.99888241, -0.00618029, -0.11255110 ],
                                              [ -0.99886706, 0.04690751, 0.00801801, 0.02457545 ],
                                              [ 0.00829895, 0.00579757, 0.99994876, -0.20631265 ],
                                              [ 0.00000000, 0.00000000, 0.00000000, 1.00000000 ]])
        # --- 掩码生成配置 ---
        self.YOLO_MODEL_PATH = "/home/wmx/graspnet-baseline/robot-stain-perception/weights/best.pt"
        self.OUTPUT_MASK_PATH = "/home/wmx/graspnet-baseline/mask.png" # 使用不同的掩码文件名
        self.CONF_THRESHOLD = 0.5
        
        # --- 导航过程中的位姿，即抓取观测位姿 ---
        self.start_pos = np.array([0.5167, 0.2731, -0.0299]) #TODO: 根据外部场景更新
        self.start_euler_deg = np.array([-179.9356, -0.0347, -175.9093])
        self.start_pose = Pose.from_xyz_rpy(self.start_pos.tolist(), self.start_euler_deg.tolist())


        # 卫生间外部垃圾桶位置
        self.bin_pos = np.array([0.8506, 0.2717, -0.0651]) #TODO: 根据外部场景更新
        self.bin_euler_deg = np.array([178.7773, 12.6751, -167.8946])
        self.bin_pose = Pose.from_xyz_rpy(self.bin_pos.tolist(), self.bin_euler_deg.tolist())
        
        # 垃圾桶上方的避障点
        bin_pos_up_array = self.bin_pos.copy()
        bin_pos_up_array[2] += 0.1
        self.bin_pose_up = Pose.from_xyz_rpy(bin_pos_up_array.tolist(), self.bin_euler_deg.tolist())


    def execute_grasp_task(self):
        """
        执行一个完整的外部垃圾抓取任务。这是该类的主要外部接口。
        """
        try:
            print("--- 正在生成卫生间外部工作区掩码 ---")
            saved_mask_path = generate_and_save_grasp_mask(
                model_weights_path=self.YOLO_MODEL_PATH, #TODO 
                output_path=self.OUTPUT_MASK_PATH,
                confidence_threshold=self.CONF_THRESHOLD
            )
            if not saved_mask_path:
                print("✗ 错误: 未能生成工作区掩码，抓取任务中止。")
                return False
            print(f"✓ 掩码已生成: {saved_mask_path}")

            print("-> 正在移动到外部拍照起始位姿...")
            
            self.robot.MoveL(self.start_pos.tolist(), self.start_euler_deg.tolist(), speed=self.ROBOT_SPEED, acc=self.ROBOT_ACC)

            success = self._perform_grasp_at_current_location(saved_mask_path)
            
            return success

        except Exception as e:
            print(f"外部抓取任务中发生意外错误: {e}")
            return False

    def _perform_grasp_at_current_location(self, workspace_mask_path):
        """
        在当前位置执行感知、决策和完整的抓取-放置序列。
        """
        # 1. 感知
        color_rgb, depth = self._get_data_from_realsense() #TODO
        end_points, cloud = self._process_data(color_rgb, depth, workspace_mask_path)
        gg = self._get_grasps(self.net, end_points)
        if self.COLLISION_THRESH > 0:
            gg = self._collision_detection(gg, cloud)

        if len(gg) == 0:
            print("✗ 未检测到有效的抓取位姿。")
            return False

        # 2. 决策
        gg.sort_by_score()
        best_grasp = gg[0]
        print(f"--------------- 检测到最佳抓取 ---------------")
        print(f"  - 分数: {best_grasp.score:.4f}, 目标宽度: {best_grasp.width:.4f} m")
        self._vis_grasps(gg, cloud)

        # 3. 坐标系变换
        rot_z_180 = R.from_euler('z', 180, degrees=True)
        rot_y_90 = R.from_euler('y', 90, degrees=True)
        R_correct_combined = (rot_y_90 * rot_z_180).as_matrix()
        grasp_rotation_corrected = best_grasp.rotation_matrix @ R_correct_combined
        
        T_C_from_G_raw = np.eye(4)
        T_C_from_G_raw[:3, :3] = grasp_rotation_corrected
        T_C_from_G_raw[:3, 3] = best_grasp.translation

        current_pos, current_euler_deg = self.robot.read_pose(Euler_flag=True)
        self.start_pose = Pose.from_xyz_rpy(current_pos, current_euler_deg)
        T_B_from_E = self._pose_euler_to_matrix(np.concatenate([current_pos, current_euler_deg]))
        T_B_from_C = T_B_from_E @ self.T_newTCP_from_camera
        T_B_from_G_grasp = T_B_from_C @ T_C_from_G_raw
        
        z_offset_matrix = np.eye(4)
        z_offset_matrix[2, 3] = self.Z_OFFSET
        T_B_from_G_final = z_offset_matrix @ T_B_from_G_grasp

        final_pose_vec = self._matrix_to_pose_euler(T_B_from_G_final)
        final_pos, final_euler = final_pose_vec[:3].tolist(), current_euler_deg.tolist()
        goal_pose = Pose.from_xyz_rpy(final_pos, final_euler)
        
        # 抓取点上方的避障点
        goal_pos_up_array = np.array(final_pos)
        goal_pos_up_array[2] += 0.3
        goal_pose_up = Pose.from_xyz_rpy(goal_pos_up_array.tolist(), final_euler)

        print(f"  > 计算出的最终抓取位姿: pos={np.round(final_pos, 4)}, euler_deg={np.round(final_euler, 4)}")
        
        # 4. 执行抓取-放置流程
        confirm = input("  ? 确认执行抓取序列? (y/n): ")
        if confirm.lower() != 'y':
            print("  - 用户取消了移动。")
            return False

        print("  > 正在打开夹爪...")
        self.robot.Move_gripper(self.GRIPPER_OPEN_WIDTH, self.GRIPPER_SPEED, self.GRIPPER_FORCE)

        print("  > 正在移动到抓取位姿...")
        self.trajectory_planner.execute_cartesian_trajectory(
            robot=self.robot, start_pose=self.start_pose, goal_pose=goal_pose, 
            planner_name="s_curve", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )
    
        print(f"  > 正在闭合夹爪...")
        self.robot.Move_gripper(0, self.GRIPPER_SPEED, self.GRIPPER_FORCE)
        time.sleep(1)
        
        print("  > 正在抬升物体...")
        self.trajectory_planner.execute_cartesian_trajectory(
            robot=self.robot, start_pose=goal_pose, goal_pose=goal_pose_up,
            planner_name="s_curve", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )
        
        print("  > 正在前往垃圾桶上方...")
        self.trajectory_planner.execute_cartesian_trajectory(
            robot=self.robot, start_pose=goal_pose_up, goal_pose=self.bin_pose_up,
            planner_name="sqp", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )
        
        print("  > 正在下降到垃圾桶...")
        self.trajectory_planner.execute_cartesian_trajectory(
            robot=self.robot, start_pose=self.bin_pose_up, goal_pose=self.bin_pose,
            planner_name="sqp", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )

        print("  > 正在扔入垃圾桶...")
        self.robot.Move_gripper(self.GRIPPER_OPEN_WIDTH, self.GRIPPER_SPEED, self.GRIPPER_FORCE)
        time.sleep(1)
        
        print("  > 正在返回抓取起始点...")
        self.trajectory_planner.execute_cartesian_trajectory(
            robot=self.robot, start_pose=self.bin_pose, goal_pose=self.start_pose,
            planner_name="sqp", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )

        print("\n  ✓ 完整抓取与放置序列执行完毕！")
        return True

    # --- 以下是所有私有辅助方法 (与原脚本逻辑相同) ---
    def _get_net(self):
        CHECKPOINT_PATH = "/home/wmx/graspnet-baseline/checkpoints/checkpoint-rs.tar"
        net = GraspNet(input_feature_dim=0, num_view=300, num_angle=12, num_depth=4,
                       cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        checkpoint = torch.load(CHECKPOINT_PATH)
        net.load_state_dict(checkpoint['model_state_dict'])
        print(f"-> GraspNet模型已加载: {CHECKPOINT_PATH} (epoch: {checkpoint['epoch']})")
        net.eval()
        return net
    
    def _get_data_from_realsense(self):
        pipeline = rs.pipeline() #TODO
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
            if not depth_frame or not color_frame: raise RuntimeError("无法从 RealSense 获取数据帧。")
            depth = np.asanyarray(depth_frame.get_data())
            color_bgr = np.asanyarray(color_frame.get_data())
            save_directory = "/home/wmx/graspnet-baseline/captured_img_out"
            os.makedirs(save_directory, exist_ok=True)
            color_rgb = color_bgr[:, :, ::-1]
            Image.fromarray(color_rgb).save(os.path.join(save_directory, "color.png"))
            Image.fromarray(depth).save(os.path.join(save_directory, "depth.png"))
            return color_rgb, depth
        finally:
            pipeline.stop()

    def _process_data(self, color_rgb, depth, workspace_mask_path):
        workspace_mask = np.array(Image.open(workspace_mask_path).resize((640, 480), Image.NEAREST))
        color_normalized = color_rgb.astype(np.float32) / 255.0
        camera = CameraInfo(640.0, 480.0, self.INTRINSIC_MATRIX[0][0], self.INTRINSIC_MATRIX[1][1], self.INTRINSIC_MATRIX[0][2], self.INTRINSIC_MATRIX[1][2], self.FACTOR_DEPTH)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)
        mask = (workspace_mask > 0) & (depth > 0)
        cloud_masked = cloud[mask]
        color_masked = color_normalized[mask]
        if len(cloud_masked) >= self.NUM_POINT: idxs = np.random.choice(len(cloud_masked), self.NUM_POINT, replace=False)
        else: idxs = np.random.choice(len(cloud_masked), self.NUM_POINT, replace=True)
        cloud_sampled = torch.from_numpy(cloud_masked[idxs][np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points = {'point_clouds': cloud_sampled}
        vis_cloud = o3d.geometry.PointCloud()
        vis_cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        vis_cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        return end_points, vis_cloud

    def _get_grasps(self, net, end_points):
        with torch.no_grad():
            end_points = net(end_points)
            grasp_preds = pred_decode(end_points)
        gg_array = grasp_preds[0].detach().cpu().numpy()
        return GraspGroup(gg_array)

    def _collision_detection(self, gg, cloud):
        mfcdetector = ModelFreeCollisionDetector(np.array(cloud.points), voxel_size=self.VOXEL_SIZE)
        collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=self.COLLISION_THRESH)
        return gg[~collision_mask]

    def _vis_grasps(self, gg, cloud):
        gg.nms()
        gg.sort_by_score()
        grippers = gg[:1].to_open3d_geometry_list()
        o3d.visualization.draw_geometries([cloud, *grippers])

    def _pose_euler_to_matrix(self, pose_euler_deg):
        matrix = np.eye(4)
        matrix[:3, :3] = R.from_euler('xyz', pose_euler_deg[3:], degrees=True).as_matrix()
        matrix[:3, 3] = pose_euler_deg[:3]
        return matrix

    def _matrix_to_pose_euler(self, matrix):
        pose = np.zeros(6)
        pose[:3] = matrix[:3, 3]
        pose[3:] = R.from_matrix(matrix[:3, :3]).as_euler('xyz', degrees=True)
        return pose
    



###调用示例
    from grasp_out_controller import GraspOutController
    self.grasp_out_handler = GraspOutController(self.robot)
    def perform_out_grasp_task(self):
        """
        执行卫生间外部的地面垃圾抓取任务。
        """

        print("\n--- 开始执行卫生间外部垃圾抓取任务 ---")
        # 调用GraspOutController的公共接口方法
        success = self.grasp_out_handler.execute_grasp_task()
        
        if success:
            print("--- 卫生间外部垃圾抓取任务成功完成 ---")
        else:
            print("--- 卫生间外部垃圾抓取任务失败或未执行 ---")
        
        return success