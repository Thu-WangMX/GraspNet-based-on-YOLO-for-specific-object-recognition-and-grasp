#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import open3d as o3d
import torch
from PIL import Image
import pyrealsense2 as rs
import time

# --- GraspNet & Robot Imports (依赖项) ---
GRASPNET_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(GRASPNET_ROOT, 'models'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'dataset'))
sys.path.append(os.path.join(GRASPNET_ROOT, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnetAPI import GraspGroup
from Grasp_ZerithC1.trajectory_planner import Pose, TrajectoryPlanner
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image
from scipy.spatial.transform import Rotation as R
from FlexivRobot import FlexivRobot

# --- Mask Generation API Import (依赖项) ---
sys.path.append('/home/wmx/graspnet-baseline/robot-stain-perception/tools')
from grasp_mask_api import generate_and_save_grasp_mask

class GraspInController:
    """
    一个封装了从感知到抓取放置全流程的控制器。
    设计为由一个更高层的系统（如System_Run）来初始化和调用。
    """
    def __init__(self, robot_instance: FlexivRobot):
        """
        初始化GraspController。
        这个过程会加载昂贵的模型，所以控制器应该只被实例化一次。

        Args:
            robot_instance (FlexivRobot): 一个已经初始化好的FlexivRobot实例。
        """
        print("正在初始化 GraspController...")
        # ---- 1. 保存由外部传入的机器人实例 ----
        self.robot = robot_instance
        
        # ---- 2. 加载GraspNet模型 ----
        self.net = self._get_net()
        
        # ---- 3. 初始化轨迹规划器 ----
        self.trajectory_planner = TrajectoryPlanner()
        
        # ---- 4. 加载所有配置参数为类的属性 ----
        self._load_configs()
        
        print("✓ GraspController 初始化成功。")

    def _load_configs(self):
        """将所有配置参数加载为类的属性，方便管理。"""
        # Robot Configs
        self.ROBOT_SPEED = 0.1
        self.ROBOT_ACC = 0.1
        # Gripper Configs
        self.GRIPPER_OPEN_WIDTH = 0.1
        self.GRIPPER_SPEED = 0.1
        self.GRIPPER_FORCE = 10.0
        # Grasp Process Configs
        self.Z_OFFSET = -0.05
        # GraspNet Model Configs
        self.NUM_POINT = 20000
        self.COLLISION_THRESH = 0.01
        self.VOXEL_SIZE = 0.01
        # Camera Intrinsics
        self.INTRINSIC_MATRIX = np.array([[608.20166016, 0., 324.31015015], [0., 608.04638672, 243.1638031], [0., 0., 1.]])
        self.FACTOR_DEPTH = 1000.0
        # Hand-eye Calibration
        self.T_newTCP_from_camera = np.array([[ 0.04685863, 0.99888241, -0.00618029, -0.11255110 ],
                                              [ -0.99886706, 0.04690751, 0.00801801, 0.02457545 ],
                                              [ 0.00829895, 0.00579757, 0.99994876, -0.20631265 ],
                                              [ 0.00000000, 0.00000000, 0.00000000, 1.00000000 ]])
        # Key Poses
        self.binstart_pos = np.array([0.5167, 0.2731, -0.0299])
        self.binstart_euler_deg = np.array([-179.9356, -0.0347, -175.9093])
        self.binstart_pose = Pose.from_xyz_rpy(self.binstart_pos.tolist(), self.binstart_euler_deg.tolist())

        self.binstart_pos_up = np.array([0.5167, 0.2731, 0.1])
        self.binstart_euler_deg_up = np.array([-179.9356, -0.0347, -175.9093])
        self.binstart_pose_up = Pose.from_xyz_rpy(self.binstart_pos_up.tolist(), self.binstart_euler_deg_up.tolist())

        self.bin_pos = np.array([0.8506, 0.2717, -0.0651])
        self.bin_euler_deg = np.array([178.7773, 12.6751, -167.8946])
        self.bin_pose = Pose.from_xyz_rpy(self.bin_pos.tolist(), self.bin_euler_deg.tolist())
        # Mask Generation Configs
        self.YOLO_MODEL_PATH = "/home/wmx/graspnet-baseline/robot-stain-perception/weights/best.pt"
        self.OUTPUT_MASK_PATH = "/home/wmx/graspnet-baseline/mask.png"
        self.CONF_THRESHOLD = 0.5

    def execute_grasp_task(self):
        """
        这是该类的主要外部接口方法。
        执行一个完整的抓取任务，从移动到拍照位开始，到最终返回结束。
        """
        try:
            # --- 1. 生成最新的工作区掩码 ---
            print("--- 正在生成工作区掩码 ---")
            saved_mask_path = generate_and_save_grasp_mask(
                model_weights_path=self.YOLO_MODEL_PATH,#TODO
                output_path=self.OUTPUT_MASK_PATH,
                confidence_threshold=self.CONF_THRESHOLD
            )
            if not saved_mask_path:
                print("✗ 错误: 未能生成工作区掩码，抓取任务中止。")
                return False
            print(f"✓ 掩码已生成: {saved_mask_path}")

            # --- 2. 移动到预设的拍照位置 ---
            print("-> 正在移动到拍照起始位姿...")

            indoor_pos, indoor_euler_deg = self.robot.read_pose(Euler_flag=True)
            indoor_pose = Pose.from_xyz_rpy(indoor_pos, indoor_euler_deg)
            self.robot.MoveL(self.binstart_pos_up.tolist(), self.binstart_euler_deg_up.tolist(), speed=self.ROBOT_SPEED, acc=self.ROBOT_ACC)
            self.robot.MoveL(self.binstart_pos.tolist(), self.binstart_euler_deg.tolist(), speed=self.ROBOT_SPEED, acc=self.ROBOT_ACC)

            # --- 3. 执行核心的感知到放置的流程 ---
            success = self._perform_grasp_at_current_location(saved_mask_path,indoor_pose)

            if success:
                print("\n==================== 抓取任务成功 ====================")
            else:
                print("\n================== 抓取任务失败或被取消 ==================")
            
            return success

        except Exception as e:
            print(f"抓取任务中发生意外错误: {e}")
            return False

    def _perform_grasp_at_current_location(self, workspace_mask_path, indoor_pose):
        """
        在机器人当前位置执行感知、决策和完整的抓取-放置序列。
        这是一个私有方法，由 execute_grasp_task 调用。
        """
        # --- A. 感知 ---
        color_rgb, depth = self._get_data_from_realsense()
        end_points, cloud = self._process_data(color_rgb, depth, workspace_mask_path)
        gg = self._get_grasps(self.net, end_points)
        if self.COLLISION_THRESH > 0:
            gg = self._collision_detection(gg, cloud)

        if len(gg) == 0:
            print("✗ 未检测到有效的抓取位姿。")
            return False

        # --- B. 决策 ---
        gg.sort_by_score()
        best_grasp = gg[0]
        print(f"--------------- 检测到最佳抓取 ---------------")
        print(f"  - 分数: {best_grasp.score:.4f}, 目标宽度: {best_grasp.width:.4f} m")
        self._vis_grasps(gg, cloud)

        # --- C. 坐标系变换，计算最终抓取位姿 ---
        rot_z_180 = R.from_euler('z', 180, degrees=True)
        rot_y_90 = R.from_euler('y', 90, degrees=True)
        R_correct_combined = (rot_y_90 * rot_z_180).as_matrix()
        grasp_rotation_corrected = best_grasp.rotation_matrix @ R_correct_combined
        
        T_C_from_G_raw = np.eye(4)
        T_C_from_G_raw[:3, :3] = grasp_rotation_corrected
        T_C_from_G_raw[:3, 3] = best_grasp.translation

        current_pos, current_euler_deg = self.robot.read_pose(Euler_flag=True)
        T_B_from_E = self._pose_euler_to_matrix(np.concatenate([current_pos, current_euler_deg]))
        T_B_from_C = T_B_from_E @ self.T_newTCP_from_camera
        T_B_from_G_grasp = T_B_from_C @ T_C_from_G_raw
        
        z_offset_matrix = np.eye(4)
        z_offset_matrix[2, 3] = self.Z_OFFSET
        T_B_from_G_final = z_offset_matrix @ T_B_from_G_grasp

        final_pose_vec = self._matrix_to_pose_euler(T_B_from_G_final)
        final_pos, final_euler = final_pose_vec[:3].tolist(), current_euler_deg.tolist()
        goal_pose = Pose.from_xyz_rpy(final_pos, final_euler)

        goal_pos_up_array = np.array(final_pos)
        goal_pos_up_array[2] += 0.3
        goal_pose_up = Pose.from_xyz_rpy(goal_pos_up_array.tolist(), final_euler)

         # 垃圾桶上方的避障点
        bin_pos_up_array = self.bin_pos.copy()
        bin_pos_up_array[2] += 0.1
        self.bin_pose_up = Pose.from_xyz_rpy(bin_pos_up_array.tolist(), self.bin_euler_deg.tolist())



        print(f"  > 计算出的最终抓取位姿: pos={np.round(final_pos, 4)}, euler_deg={np.round(final_euler, 4)}")
        
        # --- D. 执行抓取-放置流程 ---
        confirm = input("  ? 确认执行抓取序列? (y/n): ")
        if confirm.lower() != 'y':
            print("  - 用户取消了移动。")
            return False
        
        print("  > 正在打开夹爪...")
        self.robot.Move_gripper(self.GRIPPER_OPEN_WIDTH, self.GRIPPER_SPEED, self.GRIPPER_FORCE)

        print("  > 正在移动到抓取位姿...")
        self.trajectory_planner.execute_cartesian_trajectory(
            robot=self.robot, start_pose=self.binstart_pos, goal_pose=goal_pose, 
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
            robot=self.robot, start_pose=self.bin_pose, goal_pose=indoor_pose,
            planner_name="sqp", speed=0.2, acc=0.1, duration=10.0, num_samples=5
        )
        
        print("\n  ✓ 完整抓取与放置序列执行完毕！")
        return True

    # --- 以下是所有从原脚本转换来的私有辅助方法 ---
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
    
    # ... 您脚本中的其他函数如 get_data_from_realsense, process_data, ...
    # ... 都应该像这样转换成 _get_data_from_realsense, _process_data ...
    # ... 并作为类的私有方法放在这里 ...


    ##调用示例###
    from grasp_in_controller import GraspInController
    self.grasp_handler = GraspInController(self.robot)
    def perform_grasp_task(self):
        """
        执行完整的地面垃圾抓取任务。
        """

        print("\n--- 开始执行马桶内侧垃圾抓取任务 ---")
        # 直接调用GraspController的公共接口方法
        success = self.grasp_handler.execute_grasp_task()
        
        if success:
            print("--- 卫生间内部垃圾抓取任务成功完成 ---")
        else:
            print("--- 地面垃圾抓取任务失败或未执行 ---")
        
        return success