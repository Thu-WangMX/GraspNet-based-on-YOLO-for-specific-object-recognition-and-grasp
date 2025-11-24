#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from pathlib import Path
from FlexivRobot import FlexivRobot

class WipeController:
    """
    一个封装了从文件加载轨迹并执行多点MoveJ擦拭动作的控制器。
    """
    def __init__(self, robot_instance: FlexivRobot):
        """
        初始化WipeController。

        Args:
            robot_instance (FlexivRobot): 一个已经初始化好的FlexivRobot实例。
        """
        print("正在初始化 WipeController...")
        self.robot = robot_instance
        self._load_configs()
        print("✓ WipeController 初始化成功。")

    def _load_configs(self):
        """将所有配置参数加载为类的属性。"""
        # 擦拭轨迹的JSON文件路径
        #self.TRAJECTORY_JSON_PATH = "/home/wmx/graspnet-baseline/poses_20251011_224117.json" #TODO
        
        # MoveJ多点移动的速度/持续时间参数
        self.MOVEJ_SPEED_PARAM = 10

    def execute_wipe_task(self,TRAJECTORY_JSON_PATH):
        """
        执行一个完整的擦拭任务。这是该类的主要外部接口。
        它会加载轨迹文件，并按顺序执行所有轨迹。
        """
        print("--- 开始执行擦拭任务 ---")
        try:
            # 1. 从JSON文件中提取所有轨迹的关节点
            print(f"正在从 {TRAJECTORY_JSON_PATH} 加载轨迹...")
            traj_joints = self._extract_joints_from_file(TRAJECTORY_JSON_PATH)

            # 检查是否成功加载到任何轨迹
            if not traj_joints or all(not v for v in traj_joints.values()):
                print(f"✗ 错误: 未能从轨迹文件 {TRAJECTORY_JSON_PATH} 中加载到任何有效的关节点。")
                return False
            
            # 2. 准备机器人
            self.robot.switch_PRIMITIVE_Mode()
            print("机器人已切换至元操作模式。")

            # 3. 按顺序循环执行所有轨迹
            print(f"已加载 {len(traj_joints)} 条轨迹，准备执行...")
            # 使用 sorted() 保证轨迹按 traj_0, traj_1, ... 的顺序执行
            for key in sorted(traj_joints.keys()):
                joints_list = traj_joints[key]
                if not joints_list:
                    print(f"[INFO] {key}: 轨迹为空，跳过。")
                    continue
                
                print(f"--> 正在执行 {key} (包含 {len(joints_list)} 个路径点)...")
                self.robot.MoveJ_multi_points(joints_list, self.MOVEJ_SPEED_PARAM)
                print(f"✓ {key} 执行完毕。")

            print("\n==================== 擦拭任务成功 ====================")
            return True

        except FileNotFoundError:
            print(f"✗ 致命错误: 轨迹文件未找到: {TRAJECTORY_JSON_PATH}")
            return False
        except Exception as e:
            print(f"擦拭任务中发生意外错误: {e}")
            return False

    def _extract_joints_from_file(self, json_path, max_traj_idx=20):
        """
        从指定的JSON文件中解析并提取所有轨迹的关节点数据。
        这是从原脚本移植过来的私有辅助方法。
        """
        json_path = Path(json_path)
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        traj_list = data.get("trajectories", [])
        out = {}

        for idx in range(min(len(traj_list), max_traj_idx + 1)):
            traj = traj_list[idx]
            poses = traj.get("poses", [])
            joints = []
            for p in poses:
                j = p.get("joint", None)
                if isinstance(j, (list, tuple)) and len(j) == 7:
                    joints.append([float(x) for x in j])
                else:
                    continue
            out[f"traj_{idx}"] = joints
            print(f"[INFO] {f'traj_{idx}'}: 收集到 {len(joints)} 个关节点条目")
        
        return out
    

###调用示例###
from wipe_controller import WipeController
self.wipe_handler = WipeController(self.robot)
def perform_wipe_task(self, TRAJECTORY_JSON_PATH):
        """
        执行桌面或指定区域的擦拭任务。
        """
        print("\n--- 开始执行擦拭任务流程 ---")
        TRAJECTORY_JSON_PATH = "/home/wmx/graspnet-baseline/poses_20251011_224117.json" #TODO,左侧刮
        success = self.wipe_handler.execute_wipe_task(TRAJECTORY_JSON_PATH)
        
        if success:
            print("--- 擦拭任务成功完成 ---")
        else:
            print("--- 擦拭任务失败或未执行 ---")
        ###导航
        TRAJECTORY_JSON_PATH = "/home/wmx/graspnet-baseline/poses_20251011_224117.json" #TODO,右侧刮
        success = self.wipe_handler.execute_wipe_task(TRAJECTORY_JSON_PATH)
        
        return success
