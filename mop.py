import numpy as np
import cv2
import time

# 假设以下文件位于同一目录下
from realsense_d435 import RealsenseAPI
from perception_api_detect import StainPerceptionAPI
from FlexivRobot import FlexivRobot

# ###########################################################
# ## 已修复: run_cleaning_task 函数
# ## 现在它通过函数参数接收所有必需的对象和数据
# ###########################################################
def run_cleaning_task(robot, stain, realsense_api, depth_image_mm, T_TCP_from_camera):
    """
    执行完整的机器人运动以清洁检测到的污渍。
    """
    try:
        bbox = stain["bbox"]
        print("步骤1: 正在将像素坐标转换为相机坐标系坐标...")
        px = int((bbox[0] + bbox[2]) / 2)
        py = int((bbox[1] + bbox[3]) / 2)
        
        # 使用传入的 realsense_api 和 depth_image_mm
        depth_in_mm = realsense_api.get_valid_depth(depth_image_mm, bbox[0], bbox[1], bbox[2], bbox[3]) * 1000.0

        # <--- 已修复: 此处的缩进错误
        if depth_in_mm <= 0:
            print("无法获取有效的深度值，清洁任务中断。")
            return

        pixel_list = [[px, py, depth_in_mm]]
        
        # <--- 已修复: 使用了正确的变量名 'realsense_api'
        coords_in_camera_frame = realsense_api.pixels_to_camera_coords(pixel_list, camera_index=0)
        
        if coords_in_camera_frame is None or coords_in_camera_frame[0][0] is None:
            print("坐标转换失败。")
            return
            
        position_in_camera = coords_in_camera_frame[0]
        print(f"✅ 在相机坐标系下的坐标 (m): {np.round(position_in_camera, 3)}")

        print("步骤2: 正在将相机坐标转换为机械臂基座坐标系坐标...")
        # <--- 已修复: 使用传入的 T_TCP_from_camera 矩阵
        position_in_base = robot.transform_camera_to_base(position_in_camera, T_TCP_from_camera)
        print(f"✅ 在基座坐标系下的坐标 (m): {np.round(position_in_base, 3)}")
        
        print("步骤3: 正在规划并移动机械臂至污渍中心...")
        current_position, current_euler_angles = robot.read_pose(Euler_flag=True)
        
        target_position = position_in_base.tolist()
        
        robot.MoveL(position=target_position, euler=current_euler_angles, speed=0.1, acc=0.1)
        
        print("✅ 已到达污渍中心位置。")

        print("\n步骤4: 开始执行沿X轴的平移清洁动作...")
        x_offset = 0.1  # 单位：米，10厘米
        
        target_pos_np = np.array(target_position)
        position_plus_x = (target_pos_np + np.array([x_offset, 0, 0])).tolist()
        position_minus_x = (target_pos_np - np.array([x_offset, 0, 0])).tolist()

        print(f"向上平移10cm至: {np.round(position_plus_x, 3)}")
        robot.MoveL(position=position_plus_x, euler=current_euler_angles, speed=0.08, acc=0.1)

        print(f"向下平移20cm至: {np.round(position_minus_x, 3)}")
        robot.MoveL(position=position_minus_x, euler=current_euler_angles, speed=0.08, acc=0.1)
        
        print(f"返回污渍中心: {np.round(target_position, 3)}")
        robot.MoveL(position=target_position, euler=current_euler_angles, speed=0.1, acc=0.1)
        
        print("✅ 清洁动作执行完毕！")

    except Exception as e:
        print(f"!!! 在执行清洁任务时发生错误: {e}")


def main_with_chassis_control():
    """
    主函数，整合了视觉感知和机器人控制。
    """
    # --- 1. 初始化所有模块 ---
    print("正在初始化所有模块...")
    robot = None # 初始化robot为None，以便在finally块中使用
    try:
        realsense_api = RealsenseAPI(height=480, width=640, fps=30)
        
        # !! 请确保此路径正确 !!
        model_path = "path/to/your/best.pt" 
        perception_api = StainPerceptionAPI(model_weights_path=model_path)
        
        # !! 请填入您真实的机器人参数 !!
        robot = FlexivRobot(robot_sn='Rizon4R-062032', gripper_name='GripperFlexivModbus', remote_control=True)
        robot.switch_PRIMITIVE_Mode()

    except Exception as e:
        print(f"初始化失败: {e}")
        return

    # 手眼标定矩阵
    T_TCP_from_camera = np.array([
        [1.0, 0.0, 0.0, 0.05],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]
    ])

    print("✅ 所有模块初始化成功！")

    # --- 2. 定义控制参数 ---
    TARGET_DISTANCE_M = 0.5
    STOP_THRESHOLD_M = 0.05

    try:
        print("\n进入自动检测和清洁模式... 将在距离目标约 {}m 处开始清洁。".format(TARGET_DISTANCE_M))
        
        # <--- 已修复: 为实现持续检测，添加了 while True 循环
        while True:
            # a. 捕获图像
            bgr_image, depth_image_mm = realsense_api.read_frame(camera_index=0)
            if bgr_image is None or depth_image_mm is None:
                time.sleep(0.1)
                continue

            depth_image_m = depth_image_mm.astype(np.float32) / 1000.0

            # b. 检测污渍
            detected_stains = perception_api.detect_stains(bgr_image, depth_image_m)
            
            # <--- 已修复: 取消注释此代码块以防止程序崩溃
            if not detected_stains:
                print("视野内未检测到污渍，继续检测...", end='\r')
                time.sleep(0.1) # 等待一会再检测
                continue
                
            # 以检测到的第一个污渍为目标
            stain = detected_stains[0]
            if stain["depth_info"] is None:
                print("检测到污渍但无有效深度，继续检测...", end='\r')
                time.sleep(0.1)
                continue

            stain_distance_m = stain["depth_info"]["median_m"]
            print(f"检测到污渍! 当前距离: {stain_distance_m:.3f} m", end='\r')

            # d. 判断是否应该开始清洁
            if abs(stain_distance_m - TARGET_DISTANCE_M) <= STOP_THRESHOLD_M:
                print(f"\n✅ 已到达目标位置 (距离 {stain_distance_m:.3f} m)，准备开始清洁。")
                
                #移动到清洁动作的初始状态
                #robot.Move_gripper()
                #robot.MoveL_multi_points()

                # <--- 已修复: 将所有必需的变量作为参数传递给函数
                run_cleaning_task(robot, stain, realsense_api, depth_image_mm, T_TCP_from_camera)
                
                #恢复到清洁动作的初始状态
                #robot.MoveL()
                #恢复到待机状态
                #robot.MoveL_multi_points()
                #robot.Move_gripper()

                print("\n任务完成，等待新的指令或重新启动程序。")
                break # 清洁完成后退出循环
            
            time.sleep(0.1) # 短暂延时以降低CPU占用

    except KeyboardInterrupt:
        print("\n程序被用户中断。")
    except Exception as e:
        print(f"\n程序运行期间发生错误: {e}")
    finally:
        # <--- 已修复: 移除了对 chassis 的调用
        if robot:
             # 在程序退出时停止机器人是一个好习惯
             robot.Stop()
        print("\n流程结束。")


if __name__ == "__main__":
    main_with_chassis_control()