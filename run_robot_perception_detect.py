# run_robot_perception_detect.py

import cv2
import pyrealsense2 as rs
import numpy as np
import time
# 导入API类
from perception_api_detect import StainPerceptionAPI 

# --- 任务规划函数 (英文消息) ---
def plan_cleaning_tasks(detected_stains):
    """
    根据API返回的污渍检测列表，决定需要执行的清洁任务。

    Args:
        detected_stains (list[dict]): 从 StainPerceptionAPI.detect_stains 返回的列表。

    Returns:
        dict: 一个包含任务列表和英文描述性消息的字典。
    """
    if not detected_stains:
        return {
            "tasks": [],
            "message": "Area is clean. No task required."
        }

    found_classes = {stain['class_name'] for stain in detected_stains}
    tasks_to_perform = []
    message = ""
    contains_liquid = 'liquid' in found_classes
    contains_solid = 'solid' in found_classes

    if contains_liquid and contains_solid:
        tasks_to_perform = ['mopping', 'grasping']
        message = "Decision: Mopping and Grasping tasks required."
    elif contains_liquid:
        tasks_to_perform = ['mopping']
        message = "Decision: Mopping task required."
    elif contains_solid:
        tasks_to_perform = ['grasping']
        message = "Decision: Grasping task required."
    
    return {
        "tasks": tasks_to_perform,
        "message": message
    }


# --- 可视化函数 (使用标准cv2.putText) ---
def visualize_detections_and_plan(image, stains, task_plan):
    """
    一个辅助函数，用于在图像上绘制检测框、信息以及英文任务规划结果。
    """
    vis_image = image.copy()
    overlay = vis_image.copy()
    
    # 绘制检测到的污渍
    for stain in stains:
        class_name = stain['class_name']
        color = (255, 0, 0) if class_name == 'liquid' else (0, 255, 0) # BGR: Blue for liquid, Green for solid
        x1, y1, x2, y2 = stain['bbox']
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, -1)
        
        display_text = f"{class_name}"
        if stain['depth_info']:
            dist = stain['depth_info']['median_m']
            display_text += f" @ {dist:.2f}m"
        
        (text_w, text_h), _ = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(vis_image, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        cv2.putText(vis_image, display_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    vis_image = cv2.addWeighted(overlay, 0.3, vis_image, 0.7, 0)

    # 在画面底部显示英文任务规划结果
    plan_message = task_plan['message']
    (text_w, text_h), _ = cv2.getTextSize(plan_message, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
    cv2.rectangle(vis_image, (0, image.shape[0] - text_h - 15), (image.shape[0], image.shape[0]), (0,0,0), -1)
    cv2.putText(vis_image, plan_message, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    
    return vis_image

def main():
    MODEL_PATH = "/home/wmx/graspnet-baseline/robot-stain-perception/weights/best.pt"
    
    api = StainPerceptionAPI(MODEL_PATH)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        pipeline.start(config)
    except RuntimeError as e:
        print(f"❌ ERROR: Could not start camera: {e}")
        return
        
    align = rs.align(rs.stream.color)

    print("✅ Real-time perception and task planner is running (Displaying in English).")
    print("   - Live decision is shown on the video feed.")
    print("   - Detailed detections are printed to the console on every frame.")
    print("   - Press 'c', 'l', 's' for manual verification.")
    print("   - Press 'q' to quit.")
    
    last_message = ""
    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            if not depth_frame or not color_frame: continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data()) * depth_frame.get_units()
            
            detected_stains = api.detect_stains(color_image, depth_image)
            task_plan = plan_cleaning_tasks(detected_stains)

            # ###########################################################
            # ## 恢复的功能: 只要检测到污渍，就在控制台打印详细的实时日志
            # ###########################################################
            if detected_stains:
                print(f"\n--- Frame Detections (Found {len(detected_stains)} stains) ---")
                for i, stain in enumerate(detected_stains):
                    print(f"   Stain #{i+1}:")
                    print(f"    - Class:      {stain['class_name']}")
                    print(f"    - Confidence: {stain['confidence']:.3f}")
                    print(f"    - BBox (xyxy):{stain['bbox']}")
                    if stain['depth_info']:
                        print(f"    - Depth Info: Median={stain['depth_info']['median_m']}m, Mean={stain['depth_info']['mean_m']}m")
                    else:
                        print("    - Depth Info: Not available")
            # ###########################################################

            # 当任务决策发生变化时，也打印一条摘要（方便快速浏览）
            if task_plan['message'] != last_message:
                print(f"--- Task Update: {task_plan['message']} (Tasks: {task_plan['tasks']}) ---")
                last_message = task_plan['message']
            
            output_frame = visualize_detections_and_plan(color_image, detected_stains, task_plan)
            
            cv2.imshow("Robot Task Planner View", output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            
            # 手动验证功能保持不变
            verification_target = None
            if key == ord('c'): verification_target = 'all'
            elif key == ord('l'): verification_target = 'liquid'
            elif key == ord('s'): verification_target = 'solid'

            if verification_target:
                print(f"\n--- [Manual Verification] Checking for '{verification_target}' stains ---")
                verification = api.verify_cleanliness(color_image, target_class=verification_target)
                if verification['is_clean']:
                    print(f"✅ Verification Result: Clean (Target: {verification_target}).")
                else:
                    print(f"❌ Verification Result: NOT CLEAN! Found {len(verification['remaining_stains'])} target(s).")
                time.sleep(2)

            if key == ord('q'):
                break
    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()