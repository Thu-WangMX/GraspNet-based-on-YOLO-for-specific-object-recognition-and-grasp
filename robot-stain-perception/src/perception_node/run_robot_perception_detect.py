# src/perception_node/run_robot_perception_detect.py

import cv2
import time
# --- 关键改动: 导入新的相机API和我们自己的感知API ---
from realsense_api import RealsenseAPI
from perception_api_detect import StainPerceptionAPI 

# --- 任务规划和可视化函数 (与之前版本完全相同，无需改动) ---
def plan_cleaning_tasks(detected_stains):
    if not detected_stains: return {"tasks": [], "message": "Area is clean. No task required."}
    found_classes = {stain['class_name'] for stain in detected_stains}
    tasks_to_perform, message = [], ""
    contains_liquid = 'liquid' in found_classes
    contains_solid = 'solid' in found_classes
    if contains_liquid and contains_solid:
        tasks_to_perform, message = ['mopping', 'grasping'], "Decision: Mopping and Grasping tasks required."
    elif contains_liquid:
        tasks_to_perform, message = ['mopping'], "Decision: Mopping task required."
    elif contains_solid:
        tasks_to_perform, message = ['grasping'], "Decision: Grasping task required."
    return {"tasks": tasks_to_perform, "message": message}

def visualize_detections_and_plan(image, stains, task_plan):
    vis_image = image.copy()
    overlay = vis_image.copy()
    for stain in stains:
        class_name = stain['class_name']
        color = (255, 0, 0) if class_name == 'liquid' else (0, 255, 0)
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
    plan_message = task_plan['message']
    (text_w, text_h), _ = cv2.getTextSize(plan_message, cv2.FONT_HERSHEY_DUPLEX, 0.8, 1)
    cv2.rectangle(vis_image, (0, image.shape[0] - text_h - 15), (image.shape[0], image.shape[0]), (0,0,0), -1)
    cv2.putText(vis_image, plan_message, (10, image.shape[0] - 10), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
    return vis_image

def main():
    MODEL_PATH = "//home/wmx/graspnet-baseline/robot-stain-perception/weights/best.pt"  # 路径已更新为相对路径
    
    realsense_api = None
    try:
        # 1. 初始化YOLO感知API
        perception_api = StainPerceptionAPI(MODEL_PATH)
        
        # 2. 初始化新的RealsenseAPI
        realsense_api = RealsenseAPI()
        
        print("✅ 实时感知已运行。按 'q' 退出。")
        last_message = ""
        
        # 3. 主循环
        while True:
            # --- 核心改动: 使用新的API获取处理好的图像 ---
            bgr_image, depth_image_m = realsense_api.get_frames()

            if bgr_image is None or depth_image_m is None:
                print("未能从相机获取帧，跳过...")
                time.sleep(0.1)
                continue

            # 4. 调用感知API进行检测
            detected_stains = perception_api.detect_stains(bgr_image, depth_image_m)
            
            # 5. 任务规划
            task_plan = plan_cleaning_tasks(detected_stains)

            if task_plan['message'] != last_message:
                print(f"任务更新: {task_plan['message']} (需要执行: {task_plan['tasks']})")
                last_message = task_plan['message']
            
            # 6. 可视化
            output_frame = visualize_detections_and_plan(bgr_image, detected_stains, task_plan)
            cv2.imshow("Advanced Robot Perception", output_frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            # 手动验证功能已移除，以简化主循环，专注于演示核心流程

    except Exception as e:
        print(f"❌ 发生严重错误: {e}")
    finally:
        # 7. 确保相机被安全关闭
        if realsense_api:
            realsense_api.close()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()