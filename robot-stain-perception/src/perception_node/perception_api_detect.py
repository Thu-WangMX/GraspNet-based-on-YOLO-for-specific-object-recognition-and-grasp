# src/perception_node/perception_api_detect.py

import os
import cv2
import numpy as np
from ultralytics import YOLO
import sys

# 将项目根目录添加到Python路径，以确保可以找到realsense_api模块
# 这一步是为了让 `if __name__ == "__main__":` 部分能够独立运行
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

class StainPerceptionAPI:
    """
    一个封装了污渍感知功能的API类。
    它使用训练好的YOLOv8检测模型来识别液体和固体污渍，
    并输出目标在相机坐标系下的3D位置。
    """
    def __init__(self, model_weights_path):
        """
        初始化感知API，并立即加载YOLOv8模型。
        对象一旦成功创建，就处于随时可用的状态。

        Args:
            model_weights_path (str): 训练好的 .pt 模型文件路径。
        
        Raises:
            FileNotFoundError: 如果提供的模型权重文件路径不存在。
        """
        if not os.path.exists(model_weights_path):
            raise FileNotFoundError(f"模型权重文件未找到: {model_weights_path}")
            
        self.model_weights_path = model_weights_path
        
        print(f"🧠 正在加载YOLOv8模型: '{self.model_weights_path}'...")
        # 立即加载模型
        self.model = YOLO(self.model_weights_path)
        print("✅ StainPerceptionAPI 初始化成功并准备就绪。")

    def detect_stains(self, rgb_image, depth_image, camera_intrinsics, confidence_threshold=0.5):
        """
        在RGB图像中检测污渍，并计算每个污渍在相机3D坐标系下的位置。

        Args:
            rgb_image (np.ndarray): 输入的彩色图像 (BGR格式)。
            depth_image (np.ndarray): 与彩色图对齐的深度图像 (单位: 米, float类型)。
            camera_intrinsics (dict): 相机内参, 包含 'fx', 'fy', 'ppx', 'ppy'。
            confidence_threshold (float): 检测的置信度阈值。

        Returns:
            dict: 一个以类别为键的字典。每个键的值是一个列表，包含了该类别下所有
                  检测到的目标的详细信息。
        """
        # 确保内参字典包含所有必需的键
        required_keys = ['fx', 'fy', 'ppx', 'ppy']
        if not all(key in camera_intrinsics for key in required_keys):
            raise ValueError("camera_intrinsics 字典缺少必要的键 (fx, fy, ppx, ppy)。")

        detections_by_class = {'solid': [], 'liquid': []}
        results = self.model.predict(source=rgb_image, conf=confidence_threshold, verbose=False)
        result = results[0]

        if len(result.boxes) == 0:
            return detections_by_class

        fx, fy = camera_intrinsics['fx'], camera_intrinsics['fy']
        ppx, ppy = camera_intrinsics['ppx'], camera_intrinsics['ppy']

        for box in result.boxes:
            class_id = int(box.cls[0])
            class_name = self.model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            x1, y1, x2, y2 = bbox

            depth_roi = depth_image[y1:y2, x1:x2]
            valid_depth_values = depth_roi[depth_roi > 0]
            
            if valid_depth_values.size == 0: continue

            z_m = float(np.median(valid_depth_values))
            px, py = (x1 + x2) / 2, (y1 + y2) / 2
            x_m = (px - ppx) * z_m / fx
            y_m = (py - ppy) * z_m / fy
            
            stain_data = {
                'position_m': {'x': round(x_m, 4), 'y': round(y_m, 4), 'z': round(z_m, 4)},
                'confidence': round(confidence, 3),
                'bbox_pixels': bbox.tolist()
            }

            if class_name in detections_by_class:
                detections_by_class[class_name].append(stain_data)

        return detections_by_class

    def verify_cleanliness(self, rgb_image, area_of_interest=None, confidence_threshold=0.5, target_class='all'):
        """
        检查给定区域是否干净，可以指定目标类别。
        """
        image_to_check = rgb_image
        if area_of_interest:
            x1, y1, x2, y2 = map(int, area_of_interest)
            image_to_check = rgb_image[y1:y2, x1:x2]

        results = self.model.predict(source=image_to_check, conf=confidence_threshold, verbose=False)
        
        remaining_stains = []
        if len(results[0].boxes) > 0:
            for box in results[0].boxes:
                class_name = self.model.names[int(box.cls[0])]
                if target_class == 'all' or class_name == target_class:
                    remaining_stains.append({
                        "class_name": class_name,
                        "confidence": float(box.conf[0]),
                        "bbox": box.xyxy[0].cpu().numpy().astype(int).tolist()
                    })
        
        is_clean = len(remaining_stains) == 0
        
        return {
            "is_clean": is_clean,
            "remaining_stains": remaining_stains
        }

# --- 使用示例 (Usage Example with Live Camera) ---
if __name__ == "__main__":
    
    # 相对导入需要以模块方式运行
    from src.perception_node.realsense_api import RealsenseAPI
    
    def example_usage_with_live_camera():
        """
        一个演示如何使用 StainPerceptionAPI 类的自包含示例。
        此函数会连接到真实的RealSense相机来获取图像并进行测试。
        """
        print("\n" + "="*50)
        print("--- StainPerceptionAPI 真实相机使用示例 ---")
        print("="*50 + "\n")

        MODEL_PATH = "weights/multiclass_detector_best.pt"
        JSON_CONFIG_PATH = "utils/realsense-viewer.json"

        realsense_cam = None
        try:
            # --- 1. 实例化所有API ---
            detector = StainPerceptionAPI(model_weights_path=MODEL_PATH)
            realsense_cam = RealsenseAPI(config_json_path=JSON_CONFIG_PATH)
            
            # --- 2. 从相机捕获真实图像和相机内参 ---
            print("\n--- 正在从RealSense相机捕获真实图像及内参... ---")
            bgr_image, depth_image_m = realsense_cam.get_frames()
            intrinsics = realsense_cam.get_intrinsics()

            if bgr_image is None:
                print("❌ 错误: 未能从相机捕获到图像。")
                return
            
            print("✅ 真实图像捕获成功。")

            # --- 3. 演示调用 detect_stains 方法 ---
            print("\n--- 演示1: 调用 detect_stains() 并传入内参 ---")
            detected_stains_dict = detector.detect_stains(bgr_image, depth_image_m, intrinsics)

            if not detected_stains_dict['solid'] and not detected_stains_dict['liquid']:
                print("分析结果: 视野内没有发现污渍。")
            else:
                print("分析结果:")
                for class_name, stains in detected_stains_dict.items():
                    if stains:
                        print(f"  检测到 {len(stains)} 个 [{class_name}] 目标:")
                        for i, stain in enumerate(stains):
                            pos, conf, bbox = stain['position_m'], stain['confidence'], stain['bbox_pixels']
                            print(f"    - #{i+1}: Cam Coords (X,Y,Z): ({pos['x']:.3f}, {pos['y']:.3f}, {pos['z']:.3f}) m | BBox: {bbox} | Conf: {conf:.2f}")

            print("\n--- 演示2: 调用 verify_cleanliness() ---")
            
            # 演示1：检查所有类型的污渍
            verification_all = detector.verify_cleanliness(bgr_image, target_class='all')
            print("\n对捕获的图像进行【全面】清洁验证:")
            if verification_all['is_clean']:
                print("  - 结果: ✅ 干净")
            else:
                print(f"  - 结果: ❌ 未干净，仍发现 {len(verification_all['remaining_stains'])} 个污渍。")
                print(f"    - 细节: {verification_all['remaining_stains']}")

            # 演示2：只检查 'liquid' 类型的污渍
            verification_liquid = detector.verify_cleanliness(bgr_image, target_class='liquid')
            print("\n对捕获的图像【只检查液体】清洁验证:")
            if verification_liquid['is_clean']:
                print("  - 结果: ✅ 没有发现液体污渍。")
            else:
                print(f"  - 结果: ❌ 发现了 {len(verification_liquid['remaining_stains'])} 个液体污渍。")
                print(f"    - 细节: {verification_liquid['remaining_stains']}")

        except Exception as e:
            print(f"❌ 示例运行时发生错误: {e}")
        finally:
            if realsense_cam:
                realsense_cam.close()
            print("\n" + "="*50)
            print("--- 示例运行结束 ---")
            print("="*50)

    # 运行示例函数
    example_usage_with_live_camera()