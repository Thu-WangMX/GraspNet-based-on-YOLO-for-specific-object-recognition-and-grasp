# tools/create_grasp_mask.py

import os
import cv2
import numpy as np
from datetime import datetime
import sys

# 将src目录添加到Python路径中，以便可以导入我们自己的API模块
# 这是为了确保无论从哪里运行脚本，都能找到API文件
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.perception_node.realsense_api import RealsenseAPI
from perception_api_detect import StainPerceptionAPI

def generate_single_mask_on_demand(model_weights_path, output_dir="output_masks", confidence_threshold=0.5):
    """
    按需启动相机，捕获一帧图像，运行污渍检测，并生成一张包含所有
    检测结果的二值掩码图。

    Args:
        model_weights_path (str): 训练好的YOLOv8检测模型 (.pt) 文件路径。
        output_dir (str, optional): 保存生成的掩码图的目录。默认为 "output_masks"。
        confidence_threshold (float, optional): 检测的置信度阈值。

    Returns:
        str or None: 如果成功，返回生成的掩码文件的完整路径；否则返回None。
    """
    print("--- 按需生成掩码图脚本 ---")
    
    # 初始化API对象
    realsense_api = None
    perception_api = None
    
    try:
        # 1. 初始化YOLO感知API
        perception_api = StainPerceptionAPI(model_weights_path)

        # 2. 初始化并预热RealSense相机API
        realsense_api = RealsenseAPI()

        # 3. 捕获一张稳定、清晰的图像帧
        print("📷 正在捕获稳定的图像帧...")
        bgr_image, depth_image_m = realsense_api.get_frames()

        if bgr_image is None:
            print("❌ 错误：无法从相机捕获图像。")
            return None
        
        print("✅ 图像捕获成功。")
        
        # 4. 调用感知API进行污渍检测
        print("🚀 正在运行污渍检测...")
        detected_stains = perception_api.detect_stains(bgr_image, depth_image_m, confidence_threshold)

        # 5. 创建并合并掩码
        # 创建一个与原图等大的全黑图像作为基础掩码
        combined_mask = np.zeros(bgr_image.shape[:2], dtype=np.uint8)

        if not detected_stains:
            print("⚠️ 未检测到任何污渍。生成的掩码图将是全黑的。")
        else:
            print(f"🔍 检测到 {len(detected_stains)} 个污渍，正在合并掩码...")
            # 遍历所有检测到的污渍
            for stain in detected_stains:
                # 使用cv2.bitwise_or将每个污渍的矩形掩码“画”到最终的掩码图上
                # stain['mask'] 是一个与原图等大的、只有一个矩形是白色的图像
                combined_mask = cv2.bitwise_or(combined_mask, stain['mask'])
            print("✅ 掩码合并完成。")

        # 6. 保存最终的掩码图
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 创建一个带有时间戳的唯一文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"grasp_mask_{timestamp}.png"
        output_path = os.path.join(output_dir, output_filename)
        output_path = '/home/wmx/graspnet-baseline/mask.png'  
        
        print(f"💾 正在保存掩码图至: '{output_path}'")
        print(combined_mask.shape)
        cv2.imwrite(output_path, combined_mask)
        print("✅ 保存成功！")
        
        return output_path

    except Exception as e:
        print(f"❌ 发生了严重错误: {e}")
        return None
        
    finally:
        # 7. 确保相机被安全关闭，无论是否发生错误
        if realsense_api:
            realsense_api.close()
        print("--- 脚本执行完毕 ---")


if __name__ == "__main__":
    # --- 配置参数 ---
    # 指向你训练好的模型权重文件
    YOLO_MODEL_PATH = "/home/wmx/graspnet-baseline/robot-stain-perception/weights/best.pt" 
    
    # 定义掩码图的输出目录
    OUTPUT_DIRECTORY = "generated_masks"
    
    # 可以在这里调整检测的置信度阈值
    CONF_THRESHOLD = 0.5
    
    # --- 执行主函数 ---
    generate_single_mask_on_demand(
        model_weights_path=YOLO_MODEL_PATH,
        output_dir=OUTPUT_DIRECTORY,
        confidence_threshold=CONF_THRESHOLD
    )