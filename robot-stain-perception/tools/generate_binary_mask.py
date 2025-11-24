# generate_binary_mask.py

import os
import cv2
import numpy as np
from ultralytics import YOLO

def generate_binary_mask(image_path, model_weights_path, output_dir="output_masks", confidence_threshold=0.5):
    """
    加载一张图片，使用YOLOv8检测模型，然后生成一张黑白分明的二值掩码图。
    掩码图中，检测到的所有污渍区域为白色(255)，其余为黑色(0)。

    Args:
        image_path (str): 输入图片的路径。
        model_weights_path (str): 训练好的YOLOv8检测模型 (.pt) 文件路径。
        output_dir (str, optional): 保存生成掩码图的目录。默认为 "output_masks"。
        confidence_threshold (float, optional): 检测的置信度阈值。低于此阈值的检测将被忽略。
                                              默认为 0.5。

    Returns:
        tuple: (mask_image, output_mask_path)
               - mask_image (np.ndarray): 生成的二值掩码图 (如果未检测到污渍，则为全黑图)。
               - output_mask_path (str): 掩码图的保存路径。
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"输入图片未找到: {image_path}")
    if not os.path.exists(model_weights_path):
        raise FileNotFoundError(f"模型权重文件未找到: {model_weights_path}")

    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    print(f"🧠 正在加载YOLOv8模型: '{model_weights_path}'...")
    model = YOLO(model_weights_path)
    print("✅ 模型加载成功。")

    print(f"🖼️ 正在加载图片: '{image_path}'...")
    # 使用OpenCV加载图片
    original_image = cv2.imread(image_path)
    if original_image is None:
        raise ValueError(f"无法加载图片: {image_path}。请检查文件路径或文件损坏。")
    print("✅ 图片加载成功。")

    # 执行预测
    print(f"🚀 正在使用模型进行预测 (置信度阈值: {confidence_threshold})...")
    results = model.predict(source=original_image, conf=confidence_threshold, verbose=False)
    # YOLO.predict 方法返回一个 Results 对象列表，通常对于单张图片，列表只有一个元素
    result = results[0]
    print(f"✅ 预测完成。检测到 {len(result.boxes)} 个目标。")

    # 创建一个与原图大小相同的全黑图像作为掩码
    # 注意：掩码图是单通道的（灰度图），OpenCV中的0代表黑色
    mask_image = np.zeros(original_image.shape[:2], dtype=np.uint8)

    # 如果检测到任何目标
    if len(result.boxes) > 0:
        print("🔍 正在生成二值掩码...")
        # 遍历所有检测到的边界框
        for i, box in enumerate(result.boxes):
            class_id = int(box.cls[0])
            class_name = model.names[class_id]
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int) # 获取 [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox

            print(f"   - 检测到: {class_name}, 置信度: {confidence:.2f}, BBox: {bbox}")
            
            # 在掩码图上，将检测到的边界框区域填充为白色 (255)
            # 使用 cv2.rectangle 填充矩形区域
            cv2.rectangle(mask_image, (x1, y1), (x2, y2), 255, -1) # -1 表示填充整个矩形
        print("✅ 二值掩码生成成功。")
    else:
        print("⚠️ 未检测到任何污渍，将生成一张全黑掩码图。")


    # 构造输出文件名
    base_name = os.path.basename(image_path)
    name_without_ext = os.path.splitext(base_name)[0]
    output_mask_path = os.path.join(output_dir, f"{name_without_ext}_mask.png")

    # 保存生成的掩码图
    print(f"💾 正在保存二值掩码图至: '{output_mask_path}'...")
    cv2.imwrite(output_mask_path, mask_image)
    print("✅ 掩码图保存成功。")

    return mask_image, output_mask_path

if __name__ == "__main__":
    # --- 配置参数 ---
    # 你的模型权重文件路径
    YOLO_MODEL_PATH = "/home/hjj/hd10k_unet_stain_segmentation/runs/detect/multiclass_finetune_run/weights/best.pt" 
    # 待预测的图片路径（请替换为你的实际图片路径）
    INPUT_IMAGE_PATH = "/home/hjj/hd10k_unet_stain_segmentation/mix_yolo_annotation/images copy/color_20251007_122133_333270.png" 
    # 输出掩码图的目录
    OUTPUT_MASKS_DIRECTORY = "generated_masks"
    # 置信度阈值
    PREDICTION_CONF_THRESHOLD = 0.5 

    # --- 示例用法 ---
    try:
        # 调用函数生成并保存掩码图
        mask_result, saved_path = generate_binary_mask(
            image_path=INPUT_IMAGE_PATH,
            model_weights_path=YOLO_MODEL_PATH,
            output_dir=OUTPUT_MASKS_DIRECTORY,
            confidence_threshold=PREDICTION_CONF_THRESHOLD
        )

        # 可选：显示生成的掩码图 (如果需要，按任意键关闭窗口)
        # cv2.imshow("Generated Binary Mask", mask_result)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        
    except FileNotFoundError as e:
        print(f"❌ 错误: {e}")
    except ValueError as e:
        print(f"❌ 错误: {e}")
    except Exception as e:
        print(f"❌ 发生了意外错误: {e}")