import os
import cv2
import numpy as np
import sys
from datetime import datetime

# 将src目录添加到Python路径中，以便可以导入我们自己的API模块
# 注意：您可能需要根据您的项目结构调整此路径
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.perception_node.realsense_api import RealsenseAPI
from perception_api_detect import StainPerceptionAPI


def generate_and_save_grasp_mask(model_weights_path,
                                 output_path="/home/wmx/graspnet-baseline/mask.png",
                                 confidence_threshold=0.5):
    """
    【修正版】
    Capture a frame, detect stains, build a binary mask for 'solid' class only,
    and return (mask_path, grasp_pos_cam). The grasp position (in camera frame, meters)
    is chosen from the best 'solid' detection (by confidence).

    Returns:
        Tuple[str | None, dict | None]:
            - saved mask file path (or None on fatal error)
            - grasp_pos_cam as a dict {'x':, 'y':, 'z':} in meters (or None if no solid found)
    """
    import os
    import numpy as np
    import cv2

    print("--- 按需生成并保存掩码图 (仅限 Solid) ---")

    realsense_api = None
    perception_api = None
    center_pose = None  # 确保定义

    try:
        # 依赖：你需要在其他位置正确导入/实现 StainPerceptionAPI, RealsenseAPI, camera_intrinsics
        perception_api = StainPerceptionAPI(model_weights_path)
        realsense_api = RealsenseAPI()
        
        # 【修改点 1】: 调用 detect_stains 需要传入相机内参
        # 假设 realsense_api 可以提供内参
        camera_intrinsics = realsense_api.get_intrinsics()

        print("📷 正在捕获稳定的图像帧...")
        bgr_image, depth_image_m = realsense_api.get_frames()
        if bgr_image is None:
            print("❌ 错误：无法从相机捕获图像。")
            return None, None
        print("✅ 图像捕获成功。")

        print("🚀 正在运行污渍检测...")
        # 【修改点 2】: 接收 detect_stains 返回的字典
        detections_dict = perception_api.detect_stains(bgr_image, depth_image_m, camera_intrinsics, confidence_threshold)

        h, w = bgr_image.shape[:2]
        combined_mask = np.zeros((h, w), dtype=np.uint8)

        # 【修改点 3】: 从字典中安全地获取 'solid' 污渍列表
        solid_stains_list = detections_dict.get('solid', [])
        
        if solid_stains_list:
            print(f"🔍 检测到 {len(solid_stains_list)} 个 'solid' 污渍。")

            # 【修改点 4】: 直接在 solid_stains_list 上循环和排序
            # 按置信度降序排序，选出最优的
            solid_stains_list.sort(key=lambda s: s.get('confidence', 0), reverse=True)
            
            best_solid_stain = solid_stains_list[0]
            center_pose = best_solid_stain.get("position_m")
            
            print(f"✅ 已选择最优 'solid' 目标，置信度: {best_solid_stain.get('confidence')}")
            if center_pose is None:
                print("⚠️ 最优 solid 检测缺少 position_m 数据。")

            # 【修改点 5】: 遍历所有 solid 污渍，根据 bbox 生成并合并掩码
            for stain in solid_stains_list:
                bbox = stain.get("bbox_pixels")
                if bbox:
                    x1, y1, x2, y2 = bbox
                    cv2.rectangle(combined_mask, (x1, y1), (x2, y2), 255, -1)
        else:
            print("⚠️ 未发现 'solid' 类别，掩码将为全黑。")
            center_pose = None

        # 保存掩码
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        print(f"💾 正在保存掩码图至: '{output_path}'")
        ok = cv2.imwrite(output_path, combined_mask)
        if not ok:
            print("❌ cv2.imwrite 失败。")
            return None, None
        print("✅ 保存成功！")

        return output_path, False

    except Exception as e:
        import traceback
        print(f"❌ 发生了严重错误: {e}")
        traceback.print_exc() # 打印详细的错误追溯信息
        return None, None



if __name__ == "__main__":
    from PIL import Image

    # --- 配置参数 ---
    YOLO_MODEL_PATH = "/home/wmx/GraspNet-based-on-YOLO-for-specific-object-recognition-and-grasp/yolo8l_batch8_run1.pt"
    OUTPUT_MASK_PATH = "/home/wmx/graspnet-baseline/mask.png"
    CONF_THRESHOLD = 0.5

    # --- 执行主函数 ---
    saved_mask_path ,completed_grasp= generate_and_save_grasp_mask(
        model_weights_path=YOLO_MODEL_PATH,
        output_path=OUTPUT_MASK_PATH,
        confidence_threshold=CONF_THRESHOLD
    )

    # --- 处理返回的路径 ---
    if saved_mask_path:
        print(f"\n✅ 掩码已成功生成并保存到: {saved_mask_path}")
        print("现在模拟您的原始调用流程...")
        try:
            workspace_mask = np.array(Image.open(saved_mask_path).resize((640, 480), Image.NEAREST))
            print(f"成功读取并调整掩码尺寸为: {workspace_mask.shape}")
            # 在这里可以继续使用 workspace_mask

            # (可选) 显示最终生成的掩码图来验证结果
            cv2.imshow("Generated Solid-Only Mask", workspace_mask)
            print("按任意键退出显示...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        except FileNotFoundError:
            print(f"❌ 错误: 找不到文件 {saved_mask_path}")
        except Exception as e:
            print(f"❌ 读取或处理图像时出错: {e}")
    else:
        print("\n❌ 生成掩码文件失败。")