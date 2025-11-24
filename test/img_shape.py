# import cv2

# # 图片路径
# img_path = "/home/wmx/graspnet-baseline/mask.png"

# # 读取图像（彩色）
# img = cv2.imread(img_path)

# # 检查是否读取成功
# if img is None:
#     print(f"❌ 无法读取图像：{img_path}")
# else:
#     print(f"✅ 图像读取成功")
#     print(f"图像 shape: {img.shape}")   # (高度, 宽度, 通道数)
import cv2
import numpy as np

# 图像路径
img_path = "/home/wmx/graspnet-baseline/mask.png"

# 读取为灰度图
img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

if img is None:
    print(f"❌ 无法读取图像：{img_path}")
    exit()

# 二值化（确保只有 0 和 1）
binary_mask = (img > 0).astype(np.uint8)

# 打印图像的 shape
print(f"图像尺寸: {binary_mask.shape}")

# 举例：判断 (100, 200) 这个像素点
y, x = 100, 200
print(f"像素点({y}, {x}) 的值为: {binary_mask[y, x]}")

# 遍历整张图（不建议大图这么做，但可以验证逻辑）
# for i in range(binary_mask.shape[0]):
#     for j in range(binary_mask.shape[1]):
#         value = binary_mask[i, j]
#         print(f"({i},{j}): {value}")

# 统计0和1的数量
num_zero = np.sum(binary_mask == 0)
num_one = np.sum(binary_mask == 1)
print(f"像素值为0的点数: {num_zero}")
print(f"像素值为1的点数: {num_one}")
