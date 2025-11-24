import numpy as np

# 定义变换矩阵 T (4x4)
T = np.array([
[ -0.05976104, 0.79598719, -0.60235622, 1.13315002 ],
[ 0.99799063, 0.06037170, -0.01923423, -0.05149746 ],
[ 0.02105507, -0.60229533, -0.79799563, 0.72158258 ],
[ 0.00000000, 0.00000000, 0.00000000, 1.00000000 ],
])

# 定义原始点 P (3D)
P = np.array([-0.018, -0.045, 0.960])

# 将 P 转换为齐次坐标 (添加 1)
P_homogeneous = np.append(P, 1)

# 计算变换后的点 P_transformed = T * P_homogeneous
P_transformed_homogeneous = np.dot(T, P_homogeneous)

# 转换回 3D 坐标 (去掉最后一维的 1)
P_transformed = P_transformed_homogeneous[:3]

print("变换后的点坐标:", P_transformed)