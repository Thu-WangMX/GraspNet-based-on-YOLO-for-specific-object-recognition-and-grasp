#!/usr/bin/env python3
import sys
import time
import logging
import math
import numpy as np
import cv2
import pyrealsense2 as rs
import rtde_control
import rtde_receive
from scipy.spatial.transform import Rotation as R
import json

# —— Charuco 板参数 —— #
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

# 手动创建 CharucoBoard
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    size=(7, 9),           # board size (4x4) 一定要奇数
    dictionary=CHARUCO_DICT,
    squareLength=0.027,     # 格子的物理边长（米）
    markerLength=0.02      # ArUco 子标记边长（米）

    # squareLength=0.4,     # 格子的物理边长（米）
    # markerLength=0.3      # ArUco 子标记边长（米）
)

# 替代 DetectorParameters_create 的写法
DETECT_PARAMS = cv2.aruco.DetectorParameters()  # 使用 cv2.aruco.DetectorParameters() 代替

def get_aligned_images(pipeline, align):
    """从 RealSense 获取对齐后的 RGB、深度图和相机内参"""
    frames = pipeline.wait_for_frames()
    # aligned = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    if not depth_frame or not color_frame:
        return None, None, None, None

    # 相机内参
    intr = color_frame.profile.as_video_stream_profile().intrinsics
    intr_matrix = np.array([
        [intr.fx, 0,       intr.ppx],
        [0,       intr.fy, intr.ppy],
        [0,       0,       1]
    ])
    dist_coeffs = np.array(intr.coeffs)

    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data())
    return color_image, depth_image, intr_matrix, dist_coeffs

#获取末端的旋转和姿态
def get_jaka_gripper(rtde_r):
    # pose = robot.get_pose() 
    # print(pose.pos)
    # print(pose.orient)
    # pos_trans = np.array([pose.pos.x, pose.pos.y, pose.pos.z])
    # pos_R = pose.orient
    pose = rtde_r.getActualTCPPose()
    R_E2B = R.from_rotvec( np.array(pose[-3:])).as_matrix() 
    P_E2B = np.array([pose[0],pose[1],pose[2]])
    return  [P_E2B, R_E2B]

def get_charuco_pose(rgb, camera_matrix, dist_coeffs):
    """
    检测 ChArUco 板并估计位姿。
    返回：tvec (3,) 和 rvec (3,) 合并的 list，或者 None
    """
 
    gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
 
    # 1. ArUco 检测
    corners, ids, _ = cv2.aruco.detectMarkers(
        gray, CHARUCO_DICT, parameters=DETECT_PARAMS
    )

    if ids is None or len(ids) == 0:
        assert False, "No corners detected."
    
    if len(corners) < 0:
        print("No corners detected.")
        return None

    # 绘制检测到的标记
    output_img = gray.copy()
    output_img = cv2.aruco.drawDetectedMarkers(output_img, corners, ids)


    # cv2.imshow("Detected ArUco Markers", output_img)
    # cv2.waitKey(0)

    # 2. 插值出 ChArUco 角点
    # print("camera_matrix: ", camera_matrix)
    retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        markerCorners=corners,
        markerIds=ids,
        image=gray,
        board=CHARUCO_BOARD,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs
    )
    # print("camera_matrix: ", camera_matrix)
    # print("retval:", retval)
    # print("charuco_corners:", charuco_corners)

    # cv2.imshow("rgb", rgb)
    # cv2.waitKey(0)

    if retval < 16:
        raise ValueError(f"Not enough corners detected (found {retval}, need at least 4)")
    
    # 3. 估计 CharucoBoard 位姿
    success, rvec, tvec = cv2.aruco.estimatePoseCharucoBoard(
        rvec=np.empty(1),   # dummy rvec 占位
        tvec=np.empty(1),    # dummy tvec 占位
        charucoCorners=charuco_corners,
        charucoIds=charuco_ids,
        board=CHARUCO_BOARD,
        cameraMatrix=camera_matrix,
        distCoeffs=dist_coeffs

    )
    # print("success:", success)
    if not success:
        return None

    # 可视化（可选）
    cv2.aruco.drawDetectedCornersCharuco(rgb, charuco_corners, charuco_ids)
    cv2.drawFrameAxes(rgb, camera_matrix, dist_coeffs, rvec, tvec, 0.1)
    
    # 显示图像
    cv2.imshow("Charuco Detection", rgb)
    # cv2.waitKey(1)  # 这里需要确保调用了waitKey来刷新窗口

    return list(tvec.flatten()) + list(rvec.flatten())


def solve_hand_eye(robot_R_list, robot_t_list, cam_R_list, cam_t_list):
    """
    使用 OpenCV 手眼标定函数求解 X (camera->base):
      R_cam2base, t_cam2base
    """
    methods = {
        'TSAI': cv2.CALIB_HAND_EYE_TSAI,
        # 'PARK': cv2.CALIB_HAND_EYE_PARK,
        'HORAUD': cv2.CALIB_HAND_EYE_HORAUD,
        'ANDREFF': cv2.CALIB_HAND_EYE_ANDREFF,
        'DANIILIDIS': cv2.CALIB_HAND_EYE_DANIILIDIS
    }
    results = {}
    for method_name, method in methods.items():
        R_cam2base, t_cam2base = cv2.calibrateHandEye(
            robot_R_list, robot_t_list,
            cam_R_list, cam_t_list,
            method=method
        )
        results[method_name] = (R_cam2base, t_cam2base)
    return results

    # R_cam2base, t_cam2base = cv2.calibrateHandEye(
    #     robot_R_list, robot_t_list,
    #     cam_R_list,   cam_t_list,
    #     method=cv2.CALIB_HAND_EYE_TSAI
    # )
    # return R_cam2base, t_cam2base

def main():
    logging.basicConfig(level=logging.INFO)
    # —— 初始化 RealSense —— #
    pipeline = rs.pipeline()
    cfg = rs.config()
    cfg.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    cfg.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
    cfg.enable_device("152122079499")
    profile = pipeline.start(cfg)
    align   = rs.align(rs.stream.color)

    # 获取并缓存 color intrinsics
    color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
    color_intr   = color_profile.get_intrinsics()
    camera_matrix = np.array([
        [color_intr.fx, 0,             color_intr.ppx],
        [0,             color_intr.fy, color_intr.ppy],
        [0,             0,             1]
    ])
    dist_coeffs = np.array(color_intr.coeffs)

    # —— 初始化机械臂 URX —— #
    # robot = urx.Robot("192.168.101.101")
    # # robot.set_tcp((0,0,0.,0,0,0))
    # robot.set_payload(0.5, (0,0,0))
    rtde_c = rtde_control.RTDEControlInterface("192.168.101.101")
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.101.101")

    time.sleep(0.5)
    logging.info("Robot initialized at: %s", rtde_r.getActualTCPPose())

    gripper_poses = []
    charuco_poses = []

    filename = "collected_robot_poses.json"
    with open(filename, 'r') as f:
        # 3. 使用 json.load() 将文件内容解析为Python列表
        list_of_poses = json.load(f)

    # 4. 使用 np.array() 将Python列表转换为NumPy数组
    # 这就是您需要的可迭代变量
    points = np.array(list_of_poses)

    for point in points:
        # robot.movel("movel", pose=pose, relative=True)
        # robot.movel(point, acc=0.2, vel=0.2, wait=True)
        rtde_c.moveL(point,0.2,0.2)
        time.sleep(1)
        rgb, depth, K, dist = get_aligned_images(pipeline, align)
        if rgb is None:
            continue

        pose = get_charuco_pose(rgb, camera_matrix, dist_coeffs)
        key = cv2.waitKey(33) & 0xFF


        gp = get_jaka_gripper(rtde_r)
        if pose is not None:
            gripper_poses.append(gp)
            charuco_poses.append(pose)
            logging.info("Recorded sample %d", len(gripper_poses))
        else:
            logging.warning("Charuco board not detected this frame.")


    # 收集 R, t

    R_gripper_in_base, R_target_in_camera = [], []
    t_gripper_in_base, t_target_in_camera = [], []


    for marker in charuco_poses:
        #m-c的旋转矩阵和位移矩阵
        camera_rot = marker[3:6] # 3:6 是旋转向量
        camera_mat,_ = cv2.Rodrigues((camera_rot[0],camera_rot[1],camera_rot[2])) #旋转矢量到旋转矩阵
        R_target_in_camera.append(camera_mat)
        t_target_in_camera.append(np.array(marker[0:3])) 


    for gripper in gripper_poses:

        # g-b的旋转矩阵和位移矩阵
        # gripper_rot = gripper[1].get_array()
        gripper_rot = gripper[1]
        gripper_pos = gripper[0]
        R_gripper_in_base.append(gripper_rot) #欧拉角到旋转矩阵；# 表示为按照xyz
        t_gripper_in_base.append(gripper_pos) 


    data = {
        'R_target_in_camera': R_target_in_camera,
        't_target_in_camera': t_target_in_camera,
        'R_gripper_in_base':   R_gripper_in_base,
        't_gripper_in_base':   t_gripper_in_base
    }

    # 保存到 .npy 文件
    np.save('calibration_data_without_tcp.npy', data)


    # R_gripper_in_base = np.array(R_gripper_in_base)
    # t_gripper_in_base = np.array(t_gripper_in_base)
    # R_target_in_camera = np.array(R_target_in_camera)
    # t_target_in_camera = np.array(t_target_in_camera)

    # # T_end_effector_in_base
    # T_ee_b = np.zeros((len(R_gripper_in_base), 4, 4))
    # for i in range(len(T_ee_b)):
    #     T_ee_b[i][:3, :3] = R_gripper_in_base[i]
    #     T_ee_b[i][:3, 3] = t_gripper_in_base[i]


    # T_b_ee = np.zeros_like(T_ee_b)
    # for i in range(len(T_ee_b)):
    #     T_b_ee[i][:3, :3] = np.transpose(T_ee_b[i][:3, :3])
    #     T_b_ee[i][:3, 3] = -np.dot(np.transpose(T_ee_b[i][:3, :3]), T_ee_b[i][:3, 3])

    # r_b_ee = T_b_ee[:, :3, :3].reshape(-1, 3, 3)
    # t_b_ee = T_b_ee[:, :3, 3].reshape(-1, 3)
    # r_t_c = R_target_in_camera.reshape(-1, 3, 3)
    # t_t_c = t_target_in_camera.reshape(-1, 3)

    # print(len(r_b_ee),len(t_b_ee))

    # 1. 机器人位姿 (Gripper -> Base)
    # 对于眼在手上，我们直接使用 T_E->B (末端在基座下)，无需像眼在手外那样求逆。
    R_gripper_in_base = np.array(R_gripper_in_base)
    t_gripper_in_base = np.array(t_gripper_in_base)

    # 2. 相机位姿 (Target -> Camera)
    # 对于眼在手上，我们需要将 T_T->C (标定板在相机下) 求逆，得到 T_C->T (相机在标定板下)。
    R_target_in_camera = np.array(R_target_in_camera)
    t_target_in_camera = np.array(t_target_in_camera)

    # # 新增：对相机位姿列表进行求逆
    # R_camera_in_target = []
    # t_camera_in_target = []
    # for i in range(len(R_target_in_camera)):
    #     R_t_c = R_target_in_camera[i]
    #     t_t_c = t_target_in_camera[i].reshape(3, 1)  # 确保是列向量
        
    #     # 旋转部分求逆等于转置: R_c_t = (R_t_c)^T
    #     R_c_t = R_t_c.T
    #     # 平移部分求逆: t_c_t = -R_c_t * t_t_c
    #     t_c_t = -np.dot(R_c_t, t_t_c)
        
    #     R_camera_in_target.append(R_c_t)
    #     t_camera_in_target.append(t_c_t.flatten()) # 存为一维数组

    # R_camera_in_target = np.array(R_camera_in_target)
    # t_camera_in_target = np.array(t_camera_in_target)

    # 3. 调用标定函数
    # 传入的参数分别是：(gripper->base)的位姿 和 (camera->target)的位姿
    print(f"准备传入标定函数的数据点数量: {len(R_gripper_in_base)}")
    results = solve_hand_eye(R_gripper_in_base, t_gripper_in_base, 
                            R_target_in_camera, t_target_in_camera)

    for k, v in results.items():
        R_c_b, t_c_b = v

        T_c_b = np.eye(4)
        T_c_b[:3, :3] = R_c_b
        T_c_b[:3, 3] = t_c_b.flatten()

        print(f"{k}:")
        for row in T_c_b:
            print(f"[ {row[0]:.8f}, {row[1]:.8f}, {row[2]:.8f}, {row[3]:.8f} ],")

    cv2.destroyAllWindows()
    pipeline.stop()
    # Stop the RTDE control script
    rtde_c.stopScript()

if __name__ == "__main__":
    main()
