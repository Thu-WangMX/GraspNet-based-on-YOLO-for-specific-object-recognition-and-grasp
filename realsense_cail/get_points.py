#!/usr/bin/env python3
import sys
import time
import logging
import math
import numpy as np
import cv2
import pyrealsense2 as rs
# import urx
import rtde_control
import rtde_receive

# —— Charuco 板参数 —— #
CHARUCO_DICT = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_1000)

# 手动创建 CharucoBoard
CHARUCO_BOARD = cv2.aruco.CharucoBoard(
    size=(9, 7),           # board size (4x4) 一定要奇数
    dictionary=CHARUCO_DICT,
    squareLength=0.026,     # 格子的物理边长（米）
    markerLength=0.0195      # ArUco 子标记边长（米）

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

def get_jaka_gripper(robot):
    pose = robot.get_pose() 
    print(pose.pos)
    print(pose.orient)
    pos_trans = np.array([pose.pos.x, pose.pos.y, pose.pos.z])
    pos_R = pose.orient
    return  [pos_trans, pos_R]

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
        return None
    
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
        # cameraMatrix=camera_matrix,
        # distCoeffs=dist_coeffs
    )
    # print("camera_matrix: ", camera_matrix)
    # print("retval:", retval)
    # print("charuco_corners:", charuco_corners)

    # cv2.imshow("rgb", rgb)
    # cv2.waitKey(0)

    if retval < 4:
        return None

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

    # —— 初始化机械臂 URX —— #
    rtde_r = rtde_receive.RTDEReceiveInterface("192.168.101.101")
    # # robot.set_tcp((0,0,0,0,0,0))
    # robot.set_payload(0.5, (0,0,0))
    time.sleep(0.5)
    logging.info("Robot initialized at: %s", rtde_r.getActualTCPPose())

    gripper_poses = []
    charuco_poses = []

#     a = np.array([[0.556015544376486, -0.21901287467850813, 0.4117439893490715, -2.0796596801759755, -0.43005070583143057, -1.9471348676054188],
# [0.5560059992102859, -0.21897604109998295, 0.39538510006851213, -2.0796637824842823, -0.4300295275852257, -1.9471722668809655],
# [0.5651392607918211, -0.2190120574220885, 0.3954159719750363, -2.1413776173181, -0.3287667377158443, -2.0037629341398344],
# [0.565121772134597, -0.2189910007387964, 0.39535464974812456, -2.1217597639903016, -0.36199320305690197, -1.9856611734609195],
# [0.5651430165519069, -0.21894446467402753, 0.3953623407875425, -2.1033074148082656, -0.3922526988603351, -1.9687483867898412],
# [0.5651479473492378, -0.2189704233865069, 0.3953805207965347, -2.0574144193618666, -0.46439847302643505, -1.9267761746917043],
# [0.5651442358810608, -0.2189972245247138, 0.3953585948152769, -2.0130173622108347, -0.5300896600879557, -1.8860215021239441],
# [0.5751283742755533, -0.21904458860559559, 0.3953400559952672, -2.104077419309536, -0.49995678901615503, -1.7597567371144056],
# [0.5751680015654796, -0.2189881038876881, 0.3953074031238896, -2.0633500575765016, -0.5629552048586015, -1.7228602928263053],
# [0.5751563548848495, -0.219025207189552, 0.39537107464613697, -2.039863246407032, -0.5978795865428272, -1.7019046921156649],
# [0.5751684604085653, -0.21898764642972435, 0.39533946384492225, -2.00475782992927, -0.6479716611720229, -1.6702964738718593],
# [0.5751411698953601, -0.2190194806523306, 0.3953224603603943, -2.0656874251710504, -0.6165898156797117, -1.5714310360216517],
# [0.5751354546045024, -0.21901080980031504, 0.39540724180417564, -2.0804931500097643, -0.5937524938138478, -1.5849745365689893],
# [0.5751406062086636, -0.21899173052113197, 0.3849436474747692, -2.080523482324595, -0.5938531731058698, -1.5849274878491817],
# [0.5751488431441192, -0.2189383252991201, 0.3850352926671898, -2.10704827468401, -0.5518478902842218, -1.609108186067783],
# [0.5751359776472169, -0.21898786695202127, 0.37713343799193644, -2.1378741638669667, -0.5008675062290301, -1.637119216260468],
# [0.5751560176075494, -0.21897015983074183, 0.37717719693450635, -2.174370090399849, -0.4375702928210449, -1.6706630502074984],
# [0.5751629424815756, -0.21902412419436687, 0.37707802933900325, -2.2155119843803943, -0.3615992193217049, -1.7086571263103398],
# [0.5751811916372013, -0.21899801694892204, 0.37708254621790166, -2.2494982660396, -0.2946062474235994, -1.740465150660954],
# [0.5751418074827206, -0.21900462812138558, 0.3770968113672099, -2.2839161246092377, -0.2224549720202046, -1.7731017163952234],
# [0.5895348459341204, -0.2190408378946379, 0.3771189443859292, -2.283807901828857, -0.22250769915495325, -1.7732808830724485],
# [0.5895441051165856, -0.21900796387804952, 0.3771179329572434, -2.3607226682991174, -0.20512791088523322, -1.659357979093469],
# [0.5895316493965891, -0.21902494599113959, 0.3771637073361355, -2.329397032578898, -0.27664202781173997, -1.6305829096235878],
# [0.5895762584156137, -0.21899412504219776, 0.3770442285412646, -2.3161707291583618, -0.3053379024847822, -1.6181872559042876],
# [0.5895475482058495, -0.2190151380778126, 0.37711032958364704, -2.2846138676359526, -0.3712110013573615, -1.5899402166534982],
# [0.589571367145208, -0.2189736522089129, 0.377117296422248, -2.255413776635819, -0.42883971144447375, -1.5640023471516926],
# [0.5895470836848833, -0.21903257459429165, 0.37711167159933445, -2.2206653961968303, -0.4931616946749221, -1.533359598294483],
# [0.5895428762529155, -0.2190170065507448, 0.3770831531684952, -2.16764924599839, -0.5851929527382795, -1.4872133543573838],
# [0.5895547705536114, -0.21898284630860676, 0.37710186007316443, -2.110353236230393, -0.6151032777661903, -1.5899441253959605],
# [0.5895586950779289, -0.21902207702895188, 0.3807025732831484, -2.0700305576094515, -0.6341258669900087, -1.656910263292481],
# [0.5895662712804576, -0.21899270792455378, 0.38070379169053403, -2.04552605291355, -0.6690748847056976, -1.636271023045815],
# [0.5895213909512192, -0.21903925051142603, 0.3807312082295248, -2.0993125020532553, -0.6410035125000954, -1.5447392769518118],
# [0.5895309468739127, -0.2190080814082237, 0.3808001447983232, -2.0254386480191764, -0.5522032244470158, -1.5442703836207041],
# [0.5895540679419625, -0.21894796170477754, 0.3807437873876832, -1.961324536902655, -0.47796009951802415, -1.5413569402472476],
# [0.5895688480522672, -0.21901443116417008, 0.3806524472376266, -1.9049451084519053, -0.4145838899450627, -1.5371393927065662],
# [0.5895329164770977, -0.23539924561299663, 0.38076561325144104, -1.9049231614612945, -0.41464130109547664, -1.5372512301988714],
# [0.5895431807150304, -0.25017367680424624, 0.3807652855177273, -1.90487358136556, -0.4148325177171332, -1.5373203803547155],
# [0.6007397294915566, -0.2502347243915551, 0.38071943406919107, -1.9048889371043132, -0.41459430491846133, -1.537220164516112],
# [0.5819455189736953, -0.25020098132800944, 0.380723455363951, -1.9049699849652952, -0.4146202239586779, -1.5372135284296522],
# [0.5819568927330432, -0.2501700559374475, 0.3491855767103171, -1.904933154955269, -0.41482129186980715, -1.5373251297297568],
# [0.5819476580704902, -0.2501725021666812, 0.3355562897904199, -1.9048749704088073, -0.41476639988950176, -1.5372561600245924],
# [0.5819412688039493, -0.25022348274319484, 0.32930062824904943, -1.9332582967384595, -0.36058698625571184, -1.5756922447179504],
# [0.5819448311342472, -0.25021915349202356, 0.32929769418920435, -1.9728872174784058, -0.2798098540144521, -1.6307721662448122],
# [0.581961160092348, -0.2501888277591768, 0.3292640414369612, -2.0049800734315184, -0.208869907818828, -1.6767378178792711],
# [0.5819529660261976, -0.25021034537509307, 0.3292906999681269, -2.040165678538209, -0.12468682538823025, -1.728882148758949],
# [0.5819594751357409, -0.25015266340044234, 0.31129229897789956, -2.0401108700635513, -0.12482236391680043, -1.728794577717424],
# [0.5819513795234328, -0.2502105535020939, 0.31127917106913594, -2.067722372631179, -0.05264302547642821, -1.7710553883678912],
# [0.5819438238991057, -0.2502506002739229, 0.31125093851714797, -2.1006502055300884, -0.40387307801160405, -1.86411084884868],
# [0.5819455747391411, -0.2502442553535383, 0.31124866338570967, -2.180256446084686, -0.5338082098433077, -1.8838788020472101],
# [0.5819406789788437, -0.2502191621068046, 0.31127386302516513, -2.2700414181637836, -0.6940126498191477, -1.900422038494159],
# [0.5819640019822845, -0.2501431307915393, 0.34684442757702916, -2.269946120059409, -0.6939253297340797, -1.900208223560923],
# [0.581956005471815, -0.2501737919726607, 0.3789831976580829, -2.2700551436551257, -0.6939689354646098, -1.9003403022295882],
# [0.5819614565635726, -0.2502244333463177, 0.3957561885245265, -2.2701743029970616, -0.6936571807174607, -1.9001863400249839],
# [0.5819109696232345, -0.25026005014160707, 0.39578836891055985, -2.1991195483279005, -0.7821143134447774, -1.8597991757141792],
# [0.5819554590006122, -0.25023049078660525, 0.39575964997555, -2.1238193049589174, -0.8692024250088006, -1.8156439728229472],
# [0.5819686687507853, -0.25020297102602546, 0.39571931623532763, -2.036627826578824, -0.9622989251563313, -1.7631687525980622]])

    # print(a.shape)
    # robot.set_tcp((0,0,0,0,0,0))   # tool center point

    i = 0
    while True:
        key = input()
        i+=1
        print(rtde_r.getActualTCPPose())
        if i == 55:
            print("55 points")
            return
    # joint_angles = np.array([0., -90, -60, -120, 90, 0])
    # joint_angles = joint_angles / 180 * 3.14159
    # robot.movej(joint_angles, acc=0.2, vel=0.3)
    # print(robot.getl())
    # pose = np.array([ 0.57576626,-0.05204254,  0.4, -2.22167710265422, 2.2183882956246626, -0.003071341257484209])
    # robot.movel(pose, acc=0.2, vel=0.3)
    

    
    return
    y_limit = [-0.3, -0.15]
    z_limit = [0.365, 0.485]
    x = 0.55


    for j in range(4):
        pose_range = 15
        for i in range(10):
            pose = np.array((0, pose_range* i/10., h, 0, 0, 0))
            robot.movel("movel", pose=pose, relative=True)

            rgb, depth, K, dist = get_aligned_images(pipeline, align)
            if rgb is None:
                continue

            pose = get_charuco_pose(rgb, camera_matrix, dist_coeffs)
            key = cv2.waitKey(33) & 0xFF


            gp = get_jaka_gripper(robot)
            if pose is not None:
                gripper_poses.append(gp)
                charuco_poses.append(pose)
                logging.info("Recorded sample %d", len(gripper_poses))
            else:
                logging.warning("Charuco board not detected this frame.")
    
            time.sleep(1)
    
        h += 0.03       

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
        gripper_rot = gripper[1].get_array()
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


    R_gripper_in_base = np.array(R_gripper_in_base)
    t_gripper_in_base = np.array(t_gripper_in_base)
    R_target_in_camera = np.array(R_target_in_camera)
    t_target_in_camera = np.array(t_target_in_camera)

    T_ee_b = np.zeros((len(R_gripper_in_base), 4, 4))
    for i in range(len(T_ee_b)):
        T_ee_b[i][:3, :3] = R_gripper_in_base[i]
        T_ee_b[i][:3, 3] = t_gripper_in_base[i]


    T_b_ee = np.zeros_like(T_ee_b)
    for i in range(len(T_ee_b)):
        T_b_ee[i][:3, :3] = np.transpose(T_ee_b[i][:3, :3])
        T_b_ee[i][:3, 3] = -np.dot(np.transpose(T_ee_b[i][:3, :3]), T_ee_b[i][:3, 3])

    r_b_ee = T_b_ee[:, :3, :3].reshape(-1, 3, 3)
    t_b_ee = T_b_ee[:, :3, 3].reshape(-1, 3)
    r_t_c = R_target_in_camera.reshape(-1, 3, 3)
    t_t_c = t_target_in_camera.reshape(-1, 3)

    print(len(r_b_ee),len(t_b_ee))
    results = solve_hand_eye(r_b_ee, t_b_ee, r_t_c, t_t_c)

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
    robot.close()

if __name__ == "__main__":
    main()
