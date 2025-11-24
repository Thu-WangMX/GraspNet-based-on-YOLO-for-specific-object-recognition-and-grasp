from FlexivRobot import FlexivRobot
#建立机器人连接
ROBOT_SN =  'Rizon4s-062958'
GRIPPER_NAME ='Flexiv-GN01'
frequency = 100.0 #控制频率
robot = FlexivRobot(ROBOT_SN,GRIPPER_NAME,frequency,gripper_init = False,remote_control = True)

import json
import argparse
from pathlib import Path

def extract_joints(json_path, save_json=None, save_numpy_dir=None, max_traj_idx=10):
    json_path = Path(json_path)
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    traj_list = data.get("trajectories", [])
    out = {}

    # 遍历 0..6 号轨迹（若不足 7 条，会自动到头为止）
    for idx in range(min(len(traj_list), max_traj_idx + 1)):
        traj = traj_list[idx]
        poses = traj.get("poses", [])
        joints = []
        for p in poses:
            j = p.get("joint", None)
            # 只接受长度为7的列表/元组
            if isinstance(j, (list, tuple)) and len(j) == 7:
                joints.append([float(x) for x in j])
            else:
                # 若不存在或长度不为7，直接跳过
                continue
        out[f"traj_{idx}"] = joints
        print(f"[INFO] traj_{idx}: 收集到 {len(joints)} 个 joint 条目")

    # 需要的话，整体写回一个 JSON
    if save_json:
        save_json = Path(save_json)
        save_json.parent.mkdir(parents=True, exist_ok=True)
        with open(save_json, "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"[DONE] joints JSON 已保存到: {save_json}")

    # 需要的话，每条轨迹另存为 .npy
    if save_numpy_dir:
        import numpy as np
        save_dir = Path(save_numpy_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        for k, v in out.items():
            arr = np.array(v, dtype=float)  # 形状: (N, 7)
            npy_path = save_dir / f"{k}.npy"
            np.save(npy_path, arr)
            print(f"[DONE] {k}.npy -> 形状 {arr.shape} 已保存到: {npy_path}")

    return out

input_json_path = "/home/wmx/graspnet-baseline/poses_20251011_224117.json"

traj_joints = extract_joints(json_path=input_json_path)

print(traj_joints)

#多点MOVEJ使用
robot.switch_PRIMITIVE_Mode()#切换至元操作模式

#robot.move_tcp_home() #TCP回到初始位置

#循环执行traj_joints中的多个轨迹
for i in range(len(traj_joints.keys())):
    key = f"traj_{i}"
    if key in traj_joints:
        joints_list = traj_joints[key]
        print(f"{key}: {len(joints_list)} joint_poses")
        robot.MoveJ_multi_points(joints_list,10)
        #robot.move_tcp_home() #TCP回到初始位置