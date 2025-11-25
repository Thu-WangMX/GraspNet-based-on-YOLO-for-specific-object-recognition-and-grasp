
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#刮台面采集轨迹
import json
import time
import argparse
from datetime import datetime

# === 按键读取：兼容 Win / *nix ===
import sys
try:
    import msvcrt  # Windows
    _IS_WINDOWS = True
except ImportError:
    import tty
    import termios
    import select
    _IS_WINDOWS = False

from FlexivRobot import FlexivRobot


def now_ts():
    return time.time()


def getch_nonblocking():
    """非阻塞读单键；无键时返回 None。"""
    if _IS_WINDOWS:
        if msvcrt.kbhit():
            ch = msvcrt.getch()
            try:
                return ch.decode("utf-8")
            except Exception:
                return None
        return None
    else:
        # Linux/macOS: 使用 select + 原始终端模式
        dr, _, _ = select.select([sys.stdin], [], [], 0)
        if dr:
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                ch = sys.stdin.read(1)
                return ch
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        return None


def parse_pose_from_result(res):
    """
    将 robot.read_joint(True) 的返回解析为:
    { "position": [x,y,z], "euler_xyz": [rx,ry,rz] }
    """
    # 字典情况
    if isinstance(res, dict):
        # 常见可能的键名
        for k in ["tcp_pose", "pose", "tcp", "tcpPose", "tcp_pose_xyzrpy"]:
            if k in res:
                val = res[k]
                if isinstance(val, (list, tuple)) and len(val) >= 6:
                    return {
                        "position": [float(val[0]), float(val[1]), float(val[2])],
                        "euler_xyz": [float(val[3]), float(val[4]), float(val[5])]
                    }
                if isinstance(val, dict):
                    # 形如 {"position":[...], "euler":[...]} 或相近
                    pos_keys = ["position", "pos", "xyz"]
                    eul_keys = ["euler", "rpy", "euler_xyz", "rot", "rxyz"]
                    pos = None
                    eul = None
                    for pk in pos_keys:
                        if pk in val:
                            pos = val[pk]
                            break
                    for ek in eul_keys:
                        if ek in val:
                            eul = val[ek]
                            break
                    if pos is not None and eul is not None:
                        return {
                            "position": [float(pos[0]), float(pos[1]), float(pos[2])],
                            "euler_xyz": [float(eul[0]), float(eul[1]), float(eul[2])]
                        }
        # 如果字典直接有 xyz/rpy
        if all(k in res for k in ["x", "y", "z", "rx", "ry", "rz"]):
            return {
                "position": [float(res["x"]), float(res["y"]), float(res["z"])],
                "euler_xyz": [float(res["rx"]), float(res["ry"]), float(res["rz"])],
            }

    # 列表/元组情况
    if isinstance(res, (list, tuple)) and len(res) >= 6:
        return {
            "position": [float(res[0]), float(res[1]), float(res[2])],
            "euler_xyz": [float(res[3]), float(res[4]), float(res[5])]
        }

    # 未识别
    raise ValueError(
        f"无法从返回值解析末端位姿，获得类型：{type(res)} 内容示例：{str(res)[:120]}..."
    )


def read_current_pose(robot):
    """
    读取当前末端位姿（位置 + 欧拉角 xyz），返回 dict:
    {"position":[x,y,z], "euler_xyz":[rx,ry,rz]}
    """
    # 你的示例是 read_joint(True)：按你当前 SDK 习惯保留
    position, euler = robot.read_pose(True)

    # 如果你实际有 read_tcp_pose() 接口，也可以改成：
    # raw = robot.read_tcp_pose()

    # return parse_pose_from_result(raw)
    return position, euler


def read_current_joint(robot):
    joint_deg = robot.read_joint(True)

    return joint_deg

def print_help():
    print("\n=== 键盘操作 ===")
    print("  s : 保存当前位姿到【当前轨迹】")
    print("  n : 开始一条新轨迹（上一条轨迹会被收尾）")
    print("  u : 撤销当前轨迹的最后一个位姿")
    print("  h/? : 显示帮助")
    print("  q : 保存 JSON 并退出\n")


def main():
    parser = argparse.ArgumentParser(
        description="按键采集 Flexiv 机械臂末端位姿到多条轨迹并保存为 JSON"
    )
    parser.add_argument("--robot-sn", default="Rizon4s-062958", help="机器人序列号")
    parser.add_argument("--gripper", default="Flexiv-GN01", help="夹爪名称")
    parser.add_argument("--freq", type=float, default=100.0, help="控制频率")
    parser.add_argument("--out", default=None, help="输出 JSON 文件路径（可选）")
    parser.add_argument("--remote", default=False, action="store_true", help="启用远程控制标志（如需要）")
    parser.add_argument("--no-gripper-init", default=False, action="store_true", help="不初始化夹爪")
    args = parser.parse_args()

    out_path = args.out or f"poses_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print("[INFO] 正在连接机器人 ...")
    robot = FlexivRobot(
        args.robot_sn,
        args.gripper,
        args.freq,
        gripper_init=not args.no_gripper_init,
        remote_control=args.remote
    )
    print("[OK] 连接成功。开始按键采集。")
    print_help()

    data = {
        "meta": {
            "robot_sn": args.robot_sn,
            "gripper_name": args.gripper,
            "frequency": args.freq,
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "note": "按键采集末端位姿：position(x,y,z) + euler_xyz(rx,ry,rz)，单位由 SDK 决定。"
        },
        "trajectories": []
    }

    traj_idx = 0
    current_traj = {
        "name": f"traj_{traj_idx:03d}",
        "poses": []  # 每个 pose: {t, position[3], euler_xyz[3]}
    }
    data["trajectories"].append(current_traj)

    try:
        while True:
            ch = getch_nonblocking()
            if ch is None:
                # 空转小睡一下，避免占满 CPU
                time.sleep(0.01)
                continue

            ch = ch.lower()

            if ch in ("h", "?"):
                print_help()

            elif ch == "s":
                try:
                    position, euler = read_current_pose(robot)
                    joint_deg = read_current_joint(robot)
                    stamp = now_ts()
                    current_traj["poses"].append({
                        "t": stamp,
                        "pos": position,
                        "euler": euler,
                        "joint": joint_deg
                    })
                    print(f"[SAVED] {current_traj['name']} 追加 1 个位姿，总数 = {len(current_traj['poses'])}")
                except Exception as e:
                    print(f"[WARN] 读取或解析位姿失败：{e}")

            elif ch == "u":
                if current_traj["poses"]:
                    removed = current_traj["poses"].pop()
                    print(f"[UNDO] 撤销最后一个位姿，{current_traj['name']} 现在共有 {len(current_traj['poses'])} 个位姿。")
                else:
                    print("[UNDO] 当前轨迹为空，无法撤销。")

            elif ch == "n":
                # 开启新轨迹
                traj_idx += 1
                current_traj = {
                    "name": f"traj_{traj_idx:03d}",
                    "poses": []
                }
                data["trajectories"].append(current_traj)
                print(f"[NEW] 开始新轨迹：{current_traj['name']}")

            elif ch == "q":
                # 保存并退出
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print(f"[DONE] 已保存到：{out_path}")
                break

            else:
                # 过滤控制符等不可见字符
                if ch.isprintable():
                    print(f"[HINT] 未定义按键：'{ch}'；按 h 查看帮助。")

    except KeyboardInterrupt:
        # Ctrl+C 也保存一次，避免丢数据
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        print(f"\n[INTERRUPT] 捕获到中断，已保存到：{out_path}")


if __name__ == "__main__":
    main()