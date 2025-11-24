import time
import threading
import spdlog      
import flexivrdk
import math  
from scipy.spatial.transform import Rotation as R

class FlexivRobot:

    def __init__(self, robot_sn: str, gripper_name:str,frequency: float = 100.0):
            self.robot_sn = robot_sn
            self.frequency = frequency
            self.logger = spdlog.ConsoleLogger("FlexivRobot")
            self.mode = flexivrdk.Mode
            #建立连接
            try:
                self.robot = flexivrdk.Robot(self.robot_sn)
                # # Clear fault on the connected robot if any
                if self.robot .fault():
                    self.logger.warn("Fault occurred on the connected robot, trying to clear ...")
                    # Try to clear the fault
                    if not self.robot.ClearFault():
                        self.logger.error("Fault cannot be cleared, exiting ...")
                        return 1
                    self.logger.info("Fault on the connected robot is cleared")
                self.logger.info("Enabling robot ...")
                self.robot.Enable()

                # Wait for the robot to become operational
                while not self.robot.operational():
                    time.sleep(1)

                self.logger.info("Robot is now operational")
            except Exception as e:
                # Print exception error message
                self.logger.error(str(e))
            
            # #初始化夹爪
            self.gripper = flexivrdk.Gripper(self.robot)
            #初始化，在开机时可以先在示教器上手动初始化
            self.tool = flexivrdk.Tool(self.robot)
            self.gripper.Enable(gripper_name)
            self.gripper.Init()
            time.sleep(10) #等待夹爪初始化完成
            self.logger.info("Enabling gripper")
            # #切换tcp至夹爪坐标系,默认坐标系是与示教器上一致
            self.tool.Switch(gripper_name)

    
    def Stop(self):
        self.robot.Stop()
        self.gripper.Stop()
        self.logger.info("Robot is stopped")

    def quat2eulerZYX(quat, degree=False):
        """
        Convert quaternion to Euler angles with ZYX axis rotations.

        Parameters
        ----------
        quat : float list
            Quaternion input in [w,x,y,z] order.
        degree : bool
            Return values in degrees, otherwise in radians.

        Returns
        ----------
        float list
            Euler angles in [x,y,z] order, radian by default unless specified otherwise.
        """

        # Convert target quaternion to Euler ZYX using scipy package's 'xyz' extrinsic rotation
        # NOTE: scipy uses [x,y,z,w] order to represent quaternion
        eulerZYX = (
            R.from_quat([quat[1], quat[2], quat[3], quat[0]])
            .as_euler("xyz", degrees=degree)
            .tolist()
        )

        return eulerZYX
    
    def convert_radians_to_degrees(self,radian_list):
        """
        将一个包含弧度值的列表转换为角度值列表。

        Args:
            radian_list: 一个包含浮点数（弧度）的列表。

        Returns:
            一个新的列表，其中每个元素都已从弧度转换为角度。
        """
        # 使用列表推导式和 math.degrees() 函数进行转换
        degree_list = [math.degrees(rad) for rad in radian_list]
        return degree_list
    
    #Euler_flag True返回角度制欧拉角，False返回四元数，返回值可直接用于MoveL
    def read_pose(self,Euler_flag=False):
        if Euler_flag:
            quat = self.robot.states().tcp_pose[3:7] #w,x,y,z
            euler = FlexivRobot.quat2eulerZYX(quat,degree=True) #x,y,z
            position = self.robot.states().tcp_pose[0:3]
            return position, euler#list x,y,z,rx,ry,rz
        else:
            return self.robot.states().tcp_pose #list x,y,z,w,
    #degree_flag True返回角度制关节角,可用于MOVEJ，False返回弧度制关节角
    def read_joint(self,degree_flag=False):
        if degree_flag:
            radian_list = self.robot.states().q #list 1-7
            degree_list = self.convert_radians_to_degrees(radian_list)
            return degree_list #list 1-7
        else:
            return self.robot.states().q #list 1-7
    def swith_PRIMITIVE_Mode(self):
        self.robot.SwitchMode(self.mode.NRT_PRIMITIVE_EXECUTION)
    
    #speed [0.001 … 2.2]m/s acc [0.1 … 3.0] m/s^2
    def MoveL(self, position,euler, speed=0.1, acc=0.1):
        self.robot.ExecutePrimitive(
                    "MoveL",
                    {
                        "target": flexivrdk.Coord(
                            position, euler, ["WORLD", "WORLD_ORIGIN"]
                        ),
                        "vel": speed,
                        "acc": acc
                    },
                )
        # Wait for reached target
        while not self.robot.primitive_states()["reachedTarget"]:
            time.sleep(1)
        self.logger.info("Executing primitive: MoveL")
    #target_joitn为目标关节位置，7个关节角度值，单位为度
    #jntVelScale为关节速度尺度，范围[1-100]，默认20
    def MoveJ(self, target_joint, jntVelScale=20):
        self.robot.ExecutePrimitive(
                    "MoveJ",
                    {
                        "target": flexivrdk.JPos(
                            target_joint
                        ),
                        "jntVelScale": jntVelScale
                    },
                )
        # Wait for reached target
        while not self.robot.primitive_states()["reachedTarget"]:
            time.sleep(1)
        self.logger.info("Executing primitive: MoveJ")
    
    #joints_list为多个目标关节角列表，joints_list其中最后一个元素为目标终点，其余元素为中间点
    #jntVelScale为关节速度尺度，范围[1-100]，默认20
    def MoveJ_multi_points(self, joints_list, jntVelScale=20):
        if len(joints_list) > 1:
            middle_points = []
            for joints in joints_list[:-1]:
                middle_points.append(flexivrdk.JPos(joints))
            self.robot.ExecutePrimitive(
                        "MoveJ",
                        {
                            "target": flexivrdk.JPos(
                                joints_list[-1]
                            ),
                            "waypoints": middle_points,
                            "jntVelScale": jntVelScale
                        },
                    )
            # Wait for reached target
            while not self.robot.primitive_states()["reachedTarget"]:
                time.sleep(1)
            self.logger.info("Executing primitive: MoveJ")
    
    #宽度[0,0.1]m,速度[0.001,0.2]m/s，接触力[-80,80]N
    def Move_gripper(self, width:float, speed:float=0.1, force:float=10.0):
        self.gripper.Move(width, speed, force)
        time.sleep(3)#等待夹爪动作完成
    
