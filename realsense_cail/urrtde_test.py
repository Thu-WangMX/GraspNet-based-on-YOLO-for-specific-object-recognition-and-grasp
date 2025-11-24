import rtde_control
import rtde_receive
rtde_c = rtde_control.RTDEControlInterface("192.168.101.101")
rtde_r = rtde_receive.RTDEReceiveInterface("192.168.101.101")
rtde_c.moveL([-0.45314510917389555, 0.08655704292139524, 0.392161675841967, 2.048000623584075, 2.3274585573454556, 0.02067275831323949], 0.1, 0.1)
actual_t = rtde_r.getActualTCPPose()
print(actual_t)