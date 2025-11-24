import flexivrdk
import spdlog
import time

logger = spdlog.ConsoleLogger("test")
try:
    robot = flexivrdk.Robot('Rizon4R-062032')

    # Clear fault on the connected robot if any
    if robot.fault():
        logger.warn("Fault occurred on the connected robot, trying to clear ...")
        # Try to clear the fault
        if not robot.ClearFault():
            logger.error("Fault cannot be cleared, exiting ...")
        logger.info("Fault on the connected robot is cleared")

    # Enable the robot, make sure the E-stop is released before enabling
    logger.info("Enabling robot ...")
    robot.Enable()

    # Wait for the robot to become operational
    while not robot.operational():
        time.sleep(1)

    logger.info("Robot is now operational")

except Exception as e:
# Print exception error message
    logger.error(str(e))

print(f"tcp_pose: {['%.8f' % i for i in robot.states().tcp_pose]}")