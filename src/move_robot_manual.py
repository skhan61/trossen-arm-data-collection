#!/usr/bin/env python3
"""
Move robot to random positions automatically, then return home.
"""

import time
import logging
import random
from pathlib import Path
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "move_robot_manual.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ARM_IP = "192.168.1.99"

logger.info("=" * 80)
logger.info("ROBOT AUTOMATIC RANDOM MOVEMENT")
logger.info("=" * 80)
logger.info("")

# Connect to robot
logger.info(f"Connecting to robot at {ARM_IP}...")
driver = TrossenArmDriver()
driver.configure(
    model=Model.wxai_v0,
    end_effector=StandardEndEffector.wxai_v0_follower,
    serv_ip=ARM_IP,
    clear_error=True,
    timeout=10.0,
)
logger.info("Robot connected")
logger.info("")

# Clear any errors and wait
logger.info("Clearing errors and waiting 2 seconds...")
time.sleep(2.0)

# Store home position
num_joints = driver.get_num_joints()
home_position = [0.0] * num_joints

# Move to home first
logger.info("Moving to home position...")
driver.set_all_modes(Mode.position)
driver.set_all_positions(home_position, goal_time=8.0)
time.sleep(8.5)
logger.info("At home position")
logger.info("")

try:
    # Move to 5 random positions
    for i in range(5):
        # Generate bigger random movements but keep gripper joint (last) near zero
        random_position = []
        for j in range(num_joints):
            if j == num_joints - 1:  # Last joint (gripper) - keep near zero
                random_position.append(random.uniform(-0.01, 0.01))
            else:  # Other joints - bigger movements
                random_position.append(random.uniform(-0.8, 0.8))

        logger.info(f"Moving to random position {i+1}/5...")
        logger.info(f"  Joint positions: {[f'{x:.3f}' for x in random_position]}")
        driver.set_all_positions(random_position, goal_time=5.0)
        time.sleep(5.5)

        cart_pos = list(driver.get_cartesian_positions())
        logger.info(f"Position {i+1}: X={cart_pos[0]:.3f}, Y={cart_pos[1]:.3f}, Z={cart_pos[2]:.3f}")
        logger.info("")

except KeyboardInterrupt:
    logger.info("")
    logger.info("Ctrl+C detected - exiting...")

finally:
    logger.info("Returning to home position...")
    driver.set_all_modes(Mode.position)
    driver.set_all_positions(home_position, goal_time=6.0)
    time.sleep(6.5)

    driver.cleanup()
    logger.info("Robot returned to home and disconnected")
    logger.info("=" * 80)
