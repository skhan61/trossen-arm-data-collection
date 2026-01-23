#!/usr/bin/env python3
"""
Test robot joint limits by moving each joint step by step.

Moves each joint individually from its minimum to maximum limit in steps.
"""

import time
import logging
import numpy as np
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode
from pathlib import Path

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "test_joint_limits.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ARM_IP = "192.168.1.99"

# Joint limits for WidowX AI 250 (radians)
# Based on hardware specifications with 5% safety margin
JOINT_LIMITS = [
    (-3.05, 3.05),        # Joint 0: base rotation (±175°)
    (-1.57, 1.92),        # Joint 1: shoulder (±90° to 110°)
    (-1.92, 1.57),        # Joint 2: elbow (-110° to 90°)
    (-2.09, 2.09),        # Joint 3: wrist 1 (±120°)
    (-1.92, 1.92),        # Joint 4: wrist 2 (±110°)
    (-2.09, 2.09),        # Joint 5: wrist 3 (±120°)
    (-0.02, 0.02),        # Joint 6: gripper (very small range)
]

# Number of steps to divide each joint's range
NUM_STEPS = 10

# Time to wait at each step (seconds)
STEP_DURATION = 2.0


def move_joint_through_range(driver, joint_index, joint_limits, num_steps=10, step_duration=2.0):
    """
    Move a single joint through its entire range in steps.

    Args:
        driver: Robot driver instance
        joint_index: Index of joint to move (0-6)
        joint_limits: Tuple of (min, max) limits for the joint in radians
        num_steps: Number of steps to divide the range into
        step_duration: Time to wait at each step in seconds
    """
    min_limit, max_limit = joint_limits
    joint_range = max_limit - min_limit

    logger.info("")
    logger.info("=" * 60)
    logger.info(f"Testing Joint {joint_index}")
    logger.info("=" * 60)
    logger.info(f"  Range: {np.degrees(min_limit):.1f}° to {np.degrees(max_limit):.1f}°")
    logger.info(f"  Steps: {num_steps}")
    logger.info(f"  Duration per step: {step_duration}s")
    logger.info("")

    # Get current positions
    current_positions = list(driver.get_all_positions())

    # Move from min to max
    logger.info(f"Moving joint {joint_index} from MIN to MAX...")
    for step in range(num_steps + 1):
        # Calculate target angle for this step
        target_angle = min_limit + (joint_range * step / num_steps)

        # Set target position for this joint only
        target_positions = current_positions.copy()
        target_positions[joint_index] = target_angle

        # Move to target
        driver.set_all_positions(target_positions, goal_time=1.0)
        time.sleep(step_duration)

        # Log progress
        current_angle = driver.get_all_positions()[joint_index]
        logger.info(f"  Step {step}/{num_steps}: Target={np.degrees(target_angle):6.1f}°, "
                   f"Actual={np.degrees(current_angle):6.1f}°")

    logger.info(f"Joint {joint_index} reached MAX limit")
    logger.info("")

    # Move back from max to min
    logger.info(f"Moving joint {joint_index} from MAX to MIN...")
    for step in range(num_steps + 1):
        # Calculate target angle for this step (going backwards)
        target_angle = max_limit - (joint_range * step / num_steps)

        # Set target position for this joint only
        target_positions = current_positions.copy()
        target_positions[joint_index] = target_angle

        # Move to target
        driver.set_all_positions(target_positions, goal_time=1.0)
        time.sleep(step_duration)

        # Log progress
        current_angle = driver.get_all_positions()[joint_index]
        logger.info(f"  Step {step}/{num_steps}: Target={np.degrees(target_angle):6.1f}°, "
                   f"Actual={np.degrees(current_angle):6.1f}°")

    logger.info(f"Joint {joint_index} returned to MIN limit")
    logger.info("")

    # Return to starting position
    logger.info(f"Returning joint {joint_index} to starting position...")
    driver.set_all_positions(current_positions, goal_time=2.0)
    time.sleep(2.5)
    logger.info(f"Joint {joint_index} returned to start")


def main():
    logger.info("=" * 60)
    logger.info("Robot Joint Limits Test")
    logger.info("=" * 60)
    logger.info("")
    logger.info("This script will move each joint through its full range")
    logger.info("in steps, from minimum to maximum and back.")
    logger.info("")
    logger.info("WARNING: Make sure the robot has clear space to move!")
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
    num_joints = driver.get_num_joints()
    logger.info(f"Number of joints: {num_joints}")
    logger.info("")

    # Record initial position
    initial_positions = list(driver.get_all_positions())
    logger.info("Initial joint positions:")
    for i, pos in enumerate(initial_positions):
        logger.info(f"  Joint {i}: {np.degrees(pos):6.1f}°")
    logger.info("")

    # Move to home position first
    logger.info("Moving to home position...")
    driver.set_all_modes(Mode.position)
    home_position = [0.0] * num_joints
    driver.set_all_positions(home_position, goal_time=3.0)
    time.sleep(3.5)
    logger.info("At home position")
    logger.info("")

    try:
        # Test each joint (skip gripper joint 6)
        for joint_idx in range(num_joints - 1):  # 0-5, skip gripper
            logger.info(f"Press Enter to test Joint {joint_idx}, or 'q' to quit...")
            user_input = input()
            if user_input.lower() == 'q':
                logger.info("Test cancelled by user")
                break

            move_joint_through_range(
                driver=driver,
                joint_index=joint_idx,
                joint_limits=JOINT_LIMITS[joint_idx],
                num_steps=NUM_STEPS,
                step_duration=STEP_DURATION
            )

    except KeyboardInterrupt:
        logger.info("")
        logger.info("Test interrupted by user")

    finally:
        # Return to home position
        logger.info("")
        logger.info("Returning to home position...")
        driver.set_all_positions(home_position, goal_time=3.0)
        time.sleep(3.5)

        driver.cleanup()
        logger.info("Done")
        logger.info("")


if __name__ == "__main__":
    main()
