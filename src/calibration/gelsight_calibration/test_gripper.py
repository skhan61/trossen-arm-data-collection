#!/usr/bin/env python3
"""
Simple script to test gripper opening and measure what the position values mean.

Usage:
    .venv/bin/python src/calibration/gelsight_calibration/test_gripper.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import time
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode
from src.utils.config import get_arm_ip


def main():
    arm_ip = get_arm_ip()
    print(f"Connecting to robot at {arm_ip}...")

    driver = TrossenArmDriver()
    driver.configure(
        model=Model.wxai_v0,
        end_effector=StandardEndEffector.wxai_v0_leader,
        serv_ip=arm_ip,
        clear_error=True,
        timeout=10.0,
    )
    print("Connected!")

    try:
        # Read current position
        current = driver.get_gripper_position()
        print(f"\nCurrent gripper position: {current * 1000:.2f} mm")

        # Open to maximum
        print("\nOpening gripper to maximum (0.042m = 42mm)...")
        driver.set_gripper_mode(Mode.position)
        driver.set_gripper_position(0.042, goal_time=2.0)
        time.sleep(2.5)

        max_pos = driver.get_gripper_position()
        print(f"Max position reading: {max_pos * 1000:.2f} mm")
        print("\n>>> MEASURE the physical jaw-to-jaw distance with a ruler! <<<")
        print(">>> Compare to the position reading above <<<")

        input("\nPress Enter to close gripper...")

        # Close gripper
        print("Closing gripper to minimum (0.0m)...")
        driver.set_gripper_position(0.0, goal_time=2.0)
        time.sleep(2.5)

        min_pos = driver.get_gripper_position()
        print(f"Min position reading: {min_pos * 1000:.2f} mm")

    finally:
        driver.cleanup()
        print("\nDone!")


if __name__ == "__main__":
    main()
