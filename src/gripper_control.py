#!/usr/bin/env python3
"""
Gripper Control Script for Gelsight Calibration

This script provides control over the gripper opening for calibrating
T_camera_to_gelsight(u) at different gripper openings.

Features:
1. Connect to Trossen arm
2. Control gripper opening with keyboard
3. Step through predefined gripper openings for calibration

Usage:
    python src/gripper_control.py
"""

import time
import logging
from pathlib import Path
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# ============================================================================
# Configuration
# ============================================================================

# Robot IP address
ARM_IP = "192.168.1.99"

# Gripper opening range (in meters) - adjust based on your gripper
GRIPPER_MIN = 0.0  # Fully closed
GRIPPER_MAX = 0.042  # Fully open (42mm based on discussion notes)

# Step size for gripper control (in meters)
GRIPPER_STEP = 0.002  # 2mm steps

# Predefined calibration openings (in meters)
CALIBRATION_OPENINGS = [0.026, 0.028, 0.030, 0.032, 0.034, 0.036, 0.038, 0.040, 0.042]

# Directories
BASE_DIR = Path(__file__).parent.parent
LOG_DIR = BASE_DIR / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Logging setup
LOG_FILE = LOG_DIR / "gripper_control.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class GripperController:
    """Controller for Trossen arm gripper."""

    def __init__(self, ip_address: str = ARM_IP):
        self.ip_address = ip_address
        self.driver = None
        self.num_joints = None
        self.current_gripper_opening = None

    def connect(self):
        """Connect to the robot."""
        logger.info(f"Connecting to robot at {self.ip_address}...")
        self.driver = TrossenArmDriver()
        self.driver.configure(
            model=Model.wxai_v0,
            end_effector=StandardEndEffector.wxai_v0_leader,
            serv_ip=self.ip_address,
            clear_error=True,
            timeout=10.0,
        )
        self.num_joints = self.driver.get_num_joints()
        logger.info(f"Robot connected ({self.num_joints} joints)")
        return True

    def disconnect(self):
        """Disconnect from the robot."""
        if self.driver:
            self.driver.cleanup()
            logger.info("Robot disconnected")

    def go_home(self, goal_time: float = 3.0):
        """Move robot to home position."""
        logger.info("Moving to home position...")
        self.driver.set_all_modes(Mode.position)
        home_position = [0.0] * self.num_joints
        self.driver.set_all_positions(home_position, goal_time=goal_time)
        time.sleep(goal_time + 0.5)
        logger.info("At home position")

    def get_gripper_position(self) -> float:
        """Get current gripper opening in meters."""
        # The gripper position is typically the last value or accessed via specific API
        gripper_pos = self.driver.get_gripper_position()
        self.current_gripper_opening = gripper_pos
        return gripper_pos

    def set_gripper_position(self, opening: float, goal_time: float = 1.0):
        """
        Set gripper opening.

        Args:
            opening: Desired gripper opening in meters
            goal_time: Time to reach the position
        """
        # Clamp to valid range
        opening = max(GRIPPER_MIN, min(GRIPPER_MAX, opening))

        logger.info(f"Setting gripper to {opening * 1000:.1f}mm...")
        # Ensure gripper is in position mode
        self.driver.set_gripper_mode(Mode.position)
        self.driver.set_gripper_position(opening, goal_time=goal_time)
        time.sleep(goal_time + 0.2)

        # Verify position
        actual = self.get_gripper_position()
        logger.info(f"Gripper at {actual * 1000:.1f}mm")
        self.current_gripper_opening = actual
        return actual

    def open_gripper(self, goal_time: float = 1.0):
        """Fully open the gripper."""
        return self.set_gripper_position(GRIPPER_MAX, goal_time)

    def close_gripper(self, goal_time: float = 1.0):
        """Fully close the gripper."""
        return self.set_gripper_position(GRIPPER_MIN, goal_time)

    def step_gripper(self, direction: int, step: float = GRIPPER_STEP):
        """
        Step gripper opening.

        Args:
            direction: +1 to open, -1 to close
            step: Step size in meters
        """
        current = self.get_gripper_position()
        new_opening = current + (direction * step)
        return self.set_gripper_position(new_opening)


def print_controls():
    """Print keyboard controls."""
    print("\n" + "=" * 50)
    print("Gripper Control")
    print("=" * 50)
    print("Controls:")
    print("  o     - Open gripper fully")
    print("  c     - Close gripper fully")
    print("  +/=   - Open gripper by step")
    print("  -     - Close gripper by step")
    print("  1-9   - Go to calibration position 1-9")
    print("  g     - Get current gripper position")
    print("  h     - Go to home position")
    print("  q     - Quit")
    print("=" * 50)
    print(f"Step size: {GRIPPER_STEP * 1000:.1f}mm")
    print(f"Calibration openings: {[f'{x*1000:.0f}mm' for x in CALIBRATION_OPENINGS]}")
    print("=" * 50 + "\n")


def main():
    logger.info("=" * 70)
    logger.info("Gripper Control for Gelsight Calibration")
    logger.info("=" * 70)

    controller = GripperController()

    try:
        controller.connect()
        controller.go_home()

        # Get initial gripper position
        pos = controller.get_gripper_position()
        logger.info(f"Initial gripper position: {pos * 1000:.1f}mm")

        print_controls()

        while True:
            try:
                cmd = input("Command: ").strip().lower()

                if cmd == "q":
                    logger.info("Quit requested")
                    break

                elif cmd == "o":
                    controller.open_gripper()

                elif cmd == "c":
                    controller.close_gripper()

                elif cmd in ["+", "="]:
                    controller.step_gripper(+1)

                elif cmd == "-":
                    controller.step_gripper(-1)

                elif cmd == "g":
                    pos = controller.get_gripper_position()
                    print(f"Current gripper position: {pos * 1000:.1f}mm")

                elif cmd == "h":
                    controller.go_home()

                elif cmd.isdigit() and 1 <= int(cmd) <= len(CALIBRATION_OPENINGS):
                    idx = int(cmd) - 1
                    opening = CALIBRATION_OPENINGS[idx]
                    print(
                        f"Moving to calibration position {cmd}: {opening * 1000:.0f}mm"
                    )
                    controller.set_gripper_position(opening)

                elif cmd:
                    print(f"Unknown command: {cmd}")
                    print_controls()

            except KeyboardInterrupt:
                print("\nInterrupted")
                break

    finally:
        controller.go_home()
        controller.disconnect()
        logger.info("Done")

    return 0


if __name__ == "__main__":
    exit(main())
