#!/usr/bin/env python3
"""
Verify T_gripper2base: Move robot to saved poses and compare cartesian positions.

This does NOT use ROS - it uses the Trossen ARM Python API directly.
"""

import json
import numpy as np
import cv2
import logging
import time
from pathlib import Path
from datetime import datetime

from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# ============================================================================
#                               Logging Setup
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = (
    LOG_DIR / f"verify_T_gripper2base_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

ARM_IP = "192.168.1.99"

DATA_DIR = Path(__file__).parent.parent / "data" / "Xverification"
JSON_FILE = DATA_DIR / "verification_data.json"


def main():
    logger.info("=" * 70)
    logger.info("Verify T_gripper2base (using Trossen ARM API)")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"JSON file: {JSON_FILE}")
    logger.info("")

    # Load JSON
    if not JSON_FILE.exists():
        logger.error(f"File not found: {JSON_FILE}")
        return 1

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {data['num_poses']} poses from JSON")
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
    logger.info("Robot connected!")
    logger.info("")

    results = []

    try:
        for pose in data["poses"]:
            pose_id = pose["pose_id"]
            saved_joints = pose["robot"]["joint_positions"][:6]  # First 6 joints
            saved_cartesian = pose["robot"]["cartesian_xyz_rpy"]
            saved_translation = saved_cartesian[:3]

            logger.info("=" * 70)
            logger.info(f"POSE {pose_id}")
            logger.info("=" * 70)
            logger.info("")
            logger.info(f"Saved joints (rad): {[f'{j:.3f}' for j in saved_joints]}")
            logger.info(f"Saved cartesian: {[f'{c:.4f}' for c in saved_cartesian]}")
            logger.info(
                f"Saved translation (mm): [{saved_translation[0]*1000:.2f}, {saved_translation[1]*1000:.2f}, {saved_translation[2]*1000:.2f}]"
            )
            logger.info("")

            # Move robot (need 7 joints: 6 arm + 1 gripper)
            logger.info("Moving robot to saved joint positions...")
            driver.set_all_modes(Mode.position)
            joints_with_gripper = list(saved_joints) + [0.0]  # Add gripper position
            driver.set_all_positions(joints_with_gripper, goal_time=3.0)
            time.sleep(4.0)
            logger.info("Motion complete")
            logger.info("")

            # Get current cartesian
            current_cartesian = list(driver.get_cartesian_positions())
            current_translation = current_cartesian[:3]

            logger.info(f"Current cartesian: {[f'{c:.4f}' for c in current_cartesian]}")
            logger.info(
                f"Current translation (mm): [{current_translation[0]*1000:.2f}, {current_translation[1]*1000:.2f}, {current_translation[2]*1000:.2f}]"
            )
            logger.info("")

            # Compare
            diff = np.array(current_translation) - np.array(saved_translation)
            diff_norm = np.linalg.norm(diff) * 1000  # mm

            logger.info("COMPARISON:")
            logger.info(
                f"  Saved:   [{saved_translation[0]*1000:.2f}, {saved_translation[1]*1000:.2f}, {saved_translation[2]*1000:.2f}] mm"
            )
            logger.info(
                f"  Current: [{current_translation[0]*1000:.2f}, {current_translation[1]*1000:.2f}, {current_translation[2]*1000:.2f}] mm"
            )
            logger.info(
                f"  Diff:    [{diff[0]*1000:.2f}, {diff[1]*1000:.2f}, {diff[2]*1000:.2f}] mm"
            )
            logger.info(f"  Total:   {diff_norm:.2f} mm")
            logger.info("")

            results.append(
                {
                    "pose_id": pose_id,
                    "diff_mm": diff_norm,
                }
            )

        # Summary
        logger.info("=" * 70)
        logger.info("SUMMARY")
        logger.info("=" * 70)
        logger.info("")

        for r in results:
            status = "OK" if r["diff_mm"] < 1.0 else "CHECK"
            logger.info(f"Pose {r['pose_id']}: {r['diff_mm']:.2f} mm [{status}]")

        logger.info("")
        if results:
            avg_diff = np.mean([r["diff_mm"] for r in results])
            if avg_diff < 1.0:
                logger.info(f"RESULT: All poses match (avg {avg_diff:.2f} mm)")
                logger.info("        Your T_gripper2base data is CORRECT")
            else:
                logger.info(f"RESULT: Average difference = {avg_diff:.2f} mm")

    finally:
        logger.info("")
        logger.info("Cleaning up...")
        driver.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
