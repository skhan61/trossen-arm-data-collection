#!/usr/bin/env python3
"""
Analyze robot pose change between the two verification poses.

Reads verification_data.json and computes how much the robot gripper moved.

Usage:
    python3 src/analyze_robot_pose_change.py
"""

import json
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
from scipy.spatial.transform import Rotation

# ============================================================================
# Logging Setup
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"robot_pose_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode="w"),
        logging.StreamHandler()
    ],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "Xverification"
JSON_FILE = DATA_DIR / "verification_data.json"


def main():
    logger.info("=" * 70)
    logger.info("Robot Pose Change Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Data file: {JSON_FILE}")
    logger.info("")

    # Load JSON
    if not JSON_FILE.exists():
        logger.error(f"File not found: {JSON_FILE}")
        return 1

    with open(JSON_FILE, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded: {data['description']}")
    logger.info(f"Number of poses: {data['num_poses']}")
    logger.info("")

    poses = data["poses"]

    # Analyze each pose
    for pose in poses:
        pose_id = pose["pose_id"]
        logger.info("-" * 70)
        logger.info(f"POSE {pose_id}")
        logger.info("-" * 70)

        # Joint positions
        joints = pose["robot"]["joint_positions"]
        logger.info("Joint positions (rad):")
        for i, j in enumerate(joints):
            logger.info(f"  Joint {i+1}: {j:8.4f} rad ({np.degrees(j):8.2f} deg)")
        logger.info("")

        # Cartesian position
        cart = pose["robot"]["cartesian_xyz_rpy"]
        logger.info("Cartesian position (from FK):")
        logger.info(f"  X = {cart[0]*1000:8.2f} mm  ({cart[0]:.6f} m)")
        logger.info(f"  Y = {cart[1]*1000:8.2f} mm  ({cart[1]:.6f} m)")
        logger.info(f"  Z = {cart[2]*1000:8.2f} mm  ({cart[2]:.6f} m)")
        logger.info(f"  Roll  = {np.degrees(cart[3]):8.2f} deg")
        logger.info(f"  Pitch = {np.degrees(cart[4]):8.2f} deg")
        logger.info(f"  Yaw   = {np.degrees(cart[5]):8.2f} deg")
        logger.info("")

        # T_gripper2base matrix
        T = np.array(pose["robot"]["T_gripper2base"])
        logger.info("T_gripper2base (4x4 matrix):")
        for row in T:
            logger.info(f"  [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}, {row[3]:10.6f}]")
        logger.info("")

    # Compare poses
    if len(poses) >= 2:
        logger.info("=" * 70)
        logger.info("COMPARISON: Pose 1 vs Pose 2")
        logger.info("=" * 70)
        logger.info("")

        # Cartesian positions
        cart1 = np.array(poses[0]["robot"]["cartesian_xyz_rpy"])
        cart2 = np.array(poses[1]["robot"]["cartesian_xyz_rpy"])

        pos1 = cart1[:3]
        pos2 = cart2[:3]
        rpy1 = cart1[3:]
        rpy2 = cart2[3:]

        pos_diff = pos2 - pos1
        rpy_diff = rpy2 - rpy1

        logger.info("GRIPPER POSITION CHANGE:")
        logger.info(f"  Pose 1: [{pos1[0]*1000:8.2f}, {pos1[1]*1000:8.2f}, {pos1[2]*1000:8.2f}] mm")
        logger.info(f"  Pose 2: [{pos2[0]*1000:8.2f}, {pos2[1]*1000:8.2f}, {pos2[2]*1000:8.2f}] mm")
        logger.info("")
        logger.info(f"  dX = {pos_diff[0]*1000:8.2f} mm")
        logger.info(f"  dY = {pos_diff[1]*1000:8.2f} mm")
        logger.info(f"  dZ = {pos_diff[2]*1000:8.2f} mm")
        logger.info(f"  Total translation = {np.linalg.norm(pos_diff)*1000:.2f} mm")
        logger.info("")

        logger.info("GRIPPER ORIENTATION CHANGE:")
        logger.info(f"  Pose 1: [{np.degrees(rpy1[0]):8.2f}, {np.degrees(rpy1[1]):8.2f}, {np.degrees(rpy1[2]):8.2f}] deg")
        logger.info(f"  Pose 2: [{np.degrees(rpy2[0]):8.2f}, {np.degrees(rpy2[1]):8.2f}, {np.degrees(rpy2[2]):8.2f}] deg")
        logger.info("")
        logger.info(f"  dRoll  = {np.degrees(rpy_diff[0]):8.2f} deg")
        logger.info(f"  dPitch = {np.degrees(rpy_diff[1]):8.2f} deg")
        logger.info(f"  dYaw   = {np.degrees(rpy_diff[2]):8.2f} deg")
        logger.info("")

        # Joint changes
        joints1 = np.array(poses[0]["robot"]["joint_positions"])
        joints2 = np.array(poses[1]["robot"]["joint_positions"])
        joint_diff = joints2 - joints1

        logger.info("JOINT POSITION CHANGES:")
        for i, d in enumerate(joint_diff):
            logger.info(f"  Joint {i+1}: {np.degrees(d):8.2f} deg")
        logger.info("")

        # Transformation matrices
        T1 = np.array(poses[0]["robot"]["T_gripper2base"])
        T2 = np.array(poses[1]["robot"]["T_gripper2base"])

        # Relative transformation: T2 = T_rel @ T1  =>  T_rel = T2 @ inv(T1)
        T_rel = T2 @ np.linalg.inv(T1)

        logger.info("RELATIVE TRANSFORMATION (T2 = T_rel @ T1):")
        for row in T_rel:
            logger.info(f"  [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}, {row[3]:10.6f}]")
        logger.info("")

        # Extract rotation angle from T_rel
        R_rel = T_rel[:3, :3]
        rot = Rotation.from_matrix(R_rel)
        angle = rot.magnitude()
        logger.info(f"Relative rotation magnitude: {np.degrees(angle):.2f} deg")
        logger.info(f"Relative translation: {np.linalg.norm(T_rel[:3, 3])*1000:.2f} mm")

    logger.info("")
    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
