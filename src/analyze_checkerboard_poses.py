#!/usr/bin/env python3
"""
Analyze checkerboard position in camera frame from saved images.

This script reads pose images and detects the checkerboard position
relative to the camera. Since the checkerboard didn't move between
images, it should appear at the same position in the camera frame.

Usage:
    python3 src/analyze_checkerboard_poses.py
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from datetime import datetime

# ============================================================================
# Logging Setup
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = (
    LOG_DIR / f"checkerboard_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

BOARD_SIZE = (8, 6)  # Internal corners (columns, rows)
SQUARE_SIZE = 0.025  # 25mm squares

# Camera intrinsics from RealSense D405
CAMERA_MATRIX = np.array(
    [
        [385.4389953613281, 0.0, 316.2204895019531],
        [0.0, 384.87786865234375, 240.33718872070312],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)

DIST_COEFFS = np.array(
    [
        -0.05274650454521179,
        0.05903208255767822,
        -0.00027341238455846906,
        0.0007520649232901633,
        -0.018650885671377182,
    ],
    dtype=np.float32,
)

# Image paths
DATA_DIR = Path(__file__).parent.parent / "data" / "Xverification"
IMAGE_FILES = [
    DATA_DIR / "pose_001.png",
    DATA_DIR / "pose_002.png",
]


# ============================================================================
# Functions
# ============================================================================


def detect_checkerboard(image, camera_matrix, dist_coeffs):
    """
    Detect checkerboard and compute its pose in camera frame.

    Returns:
        success: bool
        corners: detected corner points
        rvec: rotation vector
        tvec: translation vector (position in camera frame)
        T_target2cam: 4x4 transformation matrix
    """
    logger.debug(f"Image shape: {image.shape}")

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    logger.debug(f"Grayscale shape: {gray.shape}")

    # Find checkerboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)

    logger.info(f"Checkerboard detection: {'SUCCESS' if ret else 'FAILED'}")

    if not ret:
        return False, None, None, None, None

    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    logger.debug(f"Refined {len(corners)} corners")

    # Create object points (checkerboard in its own frame, Z=0)
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : BOARD_SIZE[0], 0 : BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Solve PnP to get pose
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

    if not success:
        logger.error("solvePnP failed")
        return False, corners, None, None, None

    # Convert to 4x4 matrix
    R, _ = cv2.Rodrigues(rvec)
    T_target2cam = np.eye(4)
    T_target2cam[:3, :3] = R
    T_target2cam[:3, 3] = tvec.flatten()

    return True, corners, rvec, tvec, T_target2cam


def main():
    logger.info("=" * 70)
    logger.info("Checkerboard Position Analysis")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Board size: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} internal corners")
    logger.info(f"Square size: {SQUARE_SIZE * 1000:.1f} mm")
    logger.info("")

    # Store results
    results = []

    for img_path in IMAGE_FILES:
        logger.info("-" * 70)
        logger.info(f"Processing: {img_path.name}")
        logger.info("-" * 70)

        if not img_path.exists():
            logger.error(f"File not found: {img_path}")
            continue

        # Read image
        image = cv2.imread(str(img_path))
        if image is None:
            logger.error(f"Failed to read image: {img_path}")
            continue

        logger.info(f"Image loaded: {image.shape[1]}x{image.shape[0]} pixels")

        # Detect checkerboard
        success, corners, rvec, tvec, T_target2cam = detect_checkerboard(
            image, CAMERA_MATRIX, DIST_COEFFS
        )

        if not success:
            logger.error(f"Failed to detect checkerboard in {img_path.name}")
            continue

        # Log position in camera frame
        pos = tvec.flatten()
        logger.info("")
        logger.info("CHECKERBOARD POSITION IN CAMERA FRAME:")
        logger.info(f"  X = {pos[0]*1000:8.2f} mm  ({pos[0]:.6f} m)")
        logger.info(f"  Y = {pos[1]*1000:8.2f} mm  ({pos[1]:.6f} m)")
        logger.info(f"  Z = {pos[2]*1000:8.2f} mm  ({pos[2]:.6f} m)")
        logger.info(f"  Distance = {np.linalg.norm(pos)*1000:.2f} mm")
        logger.info("")

        # Log rotation
        rvec_flat = rvec.flatten()
        logger.info("ROTATION (Rodrigues vector):")
        logger.info(
            f"  rx = {rvec_flat[0]:.6f} rad ({np.degrees(rvec_flat[0]):.2f} deg)"
        )
        logger.info(
            f"  ry = {rvec_flat[1]:.6f} rad ({np.degrees(rvec_flat[1]):.2f} deg)"
        )
        logger.info(
            f"  rz = {rvec_flat[2]:.6f} rad ({np.degrees(rvec_flat[2]):.2f} deg)"
        )
        logger.info("")

        # Log full transformation matrix
        logger.info("T_target2cam (4x4 matrix):")
        for row in T_target2cam:
            logger.info(
                f"  [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}, {row[3]:10.6f}]"
            )
        logger.info("")

        results.append(
            {
                "file": img_path.name,
                "tvec": pos,
                "rvec": rvec_flat,
                "T_target2cam": T_target2cam,
            }
        )

    # Compare results
    if len(results) >= 2:
        logger.info("=" * 70)
        logger.info("COMPARISON")
        logger.info("=" * 70)
        logger.info("")

        pos1 = results[0]["tvec"]
        pos2 = results[1]["tvec"]

        diff = pos2 - pos1
        diff_norm = np.linalg.norm(diff)

        logger.info(f"Position in camera frame:")
        logger.info(
            f"  Pose 1: [{pos1[0]*1000:8.2f}, {pos1[1]*1000:8.2f}, {pos1[2]*1000:8.2f}] mm"
        )
        logger.info(
            f"  Pose 2: [{pos2[0]*1000:8.2f}, {pos2[1]*1000:8.2f}, {pos2[2]*1000:8.2f}] mm"
        )
        logger.info("")
        logger.info(f"Difference:")
        logger.info(f"  dX = {diff[0]*1000:8.2f} mm")
        logger.info(f"  dY = {diff[1]*1000:8.2f} mm")
        logger.info(f"  dZ = {diff[2]*1000:8.2f} mm")
        logger.info(f"  Total = {diff_norm*1000:.2f} mm")
        logger.info("")

        # This difference is EXPECTED because the camera moved!
        # The checkerboard is fixed, but the camera is on the gripper
        logger.info("NOTE: The positions ARE DIFFERENT because the camera moved")
        logger.info("      (camera is mounted on the gripper which moved).")
        logger.info("")
        logger.info("      To verify calibration, we need to transform these")
        logger.info("      positions to the BASE frame using:")
        logger.info("        T_target_in_base = T_gripper2base @ X @ T_target2cam")
        logger.info("")
        logger.info("      If calibration X is correct, the base-frame positions")
        logger.info("      should match (since checkerboard didn't move).")

    logger.info("=" * 70)
    logger.info("DONE")
    logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
