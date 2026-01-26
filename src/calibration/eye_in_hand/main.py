#!/usr/bin/env python3
"""
Eye-in-Hand Calibration Entry Point

This script orchestrates the eye-in-hand calibration workflow:
1. Check if camera_pose.launch.py exists (from ROS2 MoveIt hand-eye calibration)
2. Export X to .npy format
3. Verify X using collected data (or collect data first if needed)

Usage:
    python src/calibration/eye_in_hand/main.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.log import get_logger

logger = get_logger(__name__)

# Paths
BASE_DIR = Path(__file__).parent
LAUNCH_FILE = BASE_DIR / "camera_pose.launch.py"
XVERIFICATION_DIR = BASE_DIR / "Xverification"
VERIFICATION_DATA = XVERIFICATION_DIR / "verification_data.json"


def check_launch_file_exists():
    """Check if camera_pose.launch.py exists."""
    if not LAUNCH_FILE.exists():
        logger.error("=" * 70)
        logger.error("ERROR: camera_pose.launch.py not found!")
        logger.error("=" * 70)
        logger.error("")
        logger.error(f"Expected location: {LAUNCH_FILE}")
        logger.error("")
        logger.error("This file comes from ROS2 MoveIt hand-eye calibration package.")
        logger.error("Please run the hand-eye calibration first using MoveIt.")
        logger.error("")
        logger.error("Steps:")
        logger.error("  1. Launch MoveIt hand-eye calibration")
        logger.error("  2. Collect calibration samples")
        logger.error("  3. Compute calibration")
        logger.error("  4. Save the result as camera_pose.launch.py")
        logger.error("=" * 70)
        return False

    logger.info(f"Found camera_pose.launch.py: {LAUNCH_FILE}")
    return True


def export_X_to_npy():
    """Export X from launch file to .npy format."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Step 1: Export X to .npy")
    logger.info("=" * 70)

    from src.calibration.eye_in_hand.export_X_to_npy import load_X_from_launch
    import numpy as np

    # Load X
    X, params = load_X_from_launch()

    logger.info("Loaded X from camera_pose.launch.py")
    logger.info(
        f"  Translation: [{params['x']:.6f}, {params['y']:.6f}, {params['z']:.6f}] m"
    )
    logger.info(
        f"  Quaternion:  [{params['qx']:.6f}, {params['qy']:.6f}, {params['qz']:.6f}, {params['qw']:.6f}]"
    )

    # Save to dataset/calibration/
    output_dir = PROJECT_ROOT / "dataset" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "X.npy"
    np.save(output_file, X)

    logger.info(f"Saved X to: {output_file}")

    return X


def check_verification_data_exists():
    """Check if verification data exists and is not empty."""
    if not XVERIFICATION_DIR.exists():
        logger.info(f"Xverification directory not found: {XVERIFICATION_DIR}")
        return False

    if not VERIFICATION_DATA.exists():
        logger.info(f"verification_data.json not found: {VERIFICATION_DATA}")
        return False

    # Check if there are any pose images
    pose_images = list(XVERIFICATION_DIR.glob("pose_*.png"))
    if len(pose_images) == 0:
        logger.info("No pose images found in Xverification directory")
        return False

    logger.info(f"Found {len(pose_images)} pose images in Xverification directory")
    return True


def collect_verification_data():
    """Run the data collection script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Step 2: Collect Verification Data")
    logger.info("=" * 70)
    logger.info("")
    logger.info("No verification data found. Starting data collection...")
    logger.info("")

    from src.calibration.eye_in_hand.collect_X_verification_data import (
        main as collect_main,
    )

    return collect_main()


def verify_X():
    """Run the X verification script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Step 3: Verify X")
    logger.info("=" * 70)

    from src.calibration.eye_in_hand.verify_X import main as verify_main

    return verify_main()


def main():
    logger.info("=" * 70)
    logger.info("Eye-in-Hand Calibration Pipeline")
    logger.info("=" * 70)

    # Step 0: Check if launch file exists
    if not check_launch_file_exists():
        return 1

    # Step 1: Export X to .npy
    try:
        export_X_to_npy()
    except Exception as e:
        logger.error(f"Failed to export X: {e}")
        return 1

    # Step 2: Check if verification data exists
    if not check_verification_data_exists():
        # Need to collect data first
        result = collect_verification_data()
        if result != 0:
            logger.error("Data collection failed or was cancelled")
            return result

    # Step 3: Verify X
    try:
        result = verify_X()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        return 1

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline Complete")
    logger.info("=" * 70)

    return result


if __name__ == "__main__":
    exit(main())
