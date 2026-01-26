#!/usr/bin/env python3
"""
GelSight Calibration Entry Point

This script orchestrates the gelsight calibration workflow:
1. Collect calibration data (if not exists)
2. Compute calibration and save T_u params to dataset/calibration/
3. Verify calibration

Usage:
    python src/calibration/gelsight_calibration/run_gelsight_calibration.py
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
DATA_DIR = BASE_DIR / "gelsight_calibration_data"
CALIBRATION_OUTPUT_DIR = PROJECT_ROOT / "dataset" / "calibration"


def check_calibration_data_exists():
    """Check if calibration data exists in DATA_DIR."""
    data_file = DATA_DIR / "calibration_data.json"
    if not data_file.exists():
        logger.info(f"calibration_data.json not found in {DATA_DIR}")
        return False

    rgb_images = list(DATA_DIR.glob("*_rgb.png"))
    if len(rgb_images) == 0:
        logger.info("No RGB images found in DATA_DIR")
        return False

    logger.info(f"Found {len(rgb_images)} RGB images in {DATA_DIR}")
    return True


def collect_calibration_data(**kwargs):
    """Run the data collection script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Step 1: Collect Calibration Data")
    logger.info("=" * 70)
    logger.info("")

    from src.calibration.gelsight_calibration.collect_gelsight_calibration_data import (
        collect,
    )

    gripper_min = kwargs.get("gripper_min", 26.0)
    gripper_max = kwargs.get("gripper_max", 42.0)
    num_openings = kwargs.get("num_openings", 17)
    repeats = kwargs.get("repeats", 3)

    collect(
        gripper_min=gripper_min,
        gripper_max=gripper_max,
        num_openings=num_openings,
        repeats=repeats,
    )


def compute_calibration(visualize: bool = False, num_manual: int = 1):
    """Run the calibration computation script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Step 2: Compute Calibration")
    logger.info("=" * 70)

    from src.calibration.gelsight_calibration.compute_gelsight_calibration import (
        GelsightCalibrator,
    )

    calibrator = GelsightCalibrator(pose_dir=DATA_DIR)
    result = calibrator.run(visualize=visualize, num_manual_samples=num_manual)

    return result


def verify_calibration():
    """Run the calibration verification script."""
    logger.info("")
    logger.info("=" * 70)
    logger.info("Step 3: Verify Calibration")
    logger.info("=" * 70)

    # Check if npy files exist
    left_file = CALIBRATION_OUTPUT_DIR / "T_u_left_params.npy"
    right_file = CALIBRATION_OUTPUT_DIR / "T_u_right_params.npy"

    if left_file.exists():
        import numpy as np

        left_params = np.load(left_file)
        logger.info(f"LEFT params: {left_file}")
        logger.info(
            f"  t0: [{left_params[0] * 1000:.3f}, {left_params[1] * 1000:.3f}, {left_params[2] * 1000:.3f}] mm"
        )
        logger.info(
            f"  k:  [{left_params[3] * 1000:.3f}, {left_params[4] * 1000:.3f}, {left_params[5] * 1000:.3f}] mm/m"
        )

    if right_file.exists():
        import numpy as np

        right_params = np.load(right_file)
        logger.info(f"RIGHT params: {right_file}")
        logger.info(
            f"  t0: [{right_params[0] * 1000:.3f}, {right_params[1] * 1000:.3f}, {right_params[2] * 1000:.3f}] mm"
        )
        logger.info(
            f"  k:  [{right_params[3] * 1000:.3f}, {right_params[4] * 1000:.3f}, {right_params[5] * 1000:.3f}] mm/m"
        )

    logger.info("Verification complete")


def main(visualize: bool = False, num_manual: int = 1, force_collect: bool = False):
    """
    Run the full gelsight calibration pipeline.

    Args:
        visualize: Show detection visualization during compute
        num_manual: Number of samples for manual selection
        force_collect: Force data collection even if data exists
    """
    logger.info("=" * 70)
    logger.info("GelSight Calibration Pipeline")
    logger.info("=" * 70)
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Output directory: {CALIBRATION_OUTPUT_DIR}")

    # Step 1: Check if data exists, collect if not
    if force_collect or not check_calibration_data_exists():
        try:
            collect_calibration_data()
            if not check_calibration_data_exists():
                logger.error("Data collection completed but no data found")
                return 1
        except Exception as e:
            logger.error(f"Data collection failed: {e}")
            import traceback

            traceback.print_exc()
            return 1
    else:
        logger.info("")
        logger.info("=" * 70)
        logger.info("Step 1: Using Existing Calibration Data")
        logger.info("=" * 70)
        logger.info(f"Found existing data in {DATA_DIR}")

    # Step 2: Compute calibration (saves npy files directly)
    try:
        result = compute_calibration(visualize=visualize, num_manual=num_manual)
        if result is None:
            logger.error("Calibration computation failed")
            return 1
    except Exception as e:
        logger.error(f"Calibration computation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Step 3: Verify calibration
    try:
        verify_calibration()
    except Exception as e:
        logger.error(f"Verification failed: {e}")
        import traceback

        traceback.print_exc()

    logger.info("")
    logger.info("=" * 70)
    logger.info("Pipeline Complete")
    logger.info("=" * 70)
    logger.info(f"Output: {CALIBRATION_OUTPUT_DIR}/T_u_left_params.npy")
    logger.info(f"Output: {CALIBRATION_OUTPUT_DIR}/T_u_right_params.npy")

    return 0


if __name__ == "__main__":
    import fire

    fire.Fire(main)
