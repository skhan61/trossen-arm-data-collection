#!/usr/bin/env python3
"""
Compute hand-eye calibration from existing collected data.
"""

import cv2
import numpy as np
import json
import logging
from pathlib import Path

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "compute_hand_eye.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {LOG_FILE}")

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "hand_eye_calibration_data"

logger.info("=" * 60)
logger.info("Computing Hand-Eye Calibration from Existing Data")
logger.info("=" * 60)

# Load existing data
data_file = DATA_DIR / "hand_eye_data.json"
if not data_file.exists():
    logger.error(f"Data file not found: {data_file}")
    logger.error("Please run hand_eye_calibration.py first to collect data")
    exit(1)

logger.info(f"Loading data from {data_file}")
with open(data_file, "r") as f:
    data = json.load(f)

# Convert to numpy arrays
R_gripper2base = [np.array(R) for R in data["R_gripper2base"]]
t_gripper2base = [np.array(t) for t in data["t_gripper2base"]]
R_target2cam = [np.array(R) for R in data["R_target2cam"]]
t_target2cam = [np.array(t) for t in data["t_target2cam"]]

num_poses = len(R_gripper2base)
logger.info(f"Loaded {num_poses} poses")

if num_poses < 15:
    logger.error(f"Not enough poses: {num_poses} (need at least 15)")
    exit(1)

# Compute hand-eye calibration
logger.info("Computing hand-eye calibration using TSAI method...")
try:
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI,
    )

    logger.info("=" * 60)
    logger.info("Hand-Eye Calibration Result")
    logger.info("=" * 60)
    logger.info("\nRotation matrix (camera to gripper):")
    logger.info(f"\n{R_cam2gripper}")
    logger.info("\nTranslation vector (camera to gripper) [meters]:")
    logger.info(f"{t_cam2gripper.flatten()}")
    logger.info("=" * 60)

    # Save result
    result = {
        "R_cam2gripper": R_cam2gripper.tolist(),
        "t_cam2gripper": t_cam2gripper.tolist(),
        "num_poses": num_poses,
    }

    output_file = DATA_DIR / "hand_eye_calibration.json"
    with open(output_file, "w") as f:
        json.dump(result, f, indent=2)

    logger.info(f"\nCalibration saved to {output_file}")
    logger.info("Done!")

except Exception as e:
    logger.error(f"Failed to compute calibration: {type(e).__name__}: {e}")
    import traceback
    logger.error(traceback.format_exc())
    exit(1)
