#!/usr/bin/env python3
"""
Analyze X (hand-eye calibration) Verification Data - TF Version

Works with data collected using verify_X_calibration_tf.py

Usage:
    python src/analyze_X_verification_tf.py
"""

import json
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Xverification_tf"


def load_verification_data():
    """Load the collected verification data."""
    json_path = DATA_DIR / "verification_data.json"

    if not json_path.exists():
        logger.error(f"Verification data not found: {json_path}")
        logger.error("Run verify_X_calibration_tf.py first to collect data.")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {data['num_poses']} poses from {json_path}")
    logger.info(f"Data source: {data.get('tf_source', 'unknown')}")
    return data


def analyze_poses(data):
    """Analyze all poses and compute T_target_in_base for each."""
    if "X_cam_in_gripper" in data:
        X_cam2gripper = np.array(data["X_cam_in_gripper"])
    else:
        X_cam2gripper = np.array(data["X_cam2gripper"])

    poses = data["poses"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("X Matrix (Camera to Gripper)")
    logger.info("=" * 70)
    t = X_cam2gripper[:3, 3]
    logger.info(f"Translation: x={t[0]*100:.2f}cm, y={t[1]*100:.2f}cm, z={t[2]*100:.2f}cm")
    logger.info("")

    all_translations = []
    all_rotations = []

    logger.info("=" * 70)
    logger.info("Computing T_target_in_base for each pose")
    logger.info("=" * 70)
    logger.info("")

    for pose in poses:
        pose_id = pose["pose_id"]

        # Get T_gripper_in_base from TF data
        T_gripper_in_base = np.array(pose["robot"]["T_gripper2base"])

        # Get T_target_in_camera from checkerboard detection
        T_target_in_camera = np.array(pose["checkerboard"]["T_target2cam"])

        # Compute T_target_in_base
        T_target_in_base = T_gripper_in_base @ X_cam2gripper @ T_target_in_camera

        all_translations.append(T_target_in_base[:3, 3])
        all_rotations.append(T_target_in_base[:3, :3])

        t = T_target_in_base[:3, 3]
        logger.info(f"Pose {pose_id}: T_target_in_base = [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")

    return all_translations, all_rotations


def compute_consistency_metrics(all_translations, all_rotations):
    """Compute metrics showing how consistent T_target_in_base is across poses."""
    translations = np.array(all_translations)

    logger.info("")
    logger.info("=" * 70)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 70)
    logger.info("")

    if len(translations) < 2:
        logger.warning("Need at least 2 poses for comparison")
        return

    mean_translation = np.mean(translations, axis=0)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)

    logger.info("Translation Consistency (checkerboard in base frame):")
    logger.info("-" * 50)
    logger.info(f"  Mean position: [{mean_translation[0]:.4f}, {mean_translation[1]:.4f}, {mean_translation[2]:.4f}] m")
    logger.info("")
    logger.info("  Per-pose distance from mean:")
    for i, err in enumerate(translation_errors):
        logger.info(f"    Pose {i+1}: {err*1000:.2f} mm")
    logger.info("")
    logger.info(f"  Mean error: {np.mean(translation_errors)*1000:.2f} mm")
    logger.info(f"  Max error:  {np.max(translation_errors)*1000:.2f} mm")
    logger.info("")

    # Rotation consistency
    if len(all_rotations) >= 2:
        rotation_errors_deg = []
        R_ref = all_rotations[0]

        for R in all_rotations:
            R_diff = R_ref.T @ R
            trace = np.clip(np.trace(R_diff), -1, 3)
            angle_rad = np.arccos((trace - 1) / 2)
            rotation_errors_deg.append(np.degrees(angle_rad))

        logger.info("Rotation Consistency:")
        logger.info("-" * 50)
        for i, err in enumerate(rotation_errors_deg):
            logger.info(f"    Pose {i+1}: {err:.2f} deg from pose 1")
        logger.info(f"  Max rotation error: {np.max(rotation_errors_deg):.2f} deg")

    logger.info("")
    logger.info("=" * 70)
    logger.info("INTERPRETATION")
    logger.info("=" * 70)

    max_err_mm = np.max(translation_errors) * 1000

    if max_err_mm < 5:
        quality = "EXCELLENT"
        msg = "X calibration is very accurate!"
    elif max_err_mm < 15:
        quality = "GOOD"
        msg = "X calibration is acceptable for most tasks."
    elif max_err_mm < 30:
        quality = "FAIR"
        msg = "X calibration may need refinement for precision tasks."
    else:
        quality = "POOR"
        msg = "X calibration needs to be redone."

    logger.info(f"  Calibration Quality: {quality}")
    logger.info(f"  {msg}")
    logger.info("=" * 70)


def main():
    logger.info("=" * 70)
    logger.info("Hand-Eye Calibration (X) Verification Analysis - TF Version")
    logger.info("=" * 70)
    logger.info("")

    data = load_verification_data()
    if data is None:
        return 1

    if data["num_poses"] < 1:
        logger.error("No poses found in verification data")
        return 1

    all_translations, all_rotations = analyze_poses(data)
    compute_consistency_metrics(all_translations, all_rotations)

    return 0


if __name__ == "__main__":
    exit(main())
