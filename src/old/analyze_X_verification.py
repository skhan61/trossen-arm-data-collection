#!/usr/bin/env python3
"""
Analyze X (hand-eye calibration) Verification Data

Theory (eye-in-hand configuration):
-----------------------------------
The transformation chain is:
    T_target_in_base = T_gripper_in_base @ X @ T_target_in_camera

Where:
    - T_gripper_in_base: End-effector pose from robot (known)
    - X: Camera-to-gripper transform (what we're verifying)
    - T_target_in_camera: Checkerboard pose from solvePnP (known)
    - T_target_in_base: Checkerboard in robot base frame (should be constant)

Verification:
-------------
Since the checkerboard is STATIONARY (fixed in the world), T_target_in_base
should be the SAME for all poses, regardless of where the robot moved.

If X is correct:
    T_target_in_base (pose 1) ≈ T_target_in_base (pose 2) ≈ ... ≈ T_target_in_base (pose N)

We measure the spread/variance of the computed T_target_in_base across all poses.
Small variance = good calibration, large variance = poor calibration.

Usage:
    python src/analyze_X_verification.py
"""

import json
import numpy as np
import yaml
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
)
logger = logging.getLogger(__name__)

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Xverification"


def load_verification_data():
    """Load the collected verification data."""
    json_path = DATA_DIR / "verification_data.json"

    if not json_path.exists():
        logger.error(f"Verification data not found: {json_path}")
        logger.error("Run verify_X_calibration.py first to collect data.")
        return None

    with open(json_path, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {data['num_poses']} poses from {json_path}")
    return data


def compute_target_in_base(T_gripper_in_base, X_cam2gripper, T_target_in_camera):
    """
    Compute the checkerboard position in robot base frame.

    T_target_in_base = T_gripper_in_base @ X_cam2gripper @ T_target_in_camera
    """
    T_target_in_base = T_gripper_in_base @ X_cam2gripper @ T_target_in_camera
    return T_target_in_base


def rotation_matrix_to_axis_angle(R):
    """Convert rotation matrix to axis-angle representation."""
    import cv2
    rvec, _ = cv2.Rodrigues(R)
    angle = np.linalg.norm(rvec)
    if angle < 1e-6:
        axis = np.array([0, 0, 1])
    else:
        axis = rvec.flatten() / angle
    return axis, np.degrees(angle)


def analyze_poses(data):
    """Analyze all poses and compute T_target_in_base for each."""
    # Support both old and new key names
    if "X_cam_in_gripper" in data:
        X_cam2gripper = np.array(data["X_cam_in_gripper"])
    else:
        X_cam2gripper = np.array(data["X_cam2gripper"])
    poses = data["poses"]

    logger.info("")
    logger.info("=" * 70)
    logger.info("X Matrix (Camera to Gripper)")
    logger.info("=" * 70)
    logger.info(f"Translation: x={X_cam2gripper[0,3]:.4f}, y={X_cam2gripper[1,3]:.4f}, z={X_cam2gripper[2,3]:.4f} m")
    logger.info("")

    # Compute T_target_in_base for each pose
    all_T_target_in_base = []
    all_translations = []
    all_rotations = []

    logger.info("=" * 70)
    logger.info("Computing T_target_in_base for each pose")
    logger.info("=" * 70)
    logger.info("")

    for pose in poses:
        pose_id = pose["pose_id"]

        # Get T_gripper_in_base from robot data
        T_gripper_in_base = np.array(pose["robot"]["T_gripper2base"])

        # Get T_target_in_camera from checkerboard detection
        T_target_in_camera = np.array(pose["checkerboard"]["T_target2cam"])

        # Compute T_target_in_base
        T_target_in_base = compute_target_in_base(
            T_gripper_in_base, X_cam2gripper, T_target_in_camera
        )

        all_T_target_in_base.append(T_target_in_base)
        all_translations.append(T_target_in_base[:3, 3])
        all_rotations.append(T_target_in_base[:3, :3])

        # Log details for this pose
        t = T_target_in_base[:3, 3]
        axis, angle = rotation_matrix_to_axis_angle(T_target_in_base[:3, :3])

        logger.info(f"Pose {pose_id}:")
        logger.info(f"  T_target_in_base translation: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}] m")
        logger.info(f"  T_target_in_base rotation: {angle:.2f} deg about [{axis[0]:.3f}, {axis[1]:.3f}, {axis[2]:.3f}]")
        logger.info("")

    return all_T_target_in_base, all_translations, all_rotations


def compute_consistency_metrics(all_translations, all_rotations):
    """Compute metrics showing how consistent T_target_in_base is across poses."""
    translations = np.array(all_translations)

    logger.info("=" * 70)
    logger.info("VERIFICATION RESULTS")
    logger.info("=" * 70)
    logger.info("")

    if len(translations) < 2:
        logger.warning("Need at least 2 poses for comparison")
        return

    # Translation consistency
    mean_translation = np.mean(translations, axis=0)
    std_translation = np.std(translations, axis=0)
    max_translation_error = np.max(np.abs(translations - mean_translation), axis=0)

    logger.info("Translation Consistency (checkerboard position in base frame):")
    logger.info("-" * 50)
    logger.info(f"  Mean position:     [{mean_translation[0]:.4f}, {mean_translation[1]:.4f}, {mean_translation[2]:.4f}] m")
    logger.info(f"  Std deviation:     [{std_translation[0]:.4f}, {std_translation[1]:.4f}, {std_translation[2]:.4f}] m")
    logger.info(f"  Max error from mean: [{max_translation_error[0]:.4f}, {max_translation_error[1]:.4f}, {max_translation_error[2]:.4f}] m")
    logger.info("")

    # Overall translation error (Euclidean distance from mean)
    translation_errors = np.linalg.norm(translations - mean_translation, axis=1)
    logger.info(f"  Per-pose distance from mean:")
    for i, err in enumerate(translation_errors):
        logger.info(f"    Pose {i+1}: {err*1000:.2f} mm")
    logger.info("")
    logger.info(f"  Mean translation error: {np.mean(translation_errors)*1000:.2f} mm")
    logger.info(f"  Max translation error:  {np.max(translation_errors)*1000:.2f} mm")
    logger.info("")

    # Rotation consistency (using Frobenius norm of rotation difference)
    if len(all_rotations) >= 2:
        mean_rotation = all_rotations[0]  # Use first as reference
        rotation_errors_deg = []

        for i, R in enumerate(all_rotations):
            # Compute relative rotation
            R_diff = mean_rotation.T @ R
            # Convert to angle
            trace = np.trace(R_diff)
            trace = np.clip(trace, -1, 3)  # Numerical stability
            angle_rad = np.arccos((trace - 1) / 2)
            angle_deg = np.degrees(angle_rad)
            rotation_errors_deg.append(angle_deg)

        logger.info("Rotation Consistency:")
        logger.info("-" * 50)
        logger.info(f"  Per-pose rotation error from pose 1:")
        for i, err in enumerate(rotation_errors_deg):
            logger.info(f"    Pose {i+1}: {err:.2f} deg")
        logger.info("")
        logger.info(f"  Mean rotation error: {np.mean(rotation_errors_deg):.2f} deg")
        logger.info(f"  Max rotation error:  {np.max(rotation_errors_deg):.2f} deg")

    logger.info("")
    logger.info("=" * 70)
    logger.info("INTERPRETATION")
    logger.info("=" * 70)

    mean_trans_err_mm = np.mean(translation_errors) * 1000
    max_trans_err_mm = np.max(translation_errors) * 1000

    if max_trans_err_mm < 5:
        quality = "EXCELLENT"
        msg = "X calibration is very accurate!"
    elif max_trans_err_mm < 15:
        quality = "GOOD"
        msg = "X calibration is acceptable for most tasks."
    elif max_trans_err_mm < 30:
        quality = "FAIR"
        msg = "X calibration may need refinement for precision tasks."
    else:
        quality = "POOR"
        msg = "X calibration needs to be redone."

    logger.info(f"  Calibration Quality: {quality}")
    logger.info(f"  {msg}")
    logger.info("")
    logger.info("  Note: A stationary checkerboard should appear at the SAME")
    logger.info("  position in robot base frame regardless of robot pose.")
    logger.info("  Smaller errors = better calibration.")
    logger.info("=" * 70)


def main():
    logger.info("=" * 70)
    logger.info("Hand-Eye Calibration (X) Verification Analysis")
    logger.info("=" * 70)
    logger.info("")

    # Load data
    data = load_verification_data()
    if data is None:
        return 1

    if data["num_poses"] < 1:
        logger.error("No poses found in verification data")
        return 1

    # Analyze poses
    all_T_target_in_base, all_translations, all_rotations = analyze_poses(data)

    # Compute consistency metrics
    compute_consistency_metrics(all_translations, all_rotations)

    return 0


if __name__ == "__main__":
    exit(main())
