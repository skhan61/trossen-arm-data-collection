#!/usr/bin/env python3
"""
Mathematical validation of hand-eye calibration using the verification hierarchy.

Level 1: Internal Consistency - Check calibration residuals
Level 2: Geometric Consistency - Fixed point test (multi-pose verification)
"""

import numpy as np
import json
import cv2
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "hand_eye_calibration_data"

print("=" * 80)
print("MATHEMATICAL VALIDATION OF HAND-EYE CALIBRATION")
print("=" * 80)
print()

# Load calibration data
data_file = DATA_DIR / "hand_eye_data.json"
with open(data_file) as f:
    data = json.load(f)

R_gripper2base = [np.array(R) for R in data["R_gripper2base"]]
t_gripper2base = [np.array(t) for t in data["t_gripper2base"]]
R_target2cam = [np.array(R) for R in data["R_target2cam"]]
t_target2cam = [np.array(t) for t in data["t_target2cam"]]

num_poses = len(R_gripper2base)
print(f"Number of calibration poses: {num_poses}")
print()

# Load computed calibration
calib_file = DATA_DIR / "hand_eye_calibration.json"
with open(calib_file) as f:
    calib = json.load(f)

R_cam2gripper = np.array(calib["R_cam2gripper"])
t_cam2gripper = np.array(calib["t_cam2gripper"]).flatten()

print("Loaded hand-eye calibration X (camera-to-gripper transform)")
print()

# ============================================================================
# LEVEL 1: INTERNAL CONSISTENCY
# ============================================================================
print("=" * 80)
print("LEVEL 1: INTERNAL CONSISTENCY")
print("=" * 80)
print()
print("Testing: ||A_i X - X B_i|| for all pose pairs")
print("Where:")
print("  A_i = gripper motion from pose 1 to pose i")
print("  B_i = target motion from pose 1 to pose i")
print("  X = camera-to-gripper transform")
print()

# Compute pairwise motions
translation_errors = []
rotation_errors_deg = []

for i in range(1, num_poses):
    # A_i: gripper motion from pose 0 to pose i
    # R_gripper_i = R_gripper2base[i]
    # t_gripper_i = t_gripper2base[i]
    # R_gripper_0 = R_gripper2base[0]
    # t_gripper_0 = t_gripper2base[0]

    # A_i = H_gripper_0^{-1} @ H_gripper_i
    R_A = R_gripper2base[0].T @ R_gripper2base[i]
    t_A = R_gripper2base[0].T @ (t_gripper2base[i] - t_gripper2base[0])

    # B_i: target motion from pose 0 to pose i
    # B_i = H_target_i @ H_target_0^{-1}
    R_B = R_target2cam[i] @ R_target2cam[0].T
    t_B = t_target2cam[i] - R_target2cam[i] @ R_target2cam[0].T @ t_target2cam[0]

    # Check equation: A_i @ X = X @ B_i
    # Rotation part: R_A @ R_X = R_X @ R_B
    R_left = R_A @ R_cam2gripper
    R_right = R_cam2gripper @ R_B
    R_error = R_left @ R_right.T

    # Convert rotation error to angle
    rvec_error, _ = cv2.Rodrigues(R_error)
    angle_error = np.degrees(np.linalg.norm(rvec_error))

    # Translation part: R_A @ t_X + t_A = R_X @ t_B + t_X
    t_left = R_A @ t_cam2gripper + t_A.flatten()
    t_right = R_cam2gripper @ t_B.flatten() + t_cam2gripper
    t_error = np.linalg.norm(t_left - t_right)

    translation_errors.append(t_error * 1000)  # Convert to mm
    rotation_errors_deg.append(angle_error)

mean_trans_error = np.mean(translation_errors)
std_trans_error = np.std(translation_errors)
max_trans_error = np.max(translation_errors)

mean_rot_error = np.mean(rotation_errors_deg)
std_rot_error = np.std(rotation_errors_deg)
max_rot_error = np.max(rotation_errors_deg)

print(f"Translation residuals:")
print(f"  Mean: {mean_trans_error:.3f} mm")
print(f"  Std:  {std_trans_error:.3f} mm")
print(f"  Max:  {max_trans_error:.3f} mm")
print()

print(f"Rotation residuals:")
print(f"  Mean: {mean_rot_error:.3f}°")
print(f"  Std:  {std_rot_error:.3f}°")
print(f"  Max:  {max_rot_error:.3f}°")
print()

# Thresholds from your hierarchy
TRANS_THRESHOLD = 1.0  # mm
ROT_THRESHOLD = 0.5    # degrees

print("LEVEL 1 VERDICT:")
if mean_trans_error < TRANS_THRESHOLD and mean_rot_error < ROT_THRESHOLD:
    print(f"✓ PASS - Calibration is internally consistent")
    print(f"  Translation: {mean_trans_error:.3f} mm < {TRANS_THRESHOLD} mm")
    print(f"  Rotation: {mean_rot_error:.3f}° < {ROT_THRESHOLD}°")
elif mean_trans_error < 5.0 and mean_rot_error < 2.0:
    print(f"⚠ ACCEPTABLE - Calibration is reasonable but not excellent")
    print(f"  Translation: {mean_trans_error:.3f} mm (threshold: < 1 mm)")
    print(f"  Rotation: {mean_rot_error:.3f}° (threshold: < 0.5°)")
else:
    print(f"✗ FAIL - Calibration has poor internal consistency")
    print(f"  Translation: {mean_trans_error:.3f} mm >> {TRANS_THRESHOLD} mm")
    print(f"  Rotation: {mean_rot_error:.3f}° >> {ROT_THRESHOLD}°")
    print()
    print("Possible causes:")
    print("  1. Insufficient pose diversity during calibration")
    print("  2. Poor chessboard detection accuracy")
    print("  3. Robot positioning errors")
    print("  4. Board moved during data collection")

print()
print("=" * 80)
print()

print("INTERPRETATION:")
print()
print("Level 1 checks if X satisfies the calibration equation for all poses.")
print("This is a NECESSARY but NOT SUFFICIENT condition.")
print()
print("Even if Level 1 passes, you still need Level 2 (fixed point test)")
print("to verify the calibration is physically accurate.")
print()
print("Next step: Run verify_calibration_multipose.py to perform Level 2 test")
print()
print("=" * 80)
