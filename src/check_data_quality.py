#!/usr/bin/env python3
"""
Comprehensive quality check for hand-eye calibration data.

Checks:
1. Data validity (rotation matrices, etc.)
2. Pose diversity (position and orientation)
3. Motion magnitudes
4. Conditioning of the calibration problem
"""

import numpy as np
import json
import cv2
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "hand_eye_calibration_data"

print("=" * 80)
print("HAND-EYE CALIBRATION DATA QUALITY ANALYSIS")
print("=" * 80)
print()

# Load data
data_file = DATA_DIR / "hand_eye_data.json"
with open(data_file) as f:
    data = json.load(f)

R_gripper2base = [np.array(R) for R in data["R_gripper2base"]]
t_gripper2base = [np.array(t) for t in data["t_gripper2base"]]
R_target2cam = [np.array(R) for R in data["R_target2cam"]]
t_target2cam = [np.array(t) for t in data["t_target2cam"]]

num_poses = len(R_gripper2base)
print(f"Number of poses: {num_poses}")
print()

# ============================================================================
# 1. DATA VALIDITY CHECK
# ============================================================================
print("=" * 80)
print("1. DATA VALIDITY CHECK")
print("=" * 80)
print()

all_valid = True

print("Checking gripper rotation matrices...")
for i, R in enumerate(R_gripper2base):
    det = np.linalg.det(R)
    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))

    if abs(det - 1.0) > 0.01 or ortho_error > 0.01:
        print(f"  ⚠ Pose {i+1}: Invalid rotation (det={det:.4f}, ortho_err={ortho_error:.6f})")
        all_valid = False

if all_valid:
    print("  ✓ All gripper rotation matrices are valid")
print()

print("Checking target (chessboard) rotation matrices...")
all_valid = True
for i, R in enumerate(R_target2cam):
    det = np.linalg.det(R)
    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))

    if abs(det - 1.0) > 0.01 or ortho_error > 0.01:
        print(f"  ⚠ Pose {i+1}: Invalid rotation (det={det:.4f}, ortho_err={ortho_error:.6f})")
        all_valid = False

if all_valid:
    print("  ✓ All target rotation matrices are valid")
print()

# ============================================================================
# 2. POSE DIVERSITY CHECK
# ============================================================================
print("=" * 80)
print("2. POSE DIVERSITY CHECK")
print("=" * 80)
print()

# Position diversity
positions = [t.flatten() for t in t_gripper2base]
min_dist = float('inf')
max_dist = 0
all_distances = []

for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[j])
        all_distances.append(dist)
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)

print(f"Gripper position diversity:")
print(f"  Min distance: {min_dist*1000:.1f} mm")
print(f"  Max distance: {max_dist*1000:.1f} mm")
print(f"  Mean distance: {np.mean(all_distances)*1000:.1f} mm")
print(f"  Std distance: {np.std(all_distances)*1000:.1f} mm")

if min_dist < 0.03:
    print(f"  ⚠ WARNING: Some poses are very close (< 30mm)")
elif min_dist < 0.05:
    print(f"  ⚠ Some poses are close (< 50mm) - more diversity would be better")
else:
    print(f"  ✓ Good position diversity")
print()

# Rotation diversity
rotations = R_gripper2base
min_angle = float('inf')
max_angle = 0
all_angles = []

for i in range(len(rotations)):
    for j in range(i+1, len(rotations)):
        R_diff = rotations[i] @ rotations[j].T
        rvec, _ = cv2.Rodrigues(R_diff)
        angle = np.degrees(np.linalg.norm(rvec))
        all_angles.append(angle)
        min_angle = min(min_angle, angle)
        max_angle = max(max_angle, angle)

print(f"Gripper rotation diversity:")
print(f"  Min rotation difference: {min_angle:.2f}°")
print(f"  Max rotation difference: {max_angle:.2f}°")
print(f"  Mean rotation difference: {np.mean(all_angles):.2f}°")
print(f"  Std rotation difference: {np.std(all_angles):.2f}°")

if min_angle < 10:
    print(f"  ⚠ WARNING: Some poses have very similar orientations (< 10°)")
elif min_angle < 15:
    print(f"  ⚠ Some poses have similar orientations (< 15°) - more diversity would be better")
else:
    print(f"  ✓ Good rotation diversity")
print()

# ============================================================================
# 3. MOTION MAGNITUDE CHECK
# ============================================================================
print("=" * 80)
print("3. MOTION MAGNITUDE CHECK")
print("=" * 80)
print()

print("Checking relative motions between consecutive poses...")
print()

translation_motions = []
rotation_motions = []

for i in range(1, num_poses):
    # Gripper motion from i-1 to i
    dt = np.linalg.norm(t_gripper2base[i] - t_gripper2base[i-1])
    R_motion = R_gripper2base[i-1].T @ R_gripper2base[i]
    rvec, _ = cv2.Rodrigues(R_motion)
    dtheta = np.degrees(np.linalg.norm(rvec))

    translation_motions.append(dt * 1000)  # mm
    rotation_motions.append(dtheta)

print(f"Translation motions:")
print(f"  Mean: {np.mean(translation_motions):.1f} mm")
print(f"  Min: {np.min(translation_motions):.1f} mm")
print(f"  Max: {np.max(translation_motions):.1f} mm")

if np.min(translation_motions) < 20:
    print(f"  ⚠ Some motions are very small (< 20mm)")
elif np.mean(translation_motions) < 50:
    print(f"  ⚠ Average motion is small - larger motions would improve calibration")
else:
    print(f"  ✓ Good translation magnitudes")
print()

print(f"Rotation motions:")
print(f"  Mean: {np.mean(rotation_motions):.1f}°")
print(f"  Min: {np.min(rotation_motions):.1f}°")
print(f"  Max: {np.max(rotation_motions):.1f}°")

if np.min(rotation_motions) < 5:
    print(f"  ⚠ Some rotations are very small (< 5°)")
elif np.mean(rotation_motions) < 15:
    print(f"  ⚠ Average rotation is small - larger rotations would improve calibration")
else:
    print(f"  ✓ Good rotation magnitudes")
print()

# ============================================================================
# 4. CHESSBOARD DETECTION CONSISTENCY
# ============================================================================
print("=" * 80)
print("4. CHESSBOARD DETECTION CONSISTENCY")
print("=" * 80)
print()

board_distances = [np.linalg.norm(t.flatten()) for t in t_target2cam]
print(f"Chessboard distance from camera:")
print(f"  Mean: {np.mean(board_distances):.3f} m")
print(f"  Min: {np.min(board_distances):.3f} m")
print(f"  Max: {np.max(board_distances):.3f} m")
print(f"  Std: {np.std(board_distances):.3f} m")

if np.std(board_distances) > 0.15:
    print(f"  ⚠ WARNING: Large variation in board distances - board may have moved!")
elif np.std(board_distances) > 0.1:
    print(f"  ⚠ Moderate variation in board distances")
else:
    print(f"  ✓ Consistent board distances (board stayed fixed)")
print()

# ============================================================================
# 5. CONDITION NUMBER ESTIMATE
# ============================================================================
print("=" * 80)
print("5. CALIBRATION PROBLEM CONDITIONING")
print("=" * 80)
print()

# Build the motion pairs for hand-eye calibration
# The condition of the problem depends on the diversity of rotations

rotation_axes = []
for i in range(1, num_poses):
    R_A = R_gripper2base[i-1].T @ R_gripper2base[i]
    R_B = R_target2cam[i] @ R_target2cam[i-1].T

    # Extract rotation axes
    rvec_A, _ = cv2.Rodrigues(R_A)
    rvec_B, _ = cv2.Rodrigues(R_B)

    rotation_axes.append(rvec_A.flatten() / (np.linalg.norm(rvec_A) + 1e-10))
    rotation_axes.append(rvec_B.flatten() / (np.linalg.norm(rvec_B) + 1e-10))

# Check how diverse the rotation axes are
rotation_axes = np.array(rotation_axes)
if len(rotation_axes) > 0:
    # Compute pairwise angles between rotation axes
    axis_angles = []
    for i in range(len(rotation_axes)):
        for j in range(i+1, len(rotation_axes)):
            cos_angle = np.dot(rotation_axes[i], rotation_axes[j])
            cos_angle = np.clip(cos_angle, -1, 1)
            angle = np.degrees(np.arccos(abs(cos_angle)))
            axis_angles.append(angle)

    print(f"Rotation axis diversity:")
    print(f"  Mean angle between axes: {np.mean(axis_angles):.1f}°")
    print(f"  Min angle between axes: {np.min(axis_angles):.1f}°")

    if np.min(axis_angles) < 20:
        print(f"  ⚠ WARNING: Some rotation axes are nearly parallel")
        print(f"     This can lead to poor calibration!")
    elif np.min(axis_angles) < 30:
        print(f"  ⚠ Some rotation axes are similar")
    else:
        print(f"  ✓ Good rotation axis diversity")
print()

# ============================================================================
# OVERALL ASSESSMENT
# ============================================================================
print("=" * 80)
print("OVERALL DATA QUALITY ASSESSMENT")
print("=" * 80)
print()

quality_score = 0
issues = []

# Position diversity
if min_dist >= 0.05:
    quality_score += 15
    print("✓ Position diversity: GOOD")
elif min_dist >= 0.03:
    quality_score += 10
    print("⚠ Position diversity: ACCEPTABLE")
else:
    quality_score += 5
    issues.append("Poor position diversity")
    print("✗ Position diversity: POOR")

# Rotation diversity
if min_angle >= 15:
    quality_score += 15
    print("✓ Rotation diversity: GOOD")
elif min_angle >= 10:
    quality_score += 10
    print("⚠ Rotation diversity: ACCEPTABLE")
else:
    quality_score += 5
    issues.append("Poor rotation diversity")
    print("✗ Rotation diversity: POOR")

# Motion magnitudes
if np.mean(translation_motions) >= 50 and np.mean(rotation_motions) >= 15:
    quality_score += 15
    print("✓ Motion magnitudes: GOOD")
elif np.mean(translation_motions) >= 30 and np.mean(rotation_motions) >= 10:
    quality_score += 10
    print("⚠ Motion magnitudes: ACCEPTABLE")
else:
    quality_score += 5
    issues.append("Small motion magnitudes")
    print("✗ Motion magnitudes: POOR")

# Board consistency
if np.std(board_distances) < 0.1:
    quality_score += 15
    print("✓ Board consistency: GOOD")
elif np.std(board_distances) < 0.15:
    quality_score += 10
    print("⚠ Board consistency: ACCEPTABLE")
else:
    quality_score += 5
    issues.append("Board may have moved during collection")
    print("✗ Board consistency: POOR")

# Number of poses
if num_poses >= 20:
    quality_score += 20
    print(f"✓ Number of poses: EXCELLENT ({num_poses})")
elif num_poses >= 15:
    quality_score += 15
    print(f"✓ Number of poses: GOOD ({num_poses})")
elif num_poses >= 10:
    quality_score += 10
    print(f"⚠ Number of poses: ACCEPTABLE ({num_poses})")
else:
    quality_score += 5
    issues.append("Too few poses")
    print(f"✗ Number of poses: POOR ({num_poses})")

# Rotation axis diversity
if len(axis_angles) > 0:
    if np.min(axis_angles) >= 30:
        quality_score += 20
        print("✓ Rotation axis diversity: EXCELLENT")
    elif np.min(axis_angles) >= 20:
        quality_score += 15
        print("✓ Rotation axis diversity: GOOD")
    elif np.min(axis_angles) >= 10:
        quality_score += 10
        print("⚠ Rotation axis diversity: ACCEPTABLE")
    else:
        quality_score += 5
        issues.append("Poor rotation axis diversity")
        print("✗ Rotation axis diversity: POOR")

print()
print(f"Overall Quality Score: {quality_score}/100")
print()

if quality_score >= 85:
    print("🟢 EXCELLENT data quality - calibration should be accurate")
elif quality_score >= 70:
    print("🟡 GOOD data quality - calibration should work well")
elif quality_score >= 50:
    print("🟠 ACCEPTABLE data quality - calibration may work but could be improved")
else:
    print("🔴 POOR data quality - recommend recollecting with better diversity")

if issues:
    print()
    print("Issues found:")
    for issue in issues:
        print(f"  • {issue}")

print()
print("=" * 80)
print("RECOMMENDATIONS:")
print("=" * 80)

if quality_score < 70:
    print()
    print("To improve data quality:")
    print("  1. Collect 20+ poses instead of just 15")
    print("  2. Move robot to more diverse positions (>50mm apart)")
    print("  3. Vary robot orientation more (>15° between poses)")
    print("  4. Ensure board stays completely fixed during collection")
    print("  5. Make sure rotation axes are diverse (not all tilting same way)")
else:
    print()
    print("Data quality is good. If calibration still fails verification,")
    print("the issue is likely in the calibration algorithm or coordinate frame definitions.")

print()
print("=" * 80)
