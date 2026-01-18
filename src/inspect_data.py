#!/usr/bin/env python3
"""
Inspect the collected calibration data to find issues.
"""

import numpy as np
import json
import cv2
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "hand_eye_calibration_data"

print("=" * 80)
print("INSPECTING CALIBRATION DATA")
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

print("Checking gripper poses (robot end-effector positions):")
print("-" * 80)
for i in range(min(5, num_poses)):
    t = t_gripper2base[i].flatten()
    R = R_gripper2base[i]

    # Convert to axis-angle
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()

    print(f"\nPose {i+1}:")
    print(f"  Position: [{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}] m")
    print(f"  Rotation: [{rvec[0]:7.4f}, {rvec[1]:7.4f}, {rvec[2]:7.4f}] rad")

    # Check if rotation matrix is valid
    det = np.linalg.det(R)
    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))
    if abs(det - 1.0) > 0.01 or ortho_error > 0.01:
        print(f"  ⚠️  WARNING: Invalid rotation matrix (det={det:.4f}, ortho_error={ortho_error:.6f})")

print()
print("Checking target poses (chessboard in camera frame):")
print("-" * 80)
for i in range(min(5, num_poses)):
    t = t_target2cam[i].flatten()
    R = R_target2cam[i]

    # Convert to axis-angle
    rvec, _ = cv2.Rodrigues(R)
    rvec = rvec.flatten()

    print(f"\nPose {i+1}:")
    print(f"  Position: [{t[0]:7.4f}, {t[1]:7.4f}, {t[2]:7.4f}] m")
    print(f"  Rotation: [{rvec[0]:7.4f}, {rvec[1]:7.4f}, {rvec[2]:7.4f}] rad")
    print(f"  Distance from camera: {np.linalg.norm(t):.4f} m")

    # Check if rotation matrix is valid
    det = np.linalg.det(R)
    ortho_error = np.linalg.norm(R.T @ R - np.eye(3))
    if abs(det - 1.0) > 0.01 or ortho_error > 0.01:
        print(f"  ⚠️  WARNING: Invalid rotation matrix (det={det:.4f}, ortho_error={ortho_error:.6f})")

    # Check if board is too close or too far
    dist = np.linalg.norm(t)
    if dist < 0.1:
        print(f"  ⚠️  WARNING: Board very close to camera ({dist:.3f}m)")
    elif dist > 1.0:
        print(f"  ⚠️  WARNING: Board very far from camera ({dist:.3f}m)")

print()
print("=" * 80)
print("DIVERSITY CHECK")
print("=" * 80)
print()

# Check diversity of gripper poses
print("Gripper position diversity:")
positions = [t.flatten() for t in t_gripper2base]
min_dist = float('inf')
max_dist = 0
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        dist = np.linalg.norm(positions[i] - positions[j])
        min_dist = min(min_dist, dist)
        max_dist = max(max_dist, dist)

print(f"  Min distance between poses: {min_dist:.4f} m ({min_dist*1000:.1f} mm)")
print(f"  Max distance between poses: {max_dist:.4f} m ({max_dist*1000:.1f} mm)")

if min_dist < 0.05:
    print(f"  ⚠️  Some poses are very close (< 50mm)")

print()
print("Gripper rotation diversity:")
rotations = R_gripper2base
min_angle = float('inf')
max_angle = 0
for i in range(len(rotations)):
    for j in range(i+1, len(rotations)):
        R_diff = rotations[i] @ rotations[j].T
        rvec, _ = cv2.Rodrigues(R_diff)
        angle = np.degrees(np.linalg.norm(rvec))
        min_angle = min(min_angle, angle)
        max_angle = max(max_angle, angle)

print(f"  Min rotation difference: {min_angle:.2f}°")
print(f"  Max rotation difference: {max_angle:.2f}°")

if min_angle < 15:
    print(f"  ⚠️  Some poses have similar orientations (< 15°)")

print()
print("=" * 80)
print("SANITY CHECKS")
print("=" * 80)
print()

# Check if chessboard detections make sense
print("Chessboard detection sanity:")
board_distances = [np.linalg.norm(t.flatten()) for t in t_target2cam]
print(f"  Average distance to board: {np.mean(board_distances):.3f} m")
print(f"  Min distance: {np.min(board_distances):.3f} m")
print(f"  Max distance: {np.max(board_distances):.3f} m")
print(f"  Std deviation: {np.std(board_distances):.3f} m")

if np.std(board_distances) > 0.2:
    print(f"  ⚠️  Large variation in board distances - board might have moved!")

print()
print("Checking if gripper moved significantly:")
gripper_distances = [np.linalg.norm(t.flatten()) for t in t_gripper2base]
print(f"  Average gripper distance from origin: {np.mean(gripper_distances):.3f} m")
print(f"  Range: {np.min(gripper_distances):.3f} to {np.max(gripper_distances):.3f} m")

print()
print("=" * 80)
