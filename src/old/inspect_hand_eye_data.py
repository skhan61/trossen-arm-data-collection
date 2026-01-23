#!/usr/bin/env python3
"""
Simple script to inspect a single data point from hand-eye calibration data.
"""

import json
import numpy as np
from pathlib import Path

def main():
    # Load hand-eye calibration data
    data_file = Path(__file__).parent.parent / "data/hand_eye_calibration_data/hand_eye_data.json"

    with open(data_file, 'r') as f:
        data = json.load(f)

    # Get the first data point (index 0)
    pose_idx = 0

    print("=" * 60)
    print(f"Hand-Eye Calibration Data - Pose {pose_idx + 1}")
    print("=" * 60)
    print()

    # Robot gripper pose (in base frame)
    R_gripper2base = np.array(data["R_gripper2base"][pose_idx])
    t_gripper2base = np.array(data["t_gripper2base"][pose_idx])

    print("Robot Gripper Pose (in base frame):")
    print("-" * 40)
    print("Rotation matrix (3x3):")
    print(R_gripper2base)
    print()
    print("Translation vector (3x1) [meters]:")
    print(t_gripper2base.flatten())
    print()

    # Convert rotation matrix to angle-axis (what robot API returns)
    import cv2
    angle_axis, _ = cv2.Rodrigues(R_gripper2base)
    print("Robot API Cartesian Position [x, y, z, rx, ry, rz]:")
    robot_cart_pos = list(t_gripper2base.flatten()) + list(angle_axis.flatten())
    print(f"  x:  {robot_cart_pos[0]:.6f} m")
    print(f"  y:  {robot_cart_pos[1]:.6f} m")
    print(f"  z:  {robot_cart_pos[2]:.6f} m")
    print(f"  rx: {robot_cart_pos[3]:.6f} rad")
    print(f"  ry: {robot_cart_pos[4]:.6f} rad")
    print(f"  rz: {robot_cart_pos[5]:.6f} rad")
    print()

    # Chessboard pose (in camera frame)
    R_target2cam = np.array(data["R_target2cam"][pose_idx])
    t_target2cam = np.array(data["t_target2cam"][pose_idx])

    print("Chessboard Pose (in camera frame):")
    print("-" * 40)
    print("Rotation matrix (3x3):")
    print(R_target2cam)
    print()
    print("Translation vector (3x1) [meters]:")
    print(t_target2cam.flatten())
    print()

    print("=" * 60)
    print(f"Total poses in dataset: {data['num_poses']}")
    print("=" * 60)

if __name__ == "__main__":
    main()
