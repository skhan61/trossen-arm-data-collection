#!/usr/bin/env python3
"""
Check frame conventions to debug ~180° rotation error.

This script analyzes:
1. Camera optical frame convention (Z forward, X right, Y down)
2. Checkerboard coordinate system from solvePnP
3. End-effector frame orientation
"""

import json
import numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation as R


def analyze_rotation(R_mat, name):
    """Analyze what a rotation matrix does to the standard axes."""
    print(f"\n{name}:")
    print(f"  Matrix:")
    for i in range(3):
        print(f"    [{R_mat[i,0]:8.4f}, {R_mat[i,1]:8.4f}, {R_mat[i,2]:8.4f}]")

    # Where do the original axes go?
    x_axis = R_mat @ np.array([1, 0, 0])
    y_axis = R_mat @ np.array([0, 1, 0])
    z_axis = R_mat @ np.array([0, 0, 1])

    print(f"\n  Axis mapping (where original axes point after rotation):")
    print(f"    X-axis [1,0,0] -> [{x_axis[0]:6.3f}, {x_axis[1]:6.3f}, {x_axis[2]:6.3f}]")
    print(f"    Y-axis [0,1,0] -> [{y_axis[0]:6.3f}, {y_axis[1]:6.3f}, {y_axis[2]:6.3f}]")
    print(f"    Z-axis [0,0,1] -> [{z_axis[0]:6.3f}, {z_axis[1]:6.3f}, {z_axis[2]:6.3f}]")

    # Euler angles
    rot = R.from_matrix(R_mat)
    euler = rot.as_euler('xyz', degrees=True)
    print(f"\n  Euler angles (xyz): roll={euler[0]:.1f}°, pitch={euler[1]:.1f}°, yaw={euler[2]:.1f}°")

    # Determinant (should be +1 for proper rotation)
    det = np.linalg.det(R_mat)
    print(f"  Determinant: {det:.4f} (should be +1.0)")


def main():
    print("=" * 70)
    print("Frame Convention Analysis")
    print("=" * 70)

    # Load verification data
    verif_file = Path(__file__).parent.parent / "data" / "Xverification" / "verification_data.json"
    with open(verif_file, "r") as f:
        data = json.load(f)

    pose1 = data["poses"][0]
    pose2 = data["poses"][1]

    # =========================================================================
    print("\n" + "=" * 70)
    print("1. CHECKERBOARD FRAME (T_target2cam from solvePnP)")
    print("=" * 70)
    print("\nOpenCV solvePnP convention:")
    print("  - Origin: at first corner of checkerboard")
    print("  - X-axis: along first edge (typically longer)")
    print("  - Y-axis: along second edge")
    print("  - Z-axis: OUT of board (towards camera)")

    T_cam_to_target_1 = np.array(pose1["checkerboard"]["T_target2cam"])
    T_cam_to_target_2 = np.array(pose2["checkerboard"]["T_target2cam"])

    analyze_rotation(T_cam_to_target_1[:3, :3], "Pose 1: T_cam_to_target rotation")

    print(f"\n  Translation (checkerboard origin in camera frame):")
    t1 = T_cam_to_target_1[:3, 3]
    print(f"    [{t1[0]*1000:.1f}, {t1[1]*1000:.1f}, {t1[2]*1000:.1f}] mm")
    print(f"    Z = {t1[2]*1000:.1f} mm (should be POSITIVE if board is in front of camera)")

    # =========================================================================
    print("\n" + "=" * 70)
    print("2. CAMERA OPTICAL FRAME CONVENTION")
    print("=" * 70)
    print("\nROS camera_color_optical_frame convention:")
    print("  - Z-axis: forward (into scene)")
    print("  - X-axis: right")
    print("  - Y-axis: down")
    print("\nThis is DIFFERENT from camera_link which has:")
    print("  - Z-axis: forward")
    print("  - X-axis: right")
    print("  - Y-axis: down")
    print("\nBut camera_color_frame has:")
    print("  - Z-axis: forward")
    print("  - X-axis: right")
    print("  - Y-axis: up (then optical frame rotates this)")

    # =========================================================================
    print("\n" + "=" * 70)
    print("3. END-EFFECTOR FRAME (T_gripper2base)")
    print("=" * 70)

    T_base_to_ee_1 = np.array(pose1["robot"]["T_gripper2base"])
    T_base_to_ee_2 = np.array(pose2["robot"]["T_gripper2base"])

    analyze_rotation(T_base_to_ee_1[:3, :3], "Pose 1: T_base_to_ee rotation")
    analyze_rotation(T_base_to_ee_2[:3, :3], "Pose 2: T_base_to_ee rotation")

    # =========================================================================
    print("\n" + "=" * 70)
    print("4. X MATRIX ANALYSIS")
    print("=" * 70)

    # X from URDF
    X_urdf = np.array([
        [0, -0.342020, 0.939693, -0.097412],
        [-1, 0, 0, 0.009],
        [0, -0.939693, -0.342020, 0.054272],
        [0, 0, 0, 1]
    ])

    # X from launch file
    t = np.array([-0.0175075, 0.0145861, 0.0612924])
    quat = np.array([-0.578712, 0.582021, -0.409742, 0.398065])
    rot = R.from_quat(quat)
    X_launch = np.eye(4)
    X_launch[:3, :3] = rot.as_matrix()
    X_launch[:3, 3] = t

    analyze_rotation(X_urdf[:3, :3], "X from URDF")
    analyze_rotation(X_launch[:3, :3], "X from launch.py")

    # =========================================================================
    print("\n" + "=" * 70)
    print("5. CHECKING FOR 180° FLIP")
    print("=" * 70)

    # If there's a 180° error, try flipping the checkerboard Z-axis
    print("\nTrying: Flip checkerboard Z-axis (negate Z column of rotation)")

    T_cam_to_target_1_flipped = T_cam_to_target_1.copy()
    T_cam_to_target_1_flipped[:3, 2] *= -1  # Flip Z axis

    # This makes the rotation matrix improper (det = -1), so also flip another axis
    # to maintain proper rotation
    T_cam_to_target_1_flipped[:3, 0] *= -1  # Flip X to keep det = +1

    print(f"\nOriginal T_cam_to_target det: {np.linalg.det(T_cam_to_target_1[:3,:3]):.4f}")
    print(f"Flipped T_cam_to_target det: {np.linalg.det(T_cam_to_target_1_flipped[:3,:3]):.4f}")

    # =========================================================================
    print("\n" + "=" * 70)
    print("6. COMPUTING T_base_to_target WITH DIFFERENT CONVENTIONS")
    print("=" * 70)

    X = X_launch  # Use launch.py X

    # Standard computation
    T_base_to_target_1 = T_base_to_ee_1 @ X @ T_cam_to_target_1
    T_base_to_target_2 = T_base_to_ee_2 @ X @ T_cam_to_target_2

    print("\nStandard: T_base_to_target = T_base_to_ee @ X @ T_cam_to_target")
    print(f"  Pose 1 position: [{T_base_to_target_1[0,3]*1000:.1f}, {T_base_to_target_1[1,3]*1000:.1f}, {T_base_to_target_1[2,3]*1000:.1f}] mm")
    print(f"  Pose 2 position: [{T_base_to_target_2[0,3]*1000:.1f}, {T_base_to_target_2[1,3]*1000:.1f}, {T_base_to_target_2[2,3]*1000:.1f}] mm")
    diff = np.linalg.norm(T_base_to_target_1[:3,3] - T_base_to_target_2[:3,3]) * 1000
    print(f"  Difference: {diff:.1f} mm")

    # Try with inverted X
    print("\nTrying: T_base_to_target = T_base_to_ee @ inv(X) @ T_cam_to_target")
    X_inv = np.linalg.inv(X)
    T_base_to_target_1_inv = T_base_to_ee_1 @ X_inv @ T_cam_to_target_1
    T_base_to_target_2_inv = T_base_to_ee_2 @ X_inv @ T_cam_to_target_2
    print(f"  Pose 1 position: [{T_base_to_target_1_inv[0,3]*1000:.1f}, {T_base_to_target_1_inv[1,3]*1000:.1f}, {T_base_to_target_1_inv[2,3]*1000:.1f}] mm")
    print(f"  Pose 2 position: [{T_base_to_target_2_inv[0,3]*1000:.1f}, {T_base_to_target_2_inv[1,3]*1000:.1f}, {T_base_to_target_2_inv[2,3]*1000:.1f}] mm")
    diff_inv = np.linalg.norm(T_base_to_target_1_inv[:3,3] - T_base_to_target_2_inv[:3,3]) * 1000
    print(f"  Difference: {diff_inv:.1f} mm")

    # Try with T_target_to_cam instead of T_cam_to_target
    print("\nTrying: T_base_to_target = T_base_to_ee @ X @ inv(T_cam_to_target)")
    T_target_to_cam_1 = np.linalg.inv(T_cam_to_target_1)
    T_target_to_cam_2 = np.linalg.inv(T_cam_to_target_2)
    T_base_to_target_1_flip = T_base_to_ee_1 @ X @ T_target_to_cam_1
    T_base_to_target_2_flip = T_base_to_ee_2 @ X @ T_target_to_cam_2
    print(f"  Pose 1 position: [{T_base_to_target_1_flip[0,3]*1000:.1f}, {T_base_to_target_1_flip[1,3]*1000:.1f}, {T_base_to_target_1_flip[2,3]*1000:.1f}] mm")
    print(f"  Pose 2 position: [{T_base_to_target_2_flip[0,3]*1000:.1f}, {T_base_to_target_2_flip[1,3]*1000:.1f}, {T_base_to_target_2_flip[2,3]*1000:.1f}] mm")
    diff_flip = np.linalg.norm(T_base_to_target_1_flip[:3,3] - T_base_to_target_2_flip[:3,3]) * 1000
    print(f"  Difference: {diff_flip:.1f} mm")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\n  Standard formula:     {diff:.1f} mm error")
    print(f"  Using inv(X):         {diff_inv:.1f} mm error")
    print(f"  Using inv(T_cam2tgt): {diff_flip:.1f} mm error")

    best = min(diff, diff_inv, diff_flip)
    if best == diff_inv:
        print("\n  >>> inv(X) gives better result - X direction may be inverted")
    elif best == diff_flip:
        print("\n  >>> inv(T_cam_to_target) gives better result - naming convention may be wrong")

    return 0


if __name__ == "__main__":
    exit(main())
