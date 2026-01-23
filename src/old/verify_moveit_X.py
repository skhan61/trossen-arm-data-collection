#!/usr/bin/env python3
"""
Verify MoveIt Eye-to-Hand X calibration using data from data/Xverification/.

From the images: The checkerboard is held by/near the gripper,
and the camera is statically mounted.

Setup:
  - Camera is STATIC (mounted externally) - this is Eye-to-Hand
  - Checkerboard is ATTACHED TO / NEAR GRIPPER (moves with robot)

Eye-to-Hand with target attached to gripper:
  T_target_in_base = X @ T_target_in_cam        (via camera)
  T_target_in_base = T_gripper_in_base @ T_target_in_gripper  (via robot FK)

If X is correct, T_target_in_gripper should be CONSTANT across all poses.
  T_target_in_gripper = inv(T_gripper_in_base) @ X @ T_target_in_cam
"""

import json
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Xverification"


def load_moveit_X():
    """
    Load MoveIt calibration from camera_pose.launch.py values.
    T_cam_in_base: base_link -> camera_color_optical_frame
    """
    x = 0.45947
    y = 0.146574
    z = 0.357797
    qx = -0.240178
    qy = -0.614843
    qz = 0.205497
    qw = 0.722533

    rot = R.from_quat([qx, qy, qz, qw])
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def invert_transform(T):
    """Invert a 4x4 transformation matrix."""
    R_mat = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R_mat.T
    T_inv[:3, 3] = -R_mat.T @ t
    return T_inv


def rotation_error_degrees(R1, R2):
    """Compute rotation error between two rotation matrices in degrees."""
    R_diff = R1.T @ R2
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)


def main():
    print("=" * 70)
    print("Verify MoveIt X Calibration (Eye-to-Hand)")
    print("Data: data/Xverification/")
    print("=" * 70)

    # Load verification data
    json_path = DATA_DIR / "verification_data.json"
    with open(json_path, "r") as f:
        data = json.load(f)

    print(f"\nLoaded {data['num_poses']} poses from verification data")

    # Load MoveIt X (T_cam_in_base)
    X = load_moveit_X()

    print("\nMoveIt X (T_cam_in_base from camera_pose.launch.py):")
    print(f"  Translation: [{X[0,3]:.4f}, {X[1,3]:.4f}, {X[2,3]:.4f}] m")
    rot = R.from_matrix(X[:3, :3])
    euler = rot.as_euler('xyz', degrees=True)
    print(f"  Rotation (XYZ Euler): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")

    print("\n" + "=" * 70)
    print("Eye-to-Hand Verification (target moves with gripper):")
    print("T_target_in_gripper should be CONSTANT")
    print("Formula: T_target_in_gripper = inv(T_gripper_in_base) @ X @ T_target_in_cam")
    print("=" * 70)

    T_target_in_gripper_list = []

    for pose in data["poses"]:
        pose_id = pose["pose_id"]

        # T_gripper_in_base from robot FK
        T_gripper_in_base = np.array(pose["robot"]["T_gripper2base"])

        # T_target_in_cam from checkerboard detection
        T_target_in_cam = np.array(pose["checkerboard"]["T_target2cam"])

        # Compute T_target_in_base via camera path
        T_target_in_base = X @ T_target_in_cam

        # Compute T_target_in_gripper (should be constant!)
        T_base_in_gripper = invert_transform(T_gripper_in_base)
        T_target_in_gripper = T_base_in_gripper @ T_target_in_base
        T_target_in_gripper_list.append(T_target_in_gripper)

        print(f"\nPose {pose_id}:")
        print(f"  Gripper in base:   [{T_gripper_in_base[0,3]:.4f}, {T_gripper_in_base[1,3]:.4f}, {T_gripper_in_base[2,3]:.4f}] m")
        print(f"  Target in cam:     [{T_target_in_cam[0,3]:.4f}, {T_target_in_cam[1,3]:.4f}, {T_target_in_cam[2,3]:.4f}] m")
        print(f"  Target in base:    [{T_target_in_base[0,3]:.4f}, {T_target_in_base[1,3]:.4f}, {T_target_in_base[2,3]:.4f}] m")
        print(f"  Target in gripper: [{T_target_in_gripper[0,3]:.4f}, {T_target_in_gripper[1,3]:.4f}, {T_target_in_gripper[2,3]:.4f}] m")

    # Consistency Analysis
    print("\n" + "=" * 70)
    print("CONSISTENCY ANALYSIS")
    print("=" * 70)

    if len(T_target_in_gripper_list) >= 2:
        pos1 = T_target_in_gripper_list[0][:3, 3]
        pos2 = T_target_in_gripper_list[1][:3, 3]

        position_diff = np.linalg.norm(pos1 - pos2) * 1000  # mm

        R1 = T_target_in_gripper_list[0][:3, :3]
        R2 = T_target_in_gripper_list[1][:3, :3]
        rotation_diff = rotation_error_degrees(R1, R2)

        print(f"\nPose 1 target in gripper: [{pos1[0]:.4f}, {pos1[1]:.4f}, {pos1[2]:.4f}] m")
        print(f"Pose 2 target in gripper: [{pos2[0]:.4f}, {pos2[1]:.4f}, {pos2[2]:.4f}] m")
        print(f"\nPosition difference: {position_diff:.2f} mm")
        print(f"Rotation difference: {rotation_diff:.2f} deg")

        # Quality assessment
        print("\n" + "=" * 70)
        print("CALIBRATION QUALITY")
        print("=" * 70)

        if position_diff < 10 and rotation_diff < 3:
            quality = "EXCELLENT"
            symbol = "OK"
        elif position_diff < 25 and rotation_diff < 5:
            quality = "GOOD"
            symbol = "OK"
        elif position_diff < 50 and rotation_diff < 10:
            quality = "ACCEPTABLE"
            symbol = "~"
        else:
            quality = "POOR - NEEDS RECALIBRATION"
            symbol = "X"

        print(f"\n  [{symbol}] Quality: {quality}")
        print(f"  [{symbol}] Position error: {position_diff:.2f} mm (target: <25mm)")
        print(f"  [{symbol}] Rotation error: {rotation_diff:.2f} deg (target: <5deg)")

    else:
        print("Need at least 2 poses for verification")


if __name__ == "__main__":
    main()
