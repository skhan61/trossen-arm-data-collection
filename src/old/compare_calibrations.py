#!/usr/bin/env python3
"""
Compare MoveIt calibration vs computed calibration.

MoveIt calibration (camera_pose.launch.py): Eye-to-Hand, T_camera_in_base
Computed calibration (hand_eye_result.yaml): X_cam2gripper (eye-in-hand?)

These are DIFFERENT calibration types!
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


def quaternion_to_matrix(qx, qy, qz, qw):
    """Convert quaternion to rotation matrix."""
    rot = R.from_quat([qx, qy, qz, qw])
    return rot.as_matrix()


def create_transform(x, y, z, qx, qy, qz, qw):
    """Create 4x4 transformation matrix from position and quaternion."""
    T = np.eye(4)
    T[:3, :3] = quaternion_to_matrix(qx, qy, qz, qw)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def print_transform(T, name):
    """Print transformation matrix details."""
    print(f"\n{name}:")
    print(f"  Translation: [{T[0,3]:.4f}, {T[1,3]:.4f}, {T[2,3]:.4f}] m")

    rot = R.from_matrix(T[:3, :3])
    euler = rot.as_euler('xyz', degrees=True)
    quat = rot.as_quat()  # [x, y, z, w]

    print(f"  Rotation (XYZ Euler): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")
    print(f"  Quaternion (xyzw): [{quat[0]:.4f}, {quat[1]:.4f}, {quat[2]:.4f}, {quat[3]:.4f}]")


def main():
    print("=" * 70)
    print("Comparing Calibration Results")
    print("=" * 70)

    # MoveIt calibration (EYE-TO-HAND): base_link -> camera_color_optical_frame
    print("\n--- MoveIt Calibration (camera_pose.launch.py) ---")
    print("Type: EYE-TO-HAND")
    print("Transform: base_link -> camera_color_optical_frame")

    T_cam_in_base = create_transform(
        0.45947, 0.146574, 0.357797,
        -0.240178, -0.614843, 0.205497, 0.722533
    )
    print_transform(T_cam_in_base, "T_camera_in_base")

    # Computed calibration (hand_eye_result.yaml)
    print("\n--- Computed Calibration (hand_eye_result.yaml) ---")
    print("Type: Labeled as X_cam2gripper (suggests EYE-IN-HAND)")

    with open("/home/skhan61/Desktop/trossen-arm-data-collection/hand_eye_result.yaml", "r") as f:
        hand_eye = yaml.safe_load(f)

    X = np.array(hand_eye["X_cam2gripper_matrix"])
    print_transform(X, "X_cam2gripper")

    # Inverse
    X_inv = np.eye(4)
    X_inv[:3, :3] = X[:3, :3].T
    X_inv[:3, 3] = -X[:3, :3].T @ X[:3, 3]
    print_transform(X_inv, "X_gripper2cam (inverse)")

    print("\n" + "=" * 70)
    print("ANALYSIS")
    print("=" * 70)

    print("\nThe two calibrations represent DIFFERENT things:")
    print("  - MoveIt: T_camera_in_base (camera position in robot base frame)")
    print("  - Computed: X_cam2gripper (camera position in gripper frame)")
    print("")
    print("For EYE-TO-HAND (static camera):")
    print("  - MoveIt calibration IS the X we need")
    print("  - Formula: T_target_in_base = T_cam_in_base @ T_target_in_cam")
    print("")
    print("For EYE-IN-HAND (camera on gripper):")
    print("  - Computed X_cam2gripper is correct")
    print("  - Formula: T_target_in_base = T_gripper_in_base @ X @ T_target_in_cam")

    # The distance/angle difference
    dist = np.linalg.norm(T_cam_in_base[:3, 3] - X[:3, 3])
    print(f"\nTranslation difference: {dist*1000:.2f} mm")

    R_diff = T_cam_in_base[:3, :3].T @ X[:3, :3]
    trace = np.clip(np.trace(R_diff), -1, 3)
    angle_diff = np.degrees(np.arccos((trace - 1) / 2))
    print(f"Rotation difference: {angle_diff:.2f} deg")

    print("\n" + "=" * 70)
    print("RECOMMENDATION")
    print("=" * 70)
    print("\nIf your camera is STATICALLY MOUNTED (not on the gripper):")
    print("  -> Use MoveIt calibration from camera_pose.launch.py")
    print("  -> This is an EYE-TO-HAND setup")
    print("")
    print("If your camera is MOUNTED ON THE GRIPPER:")
    print("  -> Use computed X_cam2gripper from hand_eye_result.yaml")
    print("  -> This is an EYE-IN-HAND setup")
    print("")
    print("The large verification errors suggest one of:")
    print("  1. Using the wrong calibration type for your setup")
    print("  2. The calibration itself was done incorrectly")
    print("  3. The verification data has issues (different setup than calibration)")


if __name__ == "__main__":
    main()
