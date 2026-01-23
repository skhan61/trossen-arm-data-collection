#!/usr/bin/env python3
"""
Verify MoveIt Eye-to-Hand calibration from camera_pose.launch.py.

Setup: Camera is STATIC, looking at robot workspace.
Transform: base_link -> camera_color_optical_frame (T_cam_in_base)

Verification Theory:
- Checkerboard is attached to/held by gripper
- T_target_in_base = T_cam_in_base @ T_target_in_cam  (via camera)
- T_target_in_base = T_gripper_in_base @ T_target_in_gripper (via robot FK)
- If calibration is correct, T_target_in_gripper should be CONSTANT across all poses
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


def create_transform_from_moveit():
    """
    Load MoveIt calibration from camera_pose.launch.py values.
    Transform: base_link -> camera_color_optical_frame
    """
    # Values from camera_pose.launch.py
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


def list_to_matrix(flat_list):
    """Convert 16-element flat list to 4x4 matrix."""
    return np.array(flat_list).reshape(4, 4)


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
    print("MoveIt Eye-to-Hand Calibration Verification")
    print("Using: camera_pose.launch.py")
    print("=" * 70)

    # Load MoveIt calibration
    T_cam_in_base = create_transform_from_moveit()

    print("\nMoveIt Calibration (T_cam_in_base):")
    print(f"  Position: x={T_cam_in_base[0,3]:.4f}, y={T_cam_in_base[1,3]:.4f}, z={T_cam_in_base[2,3]:.4f} m")

    rot = R.from_matrix(T_cam_in_base[:3, :3])
    euler = rot.as_euler('xyz', degrees=True)
    print(f"  Rotation: roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f} deg")

    # Load verification data
    with open("/home/skhan61/Desktop/trossen-arm-data-collection/data_sample_2.yaml", "r") as f:
        data = yaml.safe_load(f)

    print(f"\nLoaded {len(data)} samples from data_sample_2.yaml")

    print("\n" + "=" * 70)
    print("Verification: T_target_in_gripper should be CONSTANT")
    print("Formula: T_target_in_gripper = inv(T_gripper_in_base) @ T_cam_in_base @ T_target_in_cam")
    print("=" * 70)

    T_target_in_gripper_list = []

    for i, sample in enumerate(data):
        # Robot FK: T_gripper_in_base (effector_wrt_world)
        T_gripper_in_base = list_to_matrix(sample["effector_wrt_world"])

        # Camera detection: T_target_in_cam (object_wrt_sensor)
        T_target_in_cam = list_to_matrix(sample["object_wrt_sensor"])

        # Compute T_target_in_base via camera path
        T_target_in_base = T_cam_in_base @ T_target_in_cam

        # Compute T_target_in_gripper
        T_target_in_gripper = invert_transform(T_gripper_in_base) @ T_target_in_base
        T_target_in_gripper_list.append(T_target_in_gripper)

        pos = T_target_in_gripper[:3, 3]
        print(f"\nSample {i+1}: T_target_in_gripper")
        print(f"  Translation: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    # Consistency analysis
    print("\n" + "=" * 70)
    print("CONSISTENCY ANALYSIS")
    print("=" * 70)

    positions = np.array([T[:3, 3] for T in T_target_in_gripper_list])
    mean_pos = np.mean(positions, axis=0)

    print(f"\nMean T_target_in_gripper position: [{mean_pos[0]:.4f}, {mean_pos[1]:.4f}, {mean_pos[2]:.4f}] m")

    # Position errors
    errors_mm = np.linalg.norm(positions - mean_pos, axis=1) * 1000

    print(f"\nPosition errors from mean:")
    for i, err in enumerate(errors_mm):
        print(f"  Sample {i+1}: {err:.2f} mm")

    print(f"\n  Max error:  {np.max(errors_mm):.2f} mm")
    print(f"  Mean error: {np.mean(errors_mm):.2f} mm")
    print(f"  Std dev:    {np.std(errors_mm):.2f} mm")

    # Rotation errors
    print(f"\nRotation errors from Sample 1:")
    R_ref = T_target_in_gripper_list[0][:3, :3]
    rot_errors = []
    for i, T in enumerate(T_target_in_gripper_list[1:], start=2):
        err = rotation_error_degrees(R_ref, T[:3, :3])
        rot_errors.append(err)
        print(f"  Sample {i}: {err:.2f} deg")

    if rot_errors:
        print(f"\n  Max rotation error: {np.max(rot_errors):.2f} deg")
        print(f"  Mean rotation error: {np.mean(rot_errors):.2f} deg")

    # Quality assessment
    print("\n" + "=" * 70)
    print("CALIBRATION QUALITY")
    print("=" * 70)

    max_pos_err = np.max(errors_mm)
    max_rot_err = np.max(rot_errors) if rot_errors else 0

    if max_pos_err < 5 and max_rot_err < 2:
        quality = "EXCELLENT"
        color = "✓"
    elif max_pos_err < 15 and max_rot_err < 5:
        quality = "GOOD"
        color = "✓"
    elif max_pos_err < 30 and max_rot_err < 10:
        quality = "ACCEPTABLE"
        color = "~"
    else:
        quality = "POOR - RECALIBRATE"
        color = "✗"

    print(f"\n  {color} Quality: {quality}")
    print(f"  {color} Max position error: {max_pos_err:.2f} mm")
    print(f"  {color} Max rotation error: {max_rot_err:.2f} deg")

    if max_pos_err > 30:
        print("\n  Possible issues:")
        print("    - MoveIt calibration was done with different camera position")
        print("    - Checkerboard was not rigidly attached to gripper")
        print("    - Camera or robot moved between calibration and verification")


if __name__ == "__main__":
    main()
