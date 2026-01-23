#!/usr/bin/env python3
"""
Verify Eye-to-Hand calibration using data from camera_pose.launch.py.

Eye-to-Hand Setup:
- Camera is STATIC (mounted externally, looking at the robot workspace)
- Transform: base_link -> camera_color_optical_frame

Transformation Chain:
- T_target_in_base = T_cam_in_base @ T_target_in_cam
- Where T_cam_in_base = X (the calibration result from MoveIt)

If X is correct, the checkerboard position in base frame should be consistent
across all robot poses (since the camera and checkerboard are both static relative
to the base during eye-to-hand calibration verification).
"""

import numpy as np
from scipy.spatial.transform import Rotation as R
import yaml


def quaternion_to_matrix(qx, qy, qz, qw):
    """Convert quaternion to 4x4 transformation matrix (rotation only)."""
    rot = R.from_quat([qx, qy, qz, qw])  # scipy uses [x, y, z, w] order
    T = np.eye(4)
    T[:3, :3] = rot.as_matrix()
    return T


def create_transform(x, y, z, qx, qy, qz, qw):
    """Create 4x4 transformation matrix from position and quaternion."""
    T = quaternion_to_matrix(qx, qy, qz, qw)
    T[0, 3] = x
    T[1, 3] = y
    T[2, 3] = z
    return T


def load_moveit_calibration():
    """
    Load the eye-to-hand calibration from camera_pose.launch.py values.

    This is T_camera_in_base (base_link -> camera_color_optical_frame)
    """
    # Values from camera_pose.launch.py
    x = 0.45947
    y = 0.146574
    z = 0.357797
    qx = -0.240178
    qy = -0.614843
    qz = 0.205497
    qw = 0.722533

    return create_transform(x, y, z, qx, qy, qz, qw)


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
    trace = np.trace(R_diff)
    trace = np.clip(trace, -1, 3)
    angle = np.arccos((trace - 1) / 2)
    return np.degrees(angle)


def main():
    print("=" * 70)
    print("Eye-to-Hand Calibration Verification")
    print("=" * 70)

    # Load MoveIt calibration (T_camera_in_base)
    T_cam_in_base = load_moveit_calibration()

    print("\nMoveIt Calibration (T_camera_in_base):")
    print(f"  Translation: [{T_cam_in_base[0,3]:.4f}, {T_cam_in_base[1,3]:.4f}, {T_cam_in_base[2,3]:.4f}] m")

    # Extract rotation as euler angles for readability
    rot = R.from_matrix(T_cam_in_base[:3, :3])
    euler = rot.as_euler('xyz', degrees=True)
    print(f"  Rotation (XYZ Euler): [{euler[0]:.2f}, {euler[1]:.2f}, {euler[2]:.2f}] deg")

    # Load verification data
    with open("/home/skhan61/Desktop/trossen-arm-data-collection/data_sample_2.yaml", "r") as f:
        data = yaml.safe_load(f)

    print(f"\nLoaded {len(data)} verification samples")

    # For eye-to-hand verification with a MOVING checkerboard attached to gripper:
    # T_target_in_base = T_gripper_in_base @ T_target_in_gripper (fixed offset)
    #
    # But we measure T_target_in_cam, so:
    # T_target_in_base = T_cam_in_base @ T_target_in_cam
    #
    # These should be equal if X is correct!

    print("\n" + "=" * 70)
    print("Method 1: Checkerboard position via camera (using X calibration)")
    print("T_target_in_base = T_cam_in_base @ T_target_in_cam")
    print("=" * 70)

    target_positions_via_camera = []
    target_rotations_via_camera = []

    for i, sample in enumerate(data):
        T_target_in_cam = list_to_matrix(sample["object_wrt_sensor"])
        T_target_in_base = T_cam_in_base @ T_target_in_cam

        pos = T_target_in_base[:3, 3]
        target_positions_via_camera.append(pos)
        target_rotations_via_camera.append(T_target_in_base[:3, :3])

        print(f"\nSample {i+1}:")
        print(f"  Target in base (via camera): [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    # Compute consistency metrics
    positions = np.array(target_positions_via_camera)
    mean_pos = np.mean(positions, axis=0)

    print("\n" + "-" * 70)
    print("Consistency Analysis (positions via camera path):")
    print("-" * 70)
    print(f"Mean position: [{mean_pos[0]:.4f}, {mean_pos[1]:.4f}, {mean_pos[2]:.4f}] m")

    errors = np.linalg.norm(positions - mean_pos, axis=1) * 1000  # mm
    print(f"\nPosition errors from mean (mm):")
    for i, err in enumerate(errors):
        print(f"  Sample {i+1}: {err:.2f} mm")

    print(f"\nMax error: {np.max(errors):.2f} mm")
    print(f"Mean error: {np.mean(errors):.2f} mm")
    print(f"Std dev: {np.std(errors):.2f} mm")

    # Rotation consistency
    print("\nRotation errors from sample 1 (degrees):")
    R_ref = target_rotations_via_camera[0]
    for i, R_curr in enumerate(target_rotations_via_camera[1:], start=2):
        rot_err = rotation_error_degrees(R_ref, R_curr)
        print(f"  Sample {i}: {rot_err:.2f} deg")

    print("\n" + "=" * 70)
    print("Method 2: Gripper positions (from robot FK)")
    print("=" * 70)

    gripper_positions = []
    for i, sample in enumerate(data):
        T_gripper_in_base = list_to_matrix(sample["effector_wrt_world"])
        pos = T_gripper_in_base[:3, 3]
        gripper_positions.append(pos)
        print(f"Sample {i+1} gripper position: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}] m")

    print("\n" + "=" * 70)
    print("Cross-Validation: Target-to-Gripper offset should be CONSTANT")
    print("(if checkerboard is rigidly attached to gripper)")
    print("=" * 70)

    # If the checkerboard is attached to the gripper, then:
    # T_target_in_gripper should be constant across all poses
    # We can compute: T_target_in_gripper = inv(T_gripper_in_base) @ T_target_in_base

    offsets = []
    for i, sample in enumerate(data):
        T_gripper_in_base = list_to_matrix(sample["effector_wrt_world"])
        T_target_in_cam = list_to_matrix(sample["object_wrt_sensor"])

        # Target position in base frame via camera
        T_target_in_base = T_cam_in_base @ T_target_in_cam

        # Target position relative to gripper
        T_target_in_gripper = invert_transform(T_gripper_in_base) @ T_target_in_base

        offset = T_target_in_gripper[:3, 3]
        offsets.append(offset)
        print(f"\nSample {i+1} - Target in gripper frame:")
        print(f"  Offset: [{offset[0]:.4f}, {offset[1]:.4f}, {offset[2]:.4f}] m")

    offsets = np.array(offsets)
    mean_offset = np.mean(offsets, axis=0)
    offset_errors = np.linalg.norm(offsets - mean_offset, axis=1) * 1000

    print("\n" + "-" * 70)
    print("Target-to-Gripper Offset Consistency:")
    print("-" * 70)
    print(f"Mean offset: [{mean_offset[0]:.4f}, {mean_offset[1]:.4f}, {mean_offset[2]:.4f}] m")
    print(f"\nOffset errors from mean (mm):")
    for i, err in enumerate(offset_errors):
        print(f"  Sample {i+1}: {err:.2f} mm")
    print(f"\nMax offset error: {np.max(offset_errors):.2f} mm")
    print(f"Mean offset error: {np.mean(offset_errors):.2f} mm")

    # Final assessment
    print("\n" + "=" * 70)
    print("CALIBRATION QUALITY ASSESSMENT")
    print("=" * 70)

    max_err = np.max(offset_errors)
    if max_err < 10:
        quality = "EXCELLENT"
        msg = "Calibration is highly accurate (< 10mm error)"
    elif max_err < 25:
        quality = "GOOD"
        msg = "Calibration is acceptable for most tasks (< 25mm error)"
    elif max_err < 50:
        quality = "FAIR"
        msg = "Calibration may need improvement (< 50mm error)"
    else:
        quality = "POOR"
        msg = "Calibration needs to be redone (> 50mm error)"

    print(f"\nQuality: {quality}")
    print(f"Assessment: {msg}")
    print(f"Max error: {max_err:.2f} mm")


if __name__ == "__main__":
    main()
