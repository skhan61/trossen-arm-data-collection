#!/usr/bin/env python3
"""
Compute X (T_ee_to_camera_color_optical) from URDF transforms.

The transformation chain from link_6 to camera_color_optical_frame:
    link_6 -> camera_mount_d405 -> camera_bottom_screw_frame ->
    camera_link -> camera_color_frame -> camera_color_optical_frame

Then we convert from link_6 reference to ee_gripper_link reference.
"""

import numpy as np
from scipy.spatial.transform import Rotation as R


def make_transform(xyz, rpy):
    """
    Create a 4x4 homogeneous transformation matrix from xyz translation and rpy rotation.

    Args:
        xyz: [x, y, z] translation in meters
        rpy: [roll, pitch, yaw] rotation in radians (XYZ Euler angles)

    Returns:
        4x4 numpy array transformation matrix
    """
    T = np.eye(4)

    # Rotation from roll, pitch, yaw (XYZ extrinsic = ZYX intrinsic)
    rot = R.from_euler('xyz', rpy)
    T[:3, :3] = rot.as_matrix()

    # Translation
    T[:3, 3] = xyz

    return T


def print_transform(name, T):
    """Pretty print a transformation matrix."""
    print(f"\n{name}:")
    print(f"  Translation: x={T[0,3]:.6f}, y={T[1,3]:.6f}, z={T[2,3]:.6f} (meters)")

    # Extract rotation as various representations
    rot = R.from_matrix(T[:3, :3])
    rpy = rot.as_euler('xyz')
    quat = rot.as_quat()  # [x, y, z, w]

    print(f"  Rotation (rpy): roll={np.degrees(rpy[0]):.2f}°, pitch={np.degrees(rpy[1]):.2f}°, yaw={np.degrees(rpy[2]):.2f}°")
    print(f"  Rotation (quat): x={quat[0]:.6f}, y={quat[1]:.6f}, z={quat[2]:.6f}, w={quat[3]:.6f}")
    print(f"  Full matrix:")
    for row in T:
        print(f"    [{row[0]:10.6f}, {row[1]:10.6f}, {row[2]:10.6f}, {row[3]:10.6f}]")


def main():
    print("=" * 70)
    print("Computing X (T_ee_to_camera_color_optical) from URDF")
    print("=" * 70)

    # =========================================================================
    # Step 1: Define all transforms from URDF (wxai_follower.urdf)
    # =========================================================================

    print("\n" + "=" * 70)
    print("Step 1: URDF Transforms (from wxai_follower.urdf)")
    print("=" * 70)

    # link_6 -> camera_mount_d405 (camera_mount_joint)
    T_link6_to_camera_mount = make_transform(
        xyz=[0.012, 0, 0],
        rpy=[0, 0, 0]
    )
    print_transform("T_link6_to_camera_mount", T_link6_to_camera_mount)

    # camera_mount_d405 -> camera_bottom_screw_frame (camera_joint)
    # Note: 0.3490658503988659 rad = 20 degrees
    T_camera_mount_to_bottom_screw = make_transform(
        xyz=[0.02927207801, 0, 0.03824951197],
        rpy=[0, 0.3490658503988659, 0]  # 20° pitch
    )
    print_transform("T_camera_mount_to_bottom_screw", T_camera_mount_to_bottom_screw)

    # camera_bottom_screw_frame -> camera_link (camera_link_joint)
    T_bottom_screw_to_camera_link = make_transform(
        xyz=[0.01085, 0.009, 0.021],
        rpy=[0, 0, 0]
    )
    print_transform("T_bottom_screw_to_camera_link", T_bottom_screw_to_camera_link)

    # camera_link -> camera_color_frame (camera_color_joint)
    T_camera_link_to_color_frame = make_transform(
        xyz=[0, 0, 0],
        rpy=[0, 0, 0]
    )
    print_transform("T_camera_link_to_color_frame", T_camera_link_to_color_frame)

    # camera_color_frame -> camera_color_optical_frame (camera_color_optical_joint)
    # Standard optical frame convention: -90° roll, -90° yaw
    T_color_frame_to_color_optical = make_transform(
        xyz=[0, 0, 0],
        rpy=[-np.pi/2, 0, -np.pi/2]
    )
    print_transform("T_color_frame_to_color_optical", T_color_frame_to_color_optical)

    # link_6 -> ee_gripper_link (ee_gripper joint)
    T_link6_to_ee = make_transform(
        xyz=[0.156062, 0, 0],
        rpy=[0, 0, 0]
    )
    print_transform("T_link6_to_ee", T_link6_to_ee)

    # =========================================================================
    # Step 2: Compute T_link6_to_camera_color_optical
    # =========================================================================

    print("\n" + "=" * 70)
    print("Step 2: Chain transforms to get T_link6_to_camera_color_optical")
    print("=" * 70)

    T_link6_to_camera_color_optical = (
        T_link6_to_camera_mount @
        T_camera_mount_to_bottom_screw @
        T_bottom_screw_to_camera_link @
        T_camera_link_to_color_frame @
        T_color_frame_to_color_optical
    )
    print_transform("T_link6_to_camera_color_optical", T_link6_to_camera_color_optical)

    # =========================================================================
    # Step 3: Compute X = T_ee_to_camera_color_optical
    # =========================================================================

    print("\n" + "=" * 70)
    print("Step 3: Compute X = T_ee_to_camera_color_optical")
    print("=" * 70)

    # T_ee_to_camera = T_ee_to_link6 @ T_link6_to_camera
    #                = inv(T_link6_to_ee) @ T_link6_to_camera

    T_ee_to_link6 = np.linalg.inv(T_link6_to_ee)
    print_transform("T_ee_to_link6 (inverse of T_link6_to_ee)", T_ee_to_link6)

    X = T_ee_to_link6 @ T_link6_to_camera_color_optical
    print_transform("X = T_ee_to_camera_color_optical", X)

    # =========================================================================
    # Step 4: Summary
    # =========================================================================

    print("\n" + "=" * 70)
    print("FINAL RESULT: X (T_ee_to_camera_color_optical)")
    print("=" * 70)

    rot = R.from_matrix(X[:3, :3])
    rpy = rot.as_euler('xyz')
    quat = rot.as_quat()

    print(f"""
This is the hand-eye calibration matrix X from URDF.

Translation (meters):
  x = {X[0,3]:.6f}
  y = {X[1,3]:.6f}
  z = {X[2,3]:.6f}

Rotation (degrees):
  roll  = {np.degrees(rpy[0]):.2f}°
  pitch = {np.degrees(rpy[1]):.2f}°
  yaw   = {np.degrees(rpy[2]):.2f}°

Quaternion [x, y, z, w]:
  [{quat[0]:.6f}, {quat[1]:.6f}, {quat[2]:.6f}, {quat[3]:.6f}]

4x4 Matrix:
""")
    print("X = np.array([")
    for row in X:
        print(f"    [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}, {row[3]:12.8f}],")
    print("])")

    # =========================================================================
    # Step 5: Save to file for later use
    # =========================================================================

    import json
    import os

    output_dir = os.path.join(os.path.dirname(__file__), '..', 'data', 'calibration')
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, 'X_from_urdf.json')

    calibration_data = {
        'description': 'X = T_ee_to_camera_color_optical from URDF (wxai_follower.urdf)',
        'source': 'computed from URDF transforms',
        'ee_frame': 'ee_gripper_link',
        'camera_frame': 'camera_color_optical_frame',
        'translation_meters': {
            'x': float(X[0, 3]),
            'y': float(X[1, 3]),
            'z': float(X[2, 3])
        },
        'rotation_degrees': {
            'roll': float(np.degrees(rpy[0])),
            'pitch': float(np.degrees(rpy[1])),
            'yaw': float(np.degrees(rpy[2]))
        },
        'quaternion_xyzw': [float(q) for q in quat],
        'matrix_4x4': X.tolist()
    }

    with open(output_file, 'w') as f:
        json.dump(calibration_data, f, indent=2)

    print(f"\nSaved to: {output_file}")

    return X


if __name__ == '__main__':
    X = main()
