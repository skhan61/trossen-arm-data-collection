#!/usr/bin/env python3
"""
Test URDF-defined X matrix using ROS TF directly.

This script queries TF to get the transform from ee_gripper_link to
camera_color_optical_frame, which should be defined in the URDF for
the follower variant.

This verifies that:
1. The URDF correctly defines the camera frame relative to the robot
2. TF is publishing the expected transform
3. No separate hand-eye calibration is needed if URDF is accurate

Usage:
    1. Launch robot with follower variant:
       ros2 launch trossen_arm_bringup arm.launch.py arm:=wxai variant:=follower ...
    2. Run this script:
       python3 src/test_urdf_X_from_tf.py
"""

import numpy as np
import rclpy
from rclpy.node import Node
from tf2_ros import Buffer, TransformListener
from scipy.spatial.transform import Rotation


def matrix_from_transform(transform):
    """Convert ROS TransformStamped to 4x4 matrix."""
    t = transform.transform.translation
    q = transform.transform.rotation

    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]

    return T


def main():
    rclpy.init()
    node = Node('test_urdf_x')

    tf_buffer = Buffer()
    tf_listener = TransformListener(tf_buffer, node)

    print("=" * 70)
    print("Testing URDF-defined X matrix from TF")
    print("=" * 70)
    print()
    print("Waiting for TF transforms...")

    # Wait for TF to be available
    for i in range(30):
        rclpy.spin_once(node, timeout_sec=0.5)

        try:
            # Try to get the transform
            transform = tf_buffer.lookup_transform(
                'ee_gripper_link',          # target frame
                'camera_color_optical_frame',  # source frame
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5)
            )
            break
        except Exception as e:
            if i % 5 == 0:
                print(f"  Waiting... ({e.__class__.__name__})")
    else:
        print("ERROR: Could not get transform from TF!")
        print()
        print("Make sure:")
        print("  1. Robot is launched with variant:=follower")
        print("  2. Camera is running and publishing TF")
        print()

        # List available frames
        try:
            frames = tf_buffer.all_frames_as_string()
            print("Available TF frames:")
            print(frames)
        except:
            pass

        rclpy.shutdown()
        return 1

    # Convert to matrix
    X_cam2gripper = matrix_from_transform(transform)

    print()
    print("=" * 70)
    print("X from TF: ee_gripper_link <- camera_color_optical_frame")
    print("=" * 70)
    print()

    # Print translation
    t = X_cam2gripper[:3, 3]
    print(f"Translation:")
    print(f"  x = {t[0]*100:7.3f} cm  ({t[0]:8.5f} m)")
    print(f"  y = {t[1]*100:7.3f} cm  ({t[1]:8.5f} m)")
    print(f"  z = {t[2]*100:7.3f} cm  ({t[2]:8.5f} m)")
    print(f"  magnitude = {np.linalg.norm(t)*100:.3f} cm")
    print()

    # Print quaternion
    q = transform.transform.rotation
    print(f"Quaternion (xyzw):")
    print(f"  [{q.x:.6f}, {q.y:.6f}, {q.z:.6f}, {q.w:.6f}]")
    print()

    # Print Euler angles
    euler = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=True)
    print(f"Euler angles (xyz, degrees):")
    print(f"  roll  = {euler[0]:7.2f}°")
    print(f"  pitch = {euler[1]:7.2f}°")
    print(f"  yaw   = {euler[2]:7.2f}°")
    print()

    # Print full matrix
    print("Full 4x4 matrix:")
    print(np.array2string(X_cam2gripper, precision=6, suppress_small=True,
                          formatter={'float': lambda x: f'{x:10.6f}'}))
    print()

    # Also get the inverse (gripper to camera)
    try:
        transform_inv = tf_buffer.lookup_transform(
            'camera_color_optical_frame',  # target
            'ee_gripper_link',             # source
            rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.5)
        )
        X_gripper2cam = matrix_from_transform(transform_inv)

        print("=" * 70)
        print("X inverse from TF: camera_color_optical_frame <- ee_gripper_link")
        print("=" * 70)
        print()
        t_inv = X_gripper2cam[:3, 3]
        print(f"Translation:")
        print(f"  x = {t_inv[0]*100:7.3f} cm")
        print(f"  y = {t_inv[1]*100:7.3f} cm")
        print(f"  z = {t_inv[2]*100:7.3f} cm")
        print()
    except:
        pass

    # Now let's also check other relevant frames
    print("=" * 70)
    print("Checking related TF frames")
    print("=" * 70)
    print()

    frame_pairs = [
        ('base_link', 'ee_gripper_link'),
        ('link_6', 'ee_gripper_link'),
        ('link_6', 'camera_mount_d405'),
        ('camera_mount_d405', 'camera_link'),
        ('camera_link', 'camera_color_optical_frame'),
    ]

    for target, source in frame_pairs:
        try:
            tf = tf_buffer.lookup_transform(target, source, rclpy.time.Time(),
                                           timeout=rclpy.duration.Duration(seconds=0.5))
            t = tf.transform.translation
            q = tf.transform.rotation
            euler = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_euler('xyz', degrees=True)
            print(f"{target} <- {source}:")
            print(f"  xyz: [{t.x*100:.2f}, {t.y*100:.2f}, {t.z*100:.2f}] cm")
            print(f"  rpy: [{euler[0]:.1f}°, {euler[1]:.1f}°, {euler[2]:.1f}°]")
            print()
        except Exception as e:
            print(f"{target} <- {source}: NOT AVAILABLE ({e.__class__.__name__})")
            print()

    rclpy.shutdown()

    print("=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    print()
    print("If the transform ee_gripper_link <- camera_color_optical_frame exists,")
    print("then the URDF already defines X and no separate calibration is needed.")
    print()
    print("You can use TF directly to transform points from camera to robot base:")
    print("  tf_buffer.lookup_transform('base_link', 'camera_color_optical_frame', ...)")
    print()

    return 0


if __name__ == "__main__":
    exit(main())
