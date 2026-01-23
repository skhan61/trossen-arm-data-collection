#!/usr/bin/env python3
"""
Compute hand-eye calibration X from data_sample.yaml
"""

import yaml
import numpy as np
import cv2
from pathlib import Path


def load_calibration_data(yaml_file):
    """Load calibration data from YAML file."""
    with open(yaml_file, 'r') as f:
        data = yaml.safe_load(f)

    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    for sample in data:
        # Extract effector_wrt_world (4x4 matrix as flat list)
        effector = np.array(sample['effector_wrt_world']).reshape(4, 4)
        R_gripper2base.append(effector[:3, :3])
        t_gripper2base.append(effector[:3, 3])

        # Extract object_wrt_sensor (4x4 matrix as flat list)
        obj = np.array(sample['object_wrt_sensor']).reshape(4, 4)
        R_target2cam.append(obj[:3, :3])
        t_target2cam.append(obj[:3, 3])

    return R_gripper2base, t_gripper2base, R_target2cam, t_target2cam


def main():
    yaml_file = Path("/home/skhan61/Desktop/trossen-arm-data-collection/data_sample.yaml")

    print("=" * 70)
    print("Hand-Eye Calibration - Computing X Transformation")
    print("=" * 70)
    print(f"\nLoading data from: {yaml_file}")

    # Load data
    R_gripper2base, t_gripper2base, R_target2cam, t_target2cam = load_calibration_data(yaml_file)

    print(f"Loaded {len(R_gripper2base)} calibration samples\n")

    # Compute hand-eye calibration using OpenCV
    # This solves: gripper2base * X = X * target2cam
    # Where X is cam2gripper transformation

    print("Computing hand-eye calibration using Tsai's method...")
    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Create 4x4 transformation matrix
    X = np.eye(4)
    X[:3, :3] = R_cam2gripper
    X[:3, 3] = t_cam2gripper.flatten()

    print("\n" + "=" * 70)
    print("RESULT: X = Camera-to-Gripper Transformation")
    print("=" * 70)
    print("\n4x4 Transformation Matrix:")
    print(X)

    print("\n\nRotation Matrix (3x3):")
    print(R_cam2gripper)

    print("\n\nTranslation Vector (meters):")
    print(f"x: {t_cam2gripper[0][0]:.6f} m")
    print(f"y: {t_cam2gripper[1][0]:.6f} m")
    print(f"z: {t_cam2gripper[2][0]:.6f} m")

    # Convert rotation to axis-angle for easier interpretation
    rvec, _ = cv2.Rodrigues(R_cam2gripper)
    angle = np.linalg.norm(rvec)
    axis = rvec / angle if angle > 0 else rvec

    print("\n\nRotation (Axis-Angle):")
    print(f"Angle: {np.degrees(angle):.2f} degrees")
    print(f"Axis: [{axis[0][0]:.4f}, {axis[1][0]:.4f}, {axis[2][0]:.4f}]")

    # Save to file
    output_file = Path("/home/skhan61/Desktop/trossen-arm-data-collection/hand_eye_result.yaml")
    result = {
        'X_cam2gripper_matrix': X.tolist(),
        'rotation_matrix': R_cam2gripper.tolist(),
        'translation_xyz_meters': {
            'x': float(t_cam2gripper[0][0]),
            'y': float(t_cam2gripper[1][0]),
            'z': float(t_cam2gripper[2][0])
        },
        'rotation_axis_angle': {
            'angle_degrees': float(np.degrees(angle)),
            'axis': [float(axis[0][0]), float(axis[1][0]), float(axis[2][0])]
        }
    }

    with open(output_file, 'w') as f:
        yaml.dump(result, f, default_flow_style=False)

    print(f"\n\nResults saved to: {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
