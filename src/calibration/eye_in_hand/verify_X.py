#!/usr/bin/env python3
"""
Verify X (hand-eye calibration) using checkerboard poses.

Verification principle:
- The checkerboard is FIXED in the world (doesn't move)
- For each robot pose, we compute: T_base_to_target = T_base_to_ee @ X @ T_cam_to_target
- If X is correct, all poses should give the SAME T_base_to_target (within noise)
- Large inconsistency = X is wrong

Usage:
    python src/calibration/eye_in_hand/verify_X.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import json
import re
import yaml
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.log import get_logger

logger = get_logger(__name__)


def rotation_matrix_to_euler(R_mat):
    """Convert 3x3 rotation matrix to euler angles (degrees)."""
    rot = R.from_matrix(R_mat)
    euler = rot.as_euler("xyz", degrees=True)
    return euler


def rotation_error_angle(R1, R2):
    """
    Compute rotation error between two rotation matrices.

    Returns angle in degrees.

    Math: R_error = R1 @ R2.T
          angle = arccos((trace(R_error) - 1) / 2)
    """
    R_error = R1 @ R2.T
    trace = np.trace(R_error)
    # Clamp to [-1, 1] to handle numerical errors
    cos_angle = np.clip((trace - 1) / 2, -1, 1)
    angle_rad = np.arccos(cos_angle)
    return np.degrees(angle_rad)


def log_matrix(name, T):
    """Log a 4x4 transformation matrix."""
    logger.info(f"  {name}:")
    for i in range(4):
        logger.info(
            f"    [{T[i, 0]:10.4f}, {T[i, 1]:10.4f}, {T[i, 2]:10.4f}, {T[i, 3]:10.4f}]"
        )


def load_X_from_launch():
    """
    Load X by parsing camera_pose.launch.py file.

    Reads the actual values from the launch file arguments.
    """
    launch_file = Path(__file__).parent / "camera_pose.launch.py"

    if not launch_file.exists():
        raise FileNotFoundError(f"Launch file not found: {launch_file}")

    # Read and parse the launch file
    with open(launch_file, "r") as f:
        content = f.read()

    def extract_value(param_name):
        # Match pattern: "--param_name",\n                "value",
        pattern = rf'"--{param_name}",\s*\n\s*"([^"]+)"'
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
        raise ValueError(f"Could not find --{param_name} in launch file")

    # Extract translation
    x = extract_value("x")
    y = extract_value("y")
    z = extract_value("z")
    t = np.array([x, y, z])

    # Extract quaternion
    qx = extract_value("qx")
    qy = extract_value("qy")
    qz = extract_value("qz")
    qw = extract_value("qw")
    quat = np.array([qx, qy, qz, qw])

    # Convert quaternion to rotation matrix
    rot = R.from_quat(quat)
    R_mat = rot.as_matrix()

    # Build 4x4 transformation matrix
    X = np.eye(4)
    X[:3, :3] = R_mat
    X[:3, 3] = t

    return X


def verify_X(X, data, name):
    """
    Verify X by checking checkerboard position consistency.

    Args:
        X: 4x4 transformation matrix (T_ee_to_camera)
        data: verification data with poses
        name: name of X source for logging

    Returns:
        dict with max_translation_error_mm and max_rotation_error_deg
    """
    logger.info("=" * 70)
    logger.info(f"Testing X from {name}")
    logger.info("=" * 70)

    # Log X matrix
    logger.info("X = T_ee_to_camera:")
    for i in range(4):
        logger.info(
            f"  [{X[i, 0]:10.6f}, {X[i, 1]:10.6f}, {X[i, 2]:10.6f}, {X[i, 3]:10.6f}]"
        )

    X_euler = rotation_matrix_to_euler(X[:3, :3])
    logger.info(
        f"X translation (mm): [{X[0, 3] * 1000:.2f}, {X[1, 3] * 1000:.2f}, {X[2, 3] * 1000:.2f}]"
    )
    logger.info(
        f"X rotation (deg):   [roll={X_euler[0]:.2f}, pitch={X_euler[1]:.2f}, yaw={X_euler[2]:.2f}]"
    )

    # Store full transforms for each pose
    target_transforms = []

    for pose in data["poses"]:
        pose_id = pose["pose_id"]

        # T_base_to_ee (gripper pose in base frame)
        T_base_to_ee = np.array(pose["robot"]["T_gripper2base"])

        # T_cam_to_target (checkerboard pose in camera frame)
        T_cam_to_target = np.array(pose["checkerboard"]["T_target2cam"])

        # Compute: T_base_to_target = T_base_to_ee @ X @ T_cam_to_target
        T_base_to_target = T_base_to_ee @ X @ T_cam_to_target

        target_transforms.append(T_base_to_target)

        # Extract components
        t = T_base_to_target[:3, 3]
        R_mat = T_base_to_target[:3, :3]
        euler = rotation_matrix_to_euler(R_mat)

        logger.info("-" * 70)
        logger.info(f"POSE {pose_id}")
        logger.info("-" * 70)

        log_matrix("T_base_to_ee (from robot)", T_base_to_ee)
        ee_euler = rotation_matrix_to_euler(T_base_to_ee[:3, :3])
        logger.info(
            f"    translation (mm): [{T_base_to_ee[0, 3] * 1000:.2f}, {T_base_to_ee[1, 3] * 1000:.2f}, {T_base_to_ee[2, 3] * 1000:.2f}]"
        )
        logger.info(
            f"    rotation (deg):   [roll={ee_euler[0]:.2f}, pitch={ee_euler[1]:.2f}, yaw={ee_euler[2]:.2f}]"
        )

        log_matrix("T_cam_to_target (from solvePnP)", T_cam_to_target)
        cam_euler = rotation_matrix_to_euler(T_cam_to_target[:3, :3])
        logger.info(
            f"    translation (mm): [{T_cam_to_target[0, 3] * 1000:.2f}, {T_cam_to_target[1, 3] * 1000:.2f}, {T_cam_to_target[2, 3] * 1000:.2f}]"
        )
        logger.info(
            f"    rotation (deg):   [roll={cam_euler[0]:.2f}, pitch={cam_euler[1]:.2f}, yaw={cam_euler[2]:.2f}]"
        )

        log_matrix(
            "T_base_to_target = T_base_to_ee @ X @ T_cam_to_target", T_base_to_target
        )
        logger.info(
            f"    translation (mm): [{t[0] * 1000:.2f}, {t[1] * 1000:.2f}, {t[2] * 1000:.2f}]"
        )
        logger.info(
            f"    rotation (deg):   [roll={euler[0]:.2f}, pitch={euler[1]:.2f}, yaw={euler[2]:.2f}]"
        )

    # Compute pairwise differences
    logger.info("=" * 70)
    logger.info(
        "ERROR ANALYSIS: Checkerboard should be at SAME position from all poses"
    )
    logger.info("=" * 70)

    max_trans_error = 0
    max_rot_error = 0

    for i in range(len(target_transforms)):
        for j in range(i + 1, len(target_transforms)):
            T1 = target_transforms[i]
            T2 = target_transforms[j]

            # Translation error
            t1 = T1[:3, 3]
            t2 = T2[:3, 3]
            trans_diff = t1 - t2
            trans_error = np.linalg.norm(trans_diff) * 1000  # mm
            max_trans_error = max(max_trans_error, trans_error)

            # Rotation error
            R1 = T1[:3, :3]
            R2 = T2[:3, :3]
            rot_error = rotation_error_angle(R1, R2)
            max_rot_error = max(max_rot_error, rot_error)

            # Error transform: T_error = T1 @ inv(T2)
            T_error = T1 @ np.linalg.inv(T2)
            error_euler = rotation_matrix_to_euler(T_error[:3, :3])

            logger.info(f"Pose {i + 1} vs Pose {j + 1}:")
            logger.info("  Translation error:")
            logger.info(
                f"    Δt = [{trans_diff[0] * 1000:.2f}, {trans_diff[1] * 1000:.2f}, {trans_diff[2] * 1000:.2f}] mm"
            )
            logger.info(f"    ||Δt|| = {trans_error:.2f} mm")
            logger.info("  Rotation error:")
            logger.info(f"    angle = {rot_error:.2f}°")
            logger.info(
                f"    Δeuler = [Δroll={error_euler[0]:.2f}°, Δpitch={error_euler[1]:.2f}°, Δyaw={error_euler[2]:.2f}°]"
            )

            log_matrix("T_error = T1 @ inv(T2)", T_error)

    # Verdict
    logger.info("=" * 70)
    logger.info("VERDICT")
    logger.info("=" * 70)
    logger.info(f"Max translation error: {max_trans_error:.2f} mm")
    logger.info(f"Max rotation error:    {max_rot_error:.2f}°")

    if max_trans_error < 10 and max_rot_error < 5:
        logger.info("RESULT: GOOD - X is accurate")
    elif max_trans_error < 30 and max_rot_error < 15:
        logger.info("RESULT: ACCEPTABLE - X has some error")
    else:
        logger.info("RESULT: POOR - X needs calibration")

    return {"translation_mm": max_trans_error, "rotation_deg": max_rot_error}


def save_calibration_yaml(X, error, source_name, output_dir):
    """Save the best X calibration to a YAML file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    yaml_path = output_dir / "hand_eye_calibration.yaml"

    # Get rotation components
    rot = R.from_matrix(X[:3, :3])
    euler = rot.as_euler("xyz", degrees=True)
    quat = rot.as_quat()  # [x, y, z, w]

    # Determine status
    if error["translation_mm"] < 10 and error["rotation_deg"] < 5:
        status = "EXCELLENT"
    elif error["translation_mm"] < 30 and error["rotation_deg"] < 15:
        status = "ACCEPTABLE"
    else:
        status = "POOR"

    calibration_data = {
        "X_cam2gripper_matrix": X.tolist(),
        "rotation_matrix": X[:3, :3].tolist(),
        "translation_xyz_meters": {
            "x": float(X[0, 3]),
            "y": float(X[1, 3]),
            "z": float(X[2, 3]),
        },
        "rotation_euler_degrees": {
            "roll": float(euler[0]),
            "pitch": float(euler[1]),
            "yaw": float(euler[2]),
        },
        "quaternion_xyzw": [float(q) for q in quat],
        "frames": {
            "parent": "ee_gripper_link",
            "child": "camera_color_optical_frame",
        },
        "verification": {
            "source": source_name,
            "max_translation_error_mm": float(error["translation_mm"]),
            "max_rotation_error_deg": float(error["rotation_deg"]),
            "status": status,
        },
        "method": source_name,
    }

    with open(yaml_path, "w") as f:
        f.write("# Hand-Eye Calibration Result (X = T_ee_to_camera)\n")
        f.write(f"# Source: {source_name}\n")
        f.write(
            f"# Verified: Max error {error['translation_mm']:.2f} mm, {error['rotation_deg']:.2f} deg\n"
        )
        f.write("#\n")
        f.write(
            "# This transforms points from camera_color_optical_frame to ee_gripper_link\n"
        )
        f.write("# Usage: T_base_to_point = T_base_to_ee @ X @ T_cam_to_point\n\n")
        yaml.dump(calibration_data, f, default_flow_style=False, sort_keys=False)

    logger.info(f"Saved calibration YAML to: {yaml_path}")
    return yaml_path


def main():
    logger.info("=" * 70)
    logger.info("Verify X (Hand-Eye Calibration)")
    logger.info("=" * 70)

    # Load verification data from: src/calibration/eye_in_hand/Xverification/
    verif_file = Path(__file__).parent / "Xverification" / "verification_data.json"

    if not verif_file.exists():
        logger.error(f"Verification data not found: {verif_file}")
        return 1

    with open(verif_file, "r") as f:
        data = json.load(f)

    logger.info(f"Loaded {data['num_poses']} poses from: {verif_file}")

    # Load and test X from launch file only
    X_launch = load_X_from_launch()
    error_launch = verify_X(X_launch, data, "camera_pose.launch.py")

    # Summary
    logger.info("=" * 70)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 70)

    logger.info("  Source              | Translation Error | Rotation Error")
    logger.info("  " + "-" * 60)
    logger.info(
        f"  X from launch.py    | {error_launch['translation_mm']:>15.2f} mm | {error_launch['rotation_deg']:>12.2f}°"
    )

    best_name = "launch.py"
    best_X = X_launch
    best_error = error_launch

    logger.info(
        f"  Result: {best_name} (translation={best_error['translation_mm']:.2f}mm, rotation={best_error['rotation_deg']:.2f}°)"
    )

    # Save best X to YAML (to project root: data/calibration/)
    output_dir = Path(__file__).parent.parent.parent.parent / "data" / "calibration"
    save_calibration_yaml(best_X, best_error, best_name, output_dir)

    return 0


if __name__ == "__main__":
    exit(main())
