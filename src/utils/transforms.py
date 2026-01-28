"""
Transform utilities for computing GelSight poses.

The key transform chain:
    T_base_to_gelsight = T_base_to_ee @ X @ T_cam_to_gelsight

Where:
    - T_base_to_ee: Robot forward kinematics (from robot)
    - X: Eye-in-hand calibration matrix (from X.npy)
    - T_cam_to_gelsight: GelSight position relative to camera (from T(u) model)

T(u) Linear Model:
    T_cam_to_gelsight = t0 + k * u
    where u = gripper opening in meters
"""

from pathlib import Path

import numpy as np

from src.utils.types import T, Transform4x4, TuParams, XMatrix


def compute_T_u(T_u_params: TuParams, gripper_opening: float) -> Transform4x4:
    """
    Compute T_cam_to_gelsight from T(u) linear model.

    T(u) = t0 + k * u

    Args:
        T_u_params: (6,) T(u) parameters, see T enum for indices
        gripper_opening: Gripper opening in meters

    Returns:
        T_cam_to_gelsight: (4, 4) homogeneous transformation matrix
    """
    t0 = np.array(
        [
            T_u_params[T.T0_X.value],
            T_u_params[T.T0_Y.value],
            T_u_params[T.T0_Z.value],
        ]
    )
    k = np.array(
        [
            T_u_params[T.K_X.value],
            T_u_params[T.K_Y.value],
            T_u_params[T.K_Z.value],
        ]
    )

    # Translation: t = t0 + k * u
    translation = t0 + k * gripper_opening

    # Build 4x4 homogeneous matrix (identity rotation)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = translation

    return transform


def compute_T_base_to_gelsight(
    T_base_to_ee: Transform4x4,
    X: XMatrix,
    T_u_params: TuParams,
    gripper_opening: float,
) -> Transform4x4:
    """
    Compute full transform from robot base to GelSight sensor.

    T_base_to_gelsight = T_base_to_ee @ X @ T_cam_to_gelsight

    Args:
        T_base_to_ee: (4, 4) Robot end-effector pose in base frame
        X: (4, 4) Eye-in-hand calibration
        T_u_params: (6,) GelSight T(u) model parameters
        gripper_opening: Gripper opening in meters

    Returns:
        T_base_to_gelsight: (4, 4) GelSight pose in base frame
    """
    T_cam_to_gelsight = compute_T_u(T_u_params, gripper_opening)
    T_base_to_gelsight = T_base_to_ee @ X @ T_cam_to_gelsight
    return T_base_to_gelsight


def compute_both_gelsight_poses(
    T_base_to_ee: Transform4x4,
    X: XMatrix,
    T_u_left_params: TuParams,
    T_u_right_params: TuParams,
    gripper_opening: float,
) -> np.ndarray:
    """
    Compute poses for both left and right GelSight sensors.

    Args:
        T_base_to_ee: (4, 4) Robot end-effector pose in base frame
        X: (4, 4) Eye-in-hand calibration
        T_u_left_params: (6,) Left GelSight T(u) parameters
        T_u_right_params: (6,) Right GelSight T(u) parameters
        gripper_opening: Gripper opening in meters

    Returns:
        poses: (2, 4, 4) float32 array [T_left, T_right]
    """
    T_left = compute_T_base_to_gelsight(
        T_base_to_ee, X, T_u_left_params, gripper_opening
    )
    T_right = compute_T_base_to_gelsight(
        T_base_to_ee, X, T_u_right_params, gripper_opening
    )
    return np.array([T_left, T_right], dtype=np.float32)


def load_calibration(
    calibration_dir: str | Path,
) -> tuple[XMatrix, TuParams, TuParams]:
    """
    Load calibration files from directory.

    Args:
        calibration_dir: Path to dataset/calibration/

    Returns:
        X: (4, 4) Eye-in-hand calibration
        T_u_left: (6,) Left GelSight T(u) parameters
        T_u_right: (6,) Right GelSight T(u) parameters
    """
    calibration_dir = Path(calibration_dir)

    X = np.load(calibration_dir / "X.npy")
    T_u_left = np.load(calibration_dir / "T_u_left_params.npy")
    T_u_right = np.load(calibration_dir / "T_u_right_params.npy")

    return X, T_u_left, T_u_right


def compute_point_in_base_frame(
    T_base_to_ee: Transform4x4,
    X: XMatrix,
    point_in_camera: np.ndarray,
) -> np.ndarray:
    """
    Transform a 3D point from camera frame to robot base frame.

    T_base_to_point = T_base_to_ee @ X @ T_cam_to_point

    Args:
        T_base_to_ee: (4, 4) Robot end-effector pose in base frame
        X: (4, 4) Eye-in-hand calibration matrix
        point_in_camera: (3,) Point in camera frame [x, y, z] in meters

    Returns:
        point_in_base: (3,) Point in robot base frame [x, y, z] in meters
    """
    # Build 4x4 transform for point (identity rotation, translation = point)
    T_cam_to_point = np.eye(4, dtype=np.float64)
    T_cam_to_point[:3, 3] = point_in_camera

    # Transform chain
    T_base_to_point = T_base_to_ee @ X @ T_cam_to_point

    # Extract translation (the point position in base frame)
    return T_base_to_point[:3, 3]


def is_valid_transform(transform: Transform4x4, tol: float = 1e-6) -> bool:
    """
    Check if a 4x4 matrix is a valid homogeneous transformation.

    Args:
        transform: (4, 4) matrix to check
        tol: Tolerance for checks

    Returns:
        True if valid transformation matrix
    """
    if transform.shape != (4, 4):
        return False

    # Check bottom row is [0, 0, 0, 1]
    if not np.allclose(transform[3, :], [0, 0, 0, 1], atol=tol):
        return False

    # Check rotation matrix is orthonormal
    R = transform[:3, :3]
    if not np.allclose(R @ R.T, np.eye(3), atol=tol):
        return False

    # Check determinant is +1 (not -1, which would be reflection)
    if not np.isclose(np.linalg.det(R), 1.0, atol=tol):
        return False

    return True
