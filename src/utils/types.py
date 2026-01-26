"""
Type definitions for the Visual-Haptic Deformation Dataset.
"""

from enum import Enum


class X(Enum):
    """Eye-in-hand calibration matrix.

    File: X.npy
    Shape: (4, 4)
    Dtype: float64

    4x4 homogeneous transformation matrix T_camera_to_gripper.
    """

    ROTATION = "X[:3, :3]"  # (3, 3) float64
    TRANSLATION = "X[:3, 3]"  # (3,) float64, meters


class T(Enum):
    """GelSight T(u) linear model params.

    Files: T_u_left_params.npy, T_u_right_params.npy
    Shape: (6,)
    Dtype: float64

    T(u) = t0 + k * u, where u = gripper opening (meters)
    """

    T0_X = 0  # float64, meters
    T0_Y = 1  # float64, meters
    T0_Z = 2  # float64, meters
    K_X = 3  # float64, meters/meter
    K_Y = 4  # float64, meters/meter
    K_Z = 5  # float64, meters/meter
