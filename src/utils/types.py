"""
Type definitions for the Visual-Haptic Deformation Dataset.
"""

from dataclasses import dataclass
from enum import Enum


@dataclass
class Sample:
    """Sample metadata.

    File: dataset/samples/{sample_id}/sample.json

    Contains:
        - rgb.mp4: RealSense RGB video (N, H, W, 3)
        - depth.npy: Depth frames in meters (N, H, W) float32
        - gelsight_left.mp4: Left GelSight tactile video (N, H, W, 3)
        - gelsight_right.mp4: Right GelSight tactile video (N, H, W, 3)
        - poses.npy: T_base_to_gelsight [left, right] per frame (N, 2, 4, 4) float32
    """

    sample_id: str  # Unique sample identifier "000001"
    object_id: str  # Reference to object being pressed
    contact_frame: int  # Frame when gripper first touches object
    max_press_frame: int  # Frame at maximum press depth
    fps: int  # Frames per second (typically 30)
    num_frames: int  # Total frames in sample


@dataclass
class Object:
    """Object metadata.

    File: dataset/objects/{object_id}.json

    Describes the deformable object being pressed.
    """

    object_id: str  # Unique object identifier "object_001"
    description: str  # Human-readable description "soft foam cube"


@dataclass
class Metadata:
    """Dataset metadata.

    File: dataset/metadata.json

    Top-level dataset information.
    """

    name: str  # Dataset name "visual_haptic_deformation"
    date: str  # Collection date "2026-01-25"
    num_objects: int  # Total number of unique objects
    num_samples: int  # Total number of press samples


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
