"""
Type definitions for the Visual-Haptic Deformation Dataset.
"""

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray


@dataclass
class Sample:
    """Sample metadata (JSON-serializable).

    File: dataset/samples/{sample_id}/sample.json
    """

    sample_id: str  # Unique sample identifier "000001"
    object_id: str  # Reference to object being pressed
    contact_frame: int  # Frame when gripper first touches object
    max_press_frame: int  # Frame at maximum press depth
    sample_rate: int  # Collection loop rate in Hz (not camera hardware rate)
    num_frames: int  # Total frames in sample


@dataclass
class SampleData:
    """Complete sample with all data arrays.

    Directory: dataset/samples/{sample_id}/
    """

    # Metadata (from sample.json)
    sample: Sample

    # Video frames
    rgb: NDArray[np.uint8]  # (N, H, W, 3) RealSense RGB
    gelsight_left: NDArray[np.uint8]  # (N, H, W, 3) left tactile
    gelsight_right: NDArray[np.uint8]  # (N, H, W, 3) right tactile

    # Numpy arrays
    depth: NDArray[np.float32]  # (N, H, W) depth in meters
    poses: NDArray[np.float32]  # (N, 2, 4, 4) T_base_to_gelsight [left, right]


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


# Type aliases for calibration data
Transform4x4 = NDArray[np.float64]  # (4, 4) homogeneous transformation matrix
TuParams = NDArray[np.float64]  # (6,) T(u) model parameters
XMatrix = NDArray[np.float64]  # (4, 4) eye-in-hand calibration matrix


class X:
    """Eye-in-hand calibration matrix slice accessors.

    File: X.npy
    Type: XMatrix (4, 4) float64

    4x4 homogeneous transformation matrix T_camera_to_gripper.

    Usage:
        x: XMatrix = np.load("X.npy")
        rotation = x[X.ROTATION]      # (3, 3) rotation matrix
        translation = x[X.TRANSLATION]  # (3,) translation vector
    """

    ROTATION = np.s_[:3, :3]  # (3, 3) float64
    TRANSLATION = np.s_[:3, 3]  # (3,) float64, meters
    HOMOGENEOUS_ROW = np.s_[3, :]  # [0, 0, 0, 1]


class T(Enum):
    """GelSight T(u) linear model param indices.

    Files: T_u_left_params.npy, T_u_right_params.npy
    Type: TuParams (6,) float64

    T(u) = t0 + k * u, where u = gripper opening (meters)
    """

    T0_X = 0  # float64, meters
    T0_Y = 1  # float64, meters
    T0_Z = 2  # float64, meters
    K_X = 3  # float64, meters/meter
    K_Y = 4  # float64, meters/meter
    K_Z = 5  # float64, meters/meter


# =============================================================================
# Collection Types
# =============================================================================


@dataclass
class CollectionConfig:
    """Data collection parameters."""

    sample_rate: int = 30  # Collection loop rate in Hz (not camera hardware rate)
    contact_tolerance: float = 0.0003  # Contact detection threshold (meters)
    max_press_depth: float = 0.010  # Max press into object (meters)
    step_size: float = 0.001  # Movement step per frame (meters)
    retract_height: float = 0.050  # Height above object after sample (meters)


@dataclass
class ObjectBoundary:
    """Object boundary defined by 4 corners in robot base frame.

    Corners define a rectangle. Robot approaches from the left edge.
    """

    top_left: NDArray[np.float64]  # (3,) xyz in base frame
    top_right: NDArray[np.float64]  # (3,)
    bottom_right: NDArray[np.float64]  # (3,)
    bottom_left: NDArray[np.float64]  # (3,)

    def get_contact_x_at_z(self, z: float) -> float:
        """Get expected contact X position at given Z height.

        Uses left edge (top_left -> bottom_left) for contact line.
        """
        t = (z - self.top_left[2]) / (self.bottom_left[2] - self.top_left[2])
        t = np.clip(t, 0.0, 1.0)
        return float(self.top_left[0] + t * (self.bottom_left[0] - self.top_left[0]))
