#!/usr/bin/env python3
"""
Export X (hand-eye calibration) from camera_pose.launch.py to .npy file.

Reads the transformation from the launch file and saves as a 4x4 numpy array.

Usage:
    python src/calibration/eye_in_hand/export_X_to_npy.py
"""

import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import re
import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.log import get_logger

logger = get_logger(__name__)


def load_X_from_launch():
    """
    Load X by parsing camera_pose.launch.py file.
    Returns 4x4 transformation matrix.
    """
    launch_file = Path(__file__).parent / "camera_pose.launch.py"

    if not launch_file.exists():
        raise FileNotFoundError(f"Launch file not found: {launch_file}")

    with open(launch_file, "r") as f:
        content = f.read()

    def extract_value(param_name):
        pattern = rf'"--{param_name}",\s*\n\s*"([^"]+)"'
        match = re.search(pattern, content)
        if match:
            return float(match.group(1))
        raise ValueError(f"Could not find --{param_name} in launch file")

    # Extract translation
    x = extract_value("x")
    y = extract_value("y")
    z = extract_value("z")

    # Extract quaternion
    qx = extract_value("qx")
    qy = extract_value("qy")
    qz = extract_value("qz")
    qw = extract_value("qw")

    # Convert quaternion to rotation matrix
    quat = np.array([qx, qy, qz, qw])
    rot = R.from_quat(quat)
    R_mat = rot.as_matrix()

    # Build 4x4 transformation matrix
    X = np.eye(4)
    X[:3, :3] = R_mat
    X[:3, 3] = [x, y, z]

    return X, {"x": x, "y": y, "z": z, "qx": qx, "qy": qy, "qz": qz, "qw": qw}


def main():
    logger.info("=" * 60)
    logger.info("Export X to .npy")
    logger.info("=" * 60)

    # Load X from launch file
    X, params = load_X_from_launch()

    logger.info("Loaded from: camera_pose.launch.py")
    logger.info("Translation:")
    logger.info(f"  x = {params['x']:.6f} m")
    logger.info(f"  y = {params['y']:.6f} m")
    logger.info(f"  z = {params['z']:.6f} m")
    logger.info("Quaternion (xyzw):")
    logger.info(f"  qx = {params['qx']:.6f}")
    logger.info(f"  qy = {params['qy']:.6f}")
    logger.info(f"  qz = {params['qz']:.6f}")
    logger.info(f"  qw = {params['qw']:.6f}")

    logger.info("4x4 Transformation Matrix X:")
    for row in X:
        logger.info(
            f"  [{row[0]:12.8f}, {row[1]:12.8f}, {row[2]:12.8f}, {row[3]:12.8f}]"
        )

    # Save to dataset/calibration/
    output_dir = Path(__file__).parent.parent.parent.parent / "dataset" / "calibration"
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / "X.npy"
    np.save(output_file, X)

    logger.info(f"Saved to: {output_file}")

    # Verify by loading back
    X_loaded = np.load(output_file)
    assert np.allclose(X, X_loaded), "Verification failed!"
    logger.info("Verification: OK")

    return 0


if __name__ == "__main__":
    exit(main())
