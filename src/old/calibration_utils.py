#!/usr/bin/env python3
"""
Utility functions for loading camera calibration data.

Supports both:
- Factory RealSense calibration (ROS YAML format)
- Custom chessboard calibration (JSON format)
"""

import json
import yaml
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional


def load_ros_calibration(yaml_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load camera calibration from ROS-format YAML file.

    Args:
        yaml_file: Path to YAML file with ROS camera_info format

    Returns:
        Tuple of (camera_matrix, dist_coeffs, metadata)
        - camera_matrix: 3x3 numpy array
        - dist_coeffs: 1x5 numpy array
        - metadata: dict with additional info
    """
    with open(yaml_file, 'r') as f:
        calib = yaml.safe_load(f)

    # Extract camera matrix (convert flat list to 3x3)
    K_data = calib['camera_matrix']['data']
    camera_matrix = np.array(K_data).reshape(3, 3)

    # Extract distortion coefficients
    dist_coeffs = np.array(calib['distortion_coefficients']['data'])

    # Metadata
    metadata = {
        'source': 'factory',
        'format': 'ros_yaml',
        'camera_name': calib['camera_name'],
        'image_width': calib['image_width'],
        'image_height': calib['image_height'],
        'distortion_model': calib['distortion_model']
    }

    return camera_matrix, dist_coeffs, metadata


def load_json_calibration(json_file: Path) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load camera calibration from JSON file (custom chessboard calibration).

    Args:
        json_file: Path to JSON file with custom calibration

    Returns:
        Tuple of (camera_matrix, dist_coeffs, metadata)
        - camera_matrix: 3x3 numpy array
        - dist_coeffs: 1x5 numpy array
        - metadata: dict with additional info
    """
    with open(json_file, 'r') as f:
        calib = json.load(f)

    # Extract camera matrix
    camera_matrix = np.array(calib['camera_matrix'])

    # Extract distortion coefficients
    dist_coeffs = np.array(calib['dist_coeffs']).flatten()

    # Metadata
    metadata = {
        'source': 'custom',
        'format': 'json',
        'reprojection_error': calib.get('reprojection_error', None),
        'num_images': calib.get('num_images', None),
        'image_width': calib.get('image_size', [None, None])[0],
        'image_height': calib.get('image_size', [None, None])[1]
    }

    return camera_matrix, dist_coeffs, metadata


def load_camera_calibration(
    calibration_type: Optional[str] = None,
    project_root: Optional[Path] = None
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load camera calibration with automatic detection.

    Args:
        calibration_type: 'factory', 'custom', or None (auto-detect)
        project_root: Path to project root (default: auto-detect)

    Returns:
        Tuple of (camera_matrix, dist_coeffs, metadata)

    Raises:
        FileNotFoundError: If no calibration files found
        ValueError: If invalid calibration_type specified
    """
    if project_root is None:
        # Auto-detect project root (assume this script is in src/)
        project_root = Path(__file__).parent.parent

    # Define calibration file paths
    factory_calib = project_root / "data" / "realsense_calibration" / "realsense_color.yaml"
    custom_calib = project_root / "data" / "camera_calibration_data" / "camera_intrinsics.json"

    # Auto-detect if not specified
    if calibration_type is None:
        if custom_calib.exists():
            calibration_type = 'custom'
            print(f"Auto-detected custom calibration: {custom_calib}")
        elif factory_calib.exists():
            calibration_type = 'factory'
            print(f"Auto-detected factory calibration: {factory_calib}")
        else:
            raise FileNotFoundError(
                "No calibration files found. Please run:\n"
                "  - python src/extract_realsense_calibration.py (for factory calibration)\n"
                "  - python src/calibrate_camera.py (for custom calibration)"
            )

    # Load specified calibration
    if calibration_type == 'factory':
        if not factory_calib.exists():
            raise FileNotFoundError(
                f"Factory calibration not found at {factory_calib}\n"
                "Run: python src/extract_realsense_calibration.py"
            )
        return load_ros_calibration(factory_calib)

    elif calibration_type == 'custom':
        if not custom_calib.exists():
            raise FileNotFoundError(
                f"Custom calibration not found at {custom_calib}\n"
                "Run: python src/calibrate_camera.py"
            )
        return load_json_calibration(custom_calib)

    else:
        raise ValueError(
            f"Invalid calibration_type: {calibration_type}\n"
            "Must be 'factory', 'custom', or None (auto-detect)"
        )


def print_calibration_info(camera_matrix: np.ndarray, dist_coeffs: np.ndarray, metadata: Dict):
    """
    Print calibration information in a readable format.

    Args:
        camera_matrix: 3x3 camera matrix
        dist_coeffs: Distortion coefficients
        metadata: Additional metadata
    """
    print("=" * 60)
    print("Camera Calibration Info")
    print("=" * 60)
    print(f"Source: {metadata.get('source', 'unknown')}")
    print(f"Format: {metadata.get('format', 'unknown')}")
    print()

    print("Camera Matrix (K):")
    print(camera_matrix)
    print()

    fx, fy = camera_matrix[0, 0], camera_matrix[1, 1]
    cx, cy = camera_matrix[0, 2], camera_matrix[1, 2]
    print(f"Focal length:     fx = {fx:.2f} pixels, fy = {fy:.2f} pixels")
    print(f"Principal point:  cx = {cx:.2f} pixels, cy = {cy:.2f} pixels")
    print()

    print("Distortion Coefficients:")
    print(f"  [k1, k2, p1, p2, k3] = {dist_coeffs}")
    print()

    if metadata.get('reprojection_error'):
        print(f"Reprojection error: {metadata['reprojection_error']:.4f} pixels")

    if metadata.get('image_width') and metadata.get('image_height'):
        print(f"Image size: {metadata['image_width']}x{metadata['image_height']}")

    print("=" * 60)


# Example usage
if __name__ == "__main__":
    import sys

    print("Camera Calibration Loader")
    print()

    # Try to load calibration
    try:
        # Auto-detect
        camera_matrix, dist_coeffs, metadata = load_camera_calibration()
        print_calibration_info(camera_matrix, dist_coeffs, metadata)

    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
