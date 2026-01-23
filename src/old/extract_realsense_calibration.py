#!/usr/bin/env python3
"""
Extract RealSense calibration in sensor_msgs/CameraInfo format
Compatible with ROS camera_info_manager

This script extracts the factory calibration from the RealSense camera
and saves it in ROS-compatible YAML format.
"""

import pyrealsense2 as rs
import yaml
from datetime import datetime
from pathlib import Path
import logging

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "realsense_calibration.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "realsense_calibration"
DATA_DIR.mkdir(exist_ok=True, parents=True)


def intrinsics_to_camera_info_dict(intrinsics, camera_name="camera"):
    """
    Convert RealSense intrinsics to standard ROS camera calibration file format.
    This matches the format from camera_calibration package.

    Args:
        intrinsics: pyrealsense2 intrinsics object
        camera_name: Name for this camera in ROS format

    Returns:
        Dictionary in ROS camera_info format
    """
    # Camera matrix (3x3)
    camera_matrix = {
        "rows": 3,
        "cols": 3,
        "data": [
            intrinsics.fx,
            0.0,
            intrinsics.ppx,
            0.0,
            intrinsics.fy,
            intrinsics.ppy,
            0.0,
            0.0,
            1.0,
        ],
    }

    # Distortion coefficients (1x5)
    # RealSense uses Brown-Conrady model: [k1, k2, p1, p2, k3]
    distortion_coefficients = {"rows": 1, "cols": 5, "data": list(intrinsics.coeffs)}

    # Rectification matrix (3x3) - identity for monocular
    rectification_matrix = {
        "rows": 3,
        "cols": 3,
        "data": [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
    }

    # Projection matrix (3x4)
    # For monocular camera: P = [fx  0  cx  0]
    #                          [ 0 fy  cy  0]
    #                          [ 0  0   1  0]
    projection_matrix = {
        "rows": 3,
        "cols": 4,
        "data": [
            intrinsics.fx,
            0.0,
            intrinsics.ppx,
            0.0,
            0.0,
            intrinsics.fy,
            intrinsics.ppy,
            0.0,
            0.0,
            0.0,
            1.0,
            0.0,
        ],
    }

    # Complete camera info in ROS format
    camera_info = {
        "image_width": intrinsics.width,
        "image_height": intrinsics.height,
        "camera_name": camera_name,
        "camera_matrix": camera_matrix,
        "distortion_model": "plumb_bob",
        "distortion_coefficients": distortion_coefficients,
        "rectification_matrix": rectification_matrix,
        "projection_matrix": projection_matrix,
    }

    return camera_info


def save_calibration_ros_format(output_dir=None):
    """
    Extract factory calibration from RealSense camera and save in ROS camera_info format.

    The RealSense camera comes pre-calibrated from the factory. This function extracts
    that calibration and saves it in a format compatible with ROS camera_info_manager.

    Args:
        output_dir: Directory to save calibration files (default: data/realsense_calibration/)
    """
    if output_dir is None:
        output_dir = DATA_DIR
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)

    logger.info("=" * 60)
    logger.info("RealSense Factory Calibration Extraction")
    logger.info("=" * 60)
    logger.info("Connecting to RealSense camera...")

    # Initialize pipeline
    pipeline = rs.pipeline()
    config = rs.config()

    # Enable streams - using common resolutions
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start pipeline
    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        logger.error(f"Failed to start RealSense pipeline: {e}")
        logger.error("Make sure the RealSense camera is connected.")
        return False

    try:
        # Get intrinsics from the stream profiles
        color_profile = profile.get_stream(rs.stream.color)
        depth_profile = profile.get_stream(rs.stream.depth)

        color_intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        depth_intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()

        # Get device info
        device = profile.get_device()
        device_name = device.get_info(rs.camera_info.name)
        device_serial = device.get_info(rs.camera_info.serial_number)
        firmware_version = device.get_info(rs.camera_info.firmware_version)

        logger.info(f"Device: {device_name}")
        logger.info(f"Serial: {device_serial}")
        logger.info(f"Firmware: {firmware_version}")
        logger.info("")

        # Print color camera info
        logger.info(f"Color Camera: {color_intrinsics.width}x{color_intrinsics.height}")
        logger.info(f"  fx = {color_intrinsics.fx:.2f} pixels")
        logger.info(f"  fy = {color_intrinsics.fy:.2f} pixels")
        logger.info(f"  cx = {color_intrinsics.ppx:.2f} pixels")
        logger.info(f"  cy = {color_intrinsics.ppy:.2f} pixels")
        logger.info(f"  Distortion model: {color_intrinsics.model}")
        logger.info(f"  Distortion coeffs: {color_intrinsics.coeffs}")
        logger.info("")

        # Print depth camera info
        logger.info(f"Depth Camera: {depth_intrinsics.width}x{depth_intrinsics.height}")
        logger.info(f"  fx = {depth_intrinsics.fx:.2f} pixels")
        logger.info(f"  fy = {depth_intrinsics.fy:.2f} pixels")
        logger.info(f"  cx = {depth_intrinsics.ppx:.2f} pixels")
        logger.info(f"  cy = {depth_intrinsics.ppy:.2f} pixels")
        logger.info(f"  Distortion model: {depth_intrinsics.model}")
        logger.info(f"  Distortion coeffs: {depth_intrinsics.coeffs}")
        logger.info("")

        # Convert to ROS format
        color_camera_info = intrinsics_to_camera_info_dict(
            color_intrinsics, camera_name="realsense_color"
        )

        depth_camera_info = intrinsics_to_camera_info_dict(
            depth_intrinsics, camera_name="realsense_depth"
        )

        # Save color camera info
        color_file = output_dir / "realsense_color.yaml"
        with open(color_file, "w") as f:
            yaml.dump(color_camera_info, f, default_flow_style=None, sort_keys=False)
        logger.info(f"✓ Color camera calibration saved to: {color_file}")

        # Save depth camera info
        depth_file = output_dir / "realsense_depth.yaml"
        with open(depth_file, "w") as f:
            yaml.dump(depth_camera_info, f, default_flow_style=None, sort_keys=False)
        logger.info(f"✓ Depth camera calibration saved to: {depth_file}")

        # Also save metadata
        metadata = {
            "device_name": device_name,
            "serial_number": device_serial,
            "firmware_version": firmware_version,
            "extraction_date": datetime.now().isoformat(),
            "color_resolution": f"{color_intrinsics.width}x{color_intrinsics.height}",
            "depth_resolution": f"{depth_intrinsics.width}x{depth_intrinsics.height}",
        }

        metadata_file = output_dir / "device_info.yaml"
        with open(metadata_file, "w") as f:
            yaml.dump(metadata, f, default_flow_style=None)
        logger.info(f"✓ Device metadata saved to: {metadata_file}")

        logger.info("")
        logger.info("=" * 60)
        logger.info("Calibration extraction complete!")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Note: These are the factory calibration parameters from Intel.")
        logger.info(
            "For hand-eye calibration, you may still want to run the chessboard"
        )
        logger.info(
            "calibration (calibrate_camera.py) to verify and refine these values."
        )
        logger.info("")

        return True

    except Exception as e:
        logger.error(f"Error extracting calibration: {e}")
        return False

    finally:
        pipeline.stop()
        logger.info("Pipeline stopped")


def main():
    """Main entry point."""
    logger.info(f"Logging to: {LOG_FILE}")
    logger.info(f"Output directory: {DATA_DIR}")
    logger.info("")

    success = save_calibration_ros_format()

    if success:
        logger.info("Successfully extracted RealSense calibration!")
        return 0
    else:
        logger.error("Failed to extract calibration")
        return 1


if __name__ == "__main__":
    exit(main())
