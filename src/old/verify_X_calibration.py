#!/usr/bin/env python3
"""
Hand-Eye Calibration (X) Verification Data Collection

This script collects data to verify the quality of the hand-eye calibration matrix X
stored in hand_eye_result.yaml.

Features:
1. Shows live camera feed with 8x6 checkerboard detection
2. Robot is in gravity compensation mode (move by hand)
3. When checkerboard is detected, robot locks position
4. Press 's' to save current pose (image + robot pose)
5. Press 'q' to quit
6. All data saved to data/Xverification/

Usage:
    python src/verify_X_calibration.py
"""

import cv2
import numpy as np
import json
import yaml
import time
import logging
import pyrealsense2 as rs
from pathlib import Path
from datetime import datetime
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# ============================================================================
# Configuration
# ============================================================================

# Robot IP address
ARM_IP = "192.168.1.99"

# Checkerboard parameters - 8x6 internal corners
BOARD_SIZE = (8, 6)  # (columns, rows) internal corners
SQUARE_SIZE = 0.025  # 25mm squares (adjust if different)

# Maximum number of samples to collect
MAX_SAMPLES = 5

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Xverification"
LOG_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Logging setup
LOG_FILE = LOG_DIR / "verify_X_calibration.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_hand_eye_result():
    """Load the hand-eye calibration result (X matrix) from MoveIt calibration."""
    result_file = BASE_DIR / "hand_eye_calibration_moveit.yaml"

    if not result_file.exists():
        logger.error(f"Hand-eye result not found: {result_file}")
        return None

    with open(result_file, "r") as f:
        result = yaml.safe_load(f)

    X_matrix = np.array(result["X_cam_in_gripper_matrix"])
    logger.info(f"Loaded X (ee_gripper_link -> camera_color_optical_frame) from {result_file}")
    logger.info(f"Translation: x={X_matrix[0,3]:.4f}, y={X_matrix[1,3]:.4f}, z={X_matrix[2,3]:.4f} m")
    logger.info(f"Translation: x={X_matrix[0,3]*100:.2f}, y={X_matrix[1,3]*100:.2f}, z={X_matrix[2,3]*100:.2f} cm")

    return X_matrix


def get_realsense_intrinsics():
    """Get camera intrinsics directly from RealSense camera."""
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    try:
        profile = pipeline.start(config)
    except RuntimeError as e:
        logger.error(f"Failed to start RealSense pipeline: {e}")
        logger.error("Make sure the RealSense camera is connected.")
        return None, None, None

    # Get intrinsics
    color_profile = profile.get_stream(rs.stream.color)
    intrinsics = color_profile.as_video_stream_profile().get_intrinsics()

    # Build camera matrix
    camera_matrix = np.array([
        [intrinsics.fx, 0, intrinsics.ppx],
        [0, intrinsics.fy, intrinsics.ppy],
        [0, 0, 1]
    ], dtype=np.float32)

    dist_coeffs = np.array(intrinsics.coeffs, dtype=np.float32)

    logger.info(f"Camera intrinsics from RealSense:")
    logger.info(f"  fx={intrinsics.fx:.2f}, fy={intrinsics.fy:.2f}")
    logger.info(f"  cx={intrinsics.ppx:.2f}, cy={intrinsics.ppy:.2f}")

    return pipeline, camera_matrix, dist_coeffs


def detect_checkerboard(image, camera_matrix, dist_coeffs):
    """Detect checkerboard and get its pose in camera frame."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find checkerboard corners
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)

    if not ret:
        return False, None, None, None

    # Refine corners for sub-pixel accuracy
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Create 3D object points (checkerboard in its own frame, Z=0)
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Solve PnP to get checkerboard pose in camera frame
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

    if success:
        return True, corners, rvec, tvec

    return False, None, None, None


def get_robot_pose(driver):
    """Get current robot end-effector pose."""
    # Get joint positions
    joint_positions = list(driver.get_all_positions())

    # Get Cartesian pose [x, y, z, rx, ry, rz]
    cartesian = list(driver.get_cartesian_positions())

    # Convert angle-axis to rotation matrix
    angle_axis = np.array(cartesian[3:6])
    R_gripper2base, _ = cv2.Rodrigues(angle_axis)
    t_gripper2base = np.array(cartesian[0:3])

    # Build 4x4 transformation matrix
    T_gripper2base = np.eye(4)
    T_gripper2base[:3, :3] = R_gripper2base
    T_gripper2base[:3, 3] = t_gripper2base

    return {
        "joint_positions": joint_positions,
        "cartesian_xyz_rpy": cartesian,
        "T_gripper2base": T_gripper2base.tolist(),
    }


def save_pose_data(pose_id, image, robot_pose, rvec, tvec, camera_matrix, dist_coeffs):
    """Save all data for a single pose."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save image
    img_path = DATA_DIR / f"pose_{pose_id:03d}.png"
    cv2.imwrite(str(img_path), image)
    logger.info(f"  Image saved: {img_path.name}")

    # Convert rvec to rotation matrix
    R_target2cam, _ = cv2.Rodrigues(rvec)
    T_target2cam = np.eye(4)
    T_target2cam[:3, :3] = R_target2cam
    T_target2cam[:3, 3] = tvec.flatten()

    # Pose data
    pose_data = {
        "pose_id": pose_id,
        "timestamp": timestamp,
        "image_file": f"pose_{pose_id:03d}.png",

        # Robot data
        "robot": robot_pose,

        # Checkerboard detection data (in camera frame)
        "checkerboard": {
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "T_target2cam": T_target2cam.tolist(),
        },

        # Camera intrinsics used
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
    }

    return pose_data


def main():
    logger.info("=" * 70)
    logger.info("Hand-Eye Calibration (X) Verification Data Collection")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"Checkerboard: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} internal corners")
    logger.info(f"Square size: {SQUARE_SIZE * 1000:.1f} mm")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info(f"Will collect up to {MAX_SAMPLES} samples")
    logger.info("")
    logger.info("Instructions:")
    logger.info("  1. Robot will be in gravity compensation mode")
    logger.info("  2. Move robot BY HAND to view the checkerboard")
    logger.info("  3. When board is detected, robot locks position")
    logger.info("  4. Press 's' to SAVE current pose")
    logger.info("  5. After save, robot returns to home and waits for next pose")
    logger.info("  6. Press 'q' to QUIT (robot returns to home)")
    logger.info("=" * 70)
    logger.info("")

    # Load hand-eye calibration result
    X_cam2gripper = load_hand_eye_result()
    if X_cam2gripper is None:
        logger.error("Cannot proceed without hand-eye calibration result")
        return 1
    logger.info("")

    # Initialize camera and get intrinsics
    logger.info("Initializing RealSense camera...")
    pipeline, camera_matrix, dist_coeffs = get_realsense_intrinsics()
    if pipeline is None:
        logger.error("Cannot proceed without camera")
        return 1
    logger.info("Camera ready")
    logger.info("")

    # Connect to robot
    logger.info(f"Connecting to robot at {ARM_IP}...")
    driver = TrossenArmDriver()
    driver.configure(
        model=Model.wxai_v0,
        end_effector=StandardEndEffector.wxai_v0_follower,
        serv_ip=ARM_IP,
        clear_error=True,
        timeout=10.0,
    )
    num_joints = driver.get_num_joints()
    logger.info(f"Robot connected ({num_joints} joints)")
    logger.info("")

    # Move to home position first
    logger.info("Moving to home position...")
    driver.set_all_modes(Mode.position)
    home_position = [0.0] * num_joints
    driver.set_all_positions(home_position, goal_time=3.0)
    time.sleep(3.5)
    logger.info("At home position")
    logger.info("")

    # Enable gravity compensation
    logger.info("Enabling gravity compensation - you can now move the robot by hand")
    driver.set_all_modes(Mode.effort)
    driver.set_all_efforts([0.0] * num_joints)
    time.sleep(0.5)
    logger.info("")

    # Create window
    cv2.namedWindow("X Verification - Data Collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("X Verification - Data Collection", 800, 600)

    # Storage for collected data
    all_poses = []
    pose_count = 0
    robot_locked = False

    try:
        while True:
            # Get camera frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            display = image.copy()

            # Detect checkerboard
            detected, corners, rvec, tvec = detect_checkerboard(
                image, camera_matrix, dist_coeffs
            )

            # Status bar at top
            status_color = (0, 255, 0) if detected else (0, 0, 255)
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (40, 40, 40), -1)

            if detected:
                # Draw checkerboard corners and axes
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                # Lock robot when board detected
                if not robot_locked:
                    current_position = driver.get_all_positions()
                    driver.set_all_modes(Mode.position)
                    driver.set_all_positions(current_position, goal_time=0.0)
                    robot_locked = True
                    logger.info("Robot LOCKED - board detected")

                # Display distance to board
                distance = np.linalg.norm(tvec)
                cv2.putText(
                    display,
                    f"DETECTED | Distance: {distance:.3f}m | Press 's' to SAVE | Poses: {pose_count}/{MAX_SAMPLES}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
            else:
                # Unlock robot when board not detected
                if robot_locked:
                    driver.set_all_modes(Mode.effort)
                    driver.set_all_efforts([0.0] * num_joints)
                    robot_locked = False
                    logger.info("Robot UNLOCKED - move by hand")

                cv2.putText(
                    display,
                    f"NO BOARD - Move robot to see checkerboard | Poses: {pose_count}/{MAX_SAMPLES}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2,
                )

            # Instructions at bottom
            cv2.rectangle(display, (0, display.shape[0]-30), (display.shape[1], display.shape[0]), (40, 40, 40), -1)
            cv2.putText(
                display,
                "Keys: 's' = Save pose | 'q' = Quit",
                (10, display.shape[0] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
            )

            cv2.imshow("X Verification - Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            # Handle key presses
            if key == ord("s") and detected:
                pose_count += 1
                logger.info(f"Saving pose {pose_count}...")

                # Get robot pose
                robot_pose = get_robot_pose(driver)

                # Save data
                pose_data = save_pose_data(
                    pose_count, image, robot_pose, rvec, tvec,
                    camera_matrix, dist_coeffs
                )
                all_poses.append(pose_data)

                # Save JSON after each pose (incremental)
                output_data = {
                    "description": "X verification data for eye-in-hand calibration",
                    "calibration_type": "eye-in-hand",
                    "parent_frame": "ee_gripper_link",
                    "child_frame": "camera_color_optical_frame",
                    "checkerboard": {
                        "size": list(BOARD_SIZE),
                        "square_size_m": SQUARE_SIZE,
                    },
                    "X_cam_in_gripper": X_cam2gripper.tolist(),
                    "num_poses": pose_count,
                    "poses": all_poses,
                }

                json_path = DATA_DIR / "verification_data.json"
                with open(json_path, "w") as f:
                    json.dump(output_data, f, indent=2)

                logger.info(f"  Pose {pose_count}/{MAX_SAMPLES} saved successfully")
                logger.info(f"  Data saved to: {json_path}")
                logger.info("")

                # Check if we've collected all samples
                if pose_count >= MAX_SAMPLES:
                    logger.info(f"Collected all {MAX_SAMPLES} samples!")
                    break

                # Return to home position
                logger.info("Returning to home position...")
                robot_locked = False
                driver.set_all_modes(Mode.position)
                driver.set_all_positions(home_position, goal_time=3.0)
                time.sleep(3.5)
                logger.info("At home position")
                logger.info("")

                # Re-enable gravity compensation for next pose
                logger.info("Enabling gravity compensation - move robot by hand for next pose")
                driver.set_all_modes(Mode.effort)
                driver.set_all_efforts([0.0] * num_joints)
                time.sleep(0.5)
                logger.info("")

            elif key == ord("q") or key == 27:  # 'q' or ESC
                logger.info("Quit requested")
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # Return to home
        logger.info("")
        logger.info("Returning to home position...")
        driver.set_all_modes(Mode.position)
        driver.set_all_positions(home_position, goal_time=3.0)
        time.sleep(3.5)

        driver.cleanup()
        logger.info("Robot disconnected")
        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Data collection complete: {pose_count} poses saved")
        if pose_count > 0:
            logger.info(f"Data saved to: {DATA_DIR}")
        logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
