#!/usr/bin/env python3
"""
Simple calibration data collection.

1. Shows live camera feed with chessboard detection
2. Press 's' to save current pose (image + robot data + chessboard data)
3. Press 'q' to finish and return robot to home
4. All data saved to JSON file
"""

import cv2
import numpy as np
import json
import time
import logging
import pyrealsense2 as rs
from pathlib import Path
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "collect_data.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ARM_IP = "192.168.1.99"
BOARD_SIZE = (7, 3)  # (columns, rows) - 7x3 chessboard
SQUARE_SIZE = 0.025  # 25mm squares

# Data directory - new folder for collected data
DATA_DIR = Path(__file__).parent.parent / "data" / "collected_calibration_data"
DATA_DIR.mkdir(exist_ok=True, parents=True)


def load_camera_intrinsics():
    """Load camera calibration from JSON."""
    calib_file = Path(__file__).parent.parent / "data" / "camera_calibration_data" / "camera_intrinsics.json"

    if not calib_file.exists():
        logger.error(f"Camera calibration not found: {calib_file}")
        logger.error("Run: python src/calibrate_camera.py first")
        exit(1)

    with open(calib_file, "r") as f:
        calib = json.load(f)

    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32).flatten()

    logger.info(f"Camera intrinsics loaded from {calib_file}")
    return camera_matrix, dist_coeffs


def detect_chessboard(image, camera_matrix, dist_coeffs):
    """Detect chessboard and get its pose."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    if not ret:
        return False, None, None, None

    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Create 3D object points (chessboard in its own frame)
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Solve PnP to get chessboard pose in camera frame
    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

    if success:
        return True, corners, rvec, tvec

    return False, None, None, None


def save_all_data(all_data, output_file):
    """Save all collected data to JSON."""
    with open(output_file, "w") as f:
        json.dump(all_data, f, indent=2)
    logger.info(f"Data saved to {output_file}")


def main():
    logger.info("=" * 60)
    logger.info("Simple Calibration Data Collection")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Instructions:")
    logger.info("  1. Move robot by hand to see chessboard")
    logger.info("  2. Press 's' to save and quit")
    logger.info("")
    logger.info(f"Chessboard: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} internal corners, {SQUARE_SIZE*1000}mm squares")
    logger.info("=" * 60)
    logger.info("")

    # Load camera intrinsics
    camera_matrix, dist_coeffs = load_camera_intrinsics()

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
    logger.info("Robot connected")
    num_joints = driver.get_num_joints()

    # Record initial robot state BEFORE moving (same format as poses)
    initial_joint_positions = list(driver.get_all_positions())
    initial_cartesian_pos = list(driver.get_cartesian_positions())

    # Convert initial orientation to rotation matrix
    initial_angle_axis = np.array(initial_cartesian_pos[3:6])
    initial_R_gripper2base, _ = cv2.Rodrigues(initial_angle_axis)

    # Create initial state with same structure as pose data
    initial_state = {
        "pose_id": 0,  # 0 indicates initial state
        "timestamp": time.time(),

        # From driver.get_all_positions()
        "get_all_positions": initial_joint_positions,

        # From driver.get_cartesian_positions()
        "get_cartesian_positions": initial_cartesian_pos,

        # From cv2.solvePnP() - None for initial state
        "rvec": None,
        "tvec": None,
    }

    logger.info("Initial robot state recorded")
    logger.info(f"  Joint positions: {[f'{np.degrees(j):.2f}°' for j in initial_joint_positions]}")
    logger.info(f"  Cartesian: x={initial_cartesian_pos[0]:.3f}, y={initial_cartesian_pos[1]:.3f}, z={initial_cartesian_pos[2]:.3f}")
    logger.info("")

    # Move to home position
    logger.info("Moving to home position...")
    driver.set_all_modes(Mode.position)
    home_position = [0.0] * num_joints
    driver.set_all_positions(home_position, goal_time=3.0)
    time.sleep(3.5)
    logger.info("At home position")
    logger.info("")

    # Enable gravity compensation for manual movement
    logger.info("Enabling gravity compensation...")
    driver.set_all_modes(Mode.effort)
    driver.set_all_efforts([0.0] * num_joints)
    time.sleep(0.5)
    logger.info("Robot ready - move it BY HAND to see the chessboard")
    logger.info("Press 's' when board is detected to save and quit")
    logger.info("")

    # Start camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Data Collection", 640, 480)

    # Storage for all collected data
    all_data = {
        "initial_robot_state": initial_state,
        "poses": [],
        "num_poses": 0
    }

    pose_count = 0
    robot_locked = False  # Track if robot is locked in position mode

    try:
        while True:
            # Get camera frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            display = image.copy()

            # Detect chessboard
            detected, corners, rvec, tvec = detect_chessboard(image, camera_matrix, dist_coeffs)

            if detected:
                # Draw detection
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                # LOCK robot when board detected
                if not robot_locked:
                    current_position = driver.get_all_positions()
                    driver.set_all_modes(Mode.position)
                    driver.set_all_positions(current_position, goal_time=0.0)
                    robot_locked = True
                    logger.info("Robot locked - board detected")

                cv2.putText(display, "BOARD DETECTED & LOCKED - Press 's' to save and quit",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                # UNLOCK robot when board not detected
                if robot_locked:
                    driver.set_all_modes(Mode.effort)
                    driver.set_all_efforts([0.0] * num_joints)
                    robot_locked = False
                    logger.info("Robot unlocked - move by hand")

                cv2.putText(display, "No board - Move robot to see chessboard",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            cv2.imshow("Data Collection", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and detected:
                # Get ALL robot data
                joint_positions = list(driver.get_all_positions())
                cartesian_pos = list(driver.get_cartesian_positions())

                # Save pose
                pose_count += 1
                logger.info(f"Saving pose {pose_count}...")

                # Convert chessboard pose to rotation matrix
                R_target2cam, _ = cv2.Rodrigues(rvec)

                # Convert robot orientation to rotation matrix
                angle_axis = np.array(cartesian_pos[3:6])
                R_gripper2base, _ = cv2.Rodrigues(angle_axis)

                # Save data for this pose
                pose_data = {
                    "pose_id": pose_count,
                    "timestamp": time.time(),

                    # From driver.get_all_positions()
                    "get_all_positions": joint_positions,

                    # From driver.get_cartesian_positions()
                    "get_cartesian_positions": cartesian_pos,

                    # From cv2.solvePnP()
                    "rvec": rvec.tolist(),
                    "tvec": tvec.tolist(),
                }

                all_data["poses"].append(pose_data)
                all_data["num_poses"] = pose_count

                # Save image
                img_path = DATA_DIR / f"pose_{pose_count:03d}.png"
                cv2.imwrite(str(img_path), image)
                logger.info(f"  Image saved: {img_path}")

                # Save JSON after each pose (incremental save)
                json_path = DATA_DIR / "calibration_data.json"
                save_all_data(all_data, json_path)

                logger.info(f"✓ Pose {pose_count} saved successfully")
                logger.info("")

                # After saving pose, quit automatically
                logger.info("Data saved - quitting...")
                break

            elif key == 27:  # ESC
                logger.info("Cancelled")
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
        logger.info("Done")
        logger.info("")
        logger.info(f"Total poses collected: {pose_count}")
        if pose_count > 0:
            logger.info(f"Data saved to: {DATA_DIR / 'calibration_data.json'}")


if __name__ == "__main__":
    main()
