#!/usr/bin/env python3
"""
Verify hand-eye calibration accuracy using multiple robot poses.

Method:
1. Fix chessboard at known position on table
2. Move robot to multiple poses where camera sees board
3. For each pose:
   - Detect board in camera frame
   - Transform to base frame using calibration
   - Compare with actual fixed board position
4. Calculate reprojection error across all poses
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
LOG_FILE = LOG_DIR / "verify_calibration_multipose.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

ARM_IP = "192.168.1.99"
DATA_DIR = Path(__file__).parent.parent / "data"

# Chessboard parameters - SAME as camera calibration
BOARD_SIZE = (7, 4)
SQUARE_SIZE = 0.025

logger.info("=" * 80)
logger.info("HAND-EYE CALIBRATION MULTI-POSE VERIFICATION")
logger.info("=" * 80)
logger.info("")

# Load camera intrinsics
calib_file = DATA_DIR / "camera_calibration_data" / "camera_intrinsics.json"
with open(calib_file) as f:
    cam_calib = json.load(f)

CAMERA_MATRIX = np.array(cam_calib["camera_matrix"], dtype=np.float32)
DIST_COEFFS = np.array(cam_calib["dist_coeffs"], dtype=np.float32).flatten()

logger.info("Loaded camera intrinsics")
logger.info(f"  Camera matrix: {CAMERA_MATRIX.diagonal()}")
logger.info(f"  Reprojection error: {cam_calib['reprojection_error']:.4f} pixels")

# Load hand-eye calibration
calib_file = DATA_DIR / "hand_eye_calibration_data" / "hand_eye_calibration.json"
with open(calib_file) as f:
    hand_eye = json.load(f)

R_cam2gripper = np.array(hand_eye["R_cam2gripper"])
t_cam2gripper = np.array(hand_eye["t_cam2gripper"]).flatten()

logger.info("")
logger.info("Loaded hand-eye calibration")
logger.info(f"  Camera offset from gripper: {t_cam2gripper} m")
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
logger.info("Robot connected")
logger.info("")

# Start camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
logger.info("Camera started")
logger.info("")

logger.info("=" * 80)
logger.info("INSTRUCTIONS:")
logger.info("=" * 80)
logger.info("1. Fix chessboard at KNOWN position on table (don't move it!)")
logger.info("2. Enable gravity compensation - move robot by hand")
logger.info("3. Move to different poses where camera sees board")
logger.info("4. Press 's' to save each pose (collect 5+ poses)")
logger.info("5. Press 'c' to compute calibration error")
logger.info("6. ESC to exit")
logger.info("=" * 80)
logger.info("")

# Enable gravity compensation
logger.info("Enabling gravity compensation mode...")
driver.set_all_modes(Mode.effort)
driver.set_all_efforts([0.0] * driver.get_num_joints())
time.sleep(0.5)
logger.info("Gravity compensation enabled - you can move robot by hand")
logger.info("")

cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Verification", 640, 480)

# Store detected board positions in base frame
board_positions_in_base = []

try:
    while True:
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        image = np.asanyarray(color_frame.get_data())
        display = image.copy()

        # Detect chessboard
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

        if ret:
            # Refine corners
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

            # Get board pose in camera frame
            objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
            objp *= SQUARE_SIZE

            success, rvec, tvec = cv2.solvePnP(objp, corners, CAMERA_MATRIX, DIST_COEFFS)

            if success:
                # Draw detection
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                cv2.drawFrameAxes(display, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)

                t_board2cam = tvec.flatten()

                cv2.putText(display, f"BOARD DETECTED - Press 's' to save pose ({len(board_positions_in_base)} saved)",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"Board distance: {np.linalg.norm(t_board2cam):.3f} m",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                if len(board_positions_in_base) >= 5:
                    cv2.putText(display, "Press 'c' to compute error",
                               (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        else:
            cv2.putText(display, "No board detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Verification", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and ret and success:
            logger.info("=" * 80)
            logger.info(f"SAVING POSE {len(board_positions_in_base) + 1}")
            logger.info("=" * 80)

            # Board position in camera frame
            t_board2cam = tvec.flatten()
            logger.info(f"Board in camera frame: {t_board2cam}")

            # Transform to gripper frame
            t_board_in_gripper = R_cam2gripper @ t_board2cam + t_cam2gripper

            # Get current gripper pose
            current_cart = list(driver.get_cartesian_positions())
            current_pos = np.array(current_cart[:3])
            current_ori = np.array(current_cart[3:6])

            # Transform to base frame
            R_gripper2base, _ = cv2.Rodrigues(current_ori)
            t_board_in_base = R_gripper2base @ t_board_in_gripper + current_pos

            logger.info(f"Board in base frame: {t_board_in_base}")
            logger.info(f"Gripper position: {current_pos}")
            logger.info("")

            board_positions_in_base.append(t_board_in_base)

        elif key == ord('c') and len(board_positions_in_base) >= 5:
            logger.info("=" * 80)
            logger.info("COMPUTING CALIBRATION ERROR")
            logger.info("=" * 80)

            # Convert to numpy array
            positions = np.array(board_positions_in_base)

            # Calculate mean position (ground truth)
            mean_position = np.mean(positions, axis=0)

            logger.info(f"Number of poses: {len(positions)}")
            logger.info(f"Mean board position in base: {mean_position}")
            logger.info("")

            # Calculate errors
            errors = positions - mean_position
            error_magnitudes = np.linalg.norm(errors, axis=1)

            logger.info("Individual pose errors:")
            for i, (pos, err, mag) in enumerate(zip(positions, errors, error_magnitudes)):
                logger.info(f"  Pose {i+1}: {pos}")
                logger.info(f"    Error: {err}")
                logger.info(f"    Magnitude: {mag*1000:.2f} mm")
                logger.info("")

            mean_error = np.mean(error_magnitudes)
            std_error = np.std(error_magnitudes)
            max_error = np.max(error_magnitudes)

            logger.info("=" * 80)
            logger.info("CALIBRATION ACCURACY SUMMARY")
            logger.info("=" * 80)
            logger.info(f"Mean error: {mean_error*1000:.2f} mm")
            logger.info(f"Std deviation: {std_error*1000:.2f} mm")
            logger.info(f"Max error: {max_error*1000:.2f} mm")
            logger.info("")

            if mean_error < 0.005:  # < 5mm
                logger.info("✓ EXCELLENT calibration (< 5mm)")
            elif mean_error < 0.010:  # < 10mm
                logger.info("✓ GOOD calibration (< 10mm)")
            elif mean_error < 0.020:  # < 20mm
                logger.info("⚠ ACCEPTABLE calibration (< 20mm)")
            else:
                logger.info("✗ POOR calibration (≥ 20mm) - recalibration recommended")

            logger.info("=" * 80)

        elif key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    driver.cleanup()
    logger.info("Cleanup complete")
