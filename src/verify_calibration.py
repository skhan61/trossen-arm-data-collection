#!/usr/bin/env python3
"""
Physical verification of hand-eye calibration.

Test: Place the chessboard, detect it with camera, transform to robot frame,
move robot to the board, see if it arrives at the correct position.
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
LOG_FILE = LOG_DIR / "verify_calibration.log"

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
logger.info("HAND-EYE CALIBRATION PHYSICAL VERIFICATION")
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

# Enable gravity compensation so user can move robot
logger.info("Enabling gravity compensation mode...")
driver.set_all_modes(Mode.effort)
driver.set_all_efforts([0.0] * driver.get_num_joints())
time.sleep(0.5)
logger.info("Gravity compensation enabled - you can move robot by hand")
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
logger.info("1. Chessboard is fixed on table")
logger.info("2. MOVE ROBOT BY HAND to position where camera sees board")
logger.info("3. When board is detected (green overlay), press 's'")
logger.info("4. Robot will lock and move gripper center to board center")
logger.info("5. Measure how accurate the position is")
logger.info("")
logger.info("Controls:")
logger.info("  's' - Lock robot and move to detected board position")
logger.info("  ESC - Exit")
logger.info("=" * 80)
logger.info("")

cv2.namedWindow("Verification", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Verification", 640, 480)

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

                # Get board position in camera frame
                R_board2cam, _ = cv2.Rodrigues(rvec)
                t_board2cam = tvec.flatten()

                cv2.putText(display, "BOARD DETECTED - Press 's' to test", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(display, f"Board distance: {np.linalg.norm(t_board2cam):.3f} m", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        else:
            cv2.putText(display, "No board detected", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        cv2.imshow("Verification", display)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('s') and ret and success:
            logger.info("=" * 80)
            logger.info("PERFORMING CALIBRATION TEST")
            logger.info("=" * 80)

            # Board position in camera frame
            logger.info(f"Board detected in camera frame:")
            logger.info(f"  Position: {t_board2cam}")
            logger.info(f"  Distance: {np.linalg.norm(t_board2cam):.4f} m")

            # Transform board position from camera frame to gripper frame
            # t_board_in_gripper = R_cam2gripper @ t_board_in_camera + t_cam2gripper
            t_board_in_gripper = R_cam2gripper @ t_board2cam + t_cam2gripper

            logger.info("")
            logger.info(f"Board position in gripper frame:")
            logger.info(f"  Position: {t_board_in_gripper}")

            # Get current gripper position
            current_cart = list(driver.get_cartesian_positions())
            current_pos = np.array(current_cart[:3])
            current_ori = np.array(current_cart[3:6])

            logger.info("")
            logger.info(f"Current gripper position:")
            logger.info(f"  Position: {current_pos}")

            # Calculate target position in robot base frame
            # We need: R_gripper2base @ t_board_in_gripper + t_gripper2base
            R_gripper2base, _ = cv2.Rodrigues(current_ori)
            t_board_in_base = R_gripper2base @ t_board_in_gripper + current_pos

            logger.info("")
            logger.info(f"Board position in robot base frame:")
            logger.info(f"  Position: {t_board_in_base}")

            logger.info("")
            logger.info("=" * 80)
            logger.info("MANUAL VERIFICATION:")
            logger.info("=" * 80)
            logger.info(f"The board center is at: X={t_board_in_base[0]:.3f}, Y={t_board_in_base[1]:.3f}, Z={t_board_in_base[2]:.3f}")
            logger.info(f"Current gripper is at: X={current_pos[0]:.3f}, Y={current_pos[1]:.3f}, Z={current_pos[2]:.3f}")
            logger.info("")
            logger.info("The board is {:.3f}m in front of current gripper position".format(
                np.linalg.norm(t_board_in_base - current_pos)))
            logger.info("")
            logger.info("To verify calibration:")
            logger.info("1. Manually move gripper by hand to the board center")
            logger.info("2. Check if the predicted position matches reality")
            logger.info("3. Measure the error distance")
            logger.info("")
            logger.info("If error < 2cm → GOOD calibration")
            logger.info("If error > 5cm → BAD calibration, needs redoing")
            logger.info("=" * 80)

        elif key == 27:  # ESC
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    driver.cleanup()
    logger.info("Cleanup complete")
