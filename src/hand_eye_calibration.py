#!/usr/bin/env python3
"""
Manual hand-eye calibration.

1. Manually move robot to different positions to see chessboard
2. Press 's' to save each pose (need 15 poses minimum)
3. Press 'q' when done to compute calibration
4. Script computes hand-eye transformation
"""

import cv2
import numpy as np
import json
import time
import logging
import pyrealsense2 as rs
from pathlib import Path
from trossen_arm import (
    TrossenArmDriver,
    Model,
    StandardEndEffector,
    Mode,
    InterpolationSpace,
)

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "hand_eye_calibration.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {LOG_FILE}")

ARM_IP = "192.168.1.99"

# Chessboard parameters for hand-eye calibration
BOARD_SIZE = (7, 3)  # (columns, rows) of inner corners - 7x3 board
SQUARE_SIZE = 0.025  # 25mm squares

# Old ArUco marker parameters (not used anymore)
# ARUCO_DICT = cv2.aruco.DICT_ARUCO_ORIGINAL
# MARKER_SIZE = 0.150

# WidowX AI Cartesian limits (from specifications)
CARTESIAN_LIMITS = {
    "x": (-0.769, 0.769),  # max reach 0.769m
    "y": (-0.769, 0.769),
    "z": (0.0, 0.769),  # height from base
}

# Orientation limits (angle-axis magnitude in radians)
# Based on wrist joint limits: pitch ±90°, yaw ±180°
MAX_ORIENTATION_MAGNITUDE = np.pi  # ~3.14 radians (~180 degrees)

# Pose diversity thresholds
MIN_POSITION_DISTANCE = 0.05  # meters - minimum distance between poses
MIN_ROTATION_ANGLE = 15.0  # degrees - minimum rotation difference

# Data directory - separate folder for hand-eye calibration
DATA_DIR = Path(__file__).parent.parent / "data" / "hand_eye_calibration_data"
DATA_DIR.mkdir(exist_ok=True, parents=True)


# Load calibrated camera intrinsics
def load_camera_intrinsics():
    """Load calibrated camera parameters from JSON file."""
    # Camera intrinsics are in the camera calibration folder
    calib_file = Path(__file__).parent.parent / "data" / "camera_calibration_data" / "camera_intrinsics.json"

    if not calib_file.exists():
        logger.error(f"Camera calibration not found at {calib_file}")
        logger.error("Please run: python src/calibrate_camera.py first")
        exit(1)

    with open(calib_file, "r") as f:
        calib = json.load(f)

    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32).flatten()

    logger.info(f"Camera intrinsics loaded from {calib_file}")
    logger.info(
        f"  Focal length: fx={camera_matrix[0,0]:.2f}, fy={camera_matrix[1,1]:.2f}"
    )
    logger.info(
        f"  Principal point: cx={camera_matrix[0,2]:.2f}, cy={camera_matrix[1,2]:.2f}"
    )
    logger.info(f"  Reprojection error: {calib['reprojection_error']:.4f} pixels")
    return camera_matrix, dist_coeffs


CAMERA_MATRIX, DIST_COEFFS = load_camera_intrinsics()


def is_pose_within_limits(cart_pos):
    """Check if Cartesian position and orientation are within arm limits."""
    x, y, z = cart_pos[0], cart_pos[1], cart_pos[2]

    # Check position limits
    if not (CARTESIAN_LIMITS["x"][0] <= x <= CARTESIAN_LIMITS["x"][1]):
        logger.debug(f"X position {x:.3f} outside limits {CARTESIAN_LIMITS['x']}")
        return False
    if not (CARTESIAN_LIMITS["y"][0] <= y <= CARTESIAN_LIMITS["y"][1]):
        logger.debug(f"Y position {y:.3f} outside limits {CARTESIAN_LIMITS['y']}")
        return False
    if not (CARTESIAN_LIMITS["z"][0] <= z <= CARTESIAN_LIMITS["z"][1]):
        logger.debug(f"Z position {z:.3f} outside limits {CARTESIAN_LIMITS['z']}")
        return False

    # Check total reach
    reach = np.sqrt(x**2 + y**2 + z**2)
    if reach > 0.769:
        logger.debug(f"Total reach {reach:.3f}m exceeds max 0.769m")
        return False

    # Check orientation limits (angle-axis representation)
    if len(cart_pos) >= 6:
        angle_axis = np.array(cart_pos[3:6])
        orientation_magnitude = np.linalg.norm(angle_axis)
        if orientation_magnitude > MAX_ORIENTATION_MAGNITUDE:
            logger.debug(f"Orientation magnitude {orientation_magnitude:.3f} exceeds max {MAX_ORIENTATION_MAGNITUDE:.3f}")
            return False

    return True


def is_pose_diverse(new_t, new_R, existing_t_list, existing_R_list):
    """Check if new pose is sufficiently different from existing poses.

    A pose is considered diverse if BOTH position AND rotation are sufficiently different
    from ALL existing poses. We want diverse views for good calibration.
    """
    if len(existing_t_list) == 0:
        return True

    for i in range(len(existing_t_list)):
        # Check position distance
        pos_dist = np.linalg.norm(new_t - existing_t_list[i])

        # Check rotation difference (angle between rotations)
        R_diff = new_R @ existing_R_list[i].T
        trace = np.trace(R_diff)
        angle_diff = np.arccos(np.clip((trace - 1) / 2, -1, 1))
        angle_diff_deg = np.degrees(angle_diff)

        # Reject if this pose is too similar to any existing pose
        # We require BOTH sufficient position separation AND sufficient rotation change
        if pos_dist < MIN_POSITION_DISTANCE or angle_diff_deg < MIN_ROTATION_ANGLE:
            logger.debug(
                f"Pose rejected: pos_dist={pos_dist:.3f}m (min {MIN_POSITION_DISTANCE}m), "
                f"angle_diff={angle_diff_deg:.1f}° (min {MIN_ROTATION_ANGLE}°)"
            )
            return False

    return True


def detect_chessboard(image):
    """Detect chessboard and return corners, rvec, tvec."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

    if not ret:
        return False, None, None, None

    # Refine corners
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    # Prepare object points
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : BOARD_SIZE[0], 0 : BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    # Solve PnP
    success, rvec, tvec = cv2.solvePnP(objp, corners, CAMERA_MATRIX, DIST_COEFFS)

    if success:
        return True, corners, rvec, tvec

    return False, None, None, None


def save_calibration_data(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam):
    """Save raw calibration data to JSON."""
    data = {
        "R_gripper2base": [R.tolist() for R in R_gripper2base],
        "t_gripper2base": [t.tolist() for t in t_gripper2base],
        "R_target2cam": [R.tolist() for R in R_target2cam],
        "t_target2cam": [t.tolist() for t in t_target2cam],
        "num_poses": len(R_gripper2base),
    }

    output_file = DATA_DIR / "hand_eye_data.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"Saved calibration data to {output_file}")


def return_to_initial_position(driver, initial_joint_pos):
    """Safely return robot to initial position."""
    try:
        logger.info("Returning robot to initial position...")
        driver.set_all_modes(Mode.position)
        # Use longer goal_time to avoid velocity limit errors
        driver.set_all_positions(initial_joint_pos, goal_time=6.0)
        time.sleep(6.5)
        logger.info("Robot returned to initial position successfully")
    except Exception as e:
        logger.error(f"Failed to return to initial position: {type(e).__name__}: {e}")
        logger.error("Attempting retry with slower movement...")
        try:
            time.sleep(0.5)
            driver.set_all_modes(Mode.position)
            driver.set_all_positions(initial_joint_pos, goal_time=10.0)
            time.sleep(10.5)
            logger.info("Robot returned to initial position on retry")
        except Exception as e2:
            logger.error(f"Retry also failed: {type(e2).__name__}: {e2}")
            logger.error("Please manually return robot to safe position!")


def main():
    logger.info("=" * 60)
    logger.info("Manual Hand-Eye Calibration")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Instructions:")
    logger.info("1. Move robot BY HAND to different positions to see chessboard")
    logger.info("2. Press 's' when board is detected to save that pose")
    logger.info("3. Collect 15+ diverse poses (different positions and angles)")
    logger.info("4. Press 'q' when done to compute calibration")
    logger.info("")
    logger.info("Chessboard: 7x3 internal corners, 25mm squares")
    logger.info("")
    logger.info("Tips:")
    logger.info("- Move board to different distances from camera")
    logger.info("- Tilt board at different angles")
    logger.info("- Ensure poses are diverse (≥5cm apart, ≥15° rotation)")
    logger.info("")
    logger.info("Controls:")
    logger.info("  's' - Save current pose")
    logger.info("  'q' - Finish and compute calibration")
    logger.info("  ESC - Cancel")
    logger.info("=" * 60)
    logger.info("")

    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)

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

    # Set initial position to zero/home position
    num_joints = driver.get_num_joints()
    initial_joint_pos = [0.0] * num_joints
    logger.info(f"Initial (home) position: {initial_joint_pos}")

    # CRITICAL: Wrap everything in try-finally to ensure robot returns home
    try:
        # Move to home position first
        logger.info("Moving to home position...")
        driver.set_all_modes(Mode.position)
        driver.set_all_positions(initial_joint_pos, goal_time=3.0)
        time.sleep(3.5)
        logger.info("At home position")

        # Enable gravity compensation
        logger.info("Enabling gravity compensation mode...")
        driver.set_all_modes(Mode.effort)
        driver.set_all_efforts([0.0] * driver.get_num_joints())
        time.sleep(0.5)

        logger.info("Robot ready - move it by hand to see the chessboard")

        # Start camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        # Storage
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        # Check if existing data exists and load it
        existing_data_file = DATA_DIR / "hand_eye_data.json"
        if existing_data_file.exists():
            logger.info(f"Found existing calibration data at {existing_data_file}")
            try:
                with open(existing_data_file, "r") as f:
                    existing_data = json.load(f)

                # Load existing poses
                R_gripper2base = [np.array(R) for R in existing_data["R_gripper2base"]]
                t_gripper2base = [np.array(t) for t in existing_data["t_gripper2base"]]
                R_target2cam = [np.array(R) for R in existing_data["R_target2cam"]]
                t_target2cam = [np.array(t) for t in existing_data["t_target2cam"]]

                logger.info(f"Loaded {len(R_gripper2base)} existing poses")
                logger.info(f"Will continue from pose #{len(R_gripper2base) + 1}")
            except Exception as e:
                logger.error(f"Failed to load existing data: {type(e).__name__}: {e}")
                logger.error("Starting fresh collection")
                R_gripper2base = []
                t_gripper2base = []
                R_target2cam = []
                t_target2cam = []
        else:
            logger.info("No existing data found, starting fresh collection")

        cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Hand-Eye Calibration", 640, 480)

        # Manual collection
        logger.info("Starting manual collection...")
        logger.info("Move robot by hand, press 's' to save poses, 'q' when done")

        robot_locked = False

        try:
            while True:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())
                display = image.copy()

                # Detect chessboard
                detected, corners, rvec, tvec = detect_chessboard(image)

                if detected:
                    # Draw chessboard corners
                    cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                    # Draw coordinate axes on board
                    cv2.drawFrameAxes(display, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)

                    # LOCK robot when board detected
                    if not robot_locked:
                        current_position = driver.get_all_positions()
                        driver.set_all_modes(Mode.position)
                        driver.set_all_positions(current_position, goal_time=0.0)
                        robot_locked = True
                        logger.debug("Robot locked")

                    cv2.putText(
                        display,
                        "BOARD DETECTED & LOCKED - Press 's' to save",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                else:
                    # UNLOCK robot when board not detected
                    if robot_locked:
                        driver.set_all_modes(Mode.effort)
                        driver.set_all_efforts([0.0] * driver.get_num_joints())
                        robot_locked = False
                        logger.debug("Robot unlocked")

                    cv2.putText(
                        display,
                        "No board - move robot by hand",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )

                # Show count
                cv2.putText(display, f"Poses collected: {len(R_gripper2base)}/15", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

                cv2.imshow("Hand-Eye Calibration", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord("s") and detected:
                    # Get robot pose
                    cart_pos = list(driver.get_cartesian_positions())

                    # Compute transforms
                    t_gripper = np.array(cart_pos[:3]).reshape(3, 1)
                    angle_axis = np.array(cart_pos[3:6])
                    R_gripper, _ = cv2.Rodrigues(angle_axis)
                    R_target, _ = cv2.Rodrigues(rvec)

                    # Check pose diversity
                    if not is_pose_diverse(
                        t_gripper, R_gripper, t_gripper2base, R_gripper2base
                    ):
                        logger.warning("Pose not diverse enough - rejected")
                        logger.warning(f"  Position: {cart_pos[:3]}")
                        logger.warning(f"  Orientation: {cart_pos[3:6]}")
                        logger.warning("  Move robot to a more different position/angle")

                        # Return to initial position
                        return_to_initial_position(driver, initial_joint_pos)

                        # Re-enable gravity compensation
                        logger.info("Re-enabling gravity compensation...")
                        driver.set_all_modes(Mode.effort)
                        driver.set_all_efforts([0.0] * driver.get_num_joints())
                        robot_locked = False
                        logger.info("Robot ready - move to a more different position")
                        continue

                    # Save calibration data to lists
                    R_gripper2base.append(R_gripper)
                    t_gripper2base.append(t_gripper)
                    R_target2cam.append(R_target)
                    t_target2cam.append(tvec)

                    # Save JSON data first (most important)
                    save_calibration_data(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

                    # Save image second
                    img_path = DATA_DIR / f"hand_eye_pose_{len(R_gripper2base):03d}.png"
                    cv2.imwrite(str(img_path), image)

                    logger.info(f"✓ Pose #{len(R_gripper2base)} saved")
                    logger.info(f"  Robot position (x,y,z): {cart_pos[:3]}")
                    logger.info(f"  Robot orientation: {cart_pos[3:6]}")
                    logger.info(f"  Image: {img_path.name}")

                    # Return to initial position
                    return_to_initial_position(driver, initial_joint_pos)

                    # Re-enable gravity compensation so user can move robot again
                    logger.info("Re-enabling gravity compensation...")
                    driver.set_all_modes(Mode.effort)
                    driver.set_all_efforts([0.0] * driver.get_num_joints())
                    robot_locked = False
                    logger.info("Robot ready - move to next position")

                    # Check if we have enough poses to finish
                    if len(R_gripper2base) >= 15:
                        logger.info(f"Collected {len(R_gripper2base)} poses - enough for calibration!")
                        logger.info("Automatically finishing collection...")
                        break

                elif key == ord("q"):
                    logger.info("Finishing collection...")
                    break

                elif key == 27:  # ESC
                    logger.info("Cancelled by user")
                    return

        finally:
            pipeline.stop()
            cv2.destroyAllWindows()
            return_to_initial_position(driver, initial_joint_pos)

        # Check if enough data
        if len(R_gripper2base) < 15:
            logger.error(f"Not enough poses: {len(R_gripper2base)} (need 15)")
            return

        # Save raw data
        save_calibration_data(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

        # STEP 3: Compute calibration
        logger.info(
            f"STEP 3: Computing hand-eye calibration from {len(R_gripper2base)} poses..."
        )

        try:
            R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
                R_gripper2base,
                t_gripper2base,
                R_target2cam,
                t_target2cam,
                method=cv2.CALIB_HAND_EYE_TSAI,
            )

            logger.info("=" * 60)
            logger.info("Hand-Eye Calibration Result")
            logger.info("=" * 60)
            logger.info("\nRotation matrix (camera to gripper):")
            logger.info(f"\n{R_cam2gripper}")
            logger.info("\nTranslation vector (camera to gripper) [meters]:")
            logger.info(f"{t_cam2gripper.flatten()}")
            logger.info("=" * 60)

            # Save result
            result = {
                "R_cam2gripper": R_cam2gripper.tolist(),
                "t_cam2gripper": t_cam2gripper.tolist(),
                "num_poses": len(R_gripper2base),
            }

            output_file = DATA_DIR / "hand_eye_calibration.json"
            with open(output_file, "w") as f:
                json.dump(result, f, indent=2)

            logger.info(f"Calibration saved to {output_file}")
            logger.info("Done!")

        except Exception as e:
            logger.error(f"Failed to compute hand-eye calibration: {type(e).__name__}: {e}")
            logger.error(f"Number of poses collected: {len(R_gripper2base)}")
            return

    finally:
        # CRITICAL: Always return robot to home and cleanup, no matter what
        logger.info("=" * 60)
        logger.info("Finalizing - returning robot to home position")
        logger.info("=" * 60)
        return_to_initial_position(driver, initial_joint_pos)
        driver.cleanup()
        logger.info("Driver cleanup completed")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error("=" * 60)
        logger.error("FATAL ERROR - Script crashed")
        logger.error("=" * 60)
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception message: {e}")
        logger.error("=" * 60)
        import traceback
        logger.error("Full traceback:")
        logger.error(traceback.format_exc())
        logger.error("=" * 60)
        raise
