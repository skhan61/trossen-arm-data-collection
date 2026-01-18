#!/usr/bin/env python3
"""
Automated hand-eye calibration using Approach 1.

APPROACH 1: Manual First Pose
1. You manually move robot to initial position where it sees chessboard
2. Press 's' when board is detected
3. Script calculates chessboard location and plans 15 diverse poses
4. Robot moves automatically through all poses
5. Data captured automatically at each pose

This solves the chicken-egg problem: we find chessboard location from first pose,
then calculate all other poses to ensure they all see the chessboard.
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
LOG_FILE = LOG_DIR / "automated_hand_eye_calibration.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {LOG_FILE}")

ARM_IP = "192.168.1.99"

# Chessboard parameters
BOARD_SIZE = (7, 3)  # (columns, rows) of internal corners
SQUARE_SIZE = 0.025  # 25mm squares

# Data directory
DATA_DIR = Path(__file__).parent.parent / "data" / "hand_eye_calibration_data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Number of poses to collect
NUM_POSES = 15

# Camera offset from gripper (approximate - will be refined by calibration)
# For RealSense D405 mounted on WidowX AI gripper
CAMERA_OFFSET = 0.08  # 8cm forward from gripper


def load_camera_intrinsics():
    """Load calibrated camera parameters from JSON file."""
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
    return camera_matrix, dist_coeffs


CAMERA_MATRIX, DIST_COEFFS = load_camera_intrinsics()


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


def generate_camera_viewpoints_around_board(board_position, board_normal, num_poses=15):
    """
    Generate camera positions that view the chessboard from different angles.

    Strategy: Generate positions on a hemisphere around the board, all pointing at it.

    Args:
        board_position: 3D position of chessboard center in base frame (x, y, z)
        board_normal: 3D normal vector of chessboard in base frame
        num_poses: number of viewing positions to generate

    Returns:
        List of (position, rotation) tuples representing camera poses in base frame
    """
    camera_poses = []

    # Distance from camera to board (meters)
    view_distance = 0.25  # 25cm from board

    # Normalize board normal
    board_normal = np.array(board_normal)
    board_normal = board_normal / np.linalg.norm(board_normal)

    # Generate viewpoints on a hemisphere
    for i in range(num_poses):
        # Spherical coordinates: vary elevation and azimuth
        # Elevation: 20° to 60° from board plane
        elevation = np.radians(20 + 40 * (i % 5) / 4)
        # Azimuth: full circle around board
        azimuth = 2 * np.pi * (i / max(num_poses-1, 1))

        # Create local coordinate frame for board
        # Z-axis = board normal (pointing away from board)
        z_axis = board_normal

        # X-axis = arbitrary perpendicular vector
        if abs(z_axis[2]) < 0.9:
            x_axis = np.cross(z_axis, [0, 0, 1])
        else:
            x_axis = np.cross(z_axis, [1, 0, 0])
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis = complete the frame
        y_axis = np.cross(z_axis, x_axis)

        # Camera position in local frame
        x = view_distance * np.sin(elevation) * np.cos(azimuth)
        y = view_distance * np.sin(elevation) * np.sin(azimuth)
        z = view_distance * np.cos(elevation)

        # Transform to base frame
        local_pos = x * x_axis + y * y_axis + z * z_axis
        cam_position = board_position + local_pos

        # Camera should point at board center (negative z direction of camera)
        cam_z = -local_pos / np.linalg.norm(local_pos)  # Point at board

        # Camera up direction (try to keep y-axis upward)
        cam_y = np.array([0, 0, -1])  # World down
        cam_x = np.cross(cam_y, cam_z)
        if np.linalg.norm(cam_x) < 0.1:  # If nearly parallel
            cam_x = np.array([1, 0, 0])
        cam_x = cam_x / np.linalg.norm(cam_x)
        cam_y = np.cross(cam_z, cam_x)

        # Build rotation matrix (camera frame in base frame)
        R_cam = np.column_stack([cam_x, cam_y, cam_z])

        camera_poses.append((cam_position, R_cam))

    return camera_poses




def is_joint_config_valid(joint_angles, joint_limits=None):
    """
    Check if joint configuration is within limits.

    Args:
        joint_angles: list of joint angles in radians
        joint_limits: optional list of (min, max) tuples for each joint

    Returns:
        True if configuration is valid
    """
    if joint_limits is None:
        # Actual hardware limits from robot (radians)
        # Add safety margin of 5% from each limit
        joint_limits = [
            (-3.05, 3.05),        # Joint 0: base rotation (±175°)
            (-1.83, 1.83),        # Joint 1: shoulder (±105°)
            (-1.57, 1.57),        # Joint 2: elbow (±90°)
            (-3.05, 3.05),        # Joint 3: forearm roll (±175°)
            (-1.74, 1.74),        # Joint 4: wrist angle (±100°)
            (-3.05, 3.05),        # Joint 5: wrist rotate (±175°)
            (-0.003, 0.042),      # Joint 6: gripper (VERY limited: -0.004 to 0.044 with margin)
        ]

    for i, angle in enumerate(joint_angles):
        if i < len(joint_limits):
            min_angle, max_angle = joint_limits[i]
            if not (min_angle <= angle <= max_angle):
                logger.debug(f"Joint {i} angle {np.degrees(angle):.1f}° outside limits [{np.degrees(min_angle):.1f}, {np.degrees(max_angle):.1f}]")
                return False

    return True


def move_to_joint_config(driver, joint_angles, goal_time=3.0):
    """
    Move robot to joint configuration in joint space.

    This avoids singularities by commanding joint angles directly,
    not through Cartesian interpolation.

    Args:
        driver: robot driver
        joint_angles: target joint configuration (list of angles)
        goal_time: time to reach configuration in seconds

    Returns:
        True if successful, False otherwise
    """
    try:
        logger.debug(f"Moving to joint config: {[np.degrees(a) for a in joint_angles]}")

        # Move in joint space using position mode
        driver.set_all_modes(Mode.position)
        driver.set_all_positions(joint_angles, goal_time=goal_time)

        # Wait for movement to complete
        import time
        time.sleep(goal_time + 0.5)

        logger.debug("Move completed")
        return True

    except Exception as e:
        logger.error(f"Failed to move to joint config: {e}")
        return False


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


def main():
    logger.info("=" * 60)
    logger.info("Automated Hand-Eye Calibration - Approach 1")
    logger.info("=" * 60)
    logger.info("")
    logger.info("STEP 1: Manual Initial Pose")
    logger.info("  - Move robot BY HAND so camera sees chessboard")
    logger.info("  - Press 's' when board is detected")
    logger.info("  - Script will find chessboard location")
    logger.info("")
    logger.info("STEP 2: Automatic Collection")
    logger.info(f"  - Script plans {NUM_POSES} diverse poses around chessboard")
    logger.info("  - Robot moves automatically to each pose")
    logger.info("  - Data captured automatically")
    logger.info("  - No more manual movement needed!")
    logger.info("")
    logger.info("Chessboard: 7x3 internal corners, 25mm squares")
    logger.info("=" * 60)
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

    # Get home position
    num_joints = driver.get_num_joints()
    initial_joint_pos = [0.0] * num_joints

    try:
        # Move to home
        logger.info("Moving to home position...")
        driver.set_all_modes(Mode.position)
        driver.set_all_positions(initial_joint_pos, goal_time=3.0)
        time.sleep(3.5)
        logger.info("At home position")

        # STEP 1: Manual first pose to find chessboard
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 1: Finding Chessboard Location")
        logger.info("=" * 60)
        logger.info("")
        logger.info("Enabling gravity compensation...")
        driver.set_all_modes(Mode.effort)
        driver.set_all_efforts([0.0] * num_joints)
        time.sleep(0.5)
        logger.info("Robot ready - move it BY HAND to see the chessboard")
        logger.info("Press 's' when board is detected to lock position")

        # Start camera
        pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)

        cv2.namedWindow("Automated Calibration - Step 1", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Automated Calibration - Step 1", 640, 480)

        board_found = False
        initial_joint_config = None
        robot_locked = False

        try:
            while not board_found:
                frames = pipeline.wait_for_frames()
                color_frame = frames.get_color_frame()
                if not color_frame:
                    continue

                image = np.asanyarray(color_frame.get_data())
                display = image.copy()

                # Detect chessboard
                detected, corners, rvec, tvec = detect_chessboard(image)

                if detected:
                    cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                    cv2.drawFrameAxes(display, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)

                    # LOCK robot when board detected
                    if not robot_locked:
                        current_position = driver.get_all_positions()
                        driver.set_all_modes(Mode.position)
                        driver.set_all_positions(current_position, goal_time=0.0)
                        robot_locked = True
                        logger.info("Robot locked - board detected")

                    cv2.putText(display, "BOARD DETECTED & LOCKED - Press 's' to continue", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    # UNLOCK robot when board not detected
                    if robot_locked:
                        driver.set_all_modes(Mode.effort)
                        driver.set_all_efforts([0.0] * num_joints)
                        robot_locked = False
                        logger.info("Robot unlocked - move by hand")

                    cv2.putText(display, "No board - move robot by hand", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                cv2.imshow("Automated Calibration - Step 1", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('s') and detected:
                    # Get current joint configuration - this is our reference configuration
                    logger.info("Saving reference joint configuration...")
                    initial_joint_config = list(driver.get_all_positions())

                    logger.info(f"Reference joint angles (degrees): {[np.degrees(a) for a in initial_joint_config]}")
                    logger.info("This configuration sees the chessboard - will generate variations around it")

                    board_found = True
                    break

                elif key == 27:  # ESC
                    logger.info("Cancelled by user")
                    cv2.destroyAllWindows()
                    return

        finally:
            pass  # Keep window open for automation phase

        if not board_found:
            logger.error("Failed to find chessboard")
            return

        # STEP 2: Generate joint space variations
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 2: Generating Joint Space Variations")
        logger.info("=" * 60)
        logger.info("")
        logger.info(f"Generating {NUM_POSES} joint configurations around reference...")
        logger.info("Variations: ±15° per joint (±0.5° for gripper)")

        all_joint_configs = generate_joint_variations(initial_joint_config, num_poses=NUM_POSES)

        logger.info(f"Generated {len(all_joint_configs)} joint configurations")

        # Filter valid configurations
        valid_configs = []
        for i, config in enumerate(all_joint_configs):
            if is_joint_config_valid(config):
                valid_configs.append(config)
                logger.debug(f"Config {i+1}: valid")
            else:
                logger.debug(f"Config {i+1}: outside joint limits, skipping")

        logger.info(f"Filtered to {len(valid_configs)} valid configurations")

        if len(valid_configs) < 10:
            logger.error(f"Only {len(valid_configs)} valid configurations - need at least 10")
            logger.error("Initial configuration may be too close to joint limits")
            return

        # STEP 3: Automatic data collection
        logger.info("")
        logger.info("=" * 60)
        logger.info("STEP 3: Automatic Data Collection")
        logger.info("=" * 60)
        logger.info("")
        logger.info(f"Robot will now move through {len(valid_configs)} joint configurations automatically")
        logger.info("Moving in JOINT SPACE - no singularity issues!")
        logger.info("This will take a few minutes - do not interfere!")
        logger.info("")

        # Storage
        R_gripper2base = []
        t_gripper2base = []
        R_target2cam = []
        t_target2cam = []

        # For each joint configuration
        for i, joint_config in enumerate(valid_configs):
            logger.info(f"Moving to configuration {i+1}/{len(valid_configs)}...")

            # Move to target joint configuration
            success = move_to_joint_config(driver, joint_config, goal_time=3.0)
            if not success:
                logger.warning(f"Failed to reach configuration {i+1}, skipping")
                continue

            # Wait for robot to settle
            time.sleep(0.5)

            # Capture image
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                logger.warning(f"No camera frame at pose {i+1}, skipping")
                continue

            image = np.asanyarray(color_frame.get_data())

            # Detect chessboard
            detected, corners, rvec, tvec = detect_chessboard(image)
            if not detected:
                logger.warning(f"Chessboard not detected at pose {i+1}, skipping")
                continue

            # Get robot pose
            cart_pos = list(driver.get_cartesian_positions())
            t_gripper = np.array(cart_pos[:3]).reshape(3, 1)
            angle_axis = np.array(cart_pos[3:6])
            R_gripper, _ = cv2.Rodrigues(angle_axis)
            R_target, _ = cv2.Rodrigues(rvec)

            # Check pose diversity - ensure this pose is different enough from existing poses
            MIN_ROTATION_DIFF = np.radians(5)  # Minimum 5° rotation difference
            pose_is_diverse = True
            if len(R_gripper2base) > 0:
                for prev_R in R_gripper2base:
                    # Compute relative rotation
                    R_rel = R_gripper @ prev_R.T
                    # Get rotation angle
                    trace = np.trace(R_rel)
                    angle = np.arccos(np.clip((trace - 1) / 2, -1, 1))
                    if angle < MIN_ROTATION_DIFF:
                        pose_is_diverse = False
                        break

            if not pose_is_diverse:
                logger.warning(f"Pose {i+1} too similar to previous pose (rotation < 5°), skipping")
                continue

            # Save data
            R_gripper2base.append(R_gripper)
            t_gripper2base.append(t_gripper)
            R_target2cam.append(R_target)
            t_target2cam.append(tvec)

            # Save image
            img_path = DATA_DIR / f"auto_pose_{len(R_gripper2base):03d}.png"
            cv2.imwrite(str(img_path), image)

            logger.info(f"✓ Pose {i+1} captured successfully")
            logger.info(f"  Total poses collected: {len(R_gripper2base)}")

            # Save incrementally
            save_calibration_data(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)

        # Pipeline stop
        pipeline.stop()
        cv2.destroyAllWindows()

        # Check if enough data
        if len(R_gripper2base) < 10:
            logger.error(f"Only collected {len(R_gripper2base)} poses, need at least 10")
            return

        # Compute calibration
        logger.info("")
        logger.info("=" * 60)
        logger.info("Computing Hand-Eye Calibration")
        logger.info("=" * 60)
        logger.info(f"Using {len(R_gripper2base)} poses...")

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

    finally:
        # Return to home
        logger.info("Returning to home position...")
        driver.set_all_modes(Mode.position)
        driver.set_all_positions(initial_joint_pos, goal_time=3.0)
        time.sleep(3.5)
        driver.cleanup()
        logger.info("Done")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise
