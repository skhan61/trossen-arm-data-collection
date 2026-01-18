#!/usr/bin/env python3
"""
Semi-automatic hand-eye calibration.

1. Manually move robot to find chessboard and press 's'
2. Robot automatically moves to hemisphere poses
3. Collects diverse calibration data
4. Computes hand-eye calibration
"""

import cv2
import numpy as np
import json
import time
import pyrealsense2 as rs
from pathlib import Path
from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

ARM_IP = "192.168.1.99"

# Chessboard parameters - DO NOT CHANGE, these are for hand-eye calibration
BOARD_SIZE = (6, 8)  # Internal corners (5x8)
SQUARE_SIZE = 0.025  # 25mm squares

# Load calibrated camera intrinsics
CALIB_FILE = Path("src/camera_calibration_data/camera_intrinsics.json")

def load_camera_intrinsics():
    """Load calibrated camera parameters."""
    if not CALIB_FILE.exists():
        print(f"WARNING: Camera calibration file not found: {CALIB_FILE}")
        print("Using approximate intrinsics. Run calibrate_camera.py for better results.")
        return (
            np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32),
            np.zeros(5, dtype=np.float32)
        )

    with open(CALIB_FILE, 'r') as f:
        calib = json.load(f)

    camera_matrix = np.array(calib['camera_matrix'], dtype=np.float32)
    dist_coeffs = np.array(calib['dist_coeffs'], dtype=np.float32).flatten()

    print(f"✓ Loaded calibrated camera intrinsics (error: {calib['reprojection_error']:.4f} pixels)")
    return camera_matrix, dist_coeffs

CAMERA_MATRIX, DIST_COEFFS = load_camera_intrinsics()

# Data directory
DATA_DIR = Path("hand_eye_calibration_data")


def generate_hemisphere_offsets(center_pos, num_poses=15, radius=0.15):
    """Generate Cartesian offsets in a hemisphere around center position."""
    offsets = []

    # Golden spiral on hemisphere
    phi = (1 + np.sqrt(5)) / 2  # Golden ratio
    for i in range(num_poses):
        # Spherical coordinates
        theta = 2 * np.pi * i / phi  # Azimuth
        z_norm = 1 - (i / num_poses)  # Normalized z from 1 to 0
        elevation = np.arccos(z_norm)  # Only upper hemisphere

        # Convert to Cartesian offset
        x_offset = radius * np.sin(elevation) * np.cos(theta)
        y_offset = radius * np.sin(elevation) * np.sin(theta)
        z_offset = radius * np.cos(elevation)

        offsets.append([x_offset, y_offset, z_offset])

    return offsets


def main():
    print("=" * 60)
    print("Semi-Automatic Hand-Eye Calibration")
    print("=" * 60)
    print()
    print("STEP 1: Manual Initial Pose")
    print("  - Move robot by hand to see the chessboard")
    print("  - Press 's' when board is detected")
    print()
    print("STEP 2: Automatic Hemisphere Sampling")
    print("  - Robot will move to 15 diverse poses")
    print("  - Data collected automatically when board visible")
    print()
    print("STEP 3: Calibration Computation")
    print("  - Automatic after data collection")
    print()
    print("Controls:")
    print("  's' - Start automatic collection (after board detected)")
    print("  ESC - Cancel")
    print("=" * 60)
    print()

    # Create data directory
    DATA_DIR.mkdir(exist_ok=True)

    # Connect to robot
    print(f"Connecting to robot at {ARM_IP}...")
    driver = TrossenArmDriver()
    driver.configure(
        model=Model.wxai_v0,
        end_effector=StandardEndEffector.wxai_v0_follower,
        serv_ip=ARM_IP,
        clear_error=True,
        timeout=10.0
    )
    print("Robot connected!")

    # Enable gravity compensation so you can move the robot by hand
    print("Enabling gravity compensation mode...")
    driver.set_all_modes(Mode.effort)
    driver.set_all_efforts([0.0] * driver.get_num_joints())
    time.sleep(0.5)

    print()
    print("Robot is in gravity compensation mode.")
    print("You can now move it by hand to different positions.")
    print("Press 's' when board is detected and robot is in a good pose.")
    print()

    # Start camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Storage for calibration data
    R_gripper2base = []
    t_gripper2base = []
    R_target2cam = []
    t_target2cam = []

    cv2.namedWindow("Hand-Eye Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Hand-Eye Calibration", 1280, 960)  # Make window bigger

    robot_locked = False  # Track if robot is locked
    current_position = None  # Store current joint positions

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
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

            board_detected = False
            rvec, tvec = None, None

            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                # Object points
                objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
                objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
                objp *= SQUARE_SIZE

                # Solve PnP to get board pose relative to camera
                success, rvec, tvec = cv2.solvePnP(objp, corners, CAMERA_MATRIX, DIST_COEFFS)

                if success:
                    board_detected = True
                    # Draw detected corners
                    cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                    # Draw coordinate axes on board
                    cv2.drawFrameAxes(display, CAMERA_MATRIX, DIST_COEFFS, rvec, tvec, 0.05)

                    # LOCK robot when board is detected
                    if not robot_locked:
                        current_position = driver.get_all_positions()
                        driver.set_all_modes(Mode.position)
                        driver.set_all_positions(current_position, goal_time=0.0)
                        robot_locked = True

                    cv2.putText(display, "BOARD DETECTED & LOCKED - Press 's' to save", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if not board_detected:
                # UNLOCK robot when board is not detected
                if robot_locked:
                    driver.set_all_modes(Mode.effort)
                    driver.set_all_efforts([0.0] * driver.get_num_joints())
                    robot_locked = False

                cv2.putText(display, "No board detected - move robot", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show number of poses collected
            cv2.putText(display, f"Poses collected: {len(R_gripper2base)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand-Eye Calibration", display)

            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF

            if key == ord('s') and board_detected:
                # Get current robot pose (Cartesian)
                cart_pos = driver.get_cartesian_positions()

                # Translation (x, y, z in meters) - first 3 elements
                t_gripper = np.array(cart_pos[:3]).reshape(3, 1)

                # Rotation (angle-axis representation) - last 3 elements
                angle_axis = np.array(cart_pos[3:6])
                R_gripper, _ = cv2.Rodrigues(angle_axis)

                # Board pose relative to camera
                R_target, _ = cv2.Rodrigues(rvec)

                # Store the calibration pair
                R_gripper2base.append(R_gripper)
                t_gripper2base.append(t_gripper)
                R_target2cam.append(R_target)
                t_target2cam.append(tvec)

                print(f"✓ Saved pose #{len(R_gripper2base)}")

            elif key == ord('q'):
                print("\nFinishing calibration...")
                break

            elif key == 27:  # ESC
                print("\nCancelled by user")
                pipeline.stop()
                cv2.destroyAllWindows()
                driver.cleanup()
                return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

        # Return robot to home position
        print("\nReturning robot to home position...")
        driver.set_all_modes(Mode.position)
        home_position = [0.0] * driver.get_num_joints()
        driver.set_all_positions(home_position, goal_time=3.0)
        time.sleep(3.5)

    # Check if we have enough poses
    if len(R_gripper2base) < 10:
        print(f"\nERROR: Need at least 10 poses, only collected {len(R_gripper2base)}")
        print("Please run again and collect more poses.")
        driver.cleanup()
        return

    # Compute hand-eye calibration
    print(f"\nComputing hand-eye calibration from {len(R_gripper2base)} poses...")
    print("This may take a moment...")

    R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(
        R_gripper2base,
        t_gripper2base,
        R_target2cam,
        t_target2cam,
        method=cv2.CALIB_HAND_EYE_TSAI
    )

    # Display results
    print("\n" + "=" * 60)
    print("Hand-Eye Calibration Result")
    print("=" * 60)
    print("\nRotation matrix (camera to gripper):")
    print(R_cam2gripper)
    print("\nTranslation vector (camera to gripper) [meters]:")
    print(t_cam2gripper.flatten())
    print("=" * 60)

    # Save to file
    result = {
        'R_cam2gripper': R_cam2gripper.tolist(),
        't_cam2gripper': t_cam2gripper.tolist(),
        'num_poses': len(R_gripper2base)
    }

    output_file = 'hand_eye_calibration.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    print(f"\nCalibration saved to {output_file}")
    print("Done!")

    driver.cleanup()


if __name__ == "__main__":
    main()
