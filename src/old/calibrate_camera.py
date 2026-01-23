#!/usr/bin/env python3
"""
Calibrate RealSense camera intrinsics using chessboard.

This will give you accurate camera matrix and distortion coefficients
needed for hand-eye calibration.
"""

import cv2
import numpy as np
import json
import logging
import pyrealsense2 as rs
from pathlib import Path

# Data folder
DATA_DIR = Path(__file__).parent.parent / "data" / "camera_calibration_data"
DATA_DIR.mkdir(exist_ok=True, parents=True)

# Setup logging
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True, parents=True)
LOG_FILE = LOG_DIR / "camera_calibration.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logger.info(f"Logging to {LOG_FILE}")

# Chessboard parameters
# Try both orientations - OpenCV is sensitive to order
BOARD_SIZE = (7, 4)  # Internal corners (columns, rows)
SQUARE_SIZE = 0.025  # 25mm squares

def main():
    logger.info("=" * 60)
    logger.info("Camera Intrinsic Calibration")
    logger.info("=" * 60)
    logger.info("")
    logger.info("Instructions:")
    logger.info("1. Hold the chessboard in front of the camera")
    logger.info("2. Move it to different positions and angles")
    logger.info("3. Press SPACE when board is detected to capture")
    logger.info("4. Collect 15-20 images from different views")
    logger.info("5. Press 'q' when done to compute calibration")
    logger.info("")
    logger.info("Tips:")
    logger.info("- Move board to corners of camera view")
    logger.info("- Tilt board at different angles")
    logger.info("- Vary distance from camera")
    logger.info("=" * 60)
    logger.info("")

    # Create data directory
    DATA_DIR.mkdir(exist_ok=True, parents=True)
    logger.info(f"Saving calibration data to: {DATA_DIR.absolute()}")
    logger.info("")

    # Start camera
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    pipeline.start(config)

    # Storage for calibration images
    all_corners = []
    all_objpoints = []
    all_images = []
    img_size = None

    # Prepare object points (0,0,0), (1,0,0), (2,0,0) ...
    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    cv2.namedWindow("Camera Calibration", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Camera Calibration", 1280, 960)

    num_captured = 0

    try:
        while True:
            # Get frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue

            image = np.asanyarray(color_frame.get_data())
            display = image.copy()

            if img_size is None:
                img_size = (image.shape[1], image.shape[0])

            # Detect chessboard
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, None)

            board_detected = False

            if ret:
                # Refine corners
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

                board_detected = True

                # Draw corners
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                cv2.putText(display, "BOARD DETECTED - Press SPACE to capture", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            else:
                cv2.putText(display, "No board detected", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            # Show count
            cv2.putText(display, f"Images captured: {num_captured}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Camera Calibration", display)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF

            if key == ord(' ') and board_detected:  # SPACE
                all_corners.append(corners)
                all_objpoints.append(objp)
                all_images.append(image.copy())

                # Save image to disk
                img_path = DATA_DIR / f"calibration_{num_captured:03d}.png"
                cv2.imwrite(str(img_path), image)

                num_captured += 1
                logger.info(f"✓ Captured image {num_captured} -> {img_path.name}")

            elif key == ord('q'):
                logger.info("Finishing calibration...")
                break

            elif key == 27:  # ESC
                logger.info("Cancelled")
                pipeline.stop()
                cv2.destroyAllWindows()
                return

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()

    # Check if we have enough images
    if num_captured < 10:
        logger.error(f"Need at least 10 images, only captured {num_captured}")
        logger.error("Please run again and capture more images.")
        return

    # Compute calibration
    logger.info(f"Computing calibration from {num_captured} images...")

    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        all_objpoints,
        all_corners,
        img_size,
        None,
        None
    )

    # Compute reprojection error
    total_error = 0
    for i in range(len(all_objpoints)):
        imgpoints2, _ = cv2.projectPoints(
            all_objpoints[i],
            rvecs[i],
            tvecs[i],
            camera_matrix,
            dist_coeffs
        )
        error = cv2.norm(all_corners[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        total_error += error

    mean_error = total_error / len(all_objpoints)

    # Display results
    logger.info("=" * 60)
    logger.info("Camera Calibration Result")
    logger.info("=" * 60)
    logger.info(f"\nCamera Matrix:\n{camera_matrix}")
    logger.info(f"\nDistortion Coefficients:\n{dist_coeffs.flatten()}")
    logger.info(f"\nMean Reprojection Error: {mean_error:.4f} pixels")
    logger.info("=" * 60)

    if mean_error > 1.0:
        logger.warning("Reprojection error is high (>1 pixel)")
        logger.warning("Consider recalibrating with more diverse images")

    # Save to file
    result = {
        'camera_matrix': camera_matrix.tolist(),
        'dist_coeffs': dist_coeffs.tolist(),
        'reprojection_error': float(mean_error),
        'num_images': num_captured,
        'image_size': list(img_size),
        'board_size': list(BOARD_SIZE),
        'square_size': SQUARE_SIZE
    }

    output_file = DATA_DIR / 'camera_intrinsics.json'
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Calibration saved to {output_file}")
    logger.info(f"Calibration images saved in: {DATA_DIR.absolute()}")
    logger.info("You can now use these intrinsics in the hand-eye calibration script.")
    logger.info("Done!")


if __name__ == "__main__":
    main()
