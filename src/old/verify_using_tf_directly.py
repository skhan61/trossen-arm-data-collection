#!/usr/bin/env python3
"""
Verify hand-eye calibration using TF directly.

This script bypasses the explicit X matrix entirely and uses TF to directly
transform the checkerboard position from camera frame to base frame.

If the URDF correctly defines the camera position, then:
    T_checkerboard_in_base = TF.lookup_transform('base_link', 'camera_color_optical_frame') @ T_checkerboard_in_camera

This is the ground truth test - if the checkerboard appears at consistent
positions across different robot poses, the URDF/TF is correct.

Usage:
    1. Launch robot: ros2 launch trossen_arm_bringup arm.launch.py arm:=wxai variant:=follower ...
    2. Launch camera: ros2 launch realsense2_camera rs_launch.py
    3. Run: python3 src/verify_using_tf_directly.py
"""

import cv2
import numpy as np
import json
import time
import logging
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

# ============================================================================
# Configuration
# ============================================================================

BOARD_SIZE = (8, 6)
SQUARE_SIZE = 0.025

BASE_FRAME = "base_link"
CAMERA_FRAME = "camera_color_optical_frame"

IMAGE_TOPIC = "/camera/camera/color/image_rect_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

MAX_SAMPLES = 5

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "tf_direct_verification"
DATA_DIR.mkdir(exist_ok=True, parents=True)

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class VerificationNode(Node):
    def __init__(self):
        super().__init__('tf_verification_node')

        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        self.bridge = CvBridge()
        self.latest_image = None
        self.camera_matrix = None
        self.dist_coeffs = None

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        self.image_sub = self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, sensor_qos)
        self.info_sub = self.create_subscription(
            CameraInfo, CAMERA_INFO_TOPIC, self.camera_info_callback, sensor_qos)

    def image_callback(self, msg):
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logger.error(f"Image conversion failed: {e}")

    def camera_info_callback(self, msg):
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3).astype(np.float32)
            self.dist_coeffs = np.array(msg.d).astype(np.float32)
            logger.info(f"Camera intrinsics: fx={self.camera_matrix[0,0]:.1f}")


def transform_to_matrix(transform):
    """Convert ROS TransformStamped to 4x4 matrix."""
    t = transform.transform.translation
    q = transform.transform.rotation
    R = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]
    return T


def detect_checkerboard(image, camera_matrix, dist_coeffs):
    """Detect checkerboard and return T_target2cam."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    ret, corners = cv2.findChessboardCorners(gray, BOARD_SIZE, flags)

    if not ret:
        return False, None, None, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

    objp = np.zeros((BOARD_SIZE[0] * BOARD_SIZE[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0:BOARD_SIZE[0], 0:BOARD_SIZE[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE

    success, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)

    if success:
        R, _ = cv2.Rodrigues(rvec)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = tvec.flatten()
        return True, corners, T, tvec

    return False, None, None, None


def main():
    logger.info("=" * 70)
    logger.info("Hand-Eye Verification using TF DIRECTLY")
    logger.info("=" * 70)
    logger.info("")
    logger.info("This test uses TF to transform directly from camera to base,")
    logger.info("bypassing any explicit X matrix. If the checkerboard appears")
    logger.info("at the same base-frame position from different poses, the")
    logger.info("URDF-defined camera transform is correct.")
    logger.info("")

    rclpy.init()
    node = VerificationNode()

    # Wait for everything
    logger.info("Waiting for TF and camera...")
    for i in range(30):
        rclpy.spin_once(node, timeout_sec=0.5)

        try:
            tf = node.tf_buffer.lookup_transform(
                BASE_FRAME, CAMERA_FRAME,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.3)
            )
            has_tf = True
        except:
            has_tf = False

        has_camera = node.latest_image is not None and node.camera_matrix is not None

        if has_tf and has_camera:
            logger.info("TF and camera ready!")
            break

        if i % 5 == 0:
            status = []
            if not has_tf:
                status.append("waiting for TF")
            if not has_camera:
                status.append("waiting for camera")
            logger.info(f"  {', '.join(status)}...")
    else:
        if not has_tf:
            logger.error(f"TF transform {BASE_FRAME} -> {CAMERA_FRAME} not available")
            logger.error("")
            logger.error("Is the robot launched with variant:=follower?")
            logger.error("Is the camera node running?")
        if not has_camera:
            logger.error(f"No camera data on {IMAGE_TOPIC}")
        rclpy.shutdown()
        return 1

    # Create window
    cv2.namedWindow("TF Direct Verification", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("TF Direct Verification", 800, 600)

    all_poses = []
    all_board_positions = []
    pose_count = 0

    logger.info("")
    logger.info("=" * 70)
    logger.info("Move robot by hand, press 's' when checkerboard is detected")
    logger.info("Press 'q' to quit")
    logger.info("=" * 70)

    try:
        while rclpy.ok():
            rclpy.spin_once(node, timeout_sec=0.01)

            if node.latest_image is None:
                continue

            image = node.latest_image.copy()
            display = image.copy()

            detected, corners, T_target_in_camera, tvec = detect_checkerboard(
                image, node.camera_matrix, node.dist_coeffs
            )

            cv2.rectangle(display, (0, 0), (display.shape[1], 50), (40, 40, 40), -1)

            if detected:
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)

                # Get TF: base_link <- camera_color_optical_frame
                try:
                    tf = node.tf_buffer.lookup_transform(
                        BASE_FRAME, CAMERA_FRAME,
                        rclpy.time.Time(),
                        timeout=rclpy.duration.Duration(seconds=0.1)
                    )
                    T_camera_in_base = transform_to_matrix(tf)

                    # Transform checkerboard to base frame
                    T_target_in_base = T_camera_in_base @ T_target_in_camera
                    board_pos = T_target_in_base[:3, 3]

                    cv2.putText(
                        display,
                        f"DETECTED | Board in base: [{board_pos[0]:.3f}, {board_pos[1]:.3f}, {board_pos[2]:.3f}]m",
                        (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 2
                    )
                    cv2.putText(
                        display,
                        f"Poses: {pose_count}/{MAX_SAMPLES} | Press 's' to save",
                        (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1
                    )

                except Exception as e:
                    cv2.putText(display, f"TF ERROR: {e}", (10, 25),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            else:
                cv2.putText(
                    display,
                    f"NO CHECKERBOARD | Poses: {pose_count}/{MAX_SAMPLES}",
                    (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2
                )

            cv2.imshow("TF Direct Verification", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and detected:
                pose_count += 1

                # Get TF
                tf = node.tf_buffer.lookup_transform(
                    BASE_FRAME, CAMERA_FRAME,
                    rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5)
                )
                T_camera_in_base = transform_to_matrix(tf)
                T_target_in_base = T_camera_in_base @ T_target_in_camera
                board_pos = T_target_in_base[:3, 3]

                all_board_positions.append(board_pos)

                # Save image
                img_path = DATA_DIR / f"pose_{pose_count:03d}.png"
                cv2.imwrite(str(img_path), image)

                pose_data = {
                    "pose_id": pose_count,
                    "timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
                    "board_position_in_base": board_pos.tolist(),
                    "T_target_in_camera": T_target_in_camera.tolist(),
                    "T_camera_in_base": T_camera_in_base.tolist(),
                    "T_target_in_base": T_target_in_base.tolist(),
                }
                all_poses.append(pose_data)

                logger.info(f"Pose {pose_count}: Board in base = [{board_pos[0]:.4f}, {board_pos[1]:.4f}, {board_pos[2]:.4f}] m")

                if pose_count >= MAX_SAMPLES:
                    logger.info(f"Collected {MAX_SAMPLES} samples!")
                    break

            elif key == ord("q") or key == 27:
                break

    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

    # Analysis
    if len(all_board_positions) >= 2:
        logger.info("")
        logger.info("=" * 70)
        logger.info("RESULTS")
        logger.info("=" * 70)
        logger.info("")

        positions = np.array(all_board_positions)
        mean_pos = np.mean(positions, axis=0)
        errors = np.linalg.norm(positions - mean_pos, axis=1)

        logger.info(f"Mean board position in base frame:")
        logger.info(f"  [{mean_pos[0]:.4f}, {mean_pos[1]:.4f}, {mean_pos[2]:.4f}] m")
        logger.info("")

        logger.info("Error from mean for each pose:")
        for i, err in enumerate(errors):
            logger.info(f"  Pose {i+1}: {err*1000:.2f} mm")
        logger.info("")

        max_err = np.max(errors) * 1000
        mean_err = np.mean(errors) * 1000

        logger.info(f"Mean error: {mean_err:.2f} mm")
        logger.info(f"Max error:  {max_err:.2f} mm")
        logger.info("")

        if max_err < 5:
            quality = "EXCELLENT"
        elif max_err < 15:
            quality = "GOOD"
        elif max_err < 30:
            quality = "FAIR"
        else:
            quality = "POOR"

        logger.info(f"Calibration quality: {quality}")
        logger.info("")

        # Save results
        results = {
            "description": "TF direct verification - no explicit X matrix",
            "base_frame": BASE_FRAME,
            "camera_frame": CAMERA_FRAME,
            "num_poses": pose_count,
            "mean_board_position": mean_pos.tolist(),
            "errors_mm": (errors * 1000).tolist(),
            "mean_error_mm": float(mean_err),
            "max_error_mm": float(max_err),
            "quality": quality,
            "poses": all_poses,
        }

        json_path = DATA_DIR / "tf_verification_results.json"
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info(f"Results saved to: {json_path}")

    return 0


if __name__ == "__main__":
    exit(main())
