#!/usr/bin/env python3
"""
Hand-Eye Calibration (X) Verification Data Collection - Using ROS TF

This version uses ROS TF to get T_gripper2base, ensuring consistency with MoveIt.
It also uses ROS image topics instead of pyrealsense2 directly.

Requirements:
- ROS2 must be running with robot state publisher
- Camera node must be running (ros2 launch realsense2_camera rs_launch.py)
- TF tree must include base_link and ee_gripper_link

Usage:
    source /opt/ros/jazzy/setup.bash
    source ~/ros2_ws/install/setup.bash
    python3 src/verify_X_calibration_tf.py
"""

import cv2
import numpy as np
import json
import yaml
import time
import logging
from pathlib import Path
from datetime import datetime

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import Image, CameraInfo
from cv_bridge import CvBridge
from scipy.spatial.transform import Rotation

# ============================================================================
# Configuration
# ============================================================================

# Checkerboard parameters - 8x6 internal corners
BOARD_SIZE = (8, 6)  # (columns, rows) internal corners
SQUARE_SIZE = 0.025  # 25mm squares

# TF frames
BASE_FRAME = "base_link"
EE_FRAME = "ee_gripper_link"

# ROS topics
IMAGE_TOPIC = "/camera/camera/color/image_raw"
CAMERA_INFO_TOPIC = "/camera/camera/color/camera_info"

# Maximum number of samples to collect
MAX_SAMPLES = 5

# Directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Xverification_tf"
LOG_DIR = BASE_DIR / "logs"

# Create directories
DATA_DIR.mkdir(exist_ok=True, parents=True)
LOG_DIR.mkdir(exist_ok=True, parents=True)

# Logging setup
LOG_FILE = LOG_DIR / "verify_X_calibration_tf.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


class VerificationNode(Node):
    """ROS2 node for verification data collection."""

    def __init__(self):
        super().__init__('verification_node')

        # TF
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # CV Bridge
        self.bridge = CvBridge()

        # Latest image and camera info
        self.latest_image = None
        self.camera_matrix = None
        self.dist_coeffs = None

        # QoS for sensor data
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to image
        self.image_sub = self.create_subscription(
            Image,
            IMAGE_TOPIC,
            self.image_callback,
            sensor_qos
        )

        # Subscribe to camera info
        self.info_sub = self.create_subscription(
            CameraInfo,
            CAMERA_INFO_TOPIC,
            self.camera_info_callback,
            sensor_qos
        )

        logger.info(f"Subscribed to {IMAGE_TOPIC}")
        logger.info(f"Subscribed to {CAMERA_INFO_TOPIC}")

    def image_callback(self, msg):
        """Store the latest image."""
        try:
            self.latest_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except Exception as e:
            logger.error(f"Failed to convert image: {e}")

    def camera_info_callback(self, msg):
        """Store camera intrinsics."""
        if self.camera_matrix is None:
            self.camera_matrix = np.array(msg.k).reshape(3, 3).astype(np.float32)
            self.dist_coeffs = np.array(msg.d).astype(np.float32)
            logger.info(f"Camera intrinsics received: fx={self.camera_matrix[0,0]:.2f}, fy={self.camera_matrix[1,1]:.2f}")

    def get_transform(self, target_frame, source_frame, timeout=1.0):
        """Get transform from source_frame to target_frame."""
        try:
            transform = self.tf_buffer.lookup_transform(
                target_frame,
                source_frame,
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout)
            )
            return transform
        except Exception as e:
            logger.error(f"Failed to get transform {source_frame} -> {target_frame}: {e}")
            return None


def transform_to_matrix(transform: TransformStamped) -> np.ndarray:
    """Convert ROS TransformStamped to 4x4 homogeneous matrix."""
    t = transform.transform.translation
    q = transform.transform.rotation

    # Build rotation matrix from quaternion
    rot = Rotation.from_quat([q.x, q.y, q.z, q.w])
    R = rot.as_matrix()

    # Build 4x4 matrix
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [t.x, t.y, t.z]

    return T


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

    return X_matrix


def detect_checkerboard(image, camera_matrix, dist_coeffs):
    """Detect checkerboard and get its pose in camera frame."""
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
        return True, corners, rvec, tvec

    return False, None, None, None


def get_robot_pose_from_tf(node: VerificationNode):
    """Get T_gripper2base from ROS TF."""
    # Get transform: base_link -> ee_gripper_link
    transform = node.get_transform(BASE_FRAME, EE_FRAME, timeout=1.0)

    if transform is None:
        return None

    # Convert to 4x4 matrix
    T_gripper2base = transform_to_matrix(transform)

    # Extract translation and quaternion for storage
    t = transform.transform.translation
    q = transform.transform.rotation

    return {
        "translation": [t.x, t.y, t.z],
        "quaternion_xyzw": [q.x, q.y, q.z, q.w],
        "T_gripper2base": T_gripper2base.tolist(),
        "source": "ROS_TF",
        "base_frame": BASE_FRAME,
        "ee_frame": EE_FRAME,
    }


def save_pose_data(pose_id, image, robot_pose, rvec, tvec, camera_matrix, dist_coeffs):
    """Save all data for a single pose."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    img_path = DATA_DIR / f"pose_{pose_id:03d}.png"
    cv2.imwrite(str(img_path), image)
    logger.info(f"  Image saved: {img_path.name}")

    R_target2cam, _ = cv2.Rodrigues(rvec)
    T_target2cam = np.eye(4)
    T_target2cam[:3, :3] = R_target2cam
    T_target2cam[:3, 3] = tvec.flatten()

    pose_data = {
        "pose_id": pose_id,
        "timestamp": timestamp,
        "image_file": f"pose_{pose_id:03d}.png",
        "robot": robot_pose,
        "checkerboard": {
            "rvec": rvec.flatten().tolist(),
            "tvec": tvec.flatten().tolist(),
            "T_target2cam": T_target2cam.tolist(),
        },
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.flatten().tolist(),
    }

    return pose_data


def main():
    logger.info("=" * 70)
    logger.info("Hand-Eye Calibration (X) Verification - Using ROS TF")
    logger.info("=" * 70)
    logger.info("")
    logger.info(f"TF frames: {BASE_FRAME} -> {EE_FRAME}")
    logger.info(f"Image topic: {IMAGE_TOPIC}")
    logger.info(f"Checkerboard: {BOARD_SIZE[0]}x{BOARD_SIZE[1]} internal corners")
    logger.info(f"Data directory: {DATA_DIR}")
    logger.info("")
    logger.info("IMPORTANT: Make sure these are running:")
    logger.info("  1. Robot: ros2 launch trossen_arm_bringup arm.launch.py ...")
    logger.info("  2. Camera: ros2 launch realsense2_camera rs_launch.py")
    logger.info("")

    # Initialize ROS2
    rclpy.init()
    node = VerificationNode()

    # Wait for TF and camera to be available
    logger.info("Waiting for TF transforms and camera...")
    for i in range(30):  # Wait up to 15 seconds
        rclpy.spin_once(node, timeout_sec=0.5)

        has_tf = node.get_transform(BASE_FRAME, EE_FRAME, timeout=0.5) is not None
        has_camera = node.latest_image is not None and node.camera_matrix is not None

        if has_tf and has_camera:
            logger.info("TF transforms and camera available!")
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
            logger.error(f"Could not get TF transform {BASE_FRAME} -> {EE_FRAME}")
        if not has_camera:
            logger.error(f"No camera data on {IMAGE_TOPIC}")
        rclpy.shutdown()
        return 1

    # Load hand-eye calibration result
    X_cam2gripper = load_hand_eye_result()
    if X_cam2gripper is None:
        logger.error("Cannot proceed without hand-eye calibration result")
        rclpy.shutdown()
        return 1

    # Create window
    cv2.namedWindow("X Verification (TF)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("X Verification (TF)", 800, 600)

    all_poses = []
    pose_count = 0

    logger.info("")
    logger.info("=" * 70)
    logger.info("Move robot by hand and press 's' when checkerboard is detected")
    logger.info("Press 'q' to quit")
    logger.info("=" * 70)

    try:
        while rclpy.ok():
            # Spin ROS to get latest data
            rclpy.spin_once(node, timeout_sec=0.01)

            # Get latest image
            if node.latest_image is None:
                continue

            image = node.latest_image.copy()
            display = image.copy()
            camera_matrix = node.camera_matrix
            dist_coeffs = node.dist_coeffs

            # Detect checkerboard
            detected, corners, rvec, tvec = detect_checkerboard(
                image, camera_matrix, dist_coeffs
            )

            # Status bar
            cv2.rectangle(display, (0, 0), (display.shape[1], 40), (40, 40, 40), -1)

            if detected:
                cv2.drawChessboardCorners(display, BOARD_SIZE, corners, True)
                cv2.drawFrameAxes(display, camera_matrix, dist_coeffs, rvec, tvec, 0.05)

                distance = np.linalg.norm(tvec)
                cv2.putText(
                    display,
                    f"DETECTED | Dist: {distance:.3f}m | 's'=SAVE | Poses: {pose_count}/{MAX_SAMPLES}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )
            else:
                cv2.putText(
                    display,
                    f"NO BOARD | Poses: {pose_count}/{MAX_SAMPLES}",
                    (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2,
                )

            cv2.imshow("X Verification (TF)", display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("s") and detected:
                pose_count += 1
                logger.info(f"Saving pose {pose_count}...")

                # Get robot pose from TF
                robot_pose = get_robot_pose_from_tf(node)
                if robot_pose is None:
                    logger.error("Failed to get robot pose from TF!")
                    pose_count -= 1
                    continue

                t = robot_pose["translation"]
                logger.info(f"  TF gripper position: [{t[0]:.4f}, {t[1]:.4f}, {t[2]:.4f}]")

                # Save data
                pose_data = save_pose_data(
                    pose_count, image, robot_pose, rvec, tvec,
                    camera_matrix, dist_coeffs
                )
                all_poses.append(pose_data)

                # Save JSON
                output_data = {
                    "description": "X verification data using ROS TF",
                    "calibration_type": "eye-in-hand",
                    "parent_frame": EE_FRAME,
                    "child_frame": "camera_color_optical_frame",
                    "tf_source": "ROS2_TF",
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

                logger.info(f"  Pose {pose_count}/{MAX_SAMPLES} saved")

                if pose_count >= MAX_SAMPLES:
                    logger.info(f"Collected all {MAX_SAMPLES} samples!")
                    break

            elif key == ord("q") or key == 27:
                logger.info("Quit requested")
                break

    finally:
        cv2.destroyAllWindows()
        rclpy.shutdown()

        logger.info("")
        logger.info("=" * 70)
        logger.info(f"Data collection complete: {pose_count} poses saved")
        if pose_count > 0:
            logger.info(f"Data saved to: {DATA_DIR}")
        logger.info("=" * 70)

    return 0


if __name__ == "__main__":
    exit(main())
