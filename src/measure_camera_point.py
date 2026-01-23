#!/usr/bin/env python3
"""
Measure GelSight sensor optical frame center.

Click 4 corners of the sensor → computes center point in camera frame.

No ROS - uses Trossen ARM API and RealSense directly.

Usage:
    python3 src/measure_camera_point.py
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import json
import logging
import time
from pathlib import Path
from datetime import datetime

from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# ============================================================================
# Logging Setup
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"measure_sensor_center_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

ARM_IP = "192.168.1.99"


class SensorMeasurer:
    def __init__(self):
        self.pipeline = None
        self.driver = None
        self.align = None
        self.depth_scale = 0.001
        self.intrinsics = None

        # Click data
        self.clicked_point = None
        self.mouse_pos = (0, 0)  # Current mouse position

        # 4 corners of sensor
        self.corners_2d = []  # pixel coordinates
        self.corners_3d = []  # 3D coordinates in camera frame

        # Auto-detected sensors
        self.detected_sensors = []  # List of detected rectangles
        self.selected_sensor_idx = 0  # Currently selected detection

    def detect_sensors(self, color_image, depth_frame):
        """Auto-detect rectangular sensor regions (25mm x 20mm sensors)."""
        gray = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

        # Multiple detection approaches
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # Canny edge detection with lower thresholds for small objects
        edges = cv2.Canny(blurred, 20, 80)

        # Dilate slightly to connect edges
        kernel = np.ones((2, 2), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        logger.info(f"  Found {len(contours)} contours")

        detected = []
        for contour in contours:
            # Use minAreaRect for rotated rectangles
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            # Get dimensions
            width, height = rect[1]
            if width == 0 or height == 0:
                continue

            # Sensor is 25x20mm, at ~120mm = ~70x56 pixels
            # Allow for range 100-150mm distance = ~50-100 pixels per side
            min_dim = min(width, height)
            max_dim = max(width, height)

            # Filter by size (expecting 40-120 pixels per side)
            if min_dim < 30 or max_dim > 150:
                continue

            # Check aspect ratio (25/20 = 1.25, allow 1.0 to 2.0)
            aspect = max_dim / min_dim
            if aspect > 2.5:
                continue

            area = width * height

            # Get corners depth
            corners = box.astype(float)
            depths = []
            for cx, cy in corners:
                cx, cy = int(cx), int(cy)
                if 0 <= cx < 640 and 0 <= cy < 480:
                    d = depth_frame.get_distance(cx, cy) * 1000
                    if 50 < d < 300:
                        depths.append(d)

            if len(depths) >= 3:
                avg_depth = np.mean(depths)
                corners = self.order_corners(corners)
                detected.append({
                    'corners': np.array(corners),
                    'area': area,
                    'avg_depth': avg_depth
                })
                logger.info(f"  Found: {width:.0f}x{height:.0f}px, area={area:.0f}, depth={avg_depth:.0f}mm")

        detected.sort(key=lambda x: x['area'], reverse=True)
        return detected[:10]

    def order_corners(self, corners):
        """Order corners as: top-left, top-right, bottom-right, bottom-left."""
        corners = [np.array(c) for c in corners]

        # Sort by y first (top to bottom)
        corners = sorted(corners, key=lambda p: p[1])

        # Top two points
        top = sorted(corners[:2], key=lambda p: p[0])
        # Bottom two points
        bottom = sorted(corners[2:], key=lambda p: p[0])

        # Order: top-left, top-right, bottom-right, bottom-left
        return [np.array(top[0]), np.array(top[1]), np.array(bottom[1]), np.array(bottom[0])]

    def connect_camera(self):
        """Connect to RealSense camera."""
        logger.info("Connecting to RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

        profile = self.pipeline.start(config)

        depth_sensor = profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        logger.info(f"Depth scale: {self.depth_scale}")

        self.align = rs.align(rs.stream.color)

        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        logger.info(f"Camera intrinsics: fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}")
        logger.info(f"  Principal point: cx={self.intrinsics.ppx:.2f}, cy={self.intrinsics.ppy:.2f}")
        logger.info("Camera connected!")

    def connect_robot(self):
        """Connect to robot."""
        logger.info(f"Connecting to robot at {ARM_IP}...")
        self.driver = TrossenArmDriver()
        self.driver.configure(
            model=Model.wxai_v0,
            end_effector=StandardEndEffector.wxai_v0_follower,
            serv_ip=ARM_IP,
            clear_error=True,
            timeout=10.0,
        )
        logger.info("Robot connected!")

    def get_robot_pose(self):
        """Get current gripper pose from robot API."""
        cartesian = self.driver.get_cartesian_positions()
        t = np.array(cartesian[:3])
        angle_axis = np.array(cartesian[3:6])
        R, _ = cv2.Rodrigues(angle_axis)

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T, cartesian

    def pixel_to_3d(self, u, v, depth_frame):
        """Convert pixel (u,v) to 3D point using depth."""
        depth = depth_frame.get_distance(int(u), int(v))
        if depth <= 0:
            return None

        point_3d = rs.rs2_deproject_pixel_to_point(
            self.intrinsics, [u, v], depth
        )
        return np.array(point_3d)

    def mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks and movement."""
        self.mouse_pos = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN:
            self.clicked_point = (x, y)

    def compute_sensor_frame(self):
        """Compute full 6 DOF sensor frame from 4 corners.

        Corners should be clicked in order:
        1 (top-left) -> 2 (top-right) -> 3 (bottom-right) -> 4 (bottom-left)

        Returns:
            center: 3D position of sensor center
            T_sensor_in_cam: 4x4 transform of sensor frame in camera frame
            rpy: roll, pitch, yaw angles (radians)
        """
        if len(self.corners_3d) != 4:
            return None, None, None

        c1, c2, c3, c4 = [np.array(c) for c in self.corners_3d]

        # Center is average of 4 corners
        center = (c1 + c2 + c3 + c4) / 4.0

        # X-axis: right direction (corner1 -> corner2 and corner4 -> corner3 average)
        x_axis = ((c2 - c1) + (c3 - c4)) / 2.0
        x_axis = x_axis / np.linalg.norm(x_axis)

        # Y-axis: down direction (corner1 -> corner4 and corner2 -> corner3 average)
        y_axis = ((c4 - c1) + (c3 - c2)) / 2.0
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Z-axis: normal to sensor surface (into sensor)
        z_axis = np.cross(x_axis, y_axis)
        z_axis = z_axis / np.linalg.norm(z_axis)

        # Re-orthogonalize Y
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)

        # Build rotation matrix (columns are axes)
        R = np.column_stack([x_axis, y_axis, z_axis])

        # Build 4x4 transform
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center

        # Convert to roll, pitch, yaw
        # Using ZYX convention (yaw-pitch-roll)
        sy = np.sqrt(R[0,0]**2 + R[1,0]**2)
        singular = sy < 1e-6

        if not singular:
            roll = np.arctan2(R[2,1], R[2,2])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = np.arctan2(R[1,0], R[0,0])
        else:
            roll = np.arctan2(-R[1,2], R[1,1])
            pitch = np.arctan2(-R[2,0], sy)
            yaw = 0

        rpy = np.array([roll, pitch, yaw])

        return center, T, rpy

    def run(self):
        """Main loop."""
        cv2.namedWindow("GelSight Sensor Measurer", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("GelSight Sensor Measurer", self.mouse_callback)

        logger.info("")
        logger.info(f"Log file: {LOG_FILE}")
        logger.info("")
        logger.info("=" * 70)
        logger.info("INSTRUCTIONS:")
        logger.info("  Press 'a' for AUTO-DETECT (recommended)")
        logger.info("  Or click 3 corners manually: top-left -> top-right -> bottom-right")
        logger.info("")
        logger.info("  Keys:")
        logger.info("    'a' - AUTO-DETECT sensors")
        logger.info("    'n' - NEXT detection (cycle through)")
        logger.info("    ENTER - ACCEPT current detection")
        logger.info("    's' - SAVE sensor and measure next")
        logger.info("    'r' - RESET current corners")
        logger.info("    'q' - QUIT and save all")
        logger.info("=" * 70)
        logger.info("")

        all_sensors = []  # List of all measured sensors
        current_result = None
        sensor_count = 0

        try:
            while True:
                frames = self.pipeline.wait_for_frames()
                aligned = self.align.process(frames)

                color_frame = aligned.get_color_frame()
                depth_frame = aligned.get_depth_frame()

                if not color_frame or not depth_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                display = color_image.copy()

                # Add depth overlay - show areas with valid depth in green tint
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_colormap = cv2.applyColorMap(
                    cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET
                )
                # Blend depth overlay (20% opacity)
                display = cv2.addWeighted(display, 0.8, depth_colormap, 0.2, 0)

                T_gripper2base, cartesian = self.get_robot_pose()

                # Draw existing corners
                for i, (u, v) in enumerate(self.corners_2d):
                    cv2.circle(display, (u, v), 5, (0, 0, 255), -1)
                    cv2.putText(display, str(i+1), (u+10, v), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

                # Draw lines between corners if we have them
                if len(self.corners_2d) >= 2:
                    for i in range(len(self.corners_2d) - 1):
                        cv2.line(display, self.corners_2d[i], self.corners_2d[i+1], (0, 255, 0), 2)
                    if len(self.corners_2d) == 4:
                        cv2.line(display, self.corners_2d[3], self.corners_2d[0], (0, 255, 0), 2)

                # Draw auto-detected sensors
                for i, det in enumerate(self.detected_sensors):
                    corners = det['corners']
                    color = (0, 255, 0) if i == self.selected_sensor_idx else (100, 100, 100)
                    thickness = 3 if i == self.selected_sensor_idx else 1

                    # Draw rectangle
                    for j in range(4):
                        pt1 = tuple(corners[j].astype(int))
                        pt2 = tuple(corners[(j+1) % 4].astype(int))
                        cv2.line(display, pt1, pt2, color, thickness)

                    # Label
                    center = np.mean(corners, axis=0).astype(int)
                    label = f"#{i+1} ({det['avg_depth']:.0f}mm)"
                    cv2.putText(display, label, tuple(center), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Show detection status
                if self.detected_sensors:
                    cv2.putText(display, f"Detected: {len(self.detected_sensors)} | Selected: #{self.selected_sensor_idx+1} | Press ENTER to accept",
                               (10, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                # Show depth at mouse position
                mx, my = self.mouse_pos
                if 0 <= mx < 640 and 0 <= my < 480:
                    hover_depth = depth_frame.get_distance(mx, my) * 1000  # mm
                    cv2.putText(display, f"Depth @ cursor: {hover_depth:.0f} mm",
                               (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                    # Draw crosshair at mouse
                    cv2.drawMarker(display, (mx, my), (0, 255, 255), cv2.MARKER_CROSS, 10, 1)

                # Process clicked point (only need 3 clicks, 4th is auto-computed)
                if self.clicked_point and len(self.corners_2d) < 3:
                    u, v = self.clicked_point
                    point_3d = self.pixel_to_3d(u, v, depth_frame)

                    if point_3d is not None:
                        depth_mm = point_3d[2] * 1000

                        # Just warn but still accept the click
                        if depth_mm < 50 or depth_mm > 500:
                            logger.warning(f"WARNING: Depth at ({u}, {v}) = {depth_mm:.0f} mm (outside 50-500mm range)")

                        # Check consistency with previous corners - warn but accept
                        if len(self.corners_3d) > 0:
                            prev_depths = [c[2] * 1000 for c in self.corners_3d]
                            avg_depth = np.mean(prev_depths)
                            if abs(depth_mm - avg_depth) > 100:  # More than 100mm difference
                                logger.warning(f"WARNING: Depth inconsistent! This: {depth_mm:.0f} mm, prev avg: {avg_depth:.0f} mm")

                        self.corners_2d.append((u, v))
                        self.corners_3d.append(point_3d)

                        logger.info(f"Corner {len(self.corners_2d)}: pixel=({u}, {v}), depth={depth_mm:.0f} mm")
                        logger.info(f"  Camera frame: [{point_3d[0]*1000:.2f}, {point_3d[1]*1000:.2f}, {point_3d[2]*1000:.2f}] mm")

                        # After 3rd corner, auto-compute 4th corner
                        if len(self.corners_2d) == 3:
                            logger.info("")
                            logger.info("Auto-computing 4th corner (bottom-left)...")

                            c1_3d, c2_3d, c3_3d = [np.array(c) for c in self.corners_3d]
                            c4_3d = c1_3d + c3_3d - c2_3d  # Parallelogram rule

                            c1_2d, c2_2d, c3_2d = [np.array(c) for c in self.corners_2d]
                            c4_2d = c1_2d + c3_2d - c2_2d
                            c4_2d = (int(c4_2d[0]), int(c4_2d[1]))

                            self.corners_2d.append(c4_2d)
                            self.corners_3d.append(c4_3d)

                            logger.info(f"Corner 4 (computed): pixel=({c4_2d[0]}, {c4_2d[1]})")
                            logger.info(f"  Camera frame: [{c4_3d[0]*1000:.2f}, {c4_3d[1]*1000:.2f}, {c4_3d[2]*1000:.2f}] mm")

                            # Now compute full 6 DOF
                            center, T_sensor, rpy = self.compute_sensor_frame()
                            logger.info("")
                            logger.info("=" * 70)
                            logger.info("SENSOR 6-DOF POSE COMPUTED!")
                            logger.info("=" * 70)
                            logger.info("")
                            logger.info("Position (in camera frame):")
                            logger.info(f"  X: {center[0]*1000:.2f} mm")
                            logger.info(f"  Y: {center[1]*1000:.2f} mm")
                            logger.info(f"  Z: {center[2]*1000:.2f} mm")
                            logger.info("")
                            logger.info("Orientation (in camera frame):")
                            logger.info(f"  Roll:  {np.degrees(rpy[0]):.2f} deg ({rpy[0]:.4f} rad)")
                            logger.info(f"  Pitch: {np.degrees(rpy[1]):.2f} deg ({rpy[1]:.4f} rad)")
                            logger.info(f"  Yaw:   {np.degrees(rpy[2]):.2f} deg ({rpy[2]:.4f} rad)")
                            logger.info("")
                            logger.info("4x4 Transform (sensor in camera frame):")
                            for row in T_sensor:
                                logger.info(f"  [{row[0]:8.4f}, {row[1]:8.4f}, {row[2]:8.4f}, {row[3]:8.4f}]")
                            logger.info("")
                            logger.info(f"Gripper position: [{cartesian[0]*1000:.2f}, {cartesian[1]*1000:.2f}, {cartesian[2]*1000:.2f}] mm")
                            logger.info("=" * 70)
                            logger.info("")

                            current_result = {
                                "sensor_id": sensor_count + 1,
                                "timestamp": datetime.now().isoformat(),
                                "corners_2d": list(self.corners_2d),
                                "corners_3d_mm": [(c * 1000).tolist() if isinstance(c, np.ndarray) else (np.array(c) * 1000).tolist() for c in self.corners_3d],
                                "sensor_position_mm": (center * 1000).tolist(),
                                "sensor_rpy_rad": rpy.tolist(),
                                "sensor_rpy_deg": np.degrees(rpy).tolist(),
                                "T_sensor_in_camera": T_sensor.tolist(),
                                "gripper_cartesian": list(cartesian),
                                "T_gripper2base": T_gripper2base.tolist(),
                                "computed_corner": 4,
                            }
                            logger.info(f"Press 's' to save sensor {sensor_count + 1} and measure next sensor")
                    else:
                        logger.info(f"No depth at ({u}, {v}) - try again")

                    self.clicked_point = None

                # Draw center and show 6 DOF if computed
                if len(self.corners_2d) == 4:
                    center_2d = np.mean(self.corners_2d, axis=0).astype(int)
                    cv2.circle(display, tuple(center_2d), 8, (255, 0, 255), -1)
                    cv2.putText(display, "CENTER", (center_2d[0]+10, center_2d[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2)

                    center, T_sensor, rpy = self.compute_sensor_frame()
                    cv2.putText(display, f"Pos: [{center[0]*1000:.1f}, {center[1]*1000:.1f}, {center[2]*1000:.1f}] mm",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)
                    cv2.putText(display, f"RPY: [{np.degrees(rpy[0]):.1f}, {np.degrees(rpy[1]):.1f}, {np.degrees(rpy[2]):.1f}] deg",
                               (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 2)

                # Show status
                status = f"Sensor {sensor_count + 1} | Corners: {len(self.corners_2d)}/3"
                if len(self.corners_2d) < 3:
                    status += " - Click corner " + str(len(self.corners_2d) + 1)
                elif len(self.corners_2d) >= 3:
                    status = f"Sensor {sensor_count + 1} | READY - Press 's' to save"
                cv2.putText(display, status, (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                cv2.putText(display, f"Saved: {sensor_count} sensors", (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

                # Show gripper position
                h, w = display.shape[:2]
                cv2.putText(display, f"Gripper: [{cartesian[0]*1000:.1f}, {cartesian[1]*1000:.1f}, {cartesian[2]*1000:.1f}] mm",
                           (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

                cv2.imshow("GelSight Sensor Measurer", display)
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break

                elif key == ord('r'):
                    # Reset
                    self.corners_2d = []
                    self.corners_3d = []
                    self.detected_sensors = []
                    self.selected_sensor_idx = 0
                    current_result = None
                    logger.info("RESET - press 'a' for auto-detect or click 3 corners")

                elif key == ord('a'):
                    # Auto-detect sensors
                    logger.info("Auto-detecting sensors...")
                    self.detected_sensors = self.detect_sensors(color_image, depth_frame)
                    self.selected_sensor_idx = 0
                    if self.detected_sensors:
                        logger.info(f"Found {len(self.detected_sensors)} sensor(s)")
                        logger.info("Press 'n' to cycle, ENTER to accept")
                    else:
                        logger.info("No sensors detected - try moving camera or click manually")

                elif key == ord('n') and self.detected_sensors:
                    # Cycle to next detection
                    self.selected_sensor_idx = (self.selected_sensor_idx + 1) % len(self.detected_sensors)
                    det = self.detected_sensors[self.selected_sensor_idx]
                    logger.info(f"Selected sensor #{self.selected_sensor_idx + 1} (depth: {det['avg_depth']:.0f} mm)")

                elif key == 13 and self.detected_sensors:  # ENTER key
                    # Accept selected detection
                    det = self.detected_sensors[self.selected_sensor_idx]
                    corners = det['corners']

                    # Clear and set corners
                    self.corners_2d = [tuple(c.astype(int)) for c in corners]
                    self.corners_3d = []

                    # Get 3D coordinates for all 4 corners
                    valid = True
                    for u, v in self.corners_2d:
                        point_3d = self.pixel_to_3d(u, v, depth_frame)
                        if point_3d is not None:
                            self.corners_3d.append(point_3d)
                            logger.info(f"Corner: pixel=({u}, {v}), depth={point_3d[2]*1000:.0f} mm")
                        else:
                            valid = False
                            logger.warning(f"No depth at ({u}, {v})")
                            break

                    if valid and len(self.corners_3d) == 4:
                        # Compute 6 DOF
                        center, T_sensor, rpy = self.compute_sensor_frame()
                        logger.info("")
                        logger.info("=" * 70)
                        logger.info("SENSOR 6-DOF POSE COMPUTED (AUTO-DETECTED)!")
                        logger.info("=" * 70)
                        logger.info(f"Position: [{center[0]*1000:.2f}, {center[1]*1000:.2f}, {center[2]*1000:.2f}] mm")
                        logger.info(f"RPY: [{np.degrees(rpy[0]):.2f}, {np.degrees(rpy[1]):.2f}, {np.degrees(rpy[2]):.2f}] deg")
                        logger.info("=" * 70)

                        current_result = {
                            "sensor_id": sensor_count + 1,
                            "timestamp": datetime.now().isoformat(),
                            "corners_2d": list(self.corners_2d),
                            "corners_3d_mm": [(c * 1000).tolist() for c in self.corners_3d],
                            "sensor_position_mm": (center * 1000).tolist(),
                            "sensor_rpy_rad": rpy.tolist(),
                            "sensor_rpy_deg": np.degrees(rpy).tolist(),
                            "T_sensor_in_camera": T_sensor.tolist(),
                            "gripper_cartesian": list(cartesian),
                            "T_gripper2base": T_gripper2base.tolist(),
                            "detection_method": "auto",
                        }
                        logger.info("Press 's' to save")
                        self.detected_sensors = []  # Clear detections
                    else:
                        logger.warning("Failed to get valid depth for all corners - try again")
                        self.corners_2d = []
                        self.corners_3d = []

                elif key == ord('s') and current_result is not None:
                    # Save current sensor and reset for next
                    all_sensors.append(current_result)
                    sensor_count += 1
                    logger.info(f"Saved sensor {sensor_count}!")
                    logger.info(f"Total sensors measured: {sensor_count}")
                    logger.info("")
                    logger.info("Click 3 corners of NEXT sensor, or press 'q' to quit")
                    logger.info("")

                    # Reset for next sensor
                    self.corners_2d = []
                    self.corners_3d = []
                    current_result = None

        finally:
            cv2.destroyAllWindows()

            # Save all measured sensors
            if all_sensors:
                output_file = Path(__file__).parent.parent / "data" / "sensor_measurements.json"
                output_file.parent.mkdir(exist_ok=True)
                output_data = {
                    "num_sensors": len(all_sensors),
                    "sensors": all_sensors
                }
                with open(output_file, "w") as f:
                    json.dump(output_data, f, indent=2)
                logger.info("")
                logger.info(f"Saved {len(all_sensors)} sensors to {output_file}")

    def cleanup(self):
        """Cleanup resources."""
        if self.pipeline:
            self.pipeline.stop()
        if self.driver:
            self.driver.cleanup()


def main():
    logger.info("=" * 70)
    logger.info("GelSight Sensor Center Measurer")
    logger.info("=" * 70)
    logger.info("")

    measurer = SensorMeasurer()

    try:
        measurer.connect_camera()
        measurer.connect_robot()
        measurer.run()
    finally:
        measurer.cleanup()
        logger.info("Done")


if __name__ == "__main__":
    main()
