"""
Manual object detection with gravity mode.

Provides:
    - position_and_detect(): Reusable function (takes hardware as params)
    - cleanup_and_return(): Reusable cleanup function
    - main(): Standalone CLI entry point

Workflow:
1. Robot enters gravity mode (manually position it)
2. Live camera feed
3. Click on image to detect object boundary
4. Shows object position relative to robot base

Usage (standalone):
    python -m src.data_collection.scan_and_detect

Usage (imported):
    from src.data_collection.scan_and_detect import position_and_detect, cleanup_and_return
    target = position_and_detect(robot, camera, X)
    cleanup_and_return(robot, target_x, target_y, final_z)
"""

import time
from dataclasses import dataclass

import cv2
import numpy as np
import pyrealsense2 as rs
import trossen_arm

from src.robot.trossen_arm import TrossenArm
from src.sensors.realsense import RealSenseCamera
from src.utils.log import get_logger
from src.utils.transforms import compute_point_in_base_frame, load_calibration

logger = get_logger(__name__)


@dataclass
class ObjectTemplate:
    """Template for known object dimensions."""

    name: str
    width_mm: float  # X dimension
    depth_mm: float  # Y dimension
    height_mm: float  # Z dimension (vertical when standing)

    @property
    def width_m(self) -> float:
        return self.width_mm / 1000.0

    @property
    def depth_m(self) -> float:
        return self.depth_mm / 1000.0

    @property
    def height_m(self) -> float:
        return self.height_mm / 1000.0


# Known object template
OBJECT_TEMPLATE = ObjectTemplate(
    name="test_box",
    width_mm=38.0,
    depth_mm=38.0,
    height_mm=148.0,
)


class RealSenseCameraWithIntrinsics(RealSenseCamera):
    """RealSense camera with full intrinsics for 3D projection."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        super().__init__(width, height, fps)
        profile = self._pipeline.get_active_profile()
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        intr = color_profile.get_intrinsics()
        self.fx = intr.fx
        self.fy = intr.fy
        self.cx = intr.ppx
        self.cy = intr.ppy

    def pixel_to_3d(
        self, u: float, v: float, depth: float
    ) -> tuple[float, float, float]:
        """Convert pixel + depth to 3D point in camera frame (meters)."""
        z = depth
        x = (u - self.cx) * z / self.fx
        y = (v - self.cy) * z / self.fy
        return x, y, z


def detect_at_click(
    rgb: np.ndarray,
    depth: np.ndarray,
    click_x: int,
    click_y: int,
    search_radius: int = 50,
) -> tuple[np.ndarray | None, tuple[int, int] | None, float]:
    """
    Detect object boundary near click point.

    Returns:
        (contour, center, depth) or (None, None, 0.0)
    """
    h, w = rgb.shape[:2]

    # Create ROI around click
    x1 = max(0, click_x - search_radius)
    x2 = min(w, click_x + search_radius)
    y1 = max(0, click_y - search_radius)
    y2 = min(h, click_y + search_radius)

    # Edge detection on full image
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    edges = cv2.dilate(edges, kernel, iterations=2)

    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None, 0.0

    # Find contour closest to click point
    best_contour = None
    best_dist = float("inf")

    for c in contours:
        if cv2.contourArea(c) < 500:
            continue
        M = cv2.moments(c)
        if M["m00"] > 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            dist = np.sqrt((cx - click_x) ** 2 + (cy - click_y) ** 2)
            if dist < best_dist and dist < search_radius * 2:
                best_dist = dist
                best_contour = c

    if best_contour is None:
        return None, None, 0.0

    # Get center
    M = cv2.moments(best_contour)
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # Get depth at center
    r = 5
    y1, y2 = max(0, cy - r), min(h, cy + r)
    x1, x2 = max(0, cx - r), min(w, cx + r)
    depth_region = depth[y1:y2, x1:x2]
    valid_d = depth_region[depth_region > 0]

    center_depth = float(np.median(valid_d)) if len(valid_d) > 0 else 0.0

    return best_contour, (cx, cy), center_depth


# Global for mouse callback
click_point = None


def mouse_callback(event, x, y, flags, param):
    global click_point
    if event == cv2.EVENT_LBUTTONDOWN:
        click_point = (x, y)


# =============================================================================
# Reusable function: position_and_detect (takes hardware as parameters)
# =============================================================================


def position_and_detect(
    robot: TrossenArm,
    camera: RealSenseCameraWithIntrinsics,
    X: np.ndarray,
) -> tuple[float, float, float] | None:
    """
    Position robot using gravity mode and detect object.

    Args:
        robot: Connected TrossenArm
        camera: RealSenseCameraWithIntrinsics
        X: Eye-in-hand calibration matrix (4x4)

    Returns:
        (target_x, target_y, target_z) or None if cancelled
    """
    global click_point
    click_point = None

    detected_contour = None
    detected_center = None
    detected_depth = 0.0

    # Enable gravity mode
    logger.info("Enabling GRAVITY MODE - manually position the robot")
    robot._driver.set_arm_modes(trossen_arm.Mode.external_effort)
    robot._driver.set_gripper_mode(trossen_arm.Mode.external_effort)

    logger.info("Click on object to detect. Press 'g' to GO, 'q' to quit, 'r' to reset")

    while True:
        # Capture
        rgb, depth = camera.capture()
        viz = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # Get robot state
        ee_pose = robot.get_cartesian_position()
        gripper_pos = robot.get_gripper_opening()

        # Show robot state
        cv2.putText(
            viz,
            "GRAVITY MODE - Move robot manually",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2,
        )
        cv2.putText(
            viz,
            f"EE: x={ee_pose[0]:.3f} y={ee_pose[1]:.3f} z={ee_pose[2]:.3f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            viz,
            f"Gripper: {gripper_pos*1000:.1f}mm",
            (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        # Handle click
        if click_point is not None:
            detected_contour, detected_center, detected_depth = detect_at_click(
                rgb, depth, click_point[0], click_point[1]
            )
            click_point = None

        # Draw detection
        if detected_contour is not None:
            cv2.drawContours(viz, [detected_contour], -1, (0, 255, 0), 2)

            if detected_center and detected_depth > 0:
                cx, cy = detected_center
                cv2.drawMarker(
                    viz, detected_center, (0, 0, 255), cv2.MARKER_CROSS, 30, 3
                )

                x_cam, y_cam, z_cam = camera.pixel_to_3d(cx, cy, detected_depth)

                cv2.putText(
                    viz,
                    "DETECTED - Press 'g' to GO",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    viz,
                    f"Depth: {detected_depth*1000:.0f} mm",
                    (10, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 0),
                    1,
                )
        else:
            cv2.putText(
                viz,
                "Click on object to detect",
                (10, 120),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (200, 200, 200),
                1,
            )

        cv2.putText(
            viz,
            "g=GO  q=quit  r=reset",
            (10, viz.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (150, 150, 150),
            1,
        )

        cv2.imshow("Detection", viz)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            logger.info("User cancelled positioning")
            robot._driver.set_arm_modes(trossen_arm.Mode.position)
            robot._driver.set_gripper_mode(trossen_arm.Mode.position)
            return None

        elif key == ord("r"):
            detected_contour = None
            detected_center = None
            detected_depth = 0.0
            logger.info("Reset detection")

        elif key == ord("g") and detected_center and detected_depth > 0:
            # Exit gravity mode
            robot._driver.set_arm_modes(trossen_arm.Mode.position)
            robot._driver.set_gripper_mode(trossen_arm.Mode.position)

            # Compute target position
            cx, cy = detected_center
            x_cam, y_cam, z_cam = camera.pixel_to_3d(cx, cy, detected_depth)
            point_in_camera = np.array([x_cam, y_cam, z_cam])

            T_base_to_ee = robot.get_ee_pose()
            point_in_base = compute_point_in_base_frame(
                T_base_to_ee, X, point_in_camera
            )
            obj_x, obj_y, obj_z = point_in_base

            # Keep current Z, move to object X,Y
            current_z = ee_pose[2]
            target_x = obj_x
            target_y = obj_y
            target_z = current_z

            logger.info(
                f"Object in base: x={obj_x*1000:.1f} y={obj_y*1000:.1f} z={obj_z*1000:.1f} mm"
            )
            logger.info(f"Target: x={target_x:.3f} y={target_y:.3f} z={target_z:.3f}")
            logger.info(f"Object Z (for safe height calculation): {obj_z*1000:.1f}mm")

            # Open gripper and move to target
            robot.open_gripper(position=0.04)
            robot.move_to_cartesian(target_x, target_y, target_z, duration=2.0)
            logger.info("Moved to object position")

            # Return target position AND object Z for safe height calculation
            return (target_x, target_y, target_z, obj_z)


# =============================================================================
# Reusable function: cleanup_and_return
# =============================================================================


def cleanup_and_return(
    robot: TrossenArm,
    target_x: float,
    target_y: float,
    current_z: float,
    clearance_mm: float = 50.0,
) -> None:
    """
    Cleanup after sample: open gripper, move up, return home.

    Args:
        robot: Connected TrossenArm
        target_x: Current X position
        target_y: Current Y position
        current_z: Current Z position
        clearance_mm: How much to move up (default 50mm)
    """
    robot.open_gripper(position=0.04)
    logger.info("Gripper opened")

    safe_z = current_z + clearance_mm / 1000.0
    robot.move_to_cartesian(target_x, target_y, safe_z, duration=0.5)
    logger.info(f"Moved up to safe height: z={safe_z*1000:.1f}mm")

    robot.go_home()
    logger.info("Returned to home position")


# =============================================================================
# Standalone CLI entry point
# =============================================================================


def main():
    """Manual detection with gravity mode."""
    global click_point

    logger.info("=== Manual Object Detection ===")

    # Connect robot
    logger.info("Connecting to robot...")
    robot = TrossenArm()
    logger.info("Robot connected")

    # Connect camera
    logger.info("Connecting to RealSense...")
    camera = RealSenseCameraWithIntrinsics()
    logger.info("Camera connected")

    # Load calibration
    logger.info("Loading calibration...")
    calibration_dir = "dataset/calibration"
    X, _, _ = load_calibration(calibration_dir)
    logger.info("Calibration loaded")

    # Warmup
    for _ in range(10):
        camera.capture()

    # Setup window and mouse callback
    cv2.namedWindow("Detection")
    cv2.setMouseCallback("Detection", mouse_callback)

    try:
        # Go home first
        logger.info("Going to home position...")
        robot.go_home()

        # Enable gravity mode
        logger.info("Enabling GRAVITY MODE - manually position the robot")
        robot._driver.set_arm_modes(trossen_arm.Mode.external_effort)
        robot._driver.set_gripper_mode(trossen_arm.Mode.external_effort)

        detected_contour = None
        detected_center = None
        detected_depth = 0.0

        logger.info(
            "Click on object to detect. Press 'q' to quit, 'r' to reset, 'p' to lock position"
        )

        while True:
            # Capture
            rgb, depth = camera.capture()
            viz = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

            # Get robot state
            joints = robot.get_joint_positions()
            ee_pose = robot.get_cartesian_position()
            gripper_pos = robot.get_gripper_opening()

            # Show robot state
            cv2.putText(
                viz,
                "GRAVITY MODE - Move robot manually",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2,
            )
            cv2.putText(
                viz,
                f"EE: x={ee_pose[0]:.3f} y={ee_pose[1]:.3f} z={ee_pose[2]:.3f}",
                (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )
            cv2.putText(
                viz,
                f"Gripper: {gripper_pos*1000:.1f}mm",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 255),
                1,
            )

            # Handle click
            if click_point is not None:
                detected_contour, detected_center, detected_depth = detect_at_click(
                    rgb, depth, click_point[0], click_point[1]
                )
                click_point = None

            # Draw detection
            if detected_contour is not None:
                cv2.drawContours(viz, [detected_contour], -1, (0, 255, 0), 2)

                if detected_center and detected_depth > 0:
                    cx, cy = detected_center
                    cv2.drawMarker(
                        viz, detected_center, (0, 0, 255), cv2.MARKER_CROSS, 30, 3
                    )

                    # Get 3D position
                    x_cam, y_cam, z_cam = camera.pixel_to_3d(cx, cy, detected_depth)

                    # Show position
                    cv2.putText(
                        viz,
                        "DETECTED",
                        (10, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2,
                    )
                    cv2.putText(
                        viz,
                        f"Camera: x={x_cam*1000:.1f} y={y_cam*1000:.1f} z={z_cam*1000:.1f} mm",
                        (10, 150),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        1,
                    )
                    cv2.putText(
                        viz,
                        f"Depth: {detected_depth*1000:.0f} mm",
                        (10, 175),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        1,
                    )

                    # Template info
                    cv2.putText(
                        viz,
                        f"Template: {OBJECT_TEMPLATE.width_mm}x{OBJECT_TEMPLATE.height_mm}mm",
                        (10, 200),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 200, 100),
                        1,
                    )
            else:
                cv2.putText(
                    viz,
                    "Click on object to detect",
                    (10, 120),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 200, 200),
                    1,
                )

            # Instructions
            if detected_center and detected_depth > 0:
                cv2.putText(
                    viz,
                    "g=GO to object  q=quit  r=reset  p=lock",
                    (10, viz.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (150, 150, 150),
                    1,
                )
            else:
                cv2.putText(
                    viz,
                    "q=quit  r=reset  p=lock position",
                    (10, viz.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (150, 150, 150),
                    1,
                )

            cv2.imshow("Detection", viz)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord("r"):
                detected_contour = None
                detected_center = None
                detected_depth = 0.0
                logger.info("Reset detection")
            elif key == ord("p"):
                # Lock current position (exit gravity mode)
                logger.info("Locking position...")
                robot._driver.set_arm_modes(trossen_arm.Mode.position)
                robot._driver.set_gripper_mode(trossen_arm.Mode.position)
                current_joints = robot.get_joint_positions()
                robot.set_arm_positions(current_joints, duration=0.1)
                logger.info(f"Locked at: {ee_pose}")
            elif key == ord("g") and detected_center and detected_depth > 0:
                # GO to object - move robot so object is centered in gripper
                logger.info("Moving to object...")

                # Exit gravity mode first
                robot._driver.set_arm_modes(trossen_arm.Mode.position)
                robot._driver.set_gripper_mode(trossen_arm.Mode.position)

                # Get object position in camera frame
                cx, cy = detected_center
                x_cam, y_cam, z_cam = camera.pixel_to_3d(cx, cy, detected_depth)
                point_in_camera = np.array([x_cam, y_cam, z_cam])

                # Get EE pose as 4x4 transform matrix
                T_base_to_ee = robot.get_ee_pose()

                # Transform object position to robot base frame using calibration
                point_in_base = compute_point_in_base_frame(
                    T_base_to_ee, X, point_in_camera
                )
                obj_x, obj_y, obj_z = point_in_base

                # Keep current Z height - only move in X,Y to center gripper over object
                current_z = ee_pose[2]
                target_x = obj_x
                target_y = obj_y
                target_z = current_z  # Stay at current height

                logger.info(
                    f"Object in camera: x={x_cam*1000:.1f} y={y_cam*1000:.1f} z={z_cam*1000:.1f} mm"
                )
                logger.info(
                    f"Object in base: x={obj_x*1000:.1f} y={obj_y*1000:.1f} z={obj_z*1000:.1f} mm"
                )
                logger.info(
                    f"Current EE: x={ee_pose[0]:.3f} y={ee_pose[1]:.3f} z={ee_pose[2]:.3f}"
                )
                logger.info(
                    f"Target EE:  x={target_x:.3f} y={target_y:.3f} z={target_z:.3f} (keeping Z)"
                )

                # Open gripper first
                robot.open_gripper(position=0.04)

                # Move to target (object X,Y but current Z height)
                robot.move_to_cartesian(target_x, target_y, target_z, duration=2.0)
                logger.info("Moved to object position")

                # Touch sequence: 10 iterations, going down 1mm each time
                num_touches = 10
                step_down_mm = 1.0
                current_touch_z = target_z

                logger.info(
                    f"Starting touch sequence: {num_touches} touches, {step_down_mm}mm step"
                )

                for i in range(num_touches):
                    # Open gripper
                    robot.open_gripper(position=0.04)
                    logger.info(f"Touch {i+1}/{num_touches}: Gripper opened")

                    # Go down 1mm
                    current_touch_z -= step_down_mm / 1000.0  # Convert mm to meters
                    robot.move_to_cartesian(
                        target_x, target_y, current_touch_z, duration=0.5
                    )
                    logger.info(
                        f"Touch {i+1}/{num_touches}: Moved to z={current_touch_z*1000:.1f}mm"
                    )

                    # Close gripper (touch)
                    robot.close_gripper()
                    logger.info(f"Touch {i+1}/{num_touches}: Gripper closed (touch)")

                    # Small pause
                    time.sleep(0.2)

                logger.info("Touch sequence complete")

                # Open gripper at end
                robot.open_gripper(position=0.04)

                # Move up 50mm to clear object (avoid collision)
                safe_z = current_touch_z + 0.050  # 50mm up
                robot.move_to_cartesian(target_x, target_y, safe_z, duration=0.5)
                logger.info(f"Moved up to safe height: z={safe_z*1000:.1f}mm")

                # Go back to home position
                logger.info("Returning to home position...")
                robot.go_home()
                logger.info("Returned to home position")

                # Quit after completing sequence
                logger.info("Sequence complete. Exiting...")
                break

    finally:
        # Exit gravity mode
        robot._driver.set_arm_modes(trossen_arm.Mode.position)
        robot._driver.set_gripper_mode(trossen_arm.Mode.position)
        cv2.destroyAllWindows()
        robot.close()
        camera.close()


if __name__ == "__main__":
    main()
