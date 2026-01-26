#!/usr/bin/env python3
"""
Compute T_camera_to_gelsight(u) calibration from collected data.

This script:
1. Loads collected calibration data (images + gripper openings)
2. Detects gelsight sensor corners using contour detection
3. Computes 3D position using depth backprojection
4. Fits linear model: t(u) = t₀ + k·u

Usage:
    .venv/bin/python src/compute_gelsight_calibration.py
    .venv/bin/python src/compute_gelsight_calibration.py --visualize
    .venv/bin/python src/compute_gelsight_calibration.py --pose_name=test
"""

import cv2
import numpy as np
import json
import logging
import fire
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LinearRegression

# ============================================================================
# Logging Setup
# ============================================================================

LOG_DIR = Path(__file__).parent.parent.parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = (
    LOG_DIR / f"compute_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent / "gelsight_calibration_data"
CALIBRATION_OUTPUT_DIR = (
    Path(__file__).parent.parent.parent.parent / "dataset" / "calibration"
)


class GelsightDetector:
    """Detect gelsight sensors using template matching from manual selection.

    For first N samples, user manually clicks pad corners.
    Uses those selections to create templates for tracking all samples.
    """

    def __init__(self, pose_dir: Path = None):
        """Initialize detector."""
        self.left_ref_center = None
        self.right_ref_center = None
        self.left_template = None
        self.right_template = None
        self.pose_dir = pose_dir

        # For interactive selection
        self._click_points = []
        self._current_image = None

        # Try to load existing templates from data directory
        self._load_templates_from_disk()

    def _load_templates_from_disk(self):
        """Load templates from disk if they exist."""
        template_dir = DATA_DIR
        left_template_file = template_dir / "template_left.png"
        right_template_file = template_dir / "template_right.png"
        ref_centers_file = template_dir / "template_ref_centers.npy"

        if (
            left_template_file.exists()
            and right_template_file.exists()
            and ref_centers_file.exists()
        ):
            self.left_template = cv2.imread(
                str(left_template_file), cv2.IMREAD_GRAYSCALE
            )
            self.right_template = cv2.imread(
                str(right_template_file), cv2.IMREAD_GRAYSCALE
            )
            ref_centers = np.load(ref_centers_file)
            self.left_ref_center = np.array([ref_centers[0], ref_centers[1]])
            self.right_ref_center = np.array([ref_centers[2], ref_centers[3]])
            logger.info(f"Loaded existing templates from {template_dir}")
            logger.info(
                f"  LEFT:  {self.left_template.shape}, ref_center=({self.left_ref_center[0]:.0f}, {self.left_ref_center[1]:.0f})"
            )
            logger.info(
                f"  RIGHT: {self.right_template.shape}, ref_center=({self.right_ref_center[0]:.0f}, {self.right_ref_center[1]:.0f})"
            )

    def has_templates(self):
        """Check if templates are loaded."""
        return self.left_template is not None and self.right_template is not None

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse clicks during manual selection."""
        if event == cv2.EVENT_LBUTTONDOWN:
            self._click_points.append((x, y))
            # Draw point on image
            cv2.circle(self._current_image, (x, y), 5, (0, 255, 0), -1)
            cv2.putText(
                self._current_image,
                str(len(self._click_points)),
                (x + 5, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
            cv2.imshow("Select Corners", self._current_image)

    def collect_manual_selections(self, samples: list, num_samples: int = 3):
        """Collect manual corner selections from first N samples.

        Args:
            samples: List of sample dicts with rgb_file paths
            num_samples: Number of samples to manually select (default 3)

        Returns:
            List of selections, each with left/right corners and centers
        """
        selections = []

        cv2.namedWindow("Select Corners", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Select Corners", self._mouse_callback)

        print("\n" + "=" * 60)
        print("MANUAL CORNER SELECTION")
        print("=" * 60)
        print("For each image:")
        print("  1. Click 4 corners of LEFT sensor pad")
        print("  2. Click 4 corners of RIGHT sensor pad")
        print("  Press 'r' to reset, 'n' to skip, 'q' to quit")
        print("=" * 60)

        for i in range(min(num_samples, len(samples))):
            sample = samples[i]
            rgb_path = self.pose_dir / sample["rgb_file"]
            rgb_image = cv2.imread(str(rgb_path))

            if rgb_image is None:
                logger.warning(f"Could not load {rgb_path}")
                continue

            print(
                f"\nSample {i}: {sample['rgb_file']} (opening: {sample['actual_opening_mm']:.1f}mm)"
            )

            self._click_points = []
            self._current_image = rgb_image.copy()

            # Add instructions to image
            cv2.putText(
                self._current_image,
                f"Sample {i}: Click LEFT (1-4), then RIGHT (5-8)",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )
            cv2.imshow("Select Corners", self._current_image)

            while True:
                key = cv2.waitKey(100) & 0xFF

                if key == ord("q"):
                    cv2.destroyAllWindows()
                    return selections

                if key == ord("r"):
                    # Reset
                    self._click_points = []
                    self._current_image = rgb_image.copy()
                    cv2.putText(
                        self._current_image,
                        f"Sample {i}: Click LEFT (1-4), then RIGHT (5-8)",
                        (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2,
                    )
                    cv2.imshow("Select Corners", self._current_image)
                    print("  Reset points")

                if key == ord("n"):
                    # Skip this sample
                    print("  Skipped")
                    break

                if len(self._click_points) == 8:
                    # Got all 8 points
                    left_corners = np.array(self._click_points[:4])
                    right_corners = np.array(self._click_points[4:8])

                    left_center = np.mean(left_corners, axis=0)
                    right_center = np.mean(right_corners, axis=0)

                    selections.append(
                        {
                            "sample_idx": i,
                            "rgb_image": rgb_image,
                            "left_corners": left_corners,
                            "right_corners": right_corners,
                            "left_center": left_center,
                            "right_center": right_center,
                        }
                    )

                    print(
                        f"  LEFT center: ({left_center[0]:.0f}, {left_center[1]:.0f})"
                    )
                    print(
                        f"  RIGHT center: ({right_center[0]:.0f}, {right_center[1]:.0f})"
                    )
                    break

        cv2.destroyAllWindows()
        return selections

    def build_templates_from_selections(self, selections: list):
        """Build templates from manual selections and save to disk.

        Uses the first selection's corners to extract templates.
        Saves templates to src/calibration/gelsight_calibration/

        Args:
            selections: List of selection dicts from collect_manual_selections
        """
        if not selections:
            logger.error("No selections provided!")
            return

        # Use first selection for templates
        sel = selections[0]
        gray = cv2.cvtColor(sel["rgb_image"], cv2.COLOR_BGR2GRAY)

        # Extract LEFT template
        left_corners = sel["left_corners"].astype(int)
        lx_min, ly_min = left_corners.min(axis=0)
        lx_max, ly_max = left_corners.max(axis=0)
        pad = 5
        lx_min, ly_min = max(0, lx_min - pad), max(0, ly_min - pad)
        lx_max, ly_max = (
            min(gray.shape[1], lx_max + pad),
            min(gray.shape[0], ly_max + pad),
        )
        self.left_template = gray[ly_min:ly_max, lx_min:lx_max].copy()

        # Extract RIGHT template
        right_corners = sel["right_corners"].astype(int)
        rx_min, ry_min = right_corners.min(axis=0)
        rx_max, ry_max = right_corners.max(axis=0)
        rx_min, ry_min = max(0, rx_min - pad), max(0, ry_min - pad)
        rx_max, ry_max = (
            min(gray.shape[1], rx_max + pad),
            min(gray.shape[0], ry_max + pad),
        )
        self.right_template = gray[ry_min:ry_max, rx_min:rx_max].copy()

        # Use first selection's centers as reference
        self.left_ref_center = sel["left_center"]
        self.right_ref_center = sel["right_center"]

        # Save templates to gelsight_calibration_data directory
        template_dir = DATA_DIR
        template_dir.mkdir(parents=True, exist_ok=True)
        left_template_file = template_dir / "template_left.png"
        right_template_file = template_dir / "template_right.png"
        cv2.imwrite(str(left_template_file), self.left_template)
        cv2.imwrite(str(right_template_file), self.right_template)

        # Save reference centers
        ref_centers = np.array(
            [
                self.left_ref_center[0],
                self.left_ref_center[1],
                self.right_ref_center[0],
                self.right_ref_center[1],
            ]
        )
        ref_centers_file = template_dir / "template_ref_centers.npy"
        np.save(ref_centers_file, ref_centers)

        logger.info("Built and saved templates:")
        logger.info(f"  LEFT:  {self.left_template.shape} -> {left_template_file}")
        logger.info(f"  RIGHT: {self.right_template.shape} -> {right_template_file}")
        logger.info(f"  Ref centers: {ref_centers_file}")

    def get_depth_at_region(
        self, depth_image: np.ndarray, center: np.ndarray, radius: int = 15
    ):
        """Get median valid depth in a region around center point."""
        h, w = depth_image.shape
        u, v = int(center[0]), int(center[1])

        u_min = max(0, u - radius)
        u_max = min(w, u + radius)
        v_min = max(0, v - radius)
        v_max = min(h, v + radius)

        region = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = region[region > 0]

        if len(valid_depths) == 0:
            return None
        return float(np.median(valid_depths))

    def detect_both_sensors(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray = None,
        visualize: bool = False,
    ):
        """
        Detect both LEFT and RIGHT gelsight sensors using template matching.

        Uses templates extracted from manual selection to track sensors across frames.

        Returns:
            dict with 'left' and 'right' keys, each containing:
                - corners: 4x2 array or None
                - center: (u, v) or None
        """
        result = {
            "left": {"corners": None, "center": None},
            "right": {"corners": None, "center": None},
        }

        # If we have templates, use template matching
        if self.left_template is not None and self.right_template is not None:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
            h, w = gray.shape

            # Template match for LEFT sensor
            # Search in left half of image with some margin
            search_margin = 100
            left_search_x1 = max(0, int(self.left_ref_center[0]) - search_margin)
            left_search_x2 = min(w, int(self.left_ref_center[0]) + search_margin)
            left_search_y1 = max(0, int(self.left_ref_center[1]) - search_margin)
            left_search_y2 = min(h, int(self.left_ref_center[1]) + search_margin)

            left_search_region = gray[
                left_search_y1:left_search_y2, left_search_x1:left_search_x2
            ]
            if (
                left_search_region.shape[0] > self.left_template.shape[0]
                and left_search_region.shape[1] > self.left_template.shape[1]
            ):
                left_match = cv2.matchTemplate(
                    left_search_region, self.left_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val_l, _, max_loc_l = cv2.minMaxLoc(left_match)

                if max_val_l > 0.5:  # Threshold for good match
                    th, tw = self.left_template.shape
                    left_center_x = left_search_x1 + max_loc_l[0] + tw // 2
                    left_center_y = left_search_y1 + max_loc_l[1] + th // 2
                    result["left"]["center"] = np.array(
                        [left_center_x, left_center_y], dtype=np.float64
                    )
                    # Create corners using actual template size
                    half_w, half_h = tw // 2, th // 2
                    result["left"]["corners"] = np.array(
                        [
                            [left_center_x - half_w, left_center_y - half_h],
                            [left_center_x + half_w, left_center_y - half_h],
                            [left_center_x + half_w, left_center_y + half_h],
                            [left_center_x - half_w, left_center_y + half_h],
                        ],
                        dtype=np.int32,
                    )

            # Template match for RIGHT sensor
            right_search_x1 = max(0, int(self.right_ref_center[0]) - search_margin)
            right_search_x2 = min(w, int(self.right_ref_center[0]) + search_margin)
            right_search_y1 = max(0, int(self.right_ref_center[1]) - search_margin)
            right_search_y2 = min(h, int(self.right_ref_center[1]) + search_margin)

            right_search_region = gray[
                right_search_y1:right_search_y2, right_search_x1:right_search_x2
            ]
            if (
                right_search_region.shape[0] > self.right_template.shape[0]
                and right_search_region.shape[1] > self.right_template.shape[1]
            ):
                right_match = cv2.matchTemplate(
                    right_search_region, self.right_template, cv2.TM_CCOEFF_NORMED
                )
                _, max_val_r, _, max_loc_r = cv2.minMaxLoc(right_match)

                if max_val_r > 0.5:  # Threshold for good match
                    th, tw = self.right_template.shape
                    right_center_x = right_search_x1 + max_loc_r[0] + tw // 2
                    right_center_y = right_search_y1 + max_loc_r[1] + th // 2
                    result["right"]["center"] = np.array(
                        [right_center_x, right_center_y], dtype=np.float64
                    )
                    # Create corners using actual template size
                    half_w, half_h = tw // 2, th // 2
                    result["right"]["corners"] = np.array(
                        [
                            [right_center_x - half_w, right_center_y - half_h],
                            [right_center_x + half_w, right_center_y - half_h],
                            [right_center_x + half_w, right_center_y + half_h],
                            [right_center_x - half_w, right_center_y + half_h],
                        ],
                        dtype=np.int32,
                    )

            if visualize:
                display = rgb_image.copy()
                if result["left"]["center"] is not None:
                    center = result["left"]["center"].astype(int)
                    # Use actual template size for visualization
                    th, tw = self.left_template.shape
                    half_w, half_h = tw // 2, th // 2
                    cv2.rectangle(
                        display,
                        (center[0] - half_w, center[1] - half_h),
                        (center[0] + half_w, center[1] + half_h),
                        (0, 0, 255),
                        2,
                    )
                    cv2.circle(display, tuple(center), 5, (0, 255, 0), -1)
                    cv2.putText(
                        display,
                        "LEFT",
                        (center[0] - 20, center[1] - half_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 0, 255),
                        2,
                    )
                if result["right"]["center"] is not None:
                    center = result["right"]["center"].astype(int)
                    # Use actual template size for visualization
                    th, tw = self.right_template.shape
                    half_w, half_h = tw // 2, th // 2
                    cv2.rectangle(
                        display,
                        (center[0] - half_w, center[1] - half_h),
                        (center[0] + half_w, center[1] + half_h),
                        (255, 0, 0),
                        2,
                    )
                    cv2.circle(display, tuple(center), 5, (0, 255, 0), -1)
                    cv2.putText(
                        display,
                        "RIGHT",
                        (center[0] - 25, center[1] - half_h - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 0, 0),
                        2,
                    )
                cv2.imshow("Detection (Template)", display)
                cv2.waitKey(100)

            return result

        # No templates available - manual selection required
        logger.warning("No manual selection found - run manual_selection.py first")
        return result

    def detect_corners(
        self,
        rgb_image: np.ndarray,
        depth_image: np.ndarray = None,
        visualize: bool = False,
    ):
        """
        Detect gelsight sensor corners - returns the midpoint between both sensors.

        This method detects both sensors and returns the combined result
        for backwards compatibility.

        Returns:
            corners: Combined 8x2 array (4 from each sensor) or None
            center: Midpoint between both sensor centers or None
        """
        results = self.detect_both_sensors(
            rgb_image, depth_image=depth_image, visualize=visualize
        )

        left = results["left"]
        right = results["right"]

        # If both detected, return combined result
        if left["corners"] is not None and right["corners"] is not None:
            # Stack corners from both sensors
            corners = np.vstack([left["corners"], right["corners"]])
            # Center is midpoint between both sensor centers
            center = (left["center"] + right["center"]) / 2
            return corners, center

        # If only one detected, return that one
        if left["corners"] is not None:
            return left["corners"], left["center"]
        if right["corners"] is not None:
            return right["corners"], right["center"]

        return None, None


class GelsightCalibrator:
    """Compute T_camera_to_gelsight(u) calibration."""

    def __init__(self, pose_name: str = None, pose_dir: Path = None):
        """Initialize calibrator.

        Args:
            pose_name: Name of pose subdirectory (optional)
            pose_dir: Direct path to data directory (optional, overrides pose_name)
        """
        if pose_dir is not None:
            self.pose_dir = pose_dir
            self.pose_name = pose_dir.name
        elif pose_name is not None:
            self.pose_name = pose_name
            self.pose_dir = DATA_DIR / pose_name
        else:
            # Check if data exists directly in DATA_DIR
            if (DATA_DIR / "calibration_data.json").exists():
                self.pose_dir = DATA_DIR
                self.pose_name = "default"
            else:
                raise ValueError(
                    "No pose_name or pose_dir provided and no data in DATA_DIR"
                )

        self.detector = GelsightDetector(pose_dir=self.pose_dir)
        self.intrinsics = None
        self.data = None

    def load_data(self):
        """Load calibration data."""
        data_file = self.pose_dir / "calibration_data.json"
        if not data_file.exists():
            raise FileNotFoundError(f"Calibration data not found: {data_file}")

        with open(data_file) as f:
            self.data = json.load(f)

        self.intrinsics = self.data["camera_intrinsics"]
        logger.info(f"Loaded {self.data['num_samples']} samples from {self.pose_dir}")
        logger.info(
            f"Camera intrinsics: fx={self.intrinsics['fx']:.2f}, fy={self.intrinsics['fy']:.2f}"
        )

    def backproject_to_3d(self, pixel_uv: np.ndarray, depth_image: np.ndarray):
        """
        Backproject pixel coordinates to 3D point using depth.

        Args:
            pixel_uv: (u, v) pixel coordinates
            depth_image: Depth image (uint16, mm values)

        Returns:
            point_3d: (X, Y, Z) in camera frame (meters)
        """
        u, v = int(pixel_uv[0]), int(pixel_uv[1])

        # Clamp to valid bounds
        h, w = depth_image.shape
        u = max(0, min(u, w - 1))
        v = max(0, min(v, h - 1))

        depth = depth_image[v, u]
        if depth == 0:
            return None

        depth_m = depth / 1000.0

        fx = self.intrinsics["fx"]
        fy = self.intrinsics["fy"]
        cx = self.intrinsics["ppx"]
        cy = self.intrinsics["ppy"]

        X = depth_m * (u - cx) / fx
        Y = depth_m * (v - cy) / fy
        Z = depth_m

        return np.array([X, Y, Z])

    def compute_transform_from_corners(self, corners_3d: np.ndarray):
        """
        Compute 4x4 transform from 4 corner points.

        Args:
            corners_3d: 4x3 array of 3D corner points

        Returns:
            T: 4x4 transformation matrix (T_camera_to_gelsight)
        """
        center = np.mean(corners_3d, axis=0)

        v1 = corners_3d[1] - corners_3d[0]
        v2 = corners_3d[3] - corners_3d[0]

        z_axis = np.cross(v1, v2)
        z_axis = z_axis / np.linalg.norm(z_axis)

        if z_axis[2] < 0:
            z_axis = -z_axis

        x_axis = v1 / np.linalg.norm(v1)
        y_axis = np.cross(z_axis, x_axis)
        y_axis = y_axis / np.linalg.norm(y_axis)
        x_axis = np.cross(y_axis, z_axis)

        R = np.column_stack([x_axis, y_axis, z_axis])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = center

        return T

    def get_depth_at_region(
        self, center: np.ndarray, depth_image: np.ndarray, radius: int = 10
    ):
        """
        Get median valid depth in a region around center point.

        Args:
            center: (u, v) center pixel coordinates
            depth_image: Depth image (uint16, mm values)
            radius: Search radius in pixels

        Returns:
            depth_m: Median depth in meters, or None if no valid depth
        """
        h, w = depth_image.shape
        u, v = int(center[0]), int(center[1])

        # Get region bounds
        u_min = max(0, u - radius)
        u_max = min(w, u + radius)
        v_min = max(0, v - radius)
        v_max = min(h, v + radius)

        region = depth_image[v_min:v_max, u_min:u_max]
        valid_depths = region[region > 0]

        if len(valid_depths) == 0:
            return None

        return float(np.median(valid_depths)) / 1000.0

    def process_samples(self, visualize: bool = False):
        """
        Process all samples and extract gelsight positions.

        Detects both LEFT and RIGHT gelsight sensors using template matching.

        Returns:
            results: List of dicts with gripper_opening and gelsight_position
        """
        results = []

        logger.info("Processing samples with template matching + depth backprojection")

        if visualize:
            cv2.namedWindow("Detection left", cv2.WINDOW_NORMAL)
            cv2.namedWindow("Detection right", cv2.WINDOW_NORMAL)

        for sample in self.data["data"]:
            rgb_path = self.pose_dir / sample["rgb_file"]
            depth_path = self.pose_dir / sample["depth_file"]

            if not rgb_path.exists() or not depth_path.exists():
                logger.warning(f"Missing files for sample {sample['sample_index']}")
                continue

            rgb_image = cv2.imread(str(rgb_path))
            depth_image = np.load(str(depth_path))

            # Detect both sensors (using depth filtering)
            detection = self.detector.detect_both_sensors(
                rgb_image, depth_image=depth_image, visualize=visualize
            )
            left = detection["left"]
            right = detection["right"]

            left_detected = left["center"] is not None
            right_detected = right["center"] is not None

            if not left_detected and not right_detected:
                logger.warning(
                    f"Sample {sample['sample_index']}: Neither sensor detected"
                )
                continue

            # Get 3D positions using center points (more robust for depth)
            left_3d = None
            right_3d = None

            if left_detected:
                # Try center point first, then search region
                left_3d = self.backproject_to_3d(left["center"], depth_image)
                if left_3d is None:
                    depth_m = self.get_depth_at_region(left["center"], depth_image)
                    if depth_m is not None:
                        fx = self.intrinsics["fx"]
                        fy = self.intrinsics["fy"]
                        cx = self.intrinsics["ppx"]
                        cy = self.intrinsics["ppy"]
                        u, v = left["center"]
                        X = depth_m * (u - cx) / fx
                        Y = depth_m * (v - cy) / fy
                        left_3d = np.array([X, Y, depth_m])

            if right_detected:
                right_3d = self.backproject_to_3d(right["center"], depth_image)
                if right_3d is None:
                    depth_m = self.get_depth_at_region(right["center"], depth_image)
                    if depth_m is not None:
                        fx = self.intrinsics["fx"]
                        fy = self.intrinsics["fy"]
                        cx = self.intrinsics["ppx"]
                        cy = self.intrinsics["ppy"]
                        u, v = right["center"]
                        X = depth_m * (u - cx) / fx
                        Y = depth_m * (v - cy) / fy
                        right_3d = np.array([X, Y, depth_m])

            # Compute gripper midpoint in 3D
            if left_3d is not None and right_3d is not None:
                center_3d = (left_3d + right_3d) / 2
                status = "both"
            elif left_3d is not None:
                center_3d = left_3d
                status = "left_only"
            elif right_3d is not None:
                center_3d = right_3d
                status = "right_only"
            else:
                logger.warning(
                    f"Sample {sample['sample_index']}: No valid depth for detected sensors"
                )
                continue

            # Build a simple transform (translation only for now)
            T = np.eye(4)
            T[:3, 3] = center_3d

            results.append(
                {
                    "sample_index": sample["sample_index"],
                    "gripper_opening_m": sample["actual_opening_m"],
                    "gripper_opening_mm": sample["actual_opening_mm"],
                    "center_3d": center_3d.tolist(),
                    "T_camera_to_gelsight": T.tolist(),
                    "detection_status": status,
                    "left_center_2d": left["center"].tolist()
                    if left["center"] is not None
                    else None,
                    "right_center_2d": right["center"].tolist()
                    if right["center"] is not None
                    else None,
                    "left_3d": left_3d.tolist() if left_3d is not None else None,
                    "right_3d": right_3d.tolist() if right_3d is not None else None,
                }
            )

            logger.info(
                f"Sample {sample['sample_index']}: u={sample['actual_opening_mm']:.1f}mm, "
                f"center=[{center_3d[0] * 1000:.1f}, {center_3d[1] * 1000:.1f}, {center_3d[2] * 1000:.1f}]mm ({status})"
            )

        if visualize:
            cv2.destroyAllWindows()

        return results

    def fit_linear_model_for_sensor(self, results: list, sensor: str):
        """
        Fit linear model for a single sensor: t(u) = t₀ + k·u

        Args:
            results: List of sample results
            sensor: 'left' or 'right'

        Returns:
            model: Dict with t0, k, r_squared for each axis, or None if insufficient data
        """
        # Filter samples that have valid 3D data for this sensor
        key = f"{sensor}_3d"
        valid_results = [r for r in results if r.get(key) is not None]

        if len(valid_results) < 2:
            logger.warning(
                f"Insufficient data for {sensor} sensor (only {len(valid_results)} samples)"
            )
            return None

        u_values = np.array([r["gripper_opening_m"] for r in valid_results]).reshape(
            -1, 1
        )
        positions = np.array([r[key] for r in valid_results])

        models = {}
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            y = positions[:, axis_idx]

            reg = LinearRegression()
            reg.fit(u_values, y)

            t0 = reg.intercept_
            k = reg.coef_[0]
            r_squared = reg.score(u_values, y)

            models[axis_name] = {
                "t0": float(t0),
                "k": float(k),
                "r_squared": float(r_squared),
            }

        return {
            "models": models,
            "num_samples": len(valid_results),
            "u_min": float(u_values.min()),
            "u_max": float(u_values.max()),
        }

    def fit_linear_model(self, results: list):
        """
        Fit linear models separately for LEFT and RIGHT sensors.

        Returns:
            model: Dict with separate models for 'left', 'right', and 'center'
        """
        if len(results) < 2:
            raise ValueError("Need at least 2 samples to fit linear model")

        # Fit model for LEFT sensor
        logger.info("\n" + "=" * 50)
        logger.info("LEFT SENSOR - Linear Model")
        logger.info("=" * 50)
        left_model = self.fit_linear_model_for_sensor(results, "left")
        if left_model:
            models = left_model["models"]
            logger.info(f"  Samples: {left_model['num_samples']}")
            for axis_name in ["x", "y", "z"]:
                m = models[axis_name]
                logger.info(
                    f"  {axis_name}: t₀={m['t0'] * 1000:.3f}mm, k={m['k'] * 1000:.3f}mm/m, R²={m['r_squared']:.4f}"
                )

        # Fit model for RIGHT sensor
        logger.info("\n" + "=" * 50)
        logger.info("RIGHT SENSOR - Linear Model")
        logger.info("=" * 50)
        right_model = self.fit_linear_model_for_sensor(results, "right")
        if right_model:
            models = right_model["models"]
            logger.info(f"  Samples: {right_model['num_samples']}")
            for axis_name in ["x", "y", "z"]:
                m = models[axis_name]
                logger.info(
                    f"  {axis_name}: t₀={m['t0'] * 1000:.3f}mm, k={m['k'] * 1000:.3f}mm/m, R²={m['r_squared']:.4f}"
                )

        # Fit model for CENTER (midpoint) - for backwards compatibility
        logger.info("\n" + "=" * 50)
        logger.info("CENTER (MIDPOINT) - Linear Model")
        logger.info("=" * 50)
        u_values = np.array([r["gripper_opening_m"] for r in results]).reshape(-1, 1)
        positions = np.array([r["center_3d"] for r in results])

        center_models = {}
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            y = positions[:, axis_idx]
            reg = LinearRegression()
            reg.fit(u_values, y)
            t0 = reg.intercept_
            k = reg.coef_[0]
            r_squared = reg.score(u_values, y)
            center_models[axis_name] = {
                "t0": float(t0),
                "k": float(k),
                "r_squared": float(r_squared),
            }
            logger.info(
                f"  {axis_name}: t₀={t0 * 1000:.3f}mm, k={k * 1000:.3f}mm/m, R²={r_squared:.4f}"
            )

        center_model = {
            "models": center_models,
            "num_samples": len(results),
            "u_min": float(u_values.min()),
            "u_max": float(u_values.max()),
        }

        # Summary
        logger.info("\n" + "=" * 50)
        logger.info("CALIBRATION QUALITY SUMMARY")
        logger.info("=" * 50)
        if left_model:
            avg_r2_left = np.mean(
                [left_model["models"][a]["r_squared"] for a in ["x", "y", "z"]]
            )
            logger.info(f"  LEFT  sensor avg R²: {avg_r2_left:.4f}")
        if right_model:
            avg_r2_right = np.mean(
                [right_model["models"][a]["r_squared"] for a in ["x", "y", "z"]]
            )
            logger.info(f"  RIGHT sensor avg R²: {avg_r2_right:.4f}")
        avg_r2_center = np.mean(
            [center_models[a]["r_squared"] for a in ["x", "y", "z"]]
        )
        logger.info(f"  CENTER (midpoint) avg R²: {avg_r2_center:.4f}")

        return {
            "left": left_model,
            "right": right_model,
            "center": center_model,
        }

    def run(self, visualize: bool = False, num_manual_samples: int = 1):
        """Run full calibration pipeline.

        Args:
            visualize: Show detection visualization during processing
            num_manual_samples: Number of samples to manually select (default 1)
        """
        self.load_data()

        # Step 1: Check if templates exist, otherwise collect manual selections
        if self.detector.has_templates():
            logger.info("\nUsing existing templates from disk")
        else:
            logger.info(
                f"\nNo templates found. Collecting manual selections from {num_manual_samples} sample(s)..."
            )
            selections = self.detector.collect_manual_selections(
                self.data["data"], num_samples=num_manual_samples
            )

            if not selections:
                logger.error("No manual selections made! Cannot proceed.")
                return None

            # Build templates from selections (saves to disk)
            self.detector.build_templates_from_selections(selections)

        # Step 3: Process all samples using template matching
        results = self.process_samples(visualize=visualize)

        if len(results) == 0:
            logger.error("No samples were successfully processed!")
            return None

        logger.info(f"\nProcessed {len(results)} samples successfully")

        models = self.fit_linear_model(results)

        output = {
            "pose_name": self.pose_name,
            "detection_method": "template_matching",
            "camera_intrinsics": self.intrinsics,
            "linear_model_left": models["left"],
            "linear_model_right": models["right"],
            "linear_model_center": models["center"],
            "samples": results,
            "calibration_time": datetime.now().isoformat(),
        }

        # Save T(u) params as npy files to dataset/calibration/
        CALIBRATION_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

        # LEFT sensor: T(u) = t0 + k*u, save as [t0_x, t0_y, t0_z, k_x, k_y, k_z]
        if models["left"]:
            left_m = models["left"]["models"]
            left_params = np.array(
                [
                    left_m["x"]["t0"],
                    left_m["y"]["t0"],
                    left_m["z"]["t0"],
                    left_m["x"]["k"],
                    left_m["y"]["k"],
                    left_m["z"]["k"],
                ]
            )
            left_file = CALIBRATION_OUTPUT_DIR / "T_u_left_params.npy"
            np.save(left_file, left_params)
            logger.info(f"\nSaved LEFT params to: {left_file}")
            logger.info(
                f"  t0: [{left_params[0] * 1000:.3f}, {left_params[1] * 1000:.3f}, {left_params[2] * 1000:.3f}] mm"
            )
            logger.info(
                f"  k:  [{left_params[3] * 1000:.3f}, {left_params[4] * 1000:.3f}, {left_params[5] * 1000:.3f}] mm/m"
            )

        # RIGHT sensor
        if models["right"]:
            right_m = models["right"]["models"]
            right_params = np.array(
                [
                    right_m["x"]["t0"],
                    right_m["y"]["t0"],
                    right_m["z"]["t0"],
                    right_m["x"]["k"],
                    right_m["y"]["k"],
                    right_m["z"]["k"],
                ]
            )
            right_file = CALIBRATION_OUTPUT_DIR / "T_u_right_params.npy"
            np.save(right_file, right_params)
            logger.info(f"Saved RIGHT params to: {right_file}")
            logger.info(
                f"  t0: [{right_params[0] * 1000:.3f}, {right_params[1] * 1000:.3f}, {right_params[2] * 1000:.3f}] mm"
            )
            logger.info(
                f"  k:  [{right_params[3] * 1000:.3f}, {right_params[4] * 1000:.3f}, {right_params[5] * 1000:.3f}] mm/m"
            )

        return output


def calibrate(pose_name: str = None, visualize: bool = False, num_manual: int = 1):
    """
    Compute T_camera_to_gelsight(u) calibration.

    Args:
        pose_name: Name of the pose/session to process
        visualize: Show detection visualization
        num_manual: Number of samples to manually select (default 3)
    """
    logger.info("=" * 70)
    logger.info("Gelsight Calibration Computation")
    logger.info("=" * 70)
    logger.info(f"Pose: {pose_name}")
    logger.info(f"Method: manual selection ({num_manual} samples) + template matching")
    logger.info("")

    calibrator = GelsightCalibrator(pose_name=pose_name)

    try:
        result = calibrator.run(visualize=visualize, num_manual_samples=num_manual)
        if result:
            logger.info("")
            logger.info("=" * 70)
            logger.info("Calibration complete!")
            logger.info("=" * 70)
    except Exception as e:
        logger.error(f"Calibration failed: {e}")
        import traceback

        traceback.print_exc()


def preview(pose_name: str = "home", sample_idx: int = 0):
    """
    Preview a single sample.

    Args:
        pose_name: Name of the pose/session
        sample_idx: Index of sample to preview
    """
    pose_dir = DATA_DIR / pose_name
    data_file = pose_dir / "calibration_data.json"

    with open(data_file) as f:
        data = json.load(f)

    if sample_idx >= len(data["data"]):
        logger.error(
            f"Sample index {sample_idx} out of range (max: {len(data['data']) - 1})"
        )
        return

    sample = data["data"][sample_idx]
    rgb_path = pose_dir / sample["rgb_file"]
    depth_path = pose_dir / sample["depth_file"]

    rgb_image = cv2.imread(str(rgb_path))
    depth_image = np.load(str(depth_path))

    logger.info(f"Sample {sample_idx}:")
    logger.info(f"  Gripper opening: {sample['actual_opening_mm']:.1f}mm")
    logger.info(f"  RGB shape: {rgb_image.shape}")
    logger.info(f"  Depth range: [{depth_image.min()}, {depth_image.max()}]")

    cv2.namedWindow("RGB", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Depth", cv2.WINDOW_NORMAL)

    depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(
        np.uint8
    )
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("RGB", rgb_image)
    cv2.imshow("Depth", depth_vis)

    logger.info("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire.Fire(
        {
            "calibrate": calibrate,
            "preview": preview,
        }
    )
