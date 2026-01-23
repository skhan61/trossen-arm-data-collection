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

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"compute_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "gelsight_calibration_data"


class GelsightDetector:
    """Detect gelsight sensors using color-based detection.

    LEFT sensor: BLACK colored
    RIGHT sensor: WHITE colored
    """

    def detect_both_sensors(self, rgb_image: np.ndarray, visualize: bool = False):
        """
        Detect both LEFT and RIGHT gelsight sensors.

        Both sensors appear as dark rectangles with colorful reflections.
        We detect all dark rectangular regions and assign left/right by X position.

        Returns:
            dict with 'left' and 'right' keys, each containing:
                - corners: 4x2 array or None
                - center: (u, v) or None
        """
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Both sensors are dark - use same threshold for both
        _, mask = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to clean up mask
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Find contours in the mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Filter by area and aspect ratio to find GelSight-sized rectangles
        valid_detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 500 or area > 50000:  # Filter by reasonable area
                continue
            rect = cv2.minAreaRect(cnt)
            w, h = rect[1]
            if w == 0 or h == 0:
                continue
            aspect = max(w, h) / min(w, h)
            if aspect > 4:  # GelSight sensors are roughly rectangular
                continue

            corners = cv2.boxPoints(rect)
            corners = np.int32(corners)
            center = np.mean(corners, axis=0)
            valid_detections.append({
                "corners": corners,
                "center": center,
                "area": area
            })

        # Sort by X coordinate (left to right)
        valid_detections.sort(key=lambda d: d["center"][0])

        result = {
            "left": {"corners": None, "center": None},
            "right": {"corners": None, "center": None}
        }

        if len(valid_detections) >= 2:
            # Take the two largest detections, then assign by X position
            valid_detections.sort(key=lambda d: d["area"], reverse=True)
            top_two = valid_detections[:2]
            top_two.sort(key=lambda d: d["center"][0])  # Sort by X

            result["left"]["corners"] = top_two[0]["corners"]
            result["left"]["center"] = top_two[0]["center"]
            result["right"]["corners"] = top_two[1]["corners"]
            result["right"]["center"] = top_two[1]["center"]
        elif len(valid_detections) == 1:
            # Only one detected - determine if left or right based on image center
            img_center_x = rgb_image.shape[1] / 2
            det = valid_detections[0]
            if det["center"][0] < img_center_x:
                result["left"]["corners"] = det["corners"]
                result["left"]["center"] = det["center"]
            else:
                result["right"]["corners"] = det["corners"]
                result["right"]["center"] = det["center"]

        if visualize:
            display = rgb_image.copy()
            cv2.imshow("Mask", mask)

            if result["left"]["center"] is not None:
                corners = result["left"]["corners"]
                center = result["left"]["center"]
                for corner in corners:
                    cv2.circle(display, tuple(corner), 5, (0, 0, 255), -1)
                cv2.circle(display, tuple(center.astype(int)), 8, (0, 255, 0), -1)
                cv2.drawContours(display, [corners], 0, (0, 0, 255), 2)
                cv2.putText(display, "LEFT", (int(center[0])-20, int(center[1])-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

            if result["right"]["center"] is not None:
                corners = result["right"]["corners"]
                center = result["right"]["center"]
                for corner in corners:
                    cv2.circle(display, tuple(corner), 5, (255, 0, 0), -1)
                cv2.circle(display, tuple(center.astype(int)), 8, (0, 255, 0), -1)
                cv2.drawContours(display, [corners], 0, (255, 0, 0), 2)
                cv2.putText(display, "RIGHT", (int(center[0])-25, int(center[1])-20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            cv2.imshow("Detection", display)
            cv2.waitKey(500)

        return result

    def detect_corners(self, rgb_image: np.ndarray, visualize: bool = False):
        """
        Detect gelsight sensor corners - returns the midpoint between both sensors.

        This method detects both sensors and returns the combined result
        for backwards compatibility.

        Returns:
            corners: Combined 8x2 array (4 from each sensor) or None
            center: Midpoint between both sensor centers or None
        """
        results = self.detect_both_sensors(rgb_image, visualize)

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

    def __init__(self, pose_name: str):
        self.pose_name = pose_name
        self.pose_dir = DATA_DIR / pose_name
        self.detector = GelsightDetector()
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
        logger.info(f"Camera intrinsics: fx={self.intrinsics['fx']:.2f}, fy={self.intrinsics['fy']:.2f}")

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

    def get_depth_at_region(self, center: np.ndarray, depth_image: np.ndarray, radius: int = 10):
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

        Detects both LEFT (black) and RIGHT (white) gelsight sensors.

        Returns:
            results: List of dicts with gripper_opening and gelsight_position
        """
        results = []

        logger.info("Processing samples with color-based detection + depth backprojection")
        logger.info("  LEFT sensor: BLACK | RIGHT sensor: WHITE")

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

            # Detect both sensors
            detection = self.detector.detect_both_sensors(rgb_image, visualize=visualize)
            left = detection["left"]
            right = detection["right"]

            left_detected = left["center"] is not None
            right_detected = right["center"] is not None

            if not left_detected and not right_detected:
                logger.warning(f"Sample {sample['sample_index']}: Neither sensor detected")
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
                logger.warning(f"Sample {sample['sample_index']}: No valid depth for detected sensors")
                continue

            # Build a simple transform (translation only for now)
            T = np.eye(4)
            T[:3, 3] = center_3d

            results.append({
                "sample_index": sample["sample_index"],
                "gripper_opening_m": sample["actual_opening_m"],
                "gripper_opening_mm": sample["actual_opening_mm"],
                "center_3d": center_3d.tolist(),
                "T_camera_to_gelsight": T.tolist(),
                "detection_status": status,
                "left_center_2d": left["center"].tolist() if left["center"] is not None else None,
                "right_center_2d": right["center"].tolist() if right["center"] is not None else None,
                "left_3d": left_3d.tolist() if left_3d is not None else None,
                "right_3d": right_3d.tolist() if right_3d is not None else None,
            })

            logger.info(
                f"Sample {sample['sample_index']}: u={sample['actual_opening_mm']:.1f}mm, "
                f"center=[{center_3d[0]*1000:.1f}, {center_3d[1]*1000:.1f}, {center_3d[2]*1000:.1f}]mm ({status})"
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
            logger.warning(f"Insufficient data for {sensor} sensor (only {len(valid_results)} samples)")
            return None

        u_values = np.array([r["gripper_opening_m"] for r in valid_results]).reshape(-1, 1)
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
        logger.info("LEFT SENSOR (BLACK) - Linear Model")
        logger.info("=" * 50)
        left_model = self.fit_linear_model_for_sensor(results, "left")
        if left_model:
            models = left_model["models"]
            logger.info(f"  Samples: {left_model['num_samples']}")
            for axis_name in ["x", "y", "z"]:
                m = models[axis_name]
                logger.info(f"  {axis_name}: t₀={m['t0']*1000:.3f}mm, k={m['k']*1000:.3f}mm/m, R²={m['r_squared']:.4f}")

        # Fit model for RIGHT sensor
        logger.info("\n" + "=" * 50)
        logger.info("RIGHT SENSOR (WHITE) - Linear Model")
        logger.info("=" * 50)
        right_model = self.fit_linear_model_for_sensor(results, "right")
        if right_model:
            models = right_model["models"]
            logger.info(f"  Samples: {right_model['num_samples']}")
            for axis_name in ["x", "y", "z"]:
                m = models[axis_name]
                logger.info(f"  {axis_name}: t₀={m['t0']*1000:.3f}mm, k={m['k']*1000:.3f}mm/m, R²={m['r_squared']:.4f}")

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
            logger.info(f"  {axis_name}: t₀={t0*1000:.3f}mm, k={k*1000:.3f}mm/m, R²={r_squared:.4f}")

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
            avg_r2_left = np.mean([left_model["models"][a]["r_squared"] for a in ["x", "y", "z"]])
            logger.info(f"  LEFT  sensor avg R²: {avg_r2_left:.4f}")
        if right_model:
            avg_r2_right = np.mean([right_model["models"][a]["r_squared"] for a in ["x", "y", "z"]])
            logger.info(f"  RIGHT sensor avg R²: {avg_r2_right:.4f}")
        avg_r2_center = np.mean([center_models[a]["r_squared"] for a in ["x", "y", "z"]])
        logger.info(f"  CENTER (midpoint) avg R²: {avg_r2_center:.4f}")

        return {
            "left": left_model,
            "right": right_model,
            "center": center_model,
        }

    def run(self, visualize: bool = False):
        """Run full calibration pipeline."""
        self.load_data()
        results = self.process_samples(visualize=visualize)

        if len(results) == 0:
            logger.error("No samples were successfully processed!")
            return None

        logger.info(f"\nProcessed {len(results)} samples successfully")

        models = self.fit_linear_model(results)

        output = {
            "pose_name": self.pose_name,
            "detection_method": "contour",
            "camera_intrinsics": self.intrinsics,
            "linear_model_left": models["left"],
            "linear_model_right": models["right"],
            "linear_model_center": models["center"],
            "samples": results,
            "calibration_time": datetime.now().isoformat(),
        }

        output_file = self.pose_dir / "gelsight_calibration.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)

        logger.info(f"\nSaved calibration to {output_file}")

        return output


def calibrate(pose_name: str = "home", visualize: bool = False):
    """
    Compute T_camera_to_gelsight(u) calibration.

    Args:
        pose_name: Name of the pose/session to process
        visualize: Show detection visualization
    """
    logger.info("=" * 70)
    logger.info("Gelsight Calibration Computation")
    logger.info("=" * 70)
    logger.info(f"Pose: {pose_name}")
    logger.info(f"Method: contour detection + depth backprojection")
    logger.info("")

    calibrator = GelsightCalibrator(pose_name=pose_name)

    try:
        result = calibrator.run(visualize=visualize)
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
        logger.error(f"Sample index {sample_idx} out of range (max: {len(data['data'])-1})")
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

    depth_vis = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    depth_vis = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

    cv2.imshow("RGB", rgb_image)
    cv2.imshow("Depth", depth_vis)

    logger.info("Press any key to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    fire.Fire({
        "calibrate": calibrate,
        "preview": preview,
    })
