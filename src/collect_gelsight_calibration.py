#!/usr/bin/env python3
"""
Collect raw data for T_camera_to_gelsight(u) calibration.

AUTOMATIC mode - no user input required.
Shows live stream, captures automatically, quits when done.

At each gripper opening, saves:
- RGB image
- Depth image (numpy array)
- Gripper opening value
- Robot pose (T_base_to_ee)

Usage:
    .venv/bin/python src/collect_gelsight_calibration.py
    .venv/bin/python src/collect_gelsight_calibration.py --num_openings=17 --repeats=3
    .venv/bin/python src/collect_gelsight_calibration.py --gripper_min=26 --gripper_max=42
"""

import cv2
import numpy as np
import pyrealsense2 as rs
import json
import logging
import time
import fire
from pathlib import Path
from datetime import datetime

from trossen_arm import TrossenArmDriver, Model, StandardEndEffector, Mode

# ============================================================================
# Logging Setup
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"gelsight_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

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
PREVIEW_TIME = 0.5
DATA_DIR = Path(__file__).parent.parent / "data" / "gelsight_calibration_data"


class DataCollector:
    """Collect raw calibration data automatically."""

    def __init__(self, pose_name: str, openings: list, repeats: int):
        self.pose_name = pose_name
        self.openings = openings
        self.repeats = repeats
        self.pipeline = None
        self.driver = None
        self.align = None
        self.intrinsics = None

    def connect_camera(self):
        """Connect to RealSense camera."""
        logger.info("Connecting to RealSense camera...")
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        profile = self.pipeline.start(config)
        self.align = rs.align(rs.stream.color)
        color_profile = profile.get_stream(rs.stream.color)
        self.intrinsics = color_profile.as_video_stream_profile().get_intrinsics()
        self.intrinsics_dict = {
            "fx": self.intrinsics.fx,
            "fy": self.intrinsics.fy,
            "ppx": self.intrinsics.ppx,
            "ppy": self.intrinsics.ppy,
            "width": self.intrinsics.width,
            "height": self.intrinsics.height,
        }
        logger.info(f"Camera connected! fx={self.intrinsics.fx:.2f}, fy={self.intrinsics.fy:.2f}")

    def connect_robot(self):
        """Connect to robot."""
        logger.info(f"Connecting to robot at {ARM_IP}...")
        self.driver = TrossenArmDriver()
        self.driver.configure(
            model=Model.wxai_v0,
            end_effector=StandardEndEffector.wxai_v0_leader,
            serv_ip=ARM_IP,
            clear_error=True,
            timeout=10.0,
        )
        logger.info("Robot connected!")

    def get_robot_pose(self):
        """Get current robot pose as 4x4 matrix and cartesian values."""
        cartesian = self.driver.get_cartesian_positions()
        t = np.array(cartesian[:3])
        angle_axis = np.array(cartesian[3:6])
        R, _ = cv2.Rodrigues(angle_axis)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T, list(cartesian)

    def set_gripper(self, opening_m: float, goal_time: float = 1.5):
        """Set gripper opening."""
        logger.info(f"Setting gripper to {opening_m * 1000:.1f}mm...")
        self.driver.set_gripper_mode(Mode.position)
        self.driver.set_gripper_position(opening_m, goal_time=goal_time)
        time.sleep(goal_time + 0.5)
        actual = self.driver.get_gripper_position()
        logger.info(f"Gripper at {actual * 1000:.1f}mm")
        return actual

    def capture_frame(self):
        """Capture one frame from camera."""
        frames = self.pipeline.wait_for_frames()
        aligned = self.align.process(frames)
        color_frame = aligned.get_color_frame()
        depth_frame = aligned.get_depth_frame()
        if not color_frame or not depth_frame:
            return None, None
        return np.asanyarray(color_frame.get_data()), np.asanyarray(depth_frame.get_data())

    def run(self):
        """Main collection loop - AUTOMATIC."""
        pose_dir = DATA_DIR / self.pose_name
        pose_dir.mkdir(parents=True, exist_ok=True)
        total_samples = len(self.openings) * self.repeats

        logger.info("")
        logger.info("=" * 70)
        logger.info("Gelsight Calibration Data Collection - AUTOMATIC")
        logger.info(f"Pose: {self.pose_name}")
        logger.info(f"Openings: {len(self.openings)} | Repeats: {self.repeats} | Total: {total_samples}")
        logger.info("=" * 70)

        cv2.namedWindow("Data Collection", cv2.WINDOW_NORMAL)
        T_base_to_ee, cartesian = self.get_robot_pose()
        logger.info(f"Robot pose: [{cartesian[0]*1000:.1f}, {cartesian[1]*1000:.1f}, {cartesian[2]*1000:.1f}] mm")

        all_data = []
        sample_idx = 0

        try:
            for opening_idx, opening in enumerate(self.openings):
                logger.info(f"=== Opening {opening_idx + 1}/{len(self.openings)}: {opening * 1000:.1f}mm ===")
                actual_opening = self.set_gripper(opening)
                T_base_to_ee, cartesian = self.get_robot_pose()

                for rep in range(self.repeats):
                    sample_idx += 1
                    start_time = time.time()
                    color_image, depth_image = None, None

                    while time.time() - start_time < PREVIEW_TIME:
                        color_image, depth_image = self.capture_frame()
                        if color_image is not None:
                            display = color_image.copy()
                            cv2.putText(display, f"Opening {opening_idx + 1}/{len(self.openings)}: {actual_opening * 1000:.1f}mm | Rep {rep + 1}/{self.repeats}",
                                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                            cv2.putText(display, f"Sample {sample_idx}/{total_samples} | q to abort",
                                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
                            cv2.imshow("Data Collection", display)
                        if cv2.waitKey(1) & 0xFF == ord('q'):
                            raise KeyboardInterrupt

                    if color_image is not None and depth_image is not None:
                        img_prefix = f"opening_{opening_idx:02d}_{int(actual_opening * 1000)}mm_rep_{rep:02d}"
                        rgb_file = pose_dir / f"{img_prefix}_rgb.png"
                        depth_file = pose_dir / f"{img_prefix}_depth.npy"
                        cv2.imwrite(str(rgb_file), color_image)
                        np.save(str(depth_file), depth_image)
                        logger.info(f"Saved: {rgb_file.name}")

                        all_data.append({
                            "sample_index": sample_idx - 1,
                            "opening_index": opening_idx,
                            "repeat": rep,
                            "target_opening_m": opening,
                            "actual_opening_m": actual_opening,
                            "actual_opening_mm": actual_opening * 1000,
                            "rgb_file": rgb_file.name,
                            "depth_file": depth_file.name,
                            "robot_cartesian": cartesian,
                            "T_base_to_ee": T_base_to_ee.tolist(),
                            "timestamp": datetime.now().isoformat(),
                        })

            logger.info("=" * 70)
            logger.info("Collection complete!")

        except KeyboardInterrupt:
            logger.info("Aborted")

        finally:
            cv2.destroyAllWindows()
            output = {
                "pose_name": self.pose_name,
                "camera_intrinsics": self.intrinsics_dict,
                "openings_m": self.openings,
                "openings_mm": [o * 1000 for o in self.openings],
                "num_openings": len(self.openings),
                "repeats_per_opening": self.repeats,
                "num_samples": len(all_data),
                "data": all_data,
                "collection_time": datetime.now().isoformat(),
            }
            output_file = pose_dir / "calibration_data.json"
            with open(output_file, "w") as f:
                json.dump(output, f, indent=2)
            logger.info(f"Saved {len(all_data)} samples to {pose_dir}")

    def go_home(self):
        """Return gripper to home (closed) position."""
        logger.info("Returning gripper to home...")
        self.driver.set_gripper_mode(Mode.position)
        self.driver.set_gripper_position(0.0, goal_time=2.0)
        time.sleep(2.5)

    def cleanup(self):
        """Cleanup."""
        if self.driver:
            self.go_home()
            self.driver.cleanup()
        if self.pipeline:
            self.pipeline.stop()


def collect(
    pose_name: str = "home",
    gripper_min: float = 26.0,
    gripper_max: float = 42.0,
    num_openings: int = 17,
    repeats: int = 3,
):
    """
    Collect gelsight calibration data automatically.

    Args:
        pose_name: Name for this pose/session
        gripper_min: Minimum gripper opening in mm
        gripper_max: Maximum gripper opening in mm
        num_openings: Number of different gripper openings
        repeats: Number of captures per opening
    """
    openings = np.linspace(gripper_max / 1000, gripper_min / 1000, num_openings).tolist()

    logger.info(f"Gelsight Calibration Data Collection")
    logger.info(f"  Pose: {pose_name}")
    logger.info(f"  Gripper: {gripper_min:.1f}mm - {gripper_max:.1f}mm")
    logger.info(f"  Openings: {num_openings} | Repeats: {repeats}")
    logger.info(f"  Total: {num_openings * repeats} samples")

    collector = DataCollector(pose_name=pose_name, openings=openings, repeats=repeats)

    try:
        collector.connect_camera()
        collector.connect_robot()
        collector.run()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        collector.cleanup()
        logger.info("Done")


if __name__ == "__main__":
    fire.Fire(collect)
