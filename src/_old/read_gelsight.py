#!/usr/bin/env python3
"""
Read data from TWO GelSight sensors connected via USB.

GelSight sensors appear as standard USB cameras (UVC devices).
This script detects, connects, and captures images from both sensors.
"""

import cv2
import numpy as np
import logging
import time
import json
from pathlib import Path
from datetime import datetime

# ============================================================================
# Logging Setup
# ============================================================================

LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = LOG_DIR / f"gelsight_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# ============================================================================
# Configuration
# ============================================================================

DATA_DIR = Path(__file__).parent.parent / "data" / "gelsight_data"


def list_cameras():
    """List all available camera devices."""
    logger.info("Scanning for available cameras...")
    available = []

    for i in range(10):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            backend = cap.getBackendName()

            available.append({
                "index": i,
                "width": w,
                "height": h,
                "fps": fps,
                "backend": backend
            })
            logger.info(f"  Camera {i}: {w}x{h} @ {fps:.1f} fps ({backend})")
            cap.release()

    if not available:
        logger.warning("No cameras found!")

    return available


class DualGelSightReader:
    """Read images from TWO GelSight tactile sensors simultaneously."""

    def __init__(self, sensor1_index, sensor2_index, width=640, height=480):
        self.sensor1_index = sensor1_index
        self.sensor2_index = sensor2_index
        self.width = width
        self.height = height
        self.cap1 = None
        self.cap2 = None
        self.frame_count = 0

    def connect(self):
        """Connect to both GelSight sensors."""
        logger.info(f"Connecting to Sensor 1 (index {self.sensor1_index})...")
        self.cap1 = cv2.VideoCapture(self.sensor1_index)
        if not self.cap1.isOpened():
            raise RuntimeError(f"Failed to open Sensor 1 (index {self.sensor1_index})")

        self.cap1.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap1.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        w1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_WIDTH))
        h1 = int(self.cap1.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"  Sensor 1 connected: {w1}x{h1}")

        logger.info(f"Connecting to Sensor 2 (index {self.sensor2_index})...")
        self.cap2 = cv2.VideoCapture(self.sensor2_index)
        if not self.cap2.isOpened():
            raise RuntimeError(f"Failed to open Sensor 2 (index {self.sensor2_index})")

        self.cap2.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap2.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        w2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_WIDTH))
        h2 = int(self.cap2.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"  Sensor 2 connected: {w2}x{h2}")

        logger.info("Both sensors connected!")
        return True

    def read_frames(self):
        """Read frames from both sensors."""
        if self.cap1 is None or self.cap2 is None:
            return None, None

        ret1, frame1 = self.cap1.read()
        ret2, frame2 = self.cap2.read()

        if ret1 and ret2:
            self.frame_count += 1
            return frame1, frame2

        return (frame1 if ret1 else None), (frame2 if ret2 else None)

    def compute_depth_map(self, frame):
        """Compute approximate depth/deformation map from GelSight image."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(float)
        gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-6)

        gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        depth = np.sqrt(gx**2 + gy**2)
        depth = (depth * 255).astype(np.uint8)
        return depth

    def save_frames(self, frame1, frame2, output_dir):
        """Save both frames to files."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        file1 = output_dir / f"sensor1_{timestamp}.png"
        file2 = output_dir / f"sensor2_{timestamp}.png"

        cv2.imwrite(str(file1), frame1)
        cv2.imwrite(str(file2), frame2)

        logger.info(f"Saved: {file1.name}, {file2.name}")
        return file1, file2

    def run_live_view(self, save_dir=None):
        """Run live view of both GelSight sensors side by side.

        Keys:
            's' - Save current frames
            'd' - Toggle depth map view
            'r' - Show raw image
            'q' - Quit
        """
        if save_dir is None:
            save_dir = DATA_DIR

        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        cv2.namedWindow("GelSight Sensors (Left: S1, Right: S2)", cv2.WINDOW_NORMAL)

        logger.info("")
        logger.info("=" * 60)
        logger.info("Dual GelSight Live View")
        logger.info("=" * 60)
        logger.info("Keys:")
        logger.info("  's' - Save current frames")
        logger.info("  'd' - Toggle depth map view")
        logger.info("  'r' - Show raw image")
        logger.info("  'q' - Quit")
        logger.info("=" * 60)
        logger.info("")

        view_mode = "raw"
        saved_count = 0
        fps_time = time.time()
        fps_frames = 0
        current_fps = 0

        try:
            while True:
                frame1, frame2 = self.read_frames()

                if frame1 is None and frame2 is None:
                    logger.warning("Failed to read from both sensors")
                    continue

                # Calculate FPS
                fps_frames += 1
                if time.time() - fps_time >= 1.0:
                    current_fps = fps_frames
                    fps_frames = 0
                    fps_time = time.time()

                # Process frames based on view mode
                if view_mode == "depth":
                    if frame1 is not None:
                        depth1 = self.compute_depth_map(frame1)
                        display1 = cv2.applyColorMap(depth1, cv2.COLORMAP_JET)
                    else:
                        display1 = np.zeros((self.height, self.width, 3), dtype=np.uint8)

                    if frame2 is not None:
                        depth2 = self.compute_depth_map(frame2)
                        display2 = cv2.applyColorMap(depth2, cv2.COLORMAP_JET)
                    else:
                        display2 = np.zeros((self.height, self.width, 3), dtype=np.uint8)
                else:
                    display1 = frame1.copy() if frame1 is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    display2 = frame2.copy() if frame2 is not None else np.zeros((self.height, self.width, 3), dtype=np.uint8)

                # Add labels
                cv2.putText(display1, "SENSOR 1", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(display2, "SENSOR 2", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                # Indicate if sensor is disconnected
                if frame1 is None:
                    cv2.putText(display1, "DISCONNECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if frame2 is None:
                    cv2.putText(display2, "DISCONNECTED", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                # Resize to same height if needed
                h1, w1 = display1.shape[:2]
                h2, w2 = display2.shape[:2]
                if h1 != h2:
                    target_h = max(h1, h2)
                    if h1 != target_h:
                        display1 = cv2.resize(display1, (int(w1 * target_h / h1), target_h))
                    if h2 != target_h:
                        display2 = cv2.resize(display2, (int(w2 * target_h / h2), target_h))

                # Combine side by side
                combined = np.hstack([display1, display2])

                # Add info overlay
                h, w = combined.shape[:2]
                cv2.putText(combined, f"Mode: {view_mode} | FPS: {current_fps} | Frame: {self.frame_count} | Saved: {saved_count}",
                           (10, h-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(combined, "Keys: s=save, d=depth, r=raw, q=quit",
                           (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

                cv2.imshow("GelSight Sensors (Left: S1, Right: S2)", combined)

                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    break
                elif key == ord('s'):
                    if frame1 is not None and frame2 is not None:
                        self.save_frames(frame1, frame2, save_dir)
                        saved_count += 1
                    else:
                        logger.warning("Cannot save - one or both sensors disconnected")
                elif key == ord('d'):
                    view_mode = "depth"
                    logger.info("Switched to depth map view")
                elif key == ord('r'):
                    view_mode = "raw"
                    logger.info("Switched to raw image view")

        finally:
            cv2.destroyAllWindows()

    def capture_sequence(self, num_frames, output_dir=None, delay_ms=100):
        """Capture a sequence of frames from both sensors."""
        if output_dir is None:
            output_dir = DATA_DIR / f"sequence_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Capturing {num_frames} frames to {output_dir}...")

        frames_data = []

        for i in range(num_frames):
            frame1, frame2 = self.read_frames()

            if frame1 is None or frame2 is None:
                logger.warning(f"Failed to read frame {i}")
                continue

            file1 = output_dir / f"sensor1_frame_{i:04d}.png"
            file2 = output_dir / f"sensor2_frame_{i:04d}.png"

            cv2.imwrite(str(file1), frame1)
            cv2.imwrite(str(file2), frame2)

            frames_data.append({
                "frame_index": i,
                "sensor1_file": file1.name,
                "sensor2_file": file2.name,
                "timestamp": datetime.now().isoformat()
            })

            if (i + 1) % 10 == 0:
                logger.info(f"  Captured {i + 1}/{num_frames} frames")

            time.sleep(delay_ms / 1000.0)

        # Save metadata
        metadata_file = output_dir / "sequence_metadata.json"
        with open(metadata_file, "w") as f:
            json.dump({
                "num_frames": len(frames_data),
                "frames": frames_data,
                "capture_time": datetime.now().isoformat()
            }, f, indent=2)

        logger.info(f"Saved {len(frames_data)} frame pairs to {output_dir}")
        return frames_data

    def cleanup(self):
        """Release camera resources."""
        if self.cap1 is not None:
            self.cap1.release()
            logger.info("Sensor 1 released")
        if self.cap2 is not None:
            self.cap2.release()
            logger.info("Sensor 2 released")


def main():
    logger.info("=" * 60)
    logger.info("Dual GelSight Sensor Reader")
    logger.info("=" * 60)
    logger.info("")

    # List available cameras
    cameras = list_cameras()

    if len(cameras) < 2:
        logger.error(f"Need at least 2 cameras, found {len(cameras)}. Check USB connections.")
        return 1

    logger.info("")
    logger.info(f"Found {len(cameras)} cameras")

    # Simply use the first two available cameras
    # GelSight Mini sensors report 3280x2464 native resolution
    if len(cameras) < 2:
        logger.error(f"Need 2 cameras, found {len(cameras)}")
        return 1

    for cam in cameras:
        logger.info(f"  -> Camera {cam['index']} available ({cam['width']}x{cam['height']})")

    # Use first two detected cameras
    sensor1_idx = cameras[0]["index"]
    sensor2_idx = cameras[1]["index"]

    logger.info("")
    logger.info(f"Using camera {sensor1_idx} as Sensor 1")
    logger.info(f"Using camera {sensor2_idx} as Sensor 2")
    logger.info("")
    logger.info("If sensors are swapped, modify sensor1_index and sensor2_index")
    logger.info("")

    # Create reader
    reader = DualGelSightReader(sensor1_index=sensor1_idx, sensor2_index=sensor2_idx)

    try:
        reader.connect()
        reader.run_live_view()
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        reader.cleanup()

    return 0


if __name__ == "__main__":
    exit(main())
