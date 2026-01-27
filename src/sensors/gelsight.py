"""GelSight tactile sensor interface.

GelSight sensors appear as USB cameras. Uses OpenCV for capture.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from src.utils.log import get_logger

logger = get_logger(__name__)


class GelSightSensor:
    """GelSight tactile sensor. Captures tactile images via USB camera."""

    def __init__(self, device_id: int, fps: int = 30, width: int = 640, height: int = 480):
        """Initialize GelSight sensor.

        Args:
            device_id: USB camera device ID (e.g., 0, 1, 2)
            fps: Target frames per second
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self.device_id = device_id
        self._cap = cv2.VideoCapture(device_id)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open GelSight camera at device {device_id}")

        # Set requested FPS and resolution
        self._cap.set(cv2.CAP_PROP_FPS, fps)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

        # Query actual values (camera may not support requested values)
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        logger.info(
            f"GelSight {device_id} connected: {self._width}x{self._height} @ {self._fps}fps"
        )

    def capture(self) -> np.ndarray:
        """Capture tactile image.

        Returns:
            frame: (H, W, 3) uint8 BGR
        """
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture GelSight frame")
        return frame

    def get_fps(self) -> float:
        """Get reported FPS from camera."""
        return self._fps

    def get_resolution(self) -> tuple[int, int]:
        """Get actual resolution (width, height)."""
        return self._width, self._height

    def measure_actual_fps(self, num_frames: int = 100) -> float:
        """Measure actual FPS by capturing frames.

        Args:
            num_frames: Number of frames to capture for measurement

        Returns:
            Measured FPS
        """
        logger.info(f"Measuring GelSight {self.device_id} FPS ({num_frames} frames)...")
        start = time.time()
        for _ in range(num_frames):
            ret, _ = self._cap.read()
            if not ret:
                raise RuntimeError("Failed to capture GelSight frame during FPS measurement")
        elapsed = time.time() - start
        actual_fps = num_frames / elapsed
        logger.info(f"GelSight {self.device_id} actual FPS: {actual_fps:.1f}")
        return actual_fps

    def close(self) -> None:
        """Disconnect."""
        self._cap.release()
        logger.info(f"GelSight {self.device_id} disconnected")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GelSight sensor")
    parser.add_argument("--device", type=int, default=0, help="USB camera device ID")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    args = parser.parse_args()

    print("=" * 50)
    print(f"GelSight Sensor Test (device {args.device})")
    print("=" * 50)

    sensor = GelSightSensor(device_id=args.device, fps=args.fps)
    print(f"Reported FPS: {sensor.get_fps()}")
    print(f"Resolution: {sensor.get_resolution()}")

    # Measure actual FPS
    actual_fps = sensor.measure_actual_fps(num_frames=100)
    print(f"Actual FPS: {actual_fps:.1f}")

    # Capture one frame to test
    frame = sensor.capture()
    print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

    sensor.close()
