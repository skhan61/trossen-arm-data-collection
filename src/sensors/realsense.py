"""RealSense RGB-D camera interface.

Requires: pip install pyrealsense2
"""

from __future__ import annotations

import time

import numpy as np
import pyrealsense2 as rs

from src.utils.log import get_logger

logger = get_logger(__name__)


class RealSenseCamera:
    """RealSense RGB-D camera. Captures synchronized RGB and depth frames."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 30):
        """Initialize RealSense camera.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
            fps: Target frames per second
        """
        self._pipeline = rs.pipeline()
        config = rs.config()

        # Configure streams with resolution and fps
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, fps)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)

        # Align depth to color frame
        self._align = rs.align(rs.stream.color)

        # Start pipeline
        profile = self._pipeline.start(config)

        # Get depth scale for converting to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        # Get actual FPS from stream profile
        color_profile = profile.get_stream(rs.stream.color).as_video_stream_profile()
        self._fps = color_profile.fps()
        self._width = color_profile.width()
        self._height = color_profile.height()

        # Warmup: discard first few frames to let camera stabilize
        for _ in range(5):
            self._pipeline.wait_for_frames()

        logger.info(
            f"RealSense connected: {self._width}x{self._height} @ {self._fps}fps, "
            f"depth_scale={self._depth_scale}"
        )

    def capture(self) -> tuple[np.ndarray, np.ndarray]:
        """Capture RGB and depth.

        Returns:
            rgb: (H, W, 3) uint8
            depth: (H, W) float32 meters
        """
        frames = self._pipeline.wait_for_frames()
        aligned = self._align.process(frames)

        rgb = np.asanyarray(aligned.get_color_frame().get_data())
        depth_raw = np.asanyarray(aligned.get_depth_frame().get_data())
        depth = depth_raw.astype(np.float32) * self._depth_scale

        return rgb, depth

    def get_fps(self) -> int:
        """Get reported FPS from camera profile."""
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
        logger.info(f"Measuring RealSense FPS ({num_frames} frames)...")
        start = time.time()
        for _ in range(num_frames):
            frames = self._pipeline.wait_for_frames()
            self._align.process(frames)
        elapsed = time.time() - start
        actual_fps = num_frames / elapsed
        logger.info(f"RealSense actual FPS: {actual_fps:.1f}")
        return actual_fps

    def close(self) -> None:
        """Disconnect."""
        self._pipeline.stop()
        logger.info("RealSense disconnected")


if __name__ == "__main__":
    # Test RealSense camera
    print("=" * 50)
    print("RealSense Camera Test")
    print("=" * 50)

    camera = RealSenseCamera(fps=30)
    print(f"Reported FPS: {camera.get_fps()}")
    print(f"Resolution: {camera.get_resolution()}")

    # Measure actual FPS
    actual_fps = camera.measure_actual_fps(num_frames=100)
    print(f"Actual FPS: {actual_fps:.1f}")

    # Capture one frame to test
    rgb, depth = camera.capture()
    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")

    camera.close()
