"""RealSense RGB-D camera interface.

Requires: pip install pyrealsense2
"""

from __future__ import annotations

import numpy as np
import pyrealsense2 as rs

from src.utils.log import get_logger

logger = get_logger(__name__)


class RealSenseCamera:
    """RealSense RGB-D camera. Captures synchronized RGB and depth frames."""

    def __init__(self, width: int = 640, height: int = 480):
        """Initialize RealSense camera.

        Args:
            width: Frame width in pixels
            height: Frame height in pixels
        """
        self._pipeline = rs.pipeline()
        config = rs.config()

        # Configure streams with resolution only, let camera run at native fps
        # fps=0 means "don't care" - camera picks best available
        config.enable_stream(rs.stream.color, width, height, rs.format.rgb8, 0)
        config.enable_stream(rs.stream.depth, width, height, rs.format.z16, 0)

        # Align depth to color frame
        self._align = rs.align(rs.stream.color)

        # Start pipeline
        profile = self._pipeline.start(config)

        # Get depth scale for converting to meters
        depth_sensor = profile.get_device().first_depth_sensor()
        self._depth_scale = depth_sensor.get_depth_scale()

        logger.info(f"RealSense connected: {width}x{height}, depth_scale={self._depth_scale}")

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

    def close(self) -> None:
        """Disconnect."""
        self._pipeline.stop()
        logger.info("RealSense disconnected")
