"""GelSight tactile sensor interface.

GelSight sensors appear as USB cameras. Uses OpenCV for capture.
"""

from __future__ import annotations

import cv2
import numpy as np

from src.utils.log import get_logger

logger = get_logger(__name__)


class GelSightSensor:
    """GelSight tactile sensor. Captures tactile images via USB camera."""

    def __init__(self, device_id: int):
        """Initialize GelSight sensor.

        Args:
            device_id: USB camera device ID (e.g., 0, 1, 2)
        """
        self.device_id = device_id
        self._cap = cv2.VideoCapture(device_id)

        if not self._cap.isOpened():
            raise RuntimeError(f"Failed to open GelSight camera at device {device_id}")

        logger.info(f"GelSight connected at device {device_id}")

    def capture(self) -> np.ndarray:
        """Capture tactile image.

        Returns:
            frame: (H, W, 3) uint8 BGR
        """
        ret, frame = self._cap.read()
        if not ret:
            raise RuntimeError("Failed to capture GelSight frame")
        return frame

    def close(self) -> None:
        """Disconnect."""
        self._cap.release()
        logger.info(f"GelSight {self.device_id} disconnected")
