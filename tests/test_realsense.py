"""Tests for RealSense camera.

Run with: pytest tests/test_realsense.py -v -s
Requires: RealSense camera connected (auto-detected)
"""

import numpy as np
import pytest

# Skip entire module if pyrealsense2 not available
pytest.importorskip("pyrealsense2")

from src.sensors.realsense import RealSenseCamera


@pytest.fixture
def camera():
    """Create camera instance."""
    cam = RealSenseCamera(width=640, height=480, fps=30)
    yield cam
    cam.close()


class TestRealSenseCamera:
    """Tests for RealSense camera (requires hardware)."""

    def test_connect(self, camera):
        """Test camera connects successfully."""
        assert camera is not None

    def test_get_fps(self, camera):
        """Test reported FPS."""
        fps = camera.get_fps()
        assert fps > 0
        print(f"Reported FPS: {fps}")

    def test_get_resolution(self, camera):
        """Test resolution."""
        width, height = camera.get_resolution()
        assert width == 640
        assert height == 480
        print(f"Resolution: {width}x{height}")

    def test_capture_rgb(self, camera):
        """Test RGB capture."""
        rgb, _ = camera.capture()
        assert rgb.shape == (480, 640, 3)
        assert rgb.dtype == np.uint8
        print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")

    def test_capture_depth(self, camera):
        """Test depth capture."""
        _, depth = camera.capture()
        assert depth.shape == (480, 640)
        assert depth.dtype == np.float32
        print(f"Depth shape: {depth.shape}, dtype: {depth.dtype}")

    def test_measure_actual_fps(self, camera):
        """Test actual FPS measurement."""
        actual_fps = camera.measure_actual_fps(num_frames=50)
        assert actual_fps > 0
        print(f"Actual FPS: {actual_fps:.1f}")

    def test_capture_multiple_frames(self, camera):
        """Test capturing multiple frames."""
        frames = []
        for _ in range(10):
            rgb, depth = camera.capture()
            frames.append((rgb, depth))
        assert len(frames) == 10
        print(f"Captured {len(frames)} frames")
