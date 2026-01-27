"""Tests for GelSight sensor.

Run with: pytest tests/test_gelsight.py -v -s
Requires: GelSight camera connected (auto-detected)
"""

import numpy as np
import pytest

from src.sensors.gelsight import GelSightSensor


@pytest.fixture
def sensor(device_id):
    """Create sensor instance with specified device."""
    s = GelSightSensor(device_id=device_id, fps=30)
    yield s
    s.close()


class TestGelSightSensor:
    """Tests for GelSight sensor (requires hardware)."""

    def test_connect(self, sensor):
        """Test sensor connects successfully."""
        assert sensor is not None

    def test_get_fps(self, sensor):
        """Test reported FPS."""
        fps = sensor.get_fps()
        assert fps >= 0
        print(f"Reported FPS: {fps}")

    def test_get_resolution(self, sensor):
        """Test resolution."""
        width, height = sensor.get_resolution()
        assert width > 0
        assert height > 0
        print(f"Resolution: {width}x{height}")

    def test_capture(self, sensor):
        """Test frame capture."""
        frame = sensor.capture()
        assert frame.ndim == 3
        assert frame.shape[2] == 3
        assert frame.dtype == np.uint8
        print(f"Frame shape: {frame.shape}, dtype: {frame.dtype}")

    def test_measure_actual_fps(self, sensor):
        """Test actual FPS measurement."""
        actual_fps = sensor.measure_actual_fps(num_frames=50)
        assert actual_fps > 0
        print(f"Actual FPS: {actual_fps:.1f}")
