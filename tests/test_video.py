"""Tests for video utilities."""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.utils.video import frames_to_mp4, get_video_info, mp4_to_frames


class TestVideo:
    """Test video encode/decode functions."""

    def test_roundtrip_shape(self):
        """Test that shape is preserved after encode/decode."""
        original = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mp4"
            frames_to_mp4(original, path, fps=30)
            loaded = mp4_to_frames(path)

        assert original.shape == loaded.shape

    def test_roundtrip_dtype(self):
        """Test that dtype is uint8 after decode."""
        original = np.random.randint(0, 255, (5, 240, 320, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mp4"
            frames_to_mp4(original, path, fps=30)
            loaded = mp4_to_frames(path)

        assert loaded.dtype == np.uint8

    def test_roundtrip_values_close(self):
        """Test that pixel values are close after lossy compression."""
        # Use solid color blocks - less affected by compression
        original = np.zeros((5, 240, 320, 3), dtype=np.uint8)
        original[:, :120, :, 0] = 255  # Red top half
        original[:, 120:, :, 2] = 255  # Blue bottom half

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mp4"
            frames_to_mp4(original, path, fps=30)
            loaded = mp4_to_frames(path)

        # MP4 is lossy, allow some difference
        mean_diff = np.abs(original.astype(float) - loaded.astype(float)).mean()
        assert mean_diff < 30, f"Mean pixel difference too high: {mean_diff}"

    def test_single_frame(self):
        """Test encoding/decoding a single frame."""
        original = np.random.randint(0, 255, (1, 480, 640, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mp4"
            frames_to_mp4(original, path, fps=30)
            loaded = mp4_to_frames(path)

        assert loaded.shape[0] >= 1  # At least 1 frame

    def test_get_video_info(self):
        """Test getting video metadata."""
        original = np.random.randint(0, 255, (10, 480, 640, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mp4"
            frames_to_mp4(original, path, fps=30)
            info = get_video_info(path)

        assert info["height"] == 480
        assert info["width"] == 640
        assert info["num_frames"] == 10
        assert info["fps"] == pytest.approx(30, rel=0.1)

    def test_file_not_found(self):
        """Test error when file doesn't exist."""
        with pytest.raises(FileNotFoundError):
            mp4_to_frames("/nonexistent/path.mp4")

    def test_creates_parent_dir(self):
        """Test that parent directories are created."""
        original = np.random.randint(0, 255, (3, 100, 100, 3), dtype=np.uint8)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "subdir" / "nested" / "test.mp4"
            frames_to_mp4(original, path, fps=30)

            assert path.exists()

    def test_rgb_order_preserved(self):
        """Test that RGB order is preserved (not swapped to BGR)."""
        # Create frame with distinct R, G, B values
        original = np.zeros((3, 100, 100, 3), dtype=np.uint8)
        original[:, :, :, 0] = 200  # R channel = 200
        original[:, :, :, 1] = 100  # G channel = 100
        original[:, :, :, 2] = 50   # B channel = 50

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.mp4"
            frames_to_mp4(original, path, fps=30)
            loaded = mp4_to_frames(path)

        # Check that R > G > B is preserved (not BGR swapped)
        r_mean = loaded[:, :, :, 0].mean()
        g_mean = loaded[:, :, :, 1].mean()
        b_mean = loaded[:, :, :, 2].mean()

        assert r_mean > g_mean > b_mean, f"RGB order wrong: R={r_mean}, G={g_mean}, B={b_mean}"
