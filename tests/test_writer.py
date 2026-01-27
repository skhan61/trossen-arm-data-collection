"""Tests for dataset writer."""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.data_collection.writer import DatasetWriter
from src.utils.types import Metadata, Object, Sample, SampleData


def make_sample_data(
    sample_id: str = "000001",
    object_id: str = "obj_001",
    contact_frame: int = 5,
    max_press_frame: int = 10,
    fps: int = 30,
    num_frames: int = 10,
    rgb_shape: tuple = (10, 100, 100, 3),
    depth_shape: tuple = (10, 100, 100),
    gs_shape: tuple = (10, 100, 100, 3),
) -> SampleData:
    """Helper to create SampleData for tests."""
    sample = Sample(
        sample_id=sample_id,
        object_id=object_id,
        contact_frame=contact_frame,
        max_press_frame=max_press_frame,
        fps=fps,
        num_frames=num_frames,
    )
    return SampleData(
        sample=sample,
        rgb=np.random.randint(0, 255, rgb_shape, dtype=np.uint8),
        gelsight_left=np.random.randint(0, 255, gs_shape, dtype=np.uint8),
        gelsight_right=np.random.randint(0, 255, gs_shape, dtype=np.uint8),
        depth=np.random.rand(*depth_shape).astype(np.float32),
        poses=np.random.rand(num_frames, 2, 4, 4).astype(np.float32),
    )


class TestDatasetWriter:
    """Test DatasetWriter class."""

    def test_creates_directories(self):
        """Test that writer creates necessary directories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            assert (Path(tmpdir) / "samples").exists()
            assert (Path(tmpdir) / "objects").exists()

    def test_write_sample(self):
        """Test writing a complete sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(
                sample_id="000001",
                num_frames=90,
                rgb_shape=(90, 480, 640, 3),
                depth_shape=(90, 480, 640),
                gs_shape=(90, 240, 320, 3),
            )

            sample_dir = writer.write_sample(data)

            # Check files exist
            assert (sample_dir / "sample.json").exists()
            assert (sample_dir / "rgb.mp4").exists()
            assert (sample_dir / "depth.npy").exists()
            assert (sample_dir / "gelsight_left.mp4").exists()
            assert (sample_dir / "gelsight_right.mp4").exists()
            assert (sample_dir / "poses.npy").exists()

    def test_write_sample_json_content(self):
        """Test sample.json has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(
                sample_id="000001",
                object_id="object_001",
                contact_frame=15,
                max_press_frame=45,
                fps=30,
                num_frames=10,
            )

            sample_dir = writer.write_sample(data)

            with open(sample_dir / "sample.json") as f:
                json_data = json.load(f)

            assert json_data["sample_id"] == "000001"
            assert json_data["object_id"] == "object_001"
            assert json_data["contact_frame"] == 15
            assert json_data["max_press_frame"] == 45
            assert json_data["fps"] == 30
            assert json_data["num_frames"] == 10

    def test_write_object(self):
        """Test writing object metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            obj = Object(object_id="object_001", description="soft foam cube")
            obj_path = writer.write_object(obj)

            assert obj_path.exists()

            with open(obj_path) as f:
                data = json.load(f)

            assert data["object_id"] == "object_001"
            assert data["description"] == "soft foam cube"

    def test_write_metadata(self):
        """Test writing dataset metadata."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            metadata = Metadata(
                name="visual_haptic_deformation",
                date="2026-01-26",
                num_objects=10,
                num_samples=500,
            )
            metadata_path = writer.write_metadata(metadata)

            assert metadata_path.exists()

            with open(metadata_path) as f:
                data = json.load(f)

            assert data["name"] == "visual_haptic_deformation"
            assert data["num_objects"] == 10
            assert data["num_samples"] == 500

    def test_get_next_sample_id_empty(self):
        """Test next sample ID when no samples exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            assert writer.get_next_sample_id() == "000001"

    def test_get_next_sample_id_with_existing(self):
        """Test next sample ID with existing samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            # Create some sample directories
            (Path(tmpdir) / "samples" / "000001").mkdir(parents=True)
            (Path(tmpdir) / "samples" / "000002").mkdir(parents=True)
            (Path(tmpdir) / "samples" / "000005").mkdir(parents=True)

            assert writer.get_next_sample_id() == "000006"

    def test_count_samples(self):
        """Test counting samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            assert writer.count_samples() == 0

            (Path(tmpdir) / "samples" / "000001").mkdir(parents=True)
            (Path(tmpdir) / "samples" / "000002").mkdir(parents=True)

            assert writer.count_samples() == 2

    def test_count_objects(self):
        """Test counting objects."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            assert writer.count_objects() == 0

            writer.write_object(Object("obj_001", "cube"))
            writer.write_object(Object("obj_002", "sphere"))

            assert writer.count_objects() == 2

    def test_update_metadata(self):
        """Test updating metadata with current counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            # Add some objects
            writer.write_object(Object("obj_001", "cube"))
            writer.write_object(Object("obj_002", "sphere"))

            # Add some sample directories
            (Path(tmpdir) / "samples" / "000001").mkdir(parents=True)
            (Path(tmpdir) / "samples" / "000002").mkdir(parents=True)
            (Path(tmpdir) / "samples" / "000003").mkdir(parents=True)

            metadata = writer.update_metadata()

            assert metadata.num_objects == 2
            assert metadata.num_samples == 3
            assert metadata.name == "visual_haptic_deformation"

    def test_depth_dtype_float32(self):
        """Test that depth is saved as float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data()
            # Override with float64 to test conversion
            data.depth = np.random.rand(10, 100, 100).astype(np.float64)

            sample_dir = writer.write_sample(data)

            loaded_depth = np.load(sample_dir / "depth.npy")
            assert loaded_depth.dtype == np.float32

    def test_poses_dtype_float32(self):
        """Test that poses are saved as float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data()
            # Override with float64 to test conversion
            data.poses = np.random.rand(10, 2, 4, 4).astype(np.float64)

            sample_dir = writer.write_sample(data)

            loaded_poses = np.load(sample_dir / "poses.npy")
            assert loaded_poses.dtype == np.float32

    def test_depth_shape(self):
        """Test depth.npy has correct shape (N, H, W)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(
                num_frames=10,
                rgb_shape=(10, 120, 160, 3),
                depth_shape=(10, 120, 160),
                gs_shape=(10, 60, 80, 3),
            )

            sample_dir = writer.write_sample(data)

            loaded_depth = np.load(sample_dir / "depth.npy")
            assert loaded_depth.shape == (10, 120, 160)

    def test_poses_shape(self):
        """Test poses.npy has correct shape (N, 2, 4, 4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(num_frames=10)

            sample_dir = writer.write_sample(data)

            loaded_poses = np.load(sample_dir / "poses.npy")
            assert loaded_poses.shape == (10, 2, 4, 4)

    def test_rgb_video_shape(self):
        """Test rgb.mp4 has correct shape (N, H, W, 3)."""
        from src.utils.video import mp4_to_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(
                num_frames=10,
                rgb_shape=(10, 120, 160, 3),
                depth_shape=(10, 120, 160),
                gs_shape=(10, 60, 80, 3),
            )

            sample_dir = writer.write_sample(data)

            loaded_rgb = mp4_to_frames(sample_dir / "rgb.mp4")
            assert loaded_rgb.shape == (10, 120, 160, 3)
            assert loaded_rgb.dtype == np.uint8

    def test_gelsight_left_video_shape(self):
        """Test gelsight_left.mp4 has correct shape (N, H, W, 3)."""
        from src.utils.video import mp4_to_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(
                num_frames=10,
                rgb_shape=(10, 120, 160, 3),
                depth_shape=(10, 120, 160),
                gs_shape=(10, 60, 80, 3),
            )

            sample_dir = writer.write_sample(data)

            loaded_gs = mp4_to_frames(sample_dir / "gelsight_left.mp4")
            assert loaded_gs.shape == (10, 60, 80, 3)
            assert loaded_gs.dtype == np.uint8

    def test_gelsight_right_video_shape(self):
        """Test gelsight_right.mp4 has correct shape (N, H, W, 3)."""
        from src.utils.video import mp4_to_frames

        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(
                num_frames=10,
                rgb_shape=(10, 120, 160, 3),
                depth_shape=(10, 120, 160),
                gs_shape=(10, 60, 80, 3),
            )

            sample_dir = writer.write_sample(data)

            loaded_gs = mp4_to_frames(sample_dir / "gelsight_right.mp4")
            assert loaded_gs.shape == (10, 60, 80, 3)
            assert loaded_gs.dtype == np.uint8
