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
    contact_frame_index: int = 3,
    max_frame_index: int = 7,
    num_frames: int = 8,
    deformation: float = 0.0085,
    rgb_shape: tuple = (8, 100, 100, 3),
    depth_shape: tuple = (8, 100, 100),
    gs_shape: tuple = (8, 100, 100, 3),
) -> SampleData:
    """Helper to create SampleData for tests."""
    sample = Sample(
        sample_id=sample_id,
        object_id=object_id,
        num_frames=num_frames,
        contact_frame_index=contact_frame_index,
        max_frame_index=max_frame_index,
        deformation=deformation,
    )
    return SampleData(
        sample=sample,
        rgb=np.random.randint(0, 255, rgb_shape, dtype=np.uint8),
        gelsight_left=np.random.randint(0, 255, gs_shape, dtype=np.uint8),
        gelsight_right=np.random.randint(0, 255, gs_shape, dtype=np.uint8),
        depth=np.random.rand(*depth_shape).astype(np.float32),
        poses_left=np.random.rand(num_frames, 4, 4).astype(np.float32),
        poses_right=np.random.rand(num_frames, 4, 4).astype(np.float32),
        timestamps=np.random.rand(num_frames).astype(np.float64),
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
                num_frames=10,
                rgb_shape=(10, 480, 640, 3),
                depth_shape=(10, 480, 640),
                gs_shape=(10, 240, 320, 3),
            )

            sample_dir = writer.write_sample(data)

            # Check files exist
            assert (sample_dir / "sample.json").exists()
            assert (sample_dir / "rgb").exists()
            assert (sample_dir / "depth").exists()
            assert (sample_dir / "gelsight_left").exists()
            assert (sample_dir / "gelsight_right").exists()
            assert (sample_dir / "poses" / "left.npy").exists()
            assert (sample_dir / "poses" / "right.npy").exists()
            assert (sample_dir / "timestamps.npy").exists()

    def test_write_sample_json_content(self):
        """Test sample.json has correct content."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(
                sample_id="000001",
                object_id="object_001",
                contact_frame_index=3,
                max_frame_index=7,
                num_frames=8,
            )

            sample_dir = writer.write_sample(data)

            with open(sample_dir / "sample.json") as f:
                json_data = json.load(f)

            assert json_data["sample_id"] == "000001"
            assert json_data["object_id"] == "object_001"
            assert json_data["contact_frame_index"] == 3
            assert json_data["max_frame_index"] == 7
            assert json_data["num_frames"] == 8
            assert json_data["deformation"] == 0.0085

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

    def test_rgb_frames_saved_as_png(self):
        """Test RGB frames are saved as individual PNGs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(num_frames=5, rgb_shape=(5, 100, 100, 3))
            sample_dir = writer.write_sample(data)

            rgb_dir = sample_dir / "rgb"
            assert rgb_dir.exists()
            assert (rgb_dir / "00.png").exists()
            assert (rgb_dir / "04.png").exists()
            assert len(list(rgb_dir.glob("*.png"))) == 5

    def test_depth_frames_saved_as_npy(self):
        """Test depth frames are saved as individual NPYs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(num_frames=5, depth_shape=(5, 100, 100))
            sample_dir = writer.write_sample(data)

            depth_dir = sample_dir / "depth"
            assert depth_dir.exists()
            assert (depth_dir / "00.npy").exists()
            assert (depth_dir / "04.npy").exists()
            assert len(list(depth_dir.glob("*.npy"))) == 5

            # Check dtype
            loaded = np.load(depth_dir / "00.npy")
            assert loaded.dtype == np.float32

    def test_poses_dtype_float32(self):
        """Test that poses are saved as float32."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data()
            sample_dir = writer.write_sample(data)

            loaded_left = np.load(sample_dir / "poses" / "left.npy")
            loaded_right = np.load(sample_dir / "poses" / "right.npy")
            assert loaded_left.dtype == np.float32
            assert loaded_right.dtype == np.float32

    def test_poses_shape(self):
        """Test poses have correct shape (N, 4, 4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(num_frames=10)
            sample_dir = writer.write_sample(data)

            loaded_left = np.load(sample_dir / "poses" / "left.npy")
            loaded_right = np.load(sample_dir / "poses" / "right.npy")
            assert loaded_left.shape == (10, 4, 4)
            assert loaded_right.shape == (10, 4, 4)

    def test_timestamps_saved(self):
        """Test timestamps are saved correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            writer = DatasetWriter(tmpdir)

            data = make_sample_data(num_frames=5)
            sample_dir = writer.write_sample(data)

            loaded = np.load(sample_dir / "timestamps.npy")
            assert loaded.shape == (5,)
            assert loaded.dtype == np.float64

