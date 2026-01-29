"""
Dataset writer for saving samples to disk.

Saves samples in the following structure:
    dataset/
    ├── objects/
    │   └── {object_id}.json
    ├── samples/
    │   └── {sample_id}/
    │       ├── sample.json
    │       ├── rgb/
    │       │   ├── 00.png
    │       │   └── ...
    │       ├── depth/
    │       │   ├── 00.npy
    │       │   └── ...
    │       ├── gelsight_left/
    │       │   ├── 00.png
    │       │   └── ...
    │       ├── gelsight_right/
    │       │   ├── 00.png
    │       │   └── ...
    │       ├── poses/
    │       │   ├── left.npy
    │       │   └── right.npy
    │       └── timestamps.npy
    └── metadata.json

sample.json schema:
    {
        "sample_id": "000001",
        "object_id": "object_001",
        "num_frames": 7,
        "contact_frame_index": 3,
        "max_frame_index": 6,
        "post_contact_squeeze": 0.0085
    }
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.utils.types import Metadata, Object, SampleData


class DatasetWriter:
    """Writer for saving dataset samples to disk."""

    def __init__(self, dataset_dir: str | Path):
        """
        Initialize dataset writer.

        Args:
            dataset_dir: Root directory for dataset (e.g., "dataset/")
        """
        self.dataset_dir = Path(dataset_dir)
        self.samples_dir = self.dataset_dir / "samples"
        self.objects_dir = self.dataset_dir / "objects"

        # Create directories
        self.samples_dir.mkdir(parents=True, exist_ok=True)
        self.objects_dir.mkdir(parents=True, exist_ok=True)

    def _save_frames_as_png(
        self, frames: np.ndarray, output_dir: Path
    ) -> None:
        """
        Save frames as individual PNG files (00.png, 01.png, ...).

        Args:
            frames: Image frames (N, H, W, 3) uint8, RGB order
            output_dir: Output directory for PNG files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(frames):
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imwrite(str(output_dir / f"{i:02d}.png"), frame_bgr)

    def _save_depth_frames(
        self, depth: np.ndarray, output_dir: Path
    ) -> None:
        """
        Save depth frames as individual NPY files (00.npy, 01.npy, ...).

        Args:
            depth: Depth frames (N, H, W) float32
            output_dir: Output directory for NPY files
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, frame in enumerate(depth):
            np.save(output_dir / f"{i:02d}.npy", frame.astype(np.float32))

    def write_sample(self, data: SampleData) -> Path:
        """
        Write a sample to disk.

        Args:
            data: Complete sample with metadata and arrays

        Returns:
            Path to sample directory
        """
        sample_dir = self.samples_dir / data.sample.sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save image frames as PNG (rgb/, gelsight_left/, gelsight_right/)
        self._save_frames_as_png(data.rgb, sample_dir / "rgb")
        self._save_frames_as_png(data.gelsight_left, sample_dir / "gelsight_left")
        self._save_frames_as_png(data.gelsight_right, sample_dir / "gelsight_right")

        # Save depth frames as NPY (depth/)
        self._save_depth_frames(data.depth, sample_dir / "depth")

        # Save poses (poses/left.npy, poses/right.npy)
        poses_dir = sample_dir / "poses"
        poses_dir.mkdir(parents=True, exist_ok=True)
        np.save(poses_dir / "left.npy", data.poses_left.astype(np.float32))
        np.save(poses_dir / "right.npy", data.poses_right.astype(np.float32))

        # Save timestamps
        np.save(sample_dir / "timestamps.npy", data.timestamps.astype(np.float64))

        # Save sample metadata
        with open(sample_dir / "sample.json", "w") as f:
            json.dump(asdict(data.sample), f, indent=2)

        return sample_dir

    def write_object(self, obj: Object) -> Path:
        """
        Write object metadata to disk.

        Args:
            obj: Object metadata

        Returns:
            Path to object JSON file
        """
        obj_path = self.objects_dir / f"{obj.object_id}.json"

        with open(obj_path, "w") as f:
            json.dump(asdict(obj), f, indent=2)

        return obj_path

    def write_metadata(self, metadata: Metadata) -> Path:
        """
        Write dataset metadata to disk.

        Args:
            metadata: Dataset metadata

        Returns:
            Path to metadata JSON file
        """
        metadata_path = self.dataset_dir / "metadata.json"

        with open(metadata_path, "w") as f:
            json.dump(asdict(metadata), f, indent=2)

        return metadata_path

    def get_next_sample_id(self) -> str:
        """
        Get the next available sample ID.

        Returns:
            Next sample ID as zero-padded string (e.g., "000001")
        """
        existing = list(self.samples_dir.glob("*"))
        existing_ids = []

        for p in existing:
            if p.is_dir():
                try:
                    existing_ids.append(int(p.name))
                except ValueError:
                    pass

        next_id = max(existing_ids, default=0) + 1
        return f"{next_id:06d}"

    def count_samples(self) -> int:
        """Count existing samples in dataset."""
        return len([p for p in self.samples_dir.glob("*") if p.is_dir()])

    def count_objects(self) -> int:
        """Count existing objects in dataset."""
        return len(list(self.objects_dir.glob("*.json")))

    def update_metadata(self, name: str = "visual_haptic_deformation") -> Metadata:
        """
        Update and save metadata with current counts.

        Args:
            name: Dataset name

        Returns:
            Updated Metadata object
        """
        metadata = Metadata(
            name=name,
            date=datetime.now().strftime("%Y-%m-%d"),
            num_objects=self.count_objects(),
            num_samples=self.count_samples(),
        )
        self.write_metadata(metadata)
        return metadata
