"""
Dataset writer for saving samples to disk.

Saves samples in the following structure:
    dataset/
    ├── calibration/
    ├── objects/
    │   └── {object_id}.json
    ├── samples/
    │   └── {sample_id}/
    │       ├── sample.json
    │       ├── rgb.mp4
    │       ├── depth.npy
    │       ├── gelsight_left.mp4
    │       ├── gelsight_right.mp4
    │       └── poses.npy
    └── metadata.json
"""

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.utils.types import Metadata, Object, SampleData
from src.utils.video import frames_to_mp4


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

        # Save videos
        frames_to_mp4(data.rgb, sample_dir / "rgb.mp4", fps=data.sample.fps)
        frames_to_mp4(
            data.gelsight_left, sample_dir / "gelsight_left.mp4", fps=data.sample.fps
        )
        frames_to_mp4(
            data.gelsight_right, sample_dir / "gelsight_right.mp4", fps=data.sample.fps
        )

        # Save numpy arrays
        np.save(sample_dir / "depth.npy", data.depth.astype(np.float32))
        np.save(sample_dir / "poses.npy", data.poses.astype(np.float32))

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
