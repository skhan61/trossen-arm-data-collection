"""
Video utilities for encoding/decoding mp4 files.

Usage:
    from src.utils.video import frames_to_mp4, mp4_to_frames

    # Save numpy frames as mp4
    frames = np.random.randint(0, 255, (90, 480, 640, 3), dtype=np.uint8)
    frames_to_mp4(frames, "output.mp4", fps=30)

    # Load mp4 as numpy frames
    frames = mp4_to_frames("output.mp4")
"""

from pathlib import Path

import cv2
import numpy as np


def frames_to_mp4(
    frames: np.ndarray,
    path: str | Path,
    fps: int = 30,
    codec: str = "mp4v",
) -> None:
    """
    Save numpy frames as mp4 video.

    Args:
        frames: Video frames (N, H, W, 3) uint8, RGB order
        path: Output file path
        fps: Frames per second
        codec: FourCC codec (default: mp4v)
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    n_frames, height, width = frames.shape[:3]

    fourcc = cv2.VideoWriter_fourcc(*codec)
    writer = cv2.VideoWriter(str(path), fourcc, fps, (width, height))

    for i in range(n_frames):
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frames[i], cv2.COLOR_RGB2BGR)
        writer.write(frame_bgr)

    writer.release()


def mp4_to_frames(path: str | Path) -> np.ndarray:
    """
    Load mp4 video as numpy frames.

    Args:
        path: Input video file path

    Returns:
        Video frames (N, H, W, 3) uint8, RGB order
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")

    frames = []
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == 0:
        raise ValueError(f"No frames read from video: {path}")

    return np.array(frames, dtype=np.uint8)


def get_video_info(path: str | Path) -> dict:
    """
    Get video metadata.

    Args:
        path: Video file path

    Returns:
        Dict with keys: num_frames, height, width, fps
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Video file not found: {path}")

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video: {path}")

    info = {
        "num_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "fps": cap.get(cv2.CAP_PROP_FPS),
    }

    cap.release()
    return info
