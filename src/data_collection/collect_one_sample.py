"""
Collect one touch sample following dataset schema.

Robot assumed already positioned above object with gripper open.
Collects frames while closing gripper until stall.

Usage:
    python -m src.data_collection.collect_one_sample \
        --sample_id 000001 --gs_left_id 0 --gs_right_id 8 --dataset_dir dataset/
"""

import time
from pathlib import Path

import cv2
import numpy as np

from src.data_collection.writer import DatasetWriter
from src.robot.trossen_arm import TrossenArm
from src.sensors.gelsight import GelSightSensor
from src.sensors.realsense import RealSenseCamera
from src.utils.log import get_logger
from src.utils.transforms import compute_T_base_to_gelsight, load_calibration
from src.utils.types import (
    Object,
    Sample,
    SampleData,
)

logger = get_logger(__name__)


def compute_diff_from_baseline(current: np.ndarray, baseline: np.ndarray) -> float:
    """Compute mean absolute difference from baseline (pixel intensity 0-255)."""
    return float(
        np.mean(np.abs(current.astype(np.float32) - baseline.astype(np.float32)))
    )


def main(
    sample_id: str,
    gs_left_id: int,
    gs_right_id: int,
    dataset_dir: str = "dataset/",
    object_id: str = "test_object",
    fps: int = 30,
    debug: bool = False,
):
    """
    Collect one touch sample.

    Closes gripper until stall, collecting all frames.

    Args:
        sample_id: Sample identifier (e.g., "000001")
        gs_left_id: Left GelSight video device ID
        gs_right_id: Right GelSight video device ID
        dataset_dir: Dataset root directory
        object_id: Object identifier
        fps: Capture frame rate
        debug: Show real-time CV2 display
    """
    dataset_dir = Path(dataset_dir)

    # Load calibration
    calibration_dir = dataset_dir / "calibration"
    if not calibration_dir.exists():
        logger.error(f"Calibration not found: {calibration_dir}")
        return
    X, T_u_left, T_u_right = load_calibration(calibration_dir)
    logger.info("Calibration loaded (X, T_u_left, T_u_right)")

    # Connect hardware
    logger.info("Connecting to robot...")
    robot = TrossenArm()
    logger.info("Robot connected")

    logger.info(f"Connecting to RealSense (fps={fps})...")
    realsense = RealSenseCamera(fps=fps)
    logger.info("RealSense connected")

    logger.info(
        f"Connecting to GelSight (L={gs_left_id}, R={gs_right_id}, fps={fps})..."
    )
    gs_left = GelSightSensor(device_id=gs_left_id, fps=fps)
    gs_right = GelSightSensor(device_id=gs_right_id, fps=fps)
    logger.info("GelSight sensors connected")

    # Sensor warmup: capture and discard first few frames to stabilize
    logger.info("Warming up sensors...")
    for _ in range(10):
        realsense.capture()
        gs_left.capture()
        gs_right.capture()
    logger.info("Sensors warmed up")

    # Writer
    writer = DatasetWriter(dataset_dir)
    writer.write_object(Object(object_id=object_id, description=""))

    try:
        # Go to home position with gripper open
        logger.info("Going to home position...")
        robot.go_home()
        logger.info("At home position with gripper open")

        # Contact detection: multiplier-based model
        # During warmup, build baseline mean
        # After warmup, detect contact when diff > multiplier * baseline_mean
        contact_frame = -1
        # ontact_diff = 0.0
        warmup_frames = 5  # Frames to build baseline statistics
        contact_multiplier = 2.0  # Require diff to be 2x baseline mean
        prev_gs_left: np.ndarray | None = None
        prev_gs_right: np.ndarray | None = None

        # Running statistics for baseline diff
        diff_count = 0
        diff_mean = 0.0

        # Storage
        rgb_frames: list[np.ndarray] = []
        depth_frames: list[np.ndarray] = []
        gs_left_frames: list[np.ndarray] = []
        gs_right_frames: list[np.ndarray] = []
        poses_left: list[np.ndarray] = []
        poses_right: list[np.ndarray] = []
        timestamps: list[float] = []

        max_frames = 50
        gripper_step = -0.002  # Close by 2mm per step

        # Log initial gripper position
        initial_gripper = robot.get_gripper_opening()
        logger.info(f"Initial gripper opening: {initial_gripper * 1000:.1f}mm")

        # Collection loop: close gripper until stall
        frame_interval = 1.0 / fps
        prev_gripper = initial_gripper
        stall_count = 0
        max_frame = -1

        for frame_idx in range(max_frames):
            frame_start = time.time()

            # Close gripper by one step
            robot.step_gripper(gripper_step)

            # Get gripper state
            gripper_opening = robot.get_gripper_opening()

            # Check if gripper stalled (stopped moving)
            if abs(gripper_opening - prev_gripper) < 0.0001:
                stall_count += 1
            else:
                stall_count = 0
            prev_gripper = gripper_opening

            # Capture all sensors
            rgb, depth = realsense.capture()
            gs_l = gs_left.capture()
            gs_r = gs_right.capture()

            T_base_ee = robot.get_ee_pose()

            # Compute GelSight poses: T_base_gelsight = T_base_ee @ X @ T_cam_gelsight
            pose_left = compute_T_base_to_gelsight(
                T_base_ee, X, T_u_left, gripper_opening
            )
            pose_right = compute_T_base_to_gelsight(
                T_base_ee, X, T_u_right, gripper_opening
            )

            # Store
            rgb_frames.append(rgb)
            depth_frames.append(depth)
            gs_left_frames.append(gs_l)
            gs_right_frames.append(gs_r)
            poses_left.append(pose_left)
            poses_right.append(pose_right)
            timestamps.append(time.time())

            # Contact detection: running statistics model
            diff_left = 0.0
            diff_right = 0.0
            diff_max = 0.0
            if prev_gs_left is not None and prev_gs_right is not None:
                diff_left = compute_diff_from_baseline(gs_l, prev_gs_left)
                diff_right = compute_diff_from_baseline(gs_r, prev_gs_right)
                diff_max = max(diff_left, diff_right)

                if frame_idx < warmup_frames:
                    # Warmup: update running mean
                    diff_count += 1
                    diff_mean += (diff_max - diff_mean) / diff_count
                elif contact_frame < 0:
                    # After warmup: detect contact when diff > multiplier * baseline
                    threshold = contact_multiplier * diff_mean

                    if diff_max > threshold:
                        contact_frame = frame_idx
                        contact_diff = diff_max
                        ratio = diff_max / diff_mean if diff_mean > 0 else 0
                        logger.info(
                            f"CONTACT at frame {frame_idx}: grip={gripper_opening*1000:.1f}mm "
                            f"diff={diff_max:.2f} (baseline={diff_mean:.2f}, ratio={ratio:.2f}x, thresh={threshold:.2f})"
                        )

            prev_gs_left = gs_l.copy()
            prev_gs_right = gs_r.copy()

            # Progress log every frame
            logger.info(
                f"Frame {frame_idx}: grip={gripper_opening*1000:.1f}mm max_diff={diff_max:.2f}"
            )

            # Debug display
            if debug:
                rgb_disp = cv2.resize(rgb, (320, 240))
                gs_l_disp = cv2.resize(gs_l, (320, 240))
                gs_r_disp = cv2.resize(gs_r, (320, 240))

                rgb_bgr = cv2.cvtColor(rgb_disp, cv2.COLOR_RGB2BGR)
                gs_l_bgr = cv2.cvtColor(gs_l_disp, cv2.COLOR_RGB2BGR)
                gs_r_bgr = cv2.cvtColor(gs_r_disp, cv2.COLOR_RGB2BGR)

                cv2.putText(
                    rgb_bgr,
                    f"Frame {frame_idx}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    rgb_bgr,
                    f"Grip: {gripper_opening*1000:.1f}mm",
                    (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    rgb_bgr,
                    f"Diff: {diff_max:.1f}",
                    (10, 75),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    gs_l_bgr,
                    f"F{frame_idx} L:{diff_left:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                if contact_frame >= 0:
                    cv2.putText(
                        gs_l_bgr,
                        "CONTACT",
                        (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 0, 255),
                        2,
                    )
                cv2.putText(
                    gs_r_bgr,
                    f"F{frame_idx} R:{diff_right:.1f}",
                    (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )

                display = np.hstack([rgb_bgr, gs_l_bgr, gs_r_bgr])
                cv2.imshow("RealSense | GelSight L | GelSight R", display)

                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    logger.info("User quit")
                    break

            # Stop if gripper stalled (not moving for 5 frames)
            if stall_count >= 5:
                max_frame = frame_idx
                logger.info(
                    f"STOP: Gripper STALLED at frame {frame_idx}, grip={gripper_opening*1000:.1f}mm"
                )
                break

            # Stop if gripper fully closed
            if gripper_opening <= 0.001:
                max_frame = frame_idx
                logger.info(f"STOP: Gripper FULLY CLOSED at frame {frame_idx}")
                break

            # Maintain frame rate
            elapsed = time.time() - frame_start
            if debug:
                time.sleep(0.5)
            elif elapsed < frame_interval:
                time.sleep(frame_interval - elapsed)

        # Set max_frame if loop ended due to max_frames limit
        if max_frame < 0:
            max_frame = len(rgb_frames) - 1
            logger.info(f"STOP: Reached MAX_FRAMES limit ({max_frames}), max_frame={max_frame}")

        # Summary (raw collection)
        logger.info(f"Collected {len(rgb_frames)} raw frames")
        logger.info(f"Contact: frame {contact_frame}, Max: frame {max_frame}")
        if contact_frame < 0:
            logger.warning("No contact detected - saving all frames")

        # Trim frames: keep 3 frames before contact to max_frame
        # Note: frame_idx is 0-based, matches file names (00.png, 01.png, etc.)
        pre_contact_frames = 3
        if contact_frame >= 0:
            start_idx = max(0, contact_frame - pre_contact_frames)
            end_idx = max_frame + 1  # +1 because max_frame is index, slice is exclusive
        else:
            # No contact detected - save all frames
            start_idx = 0
            end_idx = len(rgb_frames)

        # Trim all arrays
        rgb_frames = rgb_frames[start_idx:end_idx]
        depth_frames = depth_frames[start_idx:end_idx]
        gs_left_frames = gs_left_frames[start_idx:end_idx]
        gs_right_frames = gs_right_frames[start_idx:end_idx]
        poses_left = poses_left[start_idx:end_idx]
        poses_right = poses_right[start_idx:end_idx]
        timestamps = timestamps[start_idx:end_idx]

        # Adjust frame indices to be relative to trimmed data (0-based, matches file names)
        new_contact_frame = contact_frame - start_idx if contact_frame >= 0 else -1
        new_max_frame = max_frame - start_idx

        logger.info(
            f"Trimmed to {len(rgb_frames)} frames (start={start_idx}, end={end_idx})"
        )
        logger.info(f"New indices - Contact: {new_contact_frame}, Max: {new_max_frame}")

        # Build sample data
        sample = Sample(
            sample_id=sample_id,
            object_id=object_id,
            num_frames=len(rgb_frames),
            contact_frame_index=new_contact_frame,
            max_frame_index=new_max_frame,
        )
        sample_data = SampleData(
            sample=sample,
            rgb=np.array(rgb_frames, dtype=np.uint8),
            gelsight_left=np.array(gs_left_frames, dtype=np.uint8),
            gelsight_right=np.array(gs_right_frames, dtype=np.uint8),
            depth=np.array(depth_frames, dtype=np.float32),
            poses_left=np.array(poses_left, dtype=np.float32),
            poses_right=np.array(poses_right, dtype=np.float32),
            timestamps=np.array(timestamps, dtype=np.float64),
        )

        # Write to disk
        sample_path = writer.write_sample(sample_data)
        logger.info(f"Saved sample {sample_id} to {sample_path}")

        # Update metadata
        writer.update_metadata()

        # Return to home position
        logger.info("Returning to home position...")
        robot.go_home()

    finally:
        cv2.destroyAllWindows()
        robot.close()
        realsense.close()
        gs_left.close()
        gs_right.close()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
