"""
Data collection entry point (Simplified - ONE Touch Sample).

Collects a single touch sample. Object assumed to be already positioned.

Usage:
    python -m src.data_collection.run_collection --dataset_dir dataset/

Requires hardware:
    - Trossen arm with gripper
    - RealSense camera
    - GelSight sensors (left + right)

Workflow:
    1. Connect to hardware
    2. Load calibration
    3. Capture GelSight baseline (no contact)
    4. User positions robot above object
    5. Robot moves down, captures data
    6. Detect first touch (GelSight differential)
    7. Verify touch (sustained + increasing diff)
    8. Continue until max press depth
    9. Save sample and retract
"""

import argparse
import time
from pathlib import Path

import numpy as np

from src.data_collection.writer import DatasetWriter
from src.robot.arm import RobotArm
from src.sensors.gelsight import GelSightSensor
from src.sensors.realsense import RealSenseCamera
from src.utils.log import get_logger
from src.utils.transforms import compute_both_gelsight_poses, load_calibration
from src.utils.types import (
    CollectionConfig,
    Object,
    # ObjectBoundary,  # Not needed - no object detection
    Sample,
    SampleData,
    Transform4x4,
    TuParams,
    XMatrix,
)

logger = get_logger(__name__)


# =============================================================================
# Touch Detection (GelSight Differential)
# =============================================================================


def compute_diff_value(
    current_frame: np.ndarray,
    baseline_frame: np.ndarray,
) -> float:
    """Compute mean absolute difference between two GelSight frames.

    Args:
        current_frame: Current GelSight image (H, W, 3) uint8
        baseline_frame: Baseline GelSight image (H, W, 3) uint8

    Returns:
        Mean absolute difference value
    """
    diff = np.abs(current_frame.astype(np.float32) - baseline_frame.astype(np.float32))
    return float(np.mean(diff))


def detect_touch_differential(
    current_frame: np.ndarray,
    baseline_frame: np.ndarray,
    threshold: float = 15.0,
) -> tuple[bool, float]:
    """Detect touch using GelSight differential.

    Compares current GelSight frame against baseline (no contact).
    Touch detected when mean absolute difference exceeds threshold.

    Args:
        current_frame: Current GelSight image (H, W, 3) uint8
        baseline_frame: Baseline GelSight image (H, W, 3) uint8
        threshold: Mean pixel difference threshold (default 15.0)

    Returns:
        touched: True if touch detected
        diff_value: Mean absolute difference value
    """
    diff_value = compute_diff_value(current_frame, baseline_frame)
    touched = diff_value > threshold
    return touched, diff_value


def verify_touch(
    gs_left_frames: list[np.ndarray],
    gs_right_frames: list[np.ndarray],
    baseline_left: np.ndarray,
    baseline_right: np.ndarray,
    contact_frame: int,
    threshold: float = 15.0,
    num_frames_check: int = 3,
) -> tuple[bool, str]:
    """Verify that detected touch is real (not false positive).

    Checks:
    1. Differential is sustained for multiple frames after contact
    2. Differential increases as pressing continues (expected behavior)

    Args:
        gs_left_frames: List of left GelSight frames
        gs_right_frames: List of right GelSight frames
        baseline_left: Baseline left GelSight image
        baseline_right: Baseline right GelSight image
        contact_frame: Frame index where touch was detected
        threshold: Differential threshold
        num_frames_check: Number of frames after contact to check

    Returns:
        verified: True if touch is verified
        reason: Explanation of verification result
    """
    if contact_frame < 0:
        return False, "No touch detected"

    # Check we have enough frames after contact
    frames_after = len(gs_left_frames) - contact_frame - 1
    if frames_after < num_frames_check:
        return False, f"Not enough frames after contact ({frames_after} < {num_frames_check})"

    # Get diff values at contact and subsequent frames
    diffs_left = []
    diffs_right = []
    for i in range(contact_frame, min(contact_frame + num_frames_check + 1, len(gs_left_frames))):
        diffs_left.append(compute_diff_value(gs_left_frames[i], baseline_left))
        diffs_right.append(compute_diff_value(gs_right_frames[i], baseline_right))

    # Check 1: Differential stays above threshold (sustained touch)
    sustained_left = all(d > threshold for d in diffs_left)
    sustained_right = all(d > threshold for d in diffs_right)
    sustained = sustained_left or sustained_right  # At least one sensor sustained

    if not sustained:
        return False, f"Touch not sustained (L={diffs_left}, R={diffs_right})"

    # Check 2: Differential increases (pressing deeper = more deformation)
    increasing_left = diffs_left[-1] > diffs_left[0]
    increasing_right = diffs_right[-1] > diffs_right[0]
    increasing = increasing_left or increasing_right

    if not increasing:
        return False, f"Diff not increasing (L: {diffs_left[0]:.1f}->{diffs_left[-1]:.1f}, R: {diffs_right[0]:.1f}->{diffs_right[-1]:.1f})"

    return True, f"Touch verified (L: {diffs_left[0]:.1f}->{diffs_left[-1]:.1f}, R: {diffs_right[0]:.1f}->{diffs_right[-1]:.1f})"


# =============================================================================
# OLD: Position-based contact detection (commented out - no object detection)
# =============================================================================
# def detect_contact(
#     robot_pose: Transform4x4,
#     boundary: ObjectBoundary,
#     tolerance: float = 0.0003,
# ) -> bool:
#     """Detect if gripper is in contact with object.
#
#     Position-based contact detection (no force sensor).
#     Compares gripper X position against expected contact line.
#     """
#     robot_xyz = robot_pose[:3, 3]
#     expected_x = boundary.get_contact_x_at_z(robot_xyz[2])
#     return abs(robot_xyz[0] - expected_x) < tolerance


# =============================================================================
# Collection Loop
# =============================================================================


def collect_sample(
    robot: RobotArm,
    realsense: RealSenseCamera,
    gelsight_left: GelSightSensor,
    gelsight_right: GelSightSensor,
    X: XMatrix,
    T_u_left: TuParams,
    T_u_right: TuParams,
    baseline_left: np.ndarray,
    baseline_right: np.ndarray,
    config: CollectionConfig,
    touch_threshold: float = 15.0,
) -> tuple[list, list, list, list, list, list, list, int, int, float]:
    """Collect one sample (press sequence).

    Sample collection loop:
        1. Capture all sensors (RGB, depth, GelSight L/R)
        2. Get robot state (end-effector pose, gripper opening)
        3. Compute GelSight poses using calibration
        4. Store frame data
        5. Check for touch (GelSight differential vs baseline)
        6. Check for max press depth
        7. Move robot down by step_size
        8. Repeat until max press reached

    Args:
        robot: Robot arm interface
        realsense: RealSense camera interface
        gelsight_left: Left GelSight sensor
        gelsight_right: Right GelSight sensor
        X: Eye-in-hand calibration matrix (4x4)
        T_u_left: Left GelSight T(u) parameters (6,)
        T_u_right: Right GelSight T(u) parameters (6,)
        baseline_left: Baseline left GelSight image (no contact)
        baseline_right: Baseline right GelSight image (no contact)
        config: Collection parameters
        touch_threshold: Differential threshold for touch detection

    Returns:
        rgb_frames: List of RGB images (H, W, 3) uint8
        depth_frames: List of depth maps (H, W) float32
        gs_left_frames: List of left GelSight images (H, W, 3) uint8
        gs_right_frames: List of right GelSight images (H, W, 3) uint8
        poses_left: List of left GelSight poses (4, 4) float32
        poses_right: List of right GelSight poses (4, 4) float32
        timestamps: List of timestamps (float64)
        contact_frame: Frame index when contact detected (-1 if none)
        max_press_frame: Frame index at max press depth
        press_depth_mm: Press depth in millimeters
    """
    # Frame storage
    rgb_frames = []
    depth_frames = []
    gs_left_frames = []
    gs_right_frames = []
    poses_left = []
    poses_right = []
    timestamps = []

    # State tracking
    contact_frame = -1
    frame_idx = 0
    frame_interval = 1.0 / config.sample_rate  # Time between frames
    press_depth_mm = 0.0

    logger.info("Starting sample collection...")

    # Main collection loop - runs until max press depth reached
    while True:
        start_time = time.time()

        # ---------------------------------------------------------------------
        # Step 1: Capture all sensors synchronously
        # ---------------------------------------------------------------------
        rgb, depth = realsense.capture()  # (H, W, 3) uint8, (H, W) float32
        gs_left = gelsight_left.capture()  # (H, W, 3) uint8
        gs_right = gelsight_right.capture()  # (H, W, 3) uint8

        # ---------------------------------------------------------------------
        # Step 2: Get robot state
        # ---------------------------------------------------------------------
        T_base_to_ee = robot.get_ee_pose()  # (4, 4) end-effector in base frame
        gripper_opening = robot.get_gripper_opening()  # meters

        # ---------------------------------------------------------------------
        # Step 3: Compute GelSight poses using calibration
        # T_base_to_gelsight = T_base_to_ee @ X @ T_cam_to_gelsight
        # where T_cam_to_gelsight = T(u) = t0 + k * gripper_opening
        # ---------------------------------------------------------------------
        pose = compute_both_gelsight_poses(
            T_base_to_ee, X, T_u_left, T_u_right, gripper_opening
        )  # (2, 4, 4) [left, right]

        # ---------------------------------------------------------------------
        # Step 4: Store frame data
        # ---------------------------------------------------------------------
        rgb_frames.append(rgb)
        depth_frames.append(depth)
        gs_left_frames.append(gs_left)
        gs_right_frames.append(gs_right)
        poses_left.append(pose[0])  # (4, 4) left
        poses_right.append(pose[1])  # (4, 4) right
        timestamps.append(time.time())

        # ---------------------------------------------------------------------
        # Step 5: Check for touch (first time only)
        # Touch = GelSight differential exceeds threshold
        # ---------------------------------------------------------------------
        if contact_frame < 0:
            touched_left, diff_left = detect_touch_differential(
                gs_left, baseline_left, touch_threshold
            )
            touched_right, diff_right = detect_touch_differential(
                gs_right, baseline_right, touch_threshold
            )
            if touched_left or touched_right:
                contact_frame = frame_idx
                logger.info(
                    f"Touch detected at frame {frame_idx} "
                    f"(diff L={diff_left:.1f}, R={diff_right:.1f})"
                )

        # ---------------------------------------------------------------------
        # Step 6: Check for max press depth (after contact)
        # Press depth = Z distance traveled since contact
        # ---------------------------------------------------------------------
        if contact_frame >= 0:
            # Z position of left GelSight at contact
            initial_z = poses_left[contact_frame][2, 3]
            current_z = pose[0, 2, 3]
            press_depth = initial_z - current_z  # Positive when pressing down
            press_depth_mm = press_depth * 1000.0  # Convert to mm

            if press_depth >= config.max_press_depth:
                logger.info(f"Max press reached at frame {frame_idx}")
                break

        # ---------------------------------------------------------------------
        # Step 7: Move robot down for next frame
        # ---------------------------------------------------------------------
        robot.move_down(config.step_size)  # 1mm step
        frame_idx += 1

        # ---------------------------------------------------------------------
        # Step 8: Maintain frame rate
        # ---------------------------------------------------------------------
        elapsed = time.time() - start_time
        if elapsed < frame_interval:
            time.sleep(frame_interval - elapsed)

    max_press_frame = frame_idx

    return (
        rgb_frames,
        depth_frames,
        gs_left_frames,
        gs_right_frames,
        poses_left,
        poses_right,
        timestamps,
        contact_frame,
        max_press_frame,
        press_depth_mm,
    )


def run_collection(
    dataset_dir: str | Path,
    object_id: str = "test_object",
    gelsight_left_id: int = 0,
    gelsight_right_id: int = 8,
) -> None:
    config = CollectionConfig()
    dataset_dir = Path(dataset_dir)

    # Connect hardware
    robot = RobotArm()
    realsense = RealSenseCamera(fps=config.sample_rate)
    gelsight_left = GelSightSensor(device_id=gelsight_left_id, fps=config.sample_rate)
    gelsight_right = GelSightSensor(device_id=gelsight_right_id, fps=config.sample_rate)

    # Load calibration
    calibration_dir = dataset_dir / "calibration"
    X, T_u_left, T_u_right = load_calibration(calibration_dir)

    # Writer
    writer = DatasetWriter(dataset_dir)
    writer.write_object(Object(object_id=object_id, description=""))

    try:
        # Assume: object detected, gripper open, robot positioned above object
        # Capture baseline (gripper not touching)
        baseline_left = gelsight_left.capture()
        baseline_right = gelsight_right.capture()

        # Collect one sample
        sample_id = writer.get_next_sample_id()
        (
            rgb_frames,
            depth_frames,
            gs_left_frames,
            gs_right_frames,
            poses_left,
            poses_right,
            timestamps,
            contact_frame,
            max_press_frame,
            press_depth_mm,
        ) = collect_sample(
            robot,
            realsense,
            gelsight_left,
            gelsight_right,
            X,
            T_u_left,
            T_u_right,
            baseline_left,
            baseline_right,
            config,
        )

        # Verify touch
        verified, reason = verify_touch(
            gs_left_frames,
            gs_right_frames,
            baseline_left,
            baseline_right,
            contact_frame,
        )
        if not verified:
            logger.error(f"Touch verification FAILED: {reason}")
            robot.move_up(config.retract_height)
            return

        logger.info(f"Touch VERIFIED: {reason}")

        # Write sample
        sample = Sample(
            sample_id=sample_id,
            object_id=object_id,
            num_frames=len(rgb_frames),
            contact_frame=contact_frame,
            max_press_frame=max_press_frame,
            press_depth_mm=press_depth_mm,
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
        sample_path = writer.write_sample(sample_data)
        logger.info(f"Saved to {sample_path}")

        # Retract
        robot.move_up(config.retract_height)

    finally:
        robot.close()
        realsense.close()
        gelsight_left.close()
        gelsight_right.close()


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Collect ONE touch sample")
    parser.add_argument("--dataset_dir", type=str, default="dataset/")
    parser.add_argument("--object_id", type=str, default="test_object")
    parser.add_argument("--gs_left", type=int, default=0)
    parser.add_argument("--gs_right", type=int, default=8)

    args = parser.parse_args()

    run_collection(
        dataset_dir=args.dataset_dir,
        object_id=args.object_id,
        gelsight_left_id=args.gs_left,
        gelsight_right_id=args.gs_right,
    )


if __name__ == "__main__":
    main()
