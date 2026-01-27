"""
Data collection entry point.

Orchestrates robot, sensors, and data saving for visual-haptic dataset collection.

Usage:
    python -m src.data_collection.run_collection --dataset_dir dataset/

Requires hardware:
    - Trossen arm with gripper
    - RealSense camera
    - GelSight sensors (left + right)

Collection Workflow:
    1. Load calibration (X, T_u_left, T_u_right)
    2. Connect to hardware (robot, RealSense, GelSight x2)
    3. For each object:
        a. User enters object ID and description
        b. Define object boundary (4 corners in robot base frame)
        c. For each sample:
            i.   User positions robot above object
            ii.  Robot moves down in 1mm steps
            iii. At each step: capture RGB, depth, GelSight, compute poses
            iv.  Detect contact (gripper X within 0.3mm of boundary)
            v.   Continue until max press depth (10mm past contact)
            vi.  Save sample data (videos + numpy arrays)
            vii. Retract robot 50mm
    4. Update dataset metadata
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
    ObjectBoundary,
    Sample,
    SampleData,
    Transform4x4,
    TuParams,
    XMatrix,
)

logger = get_logger(__name__)


# =============================================================================
# Contact Detection
# =============================================================================


def detect_contact(
    robot_pose: Transform4x4,
    boundary: ObjectBoundary,
    tolerance: float = 0.0003,
) -> bool:
    """Detect if gripper is in contact with object.

    Position-based contact detection (no force sensor).
    Compares gripper X position against expected contact line.

    Args:
        robot_pose: Current end-effector pose (4x4 matrix)
        boundary: Object boundary with left edge as contact line
        tolerance: Contact threshold in meters (default 0.3mm)

    Returns:
        True if gripper X is within tolerance of boundary left edge
    """
    # Extract gripper position from pose matrix
    robot_xyz = robot_pose[:3, 3]

    # Get expected X at current Z height (interpolates along left edge)
    expected_x = boundary.get_contact_x_at_z(robot_xyz[2])

    # Contact if within tolerance
    return abs(robot_xyz[0] - expected_x) < tolerance


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
    boundary: ObjectBoundary,
    config: CollectionConfig,
) -> tuple[list, list, list, list, list, int, int]:
    """Collect one sample (press sequence).

    Sample collection loop:
        1. Capture all sensors (RGB, depth, GelSight L/R)
        2. Get robot state (end-effector pose, gripper opening)
        3. Compute GelSight poses using calibration
        4. Store frame data
        5. Check for contact (position-based)
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
        boundary: Object boundary for contact detection
        config: Collection parameters (fps, tolerances, etc.)

    Returns:
        rgb_frames: List of RGB images (H, W, 3) uint8
        depth_frames: List of depth maps (H, W) float32
        gs_left_frames: List of left GelSight images (H, W, 3) uint8
        gs_right_frames: List of right GelSight images (H, W, 3) uint8
        poses: List of GelSight poses (2, 4, 4) float32
        contact_frame: Frame index when contact detected (-1 if none)
        max_press_frame: Frame index at max press depth
    """
    # Frame storage
    rgb_frames = []
    depth_frames = []
    gs_left_frames = []
    gs_right_frames = []
    poses = []

    # State tracking
    contact_frame = -1
    frame_idx = 0
    frame_interval = 1.0 / config.sample_rate  # Time between frames

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
        poses.append(pose)

        # ---------------------------------------------------------------------
        # Step 5: Check for contact (first time only)
        # Contact = gripper X within tolerance of object boundary
        # ---------------------------------------------------------------------
        if contact_frame < 0 and detect_contact(
            T_base_to_ee, boundary, config.contact_tolerance
        ):
            contact_frame = frame_idx
            logger.info(f"Contact detected at frame {frame_idx}")

        # ---------------------------------------------------------------------
        # Step 6: Check for max press depth (after contact)
        # Press depth = Z distance traveled since contact
        # ---------------------------------------------------------------------
        if contact_frame >= 0:
            # Z position of left GelSight at contact
            initial_z = poses[contact_frame][0, 2, 3]
            current_z = pose[0, 2, 3]
            press_depth = initial_z - current_z  # Positive when pressing down

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
        poses,
        contact_frame,
        max_press_frame,
    )


def run_collection(
    dataset_dir: str | Path,
    num_samples_per_object: int = 50,
) -> None:
    """Run full data collection session.

    Main entry point for data collection. Handles:
        - Hardware initialization
        - Object loop (user enters object metadata)
        - Sample loop (collect N samples per object)
        - Data saving and cleanup

    Args:
        dataset_dir: Output directory for dataset
        num_samples_per_object: Number of samples to collect per object
    """
    config = CollectionConfig()
    dataset_dir = Path(dataset_dir)

    # =========================================================================
    # Initialization
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Starting data collection")
    logger.info("=" * 60)

    # Load calibration files from dataset/calibration/
    # X.npy: (4, 4) eye-in-hand calibration
    # T_u_left_params.npy: (6,) left GelSight T(u) model
    # T_u_right_params.npy: (6,) right GelSight T(u) model
    calibration_dir = dataset_dir / "calibration"
    X, T_u_left, T_u_right = load_calibration(calibration_dir)
    logger.info("Loaded calibration data")

    # Connect to hardware
    robot = RobotArm()
    realsense = RealSenseCamera()
    gelsight_left = GelSightSensor(device_id=1)  # USB camera ID for left
    gelsight_right = GelSightSensor(device_id=2)  # USB camera ID for right

    # Create dataset writer
    writer = DatasetWriter(dataset_dir)
    logger.info(f"Dataset directory: {dataset_dir}")

    try:
        # =====================================================================
        # Object Loop
        # User enters object metadata, then collects N samples
        # =====================================================================
        while True:
            # Get object info from user
            object_id = input("\nEnter object ID (or 'quit' to stop): ").strip()
            if object_id.lower() == "quit":
                break

            description = input("Enter object description: ").strip()

            # Save object metadata to dataset/objects/{object_id}.json
            obj = Object(object_id=object_id, description=description)
            writer.write_object(obj)
            logger.info(f"Created object: {object_id}")

            # -----------------------------------------------------------------
            # Define object boundary
            # 4 corners in robot base frame define a rectangle
            # Left edge (top_left -> bottom_left) is the contact line
            # TODO: Implement camera-based detection or manual input
            # -----------------------------------------------------------------
            print("\nDefine object boundary (4 corners in base frame):")
            print("TODO: Implement boundary definition from camera")
            boundary = ObjectBoundary(
                top_left=np.array([0.0, 0.0, 0.0]),
                top_right=np.array([0.0, 0.0, 0.0]),
                bottom_right=np.array([0.0, 0.0, 0.0]),
                bottom_left=np.array([0.0, 0.0, 0.0]),
            )

            # =================================================================
            # Sample Loop
            # Collect N press samples for this object
            # =================================================================
            for i in range(num_samples_per_object):
                sample_id = writer.get_next_sample_id()  # "000001", "000002", ...
                logger.info(
                    f"\n--- Sample {sample_id} ({i + 1}/{num_samples_per_object}) ---"
                )

                # Wait for user to position robot above object
                input("Position robot above object and press Enter...")

                # -------------------------------------------------------------
                # Collect sample (press sequence)
                # Robot moves down, captures data until max press
                # -------------------------------------------------------------
                (
                    rgb_frames,
                    depth_frames,
                    gs_left_frames,
                    gs_right_frames,
                    poses,
                    contact_frame,
                    max_press_frame,
                ) = collect_sample(
                    robot,
                    realsense,
                    gelsight_left,
                    gelsight_right,
                    X,
                    T_u_left,
                    T_u_right,
                    boundary,
                    config,
                )

                # -------------------------------------------------------------
                # Package sample data
                # -------------------------------------------------------------
                sample = Sample(
                    sample_id=sample_id,
                    object_id=object_id,
                    contact_frame=contact_frame,
                    max_press_frame=max_press_frame,
                    sample_rate=config.sample_rate,
                    num_frames=len(rgb_frames),
                )

                sample_data = SampleData(
                    sample=sample,
                    rgb=np.array(rgb_frames, dtype=np.uint8),
                    gelsight_left=np.array(gs_left_frames, dtype=np.uint8),
                    gelsight_right=np.array(gs_right_frames, dtype=np.uint8),
                    depth=np.array(depth_frames, dtype=np.float32),
                    poses=np.array(poses, dtype=np.float32),
                )

                # -------------------------------------------------------------
                # Save sample to disk
                # dataset/samples/{sample_id}/
                #   sample.json, rgb.mp4, depth.npy, gelsight_*.mp4, poses.npy
                # -------------------------------------------------------------
                sample_path = writer.write_sample(sample_data)
                logger.info(f"Saved sample to {sample_path}")

                # -------------------------------------------------------------
                # Retract robot for next sample
                # -------------------------------------------------------------
                robot.move_up(config.retract_height)  # 50mm

        # =====================================================================
        # Finalize
        # =====================================================================
        metadata = writer.update_metadata()
        logger.info(
            f"\nCollection complete: {metadata.num_samples} samples, "
            f"{metadata.num_objects} objects"
        )

    finally:
        # =====================================================================
        # Cleanup - always disconnect hardware
        # =====================================================================
        robot.close()
        realsense.close()
        gelsight_left.close()
        gelsight_right.close()


# =============================================================================
# CLI
# =============================================================================


def main():
    parser = argparse.ArgumentParser(description="Run data collection")
    parser.add_argument(
        "--dataset_dir",
        type=str,
        default="dataset/",
        help="Dataset output directory",
    )
    parser.add_argument(
        "--samples_per_object",
        type=int,
        default=50,
        help="Number of samples per object",
    )

    args = parser.parse_args()

    run_collection(
        dataset_dir=args.dataset_dir,
        num_samples_per_object=args.samples_per_object,
    )


if __name__ == "__main__":
    main()
