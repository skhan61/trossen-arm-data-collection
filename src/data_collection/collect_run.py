"""
Integrated data collection: positioning + sample collection.

Uses modular functions from:
    - scan_and_detect.py: position_and_detect(), cleanup_and_return()
    - collect_one_sample.py: collect_sample()

Workflow per sample:
1. Gravity mode - manually position robot
2. Click to detect object, press 'g' to move
3. Collect sample (close gripper, capture frames)
4. Cleanup: open gripper, move up 50mm, return home
5. Repeat for N samples

Usage:
    python -m src.data_collection.collect_run \
        --num_samples 10 --gs_left_id 0 --gs_right_id 8 --dataset_dir dataset/
"""

from pathlib import Path

import cv2

from src.data_collection.collect_one_sample import collect_sample
from src.data_collection.scan_and_detect import (
    RealSenseCameraWithIntrinsics,
    cleanup_and_return,
    mouse_callback,
    position_and_detect,
)
from src.data_collection.writer import DatasetWriter
from src.robot.trossen_arm import TrossenArm
from src.sensors.gelsight import GelSightSensor
from src.utils.log import get_logger
from src.utils.transforms import load_calibration
from src.utils.types import Object

logger = get_logger(__name__)


def main(
    num_samples: int = 10,
    gs_left_id: int = 0,
    gs_right_id: int = 8,
    dataset_dir: str = "dataset/",
    object_id: str = "test_object",
    fps: int = 30,
    debug: bool = False,
):
    """
    Collect N samples with positioning + collection loop.

    Args:
        num_samples: Number of samples to collect
        gs_left_id: Left GelSight video device ID
        gs_right_id: Right GelSight video device ID
        dataset_dir: Dataset root directory
        object_id: Object identifier
        fps: Capture frame rate
        debug: Show CV2 displays
    """
    dataset_dir = Path(dataset_dir)

    # Load calibration
    calibration_dir = dataset_dir / "calibration"
    if not calibration_dir.exists():
        logger.error(f"Calibration not found: {calibration_dir}")
        return
    X, T_u_left, T_u_right = load_calibration(calibration_dir)
    logger.info("Calibration loaded")

    # Connect all hardware once
    logger.info("Connecting to robot...")
    robot = TrossenArm()
    logger.info("Robot connected")

    logger.info(f"Connecting to RealSense (fps={fps})...")
    camera = RealSenseCameraWithIntrinsics(fps=fps)
    logger.info("RealSense connected")

    logger.info(f"Connecting to GelSight (L={gs_left_id}, R={gs_right_id})...")
    gs_left = GelSightSensor(device_id=gs_left_id, fps=fps)
    gs_right = GelSightSensor(device_id=gs_right_id, fps=fps)
    logger.info("GelSight sensors connected")

    # Sensor warmup
    logger.info("Warming up sensors...")
    for _ in range(10):
        camera.capture()
        gs_left.capture()
        gs_right.capture()
    logger.info("Sensors warmed up")

    # Writer
    writer = DatasetWriter(dataset_dir)
    writer.write_object(Object(object_id=object_id, description=""))

    # Setup CV2 window
    cv2.namedWindow("Detection")
    cv2.setMouseCallback("Detection", mouse_callback)

    try:
        # Go to home position
        logger.info("Going to home position...")
        robot.go_home()
        logger.info("At home position")

        # Get next sample ID
        samples_dir = dataset_dir / "samples"
        existing = list(samples_dir.glob("*")) if samples_dir.exists() else []
        start_id = len(existing)

        logger.info(f"=== Starting collection: {num_samples} samples ===")

        for i in range(num_samples):
            sample_id = f"{start_id + i:06d}"
            logger.info(f"\n{'='*50}")
            logger.info(f"Sample {i+1}/{num_samples} (ID: {sample_id})")
            logger.info(f"{'='*50}")

            # Module 1: Position and detect (from scan_and_detect.py)
            result = position_and_detect(robot, camera, X)
            if result is None:
                logger.info("User cancelled. Ending collection.")
                break

            target_x, target_y, target_z = result

            # Module 2: Collect sample (from collect_one_sample.py)
            sample_data, final_z = collect_sample(
                robot=robot,
                realsense=camera,
                gs_left=gs_left,
                gs_right=gs_right,
                X=X,
                T_u_left=T_u_left,
                T_u_right=T_u_right,
                sample_id=sample_id,
                object_id=object_id,
                writer=writer,
                fps=fps,
                debug=debug,
            )

            # Module 3: Cleanup and return (from scan_and_detect.py)
            cleanup_and_return(robot, target_x, target_y, final_z)

            logger.info(f"Sample {sample_id} complete!")

            # Update metadata after each sample
            writer.update_metadata()

        logger.info(f"\n=== Collection complete! ===")

    finally:
        cv2.destroyAllWindows()
        robot.close()
        camera.close()
        gs_left.close()
        gs_right.close()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
