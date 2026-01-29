"""
Integrated data collection: positioning + sample collection.

BLUEPRINT:
=========
Part 1: Scan and detect object (position gripper over object)
Part 2: Collect sample (close gripper with GelSight recording)

Usage:
    python -m src.data_collection.collect_run --num_samples 5 --object_id my_object
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
    num_samples: int = 5,
    object_id: str = "test_object",
    gs_left_id: int = 0,
    gs_right_id: int = 8,
    dataset_dir: str = "dataset/",
    fps: int = 30,
    debug: bool = False,
):
    """
    Collect samples: scan/detect then collect with GelSight.

    Args:
        num_samples: Number of samples to collect
        object_id: Object identifier for samples
        gs_left_id: Left GelSight video device ID
        gs_right_id: Right GelSight video device ID
        dataset_dir: Dataset root directory (for calibration)
        fps: Camera frame rate
        debug: Show debug CV2 display during collection
    """
    dataset_dir = Path(dataset_dir)

    # Load calibration (full: X, T_u_left, T_u_right)
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
    camera = RealSenseCameraWithIntrinsics(fps=fps)
    logger.info("RealSense connected")

    logger.info(
        f"Connecting to GelSight (L={gs_left_id}, R={gs_right_id}, fps={fps})..."
    )
    gs_left = GelSightSensor(device_id=gs_left_id, fps=fps)
    gs_right = GelSightSensor(device_id=gs_right_id, fps=fps)
    logger.info("GelSight sensors connected")

    # Warm up sensors
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

    step_down_mm = 10.0  # Go down 1mm between samples

    try:
        # Go to home
        logger.info("Going to home position...")
        robot.go_home()
        logger.info("At home position")

        # =============================================
        # PART 1: Scan and detect object (ONCE)
        # =============================================
        # - Gravity mode: manually position robot
        # - Click to detect object
        # - Press 'g' to move gripper over object
        result = position_and_detect(robot, camera, X)
        if result is None:
            logger.info("User cancelled. Ending.")
            return

        target_x, target_y, target_z = result
        current_z = target_z
        logger.info(
            f"Gripper positioned at: x={target_x:.3f} y={target_y:.3f} z={target_z:.3f}"
        )

        # =============================================
        # PART 2: Collect N samples (go down 1mm each time)
        # =============================================
        logger.info(
            f"\nStarting {num_samples} sample collection, {step_down_mm}mm step down each"
        )

        for i in range(num_samples):
            logger.info(f"\n{'='*50}")
            logger.info(f"Sample {i+1}/{num_samples}")
            logger.info(f"{'='*50}")

            # Go down 1mm (except first sample)
            if i > 0:
                current_z -= step_down_mm / 1000.0
                robot.move_to_cartesian(target_x, target_y, current_z, duration=0.5)
                logger.info(f"Moved down to z={current_z*1000:.1f}mm")

            # Collect sample
            sample_id = f"{i+1:06d}"
            logger.info(f"Collecting sample {sample_id}...")

            sample_data, _ = collect_sample(
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

            if sample_data is None:
                logger.warning(f"Sample {sample_id} collection failed")
            else:
                logger.info(
                    f"Sample {sample_id} collected: {sample_data.sample.num_frames} frames"
                )

            # Open gripper for next sample (stay at position)
            robot.open_gripper(position=0.04)
            logger.info(f"Sample {i+1} done!")

        # Update metadata after all samples collected
        writer.update_metadata()
        logger.info("Dataset metadata updated")

        # Cleanup after all samples: move up, return home
        logger.info("\nAll samples collected. Returning home...")
        cleanup_and_return(robot, target_x, target_y, current_z)

        logger.info("\n=== Collection complete! ===")

    except Exception as e:
        logger.error(f"Failed: {e}", exc_info=True)
        raise
    finally:
        cv2.destroyAllWindows()
        robot.close()
        camera.close()
        gs_left.close()
        gs_right.close()


if __name__ == "__main__":
    import fire

    fire.Fire(main)
