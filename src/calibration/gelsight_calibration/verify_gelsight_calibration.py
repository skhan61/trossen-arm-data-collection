#!/usr/bin/env python3
"""
Verify GelSight calibration T(u) linear models.

This script:
1. Loads T(u) calibration parameters from .npy files
2. Displays model parameters
3. Visualizes predicted sensor positions for gripper opening range
4. Computes expected sensor gap at different openings

Usage:
    python src/calibration/gelsight_calibration/verify_gelsight_calibration.py
    python src/calibration/gelsight_calibration/verify_gelsight_calibration.py --calibration_dir=dataset/calibration
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_CALIBRATION_DIR = PROJECT_ROOT / "dataset" / "calibration"
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

LOG_FILE = (
    LOG_DIR
    / f"verify_gelsight_calibration_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(LOG_FILE, mode="w"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


def load_tu_params(calibration_dir: Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Load T(u) parameters from .npy files.

    Args:
        calibration_dir: Path to calibration directory

    Returns:
        (T_u_left, T_u_right) - each is (6,) array [t0_x, t0_y, t0_z, k_x, k_y, k_z]
        or None if file not found
    """
    left_file = calibration_dir / "T_u_left_params.npy"
    right_file = calibration_dir / "T_u_right_params.npy"

    left_params = None
    right_params = None

    if left_file.exists():
        left_params = np.load(left_file)
        logger.info(f"Loaded left T(u) params from {left_file}")
    else:
        logger.warning(f"Left T(u) params not found: {left_file}")

    if right_file.exists():
        right_params = np.load(right_file)
        logger.info(f"Loaded right T(u) params from {right_file}")
    else:
        logger.warning(f"Right T(u) params not found: {right_file}")

    return left_params, right_params


def predict_position(params: np.ndarray, u: float) -> np.ndarray:
    """
    Predict 3D position using T(u) = t0 + k * u.

    Args:
        params: (6,) array [t0_x, t0_y, t0_z, k_x, k_y, k_z]
        u: Gripper opening in meters

    Returns:
        Predicted [x, y, z] position in meters
    """
    t0 = params[:3]
    k = params[3:6]
    return t0 + k * u


def compute_sensor_gap(left_params: np.ndarray, right_params: np.ndarray, u: float) -> float:
    """
    Compute distance between left and right sensor centers.

    Args:
        left_params: Left T(u) parameters
        right_params: Right T(u) parameters
        u: Gripper opening in meters

    Returns:
        Distance between sensor centers in meters
    """
    left_pos = predict_position(left_params, u)
    right_pos = predict_position(right_params, u)
    return float(np.linalg.norm(left_pos - right_pos))


def verify_calibration(calibration_dir: str | None = None):
    """
    Verify GelSight T(u) calibration.

    Args:
        calibration_dir: Path to calibration directory (default: dataset/calibration)
    """
    if calibration_dir is None:
        calibration_dir = DEFAULT_CALIBRATION_DIR
    else:
        calibration_dir = Path(calibration_dir)

    logger.info("=" * 70)
    logger.info("GelSight T(u) Calibration Verification")
    logger.info("=" * 70)
    logger.info(f"Calibration dir: {calibration_dir}")
    logger.info("")

    # Load T(u) parameters
    left_params, right_params = load_tu_params(calibration_dir)

    if left_params is None and right_params is None:
        logger.error("No calibration files found!")
        return

    # ========================================================================
    # Display LEFT sensor parameters
    # ========================================================================
    if left_params is not None:
        logger.info("\n" + "=" * 50)
        logger.info("LEFT SENSOR T(u) PARAMETERS")
        logger.info("=" * 50)
        logger.info("T(u) = t0 + k * u, where u = gripper opening (meters)")
        logger.info("")
        logger.info("Parameters [t0_x, t0_y, t0_z, k_x, k_y, k_z]:")
        logger.info(f"  Raw: {left_params}")
        logger.info("")
        logger.info("Interpreted:")
        logger.info(f"  t0 (offset):    [{left_params[0]*1000:.3f}, {left_params[1]*1000:.3f}, {left_params[2]*1000:.3f}] mm")
        logger.info(f"  k  (slope):     [{left_params[3]*1000:.3f}, {left_params[4]*1000:.3f}, {left_params[5]*1000:.3f}] mm/m")
        logger.info("")
        logger.info("Physical interpretation:")
        logger.info(f"  - At u=0 (closed): sensor at ({left_params[0]*1000:.2f}, {left_params[1]*1000:.2f}, {left_params[2]*1000:.2f}) mm")
        logger.info(f"  - X moves {left_params[3]*1000:.2f} mm per 1m of gripper opening")
        logger.info(f"    (or {left_params[3]:.4f} mm per 1mm opening)")

    # ========================================================================
    # Display RIGHT sensor parameters
    # ========================================================================
    if right_params is not None:
        logger.info("\n" + "=" * 50)
        logger.info("RIGHT SENSOR T(u) PARAMETERS")
        logger.info("=" * 50)
        logger.info("T(u) = t0 + k * u, where u = gripper opening (meters)")
        logger.info("")
        logger.info("Parameters [t0_x, t0_y, t0_z, k_x, k_y, k_z]:")
        logger.info(f"  Raw: {right_params}")
        logger.info("")
        logger.info("Interpreted:")
        logger.info(f"  t0 (offset):    [{right_params[0]*1000:.3f}, {right_params[1]*1000:.3f}, {right_params[2]*1000:.3f}] mm")
        logger.info(f"  k  (slope):     [{right_params[3]*1000:.3f}, {right_params[4]*1000:.3f}, {right_params[5]*1000:.3f}] mm/m")
        logger.info("")
        logger.info("Physical interpretation:")
        logger.info(f"  - At u=0 (closed): sensor at ({right_params[0]*1000:.2f}, {right_params[1]*1000:.2f}, {right_params[2]*1000:.2f}) mm")
        logger.info(f"  - X moves {right_params[3]*1000:.2f} mm per 1m of gripper opening")
        logger.info(f"    (or {right_params[3]:.4f} mm per 1mm opening)")

    # ========================================================================
    # Compute predictions for typical gripper openings
    # ========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("PREDICTED POSITIONS AT TYPICAL OPENINGS")
    logger.info("=" * 50)

    # Typical gripper opening range (26mm to 42mm based on calibration data)
    openings_mm = [26, 30, 35, 40, 42]

    for u_mm in openings_mm:
        u_m = u_mm / 1000.0
        logger.info(f"\nGripper opening: {u_mm} mm ({u_m:.3f} m)")

        if left_params is not None:
            left_pos = predict_position(left_params, u_m)
            logger.info(f"  LEFT:  ({left_pos[0]*1000:.2f}, {left_pos[1]*1000:.2f}, {left_pos[2]*1000:.2f}) mm")

        if right_params is not None:
            right_pos = predict_position(right_params, u_m)
            logger.info(f"  RIGHT: ({right_pos[0]*1000:.2f}, {right_pos[1]*1000:.2f}, {right_pos[2]*1000:.2f}) mm")

        if left_params is not None and right_params is not None:
            gap = compute_sensor_gap(left_params, right_params, u_m)
            logger.info(f"  GAP:   {gap*1000:.2f} mm")

    # ========================================================================
    # Create visualization
    # ========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("Creating visualization...")
    logger.info("=" * 50)

    # Generate smooth range of openings
    u_range_mm = np.linspace(20, 45, 100)
    u_range_m = u_range_mm / 1000.0

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("GelSight T(u) Calibration Verification", fontsize=14)

    # Plot 1: X position vs gripper opening
    ax1 = axes[0, 0]
    if left_params is not None:
        left_x = [predict_position(left_params, u)[0] * 1000 for u in u_range_m]
        ax1.plot(u_range_mm, left_x, "b-", linewidth=2, label="Left sensor")
    if right_params is not None:
        right_x = [predict_position(right_params, u)[0] * 1000 for u in u_range_m]
        ax1.plot(u_range_mm, right_x, "g-", linewidth=2, label="Right sensor")
    ax1.set_xlabel("Gripper Opening (mm)")
    ax1.set_ylabel("X Position (mm)")
    ax1.set_title("X Position vs Gripper Opening")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Y position vs gripper opening
    ax2 = axes[0, 1]
    if left_params is not None:
        left_y = [predict_position(left_params, u)[1] * 1000 for u in u_range_m]
        ax2.plot(u_range_mm, left_y, "b-", linewidth=2, label="Left sensor")
    if right_params is not None:
        right_y = [predict_position(right_params, u)[1] * 1000 for u in u_range_m]
        ax2.plot(u_range_mm, right_y, "g-", linewidth=2, label="Right sensor")
    ax2.set_xlabel("Gripper Opening (mm)")
    ax2.set_ylabel("Y Position (mm)")
    ax2.set_title("Y Position vs Gripper Opening")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Z position vs gripper opening
    ax3 = axes[1, 0]
    if left_params is not None:
        left_z = [predict_position(left_params, u)[2] * 1000 for u in u_range_m]
        ax3.plot(u_range_mm, left_z, "b-", linewidth=2, label="Left sensor")
    if right_params is not None:
        right_z = [predict_position(right_params, u)[2] * 1000 for u in u_range_m]
        ax3.plot(u_range_mm, right_z, "g-", linewidth=2, label="Right sensor")
    ax3.set_xlabel("Gripper Opening (mm)")
    ax3.set_ylabel("Z Position (mm)")
    ax3.set_title("Z Position vs Gripper Opening")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Sensor gap vs gripper opening
    ax4 = axes[1, 1]
    if left_params is not None and right_params is not None:
        gaps = [compute_sensor_gap(left_params, right_params, u) * 1000 for u in u_range_m]
        ax4.plot(u_range_mm, gaps, "r-", linewidth=2)
        ax4.set_xlabel("Gripper Opening (mm)")
        ax4.set_ylabel("Sensor Gap (mm)")
        ax4.set_title("Sensor Gap vs Gripper Opening")
        ax4.grid(True, alpha=0.3)

        # Add reference line for gripper opening = gap
        ax4.plot(u_range_mm, u_range_mm, "k--", alpha=0.5, label="Gap = Opening")
        ax4.legend()
    else:
        ax4.text(0.5, 0.5, "Need both sensors for gap calculation",
                ha="center", va="center", transform=ax4.transAxes)
        ax4.set_title("Sensor Gap vs Gripper Opening")

    plt.tight_layout()

    # Save plot
    output_file = Path(__file__).parent / "calibration_verification.png"
    plt.savefig(output_file, dpi=150)
    logger.info(f"Saved visualization to {output_file}")

    # ========================================================================
    # 3D visualization of sensor trajectories
    # ========================================================================
    fig2 = plt.figure(figsize=(12, 5))

    # Left sensor 3D trajectory
    ax3d_left = fig2.add_subplot(121, projection="3d")
    if left_params is not None:
        positions = np.array([predict_position(left_params, u) for u in u_range_m])
        scatter = ax3d_left.scatter(
            positions[:, 0] * 1000,
            positions[:, 1] * 1000,
            positions[:, 2] * 1000,
            c=u_range_mm,
            cmap="viridis",
            s=10,
        )
        ax3d_left.set_xlabel("X (mm)")
        ax3d_left.set_ylabel("Y (mm)")
        ax3d_left.set_zlabel("Z (mm)")
        ax3d_left.set_title("LEFT Sensor Trajectory")
        fig2.colorbar(scatter, ax=ax3d_left, label="Gripper Opening (mm)", shrink=0.6)
    else:
        ax3d_left.text2D(0.5, 0.5, "No left sensor data", ha="center", va="center",
                        transform=ax3d_left.transAxes)

    # Right sensor 3D trajectory
    ax3d_right = fig2.add_subplot(122, projection="3d")
    if right_params is not None:
        positions_r = np.array([predict_position(right_params, u) for u in u_range_m])
        scatter_r = ax3d_right.scatter(
            positions_r[:, 0] * 1000,
            positions_r[:, 1] * 1000,
            positions_r[:, 2] * 1000,
            c=u_range_mm,
            cmap="viridis",
            s=10,
        )
        ax3d_right.set_xlabel("X (mm)")
        ax3d_right.set_ylabel("Y (mm)")
        ax3d_right.set_zlabel("Z (mm)")
        ax3d_right.set_title("RIGHT Sensor Trajectory")
        fig2.colorbar(scatter_r, ax=ax3d_right, label="Gripper Opening (mm)", shrink=0.6)
    else:
        ax3d_right.text2D(0.5, 0.5, "No right sensor data", ha="center", va="center",
                        transform=ax3d_right.transAxes)

    plt.tight_layout()

    output_file2 = Path(__file__).parent / "calibration_3d_trajectory.png"
    plt.savefig(output_file2, dpi=150)
    logger.info(f"Saved 3D trajectory plot to {output_file2}")

    # ========================================================================
    # Summary and sanity checks
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("CALIBRATION SANITY CHECKS")
    logger.info("=" * 70)

    issues = []

    if left_params is not None:
        # LEFT sensor should have NEGATIVE k_x (moves left/outward when gripper opens)
        if left_params[3] < 0:
            logger.info(f"  LEFT: X slope = {left_params[3]*1000:.3f} mm/m (negative - moves left/outward when opening - OK)")
        else:
            logger.warning(f"  LEFT: X slope is positive ({left_params[3]*1000:.3f} mm/m) - sensor moves RIGHT/inward with opening - CHECK CALIBRATION")

        # Check if Y/Z slopes are small (should be mostly constant)
        if abs(left_params[4]) > 0.1:  # More than 100mm/m
            logger.warning(f"  LEFT: Y slope is large ({left_params[4]*1000:.3f} mm/m) - check calibration")
        if abs(left_params[5]) > 0.1:
            logger.warning(f"  LEFT: Z slope is large ({left_params[5]*1000:.3f} mm/m) - check calibration")

    if right_params is not None:
        # RIGHT sensor should have POSITIVE k_x (moves right/outward when gripper opens)
        if right_params[3] > 0:
            logger.info(f"  RIGHT: X slope = {right_params[3]*1000:.3f} mm/m (positive - moves right/outward when opening - OK)")
        else:
            logger.warning(f"  RIGHT: X slope is negative ({right_params[3]*1000:.3f} mm/m) - sensor moves LEFT/inward with opening - CHECK CALIBRATION")

        # Check if Y/Z slopes are small
        if abs(right_params[4]) > 0.1:
            logger.warning(f"  RIGHT: Y slope is large ({right_params[4]*1000:.3f} mm/m) - check calibration")
        if abs(right_params[5]) > 0.1:
            logger.warning(f"  RIGHT: Z slope is large ({right_params[5]*1000:.3f} mm/m) - check calibration")

    if left_params is not None and right_params is not None:
        # Check symmetry
        x_slope_diff = abs(left_params[3] + right_params[3])  # Should be close to 0 if symmetric
        logger.info(f"\n  Symmetry check: |k_left_x + k_right_x| = {x_slope_diff*1000:.3f} mm/m")
        if x_slope_diff < 0.01:  # Less than 10mm/m difference
            logger.info("    Sensors appear symmetric (good)")
        else:
            logger.warning("    Sensors may not be symmetric - check mounting")

    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION COMPLETE")
    logger.info("=" * 70)

    # Show plots
    plt.show()


if __name__ == "__main__":
    import fire

    fire.Fire(verify_calibration)
