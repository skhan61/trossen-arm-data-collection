#!/usr/bin/env python3
"""
Verify GelSight calibration T(u) linear models.

This script:
1. Loads calibration results
2. Computes predicted positions using T(u) = t0 + k*u
3. Compares predicted vs actual detected positions
4. Visualizes results with diagnostic plots

Usage:
    python src/calibration/gelsight_calibration/verify_gelsight_calibration.py
    python src/calibration/gelsight_calibration/verify_gelsight_calibration.py --pose_name=home
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import logging
from datetime import datetime

# ============================================================================
# Configuration
# ============================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = Path(__file__).parent / "gelsight_calibration_data"
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


def predict_position(model: dict, u: float) -> np.ndarray:
    """
    Predict 3D position using linear model T(u) = t0 + k*u.

    Args:
        model: Dict with 'models' containing x, y, z linear params
        u: Gripper opening in meters

    Returns:
        Predicted [x, y, z] position in meters
    """
    models = model["models"]
    return np.array(
        [
            models["x"]["t0"] + models["x"]["k"] * u,
            models["y"]["t0"] + models["y"]["k"] * u,
            models["z"]["t0"] + models["z"]["k"] * u,
        ]
    )


def verify_calibration():
    """
    Verify GelSight calibration.
    """
    logger.info("=" * 70)
    logger.info("GelSight Calibration Verification")
    logger.info("=" * 70)
    logger.info(f"Data dir: {DATA_DIR}")
    logger.info("")

    # Load calibration results
    calib_file = DATA_DIR / "gelsight_calibration.json"
    if not calib_file.exists():
        logger.error(f"Calibration file not found: {calib_file}")
        return

    with open(calib_file) as f:
        calib = json.load(f)

    samples = calib["samples"]
    left_model = calib.get("linear_model_left")
    right_model = calib.get("linear_model_right")

    logger.info(f"Loaded {len(samples)} samples")

    # ========================================================================
    # Analyze LEFT sensor
    # ========================================================================
    if left_model:
        logger.info("\n" + "=" * 50)
        logger.info("LEFT SENSOR VERIFICATION")
        logger.info("=" * 50)

        left_samples = [s for s in samples if s.get("left_3d") is not None]
        logger.info(f"Samples with left detection: {len(left_samples)}")

        u_values = np.array([s["gripper_opening_m"] for s in left_samples])
        actual_positions = np.array([s["left_3d"] for s in left_samples])
        predicted_positions = np.array(
            [predict_position(left_model, u) for u in u_values]
        )

        errors = actual_positions - predicted_positions
        errors_mm = errors * 1000

        logger.info("\nPrediction errors (mm):")
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            axis_errors = errors_mm[:, axis_idx]
            logger.info(
                f"  {axis_name}: mean={np.mean(axis_errors):.3f}, "
                f"std={np.std(axis_errors):.3f}, "
                f"max={np.max(np.abs(axis_errors)):.3f}"
            )

        total_error = np.linalg.norm(errors, axis=1) * 1000
        logger.info(
            f"\nTotal 3D error (mm): mean={np.mean(total_error):.3f}, "
            f"std={np.std(total_error):.3f}, max={np.max(total_error):.3f}"
        )

        # Print model parameters
        logger.info("\nLinear model parameters:")
        for axis in ["x", "y", "z"]:
            m = left_model["models"][axis]
            logger.info(
                f"  {axis}: t0={m['t0'] * 1000:.3f}mm, k={m['k'] * 1000:.3f}mm/m, R²={m['r_squared']:.4f}"
            )

    # ========================================================================
    # Analyze RIGHT sensor
    # ========================================================================
    if right_model:
        logger.info("\n" + "=" * 50)
        logger.info("RIGHT SENSOR VERIFICATION")
        logger.info("=" * 50)

        right_samples = [s for s in samples if s.get("right_3d") is not None]
        logger.info(f"Samples with right detection: {len(right_samples)}")

        u_values_r = np.array([s["gripper_opening_m"] for s in right_samples])
        actual_positions_r = np.array([s["right_3d"] for s in right_samples])
        predicted_positions_r = np.array(
            [predict_position(right_model, u) for u in u_values_r]
        )

        errors_r = actual_positions_r - predicted_positions_r
        errors_mm_r = errors_r * 1000

        logger.info("\nPrediction errors (mm):")
        for axis_idx, axis_name in enumerate(["x", "y", "z"]):
            axis_errors = errors_mm_r[:, axis_idx]
            logger.info(
                f"  {axis_name}: mean={np.mean(axis_errors):.3f}, "
                f"std={np.std(axis_errors):.3f}, "
                f"max={np.max(np.abs(axis_errors)):.3f}"
            )

        total_error_r = np.linalg.norm(errors_r, axis=1) * 1000
        logger.info(
            f"\nTotal 3D error (mm): mean={np.mean(total_error_r):.3f}, "
            f"std={np.std(total_error_r):.3f}, max={np.max(total_error_r):.3f}"
        )

        # Print model parameters
        logger.info("\nLinear model parameters:")
        for axis in ["x", "y", "z"]:
            m = right_model["models"][axis]
            logger.info(
                f"  {axis}: t0={m['t0'] * 1000:.3f}mm, k={m['k'] * 1000:.3f}mm/m, R²={m['r_squared']:.4f}"
            )

    # ========================================================================
    # Create visualization
    # ========================================================================
    logger.info("\n" + "=" * 50)
    logger.info("Creating visualization...")
    logger.info("=" * 50)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle("GelSight Calibration Verification", fontsize=14)

    axis_names = ["X", "Y", "Z"]

    for axis_idx, axis_name in enumerate(axis_names):
        # LEFT sensor plot
        ax_left = axes[axis_idx, 0]
        if left_model and len(left_samples) > 0:
            u_mm = u_values * 1000
            actual_mm = actual_positions[:, axis_idx] * 1000
            predicted_mm = predicted_positions[:, axis_idx] * 1000

            ax_left.scatter(u_mm, actual_mm, c="blue", alpha=0.6, label="Actual", s=30)
            ax_left.plot(
                u_mm, predicted_mm, "r-", linewidth=2, label="Predicted (T(u))"
            )

            m = left_model["models"][axis_name.lower()]
            ax_left.set_title(f"LEFT - {axis_name} axis (R²={m['r_squared']:.4f})")
            ax_left.set_xlabel("Gripper Opening (mm)")
            ax_left.set_ylabel(f"{axis_name} Position (mm)")
            ax_left.legend()
            ax_left.grid(True, alpha=0.3)
        else:
            ax_left.text(0.5, 0.5, "No left sensor data", ha="center", va="center")
            ax_left.set_title(f"LEFT - {axis_name} axis")

        # RIGHT sensor plot
        ax_right = axes[axis_idx, 1]
        if right_model and len(right_samples) > 0:
            u_mm_r = u_values_r * 1000
            actual_mm_r = actual_positions_r[:, axis_idx] * 1000
            predicted_mm_r = predicted_positions_r[:, axis_idx] * 1000

            ax_right.scatter(
                u_mm_r, actual_mm_r, c="green", alpha=0.6, label="Actual", s=30
            )
            ax_right.plot(
                u_mm_r, predicted_mm_r, "r-", linewidth=2, label="Predicted (T(u))"
            )

            m = right_model["models"][axis_name.lower()]
            ax_right.set_title(f"RIGHT - {axis_name} axis (R²={m['r_squared']:.4f})")
            ax_right.set_xlabel("Gripper Opening (mm)")
            ax_right.set_ylabel(f"{axis_name} Position (mm)")
            ax_right.legend()
            ax_right.grid(True, alpha=0.3)
        else:
            ax_right.text(0.5, 0.5, "No right sensor data", ha="center", va="center")
            ax_right.set_title(f"RIGHT - {axis_name} axis")

    plt.tight_layout()

    # Save plot
    output_file = Path(__file__).parent / "calibration_verification.png"
    plt.savefig(output_file, dpi=150)
    logger.info(f"Saved visualization to {output_file}")

    # ========================================================================
    # Create residual plots
    # ========================================================================
    fig2, axes2 = plt.subplots(2, 3, figsize=(15, 8))
    fig2.suptitle("Calibration Residuals (Actual - Predicted)", fontsize=14)

    # LEFT residuals
    if left_model and len(left_samples) > 0:
        for axis_idx, axis_name in enumerate(axis_names):
            ax = axes2[0, axis_idx]
            residuals = errors_mm[:, axis_idx]
            ax.scatter(u_values * 1000, residuals, c="blue", alpha=0.6, s=30)
            ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
            ax.axhline(
                y=np.mean(residuals),
                color="g",
                linestyle="-",
                linewidth=1,
                label=f"Mean: {np.mean(residuals):.2f}mm",
            )
            ax.fill_between(
                [u_values.min() * 1000, u_values.max() * 1000],
                np.mean(residuals) - np.std(residuals),
                np.mean(residuals) + np.std(residuals),
                alpha=0.2,
                color="green",
                label=f"±1σ: {np.std(residuals):.2f}mm",
            )
            ax.set_title(f"LEFT - {axis_name} Residuals")
            ax.set_xlabel("Gripper Opening (mm)")
            ax.set_ylabel("Residual (mm)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    # RIGHT residuals
    if right_model and len(right_samples) > 0:
        for axis_idx, axis_name in enumerate(axis_names):
            ax = axes2[1, axis_idx]
            residuals = errors_mm_r[:, axis_idx]
            ax.scatter(u_values_r * 1000, residuals, c="green", alpha=0.6, s=30)
            ax.axhline(y=0, color="r", linestyle="--", linewidth=1)
            ax.axhline(
                y=np.mean(residuals),
                color="orange",
                linestyle="-",
                linewidth=1,
                label=f"Mean: {np.mean(residuals):.2f}mm",
            )
            ax.fill_between(
                [u_values_r.min() * 1000, u_values_r.max() * 1000],
                np.mean(residuals) - np.std(residuals),
                np.mean(residuals) + np.std(residuals),
                alpha=0.2,
                color="orange",
                label=f"±1σ: {np.std(residuals):.2f}mm",
            )
            ax.set_title(f"RIGHT - {axis_name} Residuals")
            ax.set_xlabel("Gripper Opening (mm)")
            ax.set_ylabel("Residual (mm)")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_file2 = Path(__file__).parent / "calibration_residuals.png"
    plt.savefig(output_file2, dpi=150)
    logger.info(f"Saved residuals plot to {output_file2}")

    # ========================================================================
    # 3D visualization of sensor motion
    # ========================================================================
    fig3 = plt.figure(figsize=(12, 5))

    # Left sensor 3D trajectory
    ax3d_left = fig3.add_subplot(121, projection="3d")
    if left_model and len(left_samples) > 0:
        ax3d_left.scatter(
            actual_positions[:, 0] * 1000,
            actual_positions[:, 1] * 1000,
            actual_positions[:, 2] * 1000,
            c=u_values * 1000,
            cmap="viridis",
            s=30,
            label="Actual",
        )
        ax3d_left.plot(
            predicted_positions[:, 0] * 1000,
            predicted_positions[:, 1] * 1000,
            predicted_positions[:, 2] * 1000,
            "r-",
            linewidth=2,
            label="Predicted",
        )
        ax3d_left.set_xlabel("X (mm)")
        ax3d_left.set_ylabel("Y (mm)")
        ax3d_left.set_zlabel("Z (mm)")
        ax3d_left.set_title("LEFT Sensor 3D Trajectory")
        ax3d_left.legend()

    # Right sensor 3D trajectory
    ax3d_right = fig3.add_subplot(122, projection="3d")
    if right_model and len(right_samples) > 0:
        scatter = ax3d_right.scatter(
            actual_positions_r[:, 0] * 1000,
            actual_positions_r[:, 1] * 1000,
            actual_positions_r[:, 2] * 1000,
            c=u_values_r * 1000,
            cmap="viridis",
            s=30,
            label="Actual",
        )
        ax3d_right.plot(
            predicted_positions_r[:, 0] * 1000,
            predicted_positions_r[:, 1] * 1000,
            predicted_positions_r[:, 2] * 1000,
            "r-",
            linewidth=2,
            label="Predicted",
        )
        ax3d_right.set_xlabel("X (mm)")
        ax3d_right.set_ylabel("Y (mm)")
        ax3d_right.set_zlabel("Z (mm)")
        ax3d_right.set_title("RIGHT Sensor 3D Trajectory")
        ax3d_right.legend()
        fig3.colorbar(scatter, ax=ax3d_right, label="Gripper Opening (mm)", shrink=0.6)

    plt.tight_layout()

    output_file3 = Path(__file__).parent / "calibration_3d_trajectory.png"
    plt.savefig(output_file3, dpi=150)
    logger.info(f"Saved 3D trajectory plot to {output_file3}")

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info("\n" + "=" * 70)
    logger.info("VERIFICATION SUMMARY")
    logger.info("=" * 70)

    logger.info("\nCalibration Quality Assessment:")

    if left_model:
        avg_r2_left = np.mean(
            [left_model["models"][a]["r_squared"] for a in ["x", "y", "z"]]
        )
        x_r2 = left_model["models"]["x"]["r_squared"]
        logger.info("  LEFT sensor:")
        logger.info(
            f"    - X-axis R² = {x_r2:.4f} {'(GOOD - linear motion)' if x_r2 > 0.95 else '(CHECK - may have issues)'}"
        )
        logger.info(f"    - Average R² = {avg_r2_left:.4f}")
        logger.info(
            f"    - 3D error: mean={np.mean(total_error):.2f}mm, max={np.max(total_error):.2f}mm"
        )

    if right_model:
        avg_r2_right = np.mean(
            [right_model["models"][a]["r_squared"] for a in ["x", "y", "z"]]
        )
        x_r2_r = right_model["models"]["x"]["r_squared"]
        logger.info("  RIGHT sensor:")
        logger.info(
            f"    - X-axis R² = {x_r2_r:.4f} {'(GOOD - linear motion)' if x_r2_r > 0.95 else '(CHECK - may have issues)'}"
        )
        logger.info(f"    - Average R² = {avg_r2_right:.4f}")
        logger.info(
            f"    - 3D error: mean={np.mean(total_error_r):.2f}mm, max={np.max(total_error_r):.2f}mm"
        )

    logger.info("\nExpected behavior:")
    logger.info(
        "  - X-axis should have high R² (sensors move linearly in X with gripper)"
    )
    logger.info("  - Y/Z axes may have lower R² (should remain relatively constant)")
    logger.info("  - If Y/Z have high variation, check mask detection accuracy")

    # Show plots
    plt.show()

    logger.info("\nVerification complete!")


if __name__ == "__main__":
    import fire

    fire.Fire(verify_calibration)
