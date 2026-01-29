#!/usr/bin/env python3
"""
Analyze the relationship between gripper_opening (motor position) and
actual sensor distance using T(u) calibration.

This helps understand:
- grip=12mm (motor) → actual sensor gap = ??? mm
- Why a 36mm object shows grip=12mm at contact
"""

from pathlib import Path

import numpy as np

from src.utils.transforms import compute_T_u, load_calibration


def main():
    # Load calibration
    calibration_dir = Path("dataset/calibration")
    if not calibration_dir.exists():
        print(f"ERROR: Calibration not found at {calibration_dir}")
        return

    X, T_u_left, T_u_right = load_calibration(calibration_dir)

    print("=" * 60)
    print("T(u) Calibration Parameters")
    print("=" * 60)
    print(f"\nT_u_left:  {T_u_left}")
    print(f"T_u_right: {T_u_right}")

    print("\n" + "=" * 60)
    print("T(u) Model: T_cam_to_gelsight = t0 + k * u")
    print("  where u = gripper_opening (meters)")
    print("=" * 60)

    print("\nLeft sensor:")
    print(f"  t0 = [{T_u_left[0]:.6f}, {T_u_left[1]:.6f}, {T_u_left[2]:.6f}] m")
    print(f"  k  = [{T_u_left[3]:.6f}, {T_u_left[4]:.6f}, {T_u_left[5]:.6f}] m/m")

    print("\nRight sensor:")
    print(f"  t0 = [{T_u_right[0]:.6f}, {T_u_right[1]:.6f}, {T_u_right[2]:.6f}] m")
    print(f"  k  = [{T_u_right[3]:.6f}, {T_u_right[4]:.6f}, {T_u_right[5]:.6f}] m/m")

    print("\n" + "=" * 60)
    print("Gripper Opening → Sensor Gap Mapping")
    print("=" * 60)
    print(f"{'Gripper (mm)':>12} | {'Left Y (mm)':>12} | {'Right Y (mm)':>12} | {'Sensor Gap (mm)':>15}")
    print("-" * 60)

    # Test range of gripper openings
    gripper_openings_mm = [0, 5, 10, 12, 15, 20, 25, 30, 35, 40]

    for grip_mm in gripper_openings_mm:
        grip_m = grip_mm / 1000.0

        # Compute T_cam_to_gelsight for each sensor
        T_left = compute_T_u(T_u_left, grip_m)
        T_right = compute_T_u(T_u_right, grip_m)

        # Get Y positions (perpendicular to gripper direction)
        y_left = T_left[1, 3] * 1000  # Convert to mm
        y_right = T_right[1, 3] * 1000

        # Sensor gap = distance between sensor Y positions
        # (assuming sensors face each other in Y direction)
        sensor_gap_mm = abs(y_left - y_right)

        print(f"{grip_mm:>12.1f} | {y_left:>12.3f} | {y_right:>12.3f} | {sensor_gap_mm:>15.3f}")

    print("\n" + "=" * 60)
    print("Interpretation")
    print("=" * 60)
    print("""
The T(u) model translates gripper motor position to GelSight sensor position.

If your object is 36mm and contact occurs at grip=12mm:
- The sensor gap at grip=12mm should be approximately 36mm
- The k (slope) parameters control how sensor position changes with gripper

Key insight:
- gripper_opening (from robot API) is the motor encoder value
- Actual sensor gap depends on the T(u) calibration model
- Object deformation = object_width - sensor_gap_at_max_squeeze
""")

    # Specific analysis for the user's case
    print("\n" + "=" * 60)
    print("Your Case: 36mm object, contact at grip=12mm")
    print("=" * 60)

    grip_at_contact_mm = 12.3  # From log
    grip_m = grip_at_contact_mm / 1000.0

    T_left = compute_T_u(T_u_left, grip_m)
    T_right = compute_T_u(T_u_right, grip_m)

    # Compute full 3D distance between sensors
    pos_left = T_left[:3, 3]
    pos_right = T_right[:3, 3]
    sensor_distance_3d = np.linalg.norm(pos_left - pos_right) * 1000

    print(f"\nAt grip={grip_at_contact_mm}mm:")
    print(f"  Left sensor position:  {pos_left * 1000} mm")
    print(f"  Right sensor position: {pos_right * 1000} mm")
    print(f"  3D distance between sensors: {sensor_distance_3d:.2f} mm")

    object_width_mm = 36.0
    print(f"\nObject width: {object_width_mm} mm")
    print(f"Sensor CENTER gap at contact: {sensor_distance_3d:.2f} mm")

    # The T(u) model gives sensor CENTER position, not SURFACE position
    # If object is 36mm and sensor centers are 17mm apart, there's an offset
    surface_offset_per_sensor = (object_width_mm - sensor_distance_3d) / 2
    print(f"\nINFERRED: Elastomer surface offset from center: {surface_offset_per_sensor:.2f} mm per sensor")
    print(f"  This means the actual SURFACE gap = sensor_center_gap + 2 * {surface_offset_per_sensor:.2f}")

    # Now let's compute what we SHOULD use for deformation
    print("\n" + "=" * 60)
    print("Computing Object Deformation")
    print("=" * 60)

    # At max squeeze (grip stalls), what is the sensor distance?
    grip_at_max_mm = 12.3  # Same as contact since stall happens quickly
    grip_m = grip_at_max_mm / 1000.0

    T_left_max = compute_T_u(T_u_left, grip_m)
    T_right_max = compute_T_u(T_u_right, grip_m)

    pos_left_max = T_left_max[:3, 3]
    pos_right_max = T_right_max[:3, 3]
    sensor_gap_max = np.linalg.norm(pos_left_max - pos_right_max) * 1000

    # Add surface offset to get actual surface gap
    surface_gap_max = sensor_gap_max + 2 * surface_offset_per_sensor

    print(f"\nAt max squeeze (grip={grip_at_max_mm}mm):")
    print(f"  Sensor CENTER gap: {sensor_gap_max:.2f} mm")
    print(f"  Surface gap (with {surface_offset_per_sensor:.2f}mm offset): {surface_gap_max:.2f} mm")

    deformation = object_width_mm - surface_gap_max
    print(f"\nObject deformation = {object_width_mm} - {surface_gap_max:.2f} = {deformation:.2f} mm")

    print("\n" + "=" * 60)
    print("CONCLUSION")
    print("=" * 60)
    print(f"""
The T(u) calibration measures sensor CENTER position, not SURFACE.

For accurate deformation measurement, you need:
1. Know object_width (e.g., 36mm)
2. Compute sensor_center_gap from poses
3. Add elastomer_surface_offset (inferred as {surface_offset_per_sensor:.2f}mm per side)
4. Deformation = object_width - (sensor_center_gap + 2*surface_offset)

Formula:
  surface_gap = compute_sensor_distance(pose_left, pose_right) + 2 * SURFACE_OFFSET_MM
  deformation = object_width - surface_gap
""")


if __name__ == "__main__":
    main()
