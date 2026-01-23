#!/usr/bin/env python3
"""
Debug script to test different transformation conventions for X verification.

Tests all possible combinations to find which convention gives consistent results.
"""

import json
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data" / "Xverification"


def load_data():
    """Load verification data."""
    json_path = DATA_DIR / "verification_data.json"
    with open(json_path, "r") as f:
        return json.load(f)


def invert_transform(T):
    """Invert a 4x4 transformation matrix."""
    R = T[:3, :3]
    t = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ t
    return T_inv


def test_convention(name, T_gripper_in_base_list, X, T_target_in_cam_list, formula_func):
    """Test a specific convention and report error."""
    results = []
    for i, (T_g, T_t) in enumerate(zip(T_gripper_in_base_list, T_target_in_cam_list)):
        T_result = formula_func(T_g, X, T_t)
        results.append(T_result[:3, 3])  # Just translation

    results = np.array(results)
    if len(results) >= 2:
        # Compute spread
        mean_pos = np.mean(results, axis=0)
        errors = np.linalg.norm(results - mean_pos, axis=1)
        max_error = np.max(errors) * 1000  # mm

        print(f"\n{name}:")
        print(f"  Pose 1 position: [{results[0,0]:.4f}, {results[0,1]:.4f}, {results[0,2]:.4f}] m")
        print(f"  Pose 2 position: [{results[1,0]:.4f}, {results[1,1]:.4f}, {results[1,2]:.4f}] m")
        print(f"  Difference: {np.linalg.norm(results[0] - results[1])*1000:.2f} mm")
        print(f"  Max error from mean: {max_error:.2f} mm")
        return max_error
    return float('inf')


def main():
    print("=" * 70)
    print("DEBUG: Testing All Transformation Conventions")
    print("=" * 70)

    data = load_data()
    X = np.array(data["X_cam2gripper"])
    X_inv = invert_transform(X)

    print(f"\nX (cam2gripper) translation: [{X[0,3]:.4f}, {X[1,3]:.4f}, {X[2,3]:.4f}] m")
    print(f"X_inv (gripper2cam) translation: [{X_inv[0,3]:.4f}, {X_inv[1,3]:.4f}, {X_inv[2,3]:.4f}] m")

    # Extract transforms from poses
    T_gripper_in_base_list = []
    T_target_in_cam_list = []

    for pose in data["poses"]:
        T_g = np.array(pose["robot"]["T_gripper2base"])
        T_t = np.array(pose["checkerboard"]["T_target2cam"])
        T_gripper_in_base_list.append(T_g)
        T_target_in_cam_list.append(T_t)

    print(f"\nLoaded {len(T_gripper_in_base_list)} poses")

    # Also compute inverses
    T_base_in_gripper_list = [invert_transform(T) for T in T_gripper_in_base_list]
    T_cam_in_target_list = [invert_transform(T) for T in T_target_in_cam_list]

    print("\n" + "=" * 70)
    print("Testing different formulas for T_target_in_base:")
    print("=" * 70)

    best_name = None
    best_error = float('inf')

    # Convention 1: T_base @ X @ T_cam (what I used)
    err = test_convention(
        "1. T_gripper_in_base @ X_cam2gripper @ T_target_in_cam",
        T_gripper_in_base_list, X, T_target_in_cam_list,
        lambda Tg, X, Tt: Tg @ X @ Tt
    )
    if err < best_error:
        best_error = err
        best_name = "1"

    # Convention 2: Use X_inv instead
    err = test_convention(
        "2. T_gripper_in_base @ X_inv (gripper2cam) @ T_target_in_cam",
        T_gripper_in_base_list, X_inv, T_target_in_cam_list,
        lambda Tg, X, Tt: Tg @ X @ Tt
    )
    if err < best_error:
        best_error = err
        best_name = "2"

    # Convention 3: Different order
    err = test_convention(
        "3. T_target_in_cam @ X_cam2gripper @ T_gripper_in_base",
        T_gripper_in_base_list, X, T_target_in_cam_list,
        lambda Tg, X, Tt: Tt @ X @ Tg
    )
    if err < best_error:
        best_error = err
        best_name = "3"

    # Convention 4: With inverses
    err = test_convention(
        "4. T_gripper_in_base @ X_cam2gripper @ T_cam_in_target (inv)",
        T_gripper_in_base_list, X, T_cam_in_target_list,
        lambda Tg, X, Tt: Tg @ X @ Tt
    )
    if err < best_error:
        best_error = err
        best_name = "4"

    # Convention 5: X on right side
    err = test_convention(
        "5. T_gripper_in_base @ T_target_in_cam @ X_cam2gripper",
        T_gripper_in_base_list, X, T_target_in_cam_list,
        lambda Tg, X, Tt: Tg @ Tt @ X
    )
    if err < best_error:
        best_error = err
        best_name = "5"

    # Convention 6: X_inv on right
    err = test_convention(
        "6. T_gripper_in_base @ T_target_in_cam @ X_inv",
        T_gripper_in_base_list, X_inv, T_target_in_cam_list,
        lambda Tg, X, Tt: Tg @ Tt @ X
    )
    if err < best_error:
        best_error = err
        best_name = "6"

    # Convention 7: inv(X) @ inv(T_gripper)
    err = test_convention(
        "7. X_inv @ T_base_in_gripper @ T_target_in_cam",
        T_base_in_gripper_list, X_inv, T_target_in_cam_list,
        lambda Tg, X, Tt: X @ Tg @ Tt
    )
    if err < best_error:
        best_error = err
        best_name = "7"

    # Convention 8: Standard eye-in-hand
    err = test_convention(
        "8. T_gripper_in_base @ X @ T_target_in_cam (same as 1)",
        T_gripper_in_base_list, X, T_target_in_cam_list,
        lambda Tg, X, Tt: Tg @ X @ Tt
    )

    # Convention 9: Try base_in_gripper
    err = test_convention(
        "9. inv(T_gripper_in_base) @ X @ T_target_in_cam",
        T_base_in_gripper_list, X, T_target_in_cam_list,
        lambda Tg, X, Tt: Tg @ X @ Tt
    )
    if err < best_error:
        best_error = err
        best_name = "9"

    # Convention 10
    err = test_convention(
        "10. T_target_in_cam @ X_inv @ T_base_in_gripper",
        T_base_in_gripper_list, X_inv, T_target_in_cam_list,
        lambda Tg, X, Tt: Tt @ X @ Tg
    )
    if err < best_error:
        best_error = err
        best_name = "10"

    print("\n" + "=" * 70)
    print(f"BEST CONVENTION: {best_name} with error {best_error:.2f} mm")
    print("=" * 70)

    if best_error > 50:
        print("\nAll conventions have large errors (>50mm).")
        print("This suggests the X calibration itself is incorrect,")
        print("not just a convention mismatch.")


if __name__ == "__main__":
    main()
