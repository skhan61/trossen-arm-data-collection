#!/usr/bin/env python3
"""
Analyze hand-eye calibration quality.
"""

import numpy as np
import json
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data" / "hand_eye_calibration_data"

print("=" * 80)
print("HAND-EYE CALIBRATION ANALYSIS")
print("=" * 80)

# Load calibration
calib_file = DATA_DIR / "hand_eye_calibration.json"
with open(calib_file) as f:
    calib = json.load(f)

R_cam2gripper = np.array(calib["R_cam2gripper"])
t_cam2gripper = np.array(calib["t_cam2gripper"]).flatten()

print("\n1. ROTATION MATRIX ANALYSIS")
print("-" * 80)
print("Rotation matrix (camera to gripper):")
print(R_cam2gripper)
print()

# Check if rotation matrix is valid (orthogonal)
RTR = R_cam2gripper.T @ R_cam2gripper
det = np.linalg.det(R_cam2gripper)
print(f"Determinant: {det:.6f} (should be close to 1.0)")
print(f"Is proper rotation: {'YES' if abs(det - 1.0) < 0.01 else 'NO - WARNING!'}")
print()

print("R^T * R (should be close to identity matrix):")
print(RTR)
print()

identity_error = np.linalg.norm(RTR - np.eye(3))
print(f"Orthogonality error: {identity_error:.6f} (should be < 0.01)")
if identity_error < 0.001:
    print("✓ EXCELLENT - Matrix is very orthogonal")
elif identity_error < 0.01:
    print("✓ GOOD - Matrix is orthogonal")
else:
    print("✗ WARNING - Matrix may not be properly orthogonal")
print()

# Convert to axis-angle
import cv2
rvec, _ = cv2.Rodrigues(R_cam2gripper)
angle = np.linalg.norm(rvec)
axis = rvec / angle if angle > 0 else np.array([0, 0, 1])

print(f"Rotation angle: {np.degrees(angle):.2f}°")
print(f"Rotation axis: [{axis.flatten()[0]:.3f}, {axis.flatten()[1]:.3f}, {axis.flatten()[2]:.3f}]")
print()

# Convert to Euler angles (XYZ convention)
def rotation_matrix_to_euler_angles(R):
    """Convert rotation matrix to Euler angles (XYZ intrinsic)."""
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)

    if sy > 1e-6:
        x = np.arctan2(R[2, 1], R[2, 2])
        y = np.arctan2(-R[2, 0], sy)
        z = np.arctan2(R[1, 0], R[0, 0])
    else:
        x = np.arctan2(-R[1, 2], R[1, 1])
        y = np.arctan2(-R[2, 0], sy)
        z = 0

    return np.degrees([x, y, z])

euler = rotation_matrix_to_euler_angles(R_cam2gripper)
print(f"Euler angles (XYZ intrinsic):")
print(f"  Roll  (X): {euler[0]:7.2f}°")
print(f"  Pitch (Y): {euler[1]:7.2f}°")
print(f"  Yaw   (Z): {euler[2]:7.2f}°")
print()

print("\n2. TRANSLATION VECTOR ANALYSIS")
print("-" * 80)
print(f"Translation (camera position relative to gripper):")
print(f"  X: {t_cam2gripper[0]:8.4f} m ({t_cam2gripper[0]*1000:7.2f} mm)")
print(f"  Y: {t_cam2gripper[1]:8.4f} m ({t_cam2gripper[1]*1000:7.2f} mm)")
print(f"  Z: {t_cam2gripper[2]:8.4f} m ({t_cam2gripper[2]*1000:7.2f} mm)")
print()

distance = np.linalg.norm(t_cam2gripper)
print(f"Total distance from gripper: {distance:.4f} m ({distance*1000:.2f} mm)")
print()

print("\n3. INTERPRETATION")
print("-" * 80)
print("Camera mounting position relative to gripper frame:")
print()

# Describe the position
if abs(t_cam2gripper[0]) > 0.05:
    side = "LEFT" if t_cam2gripper[0] < 0 else "RIGHT"
    print(f"  • Camera is {abs(t_cam2gripper[0]*1000):.1f} mm to the {side}")

if abs(t_cam2gripper[1]) > 0.05:
    vert = "BELOW" if t_cam2gripper[1] < 0 else "ABOVE"
    print(f"  • Camera is {abs(t_cam2gripper[1]*1000):.1f} mm {vert}")

if abs(t_cam2gripper[2]) > 0.05:
    depth = "BEHIND" if t_cam2gripper[2] < 0 else "IN FRONT OF"
    print(f"  • Camera is {abs(t_cam2gripper[2]*1000):.1f} mm {depth}")
print()

print("Camera orientation:")
# Analyze main pointing direction
z_cam_in_gripper = R_cam2gripper[:, 2]  # Camera Z-axis (optical axis) in gripper frame
print(f"  • Camera optical axis points toward: [{z_cam_in_gripper[0]:.2f}, {z_cam_in_gripper[1]:.2f}, {z_cam_in_gripper[2]:.2f}] in gripper frame")

# Find dominant direction
abs_z = np.abs(z_cam_in_gripper)
dominant_axis = np.argmax(abs_z)
axes_names = ['X (forward)', 'Y (left)', 'Z (up)']
sign = "+" if z_cam_in_gripper[dominant_axis] > 0 else "-"
print(f"  • Mainly pointing in {sign}{axes_names[dominant_axis]} direction of gripper")
print()

print("\n4. QUALITY ASSESSMENT")
print("-" * 80)

# Load raw data to assess quality
data_file = DATA_DIR / "hand_eye_data.json"
with open(data_file) as f:
    data = json.load(f)

num_poses = data["num_poses"]
print(f"Number of calibration poses: {num_poses}")

# Check rotation matrix validity
quality_score = 0
issues = []

if abs(det - 1.0) < 0.001:
    quality_score += 25
    print("✓ Rotation matrix determinant: EXCELLENT")
elif abs(det - 1.0) < 0.01:
    quality_score += 20
    print("✓ Rotation matrix determinant: GOOD")
else:
    issues.append("Determinant not close to 1.0")
    print("✗ Rotation matrix determinant: POOR")

if identity_error < 0.001:
    quality_score += 25
    print("✓ Rotation matrix orthogonality: EXCELLENT")
elif identity_error < 0.01:
    quality_score += 20
    print("✓ Rotation matrix orthogonality: GOOD")
else:
    issues.append("Matrix not orthogonal")
    print("✗ Rotation matrix orthogonality: POOR")

if num_poses >= 15:
    quality_score += 25
    print(f"✓ Number of poses: GOOD ({num_poses} poses)")
elif num_poses >= 10:
    quality_score += 15
    print(f"⚠ Number of poses: ACCEPTABLE ({num_poses} poses, more is better)")
else:
    issues.append("Too few calibration poses")
    print(f"✗ Number of poses: POOR ({num_poses} poses)")

# Check if translation is reasonable (should be within robot workspace)
if distance < 0.5:  # Within 50cm
    quality_score += 25
    print(f"✓ Translation magnitude: REASONABLE ({distance*1000:.1f} mm)")
else:
    issues.append("Translation seems too large")
    print(f"⚠ Translation magnitude: LARGE ({distance*1000:.1f} mm)")

print()
print(f"Overall Quality Score: {quality_score}/100")
if quality_score >= 90:
    print("🟢 EXCELLENT calibration quality")
elif quality_score >= 70:
    print("🟡 GOOD calibration quality")
elif quality_score >= 50:
    print("🟠 ACCEPTABLE calibration quality - consider recalibrating")
else:
    print("🔴 POOR calibration quality - recalibration recommended")

if issues:
    print("\nIssues found:")
    for issue in issues:
        print(f"  • {issue}")

print()
print("=" * 80)
print("CALIBRATION FILE: " + str(calib_file))
print("=" * 80)
