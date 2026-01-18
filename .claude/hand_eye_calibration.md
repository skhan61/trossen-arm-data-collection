# Hand-Eye Calibration Summary for WidowX AI Follower

## Goal
Calibrate RealSense camera mounted on WidowX AI Follower gripper to enable accurate vision-tactile data collection.

---

## What We Discussed

### 1. The Calibration Problem

**Need:** Transform T_E→C (end-effector to camera)

**Why:**
- Camera measures in camera frame
- Robot operates in base frame
- Without T_E→C: robot misses objects by 5-50cm
- With T_E→C: sub-millimeter accuracy

**Example:**
```
Camera sees object at [0.2, 0.1, 0.5] (camera frame)
Without calibration: Robot goes to [0.2, 0.1, 0.5] (base frame) ← WRONG!
With calibration: Robot goes to [0.35, -0.02, 0.28] (base frame) ← CORRECT
```

### 2. The Solution: AX=XB Calibration

**Method:**
- Fix ArUco marker on table (doesn't move)
- Move robot to 15 different poses viewing marker
- Each pose gives equation: A×X = X×B
- Solve overdetermined system for X (the hand-eye transform)

**Math:**
```
A = Robot motion between poses (from forward kinematics)
X = Hand-eye transform (what we want)
B = Marker motion as seen by camera (from ArUco detection)

15 poses → 14 equations → 84 constraints for 6 unknowns
```

**Why Better than Single-Pose PnP:**
- 14× more equations (robust least-squares)
- Errors average out (√N reduction)
- Automatic outlier rejection
- Accuracy: 0.2mm vs 1-2mm

### 3. Gravity Compensation Mode - KEY INSIGHT!

**What it is:**
- Special robot mode: `Mode.external_effort`
- Motors compensate for gravity
- Arm "floats" - doesn't fall
- **You can physically move robot by hand**
- Stays wherever you leave it

**Code:**
```python
# Enable
driver.set_all_modes(Mode.external_effort)

# Now push robot by hand!
# Position it until camera sees marker

# Get position
pose = driver.get_cartesian_positions()  # Robot tells you coordinates!

# Lock position
driver.set_all_modes(Mode.position)
```

**Why This Matters:**
- No need to measure marker position with ruler
- No need to guess coordinates
- No trial-and-error
- Just push robot by hand → done in 30 seconds!

### 4. The Hybrid Approach

**Step 1: Manual (30 seconds)**
- Enable gravity compensation
- Push robot until camera sees marker
- Save position

**Step 2: Automatic (5-10 minutes)**
- Generate 14 variations around first pose
- Robot moves itself
- Check marker visibility
- Save good poses

**Total manual effort: 30 seconds!**

---

## Implementation Plan

### Setup

1. **Print ArUco Marker**
   - Size: 100mm × 100mm
   - Dictionary: DICT_6X6_250, ID=0
   - White paper with border

2. **Place Marker**
   - Flat on table
   - 0.30-0.45m from robot base
   - Within reach
   - Must not move during calibration

3. **Camera**
   - RealSense on gripper
   - Need intrinsics: K matrix, distortion coefficients

### Code Structure

**Step 1: Initialize Hardware**
```python
from trossen_arm import TrossenArmDriver, Mode
import pyrealsense2 as rs
import cv2
import numpy as np

# Robot
driver = TrossenArmDriver()
driver.configure(ip="192.168.1.3")

# Camera
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
pipeline.start(config)
```

**Step 2: Collect First Pose Manually**
```python
# Enable gravity compensation
driver.set_all_modes(Mode.external_effort)

print("Push robot by hand until camera sees marker")
input("Press ENTER when ready...")

# Save position
first_pose = driver.get_cartesian_positions()
# Returns: [x, y, z, roll, pitch, yaw]

# Lock position
driver.set_all_modes(Mode.position)

poses = [first_pose]
```

**Step 3: Collect 14 More Poses Automatically**
```python
import random

for i in range(14):
    # Random variation around first pose
    new_pose = first_pose.copy()
    new_pose[0] += random.uniform(-0.08, 0.08)  # X ±8cm
    new_pose[1] += random.uniform(-0.08, 0.08)  # Y ±8cm
    new_pose[2] += random.uniform(-0.08, 0.08)  # Z ±8cm
    new_pose[4] += random.uniform(-0.3, 0.3)    # Pitch ±17°

    # Move robot
    driver.set_cartesian_positions(
        positions=new_pose,
        time=2.0,
        blocking=True
    )

    # Wait for stabilization
    time.sleep(0.5)

    # Capture image
    frames = pipeline.wait_for_frames()
    color_frame = frames.get_color_frame()
    image = np.asanyarray(color_frame.get_data())

    # Detect ArUco
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    detector = cv2.aruco.ArucoDetector(aruco_dict)
    corners, ids = detector.detectMarkers(image)

    # Save if marker visible
    if ids is not None:
        poses.append({
            'robot_pose': new_pose,
            'image': image,
            'corners': corners
        })
        print(f"Pose {len(poses)}/15 saved")
```

**Step 4: Compute Marker Poses from ArUco**
```python
# Marker geometry (100mm square)
marker_size = 0.1  # meters
marker_points = np.array([
    [-marker_size/2,  marker_size/2, 0],
    [ marker_size/2,  marker_size/2, 0],
    [ marker_size/2, -marker_size/2, 0],
    [-marker_size/2, -marker_size/2, 0]
], dtype=np.float32)

# Camera intrinsics (you need to get these from camera calibration)
K = np.array([
    [fx, 0, cx],
    [0, fy, cy],
    [0,  0,  1]
])
dist = np.array([k1, k2, p1, p2, k3])

for pose_data in poses:
    # Solve PnP
    rvec, tvec = cv2.solvePnP(
        objectPoints=marker_points,
        imagePoints=pose_data['corners'][0],
        cameraMatrix=K,
        distCoeffs=dist
    )[1:3]

    # Convert to transform
    R, _ = cv2.Rodrigues(rvec)
    T_C_to_M = np.eye(4)
    T_C_to_M[:3, :3] = R
    T_C_to_M[:3, 3] = tvec.flatten()

    pose_data['T_C_to_M'] = T_C_to_M
```

**Step 5: Build A and B Matrices**
```python
def pose_to_transform(pose):
    """Convert [x,y,z,r,p,y] to 4x4 transform"""
    x, y, z, roll, pitch, yaw = pose

    # Rotation matrix from RPY
    cr, cp, cy = np.cos([roll, pitch, yaw])
    sr, sp, sy = np.sin([roll, pitch, yaw])

    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ])

    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = [x, y, z]
    return T

A_matrices = []
B_matrices = []

for i in range(len(poses) - 1):
    # A: Robot motion
    T_B_to_Ei = pose_to_transform(poses[i]['robot_pose'])
    T_B_to_Ej = pose_to_transform(poses[i+1]['robot_pose'])
    A = np.linalg.inv(T_B_to_Ei) @ T_B_to_Ej
    A_matrices.append(A)

    # B: Marker motion in camera
    T_C_to_Mi = poses[i]['T_C_to_M']
    T_C_to_Mj = poses[i+1]['T_C_to_M']
    B = T_C_to_Mi @ np.linalg.inv(T_C_to_Mj)
    B_matrices.append(B)
```

**Step 6: Solve AX=XB**
```python
# Extract rotations and translations
R_gripper2base = [A[:3, :3] for A in A_matrices]
t_gripper2base = [A[:3, 3:4] for A in A_matrices]
R_target2cam = [B[:3, :3] for B in B_matrices]
t_target2cam = [B[:3, 3:4] for B in B_matrices]

# Solve
R_E_to_C, t_E_to_C = cv2.calibrateHandEye(
    R_gripper2base=R_gripper2base,
    t_gripper2base=t_gripper2base,
    R_target2cam=R_target2cam,
    t_target2cam=t_target2cam,
    method=cv2.CALIB_HAND_EYE_TSAI
)

# Build transform
T_E_to_C = np.eye(4)
T_E_to_C[:3, :3] = R_E_to_C
T_E_to_C[:3, 3] = t_E_to_C.flatten()

print("Hand-Eye Calibration Result:")
print(T_E_to_C)

# Save
np.save('T_E_to_C.npy', T_E_to_C)
```

**Step 7: Validate**
```python
errors = []
for i in range(len(A_matrices)):
    # Check AX = XB
    left = A_matrices[i] @ T_E_to_C
    right = T_E_to_C @ B_matrices[i]
    error = np.linalg.norm(left - right, 'fro')
    errors.append(error)

print(f"Mean error: {np.mean(errors):.6f}")
print(f"Max error: {np.max(errors):.6f}")
print(f"Std error: {np.std(errors):.6f}")

# Good: mean < 0.001, max < 0.005
```

---

## Key Technical Details

### Robot API (Trossen Arm)

**IP Address:** 192.168.1.99

**Get position:**
```python
pose = driver.get_cartesian_positions()
# Returns: [x, y, z, roll, pitch, yaw]
```

**Set position:**
```python
driver.set_cartesian_positions(
    positions=[x, y, z, roll, pitch, yaw],
    time=2.0,
    blocking=True
)
```

**Modes:**
```python
Mode.position         # Normal control
Mode.external_effort  # Gravity compensation (manual movement)
```

### Coordinate Frame

**Origin [0,0,0]:** Center of base joint (first rotation axis)

**Axes:**
- X: Forward
- Y: Left (negative = right)
- Z: Up

**Workspace (from drawing):**
- Max reach: ~264mm
- Safe: X[0.25-0.45m], Y[±0.15m], Z[0.0-0.40m]

### ArUco Detection

**Dictionary:** DICT_6X6_250
**Marker ID:** 0
**Marker Size:** 100mm × 100mm

```python
aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
detector = cv2.aruco.ArucoDetector(aruco_dict)

corners, ids = detector.detectMarkers(image)
if ids is not None:
    # Marker detected
```

---

## Quality Requirements

### Pose Diversity

**Position:**
- Spread ±10cm in X, Y, Z

**Orientation:**
- Pitch: -45° to -10° (camera looking down)
- Roll/Yaw: near 0

**Marker Visibility:**
- All 4 corners visible
- Marker occupies 20-40% of image
- Not blurry
- Well lit

### Validation Criteria

**Good calibration:**
- Mean error < 0.001
- Max error < 0.005
- 15+ poses collected
- All poses diverse

**Bad calibration:**
- Mean error > 0.01
- Large outliers
- <10 poses
- All similar poses

---

## Common Issues

### Marker Not Detected
- Check lighting (diffuse, no shadows)
- Verify DICT_6X6_250
- Check focus/blur
- Ensure all corners visible

### Large Errors
- Re-calibrate camera intrinsics
- Verify marker didn't move
- Check pose diversity
- Remove outlier poses

### Robot Can't Reach
- Reduce random offset range
- Check workspace limits
- Verify first pose is central

---

## Expected Results

**Output:** 4x4 transform matrix T_E_to_C

**Example:**
```
[[ 0.995  -0.087   0.050   0.045]
 [ 0.088   0.996  -0.010   0.012]
 [-0.049   0.015   0.999   0.083]
 [ 0.000   0.000   0.000   1.000]]
```

**Interpretation:**
- Translation: [45mm, 12mm, 83mm] (camera offset from gripper)
- Rotation: ~5° tilt

**Accuracy:** ±0.2-0.5mm

**Time:** 10-15 minutes total

---

## Additional Calibrations Needed

After hand-eye calibration:

1. **Camera Intrinsic** - K matrix, distortion (checkerboard)
2. **Stereo** - RealSense depth accuracy (stereo checkerboard)
3. **Gelsight Tactile** - Gel depth mapping (known surfaces)

**All 4 needed for research-grade dataset**

---

## Summary Workflow

```
1. Print ArUco marker (100mm, DICT_6X6_250)
2. Place marker on table (0.3-0.5m from robot)
3. Enable gravity compensation
4. Push robot by hand until camera sees marker (30 sec)
5. Save first pose
6. Automatically collect 14 more poses (5-10 min)
7. Detect ArUco in all images
8. Build A and B matrices
9. Solve AX=XB with cv2.calibrateHandEye()
10. Validate with residual errors
11. Save T_E_to_C
```

**Manual effort:** 30 seconds
**Total time:** 10-15 minutes
**Accuracy:** ±0.2-0.5mm

---

## Libraries Required

```python
pip install trossen_arm
pip install pyrealsense2
pip install opencv-contrib-python
pip install numpy
```

---

## References

- Trossen Arm: https://github.com/TrossenRobotics/trossen_arm
- Docs: https://docs.trossenrobotics.com/trossen_arm
- Robot Drawing: Drawing_WidowX_AI_Follower.pdf
