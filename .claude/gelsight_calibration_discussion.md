# Gelsight Calibration Discussion Notes

## The Fundamental Problem

We want to know the position of the gelsight sensor center in the robot base frame. But there is a critical issue: when the gelsight is touching an object (the moment we actually need to know its position), the camera cannot see the gelsight because it is occluded by the object being touched.

---

## The Core Insight

We cannot directly observe the gelsight position at the moment of contact. Therefore, we must:

1. Pre-calibrate the relationship between camera and gelsight when we CAN see it
2. Use that calibration at runtime when we CANNOT see it

---

## The Kinematic Chain

The full transformation from base to gelsight center:

```
T_base_to_gelsight = T_base_to_ee · X · T_camera_to_gelsight(u)
```

Each term:

- **T_base_to_ee**: From robot API. Changes with robot pose.
- **X (T_ee_to_camera)**: From AX=XB hand-eye calibration. Constant because camera is rigidly bolted to end-effector.
- **T_camera_to_gelsight(u)**: From CV calibration. NOT constant — changes with gripper opening u.

---

## Why T_camera_to_gelsight is NOT Constant

This was a key misunderstanding we clarified.

The gelsight sensor is mounted on the gripper finger, not directly on the end-effector. When the gripper opens or closes, the finger moves, and thus the gelsight moves relative to the camera.

```
      Camera (fixed to EE)
         |
         v
    [Gripper]
    /       \
Gelsight_L   Gelsight_R
(moves)      (moves)
```

This is fundamentally different from X, which is constant because the camera does not move relative to the end-effector.

T_camera_to_gelsight is a function of gripper opening:

```
T_camera_to_gelsight = f(u)
```

---

## The Two-Phase Approach

### Phase 1: Calibration (Offline, Before Task)

Gripper is open. Camera CAN see gelsight.

**First:** Calibrate X using AX=XB method with fixed calibration target.

**Second:** Calibrate T_camera_to_gelsight(u) for many gripper openings:
- Open gripper to u = 26mm → observe gelsight corners → compute T_camera_to_gelsight(26)
- Open gripper to u = 28mm → observe gelsight corners → compute T_camera_to_gelsight(28)
- Continue for multiple values...
- Build a model that maps gripper opening to transform

### Phase 2: Runtime (During Task)

Gripper is closed, touching object. Camera CANNOT see gelsight.

1. Read T_base_to_ee from robot API
2. Read gripper opening u from robot API
3. Look up X from calibration (constant)
4. Look up T_camera_to_gelsight(u) from pre-calibrated model
5. Compute: T_base_to_gelsight = T_base_to_ee · X · T_camera_to_gelsight(u)

No need to see gelsight at contact time.

---

## Computing T_camera_to_gelsight(u) from Vision

For a given gripper opening u:

1. Detect 4 gelsight corner pixels in RealSense RGB image
2. Get depth for each corner from RealSense depth image
3. Backproject each corner to 3D using camera intrinsics:
   - X = depth × (pixel_u - cx) / fx
   - Y = depth × (pixel_v - cy) / fy
   - Z = depth
4. Compute center as centroid of 4 corners
5. Compute orientation from corner geometry
6. Form the 4×4 transform matrix

---

## Building the Linear Model for Translation

Since gripper moves linearly, translation varies linearly with u:

```
t(u) = t₀ + k · u
```

Where:
- t₀ = translation when u = 0 (gripper fully closed)
- k = translation direction per unit gripper opening

**The problem:** When gripper is fully closed (u = 0), we may not be able to see the gelsight corners.

**The solution:** Extrapolate t₀ from measurements at larger u values where gelsight IS visible.

Collect data at u = 26, 28, 30, ... 42mm, then fit a line. The y-intercept gives t₀.

---

## Two Sensors Require Two Models

The gripper has two fingers, each with a gelsight sensor.

When gripper opens:
- Left gelsight moves in one direction
- Right gelsight moves in opposite direction

Therefore we need two separate models:

```
t_L(u) = t₀_L + k_L · u    (left sensor)
t_R(u) = t₀_R + k_R · u    (right sensor)
```

The directions k_L and k_R are opposite. The calibration procedure must be done separately for each sensor.

---

## What Each Piece Contributes

| Component | What it provides | How obtained |
|-----------|------------------|--------------|
| Robot API | T_base_to_ee at runtime | Direct readout |
| Robot API | Gripper opening u at runtime | Direct readout |
| AX=XB calibration | X (constant) | Solve from multiple poses with fixed target |
| CV calibration | T_camera_to_gelsight(u) model | Observe gelsight at multiple gripper openings |
| Camera intrinsics | Backprojection parameters | Factory calibration or custom calibration |

---

## Verification Approaches

### Verifying T_camera_to_gelsight(u) is Consistent

Since camera and gelsight are mechanically linked (through gripper), at any fixed gripper opening u, the value T_camera_to_gelsight should be the same regardless of robot pose.

Test: Move robot to many poses with same gripper opening. Compute T_camera_to_gelsight each time. Values should be nearly identical.

This verification does NOT require X or robot API — it is purely vision-based.

### Verifying the Full Chain

After calibration, the computed gelsight position in base frame should be consistent with physical reality.

The document we reviewed used a boundary test: they fixed an object, computed where gelsight should be when touching the object boundary, and checked if the computed x-coordinate matched the known boundary x-coordinate within ~0.3mm.

---

## Key Takeaways

1. The gelsight sensor position cannot be observed at the moment of contact.

2. We pre-calibrate T_camera_to_gelsight as a function of gripper opening u.

3. X (camera to end-effector) is constant. T_camera_to_gelsight(u) is NOT constant.

4. At runtime, we use the kinematic chain with pre-calibrated values to compute gelsight position without seeing it.

5. Two gelsight sensors require two separate calibration models.

6. The linear model t(u) = t₀ + k·u captures how gelsight position changes with gripper opening, with t₀ extrapolated from visible measurements.
