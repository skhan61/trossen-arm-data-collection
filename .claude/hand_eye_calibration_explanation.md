# Hand-Eye Calibration for GelSight Data Collection

## Table of Contents
1. [Why We Need This Calibration](#why-we-need-this-calibration)
2. [What We're Computing](#what-were-computing)
3. [The Mathematical Framework](#the-mathematical-framework)
4. [How We Obtain Each Transform](#how-we-obtain-each-transform)
5. [The Complete Workflow](#the-complete-workflow)
6. [Tools and Methods](#tools-and-methods)

---

## Why We Need This Calibration

### The Problem
During GelSight tactile data collection:
- **GelSight sensor** is mounted on the robot gripper and collects tactile images when touching objects
- **RealSense camera** is also mounted on the gripper and captures RGB images of the scene
- **Robot** reports its gripper position in the robot base frame

### What We Need to Record for Each Data Sample
For YCB-Sight type datasets, each sample needs:
1. GelSight tactile image (what the sensor feels)
2. RealSense RGB image (what the camera sees)
3. Robot pose (where the gripper is)
4. **GelSight center position in robot base coordinates** ← This requires calibration!
5. **Contact location on the object** (where GelSight is touching)
6. **Surface normal at contact point**

### Why We Can't Get This Without Calibration
The robot only knows where the **gripper** is, not where the **GelSight sensor** is.

```
ROBOT BASE
    ↓ (robot knows this)
GRIPPER
    ↓ (??? unknown without calibration)
CAMERA
    ↓ (??? unknown without calibration)
GELSIGHT SENSOR
```

**Without calibration:** We have tactile images but don't know where in 3D space they were taken!

**With calibration:** We can compute the exact 3D position and orientation of the GelSight sensor for every data sample.

---

## What We're Computing

### The Goal
Compute the GelSight sensor position in robot base coordinates:

```
T_{base→gelsight} = Position and orientation of GelSight in robot base frame
```

### The Chain of Transforms
```
BASE → GRIPPER → CAMERA → GELSIGHT

T_{base→gelsight} = T_{base→ee} × T_{ee→camera} × T_{camera→gelsight}
                        ↑              ↑                  ↑
                   (robot FK)    (calibrate!)       (calibrate!)
```

---

## The Mathematical Framework

### Homogeneous Transformation Matrix

A transformation matrix **T** represents both position and orientation:

```
T = [R  |  t]
    [0ᵀ |  1]
```

Where:
- **R** = 3×3 rotation matrix (orientation)
- **t** = 3×1 translation vector (position)
- **0ᵀ** = [0, 0, 0]

Full 4×4 form:
```
    [r₁₁  r₁₂  r₁₃  tₓ]
T = [r₂₁  r₂₂  r₂₃  tᵧ]
    [r₃₁  r₃₂  r₃₃  tᵤ]
    [ 0    0    0   1 ]
```

### Matrix Multiplication Rule

For two transforms T₁ and T₂:
```
T₁ · T₂ = [R₁·R₂  |  R₁·t₂ + t₁]
          [ 0ᵀ    |      1     ]
```

---

## Mathematical Proof of Transformation Chain

### Theorem: Composition of Transformations

**Given:**
- Point **p** in GelSight frame
- Want to express **p** in robot base frame

**Prove:**
```
p_base = T_{base→ee} · T_{ee→cam} · T_{cam→gel} · p_gel
```

### Proof by Construction

#### Step 1: Point in GelSight Frame to Camera Frame

A point **p_gel** in GelSight coordinates can be expressed in camera coordinates:

```
p_cam = T_{cam→gel} · p_gel
```

Expanded:
```
[x_cam]   [R_{cam→gel}  |  t_{cam→gel}]   [x_gel]
[y_cam] = [             |              ] · [y_gel]
[z_cam]   [    0ᵀ       |      1       ]   [z_gel]
[ 1   ]                                     [ 1   ]
```

**What this means physically:**
- R_{cam→gel} rotates the point from GelSight orientation to camera orientation
- t_{cam→gel} translates the point from GelSight origin to camera origin
- Result: Point coordinates in camera frame

**Derivation:**
```
[x_cam]   [R_{cam→gel}] [x_gel]   [t_{cam→gel}]
[y_cam] = [           ] [y_gel] + [           ]
[z_cam]   [           ] [z_gel]   [           ]

Position in camera = (Rotation applied to gel position) + (translation from gel to cam)
```

---

#### Step 2: Point in Camera Frame to End-Effector Frame

The same point in end-effector coordinates:

```
p_ee = T_{ee→cam} · p_cam
```

Substituting p_cam from Step 1:
```
p_ee = T_{ee→cam} · (T_{cam→gel} · p_gel)
```

Expanded:
```
[x_ee]   [R_{ee→cam}  |  t_{ee→cam}]   [R_{cam→gel}  |  t_{cam→gel}]   [x_gel]
[y_ee] = [            |             ] · [             |              ] · [y_gel]
[z_ee]   [    0ᵀ      |      1      ]   [    0ᵀ       |      1       ]   [z_gel]
[ 1  ]                                                                    [ 1   ]
```

**What this means physically:**
- First transform moves point from GelSight to camera frame
- Second transform moves point from camera to end-effector frame
- Result: Point coordinates in end-effector frame

**Matrix multiplication (T_{ee→cam} · T_{cam→gel}):**

Using the multiplication rule:
```
T_{ee→gel} = [R_{ee→cam}·R_{cam→gel}  |  R_{ee→cam}·t_{cam→gel} + t_{ee→cam}]
             [         0ᵀ              |              1                      ]
```

Let's verify the translation component:
```
t_{ee→gel} = R_{ee→cam}·t_{cam→gel} + t_{ee→cam}

Physical meaning:
- t_{cam→gel}: Vector from camera origin to GelSight origin (in camera frame)
- R_{ee→cam}·t_{cam→gel}: Same vector rotated to ee frame orientation
- + t_{ee→cam}: Add vector from ee origin to camera origin
- Result: Total vector from ee origin to GelSight origin
```

---

#### Step 3: Point in End-Effector Frame to Base Frame

Finally, express the point in base coordinates:

```
p_base = T_{base→ee} · p_ee
```

Substituting p_ee from Step 2:
```
p_base = T_{base→ee} · (T_{ee→cam} · T_{cam→gel} · p_gel)
```

By associativity of matrix multiplication:
```
p_base = (T_{base→ee} · T_{ee→cam} · T_{cam→gel}) · p_gel
```

Let:
```
T_{base→gel} = T_{base→ee} · T_{ee→cam} · T_{cam→gel}
```

Then:
```
p_base = T_{base→gel} · p_gel
```

**This is our final transformation chain!** ∎

---

### Explicit Formula Derivation

#### Rotation Component

Starting from:
```
T_{base→gel} = T_{base→ee} · T_{ee→cam} · T_{cam→gel}
```

First multiply T_{ee→cam} · T_{cam→gel}:
```
Step A: R_{ee→gel} = R_{ee→cam} · R_{cam→gel}
```

Then multiply T_{base→ee} with the result:
```
Step B: R_{base→gel} = R_{base→ee} · R_{ee→gel}
                     = R_{base→ee} · (R_{ee→cam} · R_{cam→gel})
```

By associativity:
```
R_{base→gel} = R_{base→ee} · R_{ee→cam} · R_{cam→gel}
```

**Physical interpretation:**
- Each rotation matrix represents change of orientation between frames
- Composition means: "First rotate by R_{cam→gel}, then by R_{ee→cam}, then by R_{base→ee}"
- Order matters! Matrix multiplication is not commutative.

---

#### Translation Component

For translations, we use the multiplication rule stepwise:

**Step A:** Multiply T_{ee→cam} · T_{cam→gel}:
```
t_{ee→gel} = R_{ee→cam} · t_{cam→gel} + t_{ee→cam}
```

**Why?**
- t_{cam→gel} is in camera coordinates
- Must rotate it to ee coordinates: R_{ee→cam} · t_{cam→gel}
- Then add the ee-to-camera offset: + t_{ee→cam}

**Step B:** Multiply T_{base→ee} · T_{ee→gel}:
```
t_{base→gel} = R_{base→ee} · t_{ee→gel} + t_{base→ee}
```

Substituting t_{ee→gel} from Step A:
```
t_{base→gel} = R_{base→ee} · (R_{ee→cam} · t_{cam→gel} + t_{ee→cam}) + t_{base→ee}
```

Distributing:
```
t_{base→gel} = R_{base→ee}·R_{ee→cam}·t_{cam→gel} + R_{base→ee}·t_{ee→cam} + t_{base→ee}
```

**Final formulas:**
```
Rotation:
R_{base→gel} = R_{base→ee} · R_{ee→cam} · R_{cam→gel}

Translation:
t_{base→gel} = t_{base→ee} + R_{base→ee}·t_{ee→cam} + R_{base→ee}·R_{ee→cam}·t_{cam→gel}
```

Or compactly:
```
t_{base→gel} = t_{base→ee} + R_{base→ee}·(t_{ee→cam} + R_{ee→cam}·t_{cam→gel})
```

---

### Verification: What Comes From What

#### Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────┐
│ INPUTS (What We Measure)                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Robot Forward Kinematics:                               │
│     Input: Joint angles [θ₁, θ₂, ..., θ₆]                  │
│     Output: T_{base→ee} = [R_{base→ee} | t_{base→ee}]      │
│     Source: Robot encoders + FK computation                 │
│                                                             │
│  2. ArUco Detection (Computer Vision):                      │
│     Input: Camera RGB image + ArUco marker                  │
│     Output: T_{camera→marker} = [R_{cam→mark} | t_{cam→mark}]│
│     Source: OpenCV cv2.aruco.detectMarkers()                │
│                                                             │
│  3. PnP Solver (Computer Vision):                           │
│     Input: 3D points (datasheet) + 2D pixels (image)        │
│     Output: T_{camera→gelsight} = [R_{cam→gel} | t_{cam→gel}]│
│     Source: OpenCV cv2.solvePnP()                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ CALIBRATION (What We Solve For)                            │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  AX=XB Solver:                                              │
│     Input: Multiple pairs (T_{base→ee}, T_{camera→marker}) │
│     Process: Solve equation A·X = X·B for all pose pairs   │
│     Output: X = T_{ee→camera} = [R_{ee→cam} | t_{ee→cam}]  │
│     Source: OpenCV cv2.calibrateHandEye()                   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ FINAL COMPUTATION (What We Want)                           │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  T_{base→gelsight} = T_{base→ee} · X · T_{camera→gelsight} │
│                                                             │
│  Components:                                                │
│  R_{base→gel} = R_{base→ee} · R_{ee→cam} · R_{cam→gel}     │
│                      ↑              ↑             ↑         │
│                  (robot FK)    (AX=XB)        (PnP)        │
│                                                             │
│  t_{base→gel} = t_{base→ee} + R_{base→ee}·t_{ee→cam}       │
│                 + R_{base→ee}·R_{ee→cam}·t_{cam→gel}        │
│                      ↑              ↑             ↑         │
│                  (robot FK)    (AX=XB)        (PnP)        │
│                                                             │
│  Result: GelSight position [x, y, z] in robot base frame   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Summary: Source of Each Component

| Component | What It Is | Source Method | Tool/Algorithm |
|-----------|------------|---------------|----------------|
| **R_{base→ee}** | Gripper rotation in base | Robot FK from joint angles | Robot driver |
| **t_{base→ee}** | Gripper position in base | Robot FK from joint angles | Robot driver |
| **R_{ee→cam}** | Camera rotation in gripper | AX=XB calibration | cv2.calibrateHandEye() |
| **t_{ee→cam}** | Camera position in gripper | AX=XB calibration | cv2.calibrateHandEye() |
| **R_{cam→gel}** | GelSight rotation in camera | PnP from corner detection | cv2.solvePnP() |
| **t_{cam→gel}** | GelSight position in camera | PnP from corner detection | cv2.solvePnP() |
| **R_{base→gel}** | GelSight rotation in base | **Computed:** R₁·R₂·R₃ | Matrix multiplication |
| **t_{base→gel}** | GelSight position in base | **Computed:** formula above | Matrix multiplication |

**Key insight:**
- Direct measurements: Robot FK, ArUco detection, PnP
- Calibrated parameter: X (from AX=XB using multiple measurements)
- Final result: Computed by chaining transformations

---

## How We Obtain Each Transform

### 1. T_{base→ee} (Robot Base to End-Effector)

#### What it is:
```
T_{base→ee} = [R_{base→ee}  |  t_{base→ee}]
              [    0ᵀ       |      1      ]
```
- **R_{base→ee}** = Gripper orientation in base frame (3×3 rotation matrix)
- **t_{base→ee}** = Gripper position in base frame (3×1 vector [x, y, z]ᵀ)

#### How we get it:
**From robot forward kinematics** (robot always knows this!)

```python
# Robot API call
ee_pose = robot.get_ee_pose()
# Returns: [x, y, z, roll, pitch, yaw]

# Extract components:
t_{base→ee} = [x, y, z]ᵀ
R_{base→ee} = Rot_z(yaw) · Rot_y(pitch) · Rot_x(roll)
```

**Source:** Robot's internal joint encoders → forward kinematics computation

**Accuracy:** High (±0.1mm) - robot knows its own position well

---

### 2. T_{ee→camera} (End-Effector to Camera) = X

#### What it is:
```
X = T_{ee→camera} = [R_{ee→cam}  |  t_{ee→cam}]
                    [    0ᵀ      |      1     ]
```
- **R_{ee→cam}** = Camera orientation relative to gripper (3×3 rotation matrix)
- **t_{ee→cam}** = Camera position relative to gripper (3×1 vector)

#### How we get it:
**Hand-Eye Calibration using AX=XB method**

##### Method Overview:
1. **Setup:** Place an ArUco marker or checkerboard in a fixed position in the workspace
2. **Data Collection:** Move robot to 15-30 diverse poses where camera can see the marker
3. **At each pose, collect:**
   - T_{base→ee} (from robot forward kinematics)
   - T_{camera→marker} (from computer vision - ArUco/checkerboard detection)
4. **Solve AX=XB equation** to find X = T_{ee→camera}

##### The AX=XB Equation:

For any two robot poses i and j:
```
A_ij × X = X × B_ij

Where:
A_ij = (T_{base→ee}^j)⁻¹ · T_{base→ee}^i  (robot motion from pose i to j)
B_ij = (T_{camera→marker}^j)⁻¹ · T_{camera→marker}^i  (camera motion from pose i to j)
X = T_{ee→camera}  (what we're solving for - constant!)
```

**Key insight:** The marker doesn't move, so both motion chains must be consistent through the unknown X!

##### Tools Used:
- **Data Collection:** ROS 2 + MoveIt 2 for motion planning and safe robot movement
  - MoveIt Calibration GUI (RViz interface)
  - OR custom script using MoveIt Python API
- **Detection:** OpenCV for ArUco marker or checkerboard detection
- **Solver:** OpenCV `cv2.calibrateHandEye()` function
  - Implements Tsai-Lenz, Park, Horaud, Andreff, or Daniilidis methods

##### Workflow:
```
1. Launch ROS 2 + MoveIt
   → ros2 launch moveit_calibration hand_eye_calibration.launch.py

2. Move robot to diverse poses (manually or programmatically)
   → Ensure marker visible in all poses
   → Maximize rotation and translation diversity

3. At each pose:
   → Detect ArUco/checkerboard in camera image (CV)
   → Record T_{camera→marker} (from detection)
   → Record T_{base→ee} (from robot)

4. After collecting 15+ samples:
   → Run AX=XB solver
   → Get X = T_{ee→camera}

5. Save calibration
   → Store R_{ee→cam} and t_{ee→cam} to file
```

**Source:** Computer vision (ArUco/checkerboard detection) + Mathematical optimization (AX=XB solver)

**Accuracy:** High (±0.3-0.5mm) when properly done with diverse poses

---

### 3. T_{camera→gelsight} (Camera to GelSight Center)

#### What it is:
```
T_{cam→gel} = [R_{cam→gel}  |  t_{cam→gel}]
              [    0ᵀ       |      1      ]
```
- **R_{cam→gel}** = GelSight orientation relative to camera (3×3 rotation matrix)
- **t_{cam→gel}** = GelSight position relative to camera (3×1 vector)

#### How we get it:
**PnP (Perspective-n-Point) method using camera image**

##### Method Overview:
1. **Physical measurement:** Get GelSight dimensions from datasheet
   - Field of View: 18.6mm (H) × 14.3mm (V)
   - Compute 4 corner positions relative to GelSight center

2. **Image capture:** Position robot so camera sees GelSight clearly

3. **Corner detection:**
   - Detect or manually click 4 visible corners of GelSight in camera image
   - Get 2D pixel coordinates: [(u₁, v₁), (u₂, v₂), (u₃, v₃), (u₄, v₄)]

4. **PnP solver:**
   - Input: 3D corner positions (from datasheet) + 2D pixel positions (from image)
   - Output: R_{cam→gel} and t_{cam→gel}

##### Corner Positions (from GelSight datasheet):
```
3D positions in GelSight frame:
Corner 1 (top-left):     [ 9.3mm,  7.15mm, 0]
Corner 2 (top-right):    [-9.3mm,  7.15mm, 0]
Corner 3 (bottom-right): [-9.3mm, -7.15mm, 0]
Corner 4 (bottom-left):  [ 9.3mm, -7.15mm, 0]
```

##### PnP Algorithm:
```python
# Input
corners_3d = [[0.0093, 0.00715, 0], ...]  # From datasheet
corners_2d = [[u1, v1], [u2, v2], ...]    # From image (click or detect)
camera_matrix = [...]  # Camera intrinsics (from camera calibration)
dist_coeffs = [...]    # Distortion coefficients

# OpenCV PnP solver
success, rvec, tvec = cv2.solvePnP(
    corners_3d,
    corners_2d,
    camera_matrix,
    dist_coeffs
)

# Convert to rotation matrix and translation
R_{cam→gel}, _ = cv2.Rodrigues(rvec)
t_{cam→gel} = tvec
```

**Source:** Computer vision (corner detection in image) + Camera geometry (PnP solver)

**Accuracy:** Good (±0.5-1mm) depending on corner detection accuracy

---

### 4. T_{base→gelsight} (Base to GelSight Center) - FINAL RESULT

#### What it is:
```
T_{base→gel} = [R_{base→gel}  |  t_{base→gel}]
               [    0ᵀ        |      1       ]
```
- **R_{base→gel}** = GelSight orientation in base frame (3×3 rotation matrix)
- **t_{base→gel}** = GelSight position in base frame (3×1 vector [x, y, z]ᵀ)

#### How we compute it:
**Matrix multiplication of the three transforms above**

```
T_{base→gel} = T_{base→ee} × X × T_{cam→gel}
```

##### Expanded form:
```
[R_{base→gel}  |  t_{base→gel}]   [R_{base→ee}  |  t_{base→ee}]   [R_{ee→cam}  |  t_{ee→cam}]   [R_{cam→gel}  |  t_{cam→gel}]
[    0ᵀ        |      1       ] = [    0ᵀ       |      1      ] · [    0ᵀ     |      1     ] · [    0ᵀ       |      1      ]
```

##### Component formulas:

**Rotation:**
```
R_{base→gel} = R_{base→ee} · R_{ee→cam} · R_{cam→gel}
```

**Translation:**
```
t_{base→gel} = t_{base→ee} + R_{base→ee}·t_{ee→cam} + R_{base→ee}·R_{ee→cam}·t_{cam→gel}
```

Or more compactly:
```
t_{base→gel} = t_{base→ee} + R_{base→ee}·(t_{ee→cam} + R_{ee→cam}·t_{cam→gel})
```

**This gives us the 3D position [x, y, z] and orientation (rotation matrix) of the GelSight sensor center in robot base coordinates!**

---

## The Complete Workflow

### Phase 1: Calibrations (One-Time Setup)

#### Step 1: Camera Intrinsic Calibration
**Goal:** Get camera matrix and distortion coefficients

**Method:** Standard checkerboard calibration
```
1. Print checkerboard pattern
2. Capture 20+ images of checkerboard at different angles
3. Use OpenCV camera calibration
4. Get: camera_matrix, dist_coeffs
```

**Tool:** OpenCV `cv2.calibrateCamera()`

**Output:** Camera intrinsics file (used for all subsequent CV operations)

---

#### Step 2: Hand-Eye Calibration (Get X = T_{ee→camera})
**Goal:** Find camera position and orientation relative to gripper

**Method:** AX=XB calibration with ArUco marker

**Tools:**
- ROS 2 + MoveIt 2 (motion planning and robot control)
- OpenCV (ArUco marker detection)
- MoveIt Calibration GUI or custom script

**Detailed Process:**

1. **Setup:**
   ```
   - Fix ArUco marker to table (don't move it!)
   - Start ROS 2 robot driver
   - Launch MoveIt
   - Launch camera node
   ```

2. **Data Collection:**
   ```
   For 15-30 diverse poses:
       a) Move robot to pose where camera sees marker
          (Use MoveIt GUI or motion planning)

       b) Detect ArUco marker in camera image
          → Get T_{camera→marker} (from CV)

       c) Get robot pose
          → Get T_{base→ee} (from robot FK)

       d) Save pair: (T_{base→ee}, T_{camera→marker})
   ```

3. **Diversity Requirements:**
   ```
   Good calibration needs diverse poses:
   - Different distances from marker (30-60cm)
   - Different viewing angles (0-60° off-axis)
   - Rotations around all axes (roll, pitch, yaw)
   ```

4. **Solve AX=XB:**
   ```python
   # Input: List of (T_{base→ee}, T_{camera→marker}) pairs
   R_gripper2cam, t_gripper2cam = cv2.calibrateHandEye(
       R_base2gripper,  # List of rotation matrices
       t_base2gripper,  # List of translation vectors
       R_cam2marker,    # List of rotation matrices
       t_cam2marker,    # List of translation vectors
       method=cv2.CALIB_HAND_EYE_TSAI
   )
   ```

5. **Validation:**
   ```
   - Check reprojection error
   - Move to new poses and verify marker position
   - Should be <2 pixels error
   ```

6. **Save Result:**
   ```
   X = T_{ee→camera} = [R_{ee→cam} | t_{ee→cam}]
   Save to: hand_eye_calibration.json
   ```

**Output:** X = T_{ee→camera} (camera pose in gripper frame)

**Accuracy:** ±0.3-0.5mm (if done properly)

---

#### Step 3: Camera-to-GelSight Calibration (Get T_{camera→gelsight})
**Goal:** Find GelSight sensor position and orientation relative to camera

**Method:** PnP with GelSight 4 corners

**Tools:**
- OpenCV (PnP solver)
- Camera image

**Detailed Process:**

1. **Get GelSight Corner Positions (from datasheet):**
   ```
   Field of View: 18.6mm × 14.3mm

   3D positions in GelSight frame (meters):
   corner1 = [ 0.0093,  0.00715, 0.0]  # Top-left
   corner2 = [-0.0093,  0.00715, 0.0]  # Top-right
   corner3 = [-0.0093, -0.00715, 0.0]  # Bottom-right
   corner4 = [ 0.0093, -0.00715, 0.0]  # Bottom-left
   ```

2. **Capture Image:**
   ```
   - Position robot so camera sees GelSight clearly
   - Capture RGB image
   ```

3. **Detect Corners in Image:**
   ```
   Option A: Manual clicking
   - Click 4 corners in order
   - Get pixel coordinates [(u1,v1), (u2,v2), (u3,v3), (u4,v4)]

   Option B: Automatic detection
   - Use corner detection algorithm
   - Or detect visual markers on GelSight
   ```

4. **Solve PnP:**
   ```python
   success, rvec, tvec = cv2.solvePnP(
       corners_3d,      # From datasheet
       corners_2d,      # From image
       camera_matrix,   # From camera calibration
       dist_coeffs      # From camera calibration
   )

   R_{cam→gel}, _ = cv2.Rodrigues(rvec)
   t_{cam→gel} = tvec
   ```

5. **Save Result:**
   ```
   T_{camera→gelsight} = [R_{cam→gel} | t_{cam→gel}]
   Save to: camera_to_gelsight.json
   ```

**Output:** T_{camera→gelsight} (GelSight pose in camera frame)

**Accuracy:** ±0.5-1mm

---

### Phase 2: Data Collection (Repeated for Each Sample)

#### During GelSight Data Collection:

For each touch sample:

1. **Robot moves and GelSight touches object**
   ```
   - Robot executes motion to touch object
   - GelSight makes contact with surface
   ```

2. **Capture data:**
   ```
   - gelsight_image = GelSight tactile image
   - camera_image = RealSense RGB image
   - timestamp = Current time
   ```

3. **Get robot pose:**
   ```python
   ee_pose = robot.get_ee_pose()  # [x, y, z, roll, pitch, yaw]
   T_{base→ee} = pose_to_matrix(ee_pose)
   ```

4. **Load calibrations:**
   ```python
   X = load("hand_eye_calibration.json")  # T_{ee→camera}
   T_{cam→gel} = load("camera_to_gelsight.json")
   ```

5. **Compute GelSight position in base frame:**
   ```python
   T_{base→gel} = T_{base→ee} @ X @ T_{cam→gel}

   # Extract position and orientation
   gelsight_position = T_{base→gel}[0:3, 3]  # [x, y, z]
   gelsight_orientation = T_{base→gel}[0:3, 0:3]  # Rotation matrix
   ```

6. **Compute 4 corners of GelSight in base frame:**
   ```python
   corners_relative = [
       [ 0.0093,  0.00715, 0, 1],  # Homogeneous coordinates
       [-0.0093,  0.00715, 0, 1],
       [-0.0093, -0.00715, 0, 1],
       [ 0.0093, -0.00715, 0, 1]
   ]

   corners_in_base = []
   for corner in corners_relative:
       corner_base = T_{base→gel} @ corner
       corners_in_base.append(corner_base[0:3])
   ```

7. **Save complete data sample:**
   ```json
   {
     "gelsight_image": "frame_001_tactile.png",
     "camera_image": "frame_001_rgb.png",
     "timestamp": "2025-01-19T10:30:45.123Z",
     "robot_pose": {
       "position": [x, y, z],
       "orientation": [roll, pitch, yaw]
     },
     "gelsight_center": {
       "position": [x, y, z],
       "orientation_matrix": [[r11, r12, r13], ...]
     },
     "gelsight_corners": [
       [x1, y1, z1],
       [x2, y2, z2],
       [x3, y3, z3],
       [x4, y4, z4]
     ]
   }
   ```

---

## Tools and Methods Summary

### Computer Vision (CV) Sources

All transforms come from computer vision methods:

| Transform | CV Method | Tool | Input | Output |
|-----------|-----------|------|-------|--------|
| T_{camera→marker} | ArUco detection | OpenCV `cv2.aruco.detectMarkers()` | Camera image | Marker pose |
| T_{ee→camera} (X) | AX=XB solver | OpenCV `cv2.calibrateHandEye()` | Multiple (gripper, marker) pairs | Camera pose in gripper frame |
| T_{camera→gelsight} | PnP solver | OpenCV `cv2.solvePnP()` | 3D corners + 2D pixels | GelSight pose in camera frame |

### Robot Source

| Transform | Source | Method |
|-----------|--------|--------|
| T_{base→ee} | Robot FK | API call `robot.get_ee_pose()` |

### ROS 2 + MoveIt Role

**MoveIt does NOT solve AX=XB directly!**

MoveIt helps with:
- **Motion planning:** Safe, collision-free robot movements
- **Data collection interface:** GUI for calibration workflow
- **Pose sampling:** Generate diverse robot poses for calibration
- **Visualization:** See robot, camera, marker in RViz

**The actual calibration solving is done by OpenCV CV algorithms!**

---

## Why This Matters for YCB-Sight Dataset

For each GelSight tactile sample, researchers need to know:

1. ✅ **What was felt** (GelSight tactile image)
2. ✅ **What was seen** (RealSense camera image)
3. ✅ **Where it was felt** (3D position on object) ← **Requires this calibration!**
4. ✅ **Surface geometry** (normal vector, curvature) ← **Requires this calibration!**
5. ✅ **Alignment** (tactile-visual correspondence) ← **Requires this calibration!**

**Without accurate calibration, the dataset lacks precise 3D geometry information, making it less useful for learning tactile-visual relationships!**

---

## References

### Algorithms
- **AX=XB Calibration:** Daniilidis, "Hand-Eye Calibration Using Dual Quaternions" (1999)
- **Alternative methods:** Tsai-Lenz, Park, Horaud, Andreff

### Tools
- **ROS 2:** https://docs.ros.org/
- **MoveIt 2:** https://moveit.ros.org/
- **MoveIt Calibration:** https://github.com/moveit/moveit_calibration
- **OpenCV:** https://opencv.org/

### Your Implementation
- Code location: `/home/skhan61/Desktop/trossen-arm-data-collection/src/`
- Hand-eye calibration script: `src/hand_eye_calibration.py`
- Calibration computation: `src/compute_hand_eye.py`

---

**Document created:** 2026-01-19
**For project:** GelSight Tactile Data Collection with WidowX Robot
