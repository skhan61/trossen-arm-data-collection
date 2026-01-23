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
│ INPUTS (What We Obtain)                                     │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  1. Robot End-Effector Pose:                                │
│     Output: T_{base→ee} = [R_{base→ee} | t_{base→ee}]      │
│             (Cartesian 6DOF: x, y, z, roll, pitch, yaw)     │
│     Source: Robot API (driver computes FK internally)       │
│                                                             │
│  2. Hand-Eye Calibration (Camera-to-EE transform):          │
│     Output: X = T_{ee→camera} = [R_{ee→cam} | t_{ee→cam}]  │
│     Source: MoveIt Calibration library                      │
│             (uses ArUco marker + multiple robot poses)      │
│                                                             │
│  3. Camera-to-GelSight Calibration:                         │
│     Output: T_{camera→gelsight} = [R_{cam→gel} | t_{cam→gel}]│
│     Source: Custom script using camera image of GelSight    │
│             sensor mounted on end-effector                  │
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
│                 (Robot API)   (MoveIt)    (Custom script)  │
│                                                             │
│  t_{base→gel} = t_{base→ee} + R_{base→ee}·t_{ee→cam}       │
│                 + R_{base→ee}·R_{ee→cam}·t_{cam→gel}        │
│                      ↑              ↑             ↑         │
│                 (Robot API)   (MoveIt)    (Custom script)  │
│                                                             │
│  Result: GelSight position [x, y, z] in robot base frame   │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

### Summary: Source of Each Component

| Component | What It Is | Source Method | Tool |
|-----------|------------|---------------|------|
| **R_{base→ee}** | Gripper rotation in base | Robot API (Cartesian 6DOF) | Robot driver |
| **t_{base→ee}** | Gripper position in base | Robot API (Cartesian 6DOF) | Robot driver |
| **R_{ee→cam}** | Camera rotation in gripper | Hand-eye calibration | MoveIt Calibration |
| **t_{ee→cam}** | Camera position in gripper | Hand-eye calibration | MoveIt Calibration |
| **R_{cam→gel}** | GelSight rotation in camera | Camera-to-GelSight calibration | Custom script |
| **t_{cam→gel}** | GelSight position in camera | Camera-to-GelSight calibration | Custom script |
| **R_{base→gel}** | GelSight rotation in base | **Computed:** R₁·R₂·R₃ | Matrix multiplication |
| **t_{base→gel}** | GelSight position in base | **Computed:** formula above | Matrix multiplication |

**Key insight:**
- T_{base→ee}: From robot API directly (Cartesian 6DOF)
- T_{ee→camera} (X): From MoveIt Calibration library (hand-eye calibration)
- T_{camera→gelsight}: From custom script using camera image of GelSight on EE
- Final result: Computed by chaining the three transformations

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
**Directly from the robot API in Cartesian 6DOF format** (robot driver computes FK internally)

```python
# Robot API call - returns Cartesian pose directly
ee_pose = robot.get_ee_pose()
# Returns: [x, y, z, roll, pitch, yaw] in Cartesian coordinates

# The robot driver internally computes:
#   1. Reads joint encoders [θ₁, θ₂, ..., θ₆]
#   2. Applies forward kinematics using robot's kinematic model
#   3. Returns end-effector pose in Cartesian 6DOF
```

#### Converting 6DOF to 4×4 Transformation Matrix

Given pose = [x, y, z, roll (φ), pitch (θ), yaw (ψ)]:

**Translation vector:**
```
t = [x, y, z]ᵀ
```

**Rotation matrix (ZYX Euler angles convention):**
```
R = Rz(ψ) · Ry(θ) · Rx(φ)
```

Where the individual rotation matrices are:

```
Rx(φ) = [1      0       0   ]      Ry(θ) = [ cos(θ)  0  sin(θ)]      Rz(ψ) = [cos(ψ)  -sin(ψ)  0]
        [0   cos(φ)  -sin(φ)]              [   0     1    0   ]              [sin(ψ)   cos(ψ)  0]
        [0   sin(φ)   cos(φ)]              [-sin(θ)  0  cos(θ)]              [  0        0     1]
```

**Combined rotation matrix R = Rz(ψ) · Ry(θ) · Rx(φ):**
```
R = [cos(ψ)cos(θ)   cos(ψ)sin(θ)sin(φ)-sin(ψ)cos(φ)   cos(ψ)sin(θ)cos(φ)+sin(ψ)sin(φ)]
    [sin(ψ)cos(θ)   sin(ψ)sin(θ)sin(φ)+cos(ψ)cos(φ)   sin(ψ)sin(θ)cos(φ)-cos(ψ)sin(φ)]
    [  -sin(θ)              cos(θ)sin(φ)                       cos(θ)cos(φ)            ]
```

**Final 4×4 homogeneous transformation matrix:**
```
T_{base→ee} = [R  |  t]  =  [r₁₁  r₁₂  r₁₃  x]
              [0ᵀ |  1]     [r₂₁  r₂₂  r₂₃  y]
                            [r₃₁  r₃₂  r₃₃  z]
                            [ 0    0    0   1]
```

**Source:** Robot API call (forward kinematics computed internally by robot driver)

**Note:** You don't need to compute FK yourself - the robot driver handles this and returns the Cartesian pose directly.

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
**MoveIt Calibration Library (Hand-Eye Calibration)**

**IMPORTANT:** This calibration is done entirely by MoveIt Calibration library. Without MoveIt, this calibration cannot be performed!

##### What MoveIt Calibration Does:

MoveIt Calibration is a complete hand-eye calibration solution that handles:
1. **ArUco marker detection** - Detects the marker in camera images automatically
2. **Robot pose collection** - Gets T_{base→ee} from robot at each position
3. **Data pairing** - Collects (robot pose, marker pose) pairs at multiple positions
4. **AX=XB solving** - Solves the hand-eye calibration equation internally
5. **Result output** - Outputs T_{ee→camera} transform

##### The AX=XB Equation (solved internally by MoveIt):

For any two robot poses i and j:
```
A_ij × X = X × B_ij

Where:
A_ij = (T_{base→ee}^j)⁻¹ · T_{base→ee}^i  (robot motion from pose i to j)
B_ij = (T_{camera→marker}^j)⁻¹ · T_{camera→marker}^i  (observed camera motion)
X = T_{ee→camera}  (what MoveIt solves for)
```

**Key insight:** The marker is fixed in the world. When robot moves, both the robot motion (A) and observed camera motion (B) must be consistent through X.

##### Experiment Setup:

```
Physical Setup:
┌─────────────────────────────────────────────────────────┐
│                                                         │
│    [ArUco Marker]  ← Fixed on table, does NOT move     │
│          ↑                                              │
│          │ Camera sees marker                           │
│          │                                              │
│    [RealSense Camera] ← Mounted on gripper             │
│          │                                              │
│    [Robot Gripper/EE]                                  │
│          │                                              │
│    [Robot Arm]                                          │
│          │                                              │
│    [Robot Base]                                         │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

##### MoveIt Calibration Workflow:

```
Step 1: Setup
   - Fix ArUco marker on table (must NOT move during calibration!)
   - Mount RealSense camera on robot gripper
   - Launch MoveIt Calibration:
     → ros2 launch moveit_calibration hand_eye_calibration.launch.py

Step 2: Data Collection (15-30 poses)
   - Move robot to position where camera sees marker
   - MoveIt automatically:
     → Detects ArUco marker in camera image
     → Records T_{camera→marker} (marker pose in camera frame)
     → Records T_{base→ee} (gripper pose from robot API)
     → Stores the pair
   - Repeat at diverse positions with different:
     → Distances from marker (30-60 cm)
     → Viewing angles (0-60° off-axis)
     → Robot orientations (roll, pitch, yaw variations)

Step 3: Solve (automatic)
   - After collecting enough samples, MoveIt solves AX=XB
   - Uses optimization to find best X that satisfies all pose pairs

Step 4: Output
   - MoveIt outputs: X = T_{ee→camera}
   - Save to calibration file for later use
```

##### Why Diverse Poses Matter:

```
Good poses (maximize information):
┌──────────────────────────────────────────────────────────┐
│  Position 1      Position 2      Position 3             │
│     ╱               │               ╲                   │
│    ╱                │                ╲                  │
│   📷               📷               📷  ← Different angles│
│                                                          │
│              [ArUco Marker]                              │
│                                                          │
│  Position 4      Position 5      Position 6             │
│     📷              📷              📷   ← Different distances│
│      ↑               ↑               ↑                  │
│     far           medium          close                 │
└──────────────────────────────────────────────────────────┘

Bad poses (insufficient information):
┌──────────────────────────────────────────────────────────┐
│  📷 📷 📷 📷 📷 📷  ← All same angle, same distance      │
│        ↓                                                 │
│  [ArUco Marker]                                          │
│                                                          │
│  Result: Poor calibration, high error!                  │
└──────────────────────────────────────────────────────────┘
```

**Source:** MoveIt Calibration library (handles everything internally)

**Accuracy:** ±0.3-0.5mm when done with diverse poses

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
**Custom Script using Camera Image of GelSight Sensor**

This calibration is done by our own script that captures an image of the GelSight sensor (which is mounted on the end-effector) using the RealSense camera (also on the end-effector).

##### Physical Setup:

```
┌─────────────────────────────────────────────────────────────┐
│  END-EFFECTOR ASSEMBLY                                      │
│                                                             │
│     ┌─────────────┐                                         │
│     │  RealSense  │ ← Camera (captures image)              │
│     │   Camera    │                                         │
│     └──────┬──────┘                                         │
│            │                                                │
│            │ Camera looks at GelSight                       │
│            ↓                                                │
│     ┌─────────────┐                                         │
│     │  GelSight   │ ← Tactile sensor (visible in image)    │
│     │   Sensor    │                                         │
│     └─────────────┘                                         │
│            │                                                │
│     [Robot Gripper]                                         │
│                                                             │
└─────────────────────────────────────────────────────────────┘

Key: Both camera and GelSight are rigidly mounted on the gripper.
     Their relative position is FIXED and does not change.
```

##### Why This Works:

Since both the RealSense camera and GelSight sensor are mounted on the same rigid body (the gripper), their relative transform T_{camera→gelsight} is **constant**. We only need to measure it once!

##### Experiment Procedure:

```
┌─────────────────────────────────────────────────────────────┐
│ STEP 1: Position the Robot                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Move robot to a position where the RealSense camera      │
│   can clearly see the GelSight sensor surface.             │
│                                                             │
│   This may require:                                         │
│   - Using a mirror to reflect the GelSight into camera view│
│   - OR temporarily detaching camera to image GelSight      │
│   - OR using a second external camera                       │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 2: Capture Image                                       │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Capture RGB image from RealSense camera showing the      │
│   GelSight sensor clearly visible in the frame.            │
│                                                             │
│   Image should show:                                        │
│   - GelSight sensing surface (rectangular area)            │
│   - Clear corners or identifiable features                 │
│   - Good lighting, no blur                                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 3: Identify GelSight Features in Image                │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   In the captured image, identify known points on GelSight:│
│                                                             │
│   Option A: Click 4 corners of GelSight sensing surface    │
│   ┌─────────────────────┐                                  │
│   │ •               •   │  ← Click corners in image        │
│   │                     │                                   │
│   │                     │                                   │
│   │ •               •   │                                   │
│   └─────────────────────┘                                  │
│                                                             │
│   Option B: Use ArUco marker attached to GelSight          │
│   (if marker is placed on GelSight housing)                │
│                                                             │
│   Option C: Detect GelSight edges automatically            │
│   (using edge detection algorithms)                         │
│                                                             │
│   Result: 2D pixel coordinates [(u₁,v₁), (u₂,v₂), ...]     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 4: Define 3D Points (from GelSight Dimensions)        │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   From GelSight datasheet, we know the physical dimensions:│
│                                                             │
│   Sensing area: 18.6mm (width) × 14.3mm (height)           │
│                                                             │
│   Define 3D coordinates in GelSight frame (center = origin)│
│                                                             │
│   Corner 1 (top-left):     (+9.3mm, +7.15mm, 0)            │
│   Corner 2 (top-right):    (-9.3mm, +7.15mm, 0)            │
│   Corner 3 (bottom-right): (-9.3mm, -7.15mm, 0)            │
│   Corner 4 (bottom-left):  (+9.3mm, -7.15mm, 0)            │
│                                                             │
│   Note: Z=0 means corners lie on the GelSight surface plane│
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 5: Solve PnP (Perspective-n-Point)                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PnP Problem:                                              │
│   Given: - 3D points in GelSight frame (from dimensions)   │
│          - 2D points in image (from Step 3)                │
│          - Camera intrinsics (from camera calibration)     │
│   Find:  - T_{camera→gelsight}                             │
│                                                             │
│   Mathematical formulation:                                 │
│                                                             │
│   For each point i:                                         │
│                                                             │
│   [u_i]       [p_i^gel]                                     │
│   [v_i] = K · T_{cam→gel} · [  1  ]                        │
│   [ 1 ]                                                     │
│                                                             │
│   Where:                                                    │
│   - (u_i, v_i) = pixel coordinates                         │
│   - K = camera intrinsic matrix (3×3)                      │
│   - T_{cam→gel} = transform we want to find (4×4)         │
│   - p_i^gel = 3D point in GelSight frame                   │
│                                                             │
│   PnP solver finds R and t that minimize reprojection error│
│                                                             │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│ STEP 6: Output Transform                                    │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   PnP solver outputs:                                       │
│   - rvec: rotation vector (3×1)                            │
│   - tvec: translation vector (3×1)                         │
│                                                             │
│   Convert to transformation matrix:                         │
│                                                             │
│   R_{cam→gel} = rodrigues(rvec)   (3×3 rotation matrix)   │
│   t_{cam→gel} = tvec              (3×1 translation)        │
│                                                             │
│   T_{cam→gel} = [R_{cam→gel}  |  t_{cam→gel}]             │
│                 [    0ᵀ       |      1      ]              │
│                                                             │
│   Save to calibration file for later use.                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

##### Required Inputs:

| Input | Source | Description |
|-------|--------|-------------|
| Camera image | RealSense camera | Image showing GelSight sensor |
| 2D pixel coordinates | Manual click or detection | Corners of GelSight in image |
| 3D GelSight dimensions | Datasheet | Physical size of sensing area |
| Camera intrinsics (K) | Camera calibration | Focal length, principal point |
| Distortion coefficients | Camera calibration | Lens distortion parameters |

##### Camera Intrinsic Matrix K:

```
K = [fx   0  cx]
    [ 0  fy  cy]
    [ 0   0   1]

Where:
- fx, fy = focal lengths in pixels
- cx, cy = principal point (image center)
```

##### PnP Reprojection Error:

The solver minimizes:
```
E = Σᵢ || (u_i, v_i) - project(T_{cam→gel} · p_i^gel) ||²

Where project() applies camera projection:
project(P) = K · [P_x/P_z, P_y/P_z, 1]ᵀ
```

**Source:** Custom script (camera image + PnP solver)

**Accuracy:** ±0.5-1mm (depends on corner detection accuracy and camera calibration quality)

**Note:** This calibration only needs to be done ONCE since the camera and GelSight are rigidly mounted together

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

### Transform Sources

| Transform | Source | Tool/Method |
|-----------|--------|-------------|
| T_{base→ee} | Robot API | `robot.get_ee_pose()` returns Cartesian 6DOF |
| T_{ee→camera} (X) | Hand-eye calibration | MoveIt Calibration library |
| T_{camera→gelsight} | Camera-to-GelSight calibration | Custom script (camera image of GelSight on EE) |

### MoveIt Calibration Role

MoveIt Calibration library handles the complete hand-eye calibration:
- **ArUco marker detection:** Detects marker in camera images
- **Data collection:** Collects (robot pose, marker pose) pairs at multiple positions
- **AX=XB solver:** Solves the hand-eye calibration equation internally
- **Output:** T_{ee→camera} transform

### Robot API Role

The robot driver provides end-effector pose directly:
- **Input:** API call to robot driver
- **Output:** Cartesian 6DOF (x, y, z, roll, pitch, yaw)
- **Note:** FK is computed internally by the driver, no manual computation needed

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
