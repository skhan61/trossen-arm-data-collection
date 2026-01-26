# Visual-Haptic Deformation Dataset Schema

## Folder Structure

```
dataset/
├── calibration/
│   ├── X.npy
│   ├── T_u_left_params.npy
│   └── T_u_right_params.npy
│
├── objects/
│   ├── object_001.json
│   └── object_002.json
│
├── samples/
│   ├── 000001/
│   │   ├── sample.json
│   │   ├── rgb.mp4
│   │   ├── depth.npy
│   │   ├── gelsight_left.mp4
│   │   ├── gelsight_right.mp4
│   │   └── poses.npy
│   │
│   └── 000002/
│       └── ...
│
└── metadata.json
```

---

## metadata.json

```json
{
    "name": "visual_haptic_deformation",
    "date": "2026-01-25",
    "num_objects": 10,
    "num_samples": 500
}
```

---

## objects/object_001.json

```json
{
    "object_id": "object_001",
    "description": "soft foam cube"
}
```

---

## samples/000001/sample.json

```json
{
    "sample_id": "000001",
    "object_id": "object_001",
    "contact_frame": 15,
    "max_press_frame": 45,
    "fps": 30,
    "num_frames": 90
}
```

---

## Sample Files

| File | Format | Description |
|------|--------|-------------|
| rgb.mp4 | Video (N, H, W, 3) | RealSense RGB video |
| depth.npy | (N, H, W) float32 | Depth frames in meters |
| gelsight_left.mp4 | Video (N, H, W, 3) | Left GelSight video |
| gelsight_right.mp4 | Video (N, H, W, 3) | Right GelSight video |
| poses.npy | (N, 2, 4, 4) float32 | T_base_to_gelsight [left, right] per frame |

---

## Computed At Load Time

| Variable | Source |
|----------|--------|
| V_before | rgb.mp4 frame 0 |
| D_before | depth.npy[0] |
| V_after | rgb.mp4 frame max_press_frame |
| D_after | depth.npy[max_press_frame] |
| H_left | gelsight_left.mp4 frame max_press_frame |
| H_right | gelsight_right.mp4 frame max_press_frame |
| Pose | poses.npy[max_press_frame] |
| F | Z difference: poses[contact_frame] - poses[max_press_frame] |

---

## Code Structure

```
src/
├── calibration/              # Calibration pipelines (EXISTS)
│   ├── eye_in_hand/
│   │   ├── run_eye_in_hand_calibration.py
│   │   ├── collect_X_verification_data.py
│   │   ├── export_X_to_npy.py
│   │   ├── verify_X.py
│   │   └── camera_pose.launch.py
│   └── gelsight_calibration/
│       ├── run_gelsight_calibration.py
│       ├── collect_gelsight_calibration_data.py
│       ├── compute_gelsight_calibration.py
│       └── verify_gelsight_calibration.py
│
├── data_collection/          # Data collection pipeline (NEW)
│   ├── __init__.py
│   ├── run_collection.py     # Entry point - orchestrates collection
│   ├── writer.py             # Save sample to disk (mp4, npy, json)
│   └── loader.py             # Load sample for training
│
├── robot/                    # Robot control (NEW - needs hardware)
│   ├── __init__.py
│   └── arm.py                # Move arm, get pose, gripper control
│
├── sensors/                  # Sensor capture (NEW - needs hardware)
│   ├── __init__.py
│   ├── gelsight.py           # GelSight left + right capture
│   └── realsense.py          # RealSense RGB + depth capture
│
└── utils/                    # Shared utilities (PARTIAL)
    ├── __init__.py
    ├── config.py             # Load .env, paths (EXISTS)
    ├── log.py                # Centralized logging (EXISTS)
    ├── types.py              # Sample, Object, Metadata, X, T (EXISTS)
    ├── video.py              # frames <-> mp4 conversion (NEW)
    └── transforms.py         # T_base_to_gelsight computation (NEW)
```

---

## Entry Point: run_collection.py

```
1. INITIALIZE
   ├── Load calibration (X.npy, T_u_left.npy, T_u_right.npy)
   ├── Connect robot (arm.py)
   ├── Connect sensors (gelsight.py, realsense.py)
   └── Create DatasetWriter (writer.py)

2. FOR EACH OBJECT
   ├── Define object boundary (4 corners from camera)
   └── Save object metadata (object_id, description)

3. FOR EACH SAMPLE
   ├── Move robot above object
   ├── Start recording loop:
   │   ├── Capture RealSense (RGB, depth)
   │   ├── Capture GelSight (left, right)
   │   ├── Get robot pose + gripper opening
   │   ├── Compute T_base_to_gelsight (transforms.py)
   │   ├── Check contact (position-based, ~0.3mm tolerance)
   │   ├── Store frame in buffer
   │   └── Move robot down
   ├── Stop at max_press
   ├── Save sample via writer.py:
   │   ├── rgb.mp4, gelsight_left.mp4, gelsight_right.mp4
   │   ├── depth.npy, poses.npy
   │   └── sample.json
   └── Retract robot

4. FINALIZE
   └── Update dataset metadata.json
```

---

## Contact Detection (No Force Sensor)

Object boundary defined by 4 corners in base frame:
- Top left, Top right, Bottom right, Bottom left
- Left edge line: top_left -> bottom_left (robot touches here)

Contact detected when:
```
|robot_x - boundary_x(z)| < 0.3mm
```
Where boundary_x is computed from the left edge line at robot's z.

---

## Sensor Rates

| Sensor | Rate | Notes |
|--------|------|-------|
| GelSight | ~18 Hz | Some duplicate frames at 30 Hz sampling |
| RealSense | ~29 Hz | RGB + depth |
| Sampling | 30 Hz | Every 0.0333s grab latest from all sensors |

---

## Build Order

Can build NOW (no hardware):
1. `utils/video.py` - mp4 encode/decode
2. `utils/transforms.py` - pose math
3. `data_collection/writer.py` - save samples
4. `data_collection/loader.py` - load samples
5. Tests for all above

Build LATER (needs hardware):
1. `robot/arm.py`
2. `sensors/gelsight.py`
3. `sensors/realsense.py`
4. `data_collection/run_collection.py`
