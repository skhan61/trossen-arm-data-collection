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
│   │   ├── rgb/
│   │   │   ├── 00.png
│   │   │   ├── 01.png
│   │   │   └── ...
│   │   ├── depth/
│   │   │   ├── 00.npy
│   │   │   ├── 01.npy
│   │   │   └── ...
│   │   ├── gelsight_left/
│   │   │   ├── 00.png
│   │   │   ├── 01.png
│   │   │   └── ...
│   │   ├── gelsight_right/
│   │   │   ├── 00.png
│   │   │   ├── 01.png
│   │   │   └── ...
│   │   ├── poses/
│   │   │   ├── left.npy
│   │   │   └── right.npy
│   │   └── timestamps.npy
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
    "num_frames": 7,
    "contact_frame_index": 3,
    "max_frame_index": 6,
    "post_contact_squeeze": 0.0085
}
```

Note:
- `contact_frame_index` and `max_frame_index` are 0-based indices matching file names (e.g., contact_frame_index=3 refers to file 03.png)
- `post_contact_squeeze` = gap_contact - gap_max in meters (gripper closure after contact; soft objects allow more squeeze)

---

## Sample Files

| File | Shape | Description |
|------|-------|-------------|
| sample.json | - | Sample metadata |
| rgb/{frame:02d}.png | (H, W, 3) uint8 | RealSense RGB frame |
| depth/{frame:02d}.npy | (H, W) float32 | Depth in meters |
| gelsight_left/{frame:02d}.png | (H, W, 3) uint8 | Left GelSight frame |
| gelsight_right/{frame:02d}.png | (H, W, 3) uint8 | Right GelSight frame |
| poses/left.npy | (N, 4, 4) float32 | T_base_to_gelsight_left per frame |
| poses/right.npy | (N, 4, 4) float32 | T_base_to_gelsight_right per frame |
| timestamps.npy | (N,) float64 | Timestamp per frame |

